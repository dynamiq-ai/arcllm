"""
Google Vertex AI adapter for arcllm.

Vertex AI hosts multiple model families on top of native Gemini:

- ``gemini-*`` — Google's flagship + Gemma open models (default path).
- ``claude-*`` — Anthropic Claude on Vertex (``publishers/anthropic``,
  ``:rawPredict`` endpoint, Anthropic Messages wire format with
  ``anthropic_version: "vertex-2023-10-16"``).
- ``mistral-*`` / ``codestral-*`` — Mistral on Vertex
  (``publishers/mistralai``, ``:rawPredict``, Mistral Chat Completions
  shape).
- ``llama-*`` / ``meta-*`` — Llama on Vertex MaaS
  (``publishers/meta``, OpenAI-compat shape).

All paths share Vertex's OAuth-based authentication (Bearer access token from
``gcloud auth print-access-token`` or ADC).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import orjson

from arcllm.exceptions import (
    ArcLLMError,
    AuthenticationError,
    ResponseParseError,
)
from arcllm.providers.base import (
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.gemini_adapter import GeminiAdapter

if TYPE_CHECKING:
    from arcllm.types import ModelResponse, StreamChunk

__all__ = ["VertexAIAdapter"]


def _get_vertex_family(model: str) -> str:
    """Classify the model id into a Vertex publisher family."""
    m = model.lower()
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"
    if m.startswith("mistral") or m.startswith("codestral") or "mistralai" in m:
        return "mistral"
    if m.startswith("llama") or m.startswith("meta") or "llama" in m:
        return "meta"
    # text-embedding-* / gemini-embedding-* fall through to Gemini path.
    return "gemini"


_VERTEX_PUBLISHERS = {
    "gemini": "google",
    "anthropic": "anthropic",
    "mistral": "mistralai",
    "meta": "meta",
}


class VertexAIAdapter(GeminiAdapter):
    """Adapter for Google Vertex AI with multi-publisher dispatch."""

    provider_name = "vertex_ai"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._project = (
            config.vertex_project
            or os.environ.get("VERTEX_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        self._location = config.vertex_location or os.environ.get("VERTEX_LOCATION", "us-central1")
        self._api_base = config.api_base or f"https://{self._location}-aiplatform.googleapis.com/v1"
        # Lazy-initialised proxy adapters used purely for response-side
        # parsing on cross-provider Vertex paths (Anthropic / Mistral / Llama).
        self._proxy_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Auth + project helpers (shared across all families)
    # ------------------------------------------------------------------

    def _get_api_key(self, env_var: str = "GOOGLE_API_KEY", param_key: str = "api_key") -> str:
        """Get an OAuth2 access token for Vertex AI.

        Resolution order:

        1. Explicit ``api_key`` on :class:`ProviderConfig`.
        2. ``VERTEX_API_KEY`` / ``GOOGLE_ACCESS_TOKEN`` env vars.
        3. ``gcloud auth print-access-token`` (or Application Default
           Credentials when ``gcloud`` isn't installed).
        """
        token = (
            self.config.api_key
            or os.environ.get("VERTEX_API_KEY")
            or os.environ.get("GOOGLE_ACCESS_TOKEN")
        )
        if not token:
            try:
                import subprocess

                result = subprocess.run(
                    ["gcloud", "auth", "print-access-token"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    token = result.stdout.strip()
            except Exception:
                token = None
        if not token:
            raise AuthenticationError(
                "Vertex AI access token not provided. Set GOOGLE_ACCESS_TOKEN, "
                "use 'gcloud auth print-access-token', or configure Application "
                "Default Credentials.",
                provider=self.provider_name,
            )
        return token

    def _get_project(self) -> str:
        if not self._project:
            raise ArcLLMError(
                "Vertex AI project not provided. Set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT.",
                provider=self.provider_name,
            )
        return self._project

    def _vertex_url(self, family: str, model: str, *, method: str) -> str:
        """Build the publisher-scoped Vertex URL for a model family.

        ``method`` is one of ``generateContent``, ``streamGenerateContent``,
        ``rawPredict``, ``streamRawPredict``, or ``predict``.
        """
        publisher = _VERTEX_PUBLISHERS[family]
        project = self._get_project()
        return (
            f"{self._api_base}/projects/{project}"
            f"/locations/{self._location}/publishers/{publisher}/models/{model}:{method}"
        )

    def _vertex_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build a Vertex AI request, routing by detected model family."""
        kwargs = self._check_params(model, drop_params, **kwargs)
        family = _get_vertex_family(model)

        if family == "gemini":
            return self._build_vertex_gemini_request(model, messages, stream=stream, **kwargs)
        if family == "anthropic":
            return self._build_vertex_anthropic_request(model, messages, stream=stream, **kwargs)
        if family == "mistral":
            return self._build_vertex_mistral_request(model, messages, stream=stream, **kwargs)
        if family == "meta":
            return self._build_vertex_meta_request(model, messages, stream=stream, **kwargs)
        # Defensive default — should never hit since families are exhaustive.
        return self._build_vertex_gemini_request(model, messages, stream=stream, **kwargs)

    def _proxy_adapter(self, family: str) -> Any:
        """Spin up (and cache) a sub-adapter for response/stream parsing.

        We instantiate the underlying-provider adapter with a stub config
        because we only need its parsing methods, not its auth/endpoint logic.
        Cached on the Vertex adapter so repeated calls don't re-allocate.
        """
        if family in self._proxy_cache:
            return self._proxy_cache[family]

        sub: Any
        stub = ProviderConfig(api_key="vertex-noop")
        if family == "anthropic":
            from arcllm.providers.anthropic_adapter import AnthropicAdapter

            sub = AnthropicAdapter(stub)
        elif family == "mistral":
            from arcllm.providers.mistral_adapter import MistralAdapter

            sub = MistralAdapter(stub)
        elif family == "meta":
            from arcllm.providers.openai_adapter import OpenAIAdapter

            sub = OpenAIAdapter(stub)
        else:  # pragma: no cover — caller should only ask for non-Gemini families
            raise ArcLLMError(
                f"Unknown family for Vertex proxy: {family}",
                provider=self.provider_name,
            )
        self._proxy_cache[family] = sub
        return sub

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Dispatch response parsing to the underlying-provider parser."""
        family = _get_vertex_family(model)
        if family in {"anthropic", "mistral", "meta"}:
            response: ModelResponse = self._proxy_adapter(family).parse_response(data, model)
            return response
        return super().parse_response(data, model)

    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        family = _get_vertex_family(model)
        if family in {"anthropic", "mistral", "meta"}:
            chunk: StreamChunk | None = self._proxy_adapter(family).parse_stream_event(data, model)
            return chunk
        return super().parse_stream_event(data, model)

    # ------------------------------------------------------------------
    # Per-family request builders
    # ------------------------------------------------------------------

    def _build_vertex_gemini_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        **kwargs: Any,
    ) -> RequestData:
        """Native Gemini path — build the same body as ``GeminiAdapter`` but
        on Vertex's URL + auth."""
        # Reuse GeminiAdapter's body assembly (a private path; we already
        # inherit from it, so calling it directly is safe).
        # Simplest: ask GeminiAdapter.build_request for a body and rewrite the URL.
        gemini_request = GeminiAdapter.build_request(
            self,
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )
        method = "streamGenerateContent" if stream else "generateContent"
        url = self._vertex_url("gemini", model, method=method)
        if stream:
            url += "?alt=sse"
        return RequestData(
            method="POST",
            url=url,
            headers=self._vertex_headers(),
            body=gemini_request.body,
            timeout=self.config.timeout,
        )

    def _build_vertex_anthropic_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        **kwargs: Any,
    ) -> RequestData:
        """Claude on Vertex: Anthropic Messages body + Vertex endpoint.

        The wire shape is exactly what direct Anthropic accepts, except the
        version field is ``vertex-2023-10-16`` (Vertex-specific).
        """
        from arcllm.providers.anthropic_adapter import AnthropicAdapter

        # Reuse the Anthropic adapter purely for its content converters.
        anth = self._proxy_adapter("anthropic")
        assert isinstance(anth, AnthropicAdapter)
        # Pyright flags use of `_`-prefixed methods across modules as private
        # access; this is the established intra-package collaboration pattern.
        system_prompt, anthropic_messages = anth._convert_messages(messages)  # pyright: ignore[reportPrivateUsage]
        body: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_prompt:
            body["system"] = system_prompt
        if "thinking_budget" in kwargs and kwargs["thinking_budget"] is not None:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(kwargs["thinking_budget"]),
            }
        else:
            if "temperature" in kwargs and kwargs["temperature"] is not None:
                body["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs and kwargs["top_p"] is not None:
                body["top_p"] = kwargs["top_p"]
        if kwargs.get("stop"):
            body["stop_sequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )
        if kwargs.get("tools"):
            body["tools"] = anth._convert_tools(kwargs["tools"])  # pyright: ignore[reportPrivateUsage]

        method = "streamRawPredict" if stream else "rawPredict"
        url = self._vertex_url("anthropic", model, method=method)
        return RequestData(
            method="POST",
            url=url,
            headers=self._vertex_headers(),
            body=orjson.dumps(body),
            timeout=self.config.timeout,
        )

    def _build_vertex_mistral_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        **kwargs: Any,
    ) -> RequestData:
        """Mistral on Vertex: standard Mistral Chat Completions body."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "random_seed",
            "safe_prompt",
        ):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        if kwargs.get("stop"):
            stop_val = kwargs["stop"]
            body["stop"] = stop_val if isinstance(stop_val, list) else [stop_val]
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice") is not None:
                body["tool_choice"] = kwargs["tool_choice"]
        if kwargs.get("response_format"):
            body["response_format"] = kwargs["response_format"]

        method = "streamRawPredict" if stream else "rawPredict"
        url = self._vertex_url("mistral", model, method=method)
        return RequestData(
            method="POST",
            url=url,
            headers=self._vertex_headers(),
            body=orjson.dumps(body),
            timeout=self.config.timeout,
        )

    def _build_vertex_meta_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        **kwargs: Any,
    ) -> RequestData:
        """Llama on Vertex MaaS: OpenAI-compat Chat Completions body."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "user",
        ):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice") is not None:
                body["tool_choice"] = kwargs["tool_choice"]
        if kwargs.get("response_format"):
            body["response_format"] = kwargs["response_format"]

        # Llama on Vertex uses the OpenAI Chat Completions surface via the
        # publisher-scoped endpoint at ``:openapi``-compatible path.
        method = "streamRawPredict" if stream else "rawPredict"
        url = self._vertex_url("meta", model, method=method)
        return RequestData(
            method="POST",
            url=url,
            headers=self._vertex_headers(),
            body=orjson.dumps(body),
            timeout=self.config.timeout,
        )

    # ------------------------------------------------------------------
    # Embeddings (Gemini family only — ``text-embedding-*`` / ``gemini-embedding-*``)
    # ------------------------------------------------------------------

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Vertex embeddings: ``:predict`` endpoint, ``instances`` payload."""
        instances = [{"content": text} for text in input]
        body = {"instances": instances}
        url = self._vertex_url("gemini", model, method="predict")
        return RequestData(
            method="POST",
            url=url,
            headers=self._vertex_headers(),
            body=orjson.dumps(body),
            timeout=self.config.timeout,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> Any:
        """Vertex embedding response uses ``predictions[i].embeddings.values``."""
        try:
            payload: dict[str, Any] = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ResponseParseError(
                f"Failed to parse Vertex embedding response: {exc}",
                provider=self.provider_name,
                raw_data=data,
            ) from exc

        from arcllm.types import EmbeddingData, EmbeddingResponse, EmbeddingUsage

        rows: list[EmbeddingData] = []
        predictions = cast("list[Any]", payload.get("predictions") or [])
        for i, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                continue
            pred_dict = cast("dict[str, Any]", pred)
            emb_raw: Any = pred_dict.get("embeddings") or {}
            if not isinstance(emb_raw, dict):
                continue
            emb = cast("dict[str, Any]", emb_raw)
            values: Any = emb.get("values") or []
            if isinstance(values, list):
                rows.append(EmbeddingData(index=i, embedding=list(cast("list[float]", values))))

        return EmbeddingResponse(
            model=model,
            data=rows,
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )


# Register on import
register_provider("vertex_ai", VertexAIAdapter)
