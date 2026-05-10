"""
OpenAI adapter for arcllm.

Supports chat completions (sync + streaming), tool / function calling,
JSON-mode and JSON-schema structured output, embeddings, vision inputs,
PDF inputs, and the o-series reasoning controls.

Reference: https://platform.openai.com/docs/api-reference/chat
Models:    https://platform.openai.com/docs/models
Pricing:   https://openai.com/pricing
Errors:    https://platform.openai.com/docs/guides/error-codes
"""

from __future__ import annotations

import time
from typing import Any, cast

import orjson

from arcllm.exceptions import (
    ArcLLMError,
    AuthenticationError,
    BudgetExceededError,
    ContentFilterError,
    InternalServerError,
    InvalidRequestError,
    ProviderAPIError,
    RateLimitError,
    ResponseParseError,
    ServiceUnavailableError,
    UnsupportedModelError,
)
from arcllm.providers.base import (
    COMMON_PARAMS,
    BaseAdapter,
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.types import (
    Choice,
    ChunkChoice,
    ChunkDelta,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    ImageData,
    ImageResponse,
    Message,
    ModelResponse,
    StreamChunk,
    ToolCall,
    Usage,
)

__all__ = ["OpenAIAdapter"]


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI API."""

    provider_name = "openai"

    # OpenAI supports all common params plus some extras
    supported_params = COMMON_PARAMS | {
        "max_completion_tokens",  # For o1 models
        "logit_bias",
        "parallel_tool_calls",
        "service_tier",
        "store",
        "metadata",
        "stream_options",
        "reasoning_effort",  # For o1 models
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.openai.com/v1"

    def _build_headers(self) -> dict[str, str]:
        """Build request headers (cached after first call)."""
        api_key = self._get_api_key("OPENAI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project:
            headers["OpenAI-Project"] = self.config.project
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build OpenAI chat completion request."""
        # Check params
        kwargs = self._check_params(model, drop_params, **kwargs)

        # Build request body
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Add stream_options for usage in streaming
        if stream:
            stream_options = kwargs.pop("stream_options", None)
            if stream_options:
                body["stream_options"] = stream_options

        # Handle max_tokens vs max_completion_tokens for newer models.
        # Reasoning models (o1/o3/o4) and the GPT-5 + GPT-4.1 families all use
        # ``max_completion_tokens`` instead of the legacy ``max_tokens`` field.
        uses_completion_tokens = (
            model.startswith("o1")
            or model.startswith("o3")
            or model.startswith("o4")
            or model.startswith("gpt-5")
            or model.startswith("gpt-4.1")
        )
        if uses_completion_tokens and "max_tokens" in kwargs:
            if "max_completion_tokens" not in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            else:
                kwargs.pop("max_tokens")

        # Add optional parameters
        optional_params = [
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "n",
            "logprobs",
            "top_logprobs",
            "user",
            "parallel_tool_calls",
            "service_tier",
            "store",
            "metadata",
            "reasoning_effort",
        ]

        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                body[param] = kwargs[param]

        # Handle tools
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
            if "tool_choice" in kwargs:
                body["tool_choice"] = kwargs["tool_choice"]

        # Handle response_format
        if kwargs.get("response_format"):
            body["response_format"] = kwargs["response_format"]

        url = f"{self._api_base}/chat/completions"
        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Parse OpenAI chat completion response."""
        try:
            resp = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse response JSON: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        return self._build_model_response(resp, model)

    def _build_model_response(self, resp: dict[str, Any], model: str) -> ModelResponse:
        """Build ModelResponse from parsed JSON."""
        # Cache timestamp once for this response (avoids multiple syscalls)
        now = int(time.time())
        choices: list[Choice] = []

        for choice_data in resp.get("choices", []):
            message_data = choice_data.get("message", {})

            # Parse tool calls if present
            tool_calls: list[ToolCall] | None = None
            if message_data.get("tool_calls"):
                tool_calls = []
                for tc in message_data["tool_calls"]:
                    func_data = tc.get("function", {})
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            type=tc.get("type", "function"),
                            function=FunctionCall(
                                name=func_data.get("name", ""),
                                arguments=func_data.get("arguments", ""),
                            ),
                        )
                    )

            # Parse legacy function_call if present
            function_call: FunctionCall | None = None
            if message_data.get("function_call"):
                fc = message_data["function_call"]
                function_call = FunctionCall(
                    name=fc.get("name", ""),
                    arguments=fc.get("arguments", ""),
                )

            # ``reasoning_content`` is the de-facto field name used by
            # DeepSeek-R1, GLM-4.5+, Groq's DeepSeek/Qwen-thinking models,
            # Cerebras, Together, Fireworks, and any OpenAI-compat host
            # serving a reasoning model. ``reasoning`` is the alias
            # OpenAI ships on the chat-completions endpoint for o-series
            # responses; we accept either and normalise to one field.
            reasoning_content = message_data.get("reasoning_content") or message_data.get(
                "reasoning"
            )

            message = Message(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content"),
                tool_calls=tool_calls,
                function_call=function_call,
                refusal=message_data.get("refusal"),
                reasoning_content=reasoning_content,
            )

            choices.append(
                Choice(
                    index=choice_data.get("index", 0),
                    message=message,
                    finish_reason=choice_data.get("finish_reason"),
                    logprobs=choice_data.get("logprobs"),
                )
            )

        # Parse usage
        usage: Usage | None = None
        usage_data = resp.get("usage")
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                prompt_tokens_details=usage_data.get("prompt_tokens_details"),
                completion_tokens_details=usage_data.get("completion_tokens_details"),
            )

        return ModelResponse(
            id=resp.get("id", ""),
            object=resp.get("object", "chat.completion"),
            created=resp.get("created", now),
            model=resp.get("model", model),
            choices=choices,
            usage=usage,
            system_fingerprint=resp.get("system_fingerprint"),
            model_extra={"usage": usage.model_dump() if usage else {}},
        )

    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        """Parse OpenAI streaming event."""
        data = data.strip()
        if not data or data == "[DONE]":
            return None

        try:
            event = orjson.loads(data)
        except orjson.JSONDecodeError as e:
            raise ResponseParseError(
                f"Failed to parse stream event: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        # Cache timestamp once for this event
        now = int(time.time())
        choices: list[ChunkChoice] = []

        for choice_data in event.get("choices", []):
            delta_data = choice_data.get("delta", {})

            # Parse tool call deltas
            tool_calls: list[dict[str, Any]] | None = None
            if "tool_calls" in delta_data:
                tool_calls = delta_data["tool_calls"]

            delta = ChunkDelta(
                role=delta_data.get("role"),
                content=delta_data.get("content"),
                tool_calls=tool_calls,
                function_call=delta_data.get("function_call"),
                reasoning_content=delta_data.get("reasoning_content")
                or delta_data.get("reasoning"),
            )

            choices.append(
                ChunkChoice(
                    index=choice_data.get("index", 0),
                    delta=delta,
                    finish_reason=choice_data.get("finish_reason"),
                    logprobs=choice_data.get("logprobs"),
                )
            )

        # Parse usage if present (with stream_options.include_usage=True)
        usage: Usage | None = None
        usage_data = event.get("usage")
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                prompt_tokens_details=usage_data.get("prompt_tokens_details"),
                completion_tokens_details=usage_data.get("completion_tokens_details"),
            )

        return StreamChunk(
            id=event.get("id", ""),
            object=event.get("object", "chat.completion.chunk"),
            created=event.get("created", now),
            model=event.get("model", model),
            choices=choices,
            usage=usage,
            system_fingerprint=event.get("system_fingerprint"),
        )

    def parse_error(
        self,
        status_code: int,
        data: bytes,
        request_id: str | None = None,
    ) -> ArcLLMError:
        """Parse OpenAI error response."""
        try:
            error_data = orjson.loads(data)
            error = error_data.get("error", {})
            message = error.get("message", "Unknown error")
            error_type = error.get("type", "")
            error_code = error.get("code", "")
        except (orjson.JSONDecodeError, UnicodeDecodeError):
            message = data.decode("utf-8", errors="replace")
            error_type = ""
            error_code = ""

        # Map to appropriate exception
        common_kwargs: dict[str, Any] = {
            "provider": self.provider_name,
            "status_code": status_code,
            "request_id": request_id,
        }
        message_lower = (message or "").lower()

        if status_code == 401:
            return AuthenticationError(message, **common_kwargs)
        if status_code == 402:
            return BudgetExceededError(message, **common_kwargs)
        if status_code == 429:
            # OpenAI uses 429 for both rate limits and quota exhaustion;
            # disambiguate via the message body so callers can branch.
            if any(
                token in message_lower
                for token in ("quota", "billing", "credit", "budget", "insufficient")
            ):
                return BudgetExceededError(message, **common_kwargs)
            return RateLimitError(message, **common_kwargs)
        if status_code == 400:
            # error_code can be string or int depending on provider
            error_code_str = str(error_code) if error_code is not None else ""
            error_code_lower = error_code_str.lower()
            if "content_filter" in error_code_lower or "content_policy" in message_lower:
                return ContentFilterError(
                    message,
                    filter_reason=error_code,
                    **common_kwargs,
                )
            return InvalidRequestError(message, **common_kwargs)
        if status_code == 404:
            return UnsupportedModelError(message, **common_kwargs)
        if status_code == 503:
            return ServiceUnavailableError(message, **common_kwargs)
        if status_code >= 500:
            return InternalServerError(message, **common_kwargs)
        return ProviderAPIError(
            message,
            error_type=error_type,
            error_code=error_code,
            **common_kwargs,
        )

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Build OpenAI embedding request."""
        body: dict[str, Any] = {
            "model": model,
            "input": input,
        }

        # Optional parameters
        if "encoding_format" in kwargs:
            body["encoding_format"] = kwargs["encoding_format"]
        if "dimensions" in kwargs:
            body["dimensions"] = kwargs["dimensions"]
        if "user" in kwargs:
            body["user"] = kwargs["user"]

        url = f"{self._api_base}/embeddings"
        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Parse OpenAI embedding response."""
        try:
            resp = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse embedding response: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        embeddings: list[EmbeddingData] = []
        for item in resp.get("data", []):
            embeddings.append(
                EmbeddingData(
                    index=item.get("index", 0),
                    embedding=item.get("embedding", []),
                    object=item.get("object", "embedding"),
                )
            )

        usage_data = resp.get("usage", {})
        usage = EmbeddingUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return EmbeddingResponse(
            model=resp.get("model", model),
            data=embeddings,
            usage=usage,
            object=resp.get("object", "list"),
        )

    # ------------------------------------------------------------------
    # Image generation surface
    # ------------------------------------------------------------------

    def build_image_generation_request(
        self,
        *,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> RequestData:
        """Build a request for ``POST /v1/images/generations``.

        Body matches OpenAI's spec verbatim — DALL-E 2/3 and gpt-image-1 all
        accept the same ``{model, prompt, n, size, quality, style,
        response_format, user}`` shape.
        """
        body: dict[str, Any] = {"model": model, "prompt": prompt}
        for key in ("n", "size", "quality", "style", "response_format", "user", "background"):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        url = f"{self._api_base}/images/generations"
        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=orjson.dumps(body),
            timeout=self.config.timeout,
        )

    def build_image_variation_request(
        self,
        *,
        model: str,
        image: bytes | str,
        **kwargs: Any,
    ) -> RequestData:
        """Build a multipart ``POST /v1/images/variations`` request."""
        body, headers = self._build_image_multipart(
            model=model,
            image=image,
            extras={
                k: kwargs[k]
                for k in ("n", "size", "response_format", "user")
                if k in kwargs and kwargs[k] is not None
            },
        )
        url = f"{self._api_base}/images/variations"
        return RequestData(
            method="POST",
            url=url,
            headers=headers,
            body=body,
            timeout=self.config.timeout,
        )

    def build_image_edit_request(
        self,
        *,
        model: str,
        image: bytes | str,
        prompt: str,
        mask: bytes | str | None = None,
        **kwargs: Any,
    ) -> RequestData:
        """Build a multipart ``POST /v1/images/edits`` request."""
        body, headers = self._build_image_multipart(
            model=model,
            image=image,
            mask=mask,
            extras={
                "prompt": prompt,
                **{
                    k: kwargs[k]
                    for k in ("n", "size", "response_format", "user", "quality")
                    if k in kwargs and kwargs[k] is not None
                },
            },
        )
        url = f"{self._api_base}/images/edits"
        return RequestData(
            method="POST",
            url=url,
            headers=headers,
            body=body,
            timeout=self.config.timeout,
        )

    def _build_image_multipart(
        self,
        *,
        model: str,
        image: bytes | str,
        mask: bytes | str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        """Compose a multipart body for the variation / edit endpoints."""
        import secrets
        from io import BytesIO

        boundary = f"arcllm{secrets.token_hex(16)}"
        buf = BytesIO()

        def _write_field(name: str, value: str) -> None:
            buf.write(f"--{boundary}\r\n".encode())
            buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            buf.write(value.encode("utf-8"))
            buf.write(b"\r\n")

        def _write_file(name: str, filename: str, data: bytes) -> None:
            buf.write(f"--{boundary}\r\n".encode())
            buf.write(
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                ).encode()
            )
            buf.write(b"Content-Type: application/octet-stream\r\n\r\n")
            buf.write(data)
            buf.write(b"\r\n")

        _write_field("model", model)
        for k, v in (extras or {}).items():
            _write_field(k, str(v))

        if isinstance(image, str):
            with open(image, "rb") as f:  # noqa: PTH123 — multipart wants raw bytes
                _write_file("image", "image.png", f.read())
        else:
            _write_file("image", "image.png", image)

        if mask is not None:
            if isinstance(mask, str):
                with open(mask, "rb") as f:  # noqa: PTH123
                    _write_file("mask", "mask.png", f.read())
            else:
                _write_file("mask", "mask.png", mask)

        buf.write(f"--{boundary}--\r\n".encode())

        headers = self._get_headers().copy()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        # Drop the cached JSON content-type — multipart needs its own.
        return buf.getvalue(), headers

    def parse_image_response(self, data: bytes, model: str) -> ImageResponse:
        """Parse an OpenAI images endpoint response."""
        try:
            resp = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse image response: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        rows: list[ImageData] = []
        raw_data: Any = resp.get("data") or []
        for raw_item in raw_data:
            if not isinstance(raw_item, dict):
                continue
            item = cast("dict[str, Any]", raw_item)
            rows.append(
                ImageData(
                    url=item.get("url"),
                    b64_json=item.get("b64_json"),
                    revised_prompt=item.get("revised_prompt"),
                )
            )
        return ImageResponse(
            created=int(resp.get("created", 0)),
            data=rows,
            model=str(resp.get("model") or model),
        )


# Register on import
register_provider("openai", OpenAIAdapter)
