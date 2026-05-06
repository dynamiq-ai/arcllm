"""
Azure adapter for arcllm.

Azure exposes two distinct serving surfaces, both reachable through this
adapter (registered as ``azure`` and aliased as ``azure_ai``):

- **Azure OpenAI Service** — GPT-5 / 4o / 4.1, o-series reasoning, OpenAI
  embeddings. Hosted at ``{resource}.openai.azure.com``; URL pattern
  ``/openai/deployments/{deployment}/chat/completions?api-version=...``.
- **Azure AI Foundry serverless** — Phi, Llama, Cohere, Mistral, etc. on
  the unified ``/models/chat/completions?api-version=...`` endpoint at
  ``{resource}.services.ai.azure.com``. Wire format is OpenAI-compatible
  (Azure normalised it across vendors).

Surface dispatch is **host-first**: the configured ``api_base`` host
suffix decides which URL pattern to build. A model-name prefix sniff
covers the rare case of an unfamiliar host (e.g. private endpoints
without a canonical Azure hostname).
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import orjson

from arcllm.exceptions import (
    ArcLLMError,
    AuthenticationError,
)
from arcllm.providers.base import (
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["AzureOpenAIAdapter"]


_OPENAI_FAMILY_PREFIXES = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "text-embedding-",
    "chatgpt-",
)


def _is_foundry_host(api_base: str | None) -> bool:
    """True if ``api_base`` points at an Azure AI Foundry endpoint.

    Foundry serverless URLs end in ``.services.ai.azure.com``; the
    OpenAI-Service style ends in ``.openai.azure.com``. Host-based
    detection is more reliable than model-prefix sniffing because users
    can deploy custom-named OpenAI Service models (e.g. fine-tuned
    ``my-gpt`` deployments) that wouldn't match a static prefix list.
    """
    if not api_base:
        return False
    try:
        host = urlparse(api_base).hostname or ""
    except (ValueError, AttributeError):
        return False
    return host.endswith(".services.ai.azure.com")


def _is_azure_openai_model(model: str) -> bool:
    """Fallback: True for OpenAI-named models when host detection is ambiguous.

    Used only when the configured ``api_base`` doesn't disclose which
    Azure surface we're hitting (rare — usually the host gives it away).
    """
    m = model.lower()
    return any(m.startswith(p) for p in _OPENAI_FAMILY_PREFIXES)


class AzureOpenAIAdapter(OpenAIAdapter):
    """
    Adapter for Azure OpenAI Service.

    Inherits from OpenAIAdapter since Azure uses the same API format,
    but requires different authentication and endpoint handling.
    """

    provider_name = "azure"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_version = config.api_version or "2024-10-21"
        self._deployment = config.azure_deployment
        self._ad_token = config.azure_ad_token

    def _get_api_base(self) -> str:
        """Get Azure API base URL."""
        base = self.config.api_base
        if not base:
            # Try environment variable
            base = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not base:
            raise ArcLLMError(
                "Azure OpenAI endpoint not provided. Set AZURE_OPENAI_ENDPOINT "
                "or pass api_base parameter.",
                provider=self.provider_name,
            )
        return base.rstrip("/")

    def _build_headers(self) -> dict[str, str]:
        """Get request headers for Azure."""
        headers = {"Content-Type": "application/json"}

        # Azure supports both API key and Azure AD token authentication
        if self._ad_token or os.environ.get("AZURE_OPENAI_AD_TOKEN"):
            token = self._ad_token or os.environ.get("AZURE_OPENAI_AD_TOKEN")
            headers["Authorization"] = f"Bearer {token}"
        else:
            api_key = self.config.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise AuthenticationError(
                    "Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY "
                    "or pass api_key parameter.",
                    provider=self.provider_name,
                )
            headers["api-key"] = api_key

        if self.config.extra_headers:
            headers.update(self.config.extra_headers)

        return headers

    def _get_deployment(self, model: str) -> str:
        """Get deployment name from model or config."""
        # Deployment can be specified in config, or we use the model name
        return self._deployment or model

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build Azure OpenAI chat completion request."""
        # Check params (same as OpenAI)
        kwargs = self._check_params(model, drop_params, **kwargs)

        # Build request body (same format as OpenAI)
        body: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
        }

        # Add stream_options for usage in streaming
        if stream:
            stream_options = kwargs.pop("stream_options", None)
            if stream_options:
                body["stream_options"] = stream_options

        # Add optional parameters (same as OpenAI)
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

        # Body needs the model id when targeting Foundry serverless.
        if not _is_azure_openai_model(model):
            body["model"] = model

        body_bytes = orjson.dumps(body)
        return RequestData(
            method="POST",
            url=self._chat_url(model),
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def _chat_url(self, model: str) -> str:
        """Resolve the chat-completions URL for ``model`` per Azure family.

        Detection order (most-reliable first):

        1. **Host suffix** — ``.services.ai.azure.com`` is unambiguously
           Foundry; ``.openai.azure.com`` is unambiguously OpenAI Service.
           Custom-deployed model names (e.g. fine-tunes called ``my-gpt``)
           don't fool this check.
        2. **Model-name fallback** — when the host is unfamiliar, fall
           back to the legacy prefix sniff. Used only for callers that
           point at a private endpoint without one of the canonical
           Azure suffixes.
        """
        api_base = self._get_api_base()
        if _is_foundry_host(api_base):
            return f"{api_base}/models/chat/completions?api-version={self._api_version}"
        if _is_azure_openai_model(model) or "openai.azure.com" in (api_base or ""):
            deployment = self._get_deployment(model)
            return (
                f"{api_base}/openai/deployments/{deployment}"
                f"/chat/completions?api-version={self._api_version}"
            )
        # Unknown host + non-OpenAI model name → assume Foundry (the
        # body-driven route works on most private deployments).
        return f"{api_base}/models/chat/completions?api-version={self._api_version}"

    def _embedding_url(self, model: str) -> str:
        """Resolve the embeddings URL for ``model`` — same dispatch as ``_chat_url``."""
        api_base = self._get_api_base()
        if _is_foundry_host(api_base):
            return f"{api_base}/models/embeddings?api-version={self._api_version}"
        if _is_azure_openai_model(model) or "openai.azure.com" in (api_base or ""):
            deployment = self._get_deployment(model)
            return (
                f"{api_base}/openai/deployments/{deployment}"
                f"/embeddings?api-version={self._api_version}"
            )
        return f"{api_base}/models/embeddings?api-version={self._api_version}"

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Build Azure embedding request (OpenAI Service or Foundry)."""
        body: dict[str, Any] = {"input": input}
        if not _is_azure_openai_model(model):
            body["model"] = model
        if "encoding_format" in kwargs:
            body["encoding_format"] = kwargs["encoding_format"]
        if "dimensions" in kwargs:
            body["dimensions"] = kwargs["dimensions"]
        if "user" in kwargs:
            body["user"] = kwargs["user"]

        body_bytes = orjson.dumps(body)
        return RequestData(
            method="POST",
            url=self._embedding_url(model),
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )


# Register on import
register_provider("azure", AzureOpenAIAdapter)
# Alias for callers using the Azure AI Foundry-style provider name (parity
# with dynamiq's ``AzureAI`` node, which addresses Foundry deployments
# directly).
register_provider("azure_ai", AzureOpenAIAdapter)
