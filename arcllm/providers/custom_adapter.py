"""
Custom-endpoint adapter for arcllm.

Fall-through adapter for any OpenAI-compatible HTTP endpoint. Callers
supply the base URL plus headers via :class:`ProviderConfig` — typical
targets are self-hosted vLLM, llama.cpp servers, LiteLLM / OpenRouter-style
proxies, and internal gateways.

The model id is passed through verbatim to ``model`` in the request body.
Auth is whatever the caller supplies in ``extra_headers`` (or, by default,
``Authorization: Bearer <api_key>``).

There is no env-var fallback here on purpose — "custom" means the caller
chooses the configuration explicitly.
"""

from __future__ import annotations

from arcllm.exceptions import InvalidRequestError
from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["CustomAdapter"]


class CustomAdapter(OpenAIAdapter):
    """Adapter for user-supplied OpenAI-compatible endpoints."""

    provider_name = "custom"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        if not config.api_base:
            raise InvalidRequestError(
                "custom provider requires `api_base` (the OpenAI-compatible endpoint URL)",
                provider=self.provider_name,
            )
        self._api_base = config.api_base

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


register_provider("custom", CustomAdapter)
