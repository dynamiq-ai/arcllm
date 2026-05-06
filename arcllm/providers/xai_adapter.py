"""
xAI adapter for arcllm.

xAI's Grok family is served behind an OpenAI-compatible API at
``https://api.x.ai/v1``. We inherit from :class:`OpenAIAdapter` and only
swap the base URL and auth env var.

API Documentation:
    - Docs: https://docs.x.ai/docs
    - Models: https://docs.x.ai/docs/models
    - Chat API: https://docs.x.ai/docs/api-reference#chat-completions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcllm.exceptions import UnsupportedModelError
from arcllm.providers.base import (
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

if TYPE_CHECKING:
    from arcllm.types import EmbeddingResponse

__all__ = ["XAIAdapter"]


class XAIAdapter(OpenAIAdapter):
    """Adapter for xAI Grok API (OpenAI-compatible)."""

    provider_name = "xai"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.x.ai/v1"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("XAI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        raise UnsupportedModelError(
            "xAI does not provide an embeddings API",
            provider=self.provider_name,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        raise UnsupportedModelError(
            "xAI does not provide an embeddings API",
            provider=self.provider_name,
        )


register_provider("xai", XAIAdapter)
