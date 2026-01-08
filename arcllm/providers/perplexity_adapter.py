"""
Perplexity adapter for arcllm.

Perplexity provides an OpenAI-compatible API with search capabilities.
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

__all__ = ["PerplexityAdapter"]


class PerplexityAdapter(OpenAIAdapter):
    """
    Adapter for Perplexity API.

    Perplexity uses OpenAI-compatible format with additional search features.
    """

    provider_name = "perplexity"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.perplexity.ai"

    def _build_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("PERPLEXITY_API_KEY")
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
        """Perplexity does not support embeddings."""
        raise UnsupportedModelError(
            "Perplexity does not provide an embeddings API",
            provider=self.provider_name,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Perplexity does not support embeddings."""
        raise UnsupportedModelError(
            "Perplexity does not provide an embeddings API",
            provider=self.provider_name,
        )


# Register on import
register_provider("perplexity", PerplexityAdapter)
