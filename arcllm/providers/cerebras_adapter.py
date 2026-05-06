"""
Cerebras adapter for arcllm.

Cerebras Inference exposes Llama and Qwen models served on their CS-3
wafer-scale hardware behind an OpenAI-compatible API. Inherits from
:class:`OpenAIAdapter`.

API Documentation:
    - Docs: https://inference-docs.cerebras.ai/
    - Models: https://inference-docs.cerebras.ai/api-reference/models
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

__all__ = ["CerebrasAdapter"]


class CerebrasAdapter(OpenAIAdapter):
    """Adapter for Cerebras Inference API (OpenAI-compatible)."""

    provider_name = "cerebras"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.cerebras.ai/v1"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("CEREBRAS_API_KEY")
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
            "Cerebras does not provide an embeddings API",
            provider=self.provider_name,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        raise UnsupportedModelError(
            "Cerebras does not provide an embeddings API",
            provider=self.provider_name,
        )


register_provider("cerebras", CerebrasAdapter)
