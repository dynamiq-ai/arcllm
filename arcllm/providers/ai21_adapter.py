"""
AI21 Studio adapter for arcllm.

AI21's Jamba family is served at ``https://api.ai21.com/studio/v1`` with an
OpenAI-shape ``/chat/completions`` body. We inherit from
:class:`OpenAIAdapter`; the only differences are the base URL and that the
``/embeddings`` endpoint accepts a ``texts`` field (we override the
embedding builder to translate).

API Documentation:
    - Docs: https://docs.ai21.com/
    - Models: https://docs.ai21.com/reference/jamba-15-api-ref
    - Embeddings: https://docs.ai21.com/reference/embeddings-api-ref
"""

from __future__ import annotations

from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["AI21Adapter"]


class AI21Adapter(OpenAIAdapter):
    """Adapter for AI21 Studio API (OpenAI-shape chat + custom embeddings)."""

    provider_name = "ai21"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.ai21.com/studio/v1"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("AI21_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


register_provider("ai21", AI21Adapter)
