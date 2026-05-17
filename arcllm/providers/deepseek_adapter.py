"""
DeepSeek adapter for arcllm.

DeepSeek provides an OpenAI-compatible API.
"""

from __future__ import annotations

from typing import Any

from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["DeepSeekAdapter"]


class DeepSeekAdapter(OpenAIAdapter):
    """
    Adapter for DeepSeek API.

    DeepSeek uses OpenAI-compatible format.
    """

    provider_name = "deepseek"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.deepseek.com"

    def _build_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("DEEPSEEK_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def _extract_cache_tokens(
        self,
        usage_data: dict[str, Any],
    ) -> tuple[int | None, int | None]:
        """DeepSeek reports cache hits as a top-level
        ``prompt_cache_hit_tokens`` field (90% discount). Sibling
        ``prompt_cache_miss_tokens`` is informational — billed at the
        full input rate, no separate cache-write surcharge.
        """
        hit = usage_data.get("prompt_cache_hit_tokens")
        if hit is not None:
            return (int(hit), None)
        # Fall through for DeepSeek-via-proxy responses that pass through
        # the OpenAI-style nested ``prompt_tokens_details`` shape.
        return super()._extract_cache_tokens(usage_data)


# Register on import
register_provider("deepseek", DeepSeekAdapter)
