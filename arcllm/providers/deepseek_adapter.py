"""
DeepSeek adapter for arcllm.

DeepSeek provides an OpenAI-compatible API.
"""

from __future__ import annotations

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


# Register on import
register_provider("deepseek", DeepSeekAdapter)
