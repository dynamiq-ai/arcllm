"""
Together AI adapter for arcllm.

Together AI provides an OpenAI-compatible API.
"""

from __future__ import annotations

from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["TogetherAdapter"]


class TogetherAdapter(OpenAIAdapter):
    """
    Adapter for Together AI API.

    Together AI uses OpenAI-compatible format.
    """

    provider_name = "together_ai"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.together.xyz/v1"

    def _build_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("TOGETHER_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


# Register on import
register_provider("together_ai", TogetherAdapter)
