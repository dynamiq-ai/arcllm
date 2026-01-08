"""
Fireworks AI adapter for arcllm.

Fireworks AI provides an OpenAI-compatible API.
"""

from __future__ import annotations

from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["FireworksAdapter"]


class FireworksAdapter(OpenAIAdapter):
    """
    Adapter for Fireworks AI API.

    Fireworks AI uses OpenAI-compatible format.
    """

    provider_name = "fireworks_ai"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.fireworks.ai/inference/v1"

    def _build_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("FIREWORKS_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


# Register on import
register_provider("fireworks_ai", FireworksAdapter)
