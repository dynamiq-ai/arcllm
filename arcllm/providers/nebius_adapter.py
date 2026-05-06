"""
Nebius AI Studio adapter for arcllm.

Nebius hosts a curated open-weights catalog (Llama, Qwen, DeepSeek,
Mistral, Granite, Gemma, Nemotron, Hermes) on its own GPU cloud, and
serves them through an OpenAI-compatible chat-completions API at
``https://api.studio.nebius.ai/v1``. Inherits from :class:`OpenAIAdapter`.

API Documentation:
    - Docs: https://docs.nebius.com/studio/inference
    - Models / pricing: https://nebius.com/services/studio
"""

from __future__ import annotations

from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["NebiusAdapter"]


class NebiusAdapter(OpenAIAdapter):
    """Adapter for Nebius AI Studio (OpenAI-compatible)."""

    provider_name = "nebius"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.studio.nebius.ai/v1"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("NEBIUS_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


register_provider("nebius", NebiusAdapter)
