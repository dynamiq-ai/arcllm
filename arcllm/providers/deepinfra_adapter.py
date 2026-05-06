"""
DeepInfra adapter for arcllm.

DeepInfra serves open-weights LLMs and embedding models through an
OpenAI-compatible API. Inherits from :class:`OpenAIAdapter`.

API Documentation:
    - Docs: https://deepinfra.com/docs
    - Models: https://deepinfra.com/models
"""

from __future__ import annotations

from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["DeepInfraAdapter"]


class DeepInfraAdapter(OpenAIAdapter):
    """Adapter for DeepInfra API (OpenAI-compatible)."""

    provider_name = "deepinfra"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.deepinfra.com/v1/openai"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("DEEPINFRA_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


register_provider("deepinfra", DeepInfraAdapter)
