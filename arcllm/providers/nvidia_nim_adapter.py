"""
NVIDIA NIM adapter for arcllm.

NVIDIA NIM (NVIDIA Inference Microservices) exposes optimized open-weights
models behind an OpenAI-compatible API. Inherits from :class:`OpenAIAdapter`.

API Documentation:
    - Docs: https://docs.api.nvidia.com/nim/
    - Models catalog: https://build.nvidia.com/explore/discover
"""

from __future__ import annotations

from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["NvidiaNIMAdapter"]


class NvidiaNIMAdapter(OpenAIAdapter):
    """Adapter for NVIDIA NIM API (OpenAI-compatible)."""

    provider_name = "nvidia_nim"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://integrate.api.nvidia.com/v1"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("NVIDIA_NIM_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


register_provider("nvidia_nim", NvidiaNIMAdapter)
