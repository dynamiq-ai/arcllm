"""
OVHcloud AI Endpoints adapter for arcllm.

OVHcloud's serverless LLM offering hosts open-weights models (DeepSeek,
Llama, Qwen, Mistral, gpt-oss) on European GPU infrastructure with an
OpenAI-compatible API at
``https://oai.endpoints.kepler.ai.cloud.ovh.net/v1``. Inherits from
:class:`OpenAIAdapter`.

API Documentation:
    - Docs: https://help.ovhcloud.com/csm/en-public-cloud-ai-endpoints
    - Catalog: https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/
"""

from __future__ import annotations

from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["OVHCloudAdapter"]


class OVHCloudAdapter(OpenAIAdapter):
    """Adapter for OVHcloud AI Endpoints (OpenAI-compatible)."""

    provider_name = "ovhcloud"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("OVHCLOUD_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


register_provider("ovhcloud", OVHCloudAdapter)
