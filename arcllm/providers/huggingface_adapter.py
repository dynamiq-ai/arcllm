"""
HuggingFace Inference adapter for arcllm.

Targets the modern Hub Inference / Inference Endpoints API which now exposes
OpenAI-compatible ``/v1/chat/completions``. Inherits from
:class:`OpenAIAdapter`; only the base URL and auth env var differ.

For dedicated Inference Endpoints, callers pass ``api_base`` explicitly to
the deployment URL (e.g. ``https://abc.us-east-1.aws.endpoints.huggingface.cloud/v1``).

API Documentation:
    - Hub Inference: https://huggingface.co/docs/inference-providers
    - Inference Endpoints: https://huggingface.co/docs/inference-endpoints

Note: legacy ``/models/{model}`` text-generation endpoints (with
``inputs``/``parameters`` body shape) are not served through this adapter.
Call them directly via ``custom`` provider with a transformer if needed.
"""

from __future__ import annotations

from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["HuggingFaceAdapter"]


class HuggingFaceAdapter(OpenAIAdapter):
    """Adapter for HuggingFace Hub Inference / Inference Endpoints (OpenAI-compatible)."""

    provider_name = "huggingface"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://router.huggingface.co/v1"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("HUGGINGFACE_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


register_provider("huggingface", HuggingFaceAdapter)
