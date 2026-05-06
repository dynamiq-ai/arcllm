"""
Z.AI / Zhipu AI (GLM) adapter for arcllm.

Z.AI hosts the GLM family (GLM-4.5, GLM-4.6, GLM-5, plus reasoning,
vision, and code variants) at ``https://api.z.ai/api/paas/v4`` with an
OpenAI-compatible chat surface. Inherits from :class:`OpenAIAdapter`.

GLM exposes a ``thinking`` request parameter that toggles reasoning
mode on supported models — pass it as a kwarg to ``arcllm.completion``
and the OpenAI request builder forwards it through unchanged.

API Documentation:
    - English docs: https://docs.z.ai/
    - GLM-4.6 reference: https://docs.z.ai/guides/llm/glm-4.6
    - Pricing: https://docs.z.ai/guides/overview/pricing
"""

from __future__ import annotations

from arcllm.providers.base import (
    COMMON_PARAMS,
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["ZAIAdapter"]


class ZAIAdapter(OpenAIAdapter):
    """Adapter for Z.AI / Zhipu (OpenAI-compatible chat)."""

    provider_name = "zai"

    # GLM-specific param: ``thinking`` toggles reasoning-mode on supported
    # GLM models. Adding it to ``supported_params`` lets the capability
    # filter pass it through instead of dropping it as unknown.
    supported_params = COMMON_PARAMS | {"thinking"}

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.z.ai/api/paas/v4"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("ZAI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


register_provider("zai", ZAIAdapter)
