"""
Moonshot AI (Kimi) adapter for arcllm.

Moonshot serves the Kimi family (Kimi K2.5 / K2.6 / K2-thinking / Kimi
latest, plus the older Moonshot v1 8k/32k/128k variants) at
``https://api.moonshot.ai/v1`` with an OpenAI-compatible chat surface.
Inherits from :class:`OpenAIAdapter`.

**Caller-side responsibilities** — Moonshot deviates from pure OpenAI
in a few places that arcllm does not auto-translate. Pass kwargs that
already comply:

- **Temperature range**: Moonshot clamps ``temperature`` to ``[0, 1]``
  (OpenAI accepts ``[0, 2]``). Pass values in [0, 1] to be safe.
- **Multimodal content arrays** are only honored on the vision/video
  models (``kimi-k2.5``, ``kimi-k2.6``, ``kimi-latest``, the
  ``moonshot-v1-*-vision-preview`` line). For text-only models, pass a
  string in ``content`` rather than a content-block array.
- **``tool_choice="required"``** is not supported by Moonshot — omit it
  (use ``"auto"`` instead) or rely on tool definitions to encourage
  selection.
- **Reasoning models** (``kimi-thinking-preview``, ``kimi-k2-thinking``,
  ``kimi-k2.6``) emit a ``reasoning_content`` field in their delta
  chunks; arcllm passes it through unchanged on the assistant message.

API Documentation:
    - Docs: https://platform.moonshot.ai/docs
    - Models: https://platform.moonshot.ai/docs/models
"""

from __future__ import annotations

from arcllm.providers.base import (
    ProviderConfig,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["MoonshotAdapter"]


class MoonshotAdapter(OpenAIAdapter):
    """Adapter for Moonshot AI / Kimi (OpenAI-compatible chat)."""

    provider_name = "moonshot"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.moonshot.ai/v1"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("MOONSHOT_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers


register_provider("moonshot", MoonshotAdapter)
