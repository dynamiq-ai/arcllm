"""
OpenRouter adapter for arcllm.

OpenRouter is a unified gateway to 100+ models from many vendors, exposed
through an OpenAI-compatible API. Inherits from :class:`OpenAIAdapter`.

OpenRouter recommends two optional headers for app attribution:
- ``HTTP-Referer``: site/app URL
- ``X-Title``: app name

These can be passed via ``extra_headers`` on :class:`ProviderConfig` or via
the ``OPENROUTER_REFERER`` / ``OPENROUTER_APP_NAME`` env vars.

API Documentation:
    - Docs: https://openrouter.ai/docs
    - Models: https://openrouter.ai/docs/models
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

from arcllm.exceptions import UnsupportedModelError
from arcllm.providers.base import (
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

if TYPE_CHECKING:
    from arcllm.types import EmbeddingResponse, ModelResponse

__all__ = ["OpenRouterAdapter"]


class OpenRouterAdapter(OpenAIAdapter):
    """Adapter for OpenRouter (OpenAI-compatible gateway)."""

    provider_name = "openrouter"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://openrouter.ai/api/v1"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("OPENROUTER_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_APP_NAME")
        if title:
            headers["X-Title"] = title
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        raise UnsupportedModelError(
            "OpenRouter does not provide an embeddings API",
            provider=self.provider_name,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        raise UnsupportedModelError(
            "OpenRouter does not provide an embeddings API",
            provider=self.provider_name,
        )

    def post_process_response(
        self,
        response: ModelResponse,
        headers: dict[str, str],
    ) -> ModelResponse:
        """Lift OpenRouter's ``x-openrouter-cost`` response header (USD
        float string) onto :attr:`ModelResponse.provider_reported_cost`.

        This is the authoritative cost for the call — accounts for the
        routed provider's price plus OpenRouter's markup, which a static
        per-model table can't represent honestly. ``completion_cost()``
        prefers this value over any table lookup.

        Header lookup is case-insensitive (httpx normalizes to lower but
        some intermediaries don't). Malformed values are swallowed so the
        caller falls back to the static table instead of crashing.
        """
        for key, value in headers.items():
            if key.lower() == "x-openrouter-cost":
                with contextlib.suppress(TypeError, ValueError):
                    response.provider_reported_cost = float(value)
                break
        return response


register_provider("openrouter", OpenRouterAdapter)
