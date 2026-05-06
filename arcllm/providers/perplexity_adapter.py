"""
Perplexity adapter for arcllm.

Perplexity provides an OpenAI-compatible API with search capabilities.
The wire shape mirrors OpenAI Chat Completions, with two key extensions
that this adapter exposes through arcllm's typed surface:

- ``response.citations``: a top-level array of source URLs (legacy shape)
  or richer ``search_results`` objects with ``url`` / ``title`` /
  ``snippet`` (newer shape). Both flow into ``Message.citations``.
- Search-context controls (``search_domain_filter`` etc.) pass through
  unchanged on the request side; they're already accepted by the OpenAI
  adapter base because they're treated as opaque kwargs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import orjson

from arcllm.exceptions import ResponseParseError, UnsupportedModelError
from arcllm.providers.base import (
    COMMON_PARAMS,
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter
from arcllm.types import Citation

if TYPE_CHECKING:
    from arcllm.types import EmbeddingResponse, ModelResponse

__all__ = ["PerplexityAdapter"]


def _coerce_citations(payload: dict[str, Any]) -> list[Citation] | None:
    """Best-effort conversion of Perplexity's top-level citation fields.

    Perplexity has shipped two response shapes over time:

    1. Legacy: ``"citations": ["https://...", ...]`` — bare URLs.
    2. Current: ``"search_results": [{"url": ..., "title": ..., "snippet": ...}, ...]``.

    We accept both. ``citations`` wins if both are present (most recent
    requests still set it). Returns ``None`` if neither field is present so
    that ``Message.citations`` stays ``None`` for non-grounded responses.
    """
    raw_citations = cast("list[Any]", payload.get("citations") or [])
    if raw_citations:
        out: list[Citation] = []
        for item in raw_citations:
            if isinstance(item, str) and item:
                out.append(Citation(url=item))
                continue
            if not isinstance(item, dict):
                continue
            item_dict = cast("dict[str, Any]", item)
            url = item_dict.get("url")
            if not url:
                continue
            out.append(
                Citation(
                    url=str(url),
                    title=item_dict.get("title"),
                    snippet=item_dict.get("snippet"),
                )
            )
        if out:
            return out

    raw_search = cast("list[Any]", payload.get("search_results") or [])
    if raw_search:
        out = []
        for item in raw_search:
            if not isinstance(item, dict):
                continue
            item_dict = cast("dict[str, Any]", item)
            url = item_dict.get("url")
            if not url:
                continue
            out.append(
                Citation(
                    url=str(url),
                    title=item_dict.get("title"),
                    snippet=item_dict.get("snippet") or item_dict.get("description"),
                )
            )
        if out:
            return out

    return None


class PerplexityAdapter(OpenAIAdapter):
    """Adapter for Perplexity API. Inherits OpenAI's request building."""

    provider_name = "perplexity"

    # Perplexity-specific search controls layered on top of the OpenAI surface.
    # https://docs.perplexity.ai/api-reference/chat-completions-post
    _PERPLEXITY_SEARCH_PARAMS = (
        "search_domain_filter",
        "search_recency_filter",
        "search_after_date_filter",
        "search_before_date_filter",
        "return_images",
        "return_related_questions",
        "web_search_options",
    )
    supported_params = COMMON_PARAMS | set(_PERPLEXITY_SEARCH_PARAMS)

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build a Perplexity request, threading the search-specific params.

        We call ``OpenAIAdapter.build_request`` for the OpenAI-shape body and
        then post-fix the wire body with Perplexity's search controls, which
        the OpenAI adapter doesn't know about.
        """
        # Pull search-specific kwargs aside before delegating to OpenAI; the
        # base adapter would otherwise drop them as it builds the body.
        search_kwargs: dict[str, Any] = {}
        for name in self._PERPLEXITY_SEARCH_PARAMS:
            if name in kwargs and kwargs[name] is not None:
                search_kwargs[name] = kwargs.pop(name)

        request = super().build_request(
            model=model,
            messages=messages,
            stream=stream,
            drop_params=drop_params,
            **kwargs,
        )
        if not search_kwargs or request.body is None:
            return request

        body = orjson.loads(request.body)
        body.update(search_kwargs)
        return RequestData(
            method=request.method,
            url=request.url,
            headers=request.headers,
            body=orjson.dumps(body),
            timeout=request.timeout,
        )

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.perplexity.ai"

    def _build_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("PERPLEXITY_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Parse Perplexity response, attaching citations to each message.

        Perplexity returns citations at the **response top level**, not on the
        message. We attach them to every assistant message in the response
        (Perplexity always returns ``n=1``, so this is unambiguous).
        """
        response = super().parse_response(data, model)
        try:
            payload: dict[str, Any] = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as exc:
            # The base class already errored if the JSON was bad; if we
            # somehow got here with garbage, surface a parse error rather
            # than silently dropping citations.
            raise ResponseParseError(
                f"Failed to re-parse Perplexity response for citations: {exc}",
                provider=self.provider_name,
                raw_data=data,
            ) from exc

        citations = _coerce_citations(payload)
        if citations:
            for choice in response.choices:
                if choice.message.role == "assistant":
                    choice.message.citations = citations
        return response

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Perplexity does not support embeddings."""
        raise UnsupportedModelError(
            "Perplexity does not provide an embeddings API",
            provider=self.provider_name,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Perplexity does not support embeddings."""
        raise UnsupportedModelError(
            "Perplexity does not provide an embeddings API",
            provider=self.provider_name,
        )


# Register on import
register_provider("perplexity", PerplexityAdapter)
