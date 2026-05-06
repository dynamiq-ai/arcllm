"""
Rerank public API.

Drop-in for ``litellm.rerank``. Routes through the same provider registry as
chat-completion calls; each adapter declares whether it supports reranking by
overriding :meth:`arcllm.providers.base.BaseAdapter.build_rerank_request` /
``parse_rerank_response``.

Provider coverage in 0.4.0:

- **Cohere** (``rerank-v3.5``, ``rerank-multilingual-v3.0``,
  ``rerank-v4.0``) — primary surface used by dynamiq's ``CohereReranker``.

Voyage / Bedrock / Jina rerank can be added in 0.4.x as additional adapters
override these two methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcllm.http.async_client import AsyncHTTPClient, AsyncHTTPResponse
from arcllm.http.client import HTTPClient, HTTPResponse
from arcllm.providers.base import (
    Adapter,
    ProviderConfig,
    RequestData,
    get_provider,
    parse_model_string,
)

if TYPE_CHECKING:
    from arcllm.types import RerankResponse

__all__ = ["arerank", "rerank"]


_RERANK_CONFIG_PARAMS = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "api_version",
        "timeout",
        "max_retries",
        "provider",
        "extra_headers",
    }
)


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config_args: dict[str, Any] = {}
    request_args: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in _RERANK_CONFIG_PARAMS:
            config_args[key] = value
        else:
            request_args[key] = value
    return config_args, request_args


def _resolve(model: str, config_args: dict[str, Any]) -> tuple[Adapter, str]:
    explicit_provider = config_args.get("provider")
    if explicit_provider:
        provider_name = explicit_provider
        model_id = model
    else:
        provider_name, model_id = parse_model_string(model)
    config = ProviderConfig(
        api_key=config_args.get("api_key"),
        api_base=config_args.get("api_base") or config_args.get("base_url"),
        api_version=config_args.get("api_version"),
        timeout=config_args.get("timeout", 60.0),
        max_retries=config_args.get("max_retries", 3),
        extra_headers=config_args.get("extra_headers") or {},
    )
    adapter = get_provider(provider_name, config)
    return adapter, model_id


def _exec_sync(adapter: Adapter, request: RequestData, model_id: str) -> RerankResponse:
    client = HTTPClient()
    try:
        raw = client.request(
            method=request.method,
            url=request.url,
            headers=request.headers,
            body=request.body,
            timeout=request.timeout,
        )
        assert isinstance(raw, HTTPResponse)
        if raw.status_code >= 400:
            raise adapter.parse_error(raw.status_code, raw.body, raw.request_id)
        return adapter.parse_rerank_response(raw.body, model_id)
    finally:
        client.close()


async def _exec_async(adapter: Adapter, request: RequestData, model_id: str) -> RerankResponse:
    client = AsyncHTTPClient()
    try:
        raw = await client.request(
            method=request.method,
            url=request.url,
            headers=request.headers,
            body=request.body,
            timeout=request.timeout,
        )
        assert isinstance(raw, AsyncHTTPResponse)
        if raw.status_code >= 400:
            raise adapter.parse_error(raw.status_code, raw.body, raw.request_id)
        return adapter.parse_rerank_response(raw.body, model_id)
    finally:
        await client.close()


def rerank(
    *,
    model: str,
    query: str,
    documents: list[str],
    top_n: int | None = None,
    return_documents: bool = True,
    **kwargs: Any,
) -> RerankResponse:
    """Rerank ``documents`` against ``query`` using the named model.

    Returns a :class:`arcllm.types.RerankResponse` whose ``results`` list is
    sorted by descending relevance score. ``top_n`` caps the number of
    returned results; ``return_documents`` controls whether each result
    carries the original document text (turn off for token savings if you
    only need the indices).
    """
    config_args, request_args = _split_kwargs(kwargs)
    adapter, model_id = _resolve(model, config_args)
    request = adapter.build_rerank_request(
        model=model_id,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=return_documents,
        **request_args,
    )
    return _exec_sync(adapter, request, model_id)


async def arerank(
    *,
    model: str,
    query: str,
    documents: list[str],
    top_n: int | None = None,
    return_documents: bool = True,
    **kwargs: Any,
) -> RerankResponse:
    """Async equivalent of :func:`rerank`."""
    config_args, request_args = _split_kwargs(kwargs)
    adapter, model_id = _resolve(model, config_args)
    request = adapter.build_rerank_request(
        model=model_id,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=return_documents,
        **request_args,
    )
    return await _exec_async(adapter, request, model_id)
