"""
Image generation / variation / edit public API.

Drop-in for ``litellm.image_generation`` / ``litellm.image_variation`` /
``litellm.image_edit``. Routes through the same provider registry as
chat-completion calls; each adapter declares whether it supports images by
overriding the relevant ``build_image_*_request`` method on
:class:`arcllm.providers.base.BaseAdapter`.

Provider coverage in 0.4.0:

- **OpenAI** (``dall-e-3``, ``dall-e-2``, ``gpt-image-1``) — all three
  endpoints (generate / variation / edit).
- **Azure OpenAI** — inherits from OpenAI; works for any deployment that
  fronts an image-capable model.

Other providers (Stability, Bedrock Titan Image, Vertex Imagen) raise
:class:`UnsupportedModelError` until they're plumbed in.
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
    from arcllm.types import ImageResponse

__all__ = [
    "aimage_edit",
    "aimage_generation",
    "aimage_variation",
    "image_edit",
    "image_generation",
    "image_variation",
]


_IMAGE_CONFIG_PARAMS = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "api_version",
        "organization",
        "project",
        "timeout",
        "max_retries",
        "provider",
        "extra_headers",
    }
)


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split kwargs into ``(provider_config_args, request_args)``."""
    config_args: dict[str, Any] = {}
    request_args: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in _IMAGE_CONFIG_PARAMS:
            config_args[key] = value
        else:
            request_args[key] = value
    return config_args, request_args


def _build_provider_config(config_args: dict[str, Any]) -> ProviderConfig:
    return ProviderConfig(
        api_key=config_args.get("api_key"),
        api_base=config_args.get("api_base") or config_args.get("base_url"),
        api_version=config_args.get("api_version"),
        organization=config_args.get("organization"),
        project=config_args.get("project"),
        timeout=config_args.get("timeout", 60.0),
        max_retries=config_args.get("max_retries", 3),
        extra_headers=config_args.get("extra_headers") or {},
    )


def _resolve(model: str, config_args: dict[str, Any]) -> tuple[Adapter, str]:
    explicit_provider = config_args.get("provider")
    if explicit_provider:
        provider_name = explicit_provider
        model_id = model
    else:
        provider_name, model_id = parse_model_string(model)
    adapter = get_provider(provider_name, _build_provider_config(config_args))
    return adapter, model_id


def _exec_sync(adapter: Adapter, request: RequestData, model_id: str) -> ImageResponse:
    client = HTTPClient()
    try:
        raw = client.request(
            method=request.method,
            url=request.url,
            headers=request.headers,
            body=request.body,
            timeout=request.timeout,
        )
        # ``request`` returns ``HTTPResponse | Iterator[bytes]``; image calls
        # never set ``stream=True`` so the unbuffered branch is unreachable.
        assert isinstance(raw, HTTPResponse)
        result: HTTPResponse = raw
        if result.status_code >= 400:
            raise adapter.parse_error(
                result.status_code,
                result.body,
                result.request_id,
            )
        return adapter.parse_image_response(result.body, model_id)
    finally:
        client.close()


async def _exec_async(adapter: Adapter, request: RequestData, model_id: str) -> ImageResponse:
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
        result: AsyncHTTPResponse = raw
        if result.status_code >= 400:
            raise adapter.parse_error(
                result.status_code,
                result.body,
                result.request_id,
            )
        return adapter.parse_image_response(result.body, model_id)
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Public API: generation
# ---------------------------------------------------------------------------


def image_generation(
    *,
    model: str,
    prompt: str,
    **kwargs: Any,
) -> ImageResponse:
    """Generate one or more images from a text prompt.

    OpenAI-shape kwargs (``n``, ``size``, ``quality``, ``style``,
    ``response_format``, ``user``) pass through to the provider; provider
    config kwargs (``api_key``, ``api_base``, ``timeout``, ...) are
    extracted and applied to the underlying HTTP client.
    """
    config_args, request_args = _split_kwargs(kwargs)
    adapter, model_id = _resolve(model, config_args)
    request = adapter.build_image_generation_request(model=model_id, prompt=prompt, **request_args)
    return _exec_sync(adapter, request, model_id)


async def aimage_generation(
    *,
    model: str,
    prompt: str,
    **kwargs: Any,
) -> ImageResponse:
    """Async equivalent of :func:`image_generation`."""
    config_args, request_args = _split_kwargs(kwargs)
    adapter, model_id = _resolve(model, config_args)
    request = adapter.build_image_generation_request(model=model_id, prompt=prompt, **request_args)
    return await _exec_async(adapter, request, model_id)


# ---------------------------------------------------------------------------
# Public API: variation
# ---------------------------------------------------------------------------


def image_variation(
    *,
    model: str,
    image: bytes | str,
    **kwargs: Any,
) -> ImageResponse:
    """Generate variations of a source image.

    ``image`` is either raw bytes (PNG) or a path to a local PNG file.
    """
    config_args, request_args = _split_kwargs(kwargs)
    adapter, model_id = _resolve(model, config_args)
    request = adapter.build_image_variation_request(model=model_id, image=image, **request_args)
    return _exec_sync(adapter, request, model_id)


async def aimage_variation(
    *,
    model: str,
    image: bytes | str,
    **kwargs: Any,
) -> ImageResponse:
    """Async equivalent of :func:`image_variation`."""
    config_args, request_args = _split_kwargs(kwargs)
    adapter, model_id = _resolve(model, config_args)
    request = adapter.build_image_variation_request(model=model_id, image=image, **request_args)
    return await _exec_async(adapter, request, model_id)


# ---------------------------------------------------------------------------
# Public API: edit
# ---------------------------------------------------------------------------


def image_edit(
    *,
    model: str,
    image: bytes | str,
    prompt: str,
    mask: bytes | str | None = None,
    **kwargs: Any,
) -> ImageResponse:
    """Edit an image given a prompt and optional mask.

    ``image`` and ``mask`` are either raw PNG bytes or paths to local PNGs.
    """
    config_args, request_args = _split_kwargs(kwargs)
    adapter, model_id = _resolve(model, config_args)
    request = adapter.build_image_edit_request(
        model=model_id, image=image, prompt=prompt, mask=mask, **request_args
    )
    return _exec_sync(adapter, request, model_id)


async def aimage_edit(
    *,
    model: str,
    image: bytes | str,
    prompt: str,
    mask: bytes | str | None = None,
    **kwargs: Any,
) -> ImageResponse:
    """Async equivalent of :func:`image_edit`."""
    config_args, request_args = _split_kwargs(kwargs)
    adapter, model_id = _resolve(model, config_args)
    request = adapter.build_image_edit_request(
        model=model_id, image=image, prompt=prompt, mask=mask, **request_args
    )
    return await _exec_async(adapter, request, model_id)
