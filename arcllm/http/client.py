"""
Synchronous HTTP client using httpx.

Features:
- Connection pooling (per-host, with keep-alive)
- HTTP/2 support
- TLS support with optimized ciphers
- Timeout handling
- Streaming response support
- Automatic retries
- Gzip/deflate decompression
"""

from __future__ import annotations

import contextlib
import os
import ssl
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import orjson

from arcllm.exceptions import (
    ConnectionError,
    ProviderAPIError,
    TimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["HTTPClient", "HTTPResponse"]

# Module-level SSL context cache (matches litellm's approach)
# Created lazily on first use to avoid import-time overhead
_ssl_context: ssl.SSLContext | None = None


def _get_ssl_context() -> ssl.SSLContext:
    """
    Get or create a cached SSL context with optimized settings.

    Performance optimizations (matching litellm):
    - Minimum TLS 1.2 for faster handshakes
    - Optimized cipher suite ordering
    - Cached to avoid repeated context creation (saves ~5ms per client)
    """
    global _ssl_context
    if _ssl_context is None:
        _ssl_context = ssl.create_default_context()
        # Optimize SSL handshake performance
        _ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        # Use optimized cipher ordering (fast ciphers first)
        # These are well-supported and performant
        # Fallback to default ciphers if custom ones aren't supported on this platform.
        with contextlib.suppress(ssl.SSLError):
            _ssl_context.set_ciphers(
                "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20"
                ":ECDH+AESGCM:DH+AESGCM:ECDH+AES:DH+AES:RSA+AESGCM:RSA+AES:!aNULL"
                ":!eNULL:!MD5:!DSS"
            )
    return _ssl_context


@dataclass(slots=True)
class HTTPResponse:
    """Response from an HTTP request."""

    status_code: int
    headers: dict[str, str]
    body: bytes
    request_id: str | None = None

    def json(self) -> Any:
        """Parse response body as JSON using orjson (fast)."""
        return orjson.loads(self.body)

    @property
    def text(self) -> str:
        """Return response body as text."""
        return self.body.decode("utf-8")


class HTTPClient:
    """
    Synchronous HTTP client with connection pooling.

    Uses httpx for robust HTTP handling with connection reuse.
    """

    __slots__ = ("_client", "_max_retries", "_timeout")

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        http2: bool = True,
    ) -> None:
        # Configure timeouts
        timeouts = httpx.Timeout(
            timeout=timeout,
            connect=connect_timeout,
        )

        # Configure connection limits (aggressive settings for LLM APIs)
        limits = httpx.Limits(
            max_connections=300,  # Match litellm's aggressive limit
            max_keepalive_connections=50,  # More keepalive connections
            keepalive_expiry=120.0,  # 2 min keepalive (match litellm)
        )

        # Check for proxy
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

        # Use cached SSL context with optimized settings
        ssl_context = _get_ssl_context()

        self._client = httpx.Client(
            timeout=timeouts,
            limits=limits,
            http2=http2,
            proxy=proxy,
            follow_redirects=True,
            verify=ssl_context,  # Use optimized SSL context
        )
        self._timeout = timeout
        self._max_retries = max_retries

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> HTTPResponse | Iterator[bytes]:
        """
        Make an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL to request
            headers: Request headers
            body: Request body as bytes
            timeout: Request timeout (overrides default)
            stream: If True, return iterator for streaming response

        Returns:
            HTTPResponse for non-streaming, Iterator[bytes] for streaming
        """
        request_timeout = timeout or self._timeout
        last_error: Exception | None = None

        for _attempt in range(self._max_retries):
            try:
                if stream:
                    return self._stream_request(method, url, headers, body, request_timeout)

                response = self._client.request(
                    method,
                    url,
                    headers=headers,
                    content=body,
                    timeout=request_timeout,
                )

                # Extract request ID
                request_id = response.headers.get("x-request-id") or response.headers.get(
                    "request-id"
                )

                return HTTPResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response.content,
                    request_id=request_id,
                )

            except httpx.TimeoutException as e:
                last_error = TimeoutError(
                    f"Request timed out after {request_timeout}s: {e}",
                    timeout_type="read",
                    timeout_seconds=request_timeout,
                )
            except httpx.ConnectError as e:
                last_error = ConnectionError(f"Connection failed: {e}")
            except httpx.HTTPError as e:
                last_error = ProviderAPIError(f"HTTP error: {e}")
            except Exception as e:
                last_error = ProviderAPIError(f"Request failed: {e}")

        raise last_error or ConnectionError("Request failed after retries")

    def _stream_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        body: bytes | None,
        timeout: float,
    ) -> Iterator[bytes]:
        """Stream response body."""
        with self._client.stream(
            method,
            url,
            headers=headers,
            content=body,
            timeout=timeout,
        ) as response:
            yield from response.iter_bytes()

    def post(
        self,
        url: str,
        *,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> HTTPResponse | Iterator[bytes]:
        """Make a POST request with JSON body."""
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        body = None
        if json_data is not None:
            body = orjson.dumps(json_data)

        return self.request(
            "POST",
            url,
            headers=request_headers,
            body=body,
            timeout=timeout,
            stream=stream,
        )

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HTTPResponse:
        """Make a GET request."""
        result = self.request("GET", url, headers=headers, timeout=timeout, stream=False)
        assert isinstance(result, HTTPResponse)
        return result

    def close(self) -> None:
        """Close the client and all connections."""
        self._client.close()

    def __enter__(self) -> HTTPClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
