"""
Asynchronous HTTP client using aiohttp.

Features:
- True async I/O with connection pooling
- Optimized for high-concurrency scenarios
- TLS support with connection reuse
- Timeout handling
- Streaming response support
- Automatic retries

aiohttp is used for async operations as it provides better performance
for concurrent requests compared to httpx in async contexts.
See: https://webscrapingsite.com/resources/httpx-vs-requests-vs-aiohttp/
"""

from __future__ import annotations

import builtins
import contextlib
import os
import ssl
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp
import orjson

from arcllm.exceptions import (
    ConnectionError,
    ProviderAPIError,
    TimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["AsyncHTTPClient", "AsyncHTTPResponse"]


@dataclass(slots=True)
class AsyncHTTPResponse:
    """Response from an async HTTP request."""

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


class AsyncHTTPClient:
    """
    Asynchronous HTTP client with connection pooling.

    Uses aiohttp for high-performance async HTTP handling with connection reuse.
    aiohttp is specifically designed for async operations and provides:
    - Lower overhead per request
    - Better connection pooling
    - Optimized for high concurrency

    Note: aiohttp objects are created lazily since they require an event loop.

    Performance optimizations:
    - SSL context cached (avoid repeated creation)
    - Default timeout object cached (avoid repeated creation)
    - skip_auto_headers for reduced overhead
    """

    __slots__ = (
        "_connect_timeout",
        "_connector",
        "_default_timeout",
        "_max_retries",
        "_proxy",
        "_session",
        "_ssl_context",
        "_timeout_seconds",
    )

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize async HTTP client.

        Args:
            timeout: Total request timeout in seconds
            connect_timeout: Connection timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        # Store timeout values for lazy initialization
        self._timeout_seconds = timeout
        self._connect_timeout = connect_timeout
        self._max_retries = max_retries

        # Check for proxy (cached at init, not per-request)
        self._proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

        # Cache SSL context with optimized settings (expensive to create, reusable)
        self._ssl_context = ssl.create_default_context()
        # Optimize SSL handshake performance (matching litellm)
        self._ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        # Use optimized cipher ordering (fast ciphers first); fall back to defaults if unsupported.
        with contextlib.suppress(ssl.SSLError):
            self._ssl_context.set_ciphers(
                "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20"
                ":ECDH+AESGCM:DH+AESGCM:ECDH+AES:DH+AES:RSA+AESGCM:RSA+AES:!aNULL"
                ":!eNULL:!MD5:!DSS"
            )

        # Cache default timeout object (avoid recreation per request)
        self._default_timeout = aiohttp.ClientTimeout(
            total=timeout,
            connect=connect_timeout,
            sock_read=timeout,
        )

        # These are created lazily when first used (require event loop)
        self._connector: aiohttp.TCPConnector | None = None
        self._session: aiohttp.ClientSession | None = None

    def _get_timeout(self, timeout: float | None = None) -> aiohttp.ClientTimeout:
        """Get timeout object, using cached default if no custom timeout."""
        if timeout is None:
            return self._default_timeout
        # Create custom timeout only when needed
        return aiohttp.ClientTimeout(
            total=timeout,
            connect=self._connect_timeout,
            sock_read=timeout,
        )

    def _create_connector(self) -> aiohttp.TCPConnector:
        """
        Create a TCP connector with optimized connection pooling.

        Settings match litellm's aggressive configuration for LLM APIs:
        - High connection limits for concurrent requests
        - Long keepalive to reuse connections across requests
        - DNS caching to avoid repeated lookups
        - Cached SSL context (created once at init)
        """
        return aiohttp.TCPConnector(
            limit=300,  # Max total connections (litellm uses 300)
            limit_per_host=50,  # Max connections per host (litellm uses 50)
            keepalive_timeout=120.0,  # 2 min keepalive (litellm uses 120)
            ttl_dns_cache=300,  # DNS cache for 5 min (litellm uses 300)
            enable_cleanup_closed=True,  # Clean up closed connections
            ssl=self._ssl_context,  # Use cached SSL context
        )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure session exists and is open."""
        if self._session is None or self._session.closed:
            # Create connector if needed
            if self._connector is None or self._connector.closed:
                self._connector = self._create_connector()

            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=self._default_timeout,  # Use cached default timeout
                # Skip auto headers for performance
                skip_auto_headers={"User-Agent"},
            )
        return self._session

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> AsyncHTTPResponse | AsyncIterator[bytes]:
        """
        Make an async HTTP request.

        Args:
            method: HTTP method
            url: Full URL
            headers: Request headers
            body: Request body as bytes
            timeout: Request timeout
            stream: If True, return async iterator for streaming

        Returns:
            AsyncHTTPResponse for non-streaming, AsyncIterator[bytes] for streaming
        """
        session = await self._ensure_session()
        last_error: Exception | None = None

        # Get timeout (uses cached default if no custom timeout)
        request_timeout = self._get_timeout(timeout)

        for _attempt in range(self._max_retries):
            try:
                if stream:
                    return self._stream_request(
                        session, method, url, headers, body, request_timeout
                    )

                async with session.request(
                    method,
                    url,
                    headers=headers,
                    data=body,
                    timeout=request_timeout,
                    proxy=self._proxy,
                ) as response:
                    # Read body
                    response_body = await response.read()

                    # Extract request ID
                    request_id = response.headers.get("x-request-id") or response.headers.get(
                        "request-id"
                    )

                    return AsyncHTTPResponse(
                        status_code=response.status,
                        headers=dict(response.headers),
                        body=response_body,
                        request_id=request_id,
                    )

            except builtins.TimeoutError as e:
                last_error = TimeoutError(
                    f"Request timed out: {e}",
                    timeout_type="read",
                    timeout_seconds=timeout or self._timeout_seconds,
                )
            except aiohttp.ClientConnectorError as e:
                last_error = ConnectionError(f"Connection failed: {e}")
            except aiohttp.ClientError as e:
                last_error = ProviderAPIError(f"HTTP error: {e}")
            except Exception as e:
                last_error = ProviderAPIError(f"Request failed: {e}")

        raise last_error or ConnectionError("Request failed after retries")

    async def _stream_request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        body: bytes | None,
        timeout: aiohttp.ClientTimeout,
    ) -> AsyncIterator[bytes]:
        """Stream response body."""
        async with session.request(
            method,
            url,
            headers=headers,
            data=body,
            timeout=timeout,
            proxy=self._proxy,
        ) as response:
            async for chunk in response.content.iter_any():
                yield chunk

    async def post(
        self,
        url: str,
        *,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> AsyncHTTPResponse | AsyncIterator[bytes]:
        """Make an async POST request with JSON body."""
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        body = None
        if json_data is not None:
            body = orjson.dumps(json_data)

        return await self.request(
            "POST",
            url,
            headers=request_headers,
            body=body,
            timeout=timeout,
            stream=stream,
        )

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncHTTPResponse:
        """Make an async GET request."""
        result = await self.request("GET", url, headers=headers, timeout=timeout, stream=False)
        assert isinstance(result, AsyncHTTPResponse)
        return result

    async def close(self) -> None:
        """Close the client and all connections."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        if self._connector is not None and not self._connector.closed:
            await self._connector.close()

    async def __aenter__(self) -> AsyncHTTPClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
