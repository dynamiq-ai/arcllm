"""
Tests for arcllm.http.async_client module.

Tests the asynchronous HTTP client using aiohttp.
"""

from __future__ import annotations

import builtins
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arcllm.http.async_client import AsyncHTTPClient, AsyncHTTPResponse


class TestAsyncHTTPResponse:
    """Tests for AsyncHTTPResponse dataclass."""

    def test_create_response(self):
        """Test creating a basic async response."""
        response = AsyncHTTPResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=b'{"hello": "world"}',
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_json_method(self):
        """Test JSON parsing with orjson."""
        response = AsyncHTTPResponse(
            status_code=200,
            headers={},
            body=b'{"hello": "world"}',
        )
        data = response.json()
        assert data == {"hello": "world"}

    def test_text_property(self):
        """Test text decoding."""
        response = AsyncHTTPResponse(
            status_code=200,
            headers={},
            body=b"Hello, World!",
        )
        assert response.text == "Hello, World!"

    def test_request_id(self):
        """Test request ID extraction."""
        response = AsyncHTTPResponse(
            status_code=200,
            headers={},
            body=b"",
            request_id="req-123",
        )
        assert response.request_id == "req-123"


class TestAsyncHTTPClient:
    """Tests for AsyncHTTPClient using aiohttp."""

    def test_create_client_default(self):
        """Test creating an async HTTP client with defaults."""
        client = AsyncHTTPClient()
        assert client._timeout_seconds == 60.0
        assert client._max_retries == 3
        # Session is created lazily
        assert client._session is None

    def test_create_client_custom(self):
        """Test creating an async HTTP client with custom settings."""
        client = AsyncHTTPClient(
            timeout=30.0,
            connect_timeout=5.0,
            max_retries=5,
        )
        assert client._timeout_seconds == 30.0
        assert client._connect_timeout == 5.0
        assert client._max_retries == 5

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the client."""
        client = AsyncHTTPClient()
        await client.close()
        # Should not raise even if session was never created

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using client as async context manager."""
        async with AsyncHTTPClient() as client:
            assert client is not None
            # Session is created lazily, so it's None until first request


class TestAsyncHTTPClientRequests:
    """Tests for async request handling."""

    @pytest.mark.asyncio
    async def test_request_success(self):
        """Test successful async request."""
        # Create mock response object
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json", "x-request-id": "req-456"}
        mock_response.read = AsyncMock(return_value=b'{"result": "ok"}')

        # Create async context manager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.request", return_value=mock_cm):
            async with AsyncHTTPClient() as client:
                # Force session creation
                await client._ensure_session()
                response = await client.request("POST", "https://api.example.com/v1/test")

            assert isinstance(response, AsyncHTTPResponse)
            assert response.status_code == 200
            assert response.request_id == "req-456"
            assert response.json() == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_request_with_body(self):
        """Test async request with body."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.read = AsyncMock(return_value=b'{"status": "ok"}')

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.request", return_value=mock_cm) as mock_request:
            async with AsyncHTTPClient() as client:
                await client._ensure_session()
                await client.request(
                    "POST",
                    "https://api.example.com/v1/test",
                    body=b'{"key": "value"}',
                    headers={"Content-Type": "application/json"},
                )

            mock_request.assert_called()

    @pytest.mark.asyncio
    async def test_post_method(self):
        """Test async POST convenience method."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.read = AsyncMock(return_value=b'{"ok": true}')

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.request", return_value=mock_cm):
            async with AsyncHTTPClient() as client:
                await client._ensure_session()
                response = await client.post(
                    "https://api.example.com/v1/test",
                    json_data={"model": "gpt-4"},
                )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_method(self):
        """Test async GET convenience method."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.read = AsyncMock(return_value=b'{"models": []}')

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.request", return_value=mock_cm):
            async with AsyncHTTPClient() as client:
                await client._ensure_session()
                response = await client.get("https://api.example.com/v1/models")

            assert response.status_code == 200


class TestAsyncHTTPClientErrors:
    """Tests for error handling in AsyncHTTPClient."""

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Test timeout error handling."""

        from arcllm.exceptions import TimeoutError

        with patch("aiohttp.ClientSession.request") as mock_request:
            mock_request.side_effect = builtins.TimeoutError("Timed out")

            async with AsyncHTTPClient(max_retries=1) as client:
                await client._ensure_session()
                with pytest.raises(TimeoutError):
                    await client.request("GET", "https://api.example.com/slow")

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test connection error handling."""
        import aiohttp

        from arcllm.exceptions import ConnectionError

        with patch("aiohttp.ClientSession.request") as mock_request:
            mock_request.side_effect = aiohttp.ClientConnectorError(
                MagicMock(), OSError("Connection refused")
            )

            async with AsyncHTTPClient(max_retries=1) as client:
                await client._ensure_session()
                with pytest.raises(ConnectionError):
                    await client.request("GET", "https://api.example.com/down")

    @pytest.mark.asyncio
    async def test_retry_on_error(self):
        """Test that retries work for async client."""
        import aiohttp

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.read = AsyncMock(return_value=b'{"ok": true}')

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.request") as mock_request:
            # First call fails, second succeeds
            mock_request.side_effect = [
                aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused")),
                mock_cm,
            ]

            async with AsyncHTTPClient(max_retries=2) as client:
                await client._ensure_session()
                response = await client.request("GET", "https://api.example.com/flaky")

            assert response.status_code == 200
            assert mock_request.call_count == 2
