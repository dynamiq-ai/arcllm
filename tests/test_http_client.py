"""
Tests for arcllm.http.client module.

Tests the synchronous HTTP client using httpx.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from arcllm.http.client import HTTPClient, HTTPResponse


class TestHTTPResponse:
    """Tests for HTTPResponse dataclass."""

    def test_create_response(self):
        """Test creating a basic response."""
        response = HTTPResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=b'{"hello": "world"}',
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_json_method(self):
        """Test JSON parsing with orjson."""
        response = HTTPResponse(
            status_code=200,
            headers={},
            body=b'{"hello": "world"}',
        )
        data = response.json()
        assert data == {"hello": "world"}

    def test_text_property(self):
        """Test text decoding."""
        response = HTTPResponse(
            status_code=200,
            headers={},
            body=b"Hello, World!",
        )
        assert response.text == "Hello, World!"

    def test_request_id(self):
        """Test request ID extraction."""
        response = HTTPResponse(
            status_code=200,
            headers={},
            body=b"",
            request_id="req-123",
        )
        assert response.request_id == "req-123"


class TestHTTPClient:
    """Tests for HTTPClient using httpx."""

    def test_create_client_default(self):
        """Test creating an HTTP client with defaults."""
        client = HTTPClient()
        assert client._timeout == 60.0
        assert client._max_retries == 3
        client.close()

    def test_create_client_custom(self):
        """Test creating an HTTP client with custom settings."""
        client = HTTPClient(
            timeout=30.0,
            connect_timeout=5.0,
            max_retries=5,
            http2=False,
        )
        assert client._timeout == 30.0
        assert client._max_retries == 5
        client.close()

    def test_context_manager(self):
        """Test using client as context manager."""
        with HTTPClient() as client:
            assert client is not None
            assert client._client is not None

    def test_close_client(self):
        """Test closing the client."""
        client = HTTPClient()
        client.close()
        # Client should be closed but not raise error
        assert True

    @patch("httpx.Client.request")
    def test_request_success(self, mock_request):
        """Test successful request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json", "x-request-id": "req-123"}
        mock_response.content = b'{"result": "ok"}'
        mock_request.return_value = mock_response

        with HTTPClient() as client:
            response = client.request("POST", "https://api.example.com/v1/test")

        assert isinstance(response, HTTPResponse)
        assert response.status_code == 200
        assert response.request_id == "req-123"
        assert response.json() == {"result": "ok"}

    @patch("httpx.Client.request")
    def test_request_with_body(self, mock_request):
        """Test request with body."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'{"status": "ok"}'
        mock_request.return_value = mock_response

        with HTTPClient() as client:
            client.request(
                "POST",
                "https://api.example.com/v1/test",
                body=b'{"key": "value"}',
                headers={"Content-Type": "application/json"},
            )

        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args
        assert call_kwargs[1]["content"] == b'{"key": "value"}'

    @patch("httpx.Client.post")
    def test_post_method(self, mock_post):
        """Test POST convenience method."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'{"ok": true}'

        # The post method uses request internally, so we need to patch request
        with (
            patch("httpx.Client.request", return_value=mock_response),
            HTTPClient() as client,
        ):
            response = client.post(
                "https://api.example.com/v1/test",
                json_data={"model": "gpt-4"},
            )

        assert response.status_code == 200

    @patch("httpx.Client.request")
    def test_get_method(self, mock_request):
        """Test GET convenience method."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'{"models": []}'
        mock_request.return_value = mock_response

        with HTTPClient() as client:
            response = client.get("https://api.example.com/v1/models")

        assert response.status_code == 200
        mock_request.assert_called_with(
            "GET",
            "https://api.example.com/v1/models",
            headers=None,
            content=None,
            timeout=60.0,
        )


class TestHTTPClientErrors:
    """Tests for error handling in HTTPClient."""

    def test_timeout_error(self):
        """Test timeout error handling."""
        import httpx

        from arcllm.exceptions import TimeoutError

        with patch("httpx.Client.request") as mock_request:
            mock_request.side_effect = httpx.TimeoutException("Timed out")

            with HTTPClient(max_retries=1) as client, pytest.raises(TimeoutError):
                client.request("GET", "https://api.example.com/slow")

    def test_connection_error(self):
        """Test connection error handling."""
        import httpx

        from arcllm.exceptions import ConnectionError

        with patch("httpx.Client.request") as mock_request:
            mock_request.side_effect = httpx.ConnectError("Connection refused")

            with HTTPClient(max_retries=1) as client, pytest.raises(ConnectionError):
                client.request("GET", "https://api.example.com/down")

    def test_retry_on_error(self):
        """Test that retries work."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'{"ok": true}'

        with patch("httpx.Client.request") as mock_request:
            # First call fails, second succeeds
            mock_request.side_effect = [
                httpx.ConnectError("Connection refused"),
                mock_response,
            ]

            with HTTPClient(max_retries=2) as client:
                response = client.request("GET", "https://api.example.com/flaky")

            assert response.status_code == 200
            assert mock_request.call_count == 2
