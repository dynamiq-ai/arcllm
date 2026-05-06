"""
Tests for Gemini provider adapter.
"""

import json
import os
from unittest.mock import patch

import pytest

from arcllm.exceptions import (
    AuthenticationError,
    RateLimitError,
    ResponseParseError,
)
from arcllm.providers.base import ProviderConfig
from arcllm.providers.gemini_adapter import GeminiAdapter


class TestGeminiAdapter:
    """Tests for GeminiAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test config."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            config = ProviderConfig()
            return GeminiAdapter(config)

    @pytest.fixture
    def adapter_with_key(self):
        """Create adapter with explicit key."""
        config = ProviderConfig(api_key="test-gemini-key-123")
        return GeminiAdapter(config)


class TestBuildRequest:
    """Tests for building requests."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="test-gemini-key")
        return GeminiAdapter(config)

    def test_build_simple_request(self, adapter):
        """Test building a simple chat request."""
        messages = [{"role": "user", "content": "Hello"}]
        request = adapter.build_request(model="gemini-1.5-pro", messages=messages, stream=False)

        assert request.method == "POST"
        assert "generativelanguage.googleapis.com" in request.url
        assert "generateContent" in request.url
        assert "key=" in request.url
        assert request.headers["Content-Type"] == "application/json"

        body = json.loads(request.body.decode("utf-8"))
        assert "contents" in body
        assert len(body["contents"]) == 1
        assert body["contents"][0]["role"] == "user"

    def test_build_request_with_system(self, adapter):
        """Test building request with system message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        request = adapter.build_request(
            model="gemini-1.5-pro",
            messages=messages,
        )

        body = json.loads(request.body.decode("utf-8"))
        # Gemini uses systemInstruction for system messages
        assert "systemInstruction" in body
        assert "You are helpful" in str(body["systemInstruction"])
        # Only user message should be in contents
        assert len(body["contents"]) == 1
        assert body["contents"][0]["role"] == "user"

    def test_build_request_with_temperature(self, adapter):
        """Test building request with temperature."""
        request = adapter.build_request(
            model="gemini-1.5-pro", messages=[{"role": "user", "content": "Hi"}], temperature=0.7
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["generationConfig"]["temperature"] == 0.7

    def test_build_request_with_max_tokens(self, adapter):
        """Test building request with max_tokens."""
        request = adapter.build_request(
            model="gemini-1.5-pro", messages=[{"role": "user", "content": "Hi"}], max_tokens=100
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["generationConfig"]["maxOutputTokens"] == 100

    def test_build_request_with_tools(self, adapter, tool_definition):
        """Test building request with tools."""
        request = adapter.build_request(
            model="gemini-1.5-pro",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=tool_definition,
        )

        body = json.loads(request.body.decode("utf-8"))
        assert "tools" in body
        assert len(body["tools"]) == 1
        # Gemini wraps tools in functionDeclarations
        assert "functionDeclarations" in body["tools"][0]

    def test_build_streaming_request(self, adapter):
        """Test building streaming request."""
        request = adapter.build_request(
            model="gemini-1.5-pro", messages=[{"role": "user", "content": "Hi"}], stream=True
        )

        # Gemini uses different endpoint for streaming
        assert "streamGenerateContent" in request.url

    def test_build_request_with_thinking_config(self, adapter):
        """``thinking_budget`` and ``include_thoughts`` map to Gemini's thinkingConfig.

        Per https://ai.google.dev/gemini-api/docs/thinking, both fields live
        under ``generationConfig.thinkingConfig``.
        """
        request = adapter.build_request(
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "Hi"}],
            thinking_budget=1024,
            include_thoughts=True,
        )

        body = json.loads(request.body.decode("utf-8"))
        gen_cfg = body["generationConfig"]
        assert gen_cfg["thinkingConfig"] == {
            "thinkingBudget": 1024,
            "includeThoughts": True,
        }


class TestMessageConversion:
    """Tests for message conversion."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="test-gemini-key")
        return GeminiAdapter(config)

    def test_convert_simple_messages(self, adapter):
        """Test converting simple messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        system, converted = adapter._convert_messages(messages)

        assert system is None
        assert len(converted) == 3
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "model"  # Gemini uses 'model' instead of 'assistant'
        assert converted[2]["role"] == "user"

    def test_convert_system_message(self, adapter):
        """Test system message is extracted."""
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]

        system, converted = adapter._convert_messages(messages)

        assert system == "Be helpful"
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_convert_content_to_parts(self, adapter):
        """Test content is converted to parts format."""
        messages = [{"role": "user", "content": "Hello"}]

        _, converted = adapter._convert_messages(messages)

        # Content should be in parts array
        assert "parts" in converted[0]
        assert converted[0]["parts"][0]["text"] == "Hello"


class TestParseResponse:
    """Tests for parsing responses."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="test-gemini-key")
        return GeminiAdapter(config)

    @pytest.fixture
    def gemini_response(self, gemini_completion_response):
        """Standard Gemini response from conftest."""
        return gemini_completion_response

    def test_parse_simple_response(self, adapter, gemini_response):
        """Test parsing a simple completion response."""
        data = json.dumps(gemini_response).encode("utf-8")
        response = adapter.parse_response(data, "gemini-1.5-pro")

        assert response.model == "gemini-1.5-pro"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 8
        assert response.usage.completion_tokens == 10

    def test_parse_tool_call_response(self, adapter):
        """Test parsing response with function call."""
        resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "San Francisco"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 30,
                "totalTokenCount": 80,
            },
        }

        data = json.dumps(resp).encode("utf-8")
        response = adapter.parse_response(data, "gemini-1.5-pro")

        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

        tc = response.choices[0].message.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert "San Francisco" in tc.function.arguments

    def test_parse_response_invalid_json(self, adapter):
        """Test parsing invalid JSON raises error."""
        with pytest.raises(ResponseParseError):
            adapter.parse_response(b"not json", "gemini-1.5-pro")

    def test_model_extra_contains_usage(self, adapter, gemini_response):
        """Test model_extra contains usage."""
        data = json.dumps(gemini_response).encode("utf-8")
        response = adapter.parse_response(data, "gemini-1.5-pro")

        assert "usage" in response.model_extra
        assert response.model_extra["usage"]["total_tokens"] == 18


class TestParseStreamEvent:
    """Tests for parsing streaming events."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="test-gemini-key")
        return GeminiAdapter(config)

    def test_parse_content_chunk(self, adapter):
        """Test parsing content chunk."""
        event_data = json.dumps(
            {"candidates": [{"content": {"parts": [{"text": "Hello"}], "role": "model"}}]}
        )
        chunk = adapter.parse_stream_event(event_data, "gemini-1.5-pro")

        assert chunk is not None
        assert chunk.choices[0].delta.content == "Hello"

    def test_parse_finish_chunk(self, adapter):
        """Test parsing finish chunk."""
        event_data = json.dumps(
            {"candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "STOP"}]}
        )
        chunk = adapter.parse_stream_event(event_data, "gemini-1.5-pro")

        assert chunk is not None
        assert chunk.choices[0].finish_reason == "stop"

    def test_parse_empty_event(self, adapter):
        """Test empty event returns None."""
        chunk = adapter.parse_stream_event("", "gemini-1.5-pro")
        assert chunk is None

    def test_parse_invalid_json_returns_none(self, adapter):
        """Test invalid JSON event returns None."""
        chunk = adapter.parse_stream_event("not json", "gemini-1.5-pro")
        assert chunk is None

    def test_parse_stream_with_usage(self, adapter):
        """Test parsing stream chunk with usage."""
        event_data = json.dumps(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hi"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            }
        )
        chunk = adapter.parse_stream_event(event_data, "gemini-1.5-pro")

        assert chunk is not None
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 10
        assert chunk.usage.completion_tokens == 5


class TestParseError:
    """Tests for parsing errors."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="test-gemini-key")
        return GeminiAdapter(config)

    def test_parse_401_error(self, adapter):
        """Test parsing 401 authentication error."""
        error_data = json.dumps(
            {"error": {"message": "Invalid API key", "status": "UNAUTHENTICATED"}}
        ).encode("utf-8")

        error = adapter.parse_error(401, error_data, "req-123")
        assert isinstance(error, AuthenticationError)
        assert "Invalid API key" in str(error)

    def test_parse_429_rate_limit_error(self, adapter):
        """A pure 429 with rate-limit wording maps to RateLimitError."""
        error_data = json.dumps(
            {"error": {"message": "Rate limit reached for project", "status": "RATE_LIMITED"}}
        ).encode("utf-8")

        error = adapter.parse_error(429, error_data, "req-123")
        assert isinstance(error, RateLimitError)

    def test_parse_429_quota_exhausted_maps_to_budget_exceeded(self, adapter):
        """429 with quota wording is distinct from a rate limit — the
        capability filter routes it to BudgetExceededError so callers can
        treat it as billing rather than a transient retry."""
        from arcllm.exceptions import BudgetExceededError

        error_data = json.dumps(
            {"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
        ).encode("utf-8")

        error = adapter.parse_error(429, error_data, "req-123")
        assert isinstance(error, BudgetExceededError)
        assert not isinstance(error, RateLimitError)

    def test_parse_non_json_error(self, adapter):
        """Test parsing non-JSON error response."""
        error = adapter.parse_error(500, b"Internal Server Error", "req-123")
        assert "Internal Server Error" in str(error)


class TestToolConversion:
    """Tests for tool format conversion."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="test-gemini-key")
        return GeminiAdapter(config)

    def test_convert_tools(self, adapter, tool_definition):
        """Test converting OpenAI tool format to Gemini format."""
        gemini_tools = adapter._convert_tools(tool_definition)

        assert len(gemini_tools) == 1
        assert "functionDeclarations" in gemini_tools[0]
        func = gemini_tools[0]["functionDeclarations"][0]
        assert func["name"] == "get_weather"
        assert "description" in func
        assert "parameters" in func
