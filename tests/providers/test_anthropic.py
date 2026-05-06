"""
Tests for Anthropic provider adapter.
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
from arcllm.providers.anthropic_adapter import AnthropicAdapter
from arcllm.providers.base import ProviderConfig


class TestAnthropicAdapter:
    """Tests for AnthropicAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test config."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            config = ProviderConfig()
            return AnthropicAdapter(config)

    @pytest.fixture
    def adapter_with_key(self):
        """Create adapter with explicit key."""
        config = ProviderConfig(api_key="sk-ant-test-key-123")
        return AnthropicAdapter(config)


class TestBuildRequest:
    """Tests for building requests."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-ant-test-key")
        return AnthropicAdapter(config)

    def test_build_simple_request(self, adapter):
        """Test building a simple chat request."""
        messages = [{"role": "user", "content": "Hello"}]
        request = adapter.build_request(
            model="claude-3-5-sonnet-20241022", messages=messages, stream=False
        )

        assert request.method == "POST"
        assert "/v1/messages" in request.url
        assert request.headers["x-api-key"] == "sk-ant-test-key"
        assert request.headers["Content-Type"] == "application/json"
        assert "anthropic-version" in request.headers

        body = json.loads(request.body.decode("utf-8"))
        assert body["model"] == "claude-3-5-sonnet-20241022"
        assert body["messages"] == messages
        assert "max_tokens" in body  # Anthropic requires max_tokens

    def test_build_request_with_system(self, adapter):
        """Test building request with system message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        request = adapter.build_request(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
        )

        body = json.loads(request.body.decode("utf-8"))
        # Anthropic extracts system message to a separate field
        assert body["system"] == "You are helpful."
        # Only user message should be in messages
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    def test_build_request_with_temperature(self, adapter):
        """Test building request with temperature."""
        request = adapter.build_request(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["temperature"] == 0.7

    def test_build_request_with_max_tokens(self, adapter):
        """Test building request with max_tokens."""
        request = adapter.build_request(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["max_tokens"] == 100

    def test_build_request_with_tools(self, adapter, tool_definition):
        """Test building request with tools."""
        request = adapter.build_request(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=tool_definition,
        )

        body = json.loads(request.body.decode("utf-8"))
        assert "tools" in body
        assert len(body["tools"]) == 1
        # Anthropic format differs
        assert body["tools"][0]["name"] == "get_weather"
        assert "input_schema" in body["tools"][0]

    def test_build_streaming_request(self, adapter):
        """Test building streaming request."""
        request = adapter.build_request(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["stream"] is True

    def test_build_request_with_stop_sequences(self, adapter):
        """Test building request with stop sequences."""
        request = adapter.build_request(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hi"}],
            stop=["END", "STOP"],
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["stop_sequences"] == ["END", "STOP"]

    def test_pdf_input_file_converts_to_document_block(self, adapter):
        """OpenAI-shape ``input_file`` content -> Anthropic ``document`` block (base64)."""
        request = adapter.build_request(
            model="claude-sonnet-4-5-20250929",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file": {
                                "data": "JVBERi0xLjQKJeLjz9MK",  # fake base64 PDF header
                                "media_type": "application/pdf",
                            },
                        },
                        {"type": "text", "text": "Summarise"},
                    ],
                }
            ],
            max_tokens=64,
        )

        body = json.loads(request.body.decode("utf-8"))
        msg = body["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        # First block is the document
        doc = msg["content"][0]
        assert doc["type"] == "document"
        assert doc["source"]["type"] == "base64"
        assert doc["source"]["media_type"] == "application/pdf"
        assert doc["source"]["data"] == "JVBERi0xLjQKJeLjz9MK"
        # Second block is the text
        assert msg["content"][1] == {"type": "text", "text": "Summarise"}

    def test_pdf_data_url_converts_to_document_block(self, adapter):
        """data:application/pdf URL also converts to base64 document block."""
        request = adapter.build_request(
            model="claude-sonnet-4-5-20250929",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file": {"data": "data:application/pdf;base64,JVBERg=="},
                        },
                    ],
                }
            ],
            max_tokens=64,
        )

        body = json.loads(request.body.decode("utf-8"))
        doc = body["messages"][0]["content"][0]
        assert doc["type"] == "document"
        assert doc["source"]["type"] == "base64"
        assert doc["source"]["media_type"] == "application/pdf"
        assert doc["source"]["data"] == "JVBERg=="

    def test_anthropic_native_blocks_pass_through(self, adapter):
        """Callers can already supply Anthropic-native blocks; we don't double-wrap."""
        native = {
            "type": "document",
            "source": {"type": "url", "url": "https://example.com/doc.pdf"},
        }
        request = adapter.build_request(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": [native, {"type": "text", "text": "Read"}]}],
            max_tokens=64,
        )
        body = json.loads(request.body.decode("utf-8"))
        assert body["messages"][0]["content"][0] == native

    def test_build_request_with_thinking_budget(self, adapter):
        """thinking_budget should emit Anthropic's thinking block.

        Per Anthropic docs, when thinking is enabled the API rejects
        ``temperature``/``top_p`` — the adapter strips them defensively.
        """
        request = adapter.build_request(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "Solve this hard problem"}],
            max_tokens=4096,
            thinking_budget=2048,
            temperature=0.7,
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "temperature" not in body
        assert "top_p" not in body


class TestMessageConversion:
    """Tests for message conversion."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-ant-test-key")
        return AnthropicAdapter(config)

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
        assert converted[1]["role"] == "assistant"
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

    def test_convert_multiple_system_messages(self, adapter):
        """Test multiple system messages are concatenated."""
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
        ]

        system, _converted = adapter._convert_messages(messages)

        assert "Be helpful" in system
        assert "Be concise" in system

    def test_convert_assistant_with_tool_calls(self, adapter):
        """Test converting assistant message with tool calls."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                    }
                ],
            },
        ]

        _, converted = adapter._convert_messages(messages)

        assert len(converted) == 2
        # Assistant message should have content blocks
        assert isinstance(converted[1]["content"], list)
        assert converted[1]["content"][0]["type"] == "text"
        assert converted[1]["content"][1]["type"] == "tool_use"

    def test_convert_tool_result(self, adapter):
        """Test converting tool result message."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72F"},
        ]

        _, converted = adapter._convert_messages(messages)

        # Tool result should be in a user message (Anthropic requirement)
        assert converted[2]["role"] == "user"
        assert converted[2]["content"][0]["type"] == "tool_result"


class TestParseResponse:
    """Tests for parsing responses."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-ant-test-key")
        return AnthropicAdapter(config)

    @pytest.fixture
    def anthropic_response(self):
        """Standard Anthropic response."""
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }

    def test_parse_simple_response(self, adapter, anthropic_response):
        """Test parsing a simple completion response."""
        data = json.dumps(anthropic_response).encode("utf-8")
        response = adapter.parse_response(data, "claude-3-5-sonnet-20241022")

        assert response.id == "msg_123"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 8

    def test_parse_tool_use_response(self, adapter):
        """Test parsing response with tool use."""
        resp = {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"},
                },
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }

        data = json.dumps(resp).encode("utf-8")
        response = adapter.parse_response(data, "claude-3-5-sonnet-20241022")

        assert response.choices[0].message.content == "Let me check the weather."
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

        tc = response.choices[0].message.tool_calls[0]
        assert tc.id == "toolu_123"
        assert tc.function.name == "get_weather"
        assert "San Francisco" in tc.function.arguments
        assert response.choices[0].finish_reason == "tool_calls"

    def test_parse_response_invalid_json(self, adapter):
        """Test parsing invalid JSON raises error."""
        with pytest.raises(ResponseParseError):
            adapter.parse_response(b"not json", "claude-3-5-sonnet-20241022")

    def test_model_extra_contains_usage(self, adapter, anthropic_response):
        """Test model_extra contains usage."""
        data = json.dumps(anthropic_response).encode("utf-8")
        response = adapter.parse_response(data, "claude-3-5-sonnet-20241022")

        assert "usage" in response.model_extra
        assert response.model_extra["usage"]["total_tokens"] == 18


class TestParseStreamEvent:
    """Tests for parsing streaming events."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-ant-test-key")
        return AnthropicAdapter(config)

    def test_parse_message_start(self, adapter):
        """Test parsing message_start event."""
        event_data = json.dumps(
            {
                "type": "message_start",
                "message": {"id": "msg_123", "model": "claude-3-5-sonnet-20241022"},
            }
        )
        chunk = adapter.parse_stream_event(event_data, "claude-3-5-sonnet-20241022")

        assert chunk is not None
        assert chunk.id == "msg_123"
        assert chunk.choices[0].delta.role == "assistant"

    def test_parse_content_block_delta(self, adapter):
        """Test parsing content_block_delta event."""
        event_data = json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            }
        )
        chunk = adapter.parse_stream_event(event_data, "claude-3-5-sonnet-20241022")

        assert chunk is not None
        assert chunk.choices[0].delta.content == "Hello"

    def test_parse_message_delta(self, adapter):
        """Test parsing message_delta event (with stop reason)."""
        event_data = json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 10},
            }
        )
        chunk = adapter.parse_stream_event(event_data, "claude-3-5-sonnet-20241022")

        assert chunk is not None
        assert chunk.choices[0].finish_reason == "stop"

    def test_parse_empty_event(self, adapter):
        """Test empty event returns None."""
        chunk = adapter.parse_stream_event("", "claude-3-5-sonnet-20241022")
        assert chunk is None

    def test_parse_invalid_json_returns_none(self, adapter):
        """Test invalid JSON event returns None (doesn't raise)."""
        chunk = adapter.parse_stream_event("not json", "claude-3-5-sonnet-20241022")
        assert chunk is None


class TestParseError:
    """Tests for parsing errors."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-ant-test-key")
        return AnthropicAdapter(config)

    def test_parse_401_error(self, adapter):
        """Test parsing 401 authentication error."""
        error_data = json.dumps(
            {"error": {"type": "authentication_error", "message": "Invalid API Key"}}
        ).encode("utf-8")

        error = adapter.parse_error(401, error_data, "req-123")
        assert isinstance(error, AuthenticationError)
        assert "Invalid API Key" in str(error)

    def test_parse_429_error(self, adapter):
        """Test parsing 429 rate limit error."""
        error_data = json.dumps(
            {"error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}}
        ).encode("utf-8")

        error = adapter.parse_error(429, error_data, "req-123")
        assert isinstance(error, RateLimitError)

    def test_parse_non_json_error(self, adapter):
        """Test parsing non-JSON error response."""
        error = adapter.parse_error(500, b"Internal Server Error", "req-123")
        assert "Internal Server Error" in str(error)


class TestToolConversion:
    """Tests for tool format conversion."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-ant-test-key")
        return AnthropicAdapter(config)

    def test_convert_tools(self, adapter, tool_definition):
        """Test converting OpenAI tool format to Anthropic format."""
        anthropic_tools = adapter._convert_tools(tool_definition)

        assert len(anthropic_tools) == 1
        tool = anthropic_tools[0]
        assert tool["name"] == "get_weather"
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
