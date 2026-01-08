"""
Comprehensive tests for arcllm.types module.

Tests all type classes, model_dump methods, and edge cases.
"""

from __future__ import annotations

from arcllm.types import (
    Choice,
    ChunkChoice,
    ChunkDelta,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    Message,
    ModelResponse,
    StreamChunk,
    StreamingResponse,
    ToolCall,
    Usage,
)


class TestFunctionCallComprehensive:
    """Comprehensive tests for FunctionCall."""

    def test_parse_valid_json_arguments(self):
        """Test parsing valid JSON arguments."""
        fc = FunctionCall(
            name="get_weather",
            arguments='{"location": "NYC", "unit": "celsius"}',
        )
        args = fc.parse_arguments()
        assert args["location"] == "NYC"
        assert args["unit"] == "celsius"

    def test_parse_empty_json(self):
        """Test parsing empty JSON object."""
        fc = FunctionCall(name="test", arguments="{}")
        args = fc.parse_arguments()
        assert args == {}

    def test_parse_nested_json(self):
        """Test parsing nested JSON arguments."""
        fc = FunctionCall(
            name="complex",
            arguments='{"nested": {"key": "value"}, "array": [1, 2, 3]}',
        )
        args = fc.parse_arguments()
        assert args["nested"]["key"] == "value"
        assert args["array"] == [1, 2, 3]

    def test_model_dump_complete(self):
        """Test complete model_dump output."""
        fc = FunctionCall(name="test_func", arguments='{"arg": "value"}')
        dumped = fc.model_dump()
        assert dumped == {"name": "test_func", "arguments": '{"arg": "value"}'}


class TestToolCallComprehensive:
    """Comprehensive tests for ToolCall."""

    def test_tool_call_defaults(self):
        """Test ToolCall default values."""
        tc = ToolCall(id="call_123")
        assert tc.type == "function"
        assert tc.function is None

    def test_tool_call_with_function(self):
        """Test ToolCall with function."""
        fc = FunctionCall(name="get_weather", arguments='{"location": "NYC"}')
        tc = ToolCall(id="call_123", function=fc)
        assert tc.function.name == "get_weather"

    def test_model_dump_with_function(self):
        """Test model_dump includes function."""
        fc = FunctionCall(name="test", arguments="{}")
        tc = ToolCall(id="call_123", type="function", function=fc)
        dumped = tc.model_dump()
        assert dumped["id"] == "call_123"
        assert dumped["type"] == "function"
        assert dumped["function"]["name"] == "test"

    def test_model_dump_without_function(self):
        """Test model_dump without function."""
        tc = ToolCall(id="call_123")
        dumped = tc.model_dump()
        assert "function" not in dumped


class TestMessageComprehensive:
    """Comprehensive tests for Message."""

    def test_message_with_all_fields(self):
        """Test message with all optional fields."""
        fc = FunctionCall(name="test", arguments="{}")
        tc = ToolCall(id="call_123", function=fc)
        msg = Message(
            role="assistant",
            content="Hello",
            tool_calls=[tc],
            function_call=fc,
            refusal="I cannot help with that",
        )
        assert msg.role == "assistant"
        assert msg.content == "Hello"
        assert msg.tool_calls[0].id == "call_123"
        assert msg.function_call.name == "test"
        assert msg.refusal == "I cannot help with that"

    def test_message_model_dump_with_all_fields(self):
        """Test model_dump with all fields populated."""
        fc = FunctionCall(name="test", arguments="{}")
        tc = ToolCall(id="call_123", function=fc)
        msg = Message(
            role="assistant",
            content="Response",
            tool_calls=[tc],
            function_call=fc,
            refusal="Cannot comply",
        )
        dumped = msg.model_dump()
        assert dumped["role"] == "assistant"
        assert dumped["content"] == "Response"
        assert len(dumped["tool_calls"]) == 1
        assert "function_call" in dumped
        assert dumped["refusal"] == "Cannot comply"

    def test_message_model_dump_minimal(self):
        """Test model_dump with minimal fields."""
        msg = Message(role="user")
        dumped = msg.model_dump()
        assert dumped == {"role": "user"}

    def test_message_none_content(self):
        """Test message with None content."""
        msg = Message(role="assistant", content=None)
        dumped = msg.model_dump()
        assert "content" not in dumped


class TestUsageComprehensive:
    """Comprehensive tests for Usage."""

    def test_usage_with_details(self):
        """Test Usage with detail fields."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details={"cached_tokens": 50},
            completion_tokens_details={"reasoning_tokens": 10},
        )
        assert usage.prompt_tokens_details["cached_tokens"] == 50
        assert usage.completion_tokens_details["reasoning_tokens"] == 10

    def test_usage_model_dump_with_details(self):
        """Test model_dump includes details."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details={"cached_tokens": 50},
            completion_tokens_details={"reasoning_tokens": 10},
        )
        dumped = usage.model_dump()
        assert "prompt_tokens_details" in dumped
        assert "completion_tokens_details" in dumped

    def test_usage_model_dump_no_details(self):
        """Test model_dump without details."""
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        dumped = usage.model_dump()
        assert "prompt_tokens_details" not in dumped
        assert "completion_tokens_details" not in dumped


class TestChoiceComprehensive:
    """Comprehensive tests for Choice."""

    def test_choice_with_logprobs(self):
        """Test Choice with logprobs."""
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="Hi"),
            finish_reason="stop",
            logprobs={"tokens": ["Hi"], "token_logprobs": [-0.5]},
        )
        assert choice.logprobs is not None
        assert choice.logprobs["tokens"] == ["Hi"]

    def test_choice_model_dump_with_logprobs(self):
        """Test model_dump includes logprobs."""
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="Hi"),
            logprobs={"tokens": ["Hi"]},
        )
        dumped = choice.model_dump()
        assert "logprobs" in dumped


class TestModelResponseComprehensive:
    """Comprehensive tests for ModelResponse."""

    def test_response_post_init_adds_usage(self):
        """Test model_extra contains usage when explicitly passed (like adapters do)."""
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        # Adapters explicitly pass model_extra with usage
        response = ModelResponse(
            id="resp-1",
            choices=[],
            usage=usage,
            model_extra={"usage": usage.model_dump()},
        )
        assert "usage" in response.model_extra
        assert response.model_extra["usage"]["total_tokens"] == 15

    def test_response_model_dump(self):
        """Test complete model_dump output."""
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="Hi"),
            finish_reason="stop",
        )
        response = ModelResponse(
            id="resp-1",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[choice],
            usage=usage,
            system_fingerprint="fp_abc123",
        )
        dumped = response.model_dump()
        assert dumped["id"] == "resp-1"
        assert dumped["object"] == "chat.completion"
        assert dumped["created"] == 1234567890
        assert dumped["model"] == "gpt-4o-mini"
        assert len(dumped["choices"]) == 1
        assert dumped["usage"]["total_tokens"] == 15
        assert dumped["system_fingerprint"] == "fp_abc123"


class TestChunkDeltaComprehensive:
    """Comprehensive tests for ChunkDelta."""

    def test_chunk_delta_all_fields(self):
        """Test ChunkDelta with all fields."""
        delta = ChunkDelta(
            role="assistant",
            content="Hello",
            tool_calls=[{"index": 0, "function": {"name": "test"}}],
            function_call={"name": "legacy_func"},
        )
        assert delta.role == "assistant"
        assert delta.content == "Hello"
        assert delta.tool_calls is not None
        assert delta.function_call is not None

    def test_chunk_delta_model_dump_empty(self):
        """Test model_dump with no fields set."""
        delta = ChunkDelta()
        dumped = delta.model_dump()
        assert dumped == {}

    def test_chunk_delta_model_dump_partial(self):
        """Test model_dump with some fields set."""
        delta = ChunkDelta(content="Partial")
        dumped = delta.model_dump()
        assert dumped == {"content": "Partial"}


class TestChunkChoiceComprehensive:
    """Comprehensive tests for ChunkChoice."""

    def test_chunk_choice_with_logprobs(self):
        """Test ChunkChoice with logprobs."""
        chunk_choice = ChunkChoice(
            index=0,
            delta=ChunkDelta(content="Hello"),
            finish_reason=None,
            logprobs={"tokens": ["Hello"]},
        )
        assert chunk_choice.logprobs is not None

    def test_chunk_choice_model_dump(self):
        """Test model_dump output."""
        chunk_choice = ChunkChoice(
            index=0,
            delta=ChunkDelta(content="Hi"),
            finish_reason="stop",
            logprobs={"tokens": ["Hi"]},
        )
        dumped = chunk_choice.model_dump()
        assert dumped["index"] == 0
        assert dumped["delta"]["content"] == "Hi"
        assert dumped["finish_reason"] == "stop"
        assert "logprobs" in dumped


class TestStreamChunkComprehensive:
    """Comprehensive tests for StreamChunk."""

    def test_stream_chunk_model_dump(self):
        """Test complete model_dump output."""
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk = StreamChunk(
            id="chunk-1",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChunkDelta(content="Hi"),
                )
            ],
            usage=usage,
            system_fingerprint="fp_abc123",
        )
        dumped = chunk.model_dump()
        assert dumped["id"] == "chunk-1"
        assert dumped["object"] == "chat.completion.chunk"
        assert dumped["usage"]["total_tokens"] == 15
        assert dumped["system_fingerprint"] == "fp_abc123"


class TestStreamingResponseComprehensive:
    """Comprehensive tests for StreamingResponse."""

    def test_streaming_response_iteration(self):
        """Test iterating over streaming response."""

        def chunk_iter():
            yield StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                choices=[ChunkChoice(index=0, delta=ChunkDelta(content="Hello"))],
            )
            yield StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                choices=[ChunkChoice(index=0, delta=ChunkDelta(content="!"))],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        response = StreamingResponse(iterator=chunk_iter(), model="gpt-4o-mini")

        chunks = list(response)
        assert len(chunks) == 2
        assert response.usage is not None
        assert response.usage.total_tokens == 15

    def test_streaming_response_properties(self):
        """Test StreamingResponse properties."""

        def chunk_iter():
            yield StreamChunk(id="chunk-1", model="gpt-4o-mini", choices=[])

        response = StreamingResponse(
            iterator=chunk_iter(),
            response_id="resp-123",
            model="gpt-4o-mini",
        )

        assert response.response_id == "resp-123"
        assert response.model == "gpt-4o-mini"


class TestEmbeddingTypesComprehensive:
    """Comprehensive tests for embedding types."""

    def test_embedding_usage_model_dump(self):
        """Test EmbeddingUsage model_dump."""
        usage = EmbeddingUsage(prompt_tokens=100, total_tokens=100)
        dumped = usage.model_dump()
        assert dumped == {"prompt_tokens": 100, "total_tokens": 100}

    def test_embedding_data_model_dump(self):
        """Test EmbeddingData model_dump."""
        data = EmbeddingData(
            index=0,
            embedding=[0.1, 0.2, 0.3],
            object="embedding",
        )
        dumped = data.model_dump()
        assert dumped["index"] == 0
        assert dumped["embedding"] == [0.1, 0.2, 0.3]
        assert dumped["object"] == "embedding"

    def test_embedding_response_model_dump(self):
        """Test EmbeddingResponse model_dump."""
        response = EmbeddingResponse(
            model="text-embedding-3-small",
            data=[
                EmbeddingData(index=0, embedding=[0.1, 0.2]),
                EmbeddingData(index=1, embedding=[0.3, 0.4]),
            ],
            usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10),
            object="list",
        )
        dumped = response.model_dump()
        assert dumped["model"] == "text-embedding-3-small"
        assert len(dumped["data"]) == 2
        assert dumped["usage"]["prompt_tokens"] == 10
        assert dumped["object"] == "list"
