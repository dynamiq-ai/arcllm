"""
Tests for arcllm.types module.
"""

import pytest

from arcllm.types import (
    Choice,
    ChunkChoice,
    ChunkDelta,
    Citation,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    Message,
    ModelResponse,
    StreamChunk,
    ToolCall,
    Usage,
)


class TestCitation:
    """Tests for Citation dataclass."""

    def test_url_only(self):
        """Perplexity-legacy shape: URL with no title/snippet."""
        c = Citation(url="https://example.com/x")
        assert c.url == "https://example.com/x"
        assert c.title is None
        assert c.snippet is None
        assert c.start_index is None
        assert c.end_index is None
        assert c.model_dump() == {"url": "https://example.com/x"}

    def test_full_citation(self):
        """Anthropic web-search shape: URL + title + snippet + indices."""
        c = Citation(
            url="https://example.com/x",
            title="Example",
            snippet="An example sentence",
            start_index=12,
            end_index=42,
        )
        assert c.model_dump() == {
            "url": "https://example.com/x",
            "title": "Example",
            "snippet": "An example sentence",
            "start_index": 12,
            "end_index": 42,
        }


class TestUsageCacheTokens:
    """Cache token tracking on Usage struct."""

    def test_usage_without_cache_omits_fields(self):
        u = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        dumped = u.model_dump()
        assert "cache_read_input_tokens" not in dumped
        assert "cache_creation_input_tokens" not in dumped

    def test_usage_with_cache_serialises_fields(self):
        u = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=10,
        )
        dumped = u.model_dump()
        assert dumped["cache_read_input_tokens"] == 80
        assert dumped["cache_creation_input_tokens"] == 10


class TestMessageWithCitations:
    """Citations field threading on Message."""

    def test_message_without_citations_is_None(self):
        """Default messages don't carry citations — only grounded responses do."""
        m = Message(role="assistant", content="hi")
        assert m.citations is None
        assert "citations" not in m.model_dump()

    def test_message_with_citations_serialises(self):
        m = Message(
            role="assistant",
            content="The sky is blue.",
            citations=[
                Citation(url="https://example.com/sky", title="Sky color"),
                Citation(url="https://example.com/blue"),
            ],
        )
        dumped = m.model_dump()
        assert "citations" in dumped
        assert len(dumped["citations"]) == 2
        assert dumped["citations"][0]["title"] == "Sky color"


class TestFunctionCall:
    """Tests for FunctionCall dataclass."""

    def test_create_function_call(self):
        """Test basic FunctionCall creation."""
        fc = FunctionCall(name="get_weather", arguments='{"location": "SF"}')
        assert fc.name == "get_weather"
        assert fc.arguments == '{"location": "SF"}'

    def test_parse_arguments(self):
        """Test parsing arguments JSON."""
        fc = FunctionCall(name="get_weather", arguments='{"location": "SF", "unit": "celsius"}')
        parsed = fc.parse_arguments()
        assert parsed == {"location": "SF", "unit": "celsius"}

    def test_parse_arguments_invalid_json(self):
        """Test parsing invalid JSON raises ValueError."""
        fc = FunctionCall(name="test", arguments="not valid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            fc.parse_arguments()

    def test_model_dump(self):
        """Test model_dump serialization."""
        fc = FunctionCall(name="get_weather", arguments='{"location": "SF"}')
        dumped = fc.model_dump()
        assert dumped == {"name": "get_weather", "arguments": '{"location": "SF"}'}


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_create_tool_call(self):
        """Test basic ToolCall creation."""
        tc = ToolCall(
            id="call_123", type="function", function=FunctionCall(name="test", arguments="{}")
        )
        assert tc.id == "call_123"
        assert tc.type == "function"
        assert tc.function is not None

    def test_model_dump(self):
        """Test model_dump serialization."""
        tc = ToolCall(id="call_123", function=FunctionCall(name="test", arguments='{"x": 1}'))
        dumped = tc.model_dump()
        assert dumped["id"] == "call_123"
        assert dumped["type"] == "function"
        assert dumped["function"]["name"] == "test"


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_simple_message(self):
        """Test creating a simple message."""
        msg = Message(role="assistant", content="Hello!")
        assert msg.role == "assistant"
        assert msg.content == "Hello!"
        assert msg.tool_calls is None

    def test_create_message_with_tool_calls(self):
        """Test creating a message with tool calls."""
        tc = ToolCall(id="call_1", function=FunctionCall(name="test", arguments="{}"))
        msg = Message(role="assistant", content=None, tool_calls=[tc])
        assert msg.content is None
        assert len(msg.tool_calls) == 1

    def test_model_dump(self):
        """Test model_dump serialization."""
        msg = Message(role="assistant", content="Hi")
        dumped = msg.model_dump()
        assert dumped == {"role": "assistant", "content": "Hi"}

    def test_model_dump_with_tool_calls(self):
        """Test model_dump with tool calls."""
        tc = ToolCall(id="call_1", function=FunctionCall(name="test", arguments="{}"))
        msg = Message(role="assistant", tool_calls=[tc])
        dumped = msg.model_dump()
        assert "tool_calls" in dumped
        assert len(dumped["tool_calls"]) == 1


class TestUsage:
    """Tests for Usage dataclass."""

    def test_create_usage(self):
        """Test basic Usage creation."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_default_values(self):
        """Test default values are 0."""
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_model_dump(self):
        """Test model_dump serialization."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        dumped = usage.model_dump()
        assert dumped == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


class TestChoice:
    """Tests for Choice dataclass."""

    def test_create_choice(self):
        """Test basic Choice creation."""
        msg = Message(role="assistant", content="Hello")
        choice = Choice(index=0, message=msg, finish_reason="stop")
        assert choice.index == 0
        assert choice.message.content == "Hello"
        assert choice.finish_reason == "stop"

    def test_model_dump(self):
        """Test model_dump serialization."""
        msg = Message(role="assistant", content="Hi")
        choice = Choice(index=0, message=msg, finish_reason="stop")
        dumped = choice.model_dump()
        assert dumped["index"] == 0
        assert dumped["message"]["content"] == "Hi"
        assert dumped["finish_reason"] == "stop"


class TestModelResponse:
    """Tests for ModelResponse dataclass."""

    def test_create_model_response(self):
        """Test basic ModelResponse creation."""
        msg = Message(role="assistant", content="Hello")
        choice = Choice(index=0, message=msg, finish_reason="stop")
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        response = ModelResponse(id="resp-123", model="gpt-4o-mini", choices=[choice], usage=usage)

        assert response.id == "resp-123"
        assert response.model == "gpt-4o-mini"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello"
        assert response.usage.total_tokens == 15

    def test_model_extra_contains_usage(self):
        """Test that model_extra is populated with usage."""
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        # Adapters explicitly pass model_extra with usage
        response = ModelResponse(
            id="resp-123",
            choices=[],
            usage=usage,
            model_extra={"usage": usage.model_dump()},
        )
        assert "usage" in response.model_extra
        assert response.model_extra["usage"]["total_tokens"] == 15

    def test_model_dump(self):
        """Test model_dump serialization."""
        msg = Message(role="assistant", content="Hi")
        choice = Choice(index=0, message=msg, finish_reason="stop")
        response = ModelResponse(id="resp-123", model="gpt-4o-mini", choices=[choice])
        dumped = response.model_dump()
        assert dumped["id"] == "resp-123"
        assert dumped["model"] == "gpt-4o-mini"
        assert len(dumped["choices"]) == 1


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_create_stream_chunk(self):
        """Test basic StreamChunk creation."""
        delta = ChunkDelta(content="Hello")
        choice = ChunkChoice(index=0, delta=delta)
        chunk = StreamChunk(id="chunk-1", model="gpt-4o-mini", choices=[choice])

        assert chunk.id == "chunk-1"
        assert chunk.choices[0].delta.content == "Hello"

    def test_chunk_with_usage(self):
        """Test chunk with usage (include_usage=True)."""
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk = StreamChunk(id="chunk-1", model="gpt-4o-mini", choices=[], usage=usage)
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 15


class TestEmbedding:
    """Tests for embedding types."""

    def test_create_embedding_data(self):
        """Test EmbeddingData creation."""
        data = EmbeddingData(index=0, embedding=[0.1, 0.2, 0.3])
        assert data.index == 0
        assert len(data.embedding) == 3

    def test_create_embedding_response(self):
        """Test EmbeddingResponse creation."""
        data = [
            EmbeddingData(index=0, embedding=[0.1, 0.2]),
            EmbeddingData(index=1, embedding=[0.3, 0.4]),
        ]
        usage = EmbeddingUsage(prompt_tokens=10, total_tokens=10)
        response = EmbeddingResponse(model="text-embedding-3-small", data=data, usage=usage)
        assert response.model == "text-embedding-3-small"
        assert len(response.data) == 2
        assert response.usage.prompt_tokens == 10

    def test_embedding_model_dump(self):
        """Test embedding model_dump."""
        data = EmbeddingData(index=0, embedding=[0.1, 0.2])
        dumped = data.model_dump()
        assert dumped["index"] == 0
        assert dumped["embedding"] == [0.1, 0.2]
