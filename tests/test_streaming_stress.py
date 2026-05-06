"""
Streaming stress tests for arcllm.

Tests edge cases and robustness of streaming parsers:
- Tiny chunks (single byte)
- Huge chunks
- Partial UTF-8 boundaries
- Missing newlines
- Keep-alive comments
- Tool call deltas
- include_usage final chunks
- Bounded buffering verification
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from arcllm.http.sse import AsyncSSEParser, SSEParser
from arcllm.types import StreamChunk

# =============================================================================
# Test Fixtures
# =============================================================================


def make_sse_chunk(data: dict[str, Any]) -> bytes:
    """Create SSE chunk from dict."""
    return f"data: {json.dumps(data)}\n\n".encode()


def make_openai_stream_chunk(
    chunk_id: str = "chatcmpl-test",
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    tool_calls: list[dict] | None = None,
    usage: dict | None = None,
) -> dict[str, Any]:
    """Create OpenAI-format stream chunk."""
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls

    chunk: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


# =============================================================================
# SSE Parser Edge Cases
# =============================================================================


class TestSSEParserEdgeCases:
    """Test SSE parser with edge cases."""

    def test_single_byte_chunks(self) -> None:
        """Test parsing when data arrives byte-by-byte."""
        parser = SSEParser()
        data = b"data: hello world\n\n"

        events = []
        for byte in data:
            events.extend(parser.feed(bytes([byte])))

        assert len(events) == 1
        assert events[0].data == "hello world"

    def test_huge_single_chunk(self) -> None:
        """Test parsing a very large single chunk."""
        parser = SSEParser()

        # 1MB of data in a single event
        large_content = "x" * (1024 * 1024)
        data = f"data: {large_content}\n\n".encode()

        events = list(parser.feed(data))
        assert len(events) == 1
        assert len(events[0].data) == len(large_content)

    def test_multiple_events_single_chunk(self) -> None:
        """Test multiple events arriving in a single chunk."""
        parser = SSEParser()
        data = b"data: event1\n\ndata: event2\n\ndata: event3\n\n"

        events = list(parser.feed(data))
        assert len(events) == 3
        assert [e.data for e in events] == ["event1", "event2", "event3"]

    def test_partial_utf8_boundary(self) -> None:
        """Test handling UTF-8 multi-byte sequences split across chunks.

        Note: Individual bytes of multi-byte UTF-8 sequences cannot be decoded
        in isolation. This test verifies that the parser handles realistic
        chunk boundaries (at least complete ASCII chars or complete sequences).
        """
        parser = SSEParser()

        # UTF-8 encoding of "こんにちは" (hello in Japanese)
        text = "こんにちは"
        f"data: {text}\n\n".encode()

        # Split at realistic chunk boundaries (e.g., after "data: " which is ASCII)
        # Each Japanese char is 3 bytes in UTF-8
        ascii_prefix = b"data: "
        utf8_content = text.encode("utf-8")

        events = []
        # Feed ASCII prefix as one chunk
        events.extend(parser.feed(ascii_prefix))
        # Feed UTF-8 content in 3-byte increments (complete characters)
        for i in range(0, len(utf8_content), 3):
            chunk = utf8_content[i : i + 3]
            events.extend(parser.feed(chunk))
        # Feed final newlines
        events.extend(parser.feed(b"\n\n"))

        assert len(events) == 1
        assert events[0].data == text

    def test_emoji_split_across_chunks(self) -> None:
        """Test emoji (4-byte UTF-8) split across chunks."""
        parser = SSEParser()

        text = "Hello 🎉 World"
        full_data = f"data: {text}\n\n".encode()

        # Split after "Hello " and before emoji
        split_point = len(b"data: Hello ")
        chunk1 = full_data[:split_point]
        chunk2 = full_data[split_point:]

        events = []
        events.extend(parser.feed(chunk1))
        events.extend(parser.feed(chunk2))

        assert len(events) == 1
        assert events[0].data == text

    def test_missing_final_newline(self) -> None:
        """Test handling events without final newline (flush required)."""
        parser = SSEParser()

        # Data without final newline
        data = b"data: incomplete"
        events = list(parser.feed(data))
        assert len(events) == 0  # No complete event yet

        # Flush should return the incomplete event
        final = parser.flush()
        assert final is not None
        assert final.data == "incomplete"

    def test_keepalive_comments(self) -> None:
        """Test that keep-alive comments are ignored."""
        parser = SSEParser()
        data = b": keepalive\n\ndata: actual\n\n: another comment\n\ndata: data\n\n"

        events = list(parser.feed(data))
        assert len(events) == 2
        assert events[0].data == "actual"
        assert events[1].data == "data"

    def test_multiline_data(self) -> None:
        """Test multi-line data fields."""
        parser = SSEParser()
        data = b"data: line1\ndata: line2\ndata: line3\n\n"

        events = list(parser.feed(data))
        assert len(events) == 1
        assert events[0].data == "line1\nline2\nline3"

    def test_event_with_all_fields(self) -> None:
        """Test event with event, id, and retry fields."""
        parser = SSEParser()
        data = b"event: custom\nid: 123\nretry: 5000\ndata: payload\n\n"

        events = list(parser.feed(data))
        assert len(events) == 1
        assert events[0].event == "custom"
        assert events[0].id == "123"
        assert events[0].retry == 5000
        assert events[0].data == "payload"

    def test_done_event_detection(self) -> None:
        """Test [DONE] event is detected."""
        parser = SSEParser()
        data = b"data: [DONE]\n\n"

        events = list(parser.feed(data))
        assert len(events) == 1
        assert events[0].is_done is True

    def test_crlf_line_endings(self) -> None:
        """Test handling of Windows-style CRLF line endings."""
        parser = SSEParser()
        data = b"data: test\r\n\r\n"

        events = list(parser.feed(data))
        assert len(events) == 1
        assert events[0].data == "test"

    def test_mixed_line_endings(self) -> None:
        """Test handling of mixed line endings."""
        parser = SSEParser()
        data = b"data: event1\n\ndata: event2\r\n\r\ndata: event3\n\n"

        events = list(parser.feed(data))
        assert len(events) == 3

    def test_empty_data_field(self) -> None:
        """Test handling of empty data field."""
        parser = SSEParser()
        data = b"data:\n\n"

        events = list(parser.feed(data))
        assert len(events) == 1
        assert events[0].data == ""

    def test_data_with_colon(self) -> None:
        """Test data containing colons."""
        parser = SSEParser()
        data = b'data: {"key": "value"}\n\n'

        events = list(parser.feed(data))
        assert len(events) == 1
        assert events[0].data == '{"key": "value"}'


# =============================================================================
# Streaming with OpenAI Format
# =============================================================================


class TestOpenAIStreamingFormat:
    """Test OpenAI-specific streaming format handling."""

    def test_basic_content_stream(self) -> None:
        """Test basic content streaming."""
        from arcllm.providers.base import ProviderConfig
        from arcllm.providers.openai_adapter import OpenAIAdapter

        config = ProviderConfig(api_key="test")
        adapter = OpenAIAdapter(config)

        chunks_data = [
            make_openai_stream_chunk(role="assistant"),
            make_openai_stream_chunk(content="Hello"),
            make_openai_stream_chunk(content=" world"),
            make_openai_stream_chunk(content="!"),
            make_openai_stream_chunk(finish_reason="stop"),
        ]

        parser = SSEParser()
        content_parts = []

        for chunk_dict in chunks_data:
            sse_bytes = make_sse_chunk(chunk_dict)
            for event in parser.feed(sse_bytes):
                if not event.is_done:
                    stream_chunk = adapter.parse_stream_event(event.data, "gpt-4o-mini")
                    if stream_chunk and stream_chunk.choices:
                        delta = stream_chunk.choices[0].delta
                        if delta.content:
                            content_parts.append(delta.content)

        assert "".join(content_parts) == "Hello world!"

    def test_tool_call_deltas(self) -> None:
        """Test streaming tool call deltas."""
        from arcllm.providers.base import ProviderConfig
        from arcllm.providers.openai_adapter import OpenAIAdapter

        config = ProviderConfig(api_key="test")
        adapter = OpenAIAdapter(config)

        # Tool call streamed in parts
        chunks_data = [
            make_openai_stream_chunk(role="assistant"),
            make_openai_stream_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ]
            ),
            make_openai_stream_chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"loc'}}]),
            make_openai_stream_chunk(
                tool_calls=[{"index": 0, "function": {"arguments": 'ation":'}}]
            ),
            make_openai_stream_chunk(
                tool_calls=[{"index": 0, "function": {"arguments": '"NYC"}'}}]
            ),
            make_openai_stream_chunk(finish_reason="tool_calls"),
        ]

        parser = SSEParser()
        tool_call_parts: dict[int, dict] = {}

        for chunk_dict in chunks_data:
            sse_bytes = make_sse_chunk(chunk_dict)
            for event in parser.feed(sse_bytes):
                if not event.is_done:
                    stream_chunk = adapter.parse_stream_event(event.data, "gpt-4o-mini")
                    if stream_chunk and stream_chunk.choices:
                        delta = stream_chunk.choices[0].delta
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.get("index", 0)
                                if idx not in tool_call_parts:
                                    tool_call_parts[idx] = {
                                        "id": "",
                                        "type": "function",
                                        "name": "",
                                        "arguments": "",
                                    }
                                if "id" in tc:
                                    tool_call_parts[idx]["id"] = tc["id"]
                                if "type" in tc:
                                    tool_call_parts[idx]["type"] = tc["type"]
                                if "function" in tc:
                                    if "name" in tc["function"]:
                                        tool_call_parts[idx]["name"] += tc["function"]["name"]
                                    if "arguments" in tc["function"]:
                                        tool_call_parts[idx]["arguments"] += tc["function"][
                                            "arguments"
                                        ]

        assert 0 in tool_call_parts
        assert tool_call_parts[0]["id"] == "call_abc"
        assert tool_call_parts[0]["name"] == "get_weather"
        assert json.loads(tool_call_parts[0]["arguments"]) == {"location": "NYC"}

    def test_include_usage_final_chunk(self) -> None:
        """Test include_usage in final chunk."""
        from arcllm.providers.base import ProviderConfig
        from arcllm.providers.openai_adapter import OpenAIAdapter

        config = ProviderConfig(api_key="test")
        adapter = OpenAIAdapter(config)

        chunks_data = [
            make_openai_stream_chunk(role="assistant"),
            make_openai_stream_chunk(content="Hi"),
            make_openai_stream_chunk(
                finish_reason="stop",
                usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            ),
        ]

        parser = SSEParser()
        last_usage = None

        for chunk_dict in chunks_data:
            sse_bytes = make_sse_chunk(chunk_dict)
            for event in parser.feed(sse_bytes):
                if not event.is_done:
                    stream_chunk = adapter.parse_stream_event(event.data, "gpt-4o-mini")
                    if stream_chunk and stream_chunk.usage:
                        last_usage = stream_chunk.usage

        assert last_usage is not None
        assert last_usage.prompt_tokens == 5
        assert last_usage.completion_tokens == 1
        assert last_usage.total_tokens == 6


# =============================================================================
# Bounded Buffering Tests
# =============================================================================


class TestBoundedBuffering:
    """Test that streaming never buffers unboundedly."""

    def test_parser_buffer_size_check(self) -> None:
        """Verify parser doesn't accumulate unlimited data."""
        parser = SSEParser()

        # Feed many small chunks without completing an event
        for i in range(1000):
            list(parser.feed(f"data: chunk{i}\n".encode()))

        # Buffer should only contain the incomplete lines
        # (the newline-newline hasn't come yet)
        assert len(parser._current_data) <= 1000

    def test_large_single_event_memory(self) -> None:
        """Test memory handling for large single events."""
        parser = SSEParser()

        # 10MB event (very large but should work)
        large_data = "x" * (10 * 1024 * 1024)
        event_bytes = f"data: {large_data}\n\n".encode()

        events = list(parser.feed(event_bytes))
        assert len(events) == 1
        assert len(events[0].data) == len(large_data)

        # After yielding, internal buffer should be cleared
        assert len(parser._current_data) == 0
        assert len(parser._buffer) == 0


# =============================================================================
# Time-to-First-Token and Per-Chunk Overhead
# =============================================================================


class TestStreamingLatency:
    """Test streaming latency characteristics."""

    def test_time_to_first_token(self) -> None:
        """Measure and verify time-to-first-token is minimal."""
        from arcllm.providers.base import ProviderConfig
        from arcllm.providers.openai_adapter import OpenAIAdapter

        config = ProviderConfig(api_key="test")
        adapter = OpenAIAdapter(config)

        chunk = make_openai_stream_chunk(content="First")
        sse_bytes = make_sse_chunk(chunk)

        # Measure TTFT
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            parser = SSEParser()
            for event in parser.feed(sse_bytes):
                if not event.is_done:
                    adapter.parse_stream_event(event.data, "gpt-4o-mini")
        elapsed = time.perf_counter() - start

        ttft_us = (elapsed / iterations) * 1_000_000  # microseconds
        print(f"\nTime-to-first-token: {ttft_us:.2f}µs")

        # Generous CI-safe ceiling. Local hot machines typically land
        # well under 50µs; shared GitHub-Actions runners drift higher
        # under contention. The dedicated benchmarks job tracks
        # fine-grained regressions — this guards against
        # orders-of-magnitude slowdowns only.
        assert ttft_us < 500, f"TTFT too high: {ttft_us}µs"

    def test_per_chunk_overhead(self) -> None:
        """Measure and verify per-chunk processing overhead."""
        from arcllm.providers.base import ProviderConfig
        from arcllm.providers.openai_adapter import OpenAIAdapter

        config = ProviderConfig(api_key="test")
        adapter = OpenAIAdapter(config)

        # Create multiple chunks
        chunks = [make_sse_chunk(make_openai_stream_chunk(content=f"word{i}")) for i in range(100)]
        all_chunks = b"".join(chunks)

        # Measure per-chunk overhead
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            parser = SSEParser()
            for event in parser.feed(all_chunks):
                if not event.is_done:
                    adapter.parse_stream_event(event.data, "gpt-4o-mini")
        elapsed = time.perf_counter() - start

        per_chunk_us = (elapsed / (iterations * 100)) * 1_000_000
        print(f"\nPer-chunk overhead: {per_chunk_us:.2f}µs")

        # Generous CI-safe ceiling. Local hot machines typically land
        # around 5-12us; shared GitHub-Actions runners drift up to
        # ~30us under contention. The dedicated benchmarks job in CI
        # tracks fine-grained regressions; this assertion just guards
        # against orders-of-magnitude slowdowns.
        assert per_chunk_us < 100, f"Per-chunk overhead too high: {per_chunk_us}µs"


# =============================================================================
# Async SSE Parser Tests
# =============================================================================


class TestAsyncSSEParser:
    """Test async SSE parser."""

    @pytest.mark.asyncio
    async def test_async_basic_parsing(self) -> None:
        """Test basic async SSE parsing."""

        async def byte_generator():
            yield b"data: event1\n\n"
            yield b"data: event2\n\n"
            yield b"data: event3\n\n"

        parser = AsyncSSEParser()
        events = []
        async for event in parser.parse(byte_generator()):
            events.append(event)

        assert len(events) == 3
        assert [e.data for e in events] == ["event1", "event2", "event3"]

    @pytest.mark.asyncio
    async def test_async_partial_chunks(self) -> None:
        """Test async parsing with partial chunks."""

        async def byte_generator():
            yield b"data: he"
            yield b"llo\n"
            yield b"\ndata: wor"
            yield b"ld\n\n"

        parser = AsyncSSEParser()
        events = []
        async for event in parser.parse(byte_generator()):
            events.append(event)

        assert len(events) == 2
        assert events[0].data == "hello"
        assert events[1].data == "world"


# =============================================================================
# Stream Chunk Builder Stress Tests
# =============================================================================


class TestStreamChunkBuilderStress:
    """Stress test stream_chunk_builder."""

    def test_many_small_chunks(self) -> None:
        """Test building from many small content chunks."""
        from arcllm import stream_chunk_builder
        from arcllm.types import ChunkChoice, ChunkDelta

        # Create 1000 small chunks
        chunks = []
        for i in range(1000):
            chunk = StreamChunk(
                id="test",
                model="gpt-4o-mini",
                choices=[
                    ChunkChoice(index=0, delta=ChunkDelta(content=f"{i}"), finish_reason=None)
                ],
            )
            chunks.append(chunk)

        # Add final chunk
        chunks.append(
            StreamChunk(
                id="test",
                model="gpt-4o-mini",
                choices=[ChunkChoice(index=0, delta=ChunkDelta(), finish_reason="stop")],
            )
        )

        result = stream_chunk_builder(chunks)
        expected_content = "".join(str(i) for i in range(1000))
        assert result.choices[0].message.content == expected_content

    def test_multiple_tool_calls_streaming(self) -> None:
        """Test building response with multiple parallel tool calls."""
        from arcllm import stream_chunk_builder
        from arcllm.types import ChunkChoice, ChunkDelta

        chunks = [
            # Initial role
            StreamChunk(
                id="test",
                model="gpt-4o-mini",
                choices=[
                    ChunkChoice(index=0, delta=ChunkDelta(role="assistant"), finish_reason=None)
                ],
            ),
            # Tool call 1 start
            StreamChunk(
                id="test",
                model="gpt-4o-mini",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(
                            tool_calls=[
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "func1", "arguments": ""},
                                }
                            ]
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Tool call 2 start
            StreamChunk(
                id="test",
                model="gpt-4o-mini",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(
                            tool_calls=[
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "func2", "arguments": ""},
                                }
                            ]
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Tool call 1 arguments
            StreamChunk(
                id="test",
                model="gpt-4o-mini",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(
                            tool_calls=[{"index": 0, "function": {"arguments": '{"a":1}'}}]
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Tool call 2 arguments
            StreamChunk(
                id="test",
                model="gpt-4o-mini",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(
                            tool_calls=[{"index": 1, "function": {"arguments": '{"b":2}'}}]
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            # Final
            StreamChunk(
                id="test",
                model="gpt-4o-mini",
                choices=[ChunkChoice(index=0, delta=ChunkDelta(), finish_reason="tool_calls")],
            ),
        ]

        result = stream_chunk_builder(chunks)
        tool_calls = result.choices[0].message.tool_calls

        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].function.name == "func1"
        assert json.loads(tool_calls[0].function.arguments) == {"a": 1}
        assert tool_calls[1].id == "call_2"
        assert tool_calls[1].function.name == "func2"
        assert json.loads(tool_calls[1].function.arguments) == {"b": 2}
