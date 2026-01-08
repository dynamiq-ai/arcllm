"""
Comprehensive streaming tests for arcllm.

Tests SSE parsing, streaming edge cases, tool call deltas,
final chunk handling, and usage reporting in streams.
"""

from __future__ import annotations

import pytest

from arcllm.http.sse import AsyncSSEParser, SSEEvent, SSEParser


class TestSSEParserPartialChunks:
    """Tests for partial chunk handling in SSE parser."""

    def test_split_in_middle_of_data_field(self):
        """Test handling data: split across chunks."""
        parser = SSEParser()

        # First chunk: "dat"
        events = list(parser.feed("dat"))
        assert len(events) == 0

        # Second chunk: "a: hello\n\n"
        events = list(parser.feed("a: hello\n\n"))
        assert len(events) == 1
        assert events[0].data == "hello"

    def test_split_in_newline_sequence(self):
        """Test split in the middle of \\n\\n."""
        parser = SSEParser()

        events = list(parser.feed("data: hello\n"))
        assert len(events) == 0

        events = list(parser.feed("\n"))
        assert len(events) == 1
        assert events[0].data == "hello"

    def test_very_small_chunks(self):
        """Test with single-character chunks."""
        parser = SSEParser()
        message = "data: test\n\n"

        events = []
        for char in message:
            events.extend(parser.feed(char))

        assert len(events) == 1
        assert events[0].data == "test"


class TestSSEParserPartialUTF8:
    """Tests for partial UTF-8 handling.

    Note: The SSE parser uses errors='replace' for UTF-8 decoding, which means
    incomplete multi-byte sequences at chunk boundaries will be replaced with
    the Unicode replacement character (U+FFFD). This is a tradeoff for
    simplicity and robustness - in practice, API providers send complete
    UTF-8 sequences within SSE data fields.
    """

    def test_complete_utf8_2byte(self):
        """Test complete 2-byte UTF-8 character."""
        parser = SSEParser()
        # ñ is bytes b'\xc3\xb1' in UTF-8
        events = list(parser.feed("data: mañana\n\n"))
        assert len(events) == 1
        assert events[0].data == "mañana"

    def test_complete_utf8_3byte(self):
        """Test complete 3-byte UTF-8 character."""
        parser = SSEParser()
        # € is bytes b'\xe2\x82\xac' in UTF-8
        events = list(parser.feed("data: 100€\n\n"))
        assert len(events) == 1
        assert events[0].data == "100€"

    def test_complete_utf8_4byte_emoji(self):
        """Test complete 4-byte UTF-8 emoji."""
        parser = SSEParser()
        events = list(parser.feed("data: Hello 😀\n\n"))
        assert len(events) == 1
        assert events[0].data == "Hello 😀"

    def test_multiple_unicode_chars(self):
        """Test multiple unicode characters in one event."""
        parser = SSEParser()
        events = list(parser.feed("data: Héllo Wörld 你好 👋\n\n"))
        assert len(events) == 1
        assert events[0].data == "Héllo Wörld 你好 👋"


class TestSSEParserMissingNewlines:
    """Tests for missing/unusual newline handling."""

    def test_single_newline_not_event(self):
        """Test that single newline doesn't dispatch event."""
        parser = SSEParser()

        events = list(parser.feed("data: hello\n"))
        assert len(events) == 0

        # Add another data line
        events = list(parser.feed("data: world\n\n"))
        assert len(events) == 1
        assert events[0].data == "hello\nworld"

    def test_multiple_blank_lines(self):
        """Test multiple blank lines between events."""
        parser = SSEParser()

        events = list(parser.feed("data: first\n\n\n\ndata: second\n\n"))
        # Should get 2 events, extra blank lines are ignored
        assert len(events) == 2
        assert events[0].data == "first"
        assert events[1].data == "second"

    def test_windows_crlf(self):
        """Test Windows-style CRLF line endings."""
        parser = SSEParser()

        events = list(parser.feed("data: hello\r\n\r\n"))
        assert len(events) == 1
        assert events[0].data == "hello"


class TestSSEParserKeepAlive:
    """Tests for keep-alive (comment) handling."""

    def test_colon_comment(self):
        """Test : comment lines are ignored."""
        parser = SSEParser()

        events = list(parser.feed(": keep-alive\n\n"))
        assert len(events) == 0

    def test_comment_between_events(self):
        """Test comments between data events."""
        parser = SSEParser()

        events = list(parser.feed("data: first\n\n: ping\n\ndata: second\n\n"))
        assert len(events) == 2
        assert events[0].data == "first"
        assert events[1].data == "second"

    def test_multiple_keepalive_comments(self):
        """Test multiple keep-alive comments in sequence."""
        parser = SSEParser()

        events = list(parser.feed(": ping\n\n: pong\n\ndata: hello\n\n"))
        assert len(events) == 1
        assert events[0].data == "hello"


class TestSSEParserToolCallDeltas:
    """Tests for tool call delta parsing in SSE."""

    def test_parse_tool_call_start(self):
        """Test parsing initial tool call delta."""
        parser = SSEParser()

        data = '{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather"}}]}}]}'
        events = list(parser.feed(f"data: {data}\n\n"))

        assert len(events) == 1
        import json

        parsed = json.loads(events[0].data)
        assert parsed["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_123"

    def test_parse_tool_call_arguments_delta(self):
        """Test parsing tool call arguments delta."""
        parser = SSEParser()

        chunks = [
            '{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"loc"}}]}}]}',
            '{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\\": \\"NYC\\"}"}}]}}]}',
        ]

        events = []
        for chunk in chunks:
            events.extend(parser.feed(f"data: {chunk}\n\n"))

        assert len(events) == 2

        import json

        first = json.loads(events[0].data)
        second = json.loads(events[1].data)

        assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"loc'
        assert (
            second["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            == 'ation": "NYC"}'
        )


class TestSSEParserFinalChunk:
    """Tests for final chunk handling."""

    def test_finish_reason_in_final_chunk(self):
        """Test finish_reason in final chunk."""
        parser = SSEParser()

        data = '{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        events = list(parser.feed(f"data: {data}\n\n"))

        assert len(events) == 1
        import json

        parsed = json.loads(events[0].data)
        assert parsed["choices"][0]["finish_reason"] == "stop"

    def test_usage_in_final_chunk(self):
        """Test usage reporting in final chunk."""
        parser = SSEParser()

        data = '{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}'
        events = list(parser.feed(f"data: {data}\n\n"))

        import json

        parsed = json.loads(events[0].data)
        assert parsed["usage"]["prompt_tokens"] == 10
        assert parsed["usage"]["completion_tokens"] == 5
        assert parsed["usage"]["total_tokens"] == 15

    def test_done_event_detection(self):
        """Test [DONE] event detection."""
        parser = SSEParser()

        events = list(parser.feed("data: [DONE]\n\n"))
        assert len(events) == 1
        assert events[0].is_done is True

    def test_done_with_whitespace(self):
        """Test [DONE] with surrounding whitespace."""
        parser = SSEParser()

        events = list(parser.feed("data:   [DONE]  \n\n"))
        assert len(events) == 1
        assert events[0].is_done is True


class TestSSEParserBoundedBuffering:
    """Tests to verify bounded buffering behavior."""

    def test_buffer_cleared_after_event(self):
        """Test that buffer is cleared after each event."""
        parser = SSEParser()

        # Send first event
        events = list(parser.feed("data: first\n\n"))
        assert len(events) == 1
        assert len(parser._buffer) == 0  # Buffer should be empty (bytearray)

        # Send second event
        events = list(parser.feed("data: second\n\n"))
        assert len(events) == 1
        assert len(parser._buffer) == 0

    def test_current_data_cleared(self):
        """Test that _current_data is cleared after event dispatch."""
        parser = SSEParser()

        events = list(parser.feed("data: test\n\n"))
        assert len(events) == 1
        assert len(parser._current_data) == 0

    def test_large_event_processed(self):
        """Test processing large event without memory issues."""
        parser = SSEParser()

        # Create a large JSON payload
        large_content = "x" * 100000
        data = f'{{"content": "{large_content}"}}'
        events = list(parser.feed(f"data: {data}\n\n"))

        assert len(events) == 1
        import json

        parsed = json.loads(events[0].data)
        assert len(parsed["content"]) == 100000


class TestSSEEvent:
    """Additional tests for SSEEvent."""

    def test_is_done_caching(self):
        """Test that is_done result is cached."""
        event = SSEEvent(data="[DONE]")

        # First call
        assert event.is_done is True
        # Second call should use cached value
        assert event._is_done is True
        assert event.is_done is True

    def test_event_with_all_fields(self):
        """Test event with all optional fields."""
        event = SSEEvent(
            data="test data",
            event="custom_event",
            id="msg-123",
            retry=5000,
        )

        assert event.data == "test data"
        assert event.event == "custom_event"
        assert event.id == "msg-123"
        assert event.retry == 5000

    def test_event_retry_parsing(self):
        """Test that retry field is parsed as integer."""
        parser = SSEParser()

        events = list(parser.feed("retry: 3000\ndata: test\n\n"))
        assert len(events) == 1
        assert events[0].retry == 3000


class TestAsyncSSEParserAdditional:
    """Additional tests for AsyncSSEParser."""

    @pytest.mark.asyncio
    async def test_multiple_events_async(self):
        """Test parsing multiple events asynchronously."""

        async def async_stream():
            yield b"data: first\n\n"
            yield b"data: second\n\n"
            yield b"data: [DONE]\n\n"

        parser = AsyncSSEParser()
        events = []
        async for event in parser.parse(async_stream()):
            events.append(event)

        assert len(events) == 3
        assert events[0].data == "first"
        assert events[1].data == "second"
        assert events[2].is_done is True

    @pytest.mark.asyncio
    async def test_chunked_async_stream(self):
        """Test parsing chunked async stream."""

        async def async_stream():
            yield b"data: hel"
            yield b"lo\n\n"

        parser = AsyncSSEParser()
        events = []
        async for event in parser.parse(async_stream()):
            events.append(event)

        assert len(events) == 1
        assert events[0].data == "hello"
