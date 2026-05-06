"""
Server-Sent Events (SSE) parser for streaming responses.

Handles:
- Standard SSE format (data:, event:, id:, retry:)
- Partial chunks and multi-byte UTF-8 boundaries
- OpenAI-style streaming format

Performance optimizations:
- Zero-copy parsing using bytearray + memoryview
- Pre-compiled byte constants for hot path
- Minimized allocations - only decode when yielding events
- Direct byte operations instead of string methods
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

__all__ = ["AsyncSSEParser", "SSEEvent", "SSEParser"]

# Pre-defined byte constants for zero-copy hot path
_NEWLINE_B = ord("\n")
_CARRIAGE_B = ord("\r")
_COLON_B = ord(":")
_SPACE_B = ord(" ")
_DONE_MARKER_B = b"[DONE]"

# String constants (for decoded output)
_NEWLINE = "\n"
_DONE_MARKER = "[DONE]"
_DEFAULT_EVENT = "message"

# Field markers as bytes for fast comparison
_FIELD_DATA_B = b"data"
_FIELD_EVENT_B = b"event"
_FIELD_ID_B = b"id"
_FIELD_RETRY_B = b"retry"


class SSEEvent:
    """
    A single Server-Sent Event.

    Uses __slots__ for memory efficiency and faster attribute access.
    """

    __slots__ = ("_is_done", "data", "event", "id", "retry")

    def __init__(
        self,
        data: str,
        event: str = _DEFAULT_EVENT,
        id: str | None = None,
        retry: int | None = None,
    ) -> None:
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry
        # Cache is_done check since data is immutable
        self._is_done: bool | None = None

    @property
    def is_done(self) -> bool:
        """Check if this is the [DONE] terminator event."""
        if self._is_done is None:
            # Strip and compare - cache the result
            self._is_done = self.data.strip() == _DONE_MARKER
        return self._is_done


class SSEParser:
    """
    Zero-copy SSE parser using bytearray + memoryview.

    Handles partial chunks and maintains state across calls.

    Performance optimizations:
    - bytearray buffer avoids string concatenation overhead
    - memoryview for zero-copy line extraction
    - Decodes bytes only when yielding events (lazy decode)
    - Pre-computed byte constants for fast comparison
    """

    __slots__ = (
        "_buffer",
        "_current_data",
        "_current_event",
        "_current_id",
        "_current_retry",
        "_pos",
    )

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._pos = 0  # Current parse position
        self._current_event = _DEFAULT_EVENT
        self._current_data: list[bytes] = []  # Store as bytes, decode on yield
        self._current_id: bytes | None = None
        self._current_retry: int | None = None

    def feed(self, chunk: bytes | str) -> Iterator[SSEEvent]:
        """
        Feed a chunk of data and yield any complete events.

        Args:
            chunk: Bytes or string chunk from the stream

        Yields:
            SSEEvent for each complete event in the chunk
        """
        # Convert string to bytes if needed
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")

        # Extend buffer (efficient for bytearray)
        self._buffer.extend(chunk)
        buf_len = len(self._buffer)

        # Collect events first, then yield (avoids memoryview lifetime issues)
        events_to_yield: list[SSEEvent] = []

        while self._pos < buf_len:
            # Find next newline
            newline_pos = self._buffer.find(b"\n", self._pos)
            if newline_pos == -1:
                break

            # Extract line as bytes using slice (bytearray slice returns bytes)
            line = bytes(self._buffer[self._pos : newline_pos])

            # Strip trailing \r if present
            if line and line[-1] == _CARRIAGE_B:
                line = line[:-1]

            # Move position past the newline
            self._pos = newline_pos + 1

            # Process the line
            event = self._process_line_bytes(line)
            if event is not None:
                events_to_yield.append(event)

        # Compact buffer: remove processed data
        if self._pos > 0:
            del self._buffer[: self._pos]
            self._pos = 0

        # Now yield all collected events
        yield from events_to_yield

    def _process_line_bytes(self, line: bytes) -> SSEEvent | None:
        """Process a single line (as bytes) and return event if complete."""
        # Empty line = dispatch event
        if not line:
            if self._current_data:
                # Decode and join data lines
                data_str = _NEWLINE.join(
                    d.decode("utf-8", errors="replace") for d in self._current_data
                )
                event_str = self._current_event
                id_str = (
                    self._current_id.decode("utf-8", errors="replace") if self._current_id else None
                )

                event = SSEEvent(
                    data=data_str,
                    event=event_str,
                    id=id_str,
                    retry=self._current_retry,
                )
                # Reset for next event
                self._current_data = []
                self._current_event = _DEFAULT_EVENT
                self._current_id = None
                self._current_retry = None
                return event
            return None

        # Comment line - fast byte check
        if line[0] == _COLON_B:
            return None

        # Parse field:value using bytes.find
        colon_pos = line.find(b":")
        if colon_pos != -1:
            field = line[:colon_pos]
            value = line[colon_pos + 1 :]
            # Remove single leading space if present
            if value and value[0] == _SPACE_B:
                value = value[1:]
        else:
            field = line
            value = b""

        # Field dispatch - ordered by frequency in typical SSE streams
        if field == _FIELD_DATA_B:
            self._current_data.append(value)
        elif field == _FIELD_EVENT_B:
            self._current_event = value.decode("utf-8", errors="replace")
        elif field == _FIELD_ID_B:
            self._current_id = value
        elif field == _FIELD_RETRY_B:
            with contextlib.suppress(ValueError):
                self._current_retry = int(value)

        return None

    def flush(self) -> SSEEvent | None:
        """Flush any remaining buffered event."""
        # Process any remaining buffer content
        if self._pos < len(self._buffer):
            remaining = bytes(self._buffer[self._pos :])
            if remaining:
                # Strip trailing \r if present
                if remaining[-1] == _CARRIAGE_B:
                    remaining = remaining[:-1]
                self._process_line_bytes(remaining)

        self._buffer.clear()
        self._pos = 0

        if self._current_data:
            data_str = _NEWLINE.join(
                d.decode("utf-8", errors="replace") for d in self._current_data
            )
            event_str = self._current_event
            id_str = (
                self._current_id.decode("utf-8", errors="replace") if self._current_id else None
            )

            event = SSEEvent(
                data=data_str,
                event=event_str,
                id=id_str,
                retry=self._current_retry,
            )
            self._current_data = []
            self._current_event = _DEFAULT_EVENT
            self._current_id = None
            self._current_retry = None
            return event
        return None


class AsyncSSEParser:
    """Async wrapper for SSE parsing from async byte streams."""

    __slots__ = ("_parser",)

    def __init__(self) -> None:
        self._parser = SSEParser()

    async def parse(self, stream: AsyncIterator[bytes]) -> AsyncIterator[SSEEvent]:
        """
        Parse SSE events from an async byte stream.

        Args:
            stream: Async iterator yielding bytes chunks

        Yields:
            SSEEvent for each complete event
        """
        async for chunk in stream:
            for event in self._parser.feed(chunk):
                yield event

        # Flush any remaining event
        final = self._parser.flush()
        if final:
            yield final


def parse_sse_stream(byte_stream: Iterator[bytes]) -> Iterator[SSEEvent]:
    """
    Convenience function to parse SSE events from a byte stream.

    Args:
        byte_stream: Iterator yielding bytes chunks

    Yields:
        SSEEvent for each complete event
    """
    parser = SSEParser()
    for chunk in byte_stream:
        yield from parser.feed(chunk)

    final = parser.flush()
    if final:
        yield final


async def parse_sse_stream_async(
    byte_stream: AsyncIterator[bytes],
) -> AsyncIterator[SSEEvent]:
    """
    Convenience function to parse SSE events from an async byte stream.

    Args:
        byte_stream: Async iterator yielding bytes chunks

    Yields:
        SSEEvent for each complete event
    """
    parser = AsyncSSEParser()
    async for event in parser.parse(byte_stream):
        yield event
