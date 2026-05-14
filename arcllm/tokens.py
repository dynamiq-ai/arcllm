"""
Token counting helper.

Provides :func:`token_counter` as a drop-in for ``litellm.token_counter``.

Resolution order:

1. If `tiktoken` is installed (via ``pip install arcllm-sdk[tokenize]``) AND the
   model belongs to the OpenAI / Azure / Groq / Together / Fireworks /
   DeepSeek / Perplexity family — use the appropriate ``tiktoken`` encoder.
2. Otherwise — fall back to a chars/4 heuristic per OpenAI's published rule
   of thumb. Emits a one-time warning so callers know the count is approximate.

The heuristic is "good enough" for context-window guard rails (the typical
caller use case — token-budget enforcement, history pruning) but not for
billing. For billing, use the provider-reported usage on the response.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

__all__ = ["token_counter"]


_heuristic_warned = False


def _warn_heuristic_once(model: str) -> None:
    global _heuristic_warned
    if _heuristic_warned:
        return
    _heuristic_warned = True
    warnings.warn(
        f"arcllm.token_counter: tiktoken not installed; falling back to a "
        f"chars/4 heuristic for model {model!r}. "
        f"Install `arcllm-sdk[tokenize]` for accurate counts.",
        UserWarning,
        stacklevel=3,
    )


def _normalise_for_tiktoken(model: str) -> str | None:
    """Return the tiktoken encoder name for this model, or ``None``.

    We don't try to be exhaustive — if tiktoken doesn't recognise the model
    we fall back to ``cl100k_base`` (OpenAI's most common encoding) for any
    model id that obviously belongs to the OpenAI / OpenAI-compat family.
    """
    m = model.lower()
    # Strip provider prefix.
    if "/" in m:
        m = m.split("/", 1)[1]
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-", "text-embedding-")):
        return m
    # Common OpenAI-compat hosts.
    if any(
        token in m
        for token in (
            "llama",
            "mistral",
            "mixtral",
            "qwen",
            "deepseek",
            "command",
            "cohere",
            "phi",
            "gemma",
            "kimi",
            "glm",
        )
    ):
        return "cl100k_base"  # best-effort encoder; tiktoken accepts as a name
    return None


def _count_text_with_tiktoken(text: str, model: str) -> int | None:
    """Try tiktoken, return None if unavailable."""
    try:
        import tiktoken
    except ImportError:
        return None

    encoder_name = _normalise_for_tiktoken(model)
    if encoder_name is None:
        return None
    try:
        encoder = tiktoken.encoding_for_model(encoder_name)
    except (KeyError, ValueError):
        try:
            encoder = tiktoken.get_encoding(encoder_name)
        except (KeyError, ValueError):
            try:
                encoder = tiktoken.get_encoding("cl100k_base")
            except (KeyError, ValueError):
                return None
    return len(encoder.encode(text))


def _heuristic_count(text: str) -> int:
    """Fallback when tiktoken isn't available.

    Per OpenAI's published rule of thumb, English text averages roughly
    4 characters per token. We round up so callers don't underestimate.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Flatten OpenAI-shape messages to the text we'd pass to a tokenizer.

    Each message contributes role + content. Content can be a string or a
    list of content parts (vision, tool calls, etc.); for non-text parts we
    include a placeholder so context-window math doesn't underestimate.
    """
    pieces: list[str] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        content: Any = msg.get("content")
        pieces.append(role)
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            # mypy narrows ``content`` to ``list[Any]`` here; pyright doesn't,
            # so we cast for pyright's benefit and silence mypy's redundant
            # warning. Both type checkers run in --strict in CI.
            content_list = cast("list[Any]", content)  # type: ignore[redundant-cast]
            for raw_part in content_list:
                if not isinstance(raw_part, dict):
                    continue
                part = cast("dict[str, Any]", raw_part)
                kind = part.get("type")
                if kind == "text":
                    pieces.append(str(part.get("text", "")))
                elif kind == "image_url":
                    # Image cost varies per model; reserve ~85 tokens for the
                    # tag so guard rails don't undercount. (OpenAI documents
                    # 85 tokens for low-detail images.)
                    pieces.append("[image]" * 21)
                elif kind in {"input_file", "file"}:
                    pieces.append("[file]" * 21)
        # Tool calls aren't tokenised here; provider counts them server-side.
    return "\n".join(pieces)


def token_counter(
    *,
    model: str,
    messages: list[dict[str, Any]] | None = None,
    text: str | None = None,
) -> int:
    """Count tokens for the given model.

    Pass either ``messages`` (OpenAI-shape list) or ``text`` (raw string).
    Returns the best-available count: tiktoken-precise for OpenAI-family
    models when ``arcllm-sdk[tokenize]`` is installed, otherwise a chars/4
    heuristic with a one-time warning.

    For ``messages`` lists, follows OpenAI's published per-message
    overhead formula (3 tokens per message + 3 priming tokens for the
    final assistant turn) so counts are comparable to litellm and to
    OpenAI's own ``tiktoken`` cookbook examples. Without the overhead,
    arcllm would systematically undercount, and callers doing
    history-pruning / context-window enforcement would preserve more
    messages than the model can actually hold.

    Raises ``ValueError`` if both ``messages`` and ``text`` are missing.
    """
    if messages is None and text is None:
        raise ValueError("token_counter requires either `messages` or `text`")
    if messages is not None and text is not None:
        raise ValueError("token_counter accepts `messages` or `text`, not both")

    if text is not None:
        count = _count_text_with_tiktoken(text, model)
        if count is not None:
            return count
        _warn_heuristic_once(model)
        return _heuristic_count(text)

    # Messages path — count each field separately and add per-message
    # overhead so the total matches OpenAI's chat-completion accounting
    # (and litellm's, which uses the same formula).
    msgs = messages or []
    per_message = 3
    per_name = 1
    total = 0
    for msg in msgs:
        total += per_message
        for key, value in msg.items():
            if value is None:
                continue
            if isinstance(value, str):
                field_count = _count_text_with_tiktoken(value, model)
                if field_count is None:
                    field_count = _heuristic_count(value)
                total += field_count
            else:
                # Non-string fields (content arrays for vision, tool_calls
                # JSON, etc.) — flatten to text and count.
                flattened = _flatten_messages([{key: value}])
                field_count = _count_text_with_tiktoken(flattened, model)
                if field_count is None:
                    field_count = _heuristic_count(flattened)
                total += field_count
            if key == "name":
                total += per_name
    # Priming tokens for the assistant's reply.
    total += 3
    if _count_text_with_tiktoken("", model) is None:
        _warn_heuristic_once(model)
    return total
