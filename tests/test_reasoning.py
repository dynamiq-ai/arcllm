"""Integration-style unit tests for capability-aware param routing.

These tests exercise the path:

    arcllm.completion(...) -> adapter.build_request(...) ->
    BaseAdapter._check_params -> BaseAdapter._drop_by_capabilities

against a mocked HTTP layer. They cover the primary "modern reasoning model"
shapes:

- OpenAI o-series (o4-mini): rejects ``temperature``/``top_p``, takes
  ``reasoning_effort``.
- OpenAI gpt-5: rejects ``temperature`` per the capability table.
- Anthropic Claude with extended thinking: ``thinking_budget`` -> ``thinking``
  block; ``temperature`` is dropped pre-emptively.
- Gemini 2.5 with thinking_config: ``thinking_budget`` /
  ``include_thoughts`` -> ``generationConfig.thinkingConfig``.

Wire-format only — no live calls. We construct the adapter directly and
inspect the JSON body the adapter would have sent.
"""

from __future__ import annotations

import warnings

import orjson
import pytest

from arcllm.providers.anthropic_adapter import AnthropicAdapter
from arcllm.providers.base import ProviderConfig
from arcllm.providers.gemini_adapter import GeminiAdapter
from arcllm.providers.openai_adapter import OpenAIAdapter


@pytest.fixture
def openai_adapter() -> OpenAIAdapter:
    return OpenAIAdapter(ProviderConfig(api_key="test"))


@pytest.fixture
def anthropic_adapter() -> AnthropicAdapter:
    return AnthropicAdapter(ProviderConfig(api_key="test"))


@pytest.fixture
def gemini_adapter() -> GeminiAdapter:
    return GeminiAdapter(config=ProviderConfig(api_key="test"))


class TestReasoningParamDrops:
    """Capability-aware filter drops params reasoning models reject."""

    def test_o4_mini_drops_temperature_with_warning(self, openai_adapter: OpenAIAdapter) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            req = openai_adapter.build_request(
                model="o4-mini",
                messages=[{"role": "user", "content": "What is 7*8?"}],
                temperature=0.5,
                top_p=0.9,
                reasoning_effort="medium",
                max_tokens=64,
            )

        body = orjson.loads(req.body or b"")
        assert "temperature" not in body
        assert "top_p" not in body
        assert body["reasoning_effort"] == "medium"
        # OpenAI o-series uses max_completion_tokens instead of max_tokens.
        assert "max_tokens" not in body
        assert body["max_completion_tokens"] == 64

        warning_messages = [str(w.message) for w in captured]
        assert any("temperature" in msg and "o4-mini" in msg for msg in warning_messages)
        assert any("top_p" in msg and "o4-mini" in msg for msg in warning_messages)

    def test_gpt5_hybrid_accepts_temperature_and_reasoning_effort(
        self, openai_adapter: OpenAIAdapter
    ) -> None:
        """GPT-5 is a *hybrid*: it accepts both ``temperature`` (chat path) and
        ``reasoning_effort`` (reasoning path). Pure o-series models would 400
        on ``temperature``; gpt-5 does not."""
        req = openai_adapter.build_request(
            model="gpt-5",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            reasoning_effort="medium",
            max_tokens=10,
        )

        body = orjson.loads(req.body or b"")
        assert body["temperature"] == 0.7
        assert body["reasoning_effort"] == "medium"
        # GPT-5 family also uses max_completion_tokens.
        assert body["max_completion_tokens"] == 10
        assert "max_tokens" not in body

    def test_chat_models_keep_temperature(self, openai_adapter: OpenAIAdapter) -> None:
        """Non-reasoning chat models retain ``temperature``."""
        req = openai_adapter.build_request(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
        )
        body = orjson.loads(req.body or b"")
        assert body["temperature"] == 0.5


class TestAnthropicThinkingBudget:
    """Anthropic ``thinking_budget`` translates to the ``thinking`` block."""

    def test_thinking_budget_emits_block_and_drops_temperature(
        self, anthropic_adapter: AnthropicAdapter
    ) -> None:
        req = anthropic_adapter.build_request(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "Solve"}],
            thinking_budget=2048,
            temperature=0.7,
            max_tokens=4096,
        )
        body = orjson.loads(req.body or b"")
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "temperature" not in body
        assert "top_p" not in body

    def test_no_thinking_budget_leaves_body_unchanged(
        self, anthropic_adapter: AnthropicAdapter
    ) -> None:
        req = anthropic_adapter.build_request(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
        )
        body = orjson.loads(req.body or b"")
        assert "thinking" not in body
        assert body["temperature"] == 0.5


class TestGeminiThinkingConfig:
    """Gemini ``thinking_budget`` / ``include_thoughts`` translate to ``thinkingConfig``."""

    def test_thinking_budget_with_include_thoughts(self, gemini_adapter: GeminiAdapter) -> None:
        req = gemini_adapter.build_request(
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "Hi"}],
            thinking_budget=1024,
            include_thoughts=True,
        )
        body = orjson.loads(req.body or b"")
        cfg = body["generationConfig"]["thinkingConfig"]
        assert cfg == {"thinkingBudget": 1024, "includeThoughts": True}

    def test_thinking_budget_alone(self, gemini_adapter: GeminiAdapter) -> None:
        req = gemini_adapter.build_request(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
            thinking_budget=512,
        )
        body = orjson.loads(req.body or b"")
        assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 512}
        assert "includeThoughts" not in body["generationConfig"]["thinkingConfig"]

    def test_no_thinking_args_leaves_config_clean(self, gemini_adapter: GeminiAdapter) -> None:
        req = gemini_adapter.build_request(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
        )
        body = orjson.loads(req.body or b"")
        # generationConfig may or may not exist depending on other params; if it does
        # exist, it must not carry thinkingConfig.
        if "generationConfig" in body:
            assert "thinkingConfig" not in body["generationConfig"]


# ---------------------------------------------------------------------------
# Response-side: reasoning_content / thinking_blocks parsing
# ---------------------------------------------------------------------------
#
# Reasoning models expose chain-of-thought differently per family. arcllm
# normalises everything into ``Message.reasoning_content`` (flat str). For
# Anthropic we additionally keep ``Message.thinking_blocks`` so callers can
# replay the structured form (with signatures) on the next turn.


class TestReasoningResponseExtraction:
    """``parse_response`` populates ``reasoning_content`` + ``thinking_blocks``."""

    def test_openai_reasoning_field_is_extracted(self, openai_adapter: OpenAIAdapter) -> None:
        """OpenAI o-series chat/completions can return ``reasoning`` on message."""
        body = orjson.dumps(
            {
                "id": "x",
                "model": "o3-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "42",
                            "reasoning": "Counting Hitchhiker references...",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        resp = openai_adapter.parse_response(body, model="o3-mini")
        msg = resp.choices[0].message
        assert msg.content == "42"
        assert msg.reasoning_content == "Counting Hitchhiker references..."

    def test_deepseek_style_reasoning_content_is_extracted(
        self, openai_adapter: OpenAIAdapter
    ) -> None:
        """DeepSeek-R1 / GLM / Groq DeepSeek / Together / Fireworks DeepSeek-R1
        all use the ``reasoning_content`` field. Test through the OpenAI base
        since every OpenAI-compat host inherits this parser."""
        body = orjson.dumps(
            {
                "id": "x",
                "model": "deepseek-reasoner",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning_content": "Let me think... yes, ok.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        resp = openai_adapter.parse_response(body, model="deepseek-reasoner")
        msg = resp.choices[0].message
        assert msg.content == "ok"
        assert msg.reasoning_content == "Let me think... yes, ok."

    def test_anthropic_thinking_blocks_preserved_with_signature(
        self, anthropic_adapter: AnthropicAdapter
    ) -> None:
        """Anthropic extended-thinking returns structured blocks. The signature
        must round-trip — replaying without it breaks tool-use."""
        body = orjson.dumps(
            {
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-7",
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "User wants ok. Reply ok.",
                        "signature": "sig_abc123",
                    },
                    {"type": "text", "text": "ok"},
                ],
                "usage": {"input_tokens": 5, "output_tokens": 12},
            }
        )
        resp = anthropic_adapter.parse_response(body, model="claude-sonnet-4-7")
        msg = resp.choices[0].message
        assert msg.content == "ok"
        assert msg.reasoning_content == "User wants ok. Reply ok."
        assert msg.thinking_blocks is not None
        assert len(msg.thinking_blocks) == 1
        block = msg.thinking_blocks[0]
        assert block.type == "thinking"
        assert block.thinking == "User wants ok. Reply ok."
        assert block.signature == "sig_abc123"

    def test_anthropic_redacted_thinking_block_preserves_opaque_data(
        self, anthropic_adapter: AnthropicAdapter
    ) -> None:
        """``redacted_thinking`` blocks have no readable text — only an opaque
        payload that must round-trip back unchanged. They surface on
        ``thinking_blocks`` but contribute nothing to ``reasoning_content``."""
        body = orjson.dumps(
            {
                "id": "msg_x",
                "role": "assistant",
                "model": "claude-sonnet-4-7",
                "stop_reason": "end_turn",
                "content": [
                    {"type": "redacted_thinking", "data": "OPAQUE_BLOB"},
                    {"type": "text", "text": "ok"},
                ],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        )
        resp = anthropic_adapter.parse_response(body, model="claude-sonnet-4-7")
        msg = resp.choices[0].message
        assert msg.reasoning_content is None
        assert msg.thinking_blocks is not None
        assert msg.thinking_blocks[0].type == "redacted_thinking"
        assert msg.thinking_blocks[0].data == "OPAQUE_BLOB"

    def test_gemini_thought_parts_route_to_reasoning_content(
        self, gemini_adapter: GeminiAdapter
    ) -> None:
        """Gemini 2.5+ marks chain-of-thought parts with ``thought: true``.

        Without this split, the thought text would land in ``content`` and
        the caller would have to filter it out manually."""
        body = orjson.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "User wants ok.", "thought": True},
                                {"text": "ok"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
            }
        )
        resp = gemini_adapter.parse_response(body, model="gemini-2.5-pro")
        msg = resp.choices[0].message
        assert msg.content == "ok"
        assert msg.reasoning_content == "User wants ok."

    def test_non_reasoning_response_leaves_fields_none(self, openai_adapter: OpenAIAdapter) -> None:
        """Regular chat responses (no reasoning fields) must not invent them."""
        body = orjson.dumps(
            {
                "id": "x",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        resp = openai_adapter.parse_response(body, model="gpt-4o-mini")
        msg = resp.choices[0].message
        assert msg.reasoning_content is None
        assert msg.thinking_blocks is None


class TestReasoningStreamAccumulation:
    """``stream_chunk_builder`` accumulates reasoning across chunks."""

    def test_flat_reasoning_content_accumulates(self) -> None:
        """DeepSeek/GLM/Groq style: reasoning_content arrives in deltas."""
        from arcllm.core import stream_chunk_builder
        from arcllm.types import ChunkChoice, ChunkDelta, StreamChunk

        chunks = [
            StreamChunk(
                id="x",
                model="deepseek-reasoner",
                choices=[ChunkChoice(index=0, delta=ChunkDelta(role="assistant"))],
            ),
            StreamChunk(
                id="x",
                model="deepseek-reasoner",
                choices=[ChunkChoice(index=0, delta=ChunkDelta(reasoning_content="Let me "))],
            ),
            StreamChunk(
                id="x",
                model="deepseek-reasoner",
                choices=[ChunkChoice(index=0, delta=ChunkDelta(reasoning_content="think."))],
            ),
            StreamChunk(
                id="x",
                model="deepseek-reasoner",
                choices=[
                    ChunkChoice(index=0, delta=ChunkDelta(content="ok"), finish_reason="stop")
                ],
            ),
        ]
        final = stream_chunk_builder(chunks)
        msg = final.choices[0].message
        assert msg.content == "ok"
        assert msg.reasoning_content == "Let me think."
        assert msg.thinking_blocks is None

    def test_anthropic_thinking_deltas_grouped_by_signature(self) -> None:
        """Anthropic streaming: thinking_delta → thinking_delta → signature_delta
        is one block. The next thinking_delta opens a new block."""
        from arcllm.core import stream_chunk_builder
        from arcllm.types import ChunkChoice, ChunkDelta, StreamChunk

        chunks = [
            StreamChunk(
                id="x",
                model="claude-sonnet-4-7",
                choices=[ChunkChoice(index=0, delta=ChunkDelta(role="assistant"))],
            ),
            StreamChunk(
                id="x",
                model="claude-sonnet-4-7",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(thinking="User wants ", reasoning_content="User wants "),
                    )
                ],
            ),
            StreamChunk(
                id="x",
                model="claude-sonnet-4-7",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(thinking="ok.", reasoning_content="ok."),
                    )
                ],
            ),
            StreamChunk(
                id="x",
                model="claude-sonnet-4-7",
                choices=[ChunkChoice(index=0, delta=ChunkDelta(signature="sig_abc"))],
            ),
            StreamChunk(
                id="x",
                model="claude-sonnet-4-7",
                choices=[
                    ChunkChoice(index=0, delta=ChunkDelta(content="ok"), finish_reason="stop")
                ],
            ),
        ]
        final = stream_chunk_builder(chunks)
        msg = final.choices[0].message
        assert msg.content == "ok"
        assert msg.reasoning_content == "User wants ok."
        assert msg.thinking_blocks is not None
        assert len(msg.thinking_blocks) == 1
        assert msg.thinking_blocks[0].thinking == "User wants ok."
        assert msg.thinking_blocks[0].signature == "sig_abc"


class TestReasoningSerialization:
    """``Message.model_dump`` round-trips reasoning fields."""

    def test_dump_includes_reasoning_when_set(self) -> None:
        from arcllm.types import Message, ThinkingBlock

        msg = Message(
            role="assistant",
            content="ok",
            reasoning_content="thinking text",
            thinking_blocks=[
                ThinkingBlock(type="thinking", thinking="thinking text", signature="s")
            ],
        )
        dumped = msg.model_dump()
        assert dumped["reasoning_content"] == "thinking text"
        assert dumped["thinking_blocks"] == [
            {"type": "thinking", "thinking": "thinking text", "signature": "s"}
        ]

    def test_dump_omits_reasoning_when_absent(self) -> None:
        """Don't emit empty reasoning fields — keeps the serialised shape lean
        and matches OpenAI/litellm behaviour for non-reasoning responses."""
        from arcllm.types import Message

        msg = Message(role="assistant", content="ok")
        dumped = msg.model_dump()
        assert "reasoning_content" not in dumped
        assert "thinking_blocks" not in dumped
