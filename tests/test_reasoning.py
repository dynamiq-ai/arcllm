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
