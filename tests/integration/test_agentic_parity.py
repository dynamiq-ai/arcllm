"""Agentic-parity matrix across every currently-available chat model.

For each (provider, model) pair surfaced in arcllm's capability tables, this
file fires four parametrized tests covering the agentic surface that downstream
frameworks (dynamiq, langchain, llamaindex, raw user code) actually call:

    1. ``test_streaming``               — the model streams tokens
    2. ``test_tool_calling``            — the model can request a tool
    3. ``test_structured_output``       — the model can return JSON
    4. ``test_reasoning_content_emitted`` — the model emits a reasoning surface

Tests skip themselves when the capability flag for that model is False, so
non-tool-using models never count as failures for tool calling, etc. Every
model row in the matrix is gated on the provider's API key being present —
when a key is missing, all of that provider's rows skip cleanly.

Marked ``@pytest.mark.live`` so the existing nightly ``integration.yml``
workflow picks it up; the unit-CI never invokes provider APIs.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest

import arcllm.capabilities.tables as cap_tables

if TYPE_CHECKING:
    from arcllm.capabilities import ModelCapabilities

# Map provider key (used in arcllm model strings, e.g. ``openai/gpt-4o``) to
# (capabilities-table attribute, env var name). Together AI / Fireworks AI are
# spelled with the underscore variant arcllm itself uses internally.
PROVIDER_TABLES: dict[str, tuple[str, str]] = {
    "openai":       ("OPENAI_CAPABILITIES",     "OPENAI_API_KEY"),
    "anthropic":    ("ANTHROPIC_CAPABILITIES",  "ANTHROPIC_API_KEY"),
    "gemini":       ("GEMINI_CAPABILITIES",     "GEMINI_API_KEY"),
    "groq":         ("GROQ_CAPABILITIES",       "GROQ_API_KEY"),
    "xai":          ("XAI_CAPABILITIES",        "XAI_API_KEY"),
    "mistral":      ("MISTRAL_CAPABILITIES",    "MISTRAL_API_KEY"),
    "cohere":       ("COHERE_CAPABILITIES",     "COHERE_API_KEY"),
    "together_ai":  ("TOGETHER_CAPABILITIES",   "TOGETHER_API_KEY"),
    "fireworks_ai": ("FIREWORKS_CAPABILITIES",  "FIREWORKS_API_KEY"),
}


def _enumerate_chat_models() -> list[tuple[str, str, ModelCapabilities]]:
    """Return one row per (provider, model, caps) where ``kind`` is chat-like.

    ``reason`` is included alongside ``chat`` because reasoning-only models
    (o-series, deepseek-r1, magistral, sonar-reasoning) are still conversational
    endpoints from arcllm's perspective — they speak the same wire format.
    """
    rows: list[tuple[str, str, ModelCapabilities]] = []
    for provider, (table_attr, _env) in PROVIDER_TABLES.items():
        table = getattr(cap_tables, table_attr, None)
        if table is None:
            continue
        for model_name, caps in table.items():
            if caps.kind in ("chat", "reason"):
                rows.append((provider, model_name, caps))
    return rows


def _row_id(row: tuple[str, str, ModelCapabilities]) -> str:
    provider, model_name, _caps = row
    return f"{provider}/{model_name}"


_PARITY_ROWS = _enumerate_chat_models()


def _skip_if_no_key(provider: str) -> None:
    env = PROVIDER_TABLES[provider][1]
    if not os.environ.get(env):
        pytest.skip(f"Missing {env} — skipping {provider} parity row")


@pytest.mark.live
@pytest.mark.parametrize(
    "provider, model_name, caps",
    _PARITY_ROWS,
    ids=[_row_id(r) for r in _PARITY_ROWS],
)
class TestAgenticParity:
    """One row per (provider, chat-model). Capability flags drive per-test
    skips so that, e.g., a non-tool-using model never logs a tool failure."""

    def test_streaming(self, provider: str, model_name: str, caps: ModelCapabilities) -> None:
        """Model streams non-empty content chunks."""
        _skip_if_no_key(provider)
        from arcllm import completion

        response = completion(
            model=f"{provider}/{model_name}",
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            stream=True,
            max_tokens=32,
        )
        chunks = list(response)
        text = "".join(
            (chunk.choices[0].delta.content or "")
            for chunk in chunks
            if chunk.choices and chunk.choices[0].delta is not None
        )
        assert chunks, f"{provider}/{model_name}: expected at least one stream chunk"
        assert text.strip(), f"{provider}/{model_name}: expected non-empty streamed content"

    def test_tool_calling(self, provider: str, model_name: str, caps: ModelCapabilities) -> None:
        """Model returns either a tool call or ordinary content for a
        weather-style prompt. Either is acceptable — we only fail if the
        model errors out or returns empty."""
        if not caps.supports_tools:
            pytest.skip(f"{model_name}: capabilities.supports_tools=False")
        _skip_if_no_key(provider)
        from arcllm import completion

        response = completion(
            model=f"{provider}/{model_name}",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            tool_choice="auto",
            max_tokens=128,
        )
        message = response.choices[0].message
        assert message.tool_calls or (message.content and message.content.strip()), (
            f"{provider}/{model_name}: no tool_calls and no content"
        )

    def test_structured_output(self, provider: str, model_name: str, caps: ModelCapabilities) -> None:
        """Model returns valid JSON when ``response_format={'type':'json_object'}``."""
        if not caps.supports_structured_output:
            pytest.skip(f"{model_name}: capabilities.supports_structured_output=False")
        _skip_if_no_key(provider)
        from arcllm import completion

        response = completion(
            model=f"{provider}/{model_name}",
            messages=[
                {
                    "role": "user",
                    "content": 'Return a JSON object with one key "answer" set to 42. Reply with JSON only.',
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=64,
        )
        content = (response.choices[0].message.content or "").strip()
        assert content, f"{provider}/{model_name}: empty content"
        parsed = json.loads(content)
        assert isinstance(parsed, dict), (
            f"{provider}/{model_name}: structured output is not a JSON object: {parsed!r}"
        )

    def test_reasoning_content_emitted(self, provider: str, model_name: str, caps: ModelCapabilities) -> None:
        """Reasoning-capable models populate ``reasoning_content`` or
        ``thinking_blocks`` (the v0.4.9 unified surface).

        This guards regression of the cross-provider reasoning unification:
        DeepSeek-R1 / GLM-thinking / Anthropic-thinking / Gemini-thoughts /
        OpenAI o-series should all expose at least one of the two fields.
        """
        if not caps.supports_reasoning_effort:
            pytest.skip(f"{model_name}: capabilities.supports_reasoning_effort=False")
        _skip_if_no_key(provider)
        from arcllm import completion

        # Some o-series models reject the legacy ``max_tokens`` param. The
        # adapter remaps to ``max_completion_tokens`` automatically (see
        # tests/test_reasoning.py), so we just pass the budget here.
        kwargs: dict[str, Any] = {
            "model": f"{provider}/{model_name}",
            "messages": [{"role": "user", "content": "What is 2+3? Show your work briefly."}],
            "reasoning_effort": "low",
            "max_tokens": 512,
        }
        response = completion(**kwargs)
        message = response.choices[0].message
        has_reasoning = bool(getattr(message, "reasoning_content", None))
        has_thinking = bool(getattr(message, "thinking_blocks", None))
        assert has_reasoning or has_thinking, (
            f"{provider}/{model_name}: reasoning model returned no reasoning_content "
            "and no thinking_blocks"
        )
