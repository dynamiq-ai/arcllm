"""Built-in / server-side provider tools must pass through unchanged.

Modern providers ship native tool types that the model executes server-side
(web search, code execution, file search, etc.). These look different from
OpenAI's ``{"type": "function"}`` shape and arcllm doesn't translate them —
it forwards them verbatim so callers can opt in.

Each test asserts that a provider-native tool block survives ``build_request``
and lands in the wire body unchanged.
"""

from __future__ import annotations

import json

import pytest

from arcllm.providers.anthropic_adapter import AnthropicAdapter
from arcllm.providers.base import ProviderConfig
from arcllm.providers.gemini_adapter import GeminiAdapter
from arcllm.providers.openai_adapter import OpenAIAdapter
from arcllm.providers.perplexity_adapter import PerplexityAdapter

# ---------------------------------------------------------------------------
# OpenAI: web_search / file_search / code_interpreter pass through unchanged.
# ---------------------------------------------------------------------------


@pytest.fixture
def openai() -> OpenAIAdapter:
    return OpenAIAdapter(ProviderConfig(api_key="test"))


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "web_search"},
        {"type": "file_search"},
        {"type": "code_interpreter"},
    ],
)
def test_openai_native_tool_pass_through(openai: OpenAIAdapter, tool: dict) -> None:
    req = openai.build_request(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "search for x"}],
        tools=[tool],
    )
    body = json.loads(req.body or b"")
    assert body["tools"] == [tool]


# ---------------------------------------------------------------------------
# Anthropic: web_search_20250305, code_execution_20250825, etc. pass through
# unchanged. Custom function tools are still converted alongside.
# ---------------------------------------------------------------------------


@pytest.fixture
def anthropic() -> AnthropicAdapter:
    return AnthropicAdapter(ProviderConfig(api_key="test"))


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
        {"type": "code_execution_20250825", "name": "code_execution"},
        {"type": "text_editor_20250728", "name": "str_replace_editor"},
        {
            "type": "computer_use_20250124",
            "name": "computer",
            "display_width": 1024,
            "display_height": 768,
        },
    ],
)
def test_anthropic_native_tool_pass_through(anthropic: AnthropicAdapter, tool: dict) -> None:
    req = anthropic.build_request(
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "research this"}],
        max_tokens=128,
        tools=[tool],
    )
    body = json.loads(req.body or b"")
    assert tool in body["tools"]


def test_anthropic_mixes_native_and_function_tools(anthropic: AnthropicAdapter) -> None:
    """A request can mix server-side and user-defined tools."""
    fn_tool = {
        "type": "function",
        "function": {
            "name": "lookup_user",
            "description": "Look up a user",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
        },
    }
    native_tool = {"type": "web_search_20250305", "name": "web_search"}
    req = anthropic.build_request(
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "find user 42"}],
        max_tokens=128,
        tools=[fn_tool, native_tool],
    )
    body = json.loads(req.body or b"")
    assert len(body["tools"]) == 2
    # Function tool was converted to Anthropic's {name, description, input_schema} shape.
    function_entry = next(t for t in body["tools"] if "input_schema" in t)
    assert function_entry["name"] == "lookup_user"
    # Native tool passed through verbatim.
    assert native_tool in body["tools"]


# ---------------------------------------------------------------------------
# Gemini: google_search, code_execution, url_context, etc. pass through.
# ---------------------------------------------------------------------------


@pytest.fixture
def gemini() -> GeminiAdapter:
    return GeminiAdapter(config=ProviderConfig(api_key="test"))


@pytest.mark.parametrize(
    "tool",
    [
        {"google_search": {}},
        {"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "MODE_DYNAMIC"}}},
        {"code_execution": {}},
        {"url_context": {}},
    ],
)
def test_gemini_native_tool_pass_through(gemini: GeminiAdapter, tool: dict) -> None:
    req = gemini.build_request(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "search"}],
        tools=[tool],
    )
    body = json.loads(req.body or b"")
    # The native tool block appears in the tools array verbatim. (Gemini may
    # also include a `functionDeclarations` block alongside if function tools
    # were mixed in; this test passes a single native tool so we expect the
    # tool we sent.)
    assert tool in body["tools"]


def test_gemini_mixes_native_and_function_tools(gemini: GeminiAdapter) -> None:
    fn_tool = {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    native = {"google_search": {}}
    req = gemini.build_request(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "time?"}],
        tools=[fn_tool, native],
    )
    body = json.loads(req.body or b"")
    assert any("functionDeclarations" in t for t in body["tools"])
    assert native in body["tools"]


# ---------------------------------------------------------------------------
# Perplexity: search is implicit (no tools call needed). Confirm the search
# domain filter and similar request kwargs survive into the wire body.
# ---------------------------------------------------------------------------


def test_perplexity_search_kwargs_pass_through() -> None:
    adapter = PerplexityAdapter(ProviderConfig(api_key="test"))
    req = adapter.build_request(
        model="sonar-pro",
        messages=[{"role": "user", "content": "research this"}],
        search_domain_filter=["nytimes.com", "-reddit.com"],
        search_recency_filter="month",
        return_related_questions=True,
    )
    body = json.loads(req.body or b"")
    assert body["search_domain_filter"] == ["nytimes.com", "-reddit.com"]
    assert body["search_recency_filter"] == "month"
    assert body["return_related_questions"] is True
