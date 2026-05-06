"""Unit tests for the Bedrock adapter's per-family response parsing.

Bedrock returns a different response shape per model family. The adapter
dispatches on ``_get_model_family`` (which looks at the model id prefix:
``meta.``, ``amazon.``, ``cohere.``, ``mistral.``, ``ai21.``) and extracts
content, finish reason, and (where available) provider-reported usage.

These tests exercise the static ``_extract_generic_body`` helper directly so
we don't need AWS credentials. The Anthropic-on-Bedrock path is exercised
elsewhere (``test_anthropic.py`` covers the Messages API surface).
"""

from __future__ import annotations

import pytest

from arcllm.providers.base import ProviderConfig
from arcllm.providers.bedrock_adapter import BedrockAdapter


@pytest.fixture
def adapter() -> BedrockAdapter:
    return BedrockAdapter(ProviderConfig(aws_region="us-east-1"))


class TestModelFamilyDetection:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("anthropic.claude-sonnet-4-5-20250929-v1:0", "anthropic"),
            ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic"),
            ("meta.llama3-3-70b-instruct-v1:0", "meta"),
            ("amazon.nova-pro-v1:0", "amazon"),
            ("amazon.titan-text-express-v1", "amazon"),
            ("cohere.command-r-plus-v1:0", "cohere"),
            ("mistral.mistral-large-2407-v1:0", "mistral"),
            ("ai21.j2-ultra-v1", "ai21"),
        ],
    )
    def test_family_dispatch(self, adapter: BedrockAdapter, model: str, expected: str) -> None:
        assert adapter._get_model_family(model) == expected


class TestMetaLlamaResponse:
    def test_extracts_content_and_usage(self) -> None:
        resp = {
            "generation": "Hello there!",
            "stop_reason": "stop",
            "prompt_token_count": 12,
            "generation_token_count": 4,
        }
        content, finish, usage = BedrockAdapter._extract_generic_body("meta", resp)
        assert content == "Hello there!"
        assert finish == "stop"
        assert usage is not None
        assert usage.prompt_tokens == 12
        assert usage.completion_tokens == 4
        assert usage.total_tokens == 16

    def test_missing_usage_returns_none(self) -> None:
        resp = {"generation": "ok", "stop_reason": "length"}
        content, finish, usage = BedrockAdapter._extract_generic_body("meta", resp)
        assert content == "ok"
        assert finish == "length"
        assert usage is None


class TestAmazonNovaResponse:
    def test_extracts_nested_content_and_usage(self) -> None:
        resp = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "The answer is 42."}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 8, "outputTokens": 5, "totalTokens": 13},
        }
        content, finish, usage = BedrockAdapter._extract_generic_body("amazon", resp)
        assert content == "The answer is 42."
        assert finish == "end_turn"
        assert usage is not None
        assert usage.prompt_tokens == 8
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 13


class TestAmazonTitanTextResponse:
    def test_extracts_results_array_and_usage(self) -> None:
        resp = {
            "inputTextTokenCount": 9,
            "results": [{"outputText": "OK.", "tokenCount": 2, "completionReason": "FINISH"}],
        }
        content, finish, usage = BedrockAdapter._extract_generic_body("amazon", resp)
        assert content == "OK."
        assert finish == "FINISH"
        assert usage is not None
        assert usage.prompt_tokens == 9
        assert usage.completion_tokens == 2
        assert usage.total_tokens == 11


class TestMistralResponse:
    def test_extracts_outputs_array(self) -> None:
        resp = {
            "outputs": [{"text": "bonjour", "stop_reason": "stop"}],
        }
        content, finish, usage = BedrockAdapter._extract_generic_body("mistral", resp)
        assert content == "bonjour"
        assert finish == "stop"
        assert usage is None  # Mistral on Bedrock doesn't ship usage tokens


class TestCohereResponse:
    def test_extracts_generations_array(self) -> None:
        resp = {
            "generations": [{"text": "hi", "finish_reason": "COMPLETE"}],
        }
        content, finish, usage = BedrockAdapter._extract_generic_body("cohere", resp)
        assert content == "hi"
        assert finish == "COMPLETE"
        assert usage is None


class TestAi21Response:
    def test_extracts_completions_array(self) -> None:
        resp = {
            "completions": [
                {
                    "data": {"text": "Sure."},
                    "finishReason": {"reason": "endoftext"},
                }
            ]
        }
        content, finish, usage = BedrockAdapter._extract_generic_body("ai21", resp)
        assert content == "Sure."
        assert finish == "endoftext"
        assert usage is None


class TestUnknownFamilyFallback:
    def test_falls_back_to_historical_field_names(self) -> None:
        resp = {"completion": "yo"}
        content, finish, usage = BedrockAdapter._extract_generic_body("foo", resp)
        assert content == "yo"
        assert finish is None
        assert usage is None


class TestBedrockOpenAIFamily:
    """Bedrock now hosts ``openai.gpt-oss-*`` with the OpenAI Chat Completions
    wire format. We dispatch the body builder by family."""

    def test_openai_family_detection(self) -> None:
        adapter = BedrockAdapter(ProviderConfig(aws_region="us-east-1"))
        assert adapter._get_model_family("openai.gpt-oss-120b-1:0") == "openai"
        assert adapter._get_model_family("openai.gpt-oss-20b-1:0") == "openai"

    def test_openai_body_uses_chat_completions_shape(self) -> None:
        adapter = BedrockAdapter(ProviderConfig(aws_region="us-east-1"))
        body = adapter._build_openai_body(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            max_tokens=10,
            temperature=0.5,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {}},
                }
            ],
            tool_choice="auto",
        )
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["max_tokens"] == 10
        assert body["temperature"] == 0.5
        assert body["tools"][0]["type"] == "function"
        assert body["tool_choice"] == "auto"


class TestBedrockAnthropicToolsAndResponseFormat:
    """Anthropic-on-Bedrock honours tool_choice (auto/required/specific) and
    converts ``response_format=json_schema`` into a forced tool call."""

    def test_tool_choice_auto(self) -> None:
        adapter = BedrockAdapter(ProviderConfig(aws_region="us-east-1"))
        body = adapter._build_anthropic_body(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice="auto",
        )
        assert body["tool_choice"] == {"type": "auto"}

    def test_tool_choice_required(self) -> None:
        adapter = BedrockAdapter(ProviderConfig(aws_region="us-east-1"))
        body = adapter._build_anthropic_body(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice="required",
        )
        assert body["tool_choice"] == {"type": "any"}

    def test_response_format_json_schema_emits_forced_tool(self) -> None:
        adapter = BedrockAdapter(ProviderConfig(aws_region="us-east-1"))
        body = adapter._build_anthropic_body(
            messages=[{"role": "user", "content": "give me a user"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "user",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
            },
        )
        # A schema-shaped tool got injected.
        assert any(t.get("name") == "user" for t in body["tools"])
        assert body["tool_choice"] == {"type": "tool", "name": "user"}

    def test_thinking_budget_emits_thinking_block_and_drops_temperature(self) -> None:
        adapter = BedrockAdapter(ProviderConfig(aws_region="us-east-1"))
        body = adapter._build_anthropic_body(
            messages=[{"role": "user", "content": "solve"}],
            thinking_budget=2048,
            temperature=0.7,
        )
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "temperature" not in body
