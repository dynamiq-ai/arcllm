"""
Comprehensive Integration Tests for Anthropic Provider.

This module provides end-to-end integration tests for the Anthropic Claude API
that can be used in CI/CD pipelines with a live API key.

Requirements:
    - Environment variable: ANTHROPIC_API_KEY

Usage in CI/CD:
    # Run all Anthropic integration tests
    export ANTHROPIC_API_KEY="sk-ant-..."
    pytest tests/integration/test_anthropic_integration.py -v

    # Run with specific markers
    pytest tests/integration/test_anthropic_integration.py -v -m "not slow"

    # Run quick smoke tests only
    pytest tests/integration/test_anthropic_integration.py -v -m smoke

API Documentation References:
    - Messages API: https://docs.anthropic.com/en/api/messages
    - Streaming: https://docs.anthropic.com/en/api/messages-streaming
    - Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
    - Vision: https://docs.anthropic.com/en/docs/build-with-claude/vision
    - Models: https://docs.anthropic.com/en/docs/about-claude/models
    - Errors: https://docs.anthropic.com/en/api/errors
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.integration.base import IntegrationTestBase


class TestAnthropicIntegration(IntegrationTestBase):
    """
    Comprehensive Anthropic integration tests for CI/CD.

    These tests verify end-to-end functionality against the live Anthropic API.
    All tests include retry logic for rate limiting and proper error handling.
    """

    # Provider configuration
    PROVIDER = "anthropic"
    ENV_VAR = "ANTHROPIC_API_KEY"
    PRIMARY_MODEL = "claude-3-5-haiku-20241022"  # Fast & affordable for CI/CD

    # Feature support flags
    # NOTE: Anthropic does NOT support native JSON mode via response_format parameter.
    # See: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = False  # No native response_format support
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = False  # Anthropic doesn't offer embeddings API
    SUPPORTS_VISION = True

    # CI/CD timing configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    TIMEOUT = 60.0

    # =========================================================================
    # SMOKE TESTS - Quick validation tests for CI/CD
    # Run with: pytest -m smoke
    # =========================================================================

    @pytest.mark.smoke
    def test_basic_completion_smoke(self) -> None:
        """
        [SMOKE] Basic completion test - validates API connectivity and response format.

        This is the minimum test to verify the Anthropic integration is working.
        Should complete in < 5 seconds.
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
            max_tokens=5,
        )

        assert response is not None, "Response should not be None"
        assert response.id, "Response should have an ID"
        assert response.model, "Response should have a model"
        assert response.choices, "Response should have choices"
        assert len(response.choices) > 0, "Should have at least one choice"
        assert response.choices[0].message, "Choice should have a message"
        assert response.choices[0].message.content, "Message should have content"
        assert response.usage, "Response should have usage info"
        assert response.usage.prompt_tokens > 0, "Should have prompt tokens"
        assert response.usage.completion_tokens > 0, "Should have completion tokens"

    @pytest.mark.smoke
    def test_streaming_smoke(self) -> None:
        """
        [SMOKE] Streaming test - validates SSE streaming works.

        Should complete in < 10 seconds.
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'test'."}],
            max_tokens=5,
            stream=True,
        )

        chunks = list(response)
        assert len(chunks) > 0, "Should receive at least one chunk"

        # Verify chunk structure
        has_content = False
        for chunk in chunks:
            assert chunk.choices, "Chunk should have choices"
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                has_content = True

        assert has_content, "Should have received content in stream"

    # =========================================================================
    # CORE FUNCTIONALITY TESTS
    # =========================================================================

    def test_system_prompt(self) -> None:
        """
        Test system prompt handling (Anthropic-specific).

        Anthropic extracts system messages to a separate 'system' parameter.
        See: https://docs.anthropic.com/en/api/messages#body-messages
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {"role": "system", "content": "You are a pirate. Always say 'Arrr' first."},
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=50,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        assert len(content) > 0

    def test_multiple_system_messages(self) -> None:
        """
        Test multiple system messages are concatenated.

        arcllm concatenates multiple system messages for Anthropic.
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            max_tokens=20,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None

    def test_multi_turn_conversation(self) -> None:
        """
        Test multi-turn conversation handling.

        Verifies the adapter properly handles conversation history.
        """
        from arcllm import completion

        messages = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And if I add 3 more?"},
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=messages,
            max_tokens=20,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        # Model should reference context (answer should be 7)

    def test_temperature_parameter(self) -> None:
        """
        Test temperature parameter is properly passed.

        See: https://docs.anthropic.com/en/api/messages#body-messages
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say a random word."}],
            temperature=0.9,
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_stop_sequences(self) -> None:
        """
        Test stop sequences parameter.

        Anthropic uses 'stop_sequences' instead of OpenAI's 'stop'.
        See: https://docs.anthropic.com/en/api/messages#body-messages
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Count from 1 to 10."}],
            stop=["5"],  # Should stop at or before 5
            max_tokens=50,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        # Should stop before getting to "5" in the sequence

    # =========================================================================
    # TOOL CALLING TESTS
    # See: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
    # =========================================================================

    def test_tool_calling_basic(self) -> None:
        """
        Test basic tool/function calling capability.

        Verifies tools are converted to Anthropic format and responses parsed correctly.
        """
        from arcllm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=150,
        )

        assert response is not None
        choice = response.choices[0]

        # Model should call the tool
        has_tool_calls = (
            choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0
        )
        has_content = choice.message.content is not None

        assert has_tool_calls or has_content, "Expected tool calls or content"

        if has_tool_calls:
            tool_call = choice.message.tool_calls[0]
            assert tool_call.id, "Tool call should have an ID"
            assert tool_call.type == "function", "Tool call type should be 'function'"
            assert tool_call.function is not None
            assert tool_call.function.name == "get_weather"

            # Arguments should be valid JSON
            args = json.loads(tool_call.function.arguments)
            assert isinstance(args, dict)
            assert "location" in args

    def test_tool_calling_multiple_tools(self) -> None:
        """
        Test multiple tools can be defined.

        Model should be able to choose from multiple available tools.
        """
        from arcllm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time in a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                        "required": ["timezone"],
                    },
                },
            },
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=tools,
            max_tokens=150,
        )

        assert response is not None
        # Should get a response (either tool call or text)
        choice = response.choices[0]
        assert choice.message is not None

    def test_tool_result_handling(self) -> None:
        """
        Test tool result message handling.

        Verifies the adapter properly converts tool results for Anthropic.
        Anthropic requires tool results in user messages.
        See: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#tool-use-and-tool-result-content-blocks
        """
        from arcllm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        # Simulate a conversation with tool use
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_test123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_test123",
                "content": '{"temperature": 22, "conditions": "sunny"}',
            },
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=messages,
            tools=tools,
            max_tokens=100,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        # Model should respond about the weather

    def test_streaming_tool_calls(self) -> None:
        """
        Test streaming with tool calls.

        Verifies tool call chunks are properly accumulated.
        See: https://docs.anthropic.com/en/api/messages-streaming
        """
        from arcllm import completion, stream_chunk_builder

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "What time is it?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=100,
            stream=True,
        )

        chunks = list(response)
        assert len(chunks) > 0, "Should receive chunks"

        # Build final response
        final = stream_chunk_builder(chunks)
        assert final is not None
        assert final.choices[0].message is not None

    # =========================================================================
    # VISION TESTS
    # See: https://docs.anthropic.com/en/docs/build-with-claude/vision
    # =========================================================================

    def test_vision_base64_image(self) -> None:
        """
        Test vision capability with base64 encoded image.

        Verifies multimodal content conversion to Anthropic format.
        """
        from arcllm import completion

        # 1x1 red pixel PNG (minimal valid image)
        red_pixel_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
            "DUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Answer in one word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{red_pixel_base64}"},
                        },
                    ],
                }
            ],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_multimodal_message(self) -> None:
        """
        Test multimodal (vision) capability with text and image.

        Same as test_vision_base64_image but with different naming for base class compatibility.
        """
        self.test_vision_base64_image()

    # =========================================================================
    # STREAMING TESTS
    # See: https://docs.anthropic.com/en/api/messages-streaming
    # =========================================================================

    def test_streaming_content_accumulation(self) -> None:
        """
        Test that streaming content can be accumulated correctly.

        Verifies all content_block_delta events are properly parsed.
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=50,
            stream=True,
        )

        content_parts = []
        finish_reason = None

        for chunk in response:
            if chunk.choices:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_parts.append(chunk.choices[0].delta.content)
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

        full_content = "".join(content_parts)
        assert len(full_content) > 0, "Should have accumulated content"
        assert finish_reason == "stop", f"Expected 'stop' finish reason, got {finish_reason}"

    def test_stream_chunk_builder_complete(self) -> None:
        """
        Test stream_chunk_builder produces complete ModelResponse.

        Verifies the built response has all expected fields.
        """
        from arcllm import completion, stream_chunk_builder

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'hello'."}],
            max_tokens=10,
            stream=True,
        )

        chunks = list(response)
        final = stream_chunk_builder(chunks)

        assert final.id, "Should have response ID"
        assert final.model, "Should have model"
        assert final.choices, "Should have choices"
        assert final.choices[0].message, "Should have message"
        assert final.choices[0].message.content, "Should have content"
        assert final.choices[0].message.role == "assistant", "Role should be assistant"

    # =========================================================================
    # ASYNC TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_async_completion(self) -> None:
        """
        Test async chat completion.

        Verifies async API works correctly with Anthropic.
        """
        from arcllm import acompletion

        response = await self.retry_on_rate_limit_async(
            acompletion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'async' and nothing else."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @pytest.mark.asyncio
    async def test_async_streaming(self) -> None:
        """
        Test async streaming completion.

        Verifies async streaming works with Anthropic's SSE format.
        """
        from arcllm import acompletion

        stream = await self.retry_on_rate_limit_async(
            acompletion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Count 1 2 3."}],
            max_tokens=20,
            stream=True,
        )

        chunks = []
        content_parts = []
        async for chunk in stream:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0, "Should receive chunks"
        assert len(content_parts) > 0, "Should receive content"

    @pytest.mark.asyncio
    async def test_async_tool_calling(self) -> None:
        """
        Test async tool calling.

        Verifies async API properly handles tool calls.
        """
        from arcllm import acompletion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                },
            }
        ]

        response = await self.retry_on_rate_limit_async(
            acompletion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Calculate 5 * 5"}],
            tools=tools,
            max_tokens=100,
        )

        assert response is not None
        choice = response.choices[0]
        assert choice.message is not None
        # Should have either tool call or text response

    # =========================================================================
    # USAGE AND PRICING TESTS
    # =========================================================================

    def test_usage_reporting(self) -> None:
        """
        Test that usage information is properly reported.

        Anthropic returns input_tokens and output_tokens.
        See: https://docs.anthropic.com/en/api/messages#response-body
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        assert response is not None
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0, "Should have prompt tokens"
        assert response.usage.completion_tokens > 0, "Should have completion tokens"
        assert response.usage.total_tokens > 0, "Should have total tokens"
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        ), "Total should equal prompt + completion"

    def test_pricing_calculation(self) -> None:
        """
        Test cost calculation for Anthropic models.

        Verifies pricing tables are correctly configured.
        """
        from arcllm import completion
        from arcllm.pricing.tables import completion_cost

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        cost = completion_cost(response)
        assert cost >= 0, "Cost should be non-negative"
        assert cost < 1.0, "Cost should be reasonable for a simple request"

    def test_model_extra_contains_usage(self) -> None:
        """
        Test that model_extra dict contains usage data.

        This is for compatibility with LiteLLM-style usage access.
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        assert response.model_extra is not None
        assert "usage" in response.model_extra
        usage = response.model_extra["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    # =========================================================================
    # ERROR HANDLING TESTS
    # See: https://docs.anthropic.com/en/api/errors
    # =========================================================================

    def test_invalid_model_error(self) -> None:
        """
        Test that invalid model raises appropriate error.

        Should raise UnsupportedModelError with 404 status.
        """
        from arcllm import completion
        from arcllm.exceptions import ArcLLMError

        with pytest.raises(ArcLLMError) as exc_info:
            completion(
                model=f"{self.PROVIDER}/nonexistent-model-xyz123",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )

        error = exc_info.value
        assert error.provider == "anthropic"

    def test_invalid_api_key_error(self) -> None:
        """
        Test that invalid API key raises AuthenticationError.
        """
        from arcllm import completion
        from arcllm.exceptions import AuthenticationError

        # Temporarily use invalid key
        with pytest.raises(AuthenticationError) as exc_info:
            completion(
                model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="sk-ant-invalid-key-12345",
                max_tokens=5,
            )

        error = exc_info.value
        assert error.provider == "anthropic"
        assert error.status_code == 401

    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================

    def test_empty_assistant_content(self) -> None:
        """
        Test handling of empty assistant content with tool calls.

        When model uses tools, content may be None or empty.
        """
        from arcllm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_data",
                    "description": "Get some data",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        # Force tool use
        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Call the get_data function now."}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_data"}},
            max_tokens=100,
        )

        assert response is not None
        # Content may be None when tool is called
        choice = response.choices[0]
        assert choice.message is not None

    def test_long_context(self) -> None:
        """
        Test handling of longer context (still within limits).
        """
        from arcllm import completion

        # Create a moderately long message
        long_content = "This is a test sentence. " * 50  # ~300 words

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {"role": "system", "content": "Summarize the user message briefly."},
                {"role": "user", "content": long_content},
            ],
            max_tokens=50,
        )

        assert response is not None
        assert response.choices[0].message.content is not None


# =============================================================================
# CI/CD MARKER CONFIGURATION
# =============================================================================
# Usage:
#   pytest -m smoke        # Quick smoke tests only
#   pytest -m "not slow"   # Skip slow tests
#   pytest -m integration  # All integration tests
# =============================================================================


# Register custom markers
def pytest_configure(config: Any) -> None:
    """Register custom markers for CI/CD filtering."""
    config.addinivalue_line("markers", "smoke: Quick smoke tests for CI/CD")
    config.addinivalue_line("markers", "slow: Slow tests that may be skipped in CI")
    config.addinivalue_line("markers", "integration: All integration tests")
