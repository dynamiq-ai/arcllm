"""
Comprehensive Integration Tests for Mistral AI Provider.

This module provides end-to-end integration tests for Mistral AI
that can be used in CI/CD pipelines with a live API key.

Requirements:
    - Environment variable: MISTRAL_API_KEY

Usage in CI/CD:
    export MISTRAL_API_KEY="..."
    pytest tests/integration/test_mistral_integration.py -v

API Documentation References:
    - Mistral API Docs: https://docs.mistral.ai/
    - Models: https://docs.mistral.ai/getting-started/models/
    - Chat API: https://docs.mistral.ai/api/#tag/chat
    - Function Calling: https://docs.mistral.ai/capabilities/function_calling/
    - Vision: https://docs.mistral.ai/capabilities/vision/
    - JSON Mode: https://docs.mistral.ai/capabilities/json_mode/
"""

from __future__ import annotations

import json

import pytest

from tests.integration.base import IntegrationTestBase


class TestMistralIntegration(IntegrationTestBase):
    """
    Comprehensive Mistral AI integration tests for CI/CD.

    Mistral uses an OpenAI-compatible API format with some differences
    (e.g., random_seed instead of seed).
    """

    # Provider configuration
    PROVIDER = "mistral"
    ENV_VAR = "MISTRAL_API_KEY"
    PRIMARY_MODEL = "mistral-small-latest"  # Fast and cheap

    # Feature support flags
    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = True  # Via JSON mode
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = False  # Not tested
    SUPPORTS_VISION = True  # Via Pixtral models

    # CI/CD timing configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    TIMEOUT = 60.0

    # =========================================================================
    # SMOKE TESTS - Quick validation tests for CI/CD
    # =========================================================================

    @pytest.mark.smoke
    def test_basic_completion_smoke(self) -> None:
        """
        [SMOKE] Basic completion test - validates API connectivity.
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
            max_tokens=5,
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    @pytest.mark.smoke
    def test_streaming_smoke(self) -> None:
        """
        [SMOKE] Streaming test - validates SSE streaming works.
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
        assert len(chunks) > 0

    # =========================================================================
    # CORE FUNCTIONALITY TESTS
    # =========================================================================

    def test_system_prompt(self) -> None:
        """Test system prompt handling."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {"role": "system", "content": "Always respond with exactly one word."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            max_tokens=20,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_multi_turn_conversation(self) -> None:
        """Test multi-turn conversation handling."""
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
        assert response.choices[0].message.content is not None

    def test_temperature_parameter(self) -> None:
        """Test temperature parameter."""
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

    # =========================================================================
    # TOOL CALLING TESTS
    # =========================================================================

    def test_tool_calling_basic(self) -> None:
        """Test basic tool/function calling."""
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
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/mistral-large-latest",  # Large model for tools
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=150,
        )

        assert response is not None
        choice = response.choices[0]

        has_tool_calls = (
            choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0
        )
        has_content = choice.message.content is not None

        assert has_tool_calls or has_content

        if has_tool_calls:
            tool_call = choice.message.tool_calls[0]
            assert tool_call.function.name == "get_weather"
            args = json.loads(tool_call.function.arguments)
            assert isinstance(args, dict)

    # =========================================================================
    # STREAMING TESTS
    # =========================================================================

    def test_streaming_content_accumulation(self) -> None:
        """Test streaming content accumulation."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=50,
            stream=True,
        )

        content_parts = []
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    def test_stream_chunk_builder(self) -> None:
        """Test stream_chunk_builder produces complete response."""
        from arcllm import completion, stream_chunk_builder

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            stream=True,
        )

        chunks = list(response)
        final = stream_chunk_builder(chunks)

        assert final.choices is not None
        assert final.choices[0].message.content is not None

    # =========================================================================
    # JSON MODE TESTS
    # =========================================================================

    def test_json_mode(self) -> None:
        """Test JSON mode (structured output)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/mistral-large-latest",
            messages=[{"role": "user", "content": 'Return JSON: {"status": "ok"}'}],
            response_format={"type": "json_object"},
            max_tokens=50,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None

        # Should be valid JSON
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    # =========================================================================
    # ASYNC TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_async_completion(self) -> None:
        """Test async chat completion."""
        from arcllm import acompletion

        response = await self.retry_on_rate_limit_async(
            acompletion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'async'."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    @pytest.mark.asyncio
    async def test_async_streaming(self) -> None:
        """Test async streaming completion."""
        from arcllm import acompletion

        stream = await self.retry_on_rate_limit_async(
            acompletion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Count 1 2 3."}],
            max_tokens=20,
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0

    # =========================================================================
    # USAGE AND PRICING TESTS
    # =========================================================================

    def test_usage_reporting(self) -> None:
        """Test usage information is reported."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    def test_pricing_calculation(self) -> None:
        """Test cost calculation."""
        from arcllm import completion
        from arcllm.pricing.tables import completion_cost

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        cost = completion_cost(response)
        assert cost >= 0

    # =========================================================================
    # MODEL-SPECIFIC TESTS
    # =========================================================================

    def test_mistral_large(self) -> None:
        """Test Mistral Large model."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/mistral-large-latest",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_pixtral_12b(self) -> None:
        """Test Pixtral 12B model with text."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/pixtral-12b-2409",
            messages=[{"role": "user", "content": "Say test."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_codestral(self) -> None:
        """Test Codestral model."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/codestral-latest",
            messages=[{"role": "user", "content": "Write a hello world in Python."}],
            max_tokens=50,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_ministral_3b(self) -> None:
        """Test Ministral 3B (smallest model)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/ministral-3b-latest",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    # =========================================================================
    # VISION TESTS
    # =========================================================================

    def test_pixtral_vision_base64(self) -> None:
        """Test Pixtral vision with base64 image."""
        from arcllm import completion

        # Simple 1x1 blue pixel PNG
        blue_pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPj/HwADBwIAMCbHYQAAAABJRU5ErkJggg=="

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/pixtral-12b-2409",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{blue_pixel}"},
                        },
                    ],
                }
            ],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    # =========================================================================
    # ERROR HANDLING TESTS
    # =========================================================================

    def test_invalid_model_error(self) -> None:
        """Test invalid model raises error."""
        from arcllm import completion
        from arcllm.exceptions import ArcLLMError

        with pytest.raises(ArcLLMError):
            completion(
                model=f"{self.PROVIDER}/nonexistent-model-xyz123",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
