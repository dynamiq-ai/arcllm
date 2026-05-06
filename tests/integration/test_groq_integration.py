"""
Comprehensive Integration Tests for Groq Provider.

This module provides end-to-end integration tests for Groq
that can be used in CI/CD pipelines with a live API key.

Requirements:
    - Environment variable: GROQ_API_KEY

Usage in CI/CD:
    export GROQ_API_KEY="gsk_..."
    pytest tests/integration/test_groq_integration.py -v

API Documentation References:
    - Groq Docs: https://console.groq.com/docs
    - Models: https://console.groq.com/docs/models
    - Chat API: https://console.groq.com/docs/api-reference#chat-create
    - Rate Limits: https://console.groq.com/docs/rate-limits

Available Models (January 2026):
    - meta-llama/llama-4-maverick-17b-128e-instruct (Vision, tools)
    - meta-llama/llama-4-scout-17b-16e-instruct (Vision, tools)
    - openai/gpt-oss-120b (Strong reasoning)
    - openai/gpt-oss-20b (Fast, tool use)
    - moonshotai/kimi-k2-instruct (256K context)
    - qwen/qwen3-32b (General purpose)
    - groq/compound (Agentic with built-in tools)
    - llama-3.3-70b-versatile (Legacy flagship)
    - llama-3.1-8b-instant (Fast inference)
"""

from __future__ import annotations

import json
import os

import pytest

from tests.integration.base import IntegrationTestBase

# =============================================================================
# Configuration
# =============================================================================


def get_test_models() -> list[str]:
    """Get list of Groq models to test."""
    env_models = os.environ.get("GROQ_TEST_MODELS", "")
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]

    # Default: Test all major model families for comprehensive coverage
    return [
        "llama-3.3-70b-versatile",  # Legacy flagship (stable)
        "llama-3.1-8b-instant",  # Fast inference
        "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 (latest)
        "openai/gpt-oss-20b",  # OpenAI open-weight
        "qwen/qwen3-32b",  # Qwen 3
        "moonshotai/kimi-k2-instruct",  # Kimi K2
    ]


# Mark tests that require network access
pytestmark = [
    pytest.mark.integration,
    pytest.mark.groq,
]


# =============================================================================
# Main Test Class
# =============================================================================


class TestGroqIntegration(IntegrationTestBase):
    """
    Comprehensive Groq integration tests for CI/CD.

    Groq uses an OpenAI-compatible API format with ultra-fast inference
    powered by their custom LPU (Language Processing Unit) hardware.

    Key Features:
        - Ultra-low latency inference
        - OpenAI-compatible API
        - Wide model selection (Llama, GPT-OSS, Qwen, etc.)
        - Built-in tools with Compound models
    """

    # Provider configuration
    PROVIDER = "groq"
    ENV_VAR = "GROQ_API_KEY"
    # Use llama-3.3-70b-versatile as primary (stable, widely supported)
    PRIMARY_MODEL = "llama-3.3-70b-versatile"

    # Feature support flags
    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = True  # Via JSON mode
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = False  # Groq doesn't provide embeddings
    SUPPORTS_VISION = True  # Llama 4 models support vision

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

        Should complete in < 2 seconds (Groq is very fast).
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
        content = response.choices[0].message.content
        assert content is not None

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

    def test_stop_sequences(self) -> None:
        """Test stop sequences parameter."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Count: 1, 2, 3, 4, 5"}],
            stop=[","],
            max_tokens=50,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None

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
                        "properties": {"location": {"type": "string", "description": "City name"}},
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

    def test_tool_calling_multiple_tools(self) -> None:
        """Test multiple tools available."""
        from arcllm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "The city name"}},
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time for a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string", "description": "Timezone"}},
                        "required": ["timezone"],
                    },
                },
            },
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": "Use the get_weather function to check weather in Paris.",
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=150,
        )

        assert response is not None
        # Model should either call a tool or provide a response
        choice = response.choices[0]
        assert choice.message.tool_calls is not None or choice.message.content is not None

    # =========================================================================
    # JSON MODE TESTS
    # =========================================================================

    def test_json_mode(self) -> None:
        """Test JSON response format."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": "Return a JSON object with 'name' and 'age' fields. "
                    "Use name='Alice' and age=30.",
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None

        # Parse JSON
        data = json.loads(content)
        assert "name" in data or "age" in data

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
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)

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

    def test_llama_3_1_8b_instant(self) -> None:
        """Test Llama 3.1 8B instant model (fastest)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say test."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_llama_4_scout(self) -> None:
        """Test Llama 4 Scout model (latest)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_llama_4_maverick(self) -> None:
        """Test Llama 4 Maverick model (multimodal)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_gpt_oss_20b(self) -> None:
        """Test OpenAI GPT-OSS 20B model."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "Say test."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_gpt_oss_120b(self) -> None:
        """Test OpenAI GPT-OSS 120B model (strong reasoning)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=20,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_qwen3_32b(self) -> None:
        """Test Qwen 3 32B model."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/qwen/qwen3-32b",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_kimi_k2(self) -> None:
        """Test Moonshot Kimi K2 model."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/moonshotai/kimi-k2-instruct",
            messages=[{"role": "user", "content": "Say test."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_compound_model(self) -> None:
        """Test Groq Compound model (agentic with built-in tools)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/groq/compound",
            messages=[{"role": "user", "content": "What is 15 * 7?"}],
            max_tokens=50,
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

    def test_invalid_api_key_error(self) -> None:
        """Test invalid API key raises error."""
        import os

        from arcllm import completion
        from arcllm.exceptions import AuthenticationError

        # Temporarily set invalid key
        original_key = os.environ.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = "invalid_key_12345"

        try:
            with pytest.raises(AuthenticationError):
                completion(
                    model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                )
        finally:
            if original_key:
                os.environ["GROQ_API_KEY"] = original_key


# =============================================================================
# Multi-Model Tests
# =============================================================================


class TestGroqMultipleModels:
    """
    Test multiple Groq models to ensure broad compatibility.
    """

    @classmethod
    def setup_class(cls) -> None:
        """Check if credentials are available."""
        if not os.environ.get("GROQ_API_KEY"):
            pytest.skip("Missing GROQ_API_KEY environment variable")

    @pytest.fixture(params=get_test_models())
    def model_name(self, request) -> str:
        """Parameterized fixture for model names."""
        return request.param

    def test_model_basic_completion(self, model_name: str) -> None:
        """Test basic completion across multiple models."""
        from arcllm import completion

        response = completion(
            model=f"groq/{model_name}",
            messages=[{"role": "user", "content": "Say 'ok'."}],
            max_tokens=5,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_model_streaming(self, model_name: str) -> None:
        """Test streaming across multiple models."""
        from arcllm import completion

        response = completion(
            model=f"groq/{model_name}",
            messages=[{"role": "user", "content": "Say 1."}],
            max_tokens=5,
            stream=True,
        )

        chunks = list(response)
        assert len(chunks) > 0

    def test_model_async_completion(self, model_name: str) -> None:
        """Test async completion across multiple models."""
        import asyncio

        from arcllm import acompletion

        async def run_test():
            response = await acompletion(
                model=f"groq/{model_name}",
                messages=[{"role": "user", "content": "Say 'yes'."}],
                max_tokens=5,
            )
            return response

        response = asyncio.run(run_test())
        assert response is not None
        assert response.choices[0].message.content is not None
