"""
Comprehensive Integration Tests for Fireworks AI Provider.

This module provides end-to-end integration tests for Fireworks AI
that can be used in CI/CD pipelines with a live API key.

Requirements:
    - Environment variable: FIREWORKS_API_KEY

Usage in CI/CD:
    export FIREWORKS_API_KEY="fw_..."
    pytest tests/integration/test_fireworks_integration.py -v

API Documentation References:
    - Fireworks AI Docs: https://docs.fireworks.ai/
    - Models: https://fireworks.ai/models
    - Chat API: https://docs.fireworks.ai/api-reference/post-chatcompletions
    - Embeddings: https://docs.fireworks.ai/guides/querying-embeddings-models
    - Vision: https://docs.fireworks.ai/guides/querying-vision-language-models
    - Pricing: https://fireworks.ai/pricing
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from tests.integration.base import IntegrationTestBase


class TestFireworksIntegration(IntegrationTestBase):
    """
    Comprehensive Fireworks AI integration tests for CI/CD.

    Fireworks AI uses an OpenAI-compatible API format, so most OpenAI
    features should work out of the box.

    Model Naming Convention:
    - Serverless models use: accounts/fireworks/models/<model-name>
    - Example: accounts/fireworks/models/llama-v3p3-70b-instruct

    Available Model Categories (as of January 2026):
    - LLMs: Llama 4, Llama 3.3, Qwen3, DeepSeek R1/V3, Mixtral, etc.
    - VLMs: Qwen2.5-VL, Llama 3.2 Vision, Phi-3.5 Vision
    - Embedding: Nomic, BGE, GTE, Qwen3 Embedding
    """

    # Provider configuration
    PROVIDER = "fireworks_ai"
    ENV_VAR = "FIREWORKS_API_KEY"
    # Use Llama 3.3 70B as primary - fast, capable, good value
    PRIMARY_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"

    # Feature support flags
    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = True  # Via JSON mode
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = True
    SUPPORTS_VISION = True  # Via vision models like Qwen2.5-VL

    # Embedding model
    EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

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

        Should complete in < 5 seconds.
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
    # EMBEDDING TESTS
    # =========================================================================

    def test_embeddings_basic(self) -> None:
        """Test basic embedding generation."""
        from arcllm import embedding

        response = self.retry_on_rate_limit(
            embedding,
            model=f"{self.PROVIDER}/{self.EMBEDDING_MODEL}",
            input=["Hello, world!"],
        )

        assert response is not None
        assert response.data is not None
        assert len(response.data) > 0
        assert response.data[0].embedding is not None
        assert len(response.data[0].embedding) > 0
        assert all(isinstance(v, float) for v in response.data[0].embedding)

    def test_embeddings_multiple(self) -> None:
        """Test embedding with multiple inputs."""
        from arcllm import embedding

        texts = ["Hello, world!", "How are you?", "Goodbye!"]

        response = self.retry_on_rate_limit(
            embedding,
            model=f"{self.PROVIDER}/{self.EMBEDDING_MODEL}",
            input=texts,
        )

        assert len(response.data) == 3
        for i, item in enumerate(response.data):
            assert item.index == i
            assert len(item.embedding) > 0

    @pytest.mark.asyncio
    async def test_async_embedding(self) -> None:
        """Test async embedding generation."""
        from arcllm import aembedding

        response = await aembedding(
            model=f"{self.PROVIDER}/{self.EMBEDDING_MODEL}",
            input=["Hello, async world!"],
        )

        assert response is not None
        assert len(response.data) == 1
        assert len(response.data[0].embedding) > 0

    # =========================================================================
    # MODEL-SPECIFIC TESTS - LLMs
    # =========================================================================

    def test_llama_3p3_70b(self) -> None:
        """Test Llama 3.3 70B model."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/accounts/fireworks/models/llama-v3p3-70b-instruct",
            messages=[{"role": "user", "content": "Say test."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_qwen3_8b(self) -> None:
        """Test Qwen3 8B model (fast, cheap)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/accounts/fireworks/models/qwen3-8b",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_deepseek_v3p1(self) -> None:
        """Test DeepSeek V3.1 model."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/accounts/fireworks/models/deepseek-v3p1",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_qwen3_30b(self) -> None:
        """Test Qwen3 30B model (MoE)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/accounts/fireworks/models/qwen3-30b-a3b",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_mixtral_8x22b(self) -> None:
        """Test Mixtral 8x22B MoE model."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/accounts/fireworks/models/mixtral-8x22b-instruct",
            messages=[{"role": "user", "content": "Say test."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_deepseek_v3p2(self) -> None:
        """Test DeepSeek V3.2 model (latest)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/accounts/fireworks/models/deepseek-v3p2",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    # =========================================================================
    # MODEL-SPECIFIC TESTS - Vision/VLMs
    # =========================================================================

    def test_qwen2p5_vl_32b(self) -> None:
        """Test Qwen 2.5 VL 32B Vision model with text-only input."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/accounts/fireworks/models/qwen2p5-vl-32b-instruct",
            messages=[
                {"role": "user", "content": "Describe the concept of 'sunrise' in one sentence."}
            ],
            max_tokens=50,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_qwen3_vl_30b(self) -> None:
        """Test Qwen3 VL 30B Vision model with text-only input."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/accounts/fireworks/models/qwen3-vl-30b-a3b-instruct",
            messages=[{"role": "user", "content": "What is a good description of mountains?"}],
            max_tokens=50,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    # =========================================================================
    # STRUCTURED OUTPUT TESTS
    # =========================================================================

    def test_json_mode(self) -> None:
        """Test JSON mode structured output."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": 'Return a JSON object with "name" (string) and "age" (number). '
                    'Example: {"name": "Alice", "age": 30}',
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=50,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None

        # Should be valid JSON
        data = json.loads(content)
        assert isinstance(data, dict)

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
        """Test invalid API key raises authentication error."""
        from arcllm import completion
        from arcllm.exceptions import ArcLLMError, AuthenticationError

        with pytest.raises((ArcLLMError, AuthenticationError)):
            completion(
                model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                api_key="fw_invalid_key_for_testing",
            )


# =============================================================================
# Multi-Model Tests
# =============================================================================


class TestFireworksMultipleModels:
    """
    Test multiple Fireworks AI models to ensure broad compatibility.

    Models are grouped by category for easier testing and maintenance.
    Note: These are models available on the serverless API as of January 2026.
    """

    # Available LLM models on Fireworks AI (serverless) - verified via API
    # Note: Only models with free/standard tier access are included
    LLM_MODELS: ClassVar[list[str]] = [
        # Llama family
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        # Qwen family
        "accounts/fireworks/models/qwen3-30b-a3b",
        "accounts/fireworks/models/qwen3-8b",
        # Mixtral
        "accounts/fireworks/models/mixtral-8x22b-instruct",
        # DeepSeek
        "accounts/fireworks/models/deepseek-v3p1",
        "accounts/fireworks/models/deepseek-v3p2",
    ]

    @classmethod
    def setup_class(cls) -> None:
        """Check if credentials are available."""
        import os

        if not os.environ.get("FIREWORKS_API_KEY"):
            pytest.skip("Missing FIREWORKS_API_KEY environment variable")

    @pytest.fixture(params=LLM_MODELS)
    def model_name(self, request) -> str:
        """Parameterized fixture for model names."""
        return request.param

    def test_model_basic_completion(self, model_name: str) -> None:
        """Test basic completion across multiple models."""
        from arcllm import completion

        response = completion(
            model=f"fireworks_ai/{model_name}",
            messages=[{"role": "user", "content": "Say 'ok'."}],
            max_tokens=5,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_model_streaming(self, model_name: str) -> None:
        """Test streaming across multiple models."""
        from arcllm import completion

        response = completion(
            model=f"fireworks_ai/{model_name}",
            messages=[{"role": "user", "content": "Say 1."}],
            max_tokens=5,
            stream=True,
        )

        chunks = list(response)
        assert len(chunks) > 0
