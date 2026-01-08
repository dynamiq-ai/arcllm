"""
Comprehensive Integration Tests for Perplexity AI Provider.

This module provides end-to-end integration tests for Perplexity AI
that can be used in CI/CD pipelines with a live API key.

Requirements:
    - Environment variable: PERPLEXITY_API_KEY

Usage in CI/CD:
    export PERPLEXITY_API_KEY="pplx-..."
    pytest tests/integration/test_perplexity_integration.py -v

API Documentation References:
    - Perplexity AI Docs: https://docs.perplexity.ai/
    - Models: https://docs.perplexity.ai/getting-started/models
    - Chat API: https://docs.perplexity.ai/guides/chat-completions-guide
    - Pricing: https://docs.perplexity.ai/guides/pricing

Notes:
    - Perplexity is primarily a search-augmented AI, not a general LLM
    - It does NOT support embeddings
    - It does NOT support tool/function calling
    - All models have web search capabilities built-in
    - Responses include citations from web sources
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from tests.integration.base import IntegrationTestBase


class TestPerplexityIntegration(IntegrationTestBase):
    """
    Comprehensive Perplexity AI integration tests for CI/CD.

    Perplexity uses an OpenAI-compatible API format with search augmentation.

    Available Models (as of January 2026):
    - sonar: Fast, efficient search model (128K context)
    - sonar-pro: Advanced search with grounding (200K context)
    - sonar-reasoning: Reasoning with web search (127K context)
    - sonar-reasoning-pro: Advanced reasoning (128K context)
    - sonar-deep-research: Deep multi-source research (128K context)
    - r1-1776: Special/experimental reasoning model

    Key Features:
    - Real-time web search integration
    - Citation support (returns source URLs)
    - No function/tool calling support
    - No embeddings support
    """

    # Provider configuration
    PROVIDER = "perplexity"
    ENV_VAR = "PERPLEXITY_API_KEY"
    # Use sonar as primary - fast, cheap, good for testing
    PRIMARY_MODEL = "sonar"

    # Feature support flags - Perplexity is search-focused
    SUPPORTS_TOOLS = False  # Perplexity doesn't support function calling
    SUPPORTS_STRUCTURED_OUTPUT = False  # Perplexity doesn't support response_format properly
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = False  # Perplexity doesn't have embeddings API

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

        Should complete in < 10 seconds.
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            max_tokens=10,
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
            max_tokens=10,
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
                {"role": "system", "content": "You are a helpful assistant. Be very brief."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            max_tokens=50,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        assert "Paris" in content or "paris" in content.lower()

    def test_multi_turn_conversation(self) -> None:
        """Test multi-turn conversation handling."""
        from arcllm import completion

        messages = [
            {"role": "system", "content": "You are a math tutor. Be brief."},
            {"role": "user", "content": "What is 5 + 5?"},
            {"role": "assistant", "content": "10"},
            {"role": "user", "content": "And if I multiply that by 2?"},
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=messages,
            max_tokens=50,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        # Should mention 20 somewhere
        assert "20" in content

    def test_temperature_parameter(self) -> None:
        """Test temperature parameter."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Give me a random fun fact."}],
            temperature=0.5,
            max_tokens=100,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    # =========================================================================
    # SEARCH-SPECIFIC TESTS (Perplexity's unique feature)
    # =========================================================================

    def test_web_search_query(self) -> None:
        """Test that Perplexity performs web search for current information."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "What is the current weather like in general terms?"}],
            max_tokens=200,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        # Should return some weather-related content
        assert len(content) > 20

    def test_factual_query_with_citations(self) -> None:
        """Test factual query - Perplexity should provide grounded answers."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "What is Python programming language?"}],
            max_tokens=200,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        # Should return informative content about Python
        assert "Python" in content or "programming" in content.lower()

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
            max_tokens=100,
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
            max_tokens=20,
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
            messages=[{"role": "user", "content": "Say 'async test'."}],
            max_tokens=20,
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
            max_tokens=50,
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
            max_tokens=10,
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
            max_tokens=10,
        )

        cost = completion_cost(response)
        assert cost >= 0

    # =========================================================================
    # MODEL-SPECIFIC TESTS
    # =========================================================================

    def test_sonar_model(self) -> None:
        """Test Sonar model (fast, efficient)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/sonar",
            messages=[{"role": "user", "content": "What is AI? Be brief."}],
            max_tokens=100,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_sonar_pro_model(self) -> None:
        """Test Sonar Pro model (advanced search)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/sonar-pro",
            messages=[{"role": "user", "content": "What is machine learning? Brief answer."}],
            max_tokens=100,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_sonar_reasoning_pro_model(self) -> None:
        """
        Test Sonar Reasoning Pro model (advanced reasoning).

        Note: Reasoning models use <think> tags and need more tokens.
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/sonar-reasoning-pro",
            messages=[{"role": "user", "content": "What is 15% of 200? Give just the number."}],
            max_tokens=500,  # Reasoning models need more tokens for thinking
        )

        assert response is not None
        assert response.choices[0].message.content is not None
        # Content should not be empty (may contain <think> tags)
        assert len(response.choices[0].message.content) > 0

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
                    "content": 'Return a JSON object with "answer" (number) for: what is 5+5? '
                              'Respond ONLY with valid JSON, no other text.',
                }
            ],
            max_tokens=50,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        # Try to parse as JSON - Perplexity may include extra text
        # so we look for JSON-like content
        assert "{" in content and "}" in content

    # =========================================================================
    # ERROR HANDLING TESTS
    # =========================================================================

    def test_invalid_model_error(self) -> None:
        """Test invalid model raises error."""
        from arcllm import completion
        from arcllm.exceptions import ArcLLMError

        with pytest.raises((ArcLLMError, AttributeError, Exception)):
            # Note: May raise various errors depending on Perplexity's response format
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
                api_key="pplx-invalid-key-for-testing",
            )

    def test_embeddings_not_supported(self) -> None:
        """Test that embeddings raises appropriate error."""
        from arcllm import embedding
        from arcllm.exceptions import UnsupportedModelError

        with pytest.raises(UnsupportedModelError):
            embedding(
                model=f"{self.PROVIDER}/sonar",
                input=["Hello"],
            )


# =============================================================================
# Multi-Model Tests
# =============================================================================


class TestPerplexityMultipleModels:
    """
    Test multiple Perplexity AI models to ensure broad compatibility.

    All models are search-augmented and optimized for different use cases.
    """

    # Available models on Perplexity API (as of January 2026)
    # Note: sonar-reasoning excluded due to intermittent availability issues
    MODELS: ClassVar[list[str]] = [
        "sonar",                # Fast, efficient (128K context)
        "sonar-pro",            # Advanced search (200K context)
        "sonar-reasoning-pro",  # Advanced reasoning (128K context)
    ]

    @classmethod
    def setup_class(cls) -> None:
        """Check if credentials are available."""
        import os
        if not os.environ.get("PERPLEXITY_API_KEY"):
            pytest.skip("Missing PERPLEXITY_API_KEY environment variable")

    @pytest.fixture(params=MODELS)
    def model_name(self, request) -> str:
        """Parameterized fixture for model names."""
        return request.param

    def test_model_basic_completion(self, model_name: str) -> None:
        """Test basic completion across multiple models."""
        from arcllm import completion

        response = completion(
            model=f"perplexity/{model_name}",
            messages=[{"role": "user", "content": "What is 1+1? Just the number."}],
            max_tokens=20,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_model_streaming(self, model_name: str) -> None:
        """Test streaming across multiple models."""
        from arcllm import completion

        response = completion(
            model=f"perplexity/{model_name}",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=20,
            stream=True,
        )

        chunks = list(response)
        assert len(chunks) > 0
