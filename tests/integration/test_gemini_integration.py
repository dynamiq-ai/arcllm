"""
Comprehensive integration tests for Google Gemini (AI Studio) provider.

This module provides end-to-end tests for the Gemini integration that can be
used in CI/CD pipelines with a live Gemini API key.

=============================================================================
CI/CD Configuration
=============================================================================

Required environment variables:
    GEMINI_API_KEY          - Your Gemini API key (required)
    or
    GOOGLE_API_KEY          - Alternative env var for Gemini API key

Optional environment variables:
    GEMINI_TEST_MODELS      - Comma-separated list of models to test
                              (default: gemini-2.0-flash)
    GEMINI_TEST_EMBEDDING   - Embedding model to test
                              (default: text-embedding-004)
    ARCLLM_CI_MODE          - Set to "true" for minimal test set (faster CI)

Usage:
    # Run all Gemini integration tests
    GEMINI_API_KEY=... pytest tests/integration/test_gemini_integration.py -v

    # Run with specific models
    GEMINI_API_KEY=... GEMINI_TEST_MODELS=gemini-2.0-flash,gemini-2.5-flash pytest ...

    # Run in CI mode (minimal tests, faster)
    GEMINI_API_KEY=... ARCLLM_CI_MODE=true pytest ...

GitHub Actions example:
    jobs:
      integration-tests:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: '3.12'
          - run: pip install -e ".[dev]"
          - run: pytest tests/integration/test_gemini_integration.py -v
            env:
              GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}

=============================================================================
Gemini API Documentation References
=============================================================================

- API Reference: https://ai.google.dev/api
- Gemini API: https://ai.google.dev/gemini-api/docs
- Models: https://ai.google.dev/gemini-api/docs/models/gemini
- Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
- Embeddings: https://ai.google.dev/gemini-api/docs/embeddings

=============================================================================
Available Models (as of January 2026)
=============================================================================

Chat/Completion Models:
- gemini-2.5-pro          - Latest flagship model
- gemini-2.5-flash        - Fast, latest generation
- gemini-2.5-flash-lite   - Lightweight, fast
- gemini-2.0-flash        - Fast, stable
- gemini-2.0-flash-lite   - Lightweight, stable
- gemini-3-pro-preview    - Next-gen preview
- gemini-3-flash-preview  - Next-gen fast preview

Embedding Models:
- text-embedding-004      - Standard embedding model
- gemini-embedding-001    - Newer embedding model
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest

from tests.integration.base import IntegrationTestBase

# =============================================================================
# Configuration
# =============================================================================


def get_test_models() -> list[str]:
    """Get list of models to test from environment or defaults."""
    env_models = os.environ.get("GEMINI_TEST_MODELS", "")
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]

    # Default models to test
    if os.environ.get("ARCLLM_CI_MODE") == "true":
        return ["gemini-2.0-flash"]  # Fast CI with stable model

    # Full test suite - major models
    return [
        "gemini-2.0-flash",  # Stable, fast
        "gemini-2.0-flash-lite",  # Lightweight
        "gemini-2.5-flash",  # Latest flash
    ]


def get_embedding_model() -> str:
    """Get embedding model from environment or default."""
    return os.environ.get("GEMINI_TEST_EMBEDDING", "text-embedding-004")


# =============================================================================
# Test Markers
# =============================================================================

# Mark tests that require network access
pytestmark = [
    pytest.mark.integration,
    pytest.mark.gemini,
]


# =============================================================================
# Base Test Class
# =============================================================================


class TestGeminiIntegration(IntegrationTestBase):
    """
    Comprehensive Gemini integration tests.

    Tests all major features:
    - Basic completion
    - Async completion
    - Streaming
    - Tool/function calling
    - Structured output (JSON mode)
    - Embeddings
    - Error handling

    As of January 2026:
    - Gemini 2.5 is the latest generation
    - Gemini 3.0 is in preview

    See: https://ai.google.dev/gemini-api/docs/models/gemini
    """

    PROVIDER = "gemini"
    ENV_VAR = "GEMINI_API_KEY"
    # Use gemini-2.0-flash for primary tests (stable, fast, full feature support)
    PRIMARY_MODEL = "gemini-2.0-flash"
    EMBEDDING_MODEL = "text-embedding-004"
    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = True
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = True

    @classmethod
    def setup_class(cls) -> None:
        """Check if credentials are available."""
        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

    # =========================================================================
    # Basic Completion Tests
    # =========================================================================

    def test_simple_completion(self) -> None:
        """
        Test basic chat completion.

        Gemini Docs: https://ai.google.dev/gemini-api/docs/text-generation
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.id is not None
        assert response.model is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        assert response.choices[0].finish_reason in ("stop", "length")

    def test_completion_with_system_message(self) -> None:
        """Test completion with system message (systemInstruction in Gemini)."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {"role": "system", "content": "You are a helpful math tutor. Be concise."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            max_tokens=10,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        assert "4" in content

    def test_multi_turn_conversation(self) -> None:
        """
        Test multi-turn conversation with message history.

        Gemini Docs: https://ai.google.dev/gemini-api/docs/text-generation#multi-turn
        """
        from arcllm import completion

        messages = [
            {"role": "system", "content": "You are a math tutor. Be concise."},
            {"role": "user", "content": "What is 5 + 5?"},
            {"role": "assistant", "content": "10"},
            {"role": "user", "content": "And multiply that by 2?"},
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=messages,
            max_tokens=10,
        )

        content = response.choices[0].message.content
        assert content is not None
        assert "20" in content

    def test_completion_with_parameters(self) -> None:
        """Test completion with various parameters."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Generate a random word."}],
            max_tokens=20,
            temperature=0.8,
            top_p=0.9,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    # =========================================================================
    # Async Completion Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_async_completion(self) -> None:
        """
        Test async chat completion.

        Gemini Docs: https://ai.google.dev/gemini-api/docs/text-generation
        """
        from arcllm import acompletion

        response = await self.retry_on_rate_limit_async(
            acompletion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'async hello' and nothing else."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        content = response.choices[0].message.content
        assert content is not None

    @pytest.mark.asyncio
    async def test_async_completion_concurrent(self) -> None:
        """Test multiple concurrent async completions."""
        from arcllm import acompletion

        async def make_request(i: int):
            return await acompletion(
                model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
                messages=[{"role": "user", "content": f"Say the number {i}."}],
                max_tokens=5,
            )

        # Run 3 requests concurrently
        results = await asyncio.gather(
            make_request(1),
            make_request(2),
            make_request(3),
        )

        assert len(results) == 3
        for result in results:
            assert result is not None
            assert result.choices[0].message.content is not None

    # =========================================================================
    # Streaming Tests
    # =========================================================================

    def test_streaming_completion(self) -> None:
        """
        Test streaming chat completion.

        Gemini Docs: https://ai.google.dev/gemini-api/docs/text-generation#streaming
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=50,
            stream=True,
        )

        chunks = []
        content_parts = []

        for chunk in response:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0, "Expected at least one chunk"
        assert len(content_parts) > 0, "Expected content deltas"

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    def test_streaming_with_chunk_builder(self) -> None:
        """
        Test assembling streamed response with stream_chunk_builder.
        """
        from arcllm import completion, stream_chunk_builder

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'test'."}],
            max_tokens=10,
            stream=True,
        )

        chunks = list(response)
        final_response = stream_chunk_builder(chunks)

        assert final_response is not None
        assert final_response.id is not None
        assert final_response.choices is not None
        assert len(final_response.choices) > 0
        assert final_response.choices[0].message.content is not None
        assert final_response.choices[0].message.role == "assistant"

    @pytest.mark.asyncio
    async def test_async_streaming(self) -> None:
        """Test async streaming completion."""
        from arcllm import acompletion

        response = await acompletion(
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Count 1 to 3."}],
            max_tokens=30,
            stream=True,
        )

        chunks = []
        content_parts = []

        async for chunk in response:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0
        assert len(content_parts) > 0

    # =========================================================================
    # Tool Calling Tests
    # =========================================================================

    def test_tool_calling(self) -> None:
        """
        Test function/tool calling.

        Gemini Docs: https://ai.google.dev/gemini-api/docs/function-calling
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
                                "description": "City name, e.g. 'Paris, France'",
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
            max_tokens=100,
        )

        assert response is not None
        choice = response.choices[0]

        # Model should call the tool
        has_tool_calls = (
            choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0
        )

        if has_tool_calls:
            tool_call = choice.message.tool_calls[0]
            assert tool_call.id is not None
            assert tool_call.type == "function"
            assert tool_call.function is not None
            assert tool_call.function.name == "get_weather"

            # Arguments should be valid JSON
            args = json.loads(tool_call.function.arguments)
            assert "location" in args
            assert isinstance(args["location"], str)

    def test_tool_calling_conversation_flow(self) -> None:
        """Test complete tool calling conversation flow."""
        from arcllm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        # First call - model requests tool
        messages: list[dict[str, Any]] = [{"role": "user", "content": "What time is it?"}]
        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=messages,
            tools=tools,
            max_tokens=100,
        )

        choice = response.choices[0]
        if choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]

            # Add assistant message with tool call
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": [tc.model_dump() for tc in choice.message.tool_calls],
                }
            )

            # Add tool result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": '{"time": "3:45 PM"}',
                }
            )

            # Second call - model responds with result
            final_response = self.retry_on_rate_limit(
                completion,
                model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
                messages=messages,
                tools=tools,
                max_tokens=50,
            )

            assert final_response.choices[0].message.content is not None

    # =========================================================================
    # Structured Output Tests
    # =========================================================================

    def test_structured_output_json_mode(self) -> None:
        """
        Test JSON mode structured output.

        Gemini Docs: https://ai.google.dev/gemini-api/docs/json-mode
        """
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

        content = response.choices[0].message.content
        assert content is not None

        data = json.loads(content)
        assert isinstance(data, dict)

    # =========================================================================
    # Usage & Pricing Tests
    # =========================================================================

    def test_usage_reporting(self) -> None:
        """Test that usage information is correctly reported."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens >= 0
        assert response.usage.total_tokens > 0

        # Check model_extra compatibility
        assert "usage" in response.model_extra

    # =========================================================================
    # Embedding Tests
    # =========================================================================

    def test_embeddings(self) -> None:
        """
        Test embedding generation.

        Gemini Docs: https://ai.google.dev/gemini-api/docs/embeddings
        """
        from arcllm import embedding

        response = self.retry_on_rate_limit(
            embedding,
            model=f"{self.PROVIDER}/{self.EMBEDDING_MODEL}",
            input=["Hello, world!"],
        )

        assert response is not None
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].embedding is not None
        assert len(response.data[0].embedding) > 0
        assert all(isinstance(v, float) for v in response.data[0].embedding)
        assert response.data[0].index == 0

    def test_embedding_multiple_inputs(self) -> None:
        """Test embedding with multiple text inputs."""
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
    # Error Handling Tests
    # =========================================================================

    def test_invalid_model_error(self) -> None:
        """
        Test that invalid model raises appropriate error.

        Expected: 404 error (model not found)
        """
        from arcllm import completion
        from arcllm.exceptions import ArcLLMError, UnsupportedModelError

        with pytest.raises((ArcLLMError, UnsupportedModelError)):
            completion(
                model=f"{self.PROVIDER}/nonexistent-model-xyz123",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )

    def test_invalid_api_key_error(self) -> None:
        """
        Test that invalid API key raises authentication error.

        Expected: 401 Unauthorized
        """
        from arcllm import completion
        from arcllm.exceptions import ArcLLMError, AuthenticationError

        with pytest.raises((ArcLLMError, AuthenticationError)):
            completion(
                model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                api_key="invalid-key-for-testing",
            )

    # =========================================================================
    # Model Response Structure Tests
    # =========================================================================

    def test_response_structure(self) -> None:
        """Test that response has correct structure."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        # Check top-level fields
        assert response.id is not None
        assert response.object == "chat.completion"
        assert response.created > 0
        assert response.model is not None

        # Check choices
        assert len(response.choices) >= 1
        choice = response.choices[0]
        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.finish_reason in ("stop", "length", "tool_calls", "content_filter")

    def test_model_dump(self) -> None:
        """Test serialization with model_dump()."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        data = response.model_dump()
        assert isinstance(data, dict)
        assert "id" in data
        assert "choices" in data
        assert "usage" in data

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert isinstance(json_str, str)


# =============================================================================
# Multi-Model Tests
# =============================================================================


class TestGeminiMultipleModels:
    """
    Test multiple Gemini models to ensure broad compatibility.

    These tests run against different models based on configuration.
    """

    @classmethod
    def setup_class(cls) -> None:
        """Check if credentials are available."""
        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

    @pytest.fixture(params=get_test_models())
    def model_name(self, request) -> str:
        """Parameterized fixture for model names."""
        return request.param

    def test_model_basic_completion(self, model_name: str) -> None:
        """Test basic completion across multiple models."""
        from arcllm import completion

        response = completion(
            model=f"gemini/{model_name}",
            messages=[{"role": "user", "content": "Say 'ok'."}],
            max_tokens=5,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_model_streaming(self, model_name: str) -> None:
        """Test streaming across multiple models."""
        from arcllm import completion

        response = completion(
            model=f"gemini/{model_name}",
            messages=[{"role": "user", "content": "Say 1."}],
            max_tokens=5,
            stream=True,
        )

        chunks = list(response)
        assert len(chunks) > 0

    def test_model_tool_calling(self, model_name: str) -> None:
        """Test tool calling across multiple models."""
        from arcllm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        response = completion(
            model=f"gemini/{model_name}",
            messages=[{"role": "user", "content": "What time is it?"}],
            tools=tools,
            max_tokens=100,
        )

        assert response is not None
        # Model should either call the tool or respond
        assert (
            response.choices[0].message.content is not None
            or response.choices[0].message.tool_calls is not None
        )

    def test_model_async_completion(self, model_name: str) -> None:
        """Test async completion across multiple models."""
        import asyncio

        from arcllm import acompletion

        async def run_test():
            response = await acompletion(
                model=f"gemini/{model_name}",
                messages=[{"role": "user", "content": "Say 'yes'."}],
                max_tokens=5,
            )
            return response

        response = asyncio.run(run_test())
        assert response is not None
        assert response.choices[0].message.content is not None


# =============================================================================
# Performance/Stress Tests (marked slow)
# =============================================================================


@pytest.mark.slow
class TestGeminiPerformance:
    """
    Performance and stress tests.

    These tests are marked as slow and can be skipped with:
        pytest -m "not slow"
    """

    @classmethod
    def setup_class(cls) -> None:
        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("Missing GEMINI_API_KEY")

    def test_rapid_sequential_requests(self) -> None:
        """Test multiple sequential requests."""
        import time

        from arcllm import completion

        for i in range(5):
            response = completion(
                model="gemini/gemini-2.0-flash",
                messages=[{"role": "user", "content": f"Say {i}."}],
                max_tokens=5,
            )
            assert response is not None
            time.sleep(0.2)  # Small delay to avoid rate limits

    @pytest.mark.asyncio
    async def test_concurrent_async_requests(self) -> None:
        """Test concurrent async requests."""
        from arcllm import acompletion

        async def make_request(i: int):
            return await acompletion(
                model="gemini/gemini-2.0-flash",
                messages=[{"role": "user", "content": f"Say {i}."}],
                max_tokens=5,
            )

        results = await asyncio.gather(*[make_request(i) for i in range(5)])

        assert len(results) == 5
        for r in results:
            assert r is not None
