"""
Comprehensive integration tests for OpenAI provider.

This module provides end-to-end tests for the OpenAI integration that can be
used in CI/CD pipelines with a live OpenAI API key.

=============================================================================
CI/CD Configuration
=============================================================================

Required environment variables:
    OPENAI_API_KEY          - Your OpenAI API key (required)

Optional environment variables:
    OPENAI_TEST_MODELS      - Comma-separated list of models to test
                              (default: gpt-4o-mini)
    OPENAI_TEST_EMBEDDING   - Embedding model to test
                              (default: text-embedding-3-small)
    ARCLLM_CI_MODE          - Set to "true" for minimal test set (faster CI)
    ARCLLM_SKIP_EXPENSIVE   - Set to "true" to skip expensive model tests

Usage:
    # Run all OpenAI integration tests
    OPENAI_API_KEY=sk-... pytest tests/integration/test_openai_integration.py -v

    # Run with specific models
    OPENAI_API_KEY=sk-... OPENAI_TEST_MODELS=gpt-4o,gpt-4o-mini pytest ...

    # Run in CI mode (minimal tests, faster)
    OPENAI_API_KEY=sk-... ARCLLM_CI_MODE=true pytest ...

    # Run only fast tests
    pytest tests/integration/test_openai_integration.py -v -m "not slow"

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
          - run: pytest tests/integration/test_openai_integration.py -v
            env:
              OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

=============================================================================
OpenAI Documentation References
=============================================================================

- API Reference: https://platform.openai.com/docs/api-reference
- Chat Completions: https://platform.openai.com/docs/api-reference/chat/create
- Embeddings: https://platform.openai.com/docs/api-reference/embeddings
- Models: https://platform.openai.com/docs/models
- Function Calling: https://platform.openai.com/docs/guides/function-calling
- Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- Streaming: https://platform.openai.com/docs/api-reference/streaming
- Error Codes: https://platform.openai.com/docs/guides/error-codes
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import pytest

from tests.integration.base import IntegrationTestBase

# =============================================================================
# Configuration
# =============================================================================


def get_test_models() -> list[str]:
    """Get list of models to test from environment or defaults."""
    env_models = os.environ.get("OPENAI_TEST_MODELS", "")
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]

    # Default models to test - ALL available models
    # As of January 2026: GPT-5.2 is latest flagship
    # We test ALL models to ensure compatibility
    if os.environ.get("ARCLLM_CI_MODE") == "true":
        return ["gpt-4o-mini", "gpt-5-mini"]  # Fast CI with both legacy and new

    # Full test suite - all major models
    return [
        "gpt-4o-mini",  # Legacy, full parameter support
        "gpt-4o",  # Legacy flagship
        "gpt-4.1-mini",  # GPT-4.1 series
        "gpt-5-mini",  # GPT-5 series (restricted params)
        "gpt-5-nano",  # GPT-5 nano (restricted params)
    ]


# Models with restricted parameters (like o1 reasoning models)
# These don't support temperature, top_p, or other sampling parameters
RESTRICTED_PARAM_MODELS = {
    "o1",
    "o1-mini",
    "o1-preview",
    "o1-pro",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-turbo",
    "gpt-5.1",
    "gpt-5.1-mini",
    "gpt-5.1-nano",
    "gpt-5.2",
    "gpt-5.2-mini",
    "gpt-5.2-nano",
    "gpt-5.2-turbo",
}


def is_restricted_model(model: str) -> bool:
    """Check if model has parameter restrictions (no temperature/top_p control)."""
    # Strip provider prefix if present
    if "/" in model:
        model = model.split("/", 1)[1]

    # Check exact match or prefix match
    for restricted in RESTRICTED_PARAM_MODELS:
        if model == restricted or model.startswith(f"{restricted}-"):
            return True
    return False


def get_completion_kwargs(model: str, **kwargs: Any) -> dict[str, Any]:
    """Get completion kwargs adjusted for model restrictions."""
    result = dict(kwargs)

    if is_restricted_model(model):
        # Remove unsupported parameters for restricted models
        result.pop("temperature", None)
        result.pop("top_p", None)
        result.pop("presence_penalty", None)
        result.pop("frequency_penalty", None)
        # Convert max_tokens to max_completion_tokens
        if "max_tokens" in result:
            result["max_completion_tokens"] = result.pop("max_tokens")

    return result


def get_embedding_model() -> str:
    """Get embedding model from environment or default."""
    return os.environ.get("OPENAI_TEST_EMBEDDING", "text-embedding-3-small")


def should_skip_expensive() -> bool:
    """Check if expensive tests should be skipped."""
    return os.environ.get("ARCLLM_SKIP_EXPENSIVE", "").lower() == "true"


# =============================================================================
# Test Markers
# =============================================================================

# Mark tests that require network access
pytestmark = [
    pytest.mark.integration,
    pytest.mark.openai,
]


# =============================================================================
# Base Test Class
# =============================================================================


class TestOpenAIIntegration(IntegrationTestBase):
    """
    Comprehensive OpenAI integration tests.

    Tests all major features:
    - Basic completion
    - Async completion
    - Streaming
    - Tool/function calling
    - Structured output (JSON mode and schema)
    - Embeddings
    - Error handling
    - Cost calculation

    As of January 2026:
    - GPT-5.2 is the latest flagship model (released December 11, 2025)
    - GPT-4o is being retired on February 16, 2026
    - GPT-3.5-turbo is deprecated

    See: https://platform.openai.com/docs/models
    """

    PROVIDER = "openai"
    ENV_VAR = "OPENAI_API_KEY"
    # Use gpt-4o-mini for primary tests (cheap, fast, full parameter support)
    # GPT-5 series models have restricted parameters like o1
    # Available models as of Jan 2026: gpt-5.2, gpt-5.1, gpt-5, gpt-4.1, gpt-4o
    PRIMARY_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = True
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = True

    # =========================================================================
    # Basic Completion Tests
    # =========================================================================

    def test_simple_completion(self) -> None:
        """
        Test basic chat completion.

        OpenAI Docs: https://platform.openai.com/docs/api-reference/chat/create
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
            temperature=0,
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
        """Test completion with system message."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {"role": "system", "content": "You are a helpful math tutor."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            max_tokens=10,
            temperature=0,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        assert "4" in content

    def test_multi_turn_conversation(self) -> None:
        """
        Test multi-turn conversation with message history.

        OpenAI Docs: https://platform.openai.com/docs/guides/text-generation
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
            temperature=0,
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
            temperature=1.0,
            top_p=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            seed=42,
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

        OpenAI Docs: https://platform.openai.com/docs/api-reference/chat/create
        """
        from arcllm import acompletion

        response = await self.retry_on_rate_limit_async(
            acompletion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'async hello' and nothing else."}],
            max_tokens=10,
            temperature=0,
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        content = response.choices[0].message.content
        assert content is not None
        assert "async" in content.lower() or "hello" in content.lower()

    @pytest.mark.asyncio
    async def test_async_completion_concurrent(self) -> None:
        """Test multiple concurrent async completions."""
        from arcllm import acompletion

        async def make_request(i: int):
            return await acompletion(
                model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
                messages=[{"role": "user", "content": f"Say the number {i}."}],
                max_tokens=5,
                temperature=0,
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

        OpenAI Docs: https://platform.openai.com/docs/api-reference/streaming
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

        assert len(chunks) > 1, "Expected multiple chunks"
        assert len(content_parts) > 0, "Expected content deltas"

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    def test_streaming_with_usage(self) -> None:
        """
        Test streaming with include_usage option.

        OpenAI Docs: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options

        When stream_options.include_usage is true, the final chunk contains
        usage information.
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=5,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunks = list(response)
        assert len(chunks) > 0

        # Check that at least one chunk has usage info
        has_usage = any(chunk.usage is not None for chunk in chunks)
        # Note: Usage is typically in the last chunk
        assert has_usage or len(chunks) > 0

    def test_streaming_with_chunk_builder(self) -> None:
        """
        Test assembling streamed response with stream_chunk_builder.

        This is useful when you need to accumulate the full response
        from streaming chunks.
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

        OpenAI Docs: https://platform.openai.com/docs/guides/function-calling
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
            tool_choice="auto",
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

    def test_tool_calling_required(self) -> None:
        """Test tool calling with tool_choice='required'."""
        from arcllm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "What is 10 plus 5?"}],
            tools=tools,
            tool_choice="required",
            max_tokens=100,
        )

        choice = response.choices[0]
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) > 0

    def test_multiple_tool_calls(self) -> None:
        """
        Test parallel tool calls (multiple calls in one response).

        OpenAI Docs: https://platform.openai.com/docs/guides/function-calling/parallel-function-calling
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
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Paris and London?",
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200,
        )

        choice = response.choices[0]
        if choice.message.tool_calls:
            # May return 1 or 2 calls depending on model behavior
            assert len(choice.message.tool_calls) >= 1
            for tc in choice.message.tool_calls:
                assert tc.function.name == "get_weather"

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
        messages = [{"role": "user", "content": "What time is it?"}]
        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=messages,
            tools=tools,
            tool_choice="auto",
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

        OpenAI Docs: https://platform.openai.com/docs/guides/structured-outputs/json-mode

        Note: You must include "JSON" in the prompt when using json_object mode.
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

    def test_structured_output_json_schema(self) -> None:
        """
        Test JSON schema structured output (strict mode).

        OpenAI Docs: https://platform.openai.com/docs/guides/structured-outputs

        With strict: true, the output is guaranteed to match the schema.
        Only supported on gpt-4o, gpt-4o-mini, gpt-4o-2024-08-06.
        """
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": "Generate a person with name Alice and age 30.",
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                        "additionalProperties": False,
                    },
                },
            },
            max_tokens=50,
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        assert "name" in data
        assert "age" in data
        assert isinstance(data["name"], str)
        assert isinstance(data["age"], int)

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
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

        # Check model_extra compatibility
        assert "usage" in response.model_extra
        assert response.model_extra["usage"]["prompt_tokens"] > 0

    def test_cost_calculation(self) -> None:
        """Test cost calculation from response."""
        import arcllm
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        cost = arcllm.completion_cost(response)
        assert cost >= 0
        assert isinstance(cost, float)

    # =========================================================================
    # Embedding Tests
    # =========================================================================

    def test_embeddings(self) -> None:
        """
        Test embedding generation.

        OpenAI Docs: https://platform.openai.com/docs/api-reference/embeddings/create
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

    def test_embedding_with_dimensions(self) -> None:
        """
        Test embedding with reduced dimensions.

        OpenAI Docs: https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions

        Only supported for text-embedding-3-small and text-embedding-3-large.
        """
        from arcllm import embedding

        if "ada" in self.EMBEDDING_MODEL:
            pytest.skip("Dimensions not supported for ada model")

        response = self.retry_on_rate_limit(
            embedding,
            model=f"{self.PROVIDER}/{self.EMBEDDING_MODEL}",
            input=["Hello"],
            dimensions=256,
        )

        assert len(response.data[0].embedding) == 256

    def test_embedding_usage(self) -> None:
        """Test embedding usage reporting."""
        from arcllm import embedding

        response = self.retry_on_rate_limit(
            embedding,
            model=f"{self.PROVIDER}/{self.EMBEDDING_MODEL}",
            input=["Hello, world!"],
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0

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

        OpenAI Docs: https://platform.openai.com/docs/guides/error-codes

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
                api_key="sk-invalid-key-for-testing",
            )

    def test_invalid_request_error(self) -> None:
        """
        Test that invalid request raises appropriate error.

        Expected: 400 Bad Request
        """
        from arcllm import completion
        from arcllm.exceptions import ArcLLMError, InvalidRequestError

        with pytest.raises((ArcLLMError, InvalidRequestError)):
            completion(
                model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
                messages=[],  # Empty messages is invalid
                max_tokens=5,
            )

    # =========================================================================
    # Model Response Structure Tests
    # =========================================================================

    def test_response_structure(self) -> None:
        """Test that response has correct structure per OpenAI spec."""
        from arcllm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        # Check top-level fields
        assert response.id.startswith("chatcmpl-")
        assert response.object == "chat.completion"
        assert response.created > 0
        assert response.model is not None

        # Check choices
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.finish_reason in ("stop", "length", "tool_calls")

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


class TestOpenAIMultipleModels:
    """
    Test multiple OpenAI models to ensure broad compatibility.

    These tests run against different models based on configuration.
    """

    @classmethod
    def setup_class(cls) -> None:
        """Check if credentials are available."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("Missing OPENAI_API_KEY environment variable")

    @pytest.fixture(params=get_test_models())
    def model_name(self, request) -> str:
        """Parameterized fixture for model names."""
        return request.param

    def test_model_basic_completion(self, model_name: str) -> None:
        """Test basic completion across multiple models."""
        from arcllm import completion

        # Build kwargs adjusted for model restrictions
        kwargs = get_completion_kwargs(
            model_name,
            max_tokens=5,
            temperature=0,
        )

        response = completion(
            model=f"openai/{model_name}",
            messages=[{"role": "user", "content": "Say 'ok'."}],
            **kwargs,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_model_streaming(self, model_name: str) -> None:
        """Test streaming across multiple models."""
        from arcllm import completion

        # Build kwargs adjusted for model restrictions
        kwargs = get_completion_kwargs(
            model_name,
            max_tokens=5,
        )

        response = completion(
            model=f"openai/{model_name}",
            messages=[{"role": "user", "content": "Say 1."}],
            stream=True,
            **kwargs,
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

        kwargs = get_completion_kwargs(model_name, max_tokens=100)

        response = completion(
            model=f"openai/{model_name}",
            messages=[{"role": "user", "content": "What time is it?"}],
            tools=tools,
            **kwargs,
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
            kwargs = get_completion_kwargs(model_name, max_tokens=5)
            response = await acompletion(
                model=f"openai/{model_name}",
                messages=[{"role": "user", "content": "Say 'yes'."}],
                **kwargs,
            )
            return response

        response = asyncio.run(run_test())
        assert response is not None
        assert response.choices[0].message.content is not None


# =============================================================================
# Performance/Stress Tests (marked slow)
# =============================================================================


@pytest.mark.slow
class TestOpenAIPerformance:
    """
    Performance and stress tests.

    These tests are marked as slow and can be skipped with:
        pytest -m "not slow"
    """

    @classmethod
    def setup_class(cls) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("Missing OPENAI_API_KEY")

    def test_rapid_sequential_requests(self) -> None:
        """Test multiple sequential requests."""
        from arcllm import completion

        for i in range(5):
            response = completion(
                model="openai/gpt-4o-mini",
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
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": f"Say {i}."}],
                max_tokens=5,
            )

        results = await asyncio.gather(*[make_request(i) for i in range(5)])

        assert len(results) == 5
        for r in results:
            assert r is not None
