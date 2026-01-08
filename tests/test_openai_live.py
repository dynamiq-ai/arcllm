"""
Comprehensive live OpenAI API tests.

This module tests arcllm against the real OpenAI API with various models
and features. It validates:
- Basic completions across all supported models
- Streaming responses
- Tool/function calling
- Structured output (JSON mode and JSON schema)
- Embeddings
- Error handling

OpenAI API Documentation References:
- Chat Completions: https://platform.openai.com/docs/api-reference/chat
- Streaming: https://platform.openai.com/docs/api-reference/streaming
- Tool/Function Calling: https://platform.openai.com/docs/guides/function-calling
- Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- Embeddings: https://platform.openai.com/docs/api-reference/embeddings
- Models: https://platform.openai.com/docs/models
- Error Handling: https://platform.openai.com/docs/guides/error-codes

Run with:
    OPENAI_API_KEY=sk-... python -m pytest tests/test_openai_live.py -v -s
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import arcllm
from arcllm import acompletion, completion, embedding, stream_chunk_builder
from arcllm.exceptions import (
    ArcLLMError,
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
)

# =============================================================================
# Test Configuration
# =============================================================================

# API Key for testing
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# =============================================================================
# Models to test (organized by category)
# Reference: https://platform.openai.com/docs/models
# Last Updated: January 8, 2026
# =============================================================================

# GPT-4o family - Best for testing (full parameter support, cheap)
# Docs: https://platform.openai.com/docs/models/gpt-4o
GPT4O_MODELS = [
    "gpt-4o-mini",               # Cheap, fast, full param support (best for tests)
    "gpt-4o",                    # Full GPT-4o
]

# GPT-4.1 models (2025) - Uses max_completion_tokens
GPT41_MODELS = [
    "gpt-4.1-mini",              # Affordable, uses max_completion_tokens
    # "gpt-4.1",                 # Full 4.1
]

# GPT-5 family - Latest flagship (uses max_completion_tokens, restricted params)
# Docs: https://platform.openai.com/docs/models/gpt-5
# NOTE: GPT-5 models have restricted parameters (no temperature control on some)
GPT5_MODELS = [
    # "gpt-5.2",                 # Latest flagship (restricted params)
    # "gpt-5-mini",              # Mini version (restricted params)
    # "gpt-5-nano",              # Nano version (very restricted params)
]

# o3 Reasoning models - Next-gen reasoning (2025)
# Docs: https://platform.openai.com/docs/models/o3
O3_MODELS = [
    # "o3-mini",                 # Affordable reasoning
    # "o3",                      # Full o3 (expensive)
]

# o1 Reasoning models
O1_MODELS = [
    # "o1",                      # o1 reasoning
]

# Embedding models
# Docs: https://platform.openai.com/docs/models/embeddings
EMBEDDING_MODELS = [
    "text-embedding-3-small",    # Recommended for most use cases
    "text-embedding-3-large",    # Higher performance
]

# Combine all chat models for testing
# Use gpt-4o-mini for comprehensive testing (full param support)
CHAT_MODELS = GPT4O_MODELS

# Models that support vision (image inputs)
VISION_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-5.2", "gpt-5"]

# Models that support tool calling
TOOL_CALLING_MODELS = CHAT_MODELS

# Models that support structured output (JSON schema)
STRUCTURED_OUTPUT_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-5.2", "gpt-5"]


# =============================================================================
# Utility Functions
# =============================================================================


def retry_on_rate_limit(func, *args, max_retries: int = 3, **kwargs) -> Any:
    """Execute function with retry on rate limit errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            last_error = e
            wait_time = 2 * (2 ** attempt)  # Exponential backoff
            print(f"  Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
            time.sleep(wait_time)
    raise last_error


def print_result(test_name: str, model: str, success: bool, details: str = "") -> None:
    """Print test result with consistent formatting."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status}: {test_name} - {model}")
    if details:
        print(f"         {details}")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")


# =============================================================================
# Basic Completion Tests
# =============================================================================


def test_basic_completion(model: str) -> bool:
    """
    Test basic chat completion.
    
    OpenAI Docs: https://platform.openai.com/docs/api-reference/chat/create
    
    Request format:
    {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
    """
    try:
        response = retry_on_rate_limit(
            completion,
            model=f"openai/{model}",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello World' and nothing else."},
            ],
            max_tokens=20,
            temperature=0,
        )
        
        content = response.choices[0].message.content
        has_content = content and len(content) > 0
        has_usage = response.usage is not None
        
        details = f"Content: '{content[:50]}...' | Tokens: {response.usage.total_tokens if response.usage else 'N/A'}"
        print_result("basic_completion", model, has_content and has_usage, details)
        return has_content and has_usage
        
    except Exception as e:
        print_result("basic_completion", model, False, str(e))
        return False


def test_multi_turn_conversation(model: str) -> bool:
    """
    Test multi-turn conversation with message history.
    
    OpenAI Docs: https://platform.openai.com/docs/guides/text-generation/chat-completions-api
    
    Messages format:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."}
    ]
    """
    try:
        messages = [
            {"role": "system", "content": "You are a math tutor. Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And what is that plus 3?"},
        ]
        
        response = retry_on_rate_limit(
            completion,
            model=f"openai/{model}",
            messages=messages,
            max_tokens=10,
            temperature=0,
        )
        
        content = response.choices[0].message.content
        # Should answer "7" or contain "7"
        success = content and "7" in content
        print_result("multi_turn", model, success, f"Response: '{content}'")
        return success
        
    except Exception as e:
        print_result("multi_turn", model, False, str(e))
        return False


# =============================================================================
# Streaming Tests
# =============================================================================


def test_streaming(model: str) -> bool:
    """
    Test streaming responses.
    
    OpenAI Docs: https://platform.openai.com/docs/api-reference/streaming
    
    Response format (Server-Sent Events):
    data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}
    data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":" World"}}]}
    data: [DONE]
    """
    try:
        stream = retry_on_rate_limit(
            completion,
            model=f"openai/{model}",
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            max_tokens=30,
            stream=True,
        )
        
        chunks = []
        content_parts = []
        
        for chunk in stream:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)
        
        full_content = "".join(content_parts)
        success = len(chunks) > 1 and len(full_content) > 0
        print_result("streaming", model, success, f"Chunks: {len(chunks)}, Content: '{full_content[:50]}'")
        return success
        
    except Exception as e:
        print_result("streaming", model, False, str(e))
        return False


def test_streaming_with_usage(model: str) -> bool:
    """
    Test streaming with usage information.
    
    OpenAI Docs: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options
    
    Request:
    {
        "stream": true,
        "stream_options": {"include_usage": true}
    }
    
    Final chunk includes usage:
    {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    """
    try:
        stream = retry_on_rate_limit(
            completion,
            model=f"openai/{model}",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=10,
            stream=True,
            stream_options={"include_usage": True},
        )
        
        chunks = list(stream)
        final_response = stream_chunk_builder(chunks)
        
        has_usage = final_response.usage is not None
        success = len(chunks) > 0 and has_usage
        
        usage_str = f"Tokens: {final_response.usage.total_tokens}" if has_usage else "No usage"
        print_result("streaming_usage", model, success, usage_str)
        return success
        
    except Exception as e:
        print_result("streaming_usage", model, False, str(e))
        return False


# =============================================================================
# Tool Calling Tests
# =============================================================================


def test_tool_calling(model: str) -> bool:
    """
    Test function/tool calling.
    
    OpenAI Docs: https://platform.openai.com/docs/guides/function-calling
    
    Tools format:
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }
    
    Response tool_calls format:
    {
        "id": "call_xxx",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}
    }
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country, e.g. 'Paris, France'",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    
    try:
        response = retry_on_rate_limit(
            completion,
            model=f"openai/{model}",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=100,
        )
        
        choice = response.choices[0]
        has_tool_call = choice.message.tool_calls and len(choice.message.tool_calls) > 0
        
        if has_tool_call:
            tool_call = choice.message.tool_calls[0]
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            details = f"Function: {func_name}, Args: {args}"
        else:
            details = f"No tool call, got content: '{choice.message.content[:50] if choice.message.content else 'None'}'"
        
        print_result("tool_calling", model, has_tool_call, details)
        return has_tool_call
        
    except Exception as e:
        print_result("tool_calling", model, False, str(e))
        return False


def test_parallel_tool_calls(model: str) -> bool:
    """
    Test parallel tool calls (multiple tool calls in one response).
    
    OpenAI Docs: https://platform.openai.com/docs/guides/function-calling/parallel-function-calling
    
    The model can return multiple tool calls when appropriate:
    "tool_calls": [
        {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}},
        {"id": "call_2", "function": {"name": "get_weather", "arguments": "{\"location\": \"London\"}"}}
    ]
    """
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
        }
    ]
    
    try:
        response = retry_on_rate_limit(
            completion,
            model=f"openai/{model}",
            messages=[{"role": "user", "content": "What's the weather in Paris and London?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=200,
        )
        
        choice = response.choices[0]
        tool_calls = choice.message.tool_calls or []
        num_calls = len(tool_calls)
        
        # Model may or may not choose to make parallel calls
        success = num_calls >= 1
        details = f"Got {num_calls} tool call(s)"
        if num_calls >= 2:
            details += " (parallel!)"
        
        print_result("parallel_tools", model, success, details)
        return success
        
    except Exception as e:
        print_result("parallel_tools", model, False, str(e))
        return False


# =============================================================================
# Structured Output Tests
# =============================================================================


def test_json_mode(model: str) -> bool:
    """
    Test JSON mode structured output.
    
    OpenAI Docs: https://platform.openai.com/docs/guides/structured-outputs/json-mode
    
    Request:
    {
        "response_format": {"type": "json_object"}
    }
    
    Note: You must include "JSON" in the prompt when using json_object mode.
    """
    try:
        response = retry_on_rate_limit(
            completion,
            model=f"openai/{model}",
            messages=[
                {
                    "role": "user",
                    "content": 'Return a JSON object with "name" (string) and "age" (integer). Example: {"name": "Alice", "age": 30}',
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=50,
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        success = isinstance(data, dict) and "name" in data
        print_result("json_mode", model, success, f"Parsed: {data}")
        return success
        
    except json.JSONDecodeError as e:
        print_result("json_mode", model, False, f"Invalid JSON: {e}")
        return False
    except Exception as e:
        print_result("json_mode", model, False, str(e))
        return False


def test_json_schema(model: str) -> bool:
    """
    Test JSON schema structured output (strict mode).
    
    OpenAI Docs: https://platform.openai.com/docs/guides/structured-outputs/structured-outputs
    
    Request:
    {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"],
                    "additionalProperties": false
                }
            }
        }
    }
    
    With strict: true, output is guaranteed to match the schema.
    """
    if model not in STRUCTURED_OUTPUT_MODELS:
        print_result("json_schema", model, True, "SKIP - not supported")
        return True
    
    try:
        response = retry_on_rate_limit(
            completion,
            model=f"openai/{model}",
            messages=[
                {"role": "user", "content": "Generate a person named Bob who is 25 years old."}
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
        
        success = isinstance(data, dict) and "name" in data and "age" in data
        print_result("json_schema", model, success, f"Parsed: {data}")
        return success
        
    except json.JSONDecodeError as e:
        print_result("json_schema", model, False, f"Invalid JSON: {e}")
        return False
    except Exception as e:
        print_result("json_schema", model, False, str(e))
        return False


# =============================================================================
# Embedding Tests
# =============================================================================


def test_embeddings(model: str) -> bool:
    """
    Test embedding generation.
    
    OpenAI Docs: https://platform.openai.com/docs/api-reference/embeddings/create
    
    Request:
    {
        "model": "text-embedding-3-small",
        "input": ["Hello, world!"]
    }
    
    Response:
    {
        "data": [{"embedding": [0.1, 0.2, ...], "index": 0}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 3, "total_tokens": 3}
    }
    """
    try:
        response = retry_on_rate_limit(
            embedding,
            model=f"openai/{model}",
            input=["Hello, world!", "How are you?"],
        )
        
        has_data = len(response.data) == 2
        has_embeddings = all(len(d.embedding) > 0 for d in response.data)
        has_usage = response.usage is not None
        
        dims = len(response.data[0].embedding) if response.data else 0
        success = has_data and has_embeddings and has_usage
        print_result("embeddings", model, success, f"Dimensions: {dims}")
        return success
        
    except Exception as e:
        print_result("embeddings", model, False, str(e))
        return False


def test_embedding_dimensions(model: str) -> bool:
    """
    Test embedding with custom dimensions (text-embedding-3 only).
    
    OpenAI Docs: https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions
    
    Request:
    {
        "model": "text-embedding-3-small",
        "input": "Hello",
        "dimensions": 256
    }
    
    Note: Only text-embedding-3-small and text-embedding-3-large support dimensions parameter.
    """
    if model == "text-embedding-ada-002":
        print_result("embedding_dims", model, True, "SKIP - not supported")
        return True
    
    try:
        response = retry_on_rate_limit(
            embedding,
            model=f"openai/{model}",
            input=["Hello"],
            dimensions=256,
        )
        
        dims = len(response.data[0].embedding)
        success = dims == 256
        print_result("embedding_dims", model, success, f"Requested 256, got {dims}")
        return success
        
    except Exception as e:
        print_result("embedding_dims", model, False, str(e))
        return False


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_invalid_model_error() -> bool:
    """
    Test that invalid model returns appropriate error.
    
    OpenAI Docs: https://platform.openai.com/docs/guides/error-codes
    
    Expected: 404 error for non-existent model
    """
    try:
        completion(
            model="openai/nonexistent-model-xyz123",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        print_result("invalid_model", "N/A", False, "Should have raised error")
        return False
    except ArcLLMError as e:
        success = "404" in str(e) or "not found" in str(e).lower() or "does not exist" in str(e).lower()
        print_result("invalid_model", "N/A", success, f"Got expected error: {type(e).__name__}")
        return success
    except Exception as e:
        print_result("invalid_model", "N/A", False, f"Unexpected error: {e}")
        return False


def test_invalid_api_key() -> bool:
    """
    Test that invalid API key returns authentication error.
    
    OpenAI Docs: https://platform.openai.com/docs/guides/error-codes
    
    Expected: 401 Unauthorized
    """
    try:
        completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            api_key="sk-invalid-key-for-testing",
        )
        print_result("invalid_api_key", "N/A", False, "Should have raised error")
        return False
    except AuthenticationError:
        print_result("invalid_api_key", "N/A", True, "Got AuthenticationError as expected")
        return True
    except ArcLLMError as e:
        # Some errors may be wrapped differently
        success = "401" in str(e) or "auth" in str(e).lower() or "invalid" in str(e).lower()
        print_result("invalid_api_key", "N/A", success, f"Got: {type(e).__name__}")
        return success
    except Exception as e:
        print_result("invalid_api_key", "N/A", False, f"Unexpected error: {e}")
        return False


# =============================================================================
# Cost Calculation Tests
# =============================================================================


def test_cost_calculation(model: str) -> bool:
    """
    Test that cost calculation works for response.
    
    Uses arcllm.completion_cost() to calculate cost based on pricing tables.
    """
    try:
        response = retry_on_rate_limit(
            completion,
            model=f"openai/{model}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        
        cost = arcllm.completion_cost(response)
        success = cost > 0
        print_result("cost_calc", model, success, f"Cost: ${cost:.8f}")
        return success
        
    except Exception as e:
        print_result("cost_calc", model, False, str(e))
        return False


# =============================================================================
# Main Test Runner
# =============================================================================


def run_all_tests(api_key: str) -> dict[str, Any]:
    """Run all tests and return results summary."""
    os.environ["OPENAI_API_KEY"] = api_key
    
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "details": [],
    }
    
    # Basic Completion Tests
    print_section("Basic Completion Tests")
    for model in CHAT_MODELS:
        if test_basic_completion(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)  # Rate limit protection
    
    # Multi-turn Conversation
    print_section("Multi-turn Conversation Tests")
    for model in CHAT_MODELS:
        if test_multi_turn_conversation(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)
    
    # Streaming Tests
    print_section("Streaming Tests")
    for model in CHAT_MODELS:
        if test_streaming(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)
        
        if test_streaming_with_usage(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)
    
    # Tool Calling Tests
    print_section("Tool Calling Tests")
    for model in TOOL_CALLING_MODELS:
        if test_tool_calling(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)
        
        if test_parallel_tool_calls(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)
    
    # Structured Output Tests
    print_section("Structured Output Tests")
    for model in CHAT_MODELS:
        if test_json_mode(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)
        
        if test_json_schema(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)
    
    # Embedding Tests
    print_section("Embedding Tests")
    for model in EMBEDDING_MODELS:
        if test_embeddings(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)
        
        if test_embedding_dimensions(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)
    
    # Error Handling Tests
    print_section("Error Handling Tests")
    if test_invalid_model_error():
        results["passed"] += 1
    else:
        results["failed"] += 1
    results["total"] += 1
    
    if test_invalid_api_key():
        results["passed"] += 1
    else:
        results["failed"] += 1
    results["total"] += 1
    
    # Cost Calculation Tests
    print_section("Cost Calculation Tests")
    for model in CHAT_MODELS[:2]:  # Just test a couple
        if test_cost_calculation(model):
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
        time.sleep(0.5)
    
    return results


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print(" arcllm OpenAI Live API Tests")
    print("=" * 60)
    
    api_key = OPENAI_API_KEY
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
        return
    
    print(f"\nAPI Key: {api_key[:20]}...{api_key[-4:]}")
    print(f"Models to test: {len(CHAT_MODELS)} chat, {len(EMBEDDING_MODELS)} embedding")
    
    results = run_all_tests(api_key)
    
    # Print summary
    print_section("Test Summary")
    print(f"  Total:  {results['total']}")
    print(f"  Passed: {results['passed']} ✅")
    print(f"  Failed: {results['failed']} ❌")
    print(f"  Rate:   {100 * results['passed'] / results['total']:.1f}%")
    
    if results["failed"] == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {results['failed']} test(s) failed")


if __name__ == "__main__":
    main()
