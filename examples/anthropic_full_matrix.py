#!/usr/bin/env python3
"""
Comprehensive Anthropic Model Testing Script

This script tests all available Anthropic models to verify arcllm works correctly
with each one. It tests:
- Basic completion
- Streaming completion
- Tool/function calling
- System prompt handling
- Vision capabilities (where supported)
- Token usage reporting

Run with:
    ANTHROPIC_API_KEY="sk-ant-..." python examples/test_anthropic_models.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import arcllm
from arcllm.exceptions import ArcLLMError, RateLimitError


@dataclass
class ModelTestConfig:
    """Configuration for testing a specific model."""

    model_id: str
    supports_vision: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True
    max_tokens_default: int = 100
    is_legacy: bool = False


# All Anthropic models to test - organized by tier
ANTHROPIC_MODELS = [
    # Claude 4.5 Series (current flagship - November 2025)
    ModelTestConfig("claude-4-5-opus-20251120", supports_vision=True, supports_tools=True),
    ModelTestConfig("claude-4-5-opus-latest", supports_vision=True, supports_tools=True),
    ModelTestConfig("claude-4-5-sonnet-20251015", supports_vision=True, supports_tools=True),
    ModelTestConfig("claude-4-5-sonnet-latest", supports_vision=True, supports_tools=True),
    ModelTestConfig("claude-4-5-haiku-20251201", supports_vision=True, supports_tools=True),
    ModelTestConfig("claude-4-5-haiku-latest", supports_vision=True, supports_tools=True),
    # Claude 4 Series (June 2025)
    ModelTestConfig("claude-4-opus-20250615", supports_vision=True, supports_tools=True),
    ModelTestConfig("claude-4-opus-latest", supports_vision=True, supports_tools=True),
    ModelTestConfig("claude-4-sonnet-20250601", supports_vision=True, supports_tools=True),
    ModelTestConfig("claude-4-sonnet-latest", supports_vision=True, supports_tools=True),
    ModelTestConfig("claude-4-haiku-20250701", supports_vision=True, supports_tools=True),
    ModelTestConfig("claude-4-haiku-latest", supports_vision=True, supports_tools=True),
    # Claude 3.5 Series (DEPRECATED - retiring March 2026)
    ModelTestConfig(
        "claude-3-5-sonnet-20241022", supports_vision=True, supports_tools=True, is_legacy=True
    ),
    ModelTestConfig(
        "claude-3-5-sonnet-latest", supports_vision=True, supports_tools=True, is_legacy=True
    ),
    ModelTestConfig(
        "claude-3-5-haiku-20241022", supports_vision=True, supports_tools=True, is_legacy=True
    ),
    ModelTestConfig(
        "claude-3-5-haiku-latest", supports_vision=True, supports_tools=True, is_legacy=True
    ),
    # Claude 3 Series (DEPRECATED)
    ModelTestConfig(
        "claude-3-opus-20240229", supports_vision=True, supports_tools=True, is_legacy=True
    ),
    ModelTestConfig(
        "claude-3-opus-latest", supports_vision=True, supports_tools=True, is_legacy=True
    ),
    ModelTestConfig(
        "claude-3-sonnet-20240229", supports_vision=True, supports_tools=True, is_legacy=True
    ),
    ModelTestConfig(
        "claude-3-haiku-20240307", supports_vision=True, supports_tools=True, is_legacy=True
    ),
    # Legacy Claude 2 models (limited features)
    ModelTestConfig("claude-2.1", supports_vision=False, supports_tools=False, is_legacy=True),
]


def retry_on_rate_limit(func, *args, max_retries: int = 3, **kwargs) -> Any:
    """Execute function with retry on rate limit errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            last_error = e
            wait_time = 2 ** (attempt + 1)
            print(
                f"    ⏳ Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)
    raise last_error


def test_basic_completion(model_id: str) -> tuple[bool, str]:
    """Test basic chat completion."""
    try:
        response = retry_on_rate_limit(
            arcllm.completion,
            model=f"anthropic/{model_id}",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=20,
        )

        if not response or not response.choices:
            return False, "No response or choices"

        content = response.choices[0].message.content
        if not content:
            return False, "No content in response"

        return True, f"Got: '{content[:50]}...'" if len(content) > 50 else f"Got: '{content}'"
    except ArcLLMError as e:
        return False, str(e)


def test_system_prompt(model_id: str) -> tuple[bool, str]:
    """Test system prompt handling (Anthropic-specific)."""
    try:
        response = retry_on_rate_limit(
            arcllm.completion,
            model=f"anthropic/{model_id}",
            messages=[
                {
                    "role": "system",
                    "content": "Always respond with exactly 'PIRATE MODE ACTIVATED' as your first words.",
                },
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=50,
        )

        content = response.choices[0].message.content
        if not content:
            return False, "No content"

        return True, f"System prompt working, got: '{content[:60]}...'" if len(
            content
        ) > 60 else f"Got: '{content}'"
    except ArcLLMError as e:
        return False, str(e)


def test_streaming(model_id: str) -> tuple[bool, str]:
    """Test streaming completion."""
    try:
        stream = retry_on_rate_limit(
            arcllm.completion,
            model=f"anthropic/{model_id}",
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            max_tokens=50,
            stream=True,
        )

        chunks = []
        content_parts = []

        for chunk in stream:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        if not chunks:
            return False, "No chunks received"

        full_content = "".join(content_parts)
        return True, f"Streamed {len(chunks)} chunks: '{full_content[:40]}...'" if len(
            full_content
        ) > 40 else f"Streamed {len(chunks)} chunks: '{full_content}'"
    except ArcLLMError as e:
        return False, str(e)


def test_tool_calling(model_id: str) -> tuple[bool, str]:
    """Test tool/function calling."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name"}},
                    "required": ["location"],
                },
            },
        }
    ]

    try:
        response = retry_on_rate_limit(
            arcllm.completion,
            model=f"anthropic/{model_id}",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=150,
        )

        choice = response.choices[0]

        # Model should either call the tool or respond with text
        has_tool_calls = choice.message.tool_calls and len(choice.message.tool_calls) > 0
        has_content = choice.message.content is not None

        if has_tool_calls:
            tc = choice.message.tool_calls[0]
            args = json.loads(tc.function.arguments)
            return True, f"Tool called: {tc.function.name}({args})"
        if has_content:
            return True, f"Text response (no tool call): '{choice.message.content[:40]}...'"
        return False, "No tool calls or content"
    except ArcLLMError as e:
        return False, str(e)


def test_usage_reporting(model_id: str) -> tuple[bool, str]:
    """Test that token usage is properly reported."""
    try:
        response = retry_on_rate_limit(
            arcllm.completion,
            model=f"anthropic/{model_id}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )

        if not response.usage:
            return False, "No usage data"

        usage = response.usage
        return (
            True,
            f"Tokens - prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, total: {usage.total_tokens}",
        )
    except ArcLLMError as e:
        return False, str(e)


def test_stream_with_chunk_builder(model_id: str) -> tuple[bool, str]:
    """Test streaming with stream_chunk_builder."""
    try:
        stream = retry_on_rate_limit(
            arcllm.completion,
            model=f"anthropic/{model_id}",
            messages=[{"role": "user", "content": "Say 'test'."}],
            max_tokens=20,
            stream=True,
        )

        chunks = list(stream)

        if not chunks:
            return False, "No chunks"

        # Build final response from chunks
        final = arcllm.stream_chunk_builder(chunks)

        if not final.choices or not final.choices[0].message.content:
            return False, "Failed to build response"

        return True, f"Built response: '{final.choices[0].message.content}'"
    except ArcLLMError as e:
        return False, str(e)


def test_pricing(model_id: str) -> tuple[bool, str]:
    """Test that pricing is available for the model."""
    try:
        response = retry_on_rate_limit(
            arcllm.completion,
            model=f"anthropic/{model_id}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        cost = arcllm.completion_cost(response)
        return True, f"Cost: ${cost:.8f}"
    except Exception as e:
        return False, str(e)


def run_model_tests(config: ModelTestConfig, verbose: bool = True) -> dict[str, tuple[bool, str]]:
    """Run all applicable tests for a model."""
    results = {}

    print(f"\n{'=' * 60}")
    print(f"Testing: {config.model_id}")
    print(f"{'=' * 60}")

    # Basic tests
    tests = [
        ("Basic Completion", test_basic_completion),
        ("System Prompt", test_system_prompt),
        ("Usage Reporting", test_usage_reporting),
        ("Pricing", test_pricing),
    ]

    # Streaming (supported by all Anthropic models)
    if config.supports_streaming:
        tests.append(("Streaming", test_streaming))
        tests.append(("Stream Chunk Builder", test_stream_with_chunk_builder))

    # Tool calling (Claude 3+ only)
    if config.supports_tools:
        tests.append(("Tool Calling", test_tool_calling))

    for test_name, test_func in tests:
        print(f"  Testing {test_name}...", end=" ")
        success, message = test_func(config.model_id)
        results[test_name] = (success, message)

        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")

        # Small delay between tests to avoid rate limits
        time.sleep(0.5)

    return results


def print_summary(all_results: dict[str, dict[str, tuple[bool, str]]]) -> None:
    """Print a summary of all test results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_tests = 0
    total_passed = 0

    for model_id, results in all_results.items():
        passed = sum(1 for success, _ in results.values() if success)
        total = len(results)
        total_tests += total
        total_passed += passed

        status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
        print(f"{status} {model_id}: {passed}/{total} tests passed")

    print("-" * 80)
    print(
        f"TOTAL: {total_passed}/{total_tests} tests passed ({100 * total_passed / total_tests:.1f}%)"
    )


def main():
    """Run comprehensive Anthropic model tests."""
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("   Set it with: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    print("🔬 Anthropic Model Testing Suite")
    print(f"   API Key: {api_key[:15]}...{api_key[-4:]}")
    print(f"   Models to test: {len(ANTHROPIC_MODELS)}")

    # Select which models to test
    # By default, test only the main/latest models to save costs
    test_models = [
        m
        for m in ANTHROPIC_MODELS
        if "latest" in m.model_id
        or m.model_id
        in [
            "claude-4-5-haiku-20251201",  # Current fast model
            "claude-4-5-sonnet-20251015",  # Current balanced model
            "claude-3-5-haiku-20241022",  # Legacy (still commonly used)
        ]
    ]

    # For comprehensive testing, uncomment below:
    # test_models = ANTHROPIC_MODELS

    print(f"\n   Testing {len(test_models)} models (use --all for all models)")

    # Check command line args
    if "--all" in sys.argv:
        test_models = ANTHROPIC_MODELS
        print(f"   Full test: {len(test_models)} models")

    all_results = {}

    for config in test_models:
        try:
            results = run_model_tests(config)
            all_results[config.model_id] = results
        except Exception as e:
            print(f"\n❌ Fatal error testing {config.model_id}: {e}")
            all_results[config.model_id] = {"Fatal Error": (False, str(e))}

        # Delay between models
        time.sleep(1)

    # Print summary
    print_summary(all_results)

    # Return exit code based on results
    all_passed = all(
        all(success for success, _ in results.values()) for results in all_results.values()
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
