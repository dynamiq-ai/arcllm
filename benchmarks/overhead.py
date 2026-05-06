#!/usr/bin/env python3
"""
Mocked microbenchmarks for arcllm overhead.

Measures the SDK overhead on top of the network — no live API calls.
Each microbenchmark exercises one hot path:

    - import time (cold start)
    - type creation (msgspec.Struct allocations)
    - response parsing (JSON -> ModelResponse)
    - request building (kwargs -> wire body)
    - SSE event parsing
    - model-string parsing
    - cost calculation lookup
    - full mocked completion (round-trip with mocked HTTP)
    - stream-chunk builder

Usage:
    python benchmarks/overhead.py             # human-readable report
    python benchmarks/overhead.py --json      # stable JSON to stdout
    python benchmarks/overhead.py --ci        # enforce regression gates;
                                              # exits non-zero on regression
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from unittest.mock import patch

# Fixtures for mocked responses
MOCK_COMPLETION_RESPONSE = {
    "id": "chatcmpl-benchmark",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a benchmark test response with some content to parse.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}

MOCK_STREAMING_CHUNKS = [
    b'data: {"id":"1","choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{"content":"!"},"index":0}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
    b"data: [DONE]\n\n",
]


def benchmark_import_time():
    """Benchmark import time."""
    import subprocess
    import sys

    # Time importing arcllm
    code = "import arcllm"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import time; s=time.perf_counter(); {code}; print(time.perf_counter()-s)",
        ],
        capture_output=True,
        text=True,
    )
    import_time = float(result.stdout.strip())
    print(f"Import time: {import_time * 1000:.2f}ms")
    return import_time


def benchmark_type_creation(iterations: int = 10000):
    """Benchmark creating response types."""
    from arcllm.types import (
        Choice,
        Message,
        ModelResponse,
        Usage,
    )

    start = time.perf_counter()
    for _ in range(iterations):
        msg = Message(role="assistant", content="Hello world")
        choice = Choice(index=0, message=msg, finish_reason="stop")
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        ModelResponse(
            id="test-123",
            model="gpt-4o-mini",
            choices=[choice],
            usage=usage,
        )
    elapsed = time.perf_counter() - start

    print(
        f"Type creation ({iterations} iterations): {elapsed * 1000:.2f}ms ({elapsed / iterations * 1000000:.2f}µs/iter)"
    )
    return elapsed


def benchmark_response_parsing(iterations: int = 10000):
    """Benchmark parsing JSON response."""
    from arcllm.providers.base import ProviderConfig
    from arcllm.providers.openai_adapter import OpenAIAdapter

    config = ProviderConfig(api_key="test-key")
    adapter = OpenAIAdapter(config)
    response_bytes = json.dumps(MOCK_COMPLETION_RESPONSE).encode("utf-8")

    start = time.perf_counter()
    for _ in range(iterations):
        adapter.parse_response(response_bytes, "gpt-4o-mini")
    elapsed = time.perf_counter() - start

    print(
        f"Response parsing ({iterations} iterations): {elapsed * 1000:.2f}ms ({elapsed / iterations * 1000000:.2f}µs/iter)"
    )
    return elapsed


def benchmark_sse_parsing(iterations: int = 10000):
    """Benchmark SSE parsing."""
    from arcllm.http.sse import SSEParser

    # Create chunk data
    chunk_data = b"".join(MOCK_STREAMING_CHUNKS)

    start = time.perf_counter()
    for _ in range(iterations):
        parser = SSEParser()
        list(parser.feed(chunk_data))
    elapsed = time.perf_counter() - start

    print(
        f"SSE parsing ({iterations} iterations): {elapsed * 1000:.2f}ms ({elapsed / iterations * 1000000:.2f}µs/iter)"
    )
    return elapsed


def benchmark_request_building(iterations: int = 10000):
    """Benchmark building requests."""
    from arcllm.providers.base import ProviderConfig
    from arcllm.providers.openai_adapter import OpenAIAdapter

    config = ProviderConfig(api_key="test-key")
    adapter = OpenAIAdapter(config)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]

    start = time.perf_counter()
    for _ in range(iterations):
        adapter.build_request(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )
    elapsed = time.perf_counter() - start

    print(
        f"Request building ({iterations} iterations): {elapsed * 1000:.2f}ms ({elapsed / iterations * 1000000:.2f}µs/iter)"
    )
    return elapsed


def benchmark_model_string_parsing(iterations: int = 100000):
    """Benchmark parsing model strings."""
    from arcllm.providers.base import parse_model_string

    test_strings = [
        "gpt-4o-mini",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-latest",
        "groq/llama-3.1-70b-versatile",
        "unknown-model",
    ]

    start = time.perf_counter()
    for _ in range(iterations):
        for s in test_strings:
            parse_model_string(s)
    elapsed = time.perf_counter() - start

    total_ops = iterations * len(test_strings)
    print(
        f"Model string parsing ({total_ops} ops): {elapsed * 1000:.2f}ms ({elapsed / total_ops * 1000000:.2f}µs/op)"
    )
    return elapsed


def benchmark_cost_calculation(iterations: int = 100000):
    """Benchmark cost calculation."""
    from arcllm.pricing import cost_per_token

    start = time.perf_counter()
    for _ in range(iterations):
        cost_per_token("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    elapsed = time.perf_counter() - start

    print(
        f"Cost calculation ({iterations} iterations): {elapsed * 1000:.2f}ms ({elapsed / iterations * 1000000:.2f}µs/iter)"
    )
    return elapsed


def benchmark_full_completion_mocked(iterations: int = 1000):
    """Benchmark full completion flow with mocked HTTP."""
    import arcllm
    from arcllm.http.client import HTTPResponse

    # Mock the HTTP client
    mock_response = HTTPResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        body=json.dumps(MOCK_COMPLETION_RESPONSE).encode("utf-8"),
    )

    with patch("arcllm.core._get_http_client") as mock_client:
        mock_client.return_value.request.return_value = mock_response

        messages = [{"role": "user", "content": "Hello!"}]

        start = time.perf_counter()
        for _ in range(iterations):
            arcllm.completion(
                model="gpt-4o-mini",
                messages=messages,
                api_key="test-key",
            )
        elapsed = time.perf_counter() - start

    print(
        f"Full completion (mocked, {iterations} iterations): {elapsed * 1000:.2f}ms ({elapsed / iterations * 1000:.2f}ms/iter)"
    )
    return elapsed


def benchmark_stream_chunk_builder(iterations: int = 1000):
    """Benchmark stream_chunk_builder."""
    from arcllm import stream_chunk_builder
    from arcllm.providers.base import ProviderConfig
    from arcllm.providers.openai_adapter import OpenAIAdapter

    config = ProviderConfig(api_key="test-key")
    adapter = OpenAIAdapter(config)

    # Parse mock streaming chunks
    chunks = []
    for chunk_data in MOCK_STREAMING_CHUNKS:
        data = chunk_data.decode("utf-8")
        if "data: " in data:
            json_str = data.replace("data: ", "").strip()
            if json_str and json_str != "[DONE]":
                chunk = adapter.parse_stream_event(json_str, "gpt-4o-mini")
                if chunk:
                    chunks.append(chunk)

    start = time.perf_counter()
    for _ in range(iterations):
        stream_chunk_builder(chunks)
    elapsed = time.perf_counter() - start

    print(
        f"Stream chunk builder ({iterations} iterations): {elapsed * 1000:.2f}ms ({elapsed / iterations * 1000:.2f}ms/iter)"
    )
    return elapsed


# Regression gates for ``--ci``. Generous enough to absorb cloud-runner jitter
# (numbers are an order of magnitude above what a healthy laptop sees).
_CI_LIMITS = {
    "import_ms": 800.0,
    "type_creation_us_per_iter": 50.0,
    "response_parsing_us_per_iter": 100.0,
    "request_building_us_per_iter": 100.0,
    "sse_parsing_us_per_iter": 50.0,
    "model_parsing_us_per_iter": 5.0,
    "cost_calculation_us_per_iter": 20.0,
    "full_completion_ms_per_iter": 5.0,
    "stream_builder_ms_per_iter": 1.0,
}


def _run_all() -> dict[str, float]:
    """Run every microbenchmark and return a stable result dict."""
    results: dict[str, float] = {}
    print("--- Startup ---")
    results["import_s"] = benchmark_import_time()
    print("\n--- Core operations ---")
    results["type_creation_s"] = benchmark_type_creation()
    results["response_parsing_s"] = benchmark_response_parsing()
    results["request_building_s"] = benchmark_request_building()
    results["sse_parsing_s"] = benchmark_sse_parsing()
    print("\n--- Utility ---")
    results["model_parsing_s"] = benchmark_model_string_parsing()
    results["cost_calculation_s"] = benchmark_cost_calculation()
    print("\n--- Integration (mocked HTTP) ---")
    results["full_completion_s"] = benchmark_full_completion_mocked()
    results["stream_builder_s"] = benchmark_stream_chunk_builder()
    return results


def _collect(*, json_mode: bool) -> dict[str, float]:
    """Run microbenchmarks. In ``--json`` mode swallow human-readable prints
    so the only thing on stdout is the JSON document."""
    if json_mode:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return _run_all()

    print("=" * 60)
    print("arcllm overhead microbenchmarks (mocked HTTP)")
    print("=" * 60)
    print()
    return _run_all()


def _ci_check(results: dict[str, float]) -> list[str]:
    """Return a list of human-readable regression messages (empty if OK)."""
    failures: list[str] = []

    def check(label: str, actual_ms_or_us: float, limit: float) -> None:
        if actual_ms_or_us > limit:
            failures.append(f"{label}: {actual_ms_or_us:.2f} > {limit:.2f}")

    check("import_ms", results["import_s"] * 1000, _CI_LIMITS["import_ms"])
    check(
        "type_creation_us_per_iter",
        results["type_creation_s"] / 10000 * 1_000_000,
        _CI_LIMITS["type_creation_us_per_iter"],
    )
    check(
        "response_parsing_us_per_iter",
        results["response_parsing_s"] / 10000 * 1_000_000,
        _CI_LIMITS["response_parsing_us_per_iter"],
    )
    check(
        "request_building_us_per_iter",
        results["request_building_s"] / 10000 * 1_000_000,
        _CI_LIMITS["request_building_us_per_iter"],
    )
    check(
        "sse_parsing_us_per_iter",
        results["sse_parsing_s"] / 10000 * 1_000_000,
        _CI_LIMITS["sse_parsing_us_per_iter"],
    )
    check(
        "model_parsing_us_per_iter",
        results["model_parsing_s"] / 100000 * 1_000_000,
        _CI_LIMITS["model_parsing_us_per_iter"],
    )
    check(
        "cost_calculation_us_per_iter",
        results["cost_calculation_s"] / 100000 * 1_000_000,
        _CI_LIMITS["cost_calculation_us_per_iter"],
    )
    check(
        "full_completion_ms_per_iter",
        results["full_completion_s"] / 1000 * 1000,
        _CI_LIMITS["full_completion_ms_per_iter"],
    )
    check(
        "stream_builder_ms_per_iter",
        results["stream_builder_s"] / 1000 * 1000,
        _CI_LIMITS["stream_builder_ms_per_iter"],
    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON document on stdout instead of human-readable text.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Enforce regression gates; exit 1 on any failure.",
    )
    args = parser.parse_args()

    results = _collect(json_mode=args.json)

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")

    if args.ci:
        failures = _ci_check(results)
        if failures:
            sys.stderr.write("Regression gate failures:\n")
            for line in failures:
                sys.stderr.write(f"  - {line}\n")
            return 1
        if not args.json:
            print("\nAll regression gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
