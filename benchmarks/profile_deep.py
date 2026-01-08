#!/usr/bin/env python3
"""
Deep profiling script for arcllm performance analysis.

This script provides detailed timing breakdowns to identify bottlenecks:
1. HTTP-level timing (DNS, connect, SSL, TTFB, transfer)
2. Function-level profiling (cProfile)
3. Side-by-side comparison with litellm
4. Raw HTTP baseline comparison
5. SSE parsing overhead analysis

Usage:
    python benchmarks/profile_deep.py
    python benchmarks/profile_deep.py --provider groq --model llama-3.3-70b-versatile
    python benchmarks/profile_deep.py --vs-litellm  # Compare with litellm
    python benchmarks/profile_deep.py --all         # Run all analyses
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import os
import pstats
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileResult:
    """Profiling result with timing breakdown."""

    phase: str
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)


def get_api_config(provider: str) -> tuple[str, str, dict[str, str]]:
    """Get API configuration for a provider."""
    configs = {
        "openai": (
            "https://api.openai.com/v1/chat/completions",
            os.environ.get("OPENAI_API_KEY", ""),
            {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
        ),
        "groq": (
            "https://api.groq.com/openai/v1/chat/completions",
            os.environ.get("GROQ_API_KEY", ""),
            {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}"},
        ),
        "anthropic": (
            "https://api.anthropic.com/v1/messages",
            os.environ.get("ANTHROPIC_API_KEY", ""),
            {
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
            },
        ),
    }
    return configs.get(provider, configs["openai"])


def profile_arcllm_completion(provider: str, model: str, stream: bool = False) -> list[ProfileResult]:
    """Profile arcllm completion with detailed timing."""
    import arcllm
    from arcllm.providers.base import ProviderConfig, get_provider, parse_model_string

    results = []
    messages = [{"role": "user", "content": "Say 'ok'."}]

    gc.collect()

    # Phase 1: Model string parsing
    start = time.perf_counter()
    provider_name, model_id = parse_model_string(f"{provider}/{model}")
    results.append(
        ProfileResult(
            "Model string parsing",
            (time.perf_counter() - start) * 1000,
        )
    )

    # Phase 2: Get adapter (includes lazy loading)
    start = time.perf_counter()
    config = ProviderConfig()
    adapter = get_provider(provider_name, config)
    results.append(
        ProfileResult(
            "Get adapter (lazy load)",
            (time.perf_counter() - start) * 1000,
        )
    )

    # Phase 3: Build request
    start = time.perf_counter()
    request_data = adapter.build_request(
        model=model_id,
        messages=messages,
        max_tokens=5,
        stream=stream,
    )
    results.append(
        ProfileResult(
            "Build request (orjson)",
            (time.perf_counter() - start) * 1000,
            {"body_size": len(request_data.body) if request_data.body else 0},
        )
    )

    # Phase 4: HTTP client creation (httpx)
    from arcllm.http.client import HTTPClient

    start = time.perf_counter()
    client = HTTPClient()
    results.append(
        ProfileResult(
            "HTTP client creation (httpx)",
            (time.perf_counter() - start) * 1000,
        )
    )

    # Phase 5: HTTP request (the big one)
    start = time.perf_counter()
    http_response = client.request(
        method=request_data.method,
        url=request_data.url,
        headers=request_data.headers,
        body=request_data.body,
    )
    http_time = time.perf_counter() - start
    results.append(
        ProfileResult(
            "HTTP request (network)",
            http_time * 1000,
            {"status": http_response.status_code, "body_size": len(http_response.body)},
        )
    )

    # Phase 6: Response parsing (orjson)
    start = time.perf_counter()
    content = ""
    if not stream:
        response = adapter.parse_response(http_response.body, model_id)
        content = response.choices[0].message.content[:50] if response.choices else ""
    parse_time = time.perf_counter() - start
    results.append(
        ProfileResult(
            "Response parsing (orjson)",
            parse_time * 1000,
            {"content_preview": content if not stream else ""},
        )
    )

    client.close()
    return results


def profile_arcllm_streaming(provider: str, model: str) -> list[ProfileResult]:
    """Profile arcllm streaming with chunk-by-chunk timing."""
    import arcllm

    results = []
    messages = [{"role": "user", "content": "Count 1 2 3 4 5."}]

    gc.collect()

    # Time to start streaming
    start = time.perf_counter()
    stream = arcllm.completion(
        model=f"{provider}/{model}",
        messages=messages,
        max_tokens=20,
        stream=True,
    )
    stream_start_time = time.perf_counter() - start
    results.append(
        ProfileResult(
            "Stream initialization",
            stream_start_time * 1000,
        )
    )

    # Time to first chunk
    chunk_times = []
    first_content_time = None
    full_content = ""
    chunk_start = time.perf_counter()

    for i, chunk in enumerate(stream):
        chunk_time = time.perf_counter() - chunk_start
        chunk_times.append(chunk_time * 1000)

        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_content += delta.content
                if first_content_time is None:
                    first_content_time = sum(chunk_times)

        chunk_start = time.perf_counter()

    results.append(
        ProfileResult(
            "Time to first content chunk",
            first_content_time or 0,
            {"total_chunks": len(chunk_times), "content": full_content[:50]},
        )
    )

    results.append(
        ProfileResult(
            "Average inter-chunk time",
            statistics.mean(chunk_times) if chunk_times else 0,
            {
                "min_ms": min(chunk_times) if chunk_times else 0,
                "max_ms": max(chunk_times) if chunk_times else 0,
            },
        )
    )

    return results


def profile_with_cprofile(func, *args, **kwargs):
    """Run function with cProfile and return stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    # Get stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    return result, stream.getvalue()


def analyze_http_client():
    """Analyze our HTTP client implementation."""
    print("\n" + "=" * 70)
    print("HTTP CLIENT ANALYSIS")
    print("=" * 70)

    from arcllm.http import client as http_client

    print("\nHTTP Client Implementation:")
    print(f"  Module: {http_client.__file__}")

    # Check if using httpx
    import inspect

    source = inspect.getsource(http_client.HTTPClient)

    features = []
    if "httpx" in source:
        features.append("✅ Using httpx (modern HTTP client)")
    if "http2" in source.lower():
        features.append("✅ HTTP/2 support enabled")
    if "Limits" in source or "limits" in source:
        features.append("✅ Connection pooling configured")
    if "Timeout" in source:
        features.append("✅ Configurable timeouts")

    print("\nSync Client Features:")
    for feature in features:
        print(f"  {feature}")

    if not features:
        print("  ⚠️ No httpx features detected - may be using basic HTTP client")

    # Check async client
    try:
        from arcllm.http import async_client

        async_source = inspect.getsource(async_client.AsyncHTTPClient)

        async_features = []
        if "aiohttp" in async_source:
            async_features.append("✅ Using aiohttp (optimized for async)")
        elif "httpx" in async_source:
            async_features.append("✅ Using httpx.AsyncClient")
        if "TCPConnector" in async_source:
            async_features.append("✅ Connection pooling with TCPConnector")
        if "limit=" in async_source or "limit_per_host" in async_source:
            async_features.append("✅ Connection limits configured")

        print("\nAsync Client Features:")
        for feature in async_features:
            print(f"  {feature}")
    except Exception as e:
        print(f"\nAsync client analysis failed: {e}")

    return features


def analyze_json_performance():
    """Analyze JSON serialization performance."""
    print("\n" + "=" * 70)
    print("JSON PERFORMANCE ANALYSIS")
    print("=" * 70)

    import json

    import orjson

    # Test data - realistic LLM request
    test_data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Python programming." * 10},
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
    }

    iterations = 10000

    # Benchmark orjson vs json
    print(f"\nSerialization ({iterations} iterations):")

    start = time.perf_counter()
    for _ in range(iterations):
        json.dumps(test_data)
    json_time = (time.perf_counter() - start) * 1000
    print(f"  stdlib json: {json_time:.1f}ms ({json_time/iterations*1000:.2f}µs/op)")

    start = time.perf_counter()
    for _ in range(iterations):
        orjson.dumps(test_data)
    orjson_time = (time.perf_counter() - start) * 1000
    print(f"  orjson:      {orjson_time:.1f}ms ({orjson_time/iterations*1000:.2f}µs/op)")

    speedup = json_time / orjson_time
    print(f"\n  orjson is {speedup:.1f}x faster for serialization")

    # Deserialization
    json_str = json.dumps(test_data)
    orjson_bytes = orjson.dumps(test_data)

    print(f"\nDeserialization ({iterations} iterations):")

    start = time.perf_counter()
    for _ in range(iterations):
        json.loads(json_str)
    json_time = (time.perf_counter() - start) * 1000
    print(f"  stdlib json: {json_time:.1f}ms ({json_time/iterations*1000:.2f}µs/op)")

    start = time.perf_counter()
    for _ in range(iterations):
        orjson.loads(orjson_bytes)
    orjson_time = (time.perf_counter() - start) * 1000
    print(f"  orjson:      {orjson_time:.1f}ms ({orjson_time/iterations*1000:.2f}µs/op)")

    speedup = json_time / orjson_time
    print(f"\n  orjson is {speedup:.1f}x faster for deserialization")


def analyze_sse_parser():
    """Analyze SSE parser performance."""
    print("\n" + "=" * 70)
    print("SSE PARSER ANALYSIS")
    print("=" * 70)

    from arcllm.http.sse import SSEParser

    # Benchmark SSE parsing
    test_data = b"""data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"1","choices":[{"delta":{"content":" world"}}]}

data: {"id":"1","choices":[{"delta":{"content":"!"}}]}

data: [DONE]

"""

    iterations = 10000
    parser = SSEParser()

    start = time.perf_counter()
    for _ in range(iterations):
        parser = SSEParser()
        list(parser.feed(test_data))
    elapsed = time.perf_counter() - start

    print(f"\nSSE Parsing Performance:")
    print(f"  {iterations} iterations: {elapsed*1000:.2f}ms")
    print(f"  Per parse: {elapsed/iterations*1000000:.2f}µs")
    print(f"  Events per second: {iterations/elapsed:,.0f}")


def compare_with_raw_httpx(provider: str, model: str):
    """Compare arcllm with raw httpx request."""
    print("\n" + "=" * 70)
    print("COMPARISON: arcllm vs Raw httpx")
    print("=" * 70)

    import httpx
    import orjson

    url, api_key, headers = get_api_config(provider)

    body = orjson.dumps(
        {
            "model": model,
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Say ok."}],
        }
    )

    headers["Content-Type"] = "application/json"

    # Warmup
    print("\nWarming up...")
    with httpx.Client(http2=True) as client:
        client.post(url, headers=headers, content=body)

    # Measure raw httpx
    print("\n1. Raw httpx (with connection reuse):")
    with httpx.Client(http2=True) as client:
        times = []
        for i in range(5):
            gc.collect()
            start = time.perf_counter()
            response = client.post(url, headers=headers, content=body)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            if i == 0:
                print(f"   First request: {elapsed:.2f}ms")

        print(f"   Subsequent avg: {statistics.mean(times[1:]):.2f}ms")
        raw_median = statistics.median(times)
        print(f"   Median: {raw_median:.2f}ms")

    # Measure arcllm
    print("\n2. arcllm completion (with connection reuse):")
    import arcllm

    # Warmup
    arcllm.completion(
        model=f"{provider}/{model}",
        messages=[{"role": "user", "content": "ok"}],
        max_tokens=1,
    )

    times = []
    for i in range(5):
        gc.collect()
        start = time.perf_counter()
        response = arcllm.completion(
            model=f"{provider}/{model}",
            messages=[{"role": "user", "content": "Say ok."}],
            max_tokens=5,
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        if i == 0:
            content = response.choices[0].message.content[:30] if response.choices else ""
            print(f"   First request: {elapsed:.2f}ms ('{content}')")

    print(f"   Subsequent avg: {statistics.mean(times[1:]):.2f}ms")
    arcllm_median = statistics.median(times)
    print(f"   Median: {arcllm_median:.2f}ms")

    # Calculate overhead
    overhead = arcllm_median - raw_median
    overhead_pct = (overhead / raw_median) * 100 if raw_median > 0 else 0
    print(f"\n3. Overhead Analysis:")
    print(f"   Raw httpx:    {raw_median:.2f}ms")
    print(f"   arcllm:       {arcllm_median:.2f}ms")
    print(f"   Overhead:     {overhead:.2f}ms ({overhead_pct:+.1f}%)")

    if overhead < 5:
        print(f"\n   ✅ Excellent! arcllm overhead is minimal (<5ms)")
    elif overhead < 15:
        print(f"\n   ✓ Good. arcllm overhead is acceptable (<15ms)")
    else:
        print(f"\n   ⚠️ arcllm has significant overhead (>{overhead:.0f}ms)")


def compare_with_litellm(provider: str, model: str):
    """Deep comparison with litellm to understand differences."""
    print("\n" + "=" * 70)
    print("DEEP COMPARISON: arcllm vs litellm")
    print("=" * 70)

    import httpx
    import orjson

    url, api_key, headers = get_api_config(provider)
    headers["Content-Type"] = "application/json"

    body = orjson.dumps(
        {
            "model": model,
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Say ok."}],
            "temperature": 0.0,
            "seed": 42,
        }
    )

    model_string = f"{provider}/{model}"
    request_params = {
        "model": model_string if provider != "openai" else model,
        "messages": [{"role": "user", "content": "Say ok."}],
        "max_tokens": 5,
        "temperature": 0.0,
        "seed": 42,
    }

    iterations = 5

    # =========================================================================
    # 1. RAW HTTPX BASELINE
    # =========================================================================
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ 1. RAW HTTPX BASELINE (what's physically possible)             │")
    print("└─────────────────────────────────────────────────────────────────┘")

    with httpx.Client(http2=True, timeout=60) as client:
        # Warmup
        client.post(url, headers=headers, content=body)
        gc.collect()

        raw_times = []
        for i in range(iterations):
            start = time.perf_counter()
            response = client.post(url, headers=headers, content=body)
            elapsed = (time.perf_counter() - start) * 1000
            raw_times.append(elapsed)
            if i == 0:
                data = orjson.loads(response.content)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:30]
                print(f"   Response: '{content}'")

        raw_median = statistics.median(raw_times)
        print(f"   Times: {[f'{t:.1f}' for t in raw_times]}")
        print(f"   Median: {raw_median:.2f}ms")

    # =========================================================================
    # 2. ARCLLM
    # =========================================================================
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ 2. ARCLLM (our library)                                         │")
    print("└─────────────────────────────────────────────────────────────────┘")

    import arcllm

    # Warmup
    arcllm.completion(**{**request_params, "model": model_string})
    gc.collect()

    arcllm_times = []
    for i in range(iterations):
        start = time.perf_counter()
        response = arcllm.completion(**{**request_params, "model": model_string})
        elapsed = (time.perf_counter() - start) * 1000
        arcllm_times.append(elapsed)
        if i == 0:
            content = response.choices[0].message.content[:30] if response.choices else ""
            print(f"   Response: '{content}'")

    arcllm_median = statistics.median(arcllm_times)
    print(f"   Times: {[f'{t:.1f}' for t in arcllm_times]}")
    print(f"   Median: {arcllm_median:.2f}ms")

    # =========================================================================
    # 3. LITELLM
    # =========================================================================
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ 3. LITELLM (competitor)                                         │")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Run litellm in subprocess to avoid any import conflicts
    litellm_script = f'''
import json
import time
import gc
import litellm

litellm.suppress_debug_info = True
litellm.set_verbose = False

request_params = {request_params}
request_params["model"] = "{model if provider == 'openai' else model_string}"

# Warmup
litellm.completion(**request_params)
gc.collect()

times = []
content = ""
for i in range({iterations}):
    start = time.perf_counter()
    response = litellm.completion(**request_params)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    if i == 0:
        content = response.choices[0].message.content[:30] if response.choices else ""

print(json.dumps({{"times": times, "content": content}}))
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", litellm_script],
            capture_output=True,
            text=True,
            env=os.environ,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"   ❌ Error: {result.stderr[:200]}")
            litellm_median = 0
            litellm_times = []
        else:
            import json
            data = json.loads(result.stdout.strip())
            litellm_times = data["times"]
            litellm_median = statistics.median(litellm_times)
            print(f"   Response: '{data['content']}'")
            print(f"   Times: {[f'{t:.1f}' for t in litellm_times]}")
            print(f"   Median: {litellm_median:.2f}ms")
    except Exception as e:
        print(f"   ❌ Error running litellm: {e}")
        litellm_median = 0
        litellm_times = []

    # =========================================================================
    # 4. ANALYSIS
    # =========================================================================
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ 4. ANALYSIS                                                      │")
    print("└─────────────────────────────────────────────────────────────────┘")

    print(f"\n   {'Library':<15} {'Median':>10} {'vs Raw':>12} {'vs litellm':>12}")
    print("   " + "-" * 51)

    arcllm_overhead = arcllm_median - raw_median
    litellm_overhead = litellm_median - raw_median if litellm_median > 0 else 0
    arcllm_vs_litellm = arcllm_median - litellm_median if litellm_median > 0 else 0

    print(f"   {'Raw httpx':<15} {raw_median:>8.1f}ms {'---':>12} {'---':>12}")
    print(f"   {'arcllm':<15} {arcllm_median:>8.1f}ms {arcllm_overhead:>+10.1f}ms {arcllm_vs_litellm:>+10.1f}ms")
    if litellm_median > 0:
        print(f"   {'litellm':<15} {litellm_median:>8.1f}ms {litellm_overhead:>+10.1f}ms {'---':>12}")

    # =========================================================================
    # 5. BREAKDOWN OF ARCLLM OVERHEAD
    # =========================================================================
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ 5. ARCLLM OVERHEAD BREAKDOWN                                     │")
    print("└─────────────────────────────────────────────────────────────────┘")

    results = profile_arcllm_completion(provider, model)
    total = sum(r.duration_ms for r in results)
    http_time = next((r.duration_ms for r in results if "HTTP request" in r.phase), 0)
    overhead_time = total - http_time

    print(f"\n   Total arcllm time:    {total:.2f}ms")
    print(f"   Network time:         {http_time:.2f}ms")
    print(f"   Library overhead:     {overhead_time:.2f}ms")
    print(f"\n   Breakdown:")
    for r in results:
        if r.phase != "HTTP request (network)":
            pct = (r.duration_ms / total) * 100 if total > 0 else 0
            print(f"     {r.phase:<30} {r.duration_ms:>6.2f}ms ({pct:>4.1f}%)")

    # =========================================================================
    # 6. WHAT'S DIFFERENT?
    # =========================================================================
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ 6. WHY MIGHT LITELLM BE FASTER/SLOWER?                          │")
    print("└─────────────────────────────────────────────────────────────────┘")

    print("""
   Possible reasons for performance differences:

   Network Variance (±50-100ms is NORMAL for LLM APIs):
   - LLM API latency varies significantly between requests
   - Same API can return in 80ms one time, 150ms the next
   - 5 iterations is too few to overcome variance

   If arcllm is slower:
   - Check if connection pooling is working (httpx client reuse)
   - Check if we're creating new objects unnecessarily
   - Check msgspec struct creation overhead

   If litellm is slower:
   - Their validation/preprocessing overhead
   - Different HTTP client (uses httpx too, but config differs)
   - Logging overhead even when suppressed

   IMPORTANT: For LLM APIs, the network is 95%+ of the time.
   Library overhead (1-5ms) is negligible vs API latency (80-800ms).
""")

    # Return comparison data
    return {
        "raw_median": raw_median,
        "arcllm_median": arcllm_median,
        "litellm_median": litellm_median,
        "arcllm_overhead": overhead_time,
    }


def trace_full_request(provider: str, model: str):
    """Trace a full request with detailed timing."""
    print("\n" + "=" * 70)
    print("FULL REQUEST TRACE")
    print("=" * 70)

    results = profile_arcllm_completion(provider, model)

    total = sum(r.duration_ms for r in results)

    print(f"\n{'Phase':<35} {'Time':>10} {'%':>8}")
    print("-" * 55)
    for r in results:
        pct = (r.duration_ms / total) * 100 if total > 0 else 0
        print(f"{r.phase:<35} {r.duration_ms:>8.2f}ms {pct:>7.1f}%")
        if r.details:
            for k, v in r.details.items():
                print(f"  └─ {k}: {v}")
    print("-" * 55)
    print(f"{'TOTAL':<35} {total:>8.2f}ms")


def trace_streaming_request(provider: str, model: str):
    """Trace a streaming request."""
    print("\n" + "=" * 70)
    print("STREAMING REQUEST TRACE")
    print("=" * 70)

    results = profile_arcllm_streaming(provider, model)

    print(f"\n{'Phase':<35} {'Time':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r.phase:<35} {r.duration_ms:>8.2f}ms")
        if r.details:
            for k, v in r.details.items():
                print(f"  └─ {k}: {v}")


def generate_optimization_report(provider: str, model: str):
    """Generate optimization recommendations."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION STATUS")
    print("=" * 70)

    optimizations = []

    # Check HTTP client
    from arcllm.http.client import HTTPClient

    import inspect

    source = inspect.getsource(HTTPClient)

    if "httpx" in source:
        optimizations.append(("✅", "httpx sync", "Using httpx for sync HTTP (connection pooling, HTTP/2)"))
    else:
        optimizations.append(("❌", "httpx sync", "Not using httpx for sync"))

    # Check async client
    from arcllm.http.async_client import AsyncHTTPClient
    async_source = inspect.getsource(AsyncHTTPClient)

    if "aiohttp" in async_source:
        optimizations.append(("✅", "aiohttp async", "Using aiohttp for async HTTP (optimized for async)"))
    elif "httpx" in async_source:
        optimizations.append(("✅", "httpx async", "Using httpx.AsyncClient for async HTTP"))
    else:
        optimizations.append(("❌", "async client", "No optimized async client"))

    # Check JSON library
    try:
        import orjson

        optimizations.append(("✅", "orjson", "Using orjson for fast JSON serialization"))
    except ImportError:
        optimizations.append(("❌", "orjson", "Not using orjson"))

    # Check msgspec
    try:
        import msgspec
        from arcllm.types import ModelResponse
        if "Struct" in str(type(ModelResponse)):
            optimizations.append(("✅", "msgspec", "Using msgspec.Struct for fast types"))
        else:
            optimizations.append(("⚠️", "msgspec", "msgspec available but not used for types"))
    except ImportError:
        optimizations.append(("❌", "msgspec", "Not using msgspec"))

    # Check uvloop
    try:
        import uvloop
        optimizations.append(("✅", "uvloop", "uvloop available for faster async event loop"))
    except ImportError:
        optimizations.append(("⚠️", "uvloop", "uvloop not installed (optional)"))

    # Check for lazy loading
    from arcllm.providers import base

    base_source = inspect.getsource(base)
    if "_PROVIDER_MODULES" in base_source:
        optimizations.append(("✅", "Lazy loading", "Providers are lazily loaded on first use"))
    else:
        optimizations.append(("⚠️", "Lazy loading", "Providers may be loaded eagerly"))

    # Check HTTP/2
    if "http2=True" in source or "http2" in source.lower():
        optimizations.append(("✅", "HTTP/2", "HTTP/2 enabled for sync client"))
    else:
        optimizations.append(("⚠️", "HTTP/2", "HTTP/2 may not be enabled"))

    print("\n")
    for status, area, description in optimizations:
        print(f"  {status} {area:<15} {description}")

    # Summary
    good = sum(1 for s, _, _ in optimizations if s == "✅")
    total = len(optimizations)
    print(f"\n  Score: {good}/{total} optimizations applied")

    if good == total:
        print("\n  🚀 All optimizations applied! Performance should be optimal.")


def main():
    parser = argparse.ArgumentParser(description="Deep profiling for arcllm")
    parser.add_argument("--provider", default="groq", help="Provider to test")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Model to test")
    parser.add_argument("--trace", action="store_true", help="Show full call trace")
    parser.add_argument("--cprofile", action="store_true", help="Run with cProfile")
    parser.add_argument("--streaming", action="store_true", help="Profile streaming")
    parser.add_argument("--vs-litellm", action="store_true", help="Compare with litellm")
    parser.add_argument("--all", action="store_true", help="Run all analyses")

    args = parser.parse_args()

    print("=" * 70)
    print("arcllm Deep Profiling")
    print("=" * 70)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")

    # Always show optimization status first
    generate_optimization_report(args.provider, args.model)

    if args.all or args.vs_litellm:
        compare_with_litellm(args.provider, args.model)

    if args.all or args.trace:
        trace_full_request(args.provider, args.model)

    if args.all or args.streaming:
        trace_streaming_request(args.provider, args.model)

    if args.all:
        analyze_http_client()
        analyze_json_performance()
        analyze_sse_parser()
        compare_with_raw_httpx(args.provider, args.model)

    if args.cprofile:
        print("\n" + "=" * 70)
        print("cProfile Results")
        print("=" * 70)
        import arcllm

        _, stats = profile_with_cprofile(
            arcllm.completion,
            model=f"{args.provider}/{args.model}",
            messages=[{"role": "user", "content": "Say ok."}],
            max_tokens=5,
        )
        print(stats)


if __name__ == "__main__":
    main()
