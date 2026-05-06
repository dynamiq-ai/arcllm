#!/usr/bin/env python3
"""
Benchmark arcllm vs litellm performance.

This script compares performance of arcllm against litellm across various metrics:
- Import time (cold start)
- Memory usage
- Sync completion latency
- Streaming TTFT (time to first token)
- Async completion latency
- Concurrent async requests (where uvloop shines)

Key features:
- Both libraries run in ISOLATED subprocesses (no cross-contamination)
- arcllm tested with uvloop for max performance
- Identical request parameters (seed, temperature, message)
- Proper validation of responses
- Statistical analysis with multiple iterations

Usage:
    python benchmarks/benchmark_vs_litellm.py --provider groq --model llama-3.3-70b-versatile
    python benchmarks/benchmark_vs_litellm.py --provider openai --model gpt-4o-mini --iterations 10
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Common request parameters for fair comparison
BENCHMARK_SEED = 42
BENCHMARK_TEMPERATURE = 0.0  # Deterministic
BENCHMARK_MAX_TOKENS = 10
BENCHMARK_MESSAGE = "Count from 1 to 5."


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    library: str
    metric: str
    values: list[float]
    unit: str
    errors: list[str] = field(default_factory=list)
    sample_responses: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.values)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0


def measure_import_time(library: str, use_uvloop: bool = False) -> float:
    """Measure import time for a library in isolated subprocess."""
    uvloop_setup = (
        """
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass
"""
        if use_uvloop
        else ""
    )

    script = f"""
import time
{uvloop_setup}
start = time.perf_counter()
import {library}
elapsed = (time.perf_counter() - start) * 1000
print(elapsed)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Import failed: {result.stderr}")
    return float(result.stdout.strip())


def measure_memory(library: str) -> float:
    """Measure memory usage after importing a library."""
    script = f"""
import {library}
import os
try:
    import resource
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns KB
    if os.uname().sysname == 'Darwin':
        mem = mem / 1024 / 1024  # bytes to MB
    else:
        mem = mem / 1024  # KB to MB
except ImportError:
    import psutil
    mem = psutil.Process().memory_info().rss / 1024 / 1024
print(mem)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Memory measurement failed: {result.stderr}")
    return float(result.stdout.strip())


def run_arcllm_benchmark_isolated(
    provider: str,
    model: str,
    iterations: int,
    warmup: int = 2,
    use_uvloop: bool = True,
    concurrency: int = 5,
) -> dict[str, Any]:
    """Run arcllm benchmarks in isolated subprocess with uvloop."""

    uvloop_setup = (
        """
# Install uvloop for max performance
try:
    import uvloop
    uvloop.install()
    UVLOOP_ENABLED = True
except ImportError:
    UVLOOP_ENABLED = False
"""
        if use_uvloop
        else "UVLOOP_ENABLED = False"
    )

    script = f'''
import json
import time
import gc
import asyncio

{uvloop_setup}

import arcllm

provider = "{provider}"
model = "{model}"
iterations = {iterations}
warmup = {warmup}
concurrency = {concurrency}

SEED = {BENCHMARK_SEED}
TEMPERATURE = {BENCHMARK_TEMPERATURE}
MAX_TOKENS = {BENCHMARK_MAX_TOKENS}
MESSAGE = "{BENCHMARK_MESSAGE}"

results = {{"uvloop": UVLOOP_ENABLED}}
model_string = f"{{provider}}/{{model}}"

request_params = {{
    "model": model_string,
    "messages": [{{"role": "user", "content": MESSAGE}}],
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "seed": SEED,
}}

def validate_response(response):
    try:
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and choice.message:
                content = choice.message.content or ""
                if content.strip():
                    return True, content, ""
        return False, "", "No content"
    except Exception as e:
        return False, "", str(e)

def validate_chunk(chunk):
    try:
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, 'delta') and choice.delta:
                content = getattr(choice.delta, 'content', None) or ""
                return bool(content), content, ""
        return False, "", ""
    except Exception as e:
        return False, "", str(e)

# Warmup
for _ in range(warmup):
    try:
        arcllm.completion(**request_params)
    except:
        pass

# Force GC before benchmarks
gc.collect()

# Sync completion latency
latencies = []
errors = []
samples = []
for i in range(iterations):
    gc.collect()
    start = time.perf_counter()
    try:
        response = arcllm.completion(**request_params)
        elapsed = (time.perf_counter() - start) * 1000
        is_valid, content, error = validate_response(response)
        if is_valid:
            latencies.append(elapsed)
            if i < 3:
                samples.append(content[:100])
        else:
            errors.append(error)
    except Exception as e:
        errors.append(f"{{type(e).__name__}}: {{e}}")

results["sync_latency"] = {{"values": latencies, "errors": errors, "samples": samples}}

# Streaming TTFT
ttft_values = []
stream_errors = []
stream_samples = []
for i in range(iterations):
    gc.collect()
    start = time.perf_counter()
    full_content = ""
    try:
        stream = arcllm.completion(**{{**request_params, "stream": True}})
        got_first_token = False
        for chunk in stream:
            has_content, content, _ = validate_chunk(chunk)
            if has_content:
                full_content += content
                if not got_first_token:
                    ttft = (time.perf_counter() - start) * 1000
                    ttft_values.append(ttft)
                    got_first_token = True

        if not got_first_token:
            stream_errors.append("No content in stream")
        elif i < 3:
            stream_samples.append(full_content[:100])
    except Exception as e:
        stream_errors.append(f"{{type(e).__name__}}: {{e}}")

results["streaming_ttft"] = {{"values": ttft_values, "errors": stream_errors, "samples": stream_samples}}

# Async completion (sequential)
async def measure_async_sequential():
    latencies = []
    errors = []
    samples = []
    for i in range(iterations):
        gc.collect()
        start = time.perf_counter()
        try:
            response = await arcllm.acompletion(**request_params)
            elapsed = (time.perf_counter() - start) * 1000
            is_valid, content, error = validate_response(response)
            if is_valid:
                latencies.append(elapsed)
                if i < 3:
                    samples.append(content[:100])
            else:
                errors.append(error)
        except Exception as e:
            errors.append(f"{{type(e).__name__}}: {{e}}")
    return latencies, errors, samples

async_latencies, async_errors, async_samples = asyncio.run(measure_async_sequential())
results["async_latency"] = {{"values": async_latencies, "errors": async_errors, "samples": async_samples}}

# Concurrent async (where uvloop really helps)
async def measure_concurrent():
    async def single_request():
        start = time.perf_counter()
        try:
            response = await arcllm.acompletion(**request_params)
            elapsed = (time.perf_counter() - start) * 1000
            is_valid, content, _ = validate_response(response)
            return elapsed if is_valid else None, None
        except Exception as e:
            return None, str(e)

    # Run multiple batches
    batch_times = []
    all_latencies = []
    batch_errors = []

    for _ in range(iterations):
        gc.collect()
        batch_start = time.perf_counter()
        tasks = [single_request() for _ in range(concurrency)]
        results_batch = await asyncio.gather(*tasks)
        batch_time = (time.perf_counter() - batch_start) * 1000
        batch_times.append(batch_time)

        for latency, error in results_batch:
            if latency is not None:
                all_latencies.append(latency)
            if error is not None:
                batch_errors.append(error)

    return batch_times, all_latencies, batch_errors

batch_times, concurrent_latencies, concurrent_errors = asyncio.run(measure_concurrent())
results["concurrent_batch_time"] = {{"values": batch_times, "errors": concurrent_errors, "samples": []}}
results["concurrent_individual"] = {{"values": concurrent_latencies, "errors": [], "samples": []}}

print(json.dumps(results))
'''

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ,
    )

    if result.returncode != 0:
        raise RuntimeError(f"arcllm benchmark failed: {result.stderr}")

    return json.loads(result.stdout.strip())


def run_litellm_benchmark_isolated(
    provider: str,
    model: str,
    iterations: int,
    warmup: int = 2,
    concurrency: int = 5,
) -> dict[str, Any]:
    """Run litellm benchmarks in isolated subprocess."""

    script = f'''
import json
import time
import gc
import asyncio
import litellm

# Suppress litellm logs
litellm.suppress_debug_info = True
litellm.set_verbose = False

provider = "{provider}"
model = "{model}"
iterations = {iterations}
warmup = {warmup}
concurrency = {concurrency}

SEED = {BENCHMARK_SEED}
TEMPERATURE = {BENCHMARK_TEMPERATURE}
MAX_TOKENS = {BENCHMARK_MAX_TOKENS}
MESSAGE = "{BENCHMARK_MESSAGE}"

results = {{}}
model_string = f"{{provider}}/{{model}}" if provider != "openai" else model

request_params = {{
    "model": model_string,
    "messages": [{{"role": "user", "content": MESSAGE}}],
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "seed": SEED,
}}

def validate_response(response):
    try:
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and choice.message:
                content = choice.message.content or ""
                if content.strip():
                    return True, content, ""
        return False, "", "No content"
    except Exception as e:
        return False, "", str(e)

def validate_chunk(chunk):
    try:
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, 'delta') and choice.delta:
                content = getattr(choice.delta, 'content', None) or ""
                return bool(content), content, ""
        return False, "", ""
    except Exception as e:
        return False, "", str(e)

# Warmup
for _ in range(warmup):
    try:
        litellm.completion(**request_params)
    except:
        pass

gc.collect()

# Sync completion latency
latencies = []
errors = []
samples = []
for i in range(iterations):
    gc.collect()
    start = time.perf_counter()
    try:
        response = litellm.completion(**request_params)
        elapsed = (time.perf_counter() - start) * 1000
        is_valid, content, error = validate_response(response)
        if is_valid:
            latencies.append(elapsed)
            if i < 3:
                samples.append(content[:100])
        else:
            errors.append(error)
    except Exception as e:
        errors.append(f"{{type(e).__name__}}: {{e}}")

results["sync_latency"] = {{"values": latencies, "errors": errors, "samples": samples}}

# Streaming TTFT
ttft_values = []
stream_errors = []
stream_samples = []
for i in range(iterations):
    gc.collect()
    start = time.perf_counter()
    full_content = ""
    try:
        stream = litellm.completion(**{{**request_params, "stream": True}})
        got_first_token = False
        for chunk in stream:
            has_content, content, _ = validate_chunk(chunk)
            if has_content:
                full_content += content
                if not got_first_token:
                    ttft = (time.perf_counter() - start) * 1000
                    ttft_values.append(ttft)
                    got_first_token = True

        if not got_first_token:
            stream_errors.append("No content in stream")
        elif i < 3:
            stream_samples.append(full_content[:100])
    except Exception as e:
        stream_errors.append(f"{{type(e).__name__}}: {{e}}")

results["streaming_ttft"] = {{"values": ttft_values, "errors": stream_errors, "samples": stream_samples}}

# Async completion
async def measure_async_sequential():
    latencies = []
    errors = []
    samples = []
    for i in range(iterations):
        gc.collect()
        start = time.perf_counter()
        try:
            response = await litellm.acompletion(**request_params)
            elapsed = (time.perf_counter() - start) * 1000
            is_valid, content, error = validate_response(response)
            if is_valid:
                latencies.append(elapsed)
                if i < 3:
                    samples.append(content[:100])
            else:
                errors.append(error)
        except Exception as e:
            errors.append(f"{{type(e).__name__}}: {{e}}")
    return latencies, errors, samples

async_latencies, async_errors, async_samples = asyncio.run(measure_async_sequential())
results["async_latency"] = {{"values": async_latencies, "errors": async_errors, "samples": async_samples}}

# Concurrent async
async def measure_concurrent():
    async def single_request():
        start = time.perf_counter()
        try:
            response = await litellm.acompletion(**request_params)
            elapsed = (time.perf_counter() - start) * 1000
            is_valid, content, _ = validate_response(response)
            return elapsed if is_valid else None, None
        except Exception as e:
            return None, str(e)

    batch_times = []
    all_latencies = []
    batch_errors = []

    for _ in range(iterations):
        gc.collect()
        batch_start = time.perf_counter()
        tasks = [single_request() for _ in range(concurrency)]
        results_batch = await asyncio.gather(*tasks)
        batch_time = (time.perf_counter() - batch_start) * 1000
        batch_times.append(batch_time)

        for latency, error in results_batch:
            if latency is not None:
                all_latencies.append(latency)
            if error is not None:
                batch_errors.append(error)

    return batch_times, all_latencies, batch_errors

batch_times, concurrent_latencies, concurrent_errors = asyncio.run(measure_concurrent())
results["concurrent_batch_time"] = {{"values": batch_times, "errors": concurrent_errors, "samples": []}}
results["concurrent_individual"] = {{"values": concurrent_latencies, "errors": [], "samples": []}}

print(json.dumps(results))
'''

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ,
    )

    if result.returncode != 0:
        raise RuntimeError(f"litellm benchmark failed: {result.stderr}")

    return json.loads(result.stdout.strip())


def print_comparison(
    arcllm_data: dict[str, Any],
    litellm_data: dict[str, Any],
    arcllm_import: float,
    litellm_import: float,
    arcllm_memory: float,
    litellm_memory: float,
    concurrency: int,
) -> None:
    """Print a comparison table of results."""

    def make_result(data: dict, key: str, lib: str) -> BenchmarkResult:
        d = data.get(key, {"values": [], "errors": [], "samples": []})
        return BenchmarkResult(
            library=lib,
            metric=key,
            values=d.get("values", []),
            unit="ms",
            errors=d.get("errors", []),
            sample_responses=d.get("samples", []),
        )

    print()
    print("=" * 80)
    print("BENCHMARK RESULTS: arcllm vs litellm")
    print("=" * 80)

    # Configuration
    uvloop_status = "✅ ENABLED" if arcllm_data.get("uvloop", False) else "❌ DISABLED"
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ CONFIGURATION                                                               │")
    print("├───────────────────────┬─────────────────────────────────────────────────────┤")
    print(f"│ arcllm uvloop         │ {uvloop_status:<51} │")
    print(f"│ Concurrent requests   │ {concurrency:<51} │")
    print(f"│ Message               │ {BENCHMARK_MESSAGE:<51} │")
    print(f"│ Seed                  │ {BENCHMARK_SEED:<51} │")
    print("└───────────────────────┴─────────────────────────────────────────────────────┘")

    # Startup metrics
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ STARTUP PERFORMANCE                                                         │")
    print("├───────────────────────┬──────────────┬──────────────┬────────────┬──────────┤")
    print("│ Metric                │ arcllm       │ litellm      │ Difference │ Winner   │")
    print("├───────────────────────┼──────────────┼──────────────┼────────────┼──────────┤")

    litellm_import - arcllm_import
    import_winner = "arcllm" if arcllm_import < litellm_import else "litellm"
    import_ratio = litellm_import / arcllm_import if arcllm_import > 0 else 0
    print(
        f"│ Import Time           │ {arcllm_import:>8.1f} ms  │ {litellm_import:>8.1f} ms  │   {import_ratio:>5.1f}x   │ {import_winner:>8} │"
    )

    litellm_memory - arcllm_memory
    memory_winner = "arcllm" if arcllm_memory < litellm_memory else "litellm"
    memory_ratio = litellm_memory / arcllm_memory if arcllm_memory > 0 else 0
    print(
        f"│ Memory Usage          │ {arcllm_memory:>8.1f} MB  │ {litellm_memory:>8.1f} MB  │   {memory_ratio:>5.1f}x   │ {memory_winner:>8} │"
    )

    print("└───────────────────────┴──────────────┴──────────────┴────────────┴──────────┘")

    # Request metrics
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ REQUEST PERFORMANCE (median latency)                                        │")
    print("├───────────────────────┬──────────────┬──────────────┬────────────┬──────────┤")
    print("│ Metric                │ arcllm       │ litellm      │ Difference │ Winner   │")
    print("├───────────────────────┼──────────────┼──────────────┼────────────┼──────────┤")

    metrics = [
        ("Sync Completion", "sync_latency"),
        ("Streaming TTFT", "streaming_ttft"),
        ("Async Sequential", "async_latency"),
        (f"Concurrent ({concurrency}x)", "concurrent_batch_time"),
    ]

    for name, key in metrics:
        arc = make_result(arcllm_data, key, "arcllm")
        lit = make_result(litellm_data, key, "litellm")

        if arc.values and lit.values:
            diff = lit.median - arc.median
            winner = "arcllm" if arc.median < lit.median else "litellm"
            print(
                f"│ {name:<21} │ {arc.median:>8.1f} ms  │ {lit.median:>8.1f} ms  │ {diff:>+8.1f} ms │ {winner:>8} │"
            )
        else:
            print(f"│ {name:<21} │ {'N/A':>11}  │ {'N/A':>11}  │ {'N/A':>10} │ {'N/A':>8} │")

    print("└───────────────────────┴──────────────┴──────────────┴────────────┴──────────┘")

    # Sample responses
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ SAMPLE RESPONSES                                                            │")
    print("├───────────────────────┬─────────────────────────────────────────────────────┤")

    arc_sync = make_result(arcllm_data, "sync_latency", "arcllm")
    lit_sync = make_result(litellm_data, "sync_latency", "litellm")
    arc_sample = arc_sync.sample_responses[0][:50] if arc_sync.sample_responses else "(no response)"
    lit_sample = lit_sync.sample_responses[0][:50] if lit_sync.sample_responses else "(no response)"
    print(f"│ arcllm                │ {arc_sample:<51} │")
    print(f"│ litellm               │ {lit_sample:<51} │")
    print("└───────────────────────┴─────────────────────────────────────────────────────┘")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"  Import time:     arcllm is {import_ratio:.1f}x faster")
    print(f"  Memory usage:    arcllm uses {memory_ratio:.1f}x less memory")

    arc_sync = make_result(arcllm_data, "sync_latency", "arcllm")
    lit_sync = make_result(litellm_data, "sync_latency", "litellm")
    if arc_sync.values and lit_sync.values:
        sync_diff = lit_sync.median - arc_sync.median
        if sync_diff > 0:
            print(f"  Sync requests:   arcllm is {sync_diff:.1f}ms faster")
        else:
            print(f"  Sync requests:   litellm is {-sync_diff:.1f}ms faster")

    arc_stream = make_result(arcllm_data, "streaming_ttft", "arcllm")
    lit_stream = make_result(litellm_data, "streaming_ttft", "litellm")
    if arc_stream.values and lit_stream.values:
        stream_diff = lit_stream.median - arc_stream.median
        if stream_diff > 0:
            print(f"  Streaming TTFT:  arcllm is {stream_diff:.1f}ms faster")
        else:
            print(f"  Streaming TTFT:  litellm is {-stream_diff:.1f}ms faster")

    arc_conc = make_result(arcllm_data, "concurrent_batch_time", "arcllm")
    lit_conc = make_result(litellm_data, "concurrent_batch_time", "litellm")
    if arc_conc.values and lit_conc.values:
        conc_diff = lit_conc.median - arc_conc.median
        if conc_diff > 0:
            print(
                f"  Concurrent:      arcllm is {conc_diff:.1f}ms faster ({concurrency} parallel requests)"
            )
        else:
            print(
                f"  Concurrent:      litellm is {-conc_diff:.1f}ms faster ({concurrency} parallel requests)"
            )

    print(
        f"  uvloop:          {'✅ Enabled for arcllm' if arcllm_data.get('uvloop') else '❌ Not available'}"
    )

    print("=" * 80)


def save_results(
    output_path: str,
    arcllm_data: dict[str, Any],
    litellm_data: dict[str, Any],
    arcllm_import: float,
    litellm_import: float,
    arcllm_memory: float,
    litellm_memory: float,
    provider: str,
    model: str,
    concurrency: int,
) -> None:
    """Save benchmark results to JSON file."""
    data = {
        "metadata": {
            "provider": provider,
            "model": model,
            "seed": BENCHMARK_SEED,
            "temperature": BENCHMARK_TEMPERATURE,
            "max_tokens": BENCHMARK_MAX_TOKENS,
            "message": BENCHMARK_MESSAGE,
            "concurrency": concurrency,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "arcllm": {
            "import_time_ms": arcllm_import,
            "memory_mb": arcllm_memory,
            "uvloop_enabled": arcllm_data.get("uvloop", False),
            **{k: v for k, v in arcllm_data.items() if k != "uvloop"},
        },
        "litellm": {
            "import_time_ms": litellm_import,
            "memory_mb": litellm_memory,
            **litellm_data,
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark arcllm vs litellm performance")
    parser.add_argument(
        "--provider",
        default="groq",
        help="Provider to test (default: groq)",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Model to test (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per test (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup requests (default: 2)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent requests for async test (default: 5)",
    )
    parser.add_argument(
        "--no-uvloop",
        action="store_true",
        help="Disable uvloop for arcllm",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--skip-litellm",
        action="store_true",
        help="Skip litellm benchmarks (arcllm only)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("arcllm vs litellm Benchmark (ISOLATED SUBPROCESSES)")
    print("=" * 80)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Concurrency: {args.concurrency}")
    print(f"uvloop: {'disabled' if args.no_uvloop else 'enabled for arcllm'}")
    print(f"Seed: {BENCHMARK_SEED}")
    print(f"Temperature: {BENCHMARK_TEMPERATURE}")
    print("=" * 80)
    print()

    # Measure import times (in isolated processes)
    print("Measuring import times (isolated processes)...")
    arcllm_import = measure_import_time("arcllm", use_uvloop=not args.no_uvloop)
    print(f"  arcllm: {arcllm_import:.1f}ms")

    if not args.skip_litellm:
        litellm_import = measure_import_time("litellm")
        print(f"  litellm: {litellm_import:.1f}ms")
    else:
        litellm_import = 0

    # Measure memory
    print("\nMeasuring memory usage (isolated processes)...")
    arcllm_memory = measure_memory("arcllm")
    print(f"  arcllm: {arcllm_memory:.1f}MB")

    if not args.skip_litellm:
        litellm_memory = measure_memory("litellm")
        print(f"  litellm: {litellm_memory:.1f}MB")
    else:
        litellm_memory = 0

    # Run arcllm benchmarks (isolated subprocess with uvloop)
    print("\nRunning arcllm benchmarks (isolated subprocess)...")
    print(f"  uvloop: {'disabled' if args.no_uvloop else 'enabled'}")
    arcllm_data = run_arcllm_benchmark_isolated(
        args.provider,
        args.model,
        args.iterations,
        args.warmup,
        use_uvloop=not args.no_uvloop,
        concurrency=args.concurrency,
    )
    print(f"  ✅ Completed (uvloop={arcllm_data.get('uvloop', False)})")

    if not args.skip_litellm:
        print("\nRunning litellm benchmarks (isolated subprocess)...")
        litellm_data = run_litellm_benchmark_isolated(
            args.provider,
            args.model,
            args.iterations,
            args.warmup,
            concurrency=args.concurrency,
        )
        print("  ✅ Completed")
    else:
        litellm_data = {}

    # Print comparison
    print_comparison(
        arcllm_data,
        litellm_data,
        arcllm_import,
        litellm_import,
        arcllm_memory,
        litellm_memory,
        args.concurrency,
    )

    # Save results
    save_results(
        args.output,
        arcllm_data,
        litellm_data,
        arcllm_import,
        litellm_import,
        arcllm_memory,
        litellm_memory,
        args.provider,
        args.model,
        args.concurrency,
    )


if __name__ == "__main__":
    main()
