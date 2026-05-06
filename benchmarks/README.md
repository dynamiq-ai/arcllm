# arcllm benchmarks

Three production-grade benchmark scripts. None of them mock provider
responses except where explicitly marked; if a script needs an API key it
fails loud rather than skipping silently.

## `overhead.py` — mocked microbenchmarks

Measures arcllm's own SDK overhead — import time, type creation,
request/response building, SSE parsing, full completion round-trip with a
mocked HTTP layer. Doesn't touch any provider.

```bash
python benchmarks/overhead.py             # human-readable report
python benchmarks/overhead.py --json      # stable JSON to stdout
python benchmarks/overhead.py --ci        # enforce regression gates (used in CI)
```

CI gates are intentionally generous (~10× expected) so cloud-runner jitter
doesn't flake them. Tightening them later is fine; they're inline in
`overhead.py` under `_CI_LIMITS`.

## `live_matrix.py` — live cross-provider TTFT + latency

Fires the same tiny prompt at every configured provider's flagship + cheap
model concurrently (`asyncio.gather`) and reports time-to-first-token, total
latency, and tokens-per-second from the provider's own usage block.

Skips silently for any provider whose API key is missing — so a developer
with one or two keys gets a useful local run.

```bash
python benchmarks/live_matrix.py                 # all configured providers, parallel
python benchmarks/live_matrix.py --sequential    # one provider at a time
python benchmarks/live_matrix.py --filter gemini # only Gemini scenarios
```

Output is a single JSON document on stdout. Pipe to `jq` for one-liner
analysis:

```bash
python benchmarks/live_matrix.py | jq '.results[] | select(.error == null) | {model, ttft_ms, tps_out}'
```

## `vs_litellm.py` — arcllm vs litellm comparison

Spawns isolated subprocesses for arcllm and litellm and measures the same
operations head-to-head (import, sync completion, async, streaming, memory,
concurrency). Requires both libraries plus an API key.

```bash
pip install litellm
OPENAI_API_KEY=sk-... python benchmarks/vs_litellm.py --provider openai --model gpt-4o-mini
```

The subprocess isolation matters: it ensures the result reflects each
library's full cold-start cost (lazy imports, module init, first-request
warm-up).

## Why no `quick_benchmark.sh` or `profile_deep.py`

Both have been removed. `quick_benchmark.sh` was a 3-liner shell wrapper
that's now a one-line invocation of `vs_litellm.py`. `profile_deep.py`
hardcoded model names (drift risk), embedded its own subprocess scaffolding,
and overlapped with `vs_litellm.py` — it's been retired.

If you need cProfile-level introspection, run any benchmark under cProfile
directly:

```bash
python -m cProfile -o /tmp/arcllm.prof benchmarks/overhead.py
python -c "import pstats; pstats.Stats('/tmp/arcllm.prof').sort_stats('cumulative').print_stats(40)"
```
