# arcllm examples

Each example is a standalone Python script. Run from the repo root:

```bash
pip install -e ".[dev]"           # editable install with test deps
export OPENAI_API_KEY="sk-..."    # whatever key the example needs
python examples/<name>.py
```

## Index

| Example | Demonstrates | Requires |
|---|---|---|
| [`simple_completion.py`](simple_completion.py) | Basic chat-completion call, response inspection, cost calculation | `OPENAI_API_KEY` |
| [`streaming.py`](streaming.py) | Sync + async streaming, `stream_chunk_builder` to assemble final response | `OPENAI_API_KEY` |
| [`tool_calling.py`](tool_calling.py) | Function-calling tools, multi-turn tool-use loop | `OPENAI_API_KEY` |
| [`structured_output.py`](structured_output.py) | JSON mode + JSON schema for typed extraction | `OPENAI_API_KEY` |
| [`embeddings.py`](embeddings.py) | Sync + async embedding, cosine similarity helper | `OPENAI_API_KEY` |
| [`rerank.py`](rerank.py) | Cohere `/v2/rerank` — sync + async | `COHERE_API_KEY` |
| [`image_generation.py`](image_generation.py) | DALL-E 3 / gpt-image-1 — generation, b64 response, async | `OPENAI_API_KEY` |
| [`token_counter.py`](token_counter.py) | Heuristic vs tiktoken-precise token counting | none (`pip install "arcllm-sdk[tokenize]"` for exact counts) |
| [`capability_helpers.py`](capability_helpers.py) | `supports_*`, `get_max_tokens`, `get_model_info`, `get_supported_openai_params` — pure-Python | none |
| [`error_handling.py`](error_handling.py) | Catching the full exception hierarchy, retry-vs-fail decisions | `OPENAI_API_KEY` |
| [`anthropic_full_matrix.py`](anthropic_full_matrix.py) | Comprehensive Claude-family compat matrix (vision, PDF, thinking, tools, structured output) | `ANTHROPIC_API_KEY` |

## Conventions

- **Async equivalents** for sync APIs are exposed as `acompletion` /
  `aembedding` / `arerank` / `aimage_*`. The synchronous examples above
  include async demos at the bottom of the same file when relevant.
- **API keys** are read from environment variables by default — see the
  README's "Authentication" section for the full table. You can also pass
  `api_key=` to any function to override per-call.
- **Model strings** use the `<provider>/<model_id>` convention. Bare model
  ids without a prefix are treated as OpenAI by default.
- **No `print()` in library code** — these examples deliberately print
  to stdout; that's fine for examples, not for production code paths.
