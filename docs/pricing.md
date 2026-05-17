# Pricing & Cost Calculation

> **What this is.** A user-facing reference for how `arcllm` calculates dollar costs from model responses. Read this if you're building usage / billing infrastructure on top of arcllm, you want to know *exactly* what numbers `completion_cost()` returns and why, or you want to add or override a model's pricing.

## TL;DR

```python
import arcllm

response = arcllm.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
)

# Total USD cost for this call.
cost = arcllm.completion_cost(response)
```

`completion_cost()` does one of three things, in order of preference:

1. If the provider reported the cost in a response header (currently: OpenRouter's `x-openrouter-cost`), return that value verbatim. **Provider truth beats local lookup.**
2. Otherwise, look the model up in the static pricing table and compute `prompt_cost + completion_cost` from the token counts in `response.usage`, applying cache-read / cache-write / reasoning-token rates where the model and the response support them.
3. Return `0.0` if `response.usage` is `None`.

## Public API

| Symbol | Returns | Notes |
| --- | --- | --- |
| `arcllm.completion_cost(response, model=None)` | `float` USD | Total cost for a chat / embedding call. |
| `arcllm.cost_per_token(model, prompt_tokens, completion_tokens, *, cache_read_input_tokens=0, cache_creation_input_tokens=0, reasoning_tokens=0)` | `(prompt_cost, completion_cost)` USD tuple | Lower-level: compute cost from raw token counts. |
| `arcllm.get_model_pricing(model)` | `ModelPricing` dataclass | Inspect raw rates without computing a cost. Raises `UnknownModelPricingError` if the model isn't in the table. |
| `arcllm.image_cost(model, n=1)` | `float` USD | Per-image generation cost (DALL-E, Imagen, etc.). |
| `arcllm.audio_cost(model, *, characters=None, seconds=None)` | `float` USD | TTS (per-character) or STT (per-second). |
| `arcllm.rerank_cost(model, queries=1)` | `float` USD | Per-query rerank cost (Cohere, Vertex Reranker). |

All cost helpers raise `arcllm.pricing.tables.UnknownModelPricingError` when invoked on a model that isn't configured for the relevant modality.

## Data model: `ModelPricing`

`ModelPricing` is a frozen slots-only dataclass at `arcllm.pricing.tables.ModelPricing`. Required fields are positional; modality-specific fields are optional and default to `None`.

```python
@dataclass(slots=True, frozen=True)
class ModelPricing:
    input_cost_per_million: float
    output_cost_per_million: float
    cached_input_cost_per_million: float | None = None
    cache_creation_cost_per_million: float | None = None
    output_cost_per_reasoning_token: float | None = None
    image_per_request: float | None = None
    audio_per_character: float | None = None
    audio_per_second: float | None = None
    rerank_per_query: float | None = None
```

| Field | Unit | Populated for | Used by |
| --- | --- | --- | --- |
| `input_cost_per_million` | USD / 1M tokens | every chat & embedding model | base prompt cost |
| `output_cost_per_million` | USD / 1M tokens | every chat model | base completion cost |
| `cached_input_cost_per_million` | USD / 1M tokens | models with prompt caching (Anthropic, OpenAI prompt cache, Gemini context cache, DeepSeek, Bedrock-Anthropic, …) | cache-read tier |
| `cache_creation_cost_per_million` | USD / 1M tokens | Anthropic-family cache writes | cache-write tier (replaces the hard-coded 1.25x multiplier when set) |
| `output_cost_per_reasoning_token` | USD / 1M tokens | OpenAI o-series, DeepSeek-R1, etc. | reasoning tokens billed at this rate; the rest of completion stays at `output_cost_per_million` |
| `image_per_request` | USD / image | DALL-E, GPT-Image-1, Imagen | `image_cost()` |
| `audio_per_character` | USD / input character | TTS models | `audio_cost(model, characters=…)` |
| `audio_per_second` | USD / second of audio | STT models | `audio_cost(model, seconds=…)` |
| `rerank_per_query` | USD / query | Cohere rerank, Vertex Reranker | `rerank_cost()` |

## Cost formula

### Chat / embedding (no caching, no reasoning)

```
total = (prompt_tokens × input_rate) + (completion_tokens × output_rate)
```

Worked example: 1,000 prompt tokens + 500 completion tokens against `openai/gpt-4o-mini`:

```
input_rate  = 0.15 / 1_000_000 = 1.5e-07 USD/token
output_rate = 0.60 / 1_000_000 = 6.0e-07 USD/token

prompt_cost     = 1_000 × 1.5e-07 = 0.000150
completion_cost = 500   × 6.0e-07 = 0.000300
total                            = 0.000450 USD
```

### Chat with prompt caching

When the response carries cache-read and/or cache-creation token counts (`Usage.cache_read_input_tokens`, `Usage.cache_creation_input_tokens`), the prompt is split into three slices:

```
base_prompt = prompt_tokens − cache_read − cache_creation

prompt_cost = (base_prompt × input_rate)
            + (cache_read × cached_rate)
            + (cache_creation × creation_rate)
```

- `cached_rate` comes from `cached_input_cost_per_million` (typically 10% of the base rate). Falls back to `input_rate` if the model has no cached rate configured.
- `creation_rate` comes from `cache_creation_cost_per_million` (varies by provider / TTL). Falls back to `input_rate × 1.25` (the historical Anthropic 5-minute-write assumption) for legacy entries.

Worked example: Anthropic `claude-sonnet-4-5` with 10,000 prompt tokens of which 5,000 are cache-creation writes:

```
input_rate                    = 3.00 / 1_000_000 = 3.0e-06
cache_creation_rate (manifest)= 3.75 / 1_000_000 = 3.75e-06

base_prompt   = 10_000 − 0 − 5_000 = 5_000
prompt_cost   = (5_000 × 3.0e-06) + (5_000 × 3.75e-06)
              = 0.015 + 0.01875
              = 0.03375 USD
```

### Reasoning tokens

When a model has `output_cost_per_reasoning_token` set and the response carries `usage.completion_tokens_details.reasoning_tokens`, the completion side splits in two:

```
non_reasoning   = completion_tokens − reasoning_tokens

completion_cost = (non_reasoning × output_rate)
                + (reasoning_tokens × reasoning_rate)
```

Important: provider responses report `reasoning_tokens` *within* `completion_tokens`. The formula subtracts to avoid double-billing.

### Image / audio / rerank

These don't use the per-token formula at all:

```python
image_cost  = pricing.image_per_request × n
audio_cost  = pricing.audio_per_character × characters   # TTS
audio_cost  = pricing.audio_per_second × seconds         # STT
rerank_cost = pricing.rerank_per_query × queries
```

## Provider-native cost surfaces

Some providers report the dollar cost directly in the response. arcllm prefers these values over its static table.

| Provider | How | Captured into |
| --- | --- | --- |
| **OpenRouter** | `x-openrouter-cost` response header (USD float string) | `ModelResponse.provider_reported_cost` |

When `provider_reported_cost` is non-`None`, `completion_cost()` returns it directly and skips the table lookup entirely. This sidesteps the meta-router enumeration problem — OpenRouter exposes thousands of routed models with markup, and the authoritative cost is what they say it is.

Other providers (Cohere's `billed_units`, Bedrock's `x-amzn-bedrock-*` headers) report *token counts* the provider says they billed, not USD. arcllm doesn't currently lift these — the static table is used. If you need provider-native counts for audit purposes, they're available on the response's `usage` field.

## Where prices come from

arcllm doesn't fetch prices at runtime. The flow is:

```
tmp/model_manifests/<provider>.json   ← hand-edited from each provider's pricing page
        │
        │   $ python scripts/sync_tables.py
        ▼
arcllm/pricing/tables.py              ← generated artifact, byte-deterministic
arcllm/capabilities/tables.py
```

- `tmp/model_manifests/` is *gitignored* by design — the JSON files are local scratch. Treat them as source-of-truth for *one machine*, regenerated from the providers' pricing pages.
- `scripts/sync_tables.py` is idempotent and runs `ruff format` on its output. Re-running it without manifest changes produces a byte-identical file.
- `scripts/check_model_drift.py` runs in CI (`.github/workflows/model-drift.yml`) on a weekly schedule. It compares each provider's `/v1/models` endpoint to the pricing table and emits a Markdown report flagging new / disappeared models. It does **not** validate price values — only model presence.

### Canonical pricing sources

| Provider | URL |
| --- | --- |
| OpenAI | https://openai.com/api/pricing/ |
| Anthropic | https://www.anthropic.com/pricing |
| Gemini | https://ai.google.dev/gemini-api/docs/pricing |
| Vertex AI | https://cloud.google.com/vertex-ai/generative-ai/pricing |
| Azure OpenAI | https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/ |
| AWS Bedrock | https://aws.amazon.com/bedrock/pricing/ |
| Mistral | https://docs.mistral.ai/api |
| Cohere | https://cohere.com/pricing |
| Groq | https://groq.com/pricing |
| DeepSeek | https://api-docs.deepseek.com/quick_start/pricing |
| Perplexity | https://docs.perplexity.ai/docs/sonar/quickstart |
| xAI | https://x.ai/api |
| Together AI | https://docs.together.ai/docs/billing |
| Fireworks AI | https://fireworks.ai/pricing |
| Databricks | https://www.databricks.com/product/pricing/foundation-model-serving |
| Watsonx | https://www.ibm.com/products/watsonx-ai/pricing |
| Cerebras | https://www.cerebras.ai/pricing |
| SambaNova | https://cloud.sambanova.ai/plans/pricing |
| DeepInfra | https://deepinfra.com/pricing |
| Nebius | https://docs.nebius.com/applications/standalone/pricing |
| OVHcloud | https://help.ovhcloud.com/csm/en-public-cloud-compute-billing-options |
| Z.AI | https://docs.z.ai/guides/overview/pricing |
| Moonshot | https://platform.kimi.ai/docs/pricing/chat |
| NVIDIA NIM | https://build.nvidia.com |
| Ollama | local — $0 |

## Adding or overriding a model

If a model isn't in the table, or you want to override a rate (custom enterprise contract, regional pricing variation, a model arcllm hasn't picked up yet), edit the relevant `tmp/model_manifests/<provider>.json` and regenerate:

```json
{
  "id": "my-custom-claude-deployment",
  "kind": "chat",
  "context_window": 200000,
  "max_output_tokens": 8192,
  "supports_vision": true,
  "supports_pdf_input": true,
  "supports_tools": true,
  "supports_structured_output": true,
  "input_per_m": 2.50,
  "output_per_m": 12.00,
  "cached_input_per_m": 0.25,
  "cache_creation_per_m": 3.125,
  "deprecated": false
}
```

For modality-specific entries:

```json
{
  "id": "my-image-model",
  "kind": "image",
  "context_window": 0,
  "max_output_tokens": 0,
  "supports_vision": false,
  "supports_pdf_input": false,
  "supports_tools": false,
  "supports_structured_output": false,
  "input_per_m": 0.0,
  "output_per_m": 0.0,
  "image_per_request": 0.03
}
```

Then:

```bash
python scripts/sync_tables.py
pytest tests/test_pricing.py tests/test_tables_parity.py
```

## Out of scope

The following are intentionally not modeled — they'd require either provider-specific code paths or runtime API calls that don't justify the complexity for most use cases. If you need any of these, file an issue at https://github.com/dynamiq-ai/arcllm/issues.

- **Per-region pricing overrides.** Bedrock charges different rates per AWS region; arcllm uses US East as the baseline. EU and APAC rates can differ by 10-30%.
- **Batch-API discounts.** Most providers offer ~50% off via Batch API; we don't currently track which models qualify or the discount tier.
- **Tiered pricing thresholds.** Anthropic prices above 200k tokens at 2x; OpenAI has `priority` / `flex` service tiers; arcllm uses the base rate.
- **Volume / enterprise contracts.** Custom-negotiated rates aren't representable in the static table.
- **OpenRouter per-model entries.** Use the `x-openrouter-cost` header path instead — it's authoritative through OpenRouter's routing layer.
- **HuggingFace per-model entries.** Variable per-model rates; users supply their own via the manifest workflow above.
- **Cohere `billed_units` extraction.** Cohere reports their idea of billed tokens in the response; we could lift them onto `Usage` but currently use the static table. (No clear caller yet.)
- **Audio-token billing for chat-with-audio models** (e.g. gpt-4o-audio-preview, Gemini live API). Token counts for input/output audio are reported separately by the provider; arcllm sums everything into the standard `prompt_tokens`/`completion_tokens` fields. Track at the linked issues URL if you need per-modality breakdown.

## Honest accuracy disclaimer

arcllm's pricing table is a snapshot maintained by hand. Prices change. Models are deprecated. New SKUs appear weekly. Two things to keep in mind:

1. **Check the canonical doc URL** (table above) before trusting a six-figure-budget number from arcllm's table. The drift workflow flags missing/new *models* but doesn't validate *prices*.
2. **For absolute billing accuracy**, use the provider's own usage dashboard or invoice (AWS Cost Management, Azure Cost Management, OpenAI usage page, etc.). arcllm's `completion_cost` is intended for context-window enforcement, request-time logging, prompt-engineering experiments, and pre-flight budget checks — not for line-item billing reconciliation.

If you spot a stale or wrong price: open a PR updating the relevant `tmp/model_manifests/<provider>.json` entry with the canonical doc URL as evidence, then run `scripts/sync_tables.py`.
