# Provider model matrix

_Auto-generated from `arcllm/capabilities/tables.py` (version `2026.05.07`)._

Run `python scripts/render_matrix.py` to refresh after updating the manifests.


Columns: `kind` is `chat` / `reason` / `embed`; ✅ marks a supported capability flag. Prices are USD per 1M tokens. `reasoning` indicates the model accepts the `reasoning_effort` parameter.

## anthropic (11 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `claude-haiku-4-5` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $1.00 | $5.00 |
| `claude-haiku-4-5-20251001` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $1.00 | $5.00 |
| `claude-opus-4-1` | chat | 200,000 | 32,000 | ✅ | ✅ | ✅ | — | — | $15.00 | $75.00 |
| `claude-opus-4-1-20250805` | chat | 200,000 | 32,000 | ✅ | ✅ | ✅ | — | — | $15.00 | $75.00 |
| `claude-opus-4-5` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $5.00 | $25.00 |
| `claude-opus-4-5-20251101` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $5.00 | $25.00 |
| `claude-opus-4-6` | chat | 1,000,000 | 128,000 | ✅ | ✅ | ✅ | — | — | $5.00 | $25.00 |
| `claude-opus-4-7` | chat | 1,000,000 | 128,000 | ✅ | ✅ | ✅ | — | — | $5.00 | $25.00 |
| `claude-sonnet-4-5` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $3.00 | $15.00 |
| `claude-sonnet-4-5-20250929` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $3.00 | $15.00 |
| `claude-sonnet-4-6` | chat | 1,000,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $3.00 | $15.00 |

## azure (36 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `gpt-4.1` | chat | 1,047,576 | 32,768 | ✅ | — | ✅ | ✅ | — | $2.00 | $8.00 |
| `gpt-4.1-mini` | chat | 1,047,576 | 32,768 | ✅ | — | ✅ | ✅ | — | $0.400 | $1.60 |
| `gpt-4.1-nano` | chat | 1,047,576 | 32,768 | ✅ | — | ✅ | ✅ | — | $0.100 | $0.400 |
| `gpt-4o` | chat | 128,000 | 16,384 | ✅ | — | ✅ | ✅ | — | $2.50 | $10.00 |
| `gpt-4o-mini` | chat | 128,000 | 16,384 | ✅ | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `gpt-5` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.25 | $10.00 |
| `gpt-5-mini` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.250 | $2.00 |
| `gpt-5-nano` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.050 | $0.400 |
| `gpt-5-pro` | reason | 400,000 | 272,000 | ✅ | — | ✅ | ✅ | ✅ | $15.00 | $120.00 |
| `gpt-5.1` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.25 | $10.00 |
| `gpt-5.1-codex` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.25 | $10.00 |
| `gpt-5.1-codex-max` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.25 | $10.00 |
| `gpt-5.1-codex-mini` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.250 | $2.00 |
| `gpt-5.2` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.75 | $14.00 |
| `gpt-5.2-codex` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.75 | $14.00 |
| `gpt-5.2-pro` | reason | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $21.00 | $168.00 |
| `gpt-5.4` | chat | 1,050,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $2.50 | $15.00 |
| `gpt-5.4-mini` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.750 | $4.50 |
| `gpt-5.4-nano` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.200 | $1.25 |
| `gpt-5.4-pro` | reason | 1,050,000 | 128,000 | ✅ | — | ✅ | — | ✅ | $30.00 | $180.00 |
| `gpt-5.5` | chat | 1,050,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $5.00 | $30.00 |
| `gpt-5.5-pro` | reason | 1,050,000 | 128,000 | ✅ | — | ✅ | — | ✅ | $30.00 | $180.00 |
| `gpt-audio` | chat | 128,000 | 16,384 | — | — | ✅ | — | — | $2.50 | $10.00 |
| `gpt-audio-1.5` | chat | 128,000 | 16,384 | — | — | ✅ | — | — | $2.50 | $10.00 |
| `gpt-audio-mini` | chat | 128,000 | 16,384 | — | — | ✅ | — | — | $0.600 | $2.40 |
| `gpt-realtime` | chat | 32,000 | 4,096 | ✅ | — | ✅ | — | — | $4.00 | $16.00 |
| `gpt-realtime-1.5` | chat | 32,000 | 4,096 | ✅ | — | ✅ | — | — | $4.00 | $16.00 |
| `gpt-realtime-mini` | chat | 32,000 | 4,096 | — | — | ✅ | — | — | $0.600 | $2.40 |
| `o1` | reason | 200,000 | 100,000 | ✅ | — | ✅ | ✅ | ✅ | $15.00 | $60.00 |
| `o3` | reason | 200,000 | 100,000 | ✅ | — | ✅ | ✅ | ✅ | $2.00 | $8.00 |
| `o3-mini` | reason | 200,000 | 100,000 | — | — | ✅ | ✅ | ✅ | $1.10 | $4.40 |
| `o3-pro` | reason | 200,000 | 100,000 | ✅ | — | ✅ | ✅ | ✅ | $20.00 | $80.00 |
| `o4-mini` | reason | 200,000 | 100,000 | ✅ | — | ✅ | ✅ | ✅ | $1.10 | $4.40 |
| `text-embedding-3-large` | embed | 8,191 | — | — | — | — | — | — | $0.130 | free |
| `text-embedding-3-small` | embed | 8,191 | — | — | — | — | — | — | $0.020 | free |
| `text-embedding-ada-002` | embed | 8,191 | — | — | — | — | — | — | $0.100 | free |

## bedrock (31 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `amazon.nova-lite-v1:0` | chat | 300,000 | 10,000 | ✅ | ✅ | ✅ | ✅ | — | $0.060 | $0.240 |
| `amazon.nova-micro-v1:0` | chat | 128,000 | 10,000 | — | — | ✅ | ✅ | — | $0.035 | $0.140 |
| `amazon.nova-pro-v1:0` | chat | 300,000 | 10,000 | ✅ | ✅ | ✅ | ✅ | — | $0.800 | $3.20 |
| `amazon.titan-embed-text-v1` | embed | 8,192 | — | — | — | — | — | — | $0.100 | free |
| `amazon.titan-embed-text-v2:0` | embed | 8,192 | — | — | — | — | — | — | $0.200 | free |
| `anthropic.claude-3-5-haiku-20241022-v1:0` | chat | 200,000 | 8,192 | — | ✅ | ✅ | ✅ | — | $0.800 | $4.00 |
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | chat | 200,000 | 8,192 | ✅ | ✅ | ✅ | ✅ | — | $3.00 | $15.00 |
| `anthropic.claude-3-7-sonnet-20250219-v1:0` | chat | 200,000 | 8,192 | ✅ | ✅ | ✅ | ✅ | — | $3.00 | $15.00 |
| `anthropic.claude-3-haiku-20240307-v1:0` | chat | 200,000 | 4,096 | ✅ | ✅ | ✅ | ✅ | — | $0.250 | $1.25 |
| `anthropic.claude-3-opus-20240229-v1:0` | chat | 200,000 | 4,096 | ✅ | — | ✅ | ✅ | — | $15.00 | $75.00 |
| `anthropic.claude-haiku-4-5-20251001-v1:0` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | ✅ | — | $1.00 | $5.00 |
| `anthropic.claude-opus-4-1-20250805-v1:0` | chat | 200,000 | 32,000 | ✅ | ✅ | ✅ | ✅ | — | $15.00 | $75.00 |
| `anthropic.claude-opus-4-20250514-v1:0` | chat | 200,000 | 32,000 | ✅ | ✅ | ✅ | ✅ | — | $15.00 | $75.00 |
| `anthropic.claude-opus-4-5-20251101-v1:0` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | ✅ | — | $5.00 | $25.00 |
| `anthropic.claude-sonnet-4-20250514-v1:0` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | ✅ | — | $3.00 | $15.00 |
| `anthropic.claude-sonnet-4-5-20250929-v1:0` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | ✅ | — | $3.00 | $15.00 |
| `cohere.command-r-plus-v1:0` | chat | 128,000 | 4,096 | — | — | ✅ | — | — | $3.00 | $15.00 |
| `cohere.command-r-v1:0` | chat | 128,000 | 4,096 | — | — | ✅ | — | — | $0.500 | $1.50 |
| `cohere.embed-english-v3` | embed | 512 | — | — | — | — | — | — | $0.100 | free |
| `cohere.embed-multilingual-v3` | embed | 512 | — | — | — | — | — | — | $0.100 | free |
| `meta.llama3-1-405b-instruct-v1:0` | chat | 128,000 | 4,096 | — | — | ✅ | — | — | $5.32 | $16.00 |
| `meta.llama3-1-70b-instruct-v1:0` | chat | 128,000 | 2,048 | — | — | ✅ | — | — | $0.990 | $0.990 |
| `meta.llama3-1-8b-instruct-v1:0` | chat | 128,000 | 2,048 | — | — | ✅ | — | — | $0.220 | $0.220 |
| `meta.llama3-2-11b-instruct-v1:0` | chat | 128,000 | 4,096 | ✅ | — | ✅ | — | — | $0.350 | $0.350 |
| `meta.llama3-2-1b-instruct-v1:0` | chat | 128,000 | 4,096 | — | — | ✅ | — | — | $0.100 | $0.100 |
| `meta.llama3-2-3b-instruct-v1:0` | chat | 128,000 | 4,096 | — | — | ✅ | — | — | $0.150 | $0.150 |
| `meta.llama3-2-90b-instruct-v1:0` | chat | 128,000 | 4,096 | ✅ | — | ✅ | — | — | $2.00 | $2.00 |
| `meta.llama3-3-70b-instruct-v1:0` | chat | 128,000 | 4,096 | — | — | ✅ | — | — | $0.720 | $0.720 |
| `meta.llama4-maverick-17b-instruct-v1:0` | chat | 128,000 | 4,096 | ✅ | — | ✅ | — | — | $0.240 | $0.970 |
| `meta.llama4-scout-17b-instruct-v1:0` | chat | 128,000 | 4,096 | ✅ | — | ✅ | — | — | $0.170 | $0.660 |
| `mistral.mistral-large-2407-v1:0` | chat | 128,000 | 8,192 | — | — | ✅ | — | — | $3.00 | $9.00 |

## cohere (11 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `command-a-03-2025` | chat | 256,000 | 8,000 | — | — | ✅ | ✅ | — | $2.50 | $10.00 |
| `command-a-reasoning-08-2025` | reason | 256,000 | 32,000 | — | — | ✅ | ✅ | ✅ | $2.50 | $10.00 |
| `command-a-vision-07-2025` | chat | 128,000 | 8,000 | ✅ | — | — | ✅ | — | $2.50 | $10.00 |
| `command-r-08-2024` | chat | 128,000 | 4,096 | — | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `command-r-plus-08-2024` | chat | 128,000 | 4,096 | — | — | ✅ | ✅ | — | $2.50 | $10.00 |
| `command-r7b-12-2024` | chat | 128,000 | 4,096 | — | — | ✅ | ✅ | — | $0.037 | $0.150 |
| `embed-english-light-v3.0` | embed | 512 | — | — | — | — | — | — | $0.100 | free |
| `embed-english-v3.0` | embed | 512 | — | — | — | — | — | — | $0.100 | free |
| `embed-multilingual-light-v3.0` | embed | 512 | — | — | — | — | — | — | $0.100 | free |
| `embed-multilingual-v3.0` | embed | 512 | — | — | — | — | — | — | $0.100 | free |
| `embed-v4.0` | embed | 128,000 | — | — | — | — | — | — | $0.120 | free |

## databricks (23 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `databricks-bge-large-en` | embed | 512 | — | — | — | — | — | — | $0.100 | free |
| `databricks-claude-3-7-sonnet` | chat | 200,000 | 128,000 | ✅ | ✅ | ✅ | — | — | $3.00 | $15.00 |
| `databricks-claude-haiku-4-5` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $1.00 | $5.00 |
| `databricks-claude-opus-4` | chat | 200,000 | 32,000 | ✅ | ✅ | ✅ | — | — | $15.00 | $75.00 |
| `databricks-claude-opus-4-1` | chat | 200,000 | 32,000 | ✅ | ✅ | ✅ | — | — | $15.00 | $75.00 |
| `databricks-claude-opus-4-5` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $5.00 | $25.00 |
| `databricks-claude-sonnet-4` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $3.00 | $15.00 |
| `databricks-claude-sonnet-4-1` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $3.00 | $15.00 |
| `databricks-claude-sonnet-4-5` | chat | 200,000 | 64,000 | ✅ | ✅ | ✅ | — | — | $3.00 | $15.00 |
| `databricks-gemini-2-5-flash` | chat | 1,048,576 | 65,535 | ✅ | ✅ | ✅ | ✅ | — | $0.300 | $2.50 |
| `databricks-gemini-2-5-pro` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $1.25 | $10.00 |
| `databricks-gemma-3-12b` | chat | 128,000 | 32,000 | ✅ | — | — | — | — | $0.150 | $0.500 |
| `databricks-gpt-5` | chat | 272,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.25 | $10.00 |
| `databricks-gpt-5-1` | chat | 272,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.25 | $10.00 |
| `databricks-gpt-5-mini` | chat | 272,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.250 | $2.00 |
| `databricks-gpt-5-nano` | chat | 272,000 | 128,000 | — | — | ✅ | ✅ | ✅ | $0.050 | $0.400 |
| `databricks-gpt-oss-120b` | chat | 131,072 | 131,072 | — | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `databricks-gpt-oss-20b` | chat | 131,072 | 131,072 | — | — | ✅ | ✅ | — | $0.070 | $0.300 |
| `databricks-gte-large-en` | embed | 8,192 | — | — | — | — | — | — | $0.130 | free |
| `databricks-llama-4-maverick` | chat | 128,000 | 8,192 | ✅ | — | ✅ | ✅ | — | $0.500 | $1.50 |
| `databricks-meta-llama-3-1-405b-instruct` | chat | 128,000 | 8,192 | — | — | ✅ | ✅ | — | $5.00 | $15.00 |
| `databricks-meta-llama-3-1-8b-instruct` | chat | 128,000 | 8,192 | — | — | ✅ | ✅ | — | $0.150 | $0.450 |
| `databricks-meta-llama-3-3-70b-instruct` | chat | 128,000 | 8,192 | — | — | ✅ | ✅ | — | $0.500 | $1.50 |

## deepseek (4 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-chat` | chat | 1,000,000 | 384,000 | — | — | ✅ | ✅ | — | $0.140 | $0.280 |
| `deepseek-reasoner` | reason | 1,000,000 | 384,000 | — | — | ✅ | ✅ | ✅ | $0.140 | $0.280 |
| `deepseek-v4-flash` | chat | 1,000,000 | 384,000 | — | — | ✅ | ✅ | — | $0.140 | $0.280 |
| `deepseek-v4-pro` | chat | 1,000,000 | 384,000 | — | — | ✅ | ✅ | — | $1.74 | $3.48 |

## fireworks_ai (18 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `accounts/fireworks/models/deepseek-v3p1` | chat | 163,840 | 163,840 | — | — | ✅ | ✅ | — | $0.560 | $1.68 |
| `accounts/fireworks/models/deepseek-v3p2` | chat | 163,840 | 163,840 | — | — | ✅ | ✅ | — | $0.560 | $1.68 |
| `accounts/fireworks/models/deepseek-v4-pro` | chat | 1,048,576 | 32,768 | — | — | ✅ | ✅ | — | $1.74 | $3.48 |
| `accounts/fireworks/models/glm-4p7` | chat | 202,752 | 202,752 | — | — | ✅ | ✅ | — | $0.600 | $2.20 |
| `accounts/fireworks/models/glm-5` | chat | 202,752 | 202,752 | — | — | ✅ | ✅ | — | $1.00 | $3.20 |
| `accounts/fireworks/models/glm-5p1` | chat | 202,752 | 202,752 | — | — | ✅ | ✅ | — | $1.40 | $4.40 |
| `accounts/fireworks/models/gpt-oss-120b` | chat | 131,072 | 131,072 | — | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `accounts/fireworks/models/gpt-oss-20b` | chat | 131,072 | 131,072 | — | — | ✅ | ✅ | — | $0.070 | $0.300 |
| `accounts/fireworks/models/kimi-k2p5` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.600 | $3.00 |
| `accounts/fireworks/models/kimi-k2p6` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.950 | $4.00 |
| `accounts/fireworks/models/llama-v3p3-70b-instruct` | chat | 131,072 | 131,072 | — | — | ✅ | ✅ | — | $0.900 | $0.900 |
| `accounts/fireworks/models/minimax-m2p5` | chat | 196,608 | 196,608 | — | — | ✅ | ✅ | — | $0.300 | $1.20 |
| `accounts/fireworks/models/minimax-m2p7` | chat | 196,608 | 196,608 | — | — | ✅ | ✅ | — | $0.300 | $1.20 |
| `accounts/fireworks/models/qwen3-235b-a22b-thinking-2507` | chat | 262,144 | 81,920 | — | — | ✅ | ✅ | — | $0.220 | $0.880 |
| `accounts/fireworks/models/qwen3-embedding-8b` | embed | 32,000 | — | — | — | — | — | — | free | free |
| `accounts/fireworks/models/qwen3-vl-30b-a3b-instruct` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `accounts/fireworks/models/qwen3-vl-30b-a3b-thinking` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `accounts/fireworks/models/qwen3p6-plus` | chat | 262,144 | 32,768 | ✅ | — | ✅ | ✅ | — | $0.500 | $3.00 |

## gemini (15 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemini-2.0-flash` | chat | 1,048,576 | 8,192 | ✅ | ✅ | ✅ | ✅ | — | $0.100 | $0.400 |
| `gemini-2.0-flash-001` | chat | 1,048,576 | 8,192 | ✅ | ✅ | ✅ | ✅ | — | $0.100 | $0.400 |
| `gemini-2.0-flash-lite` | chat | 1,048,576 | 8,192 | ✅ | ✅ | ✅ | ✅ | — | $0.075 | $0.300 |
| `gemini-2.0-flash-lite-001` | chat | 1,048,576 | 8,192 | ✅ | ✅ | ✅ | ✅ | — | $0.075 | $0.300 |
| `gemini-2.5-flash` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $0.300 | $2.50 |
| `gemini-2.5-flash-lite` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $0.100 | $0.400 |
| `gemini-2.5-pro` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $1.25 | $10.00 |
| `gemini-3-flash-preview` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $0.500 | $3.00 |
| `gemini-3.1-flash-lite-preview` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $0.250 | $1.50 |
| `gemini-3.1-pro-preview` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $2.00 | $12.00 |
| `gemini-embedding-001` | embed | 2,048 | — | — | — | — | — | — | $0.150 | free |
| `gemini-embedding-2` | embed | 8,192 | — | — | — | — | — | — | $0.200 | free |
| `gemini-flash-latest` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $0.300 | $2.50 |
| `gemini-flash-lite-latest` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $0.100 | $0.400 |
| `gemini-pro-latest` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $1.25 | $10.00 |

## groq (7 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `llama-3.1-8b-instant` | chat | 131,072 | 131,072 | — | — | ✅ | ✅ | — | $0.050 | $0.080 |
| `llama-3.3-70b-versatile` | chat | 131,072 | 32,768 | — | — | ✅ | ✅ | — | $0.590 | $0.790 |
| `meta-llama/llama-4-scout-17b-16e-instruct` | chat | 131,072 | 8,192 | ✅ | — | ✅ | ✅ | — | $0.110 | $0.340 |
| `openai/gpt-oss-120b` | chat | 131,072 | 65,536 | — | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `openai/gpt-oss-20b` | chat | 131,072 | 65,536 | — | — | ✅ | ✅ | — | $0.075 | $0.300 |
| `openai/gpt-oss-safeguard-20b` | chat | 131,072 | 65,536 | — | — | ✅ | ✅ | — | $0.075 | $0.300 |
| `qwen/qwen3-32b` | chat | 131,072 | 40,960 | — | — | ✅ | — | — | $0.290 | $0.590 |

## mistral (25 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `codestral-2508` | chat | 256,000 | 256,000 | — | — | ✅ | ✅ | — | $0.300 | $0.900 |
| `codestral-embed` | embed | 8,192 | — | — | — | — | — | — | $0.150 | free |
| `codestral-latest` | chat | 256,000 | 256,000 | — | — | ✅ | ✅ | — | $0.300 | $0.900 |
| `devstral-2512` | chat | 262,144 | 262,144 | — | — | ✅ | ✅ | — | $0.400 | $2.00 |
| `devstral-latest` | chat | 262,144 | 262,144 | — | — | ✅ | ✅ | — | $0.400 | $2.00 |
| `devstral-medium-latest` | chat | 262,144 | 262,144 | — | — | ✅ | ✅ | — | $0.400 | $2.00 |
| `magistral-medium-2509` | reason | 131,072 | 131,072 | ✅ | — | ✅ | ✅ | ✅ | $2.00 | $5.00 |
| `magistral-medium-latest` | reason | 131,072 | 131,072 | ✅ | — | ✅ | ✅ | ✅ | $2.00 | $5.00 |
| `magistral-small-2509` | reason | 131,072 | 131,072 | ✅ | — | ✅ | ✅ | ✅ | $0.500 | $1.50 |
| `magistral-small-latest` | reason | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | ✅ | $0.500 | $1.50 |
| `ministral-14b-2512` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.200 | $0.200 |
| `ministral-14b-latest` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.200 | $0.200 |
| `ministral-3b-2512` | chat | 131,072 | 131,072 | ✅ | — | ✅ | ✅ | — | $0.100 | $0.100 |
| `ministral-3b-latest` | chat | 131,072 | 131,072 | ✅ | — | ✅ | ✅ | — | $0.100 | $0.100 |
| `ministral-8b-2512` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.150 | $0.150 |
| `ministral-8b-latest` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.150 | $0.150 |
| `mistral-embed` | embed | 8,192 | — | — | — | — | — | — | $0.100 | free |
| `mistral-large-2512` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.500 | $1.50 |
| `mistral-large-latest` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.500 | $1.50 |
| `mistral-medium-2508` | chat | 131,072 | 131,072 | ✅ | — | ✅ | ✅ | — | $0.400 | $2.00 |
| `mistral-medium-3-5` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.400 | $2.00 |
| `mistral-medium-latest` | chat | 131,072 | 131,072 | ✅ | — | ✅ | ✅ | — | $0.400 | $2.00 |
| `mistral-small-2603` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `mistral-small-latest` | chat | 262,144 | 262,144 | ✅ | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `open-mistral-nemo` | chat | 131,072 | 131,072 | — | — | ✅ | ✅ | — | $0.300 | $0.300 |

## ollama (18 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-r1` | reason | 131,072 | 32,768 | — | — | ✅ | ✅ | ✅ | free | free |
| `deepseek-r1:70b` | reason | 131,072 | 32,768 | — | — | ✅ | ✅ | ✅ | free | free |
| `gemma3` | chat | 131,072 | 8,192 | ✅ | — | — | ✅ | — | free | free |
| `gemma3:27b` | chat | 131,072 | 8,192 | ✅ | — | — | ✅ | — | free | free |
| `gpt-oss` | reason | 131,072 | 32,768 | — | — | ✅ | ✅ | ✅ | free | free |
| `llama3.1` | chat | 131,072 | 4,096 | — | — | ✅ | ✅ | — | free | free |
| `llama3.2-vision` | chat | 131,072 | 4,096 | ✅ | — | — | ✅ | — | free | free |
| `llama3.3` | chat | 131,072 | 4,096 | — | — | ✅ | ✅ | — | free | free |
| `llava` | chat | 32,768 | 4,096 | ✅ | — | — | — | — | free | free |
| `mistral-nemo` | chat | 131,072 | 8,192 | — | — | ✅ | ✅ | — | free | free |
| `mixtral:8x7b` | chat | 32,768 | 8,192 | — | — | — | ✅ | — | free | free |
| `mxbai-embed-large` | embed | 512 | — | — | — | — | — | — | free | free |
| `nomic-embed-text` | embed | 2,048 | — | — | — | — | — | — | free | free |
| `phi4` | chat | 16,384 | 16,384 | — | — | — | ✅ | — | free | free |
| `qwen2.5` | chat | 32,768 | 8,192 | — | — | ✅ | ✅ | — | free | free |
| `qwen2.5-coder` | chat | 32,768 | 8,192 | — | — | ✅ | ✅ | — | free | free |
| `qwen3` | chat | 40,960 | 8,192 | — | — | ✅ | ✅ | — | free | free |
| `qwen3:32b` | chat | 40,960 | 8,192 | — | — | ✅ | ✅ | — | free | free |

## openai (37 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `gpt-4.1` | chat | 1,047,576 | 32,768 | ✅ | — | ✅ | ✅ | — | $2.00 | $8.00 |
| `gpt-4.1-mini` | chat | 1,047,576 | 32,768 | ✅ | — | ✅ | ✅ | — | $0.400 | $1.60 |
| `gpt-4.1-nano` | chat | 1,047,576 | 32,768 | ✅ | — | ✅ | ✅ | — | $0.100 | $0.400 |
| `gpt-4o` | chat | 128,000 | 16,384 | ✅ | — | ✅ | ✅ | — | $2.50 | $10.00 |
| `gpt-4o-mini` | chat | 128,000 | 16,384 | ✅ | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `gpt-5` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.25 | $10.00 |
| `gpt-5-mini` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.250 | $2.00 |
| `gpt-5-nano` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.050 | $0.400 |
| `gpt-5-pro` | reason | 400,000 | 272,000 | ✅ | — | ✅ | ✅ | ✅ | $15.00 | $120.00 |
| `gpt-5.1` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.25 | $10.00 |
| `gpt-5.1-codex` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.25 | $10.00 |
| `gpt-5.1-codex-max` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.25 | $10.00 |
| `gpt-5.1-codex-mini` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.250 | $2.00 |
| `gpt-5.2` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.75 | $14.00 |
| `gpt-5.2-codex` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $1.75 | $14.00 |
| `gpt-5.2-pro` | reason | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $21.00 | $168.00 |
| `gpt-5.4` | chat | 1,050,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $2.50 | $15.00 |
| `gpt-5.4-mini` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.750 | $4.50 |
| `gpt-5.4-nano` | chat | 400,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $0.200 | $1.25 |
| `gpt-5.4-pro` | reason | 1,050,000 | 128,000 | ✅ | — | ✅ | — | ✅ | $30.00 | $180.00 |
| `gpt-5.5` | chat | 1,050,000 | 128,000 | ✅ | — | ✅ | ✅ | ✅ | $5.00 | $30.00 |
| `gpt-5.5-pro` | reason | 1,050,000 | 128,000 | ✅ | — | ✅ | — | ✅ | $30.00 | $180.00 |
| `gpt-audio` | chat | 128,000 | 16,384 | — | — | ✅ | — | — | $2.50 | $10.00 |
| `gpt-audio-1.5` | chat | 128,000 | 16,384 | — | — | ✅ | — | — | $2.50 | $10.00 |
| `gpt-audio-mini` | chat | 128,000 | 16,384 | — | — | ✅ | — | — | $0.600 | $2.40 |
| `gpt-realtime` | chat | 32,000 | 4,096 | ✅ | — | ✅ | — | — | $4.00 | $16.00 |
| `gpt-realtime-1.5` | chat | 32,000 | 4,096 | ✅ | — | ✅ | — | — | $4.00 | $16.00 |
| `gpt-realtime-mini` | chat | 32,000 | 4,096 | — | — | ✅ | — | — | $0.600 | $2.40 |
| `o1` | reason | 200,000 | 100,000 | ✅ | — | ✅ | ✅ | ✅ | $15.00 | $60.00 |
| `o1-pro` | reason | 200,000 | 100,000 | ✅ | — | ✅ | ✅ | ✅ | $150.00 | $600.00 |
| `o3` | reason | 200,000 | 100,000 | ✅ | — | ✅ | ✅ | ✅ | $2.00 | $8.00 |
| `o3-mini` | reason | 200,000 | 100,000 | — | — | ✅ | ✅ | ✅ | $1.10 | $4.40 |
| `o3-pro` | reason | 200,000 | 100,000 | ✅ | — | ✅ | ✅ | ✅ | $20.00 | $80.00 |
| `o4-mini` | reason | 200,000 | 100,000 | ✅ | — | ✅ | ✅ | ✅ | $1.10 | $4.40 |
| `text-embedding-3-large` | embed | 8,191 | — | — | — | — | — | — | $0.130 | free |
| `text-embedding-3-small` | embed | 8,191 | — | — | — | — | — | — | $0.020 | free |
| `text-embedding-ada-002` | embed | 8,191 | — | — | — | — | — | — | $0.100 | free |

## perplexity (4 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `sonar` | chat | 128,000 | 128,000 | ✅ | ✅ | — | ✅ | — | $1.00 | $1.00 |
| `sonar-deep-research` | reason | 128,000 | 128,000 | ✅ | ✅ | — | ✅ | ✅ | $2.00 | $8.00 |
| `sonar-pro` | chat | 200,000 | 8,000 | ✅ | ✅ | — | ✅ | — | $3.00 | $15.00 |
| `sonar-reasoning-pro` | reason | 128,000 | 128,000 | ✅ | ✅ | — | ✅ | ✅ | $2.00 | $8.00 |

## together_ai (47 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `LiquidAI/LFM2-24B-A2B` | chat | 32,768 | — | — | — | — | — | — | $0.030 | $0.120 |
| `MiniMaxAI/MiniMax-M2.7` | chat | 196,608 | — | — | — | ✅ | ✅ | — | $0.300 | $1.20 |
| `Qwen/QwQ-32B` | reason | 131,072 | — | — | — | — | — | ✅ | $1.20 | $1.20 |
| `Qwen/Qwen2.5-72B-Instruct-Turbo` | chat | 131,072 | — | — | — | ✅ | ✅ | — | $1.20 | $1.20 |
| `Qwen/Qwen2.5-7B-Instruct-Turbo` | chat | 32,768 | — | — | — | ✅ | ✅ | — | $0.300 | $0.300 |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | chat | 16,384 | — | — | — | — | — | — | $0.800 | $0.800 |
| `Qwen/Qwen2.5-VL-72B-Instruct` | chat | 32,768 | — | ✅ | — | — | — | — | $1.95 | $8.00 |
| `Qwen/Qwen3-235B-A22B-Instruct-2507-tput` | chat | 262,144 | — | — | — | ✅ | ✅ | — | $0.200 | $0.600 |
| `Qwen/Qwen3-235B-A22B-Thinking-2507` | reason | 262,144 | — | — | — | ✅ | ✅ | ✅ | $0.650 | $3.00 |
| `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | chat | 262,144 | — | — | — | ✅ | ✅ | — | $2.00 | $2.00 |
| `Qwen/Qwen3-Coder-Next-FP8` | chat | 262,144 | — | — | — | — | — | — | $0.500 | $1.20 |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | chat | 262,144 | — | — | — | ✅ | ✅ | — | $0.150 | $1.50 |
| `Qwen/Qwen3-Next-80B-A3B-Thinking` | reason | 262,144 | — | — | — | ✅ | ✅ | ✅ | $0.150 | $1.50 |
| `Qwen/Qwen3-VL-32B-Instruct` | chat | 262,144 | — | ✅ | — | — | — | — | $0.500 | $1.50 |
| `Qwen/Qwen3-VL-8B-Instruct` | chat | 262,144 | — | ✅ | — | — | — | — | $0.180 | $0.680 |
| `Qwen/Qwen3.5-397B-A17B` | chat | 262,144 | — | ✅ | — | ✅ | ✅ | — | $0.600 | $3.60 |
| `Qwen/Qwen3.5-9B` | chat | 262,144 | — | ✅ | — | ✅ | ✅ | — | $0.100 | $0.150 |
| `Qwen/Qwen3.6-Plus` | chat | 1,000,000 | — | — | — | — | — | — | $0.500 | $3.00 |
| `deepcogito/cogito-v2-1-671b` | chat | 163,840 | — | — | — | — | — | — | $1.25 | $1.25 |
| `deepseek-ai/DeepSeek-R1` | reason | 163,840 | 20,480 | — | — | ✅ | ✅ | ✅ | $3.00 | $7.00 |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | reason | 131,072 | — | — | — | — | — | ✅ | $2.00 | $2.00 |
| `deepseek-ai/DeepSeek-V3.1` | chat | 131,072 | 16,384 | — | — | ✅ | ✅ | — | $0.600 | $1.70 |
| `deepseek-ai/DeepSeek-V4-Pro` | chat | 512,000 | — | — | — | ✅ | ✅ | — | $2.10 | $4.40 |
| `essentialai/rnj-1-instruct` | chat | 32,768 | — | — | — | ✅ | ✅ | — | $0.150 | $0.150 |
| `google/gemma-3n-E4B-it` | chat | 32,768 | — | — | — | — | ✅ | — | $0.060 | $0.120 |
| `google/gemma-4-31B-it` | chat | 262,144 | — | — | — | ✅ | ✅ | — | $0.200 | $0.500 |
| `intfloat/multilingual-e5-large-instruct` | embed | 514 | — | — | — | — | — | — | $0.020 | free |
| `meta-llama/Llama-3.1-405B-Instruct` | chat | 4,096 | — | — | — | ✅ | ✅ | — | $3.50 | $3.50 |
| `meta-llama/Llama-3.2-1B-Instruct` | chat | 131,072 | — | — | — | — | — | — | $0.060 | $0.060 |
| `meta-llama/Llama-3.3-70B-Instruct-Turbo` | chat | 131,072 | — | — | — | ✅ | ✅ | — | $0.880 | $0.880 |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | chat | 1,048,576 | — | ✅ | — | ✅ | ✅ | — | $0.270 | $0.850 |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | chat | 1,048,576 | — | ✅ | — | ✅ | ✅ | — | $0.180 | $0.590 |
| `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` | chat | 131,072 | — | — | — | ✅ | ✅ | — | $0.880 | $0.880 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | chat | 131,072 | — | — | — | ✅ | ✅ | — | $0.180 | $0.180 |
| `mistralai/Ministral-3-14B-Instruct-2512` | chat | 262,144 | — | — | — | — | — | — | $0.200 | $0.200 |
| `mistralai/Mistral-Small-24B-Instruct-2501` | chat | 32,768 | — | — | — | ✅ | — | — | $0.100 | $0.300 |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | chat | 32,768 | — | — | — | ✅ | ✅ | — | $0.600 | $0.600 |
| `moonshotai/Kimi-K2.5` | chat | 262,144 | — | ✅ | — | ✅ | ✅ | — | $0.500 | $2.80 |
| `moonshotai/Kimi-K2.6` | chat | 262,144 | — | — | — | ✅ | ✅ | — | $1.20 | $4.50 |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | chat | 131,072 | — | — | — | — | — | — | $0.060 | $0.250 |
| `openai/gpt-oss-120b` | chat | 131,072 | 131,072 | — | — | ✅ | ✅ | — | $0.150 | $0.600 |
| `openai/gpt-oss-20b` | chat | 131,072 | — | — | — | ✅ | ✅ | — | $0.050 | $0.200 |
| `zai-org/GLM-4.5-Air-FP8` | chat | 131,072 | — | — | — | ✅ | ✅ | — | $0.200 | $1.10 |
| `zai-org/GLM-4.6` | chat | 202,752 | 200,000 | — | — | ✅ | ✅ | — | $0.600 | $2.20 |
| `zai-org/GLM-4.7` | chat | 202,752 | 200,000 | — | — | ✅ | ✅ | — | $0.450 | $2.00 |
| `zai-org/GLM-5` | chat | 202,752 | — | — | — | ✅ | ✅ | — | $1.00 | $3.20 |
| `zai-org/GLM-5.1` | chat | 202,752 | — | — | — | ✅ | ✅ | — | $1.40 | $4.40 |

## vertex_ai (15 models)

| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemini-2.0-flash` | chat | 1,048,576 | 8,192 | ✅ | ✅ | ✅ | ✅ | — | $0.150 | $0.600 |
| `gemini-2.0-flash-001` | chat | 1,048,576 | 8,192 | ✅ | ✅ | ✅ | ✅ | — | $0.150 | $0.600 |
| `gemini-2.0-flash-lite` | chat | 1,048,576 | 8,192 | ✅ | ✅ | ✅ | ✅ | — | $0.075 | $0.300 |
| `gemini-2.0-flash-lite-001` | chat | 1,048,576 | 8,192 | ✅ | ✅ | ✅ | ✅ | — | $0.075 | $0.300 |
| `gemini-2.5-flash` | chat | 1,048,576 | 65,535 | ✅ | ✅ | ✅ | ✅ | — | $0.300 | $2.50 |
| `gemini-2.5-flash-lite` | chat | 1,048,576 | 65,535 | ✅ | ✅ | ✅ | ✅ | — | $0.100 | $0.400 |
| `gemini-2.5-pro` | chat | 1,048,576 | 65,535 | ✅ | ✅ | ✅ | ✅ | — | $1.25 | $10.00 |
| `gemini-3-flash-preview` | chat | 1,048,576 | 65,535 | ✅ | ✅ | ✅ | ✅ | — | $0.500 | $3.00 |
| `gemini-3-pro-preview` | chat | 1,048,576 | 65,535 | ✅ | ✅ | ✅ | ✅ | — | $2.00 | $12.00 |
| `gemini-3.1-flash-lite-preview` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $0.250 | $1.50 |
| `gemini-3.1-pro-preview` | chat | 1,048,576 | 65,536 | ✅ | ✅ | ✅ | ✅ | — | $2.00 | $12.00 |
| `gemini-embedding-001` | embed | 2,048 | — | — | — | — | — | — | $0.150 | free |
| `gemini-embedding-2` | embed | 8,192 | — | — | — | — | — | — | $0.200 | free |
| `text-embedding-005` | embed | 2,048 | — | — | — | — | — | — | $0.100 | free |
| `text-multilingual-embedding-002` | embed | 2,048 | — | — | — | — | — | — | $0.100 | free |
