"""
Pricing tables for supported LLM models.

Prices are in USD per 1 million tokens.
Generated from `tmp/model_manifests/` by `scripts/sync_tables.py`.
Last updated: 2026-05-07

To update prices:
    1. Refresh the manifests: see plan + AGENTS.md for the agent-driven workflow.
    2. Run `python scripts/sync_tables.py`.
    3. Bump `PRICING_VERSION` if the format / public API changes.
    4. Run `pytest tests/test_pricing.py tests/test_tables_parity.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from arcllm.exceptions import ArcLLMError

if TYPE_CHECKING:
    from arcllm.types import ModelResponse

__all__ = [
    "PRICING_VERSION",
    "ModelPricing",
    "completion_cost",
    "cost_per_token",
    "get_model_pricing",
]


PRICING_VERSION = "2026.05.07"


@dataclass(slots=True, frozen=True)
class ModelPricing:
    """Pricing information for a model (USD per 1M tokens)."""

    input_cost_per_million: float
    output_cost_per_million: float
    # Optional: cached input price for prompt caching (None if not supported).
    cached_input_cost_per_million: float | None = None


# =============================================================================
# openai
# =============================================================================
OPENAI_PRICING: dict[str, ModelPricing] = {
    "gpt-5.5": ModelPricing(5.0, 30.0, 0.5),
    "gpt-5.5-pro": ModelPricing(30.0, 180.0),
    "gpt-5.4": ModelPricing(2.5, 15.0, 0.25),
    "gpt-5.4-mini": ModelPricing(0.75, 4.5, 0.075),
    "gpt-5.4-nano": ModelPricing(0.2, 1.25, 0.02),
    "gpt-5.4-pro": ModelPricing(30.0, 180.0),
    "gpt-5.2": ModelPricing(1.75, 14.0, 0.175),
    "gpt-5.2-codex": ModelPricing(1.75, 14.0, 0.175),
    "gpt-5.2-pro": ModelPricing(21.0, 168.0),
    "gpt-5.1": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5.1-codex": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5.1-codex-mini": ModelPricing(0.25, 2.0, 0.025),
    "gpt-5.1-codex-max": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5-mini": ModelPricing(0.25, 2.0, 0.025),
    "gpt-5-nano": ModelPricing(0.05, 0.4, 0.005),
    "gpt-5-pro": ModelPricing(15.0, 120.0),
    "gpt-4.1": ModelPricing(2.0, 8.0, 0.5),
    "gpt-4.1-mini": ModelPricing(0.4, 1.6, 0.1),
    "gpt-4.1-nano": ModelPricing(0.1, 0.4, 0.025),
    "gpt-4o": ModelPricing(2.5, 10.0, 1.25),
    "gpt-4o-mini": ModelPricing(0.15, 0.6, 0.075),
    "o1": ModelPricing(15.0, 60.0, 7.5),
    "o1-pro": ModelPricing(150.0, 600.0),
    "o3": ModelPricing(2.0, 8.0, 0.5),
    "o3-mini": ModelPricing(1.1, 4.4, 0.55),
    "o3-pro": ModelPricing(20.0, 80.0),
    "o4-mini": ModelPricing(1.1, 4.4, 0.275),
    "gpt-audio": ModelPricing(2.5, 10.0),
    "gpt-audio-mini": ModelPricing(0.6, 2.4),
    "gpt-audio-1.5": ModelPricing(2.5, 10.0),
    "gpt-realtime": ModelPricing(4.0, 16.0, 0.4),
    "gpt-realtime-mini": ModelPricing(0.6, 2.4),
    "gpt-realtime-1.5": ModelPricing(4.0, 16.0, 0.4),
    "text-embedding-3-small": ModelPricing(0.02, 0.0),
    "text-embedding-3-large": ModelPricing(0.13, 0.0),
    "text-embedding-ada-002": ModelPricing(0.1, 0.0),
}

# =============================================================================
# azure
# =============================================================================
AZURE_PRICING: dict[str, ModelPricing] = {
    "gpt-5.5": ModelPricing(5.0, 30.0, 0.5),
    "gpt-5.5-pro": ModelPricing(30.0, 180.0),
    "gpt-5.4": ModelPricing(2.5, 15.0, 0.25),
    "gpt-5.4-mini": ModelPricing(0.75, 4.5, 0.075),
    "gpt-5.4-nano": ModelPricing(0.2, 1.25, 0.02),
    "gpt-5.4-pro": ModelPricing(30.0, 180.0),
    "gpt-5.2": ModelPricing(1.75, 14.0, 0.175),
    "gpt-5.2-codex": ModelPricing(1.75, 14.0, 0.175),
    "gpt-5.2-pro": ModelPricing(21.0, 168.0),
    "gpt-5.1": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5.1-codex": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5.1-codex-mini": ModelPricing(0.25, 2.0, 0.025),
    "gpt-5.1-codex-max": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5-mini": ModelPricing(0.25, 2.0, 0.025),
    "gpt-5-nano": ModelPricing(0.05, 0.4, 0.005),
    "gpt-5-pro": ModelPricing(15.0, 120.0),
    "gpt-4.1": ModelPricing(2.0, 8.0, 0.5),
    "gpt-4.1-mini": ModelPricing(0.4, 1.6, 0.1),
    "gpt-4.1-nano": ModelPricing(0.1, 0.4, 0.025),
    "gpt-4o": ModelPricing(2.5, 10.0, 1.25),
    "gpt-4o-mini": ModelPricing(0.15, 0.6, 0.075),
    "o1": ModelPricing(15.0, 60.0, 7.5),
    "o3": ModelPricing(2.0, 8.0, 0.5),
    "o3-mini": ModelPricing(1.1, 4.4, 0.55),
    "o3-pro": ModelPricing(20.0, 80.0),
    "o4-mini": ModelPricing(1.1, 4.4, 0.275),
    "gpt-audio": ModelPricing(2.5, 10.0),
    "gpt-audio-mini": ModelPricing(0.6, 2.4),
    "gpt-audio-1.5": ModelPricing(2.5, 10.0),
    "gpt-realtime": ModelPricing(4.0, 16.0, 0.4),
    "gpt-realtime-mini": ModelPricing(0.6, 2.4),
    "gpt-realtime-1.5": ModelPricing(4.0, 16.0, 0.4),
    "text-embedding-3-small": ModelPricing(0.02, 0.0),
    "text-embedding-3-large": ModelPricing(0.13, 0.0),
    "text-embedding-ada-002": ModelPricing(0.1, 0.0),
}

# =============================================================================
# anthropic
# =============================================================================
ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-7": ModelPricing(5.0, 25.0, 0.5),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0, 0.3),
    "claude-opus-4-6": ModelPricing(5.0, 25.0, 0.5),
    "claude-haiku-4-5-20251001": ModelPricing(1.0, 5.0, 0.1),
    "claude-haiku-4-5": ModelPricing(1.0, 5.0, 0.1),
    "claude-sonnet-4-5-20250929": ModelPricing(3.0, 15.0, 0.3),
    "claude-sonnet-4-5": ModelPricing(3.0, 15.0, 0.3),
    "claude-opus-4-5-20251101": ModelPricing(5.0, 25.0, 0.5),
    "claude-opus-4-5": ModelPricing(5.0, 25.0, 0.5),
    "claude-opus-4-1-20250805": ModelPricing(15.0, 75.0, 1.5),
    "claude-opus-4-1": ModelPricing(15.0, 75.0, 1.5),
}

# =============================================================================
# gemini
# =============================================================================
GEMINI_PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-pro": ModelPricing(1.25, 10.0, 0.125),
    "gemini-2.5-flash": ModelPricing(0.3, 2.5, 0.03),
    "gemini-2.5-flash-lite": ModelPricing(0.1, 0.4, 0.01),
    "gemini-2.0-flash": ModelPricing(0.1, 0.4, 0.025),
    "gemini-2.0-flash-001": ModelPricing(0.1, 0.4, 0.025),
    "gemini-2.0-flash-lite": ModelPricing(0.075, 0.3, 0.01875),
    "gemini-2.0-flash-lite-001": ModelPricing(0.075, 0.3, 0.01875),
    "gemini-flash-latest": ModelPricing(0.3, 2.5, 0.03),
    "gemini-flash-lite-latest": ModelPricing(0.1, 0.4, 0.01),
    "gemini-pro-latest": ModelPricing(1.25, 10.0, 0.125),
    "gemini-3.1-pro-preview": ModelPricing(2.0, 12.0, 0.2),
    "gemini-3-flash-preview": ModelPricing(0.5, 3.0, 0.05),
    "gemini-3.1-flash-lite-preview": ModelPricing(0.25, 1.5, 0.025),
    "gemini-embedding-001": ModelPricing(0.15, 0.0),
    "gemini-embedding-2": ModelPricing(0.2, 0.0),
}

# =============================================================================
# vertex_ai
# =============================================================================
VERTEX_AI_PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-pro": ModelPricing(1.25, 10.0, 0.125),
    "gemini-2.5-flash": ModelPricing(0.3, 2.5, 0.03),
    "gemini-2.5-flash-lite": ModelPricing(0.1, 0.4, 0.01),
    "gemini-2.0-flash": ModelPricing(0.15, 0.6, 0.0375),
    "gemini-2.0-flash-001": ModelPricing(0.15, 0.6, 0.0375),
    "gemini-2.0-flash-lite": ModelPricing(0.075, 0.3, 0.01875),
    "gemini-2.0-flash-lite-001": ModelPricing(0.075, 0.3, 0.01875),
    "gemini-3-pro-preview": ModelPricing(2.0, 12.0, 0.2),
    "gemini-3-flash-preview": ModelPricing(0.5, 3.0, 0.05),
    "gemini-3.1-pro-preview": ModelPricing(2.0, 12.0, 0.2),
    "gemini-3.1-flash-lite-preview": ModelPricing(0.25, 1.5, 0.025),
    "text-embedding-005": ModelPricing(0.1, 0.0),
    "text-multilingual-embedding-002": ModelPricing(0.1, 0.0),
    "gemini-embedding-001": ModelPricing(0.15, 0.0),
    "gemini-embedding-2": ModelPricing(0.2, 0.0),
}

# =============================================================================
# bedrock
# =============================================================================
BEDROCK_PRICING: dict[str, ModelPricing] = {
    "anthropic.claude-sonnet-4-5-20250929-v1:0": ModelPricing(3.0, 15.0, 0.3),
    "anthropic.claude-haiku-4-5-20251001-v1:0": ModelPricing(1.0, 5.0, 0.1),
    "anthropic.claude-opus-4-5-20251101-v1:0": ModelPricing(5.0, 25.0, 0.5),
    "anthropic.claude-opus-4-1-20250805-v1:0": ModelPricing(15.0, 75.0, 1.5),
    "anthropic.claude-opus-4-20250514-v1:0": ModelPricing(15.0, 75.0, 1.5),
    "anthropic.claude-sonnet-4-20250514-v1:0": ModelPricing(3.0, 15.0, 0.3),
    "anthropic.claude-3-7-sonnet-20250219-v1:0": ModelPricing(3.0, 15.0, 0.3),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelPricing(3.0, 15.0, 0.3),
    "anthropic.claude-3-5-haiku-20241022-v1:0": ModelPricing(0.8, 4.0, 0.08),
    "anthropic.claude-3-haiku-20240307-v1:0": ModelPricing(0.25, 1.25, 0.03),
    "anthropic.claude-3-opus-20240229-v1:0": ModelPricing(15.0, 75.0, 1.5),
    "amazon.nova-pro-v1:0": ModelPricing(0.8, 3.2),
    "amazon.nova-lite-v1:0": ModelPricing(0.06, 0.24),
    "amazon.nova-micro-v1:0": ModelPricing(0.035, 0.14),
    "meta.llama3-3-70b-instruct-v1:0": ModelPricing(0.72, 0.72),
    "meta.llama3-2-90b-instruct-v1:0": ModelPricing(2.0, 2.0),
    "meta.llama3-2-11b-instruct-v1:0": ModelPricing(0.35, 0.35),
    "meta.llama3-2-3b-instruct-v1:0": ModelPricing(0.15, 0.15),
    "meta.llama3-2-1b-instruct-v1:0": ModelPricing(0.1, 0.1),
    "meta.llama3-1-405b-instruct-v1:0": ModelPricing(5.32, 16.0),
    "meta.llama3-1-70b-instruct-v1:0": ModelPricing(0.99, 0.99),
    "meta.llama3-1-8b-instruct-v1:0": ModelPricing(0.22, 0.22),
    "meta.llama4-maverick-17b-instruct-v1:0": ModelPricing(0.24, 0.97),
    "meta.llama4-scout-17b-instruct-v1:0": ModelPricing(0.17, 0.66),
    "mistral.mistral-large-2407-v1:0": ModelPricing(3.0, 9.0),
    "cohere.command-r-plus-v1:0": ModelPricing(3.0, 15.0),
    "cohere.command-r-v1:0": ModelPricing(0.5, 1.5),
    "amazon.titan-embed-text-v2:0": ModelPricing(0.2, 0.0),
    "amazon.titan-embed-text-v1": ModelPricing(0.1, 0.0),
    "cohere.embed-english-v3": ModelPricing(0.1, 0.0),
    "cohere.embed-multilingual-v3": ModelPricing(0.1, 0.0),
}

# =============================================================================
# mistral
# =============================================================================
MISTRAL_PRICING: dict[str, ModelPricing] = {
    "mistral-large-latest": ModelPricing(0.5, 1.5),
    "mistral-large-2512": ModelPricing(0.5, 1.5),
    "mistral-medium-latest": ModelPricing(0.4, 2.0),
    "mistral-medium-2508": ModelPricing(0.4, 2.0),
    "mistral-medium-3-5": ModelPricing(0.4, 2.0),
    "mistral-small-latest": ModelPricing(0.15, 0.6),
    "mistral-small-2603": ModelPricing(0.15, 0.6),
    "ministral-3b-latest": ModelPricing(0.1, 0.1),
    "ministral-3b-2512": ModelPricing(0.1, 0.1),
    "ministral-8b-latest": ModelPricing(0.15, 0.15),
    "ministral-8b-2512": ModelPricing(0.15, 0.15),
    "ministral-14b-latest": ModelPricing(0.2, 0.2),
    "ministral-14b-2512": ModelPricing(0.2, 0.2),
    "open-mistral-nemo": ModelPricing(0.3, 0.3),
    "codestral-latest": ModelPricing(0.3, 0.9),
    "codestral-2508": ModelPricing(0.3, 0.9),
    "devstral-latest": ModelPricing(0.4, 2.0),
    "devstral-2512": ModelPricing(0.4, 2.0),
    "devstral-medium-latest": ModelPricing(0.4, 2.0),
    "magistral-medium-latest": ModelPricing(2.0, 5.0),
    "magistral-medium-2509": ModelPricing(2.0, 5.0),
    "magistral-small-latest": ModelPricing(0.5, 1.5),
    "magistral-small-2509": ModelPricing(0.5, 1.5),
    "mistral-embed": ModelPricing(0.1, 0.0),
    "codestral-embed": ModelPricing(0.15, 0.0),
}

# =============================================================================
# cohere
# =============================================================================
COHERE_PRICING: dict[str, ModelPricing] = {
    "command-a-03-2025": ModelPricing(2.5, 10.0),
    "command-a-reasoning-08-2025": ModelPricing(2.5, 10.0),
    "command-a-vision-07-2025": ModelPricing(2.5, 10.0),
    "command-r-plus-08-2024": ModelPricing(2.5, 10.0),
    "command-r-08-2024": ModelPricing(0.15, 0.6),
    "command-r7b-12-2024": ModelPricing(0.0375, 0.15),
    "embed-v4.0": ModelPricing(0.12, 0.0),
    "embed-english-v3.0": ModelPricing(0.1, 0.0),
    "embed-multilingual-v3.0": ModelPricing(0.1, 0.0),
    "embed-english-light-v3.0": ModelPricing(0.1, 0.0),
    "embed-multilingual-light-v3.0": ModelPricing(0.1, 0.0),
}

# =============================================================================
# groq
# =============================================================================
GROQ_PRICING: dict[str, ModelPricing] = {
    "llama-3.3-70b-versatile": ModelPricing(0.59, 0.79),
    "llama-3.1-8b-instant": ModelPricing(0.05, 0.08),
    "openai/gpt-oss-120b": ModelPricing(0.15, 0.6),
    "openai/gpt-oss-20b": ModelPricing(0.075, 0.3),
    "meta-llama/llama-4-scout-17b-16e-instruct": ModelPricing(0.11, 0.34),
    "openai/gpt-oss-safeguard-20b": ModelPricing(0.075, 0.3),
    "qwen/qwen3-32b": ModelPricing(0.29, 0.59),
}

# =============================================================================
# together_ai
# =============================================================================
TOGETHER_PRICING: dict[str, ModelPricing] = {
    "deepseek-ai/DeepSeek-V4-Pro": ModelPricing(2.1, 4.4, 0.2),
    "deepseek-ai/DeepSeek-V3.1": ModelPricing(0.6, 1.7),
    "deepseek-ai/DeepSeek-R1": ModelPricing(3.0, 7.0),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ModelPricing(2.0, 2.0),
    "deepcogito/cogito-v2-1-671b": ModelPricing(1.25, 1.25),
    "moonshotai/Kimi-K2.6": ModelPricing(1.2, 4.5, 0.2),
    "moonshotai/Kimi-K2.5": ModelPricing(0.5, 2.8),
    "MiniMaxAI/MiniMax-M2.7": ModelPricing(0.3, 1.2, 0.06),
    "zai-org/GLM-5.1": ModelPricing(1.4, 4.4),
    "zai-org/GLM-5": ModelPricing(1.0, 3.2),
    "zai-org/GLM-4.7": ModelPricing(0.45, 2.0),
    "zai-org/GLM-4.6": ModelPricing(0.6, 2.2),
    "zai-org/GLM-4.5-Air-FP8": ModelPricing(0.2, 1.1),
    "Qwen/Qwen3.6-Plus": ModelPricing(0.5, 3.0),
    "Qwen/Qwen3.5-397B-A17B": ModelPricing(0.6, 3.6),
    "Qwen/Qwen3.5-9B": ModelPricing(0.1, 0.15),
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": ModelPricing(0.2, 0.6),
    "Qwen/Qwen3-235B-A22B-Thinking-2507": ModelPricing(0.65, 3.0),
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": ModelPricing(2.0, 2.0),
    "Qwen/Qwen3-Coder-Next-FP8": ModelPricing(0.5, 1.2),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": ModelPricing(0.15, 1.5),
    "Qwen/Qwen3-Next-80B-A3B-Thinking": ModelPricing(0.15, 1.5),
    "Qwen/Qwen3-VL-32B-Instruct": ModelPricing(0.5, 1.5),
    "Qwen/Qwen3-VL-8B-Instruct": ModelPricing(0.18, 0.68),
    "Qwen/QwQ-32B": ModelPricing(1.2, 1.2),
    "Qwen/Qwen2.5-72B-Instruct-Turbo": ModelPricing(1.2, 1.2),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": ModelPricing(0.3, 0.3),
    "Qwen/Qwen2.5-Coder-32B-Instruct": ModelPricing(0.8, 0.8),
    "Qwen/Qwen2.5-VL-72B-Instruct": ModelPricing(1.95, 8.0),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": ModelPricing(0.27, 0.85),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": ModelPricing(0.18, 0.59),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelPricing(0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelPricing(0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelPricing(0.18, 0.18),
    "meta-llama/Llama-3.1-405B-Instruct": ModelPricing(3.5, 3.5),
    "meta-llama/Llama-3.2-1B-Instruct": ModelPricing(0.06, 0.06),
    "openai/gpt-oss-120b": ModelPricing(0.15, 0.6),
    "openai/gpt-oss-20b": ModelPricing(0.05, 0.2),
    "google/gemma-4-31B-it": ModelPricing(0.2, 0.5),
    "google/gemma-3n-E4B-it": ModelPricing(0.06, 0.12),
    "mistralai/Mistral-Small-24B-Instruct-2501": ModelPricing(0.1, 0.3),
    "mistralai/Ministral-3-14B-Instruct-2512": ModelPricing(0.2, 0.2),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelPricing(0.6, 0.6),
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": ModelPricing(0.06, 0.25),
    "LiquidAI/LFM2-24B-A2B": ModelPricing(0.03, 0.12),
    "essentialai/rnj-1-instruct": ModelPricing(0.15, 0.15),
    "intfloat/multilingual-e5-large-instruct": ModelPricing(0.02, 0.0),
}

# =============================================================================
# fireworks_ai
# =============================================================================
FIREWORKS_PRICING: dict[str, ModelPricing] = {
    "accounts/fireworks/models/deepseek-v4-pro": ModelPricing(1.74, 3.48, 0.145),
    "accounts/fireworks/models/deepseek-v3p2": ModelPricing(0.56, 1.68, 0.28),
    "accounts/fireworks/models/deepseek-v3p1": ModelPricing(0.56, 1.68, 0.28),
    "accounts/fireworks/models/glm-5p1": ModelPricing(1.4, 4.4, 0.26),
    "accounts/fireworks/models/glm-5": ModelPricing(1.0, 3.2, 0.2),
    "accounts/fireworks/models/glm-4p7": ModelPricing(0.6, 2.2, 0.3),
    "accounts/fireworks/models/kimi-k2p6": ModelPricing(0.95, 4.0, 0.16),
    "accounts/fireworks/models/kimi-k2p5": ModelPricing(0.6, 3.0, 0.1),
    "accounts/fireworks/models/qwen3p6-plus": ModelPricing(0.5, 3.0, 0.1),
    "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507": ModelPricing(0.22, 0.88, 0.11),
    "accounts/fireworks/models/qwen3-vl-30b-a3b-thinking": ModelPricing(0.15, 0.6, 0.07),
    "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct": ModelPricing(0.15, 0.6),
    "accounts/fireworks/models/minimax-m2p7": ModelPricing(0.3, 1.2, 0.06),
    "accounts/fireworks/models/minimax-m2p5": ModelPricing(0.3, 1.2, 0.03),
    "accounts/fireworks/models/gpt-oss-120b": ModelPricing(0.15, 0.6, 0.01),
    "accounts/fireworks/models/gpt-oss-20b": ModelPricing(0.07, 0.3, 0.04),
    "accounts/fireworks/models/llama-v3p3-70b-instruct": ModelPricing(0.9, 0.9, 0.45),
    "accounts/fireworks/models/qwen3-8b": ModelPricing(0.2, 0.2, 0.1),
    "accounts/fireworks/models/qwen3-embedding-8b": ModelPricing(0.0, 0.0),
}

# =============================================================================
# deepseek
# =============================================================================
DEEPSEEK_PRICING: dict[str, ModelPricing] = {
    "deepseek-v4-flash": ModelPricing(0.14, 0.28, 0.0028),
    "deepseek-v4-pro": ModelPricing(1.74, 3.48, 0.0145),
    "deepseek-chat": ModelPricing(0.14, 0.28, 0.0028),
    "deepseek-reasoner": ModelPricing(0.14, 0.28, 0.0028),
}

# =============================================================================
# perplexity
# =============================================================================
PERPLEXITY_PRICING: dict[str, ModelPricing] = {
    "sonar": ModelPricing(1.0, 1.0),
    "sonar-pro": ModelPricing(3.0, 15.0),
    "sonar-reasoning-pro": ModelPricing(2.0, 8.0),
    "sonar-deep-research": ModelPricing(2.0, 8.0),
}

# =============================================================================
# databricks
# =============================================================================
DATABRICKS_PRICING: dict[str, ModelPricing] = {
    "databricks-meta-llama-3-3-70b-instruct": ModelPricing(0.5, 1.5),
    "databricks-meta-llama-3-1-405b-instruct": ModelPricing(5.0, 15.0),
    "databricks-meta-llama-3-1-8b-instruct": ModelPricing(0.15, 0.45),
    "databricks-llama-4-maverick": ModelPricing(0.5, 1.5),
    "databricks-gpt-oss-120b": ModelPricing(0.15, 0.6),
    "databricks-gpt-oss-20b": ModelPricing(0.07, 0.3),
    "databricks-gemma-3-12b": ModelPricing(0.15, 0.5),
    "databricks-claude-3-7-sonnet": ModelPricing(3.0, 15.0),
    "databricks-claude-sonnet-4": ModelPricing(3.0, 15.0),
    "databricks-claude-sonnet-4-1": ModelPricing(3.0, 15.0),
    "databricks-claude-sonnet-4-5": ModelPricing(3.0, 15.0),
    "databricks-claude-haiku-4-5": ModelPricing(1.0, 5.0),
    "databricks-claude-opus-4": ModelPricing(15.0, 75.0),
    "databricks-claude-opus-4-1": ModelPricing(15.0, 75.0),
    "databricks-claude-opus-4-5": ModelPricing(5.0, 25.0),
    "databricks-gemini-2-5-flash": ModelPricing(0.3, 2.5),
    "databricks-gemini-2-5-pro": ModelPricing(1.25, 10.0),
    "databricks-gpt-5": ModelPricing(1.25, 10.0),
    "databricks-gpt-5-1": ModelPricing(1.25, 10.0),
    "databricks-gpt-5-mini": ModelPricing(0.25, 2.0),
    "databricks-gpt-5-nano": ModelPricing(0.05, 0.4),
    "databricks-gte-large-en": ModelPricing(0.13, 0.0),
    "databricks-bge-large-en": ModelPricing(0.1, 0.0),
}

# =============================================================================
# ollama
# =============================================================================
OLLAMA_PRICING: dict[str, ModelPricing] = {
    "llama3.3": ModelPricing(0.0, 0.0),
    "llama3.2-vision": ModelPricing(0.0, 0.0),
    "llama3.1": ModelPricing(0.0, 0.0),
    "qwen3": ModelPricing(0.0, 0.0),
    "qwen3:32b": ModelPricing(0.0, 0.0),
    "qwen2.5": ModelPricing(0.0, 0.0),
    "qwen2.5-coder": ModelPricing(0.0, 0.0),
    "gemma3": ModelPricing(0.0, 0.0),
    "gemma3:27b": ModelPricing(0.0, 0.0),
    "deepseek-r1": ModelPricing(0.0, 0.0),
    "deepseek-r1:70b": ModelPricing(0.0, 0.0),
    "phi4": ModelPricing(0.0, 0.0),
    "mistral-nemo": ModelPricing(0.0, 0.0),
    "mixtral:8x7b": ModelPricing(0.0, 0.0),
    "gpt-oss": ModelPricing(0.0, 0.0),
    "llava": ModelPricing(0.0, 0.0),
    "nomic-embed-text": ModelPricing(0.0, 0.0),
    "mxbai-embed-large": ModelPricing(0.0, 0.0),
}

# =============================================================================
# xai
# =============================================================================
XAI_PRICING: dict[str, ModelPricing] = {
    "grok-4.3": ModelPricing(2.0, 6.0, 0.2),
    "grok-4-latest": ModelPricing(3.0, 15.0),
    "grok-4": ModelPricing(3.0, 15.0),
    "grok-4-0709": ModelPricing(3.0, 15.0),
    "grok-4-fast-reasoning": ModelPricing(0.2, 0.5, 0.05),
    "grok-4-fast-non-reasoning": ModelPricing(0.2, 0.5, 0.05),
    "grok-4-1-fast-reasoning": ModelPricing(0.2, 0.5, 0.05),
    "grok-4-1-fast-non-reasoning": ModelPricing(0.2, 0.5, 0.05),
    "grok-4.20-0309-reasoning": ModelPricing(2.0, 6.0, 0.2),
    "grok-4.20-0309-non-reasoning": ModelPricing(2.0, 6.0, 0.2),
    "grok-4.20-multi-agent-0309": ModelPricing(2.0, 6.0, 0.2),
    "grok-code-fast-1": ModelPricing(0.2, 1.5, 0.02),
    "grok-3": ModelPricing(3.0, 15.0, 0.75),
    "grok-3-mini": ModelPricing(0.3, 0.5, 0.075),
    "grok-2-vision": ModelPricing(2.0, 10.0),
}

# =============================================================================
# openrouter
# =============================================================================
OPENROUTER_PRICING: dict[str, ModelPricing] = {
    # No models in manifest
}

# =============================================================================
# nvidia_nim
# =============================================================================
NVIDIA_NIM_PRICING: dict[str, ModelPricing] = {
    "meta/llama-3.1-8b-instruct": ModelPricing(0.0, 0.0),
    "meta/llama-3.1-70b-instruct": ModelPricing(0.0, 0.0),
    "meta/llama-3.1-405b-instruct": ModelPricing(0.0, 0.0),
    "meta/llama-3.2-1b-instruct": ModelPricing(0.0, 0.0),
    "meta/llama-3.2-3b-instruct": ModelPricing(0.0, 0.0),
    "meta/llama-3.2-11b-vision-instruct": ModelPricing(0.0, 0.0),
    "meta/llama-3.2-90b-vision-instruct": ModelPricing(0.0, 0.0),
    "meta/llama-3.3-70b-instruct": ModelPricing(0.0, 0.0),
    "meta/llama-4-maverick-17b-128e-instruct": ModelPricing(0.0, 0.0),
    "nvidia/llama-3.1-nemotron-nano-8b-v1": ModelPricing(0.0, 0.0),
    "nvidia/llama-3.1-nemotron-51b-instruct": ModelPricing(0.0, 0.0),
    "nvidia/llama-3.1-nemotron-70b-instruct": ModelPricing(0.0, 0.0),
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": ModelPricing(0.0, 0.0),
    "nvidia/llama-3.3-nemotron-super-49b-v1": ModelPricing(0.0, 0.0),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": ModelPricing(0.0, 0.0),
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1": ModelPricing(0.0, 0.0),
    "nvidia/nemotron-mini-4b-instruct": ModelPricing(0.0, 0.0),
    "nvidia/nemotron-4-340b-instruct": ModelPricing(0.0, 0.0),
    "nvidia/nemotron-3-nano-30b-a3b": ModelPricing(0.0, 0.0),
    "nvidia/nemotron-3-super-120b-a12b": ModelPricing(0.0, 0.0),
    "nvidia/nemotron-nano-3-30b-a3b": ModelPricing(0.0, 0.0),
    "nvidia/nemotron-nano-12b-v2-vl": ModelPricing(0.0, 0.0),
    "nvidia/nvidia-nemotron-nano-9b-v2": ModelPricing(0.0, 0.0),
    "nvidia/llama3-chatqa-1.5-70b": ModelPricing(0.0, 0.0),
    "nvidia/mistral-nemo-minitron-8b-8k-instruct": ModelPricing(0.0, 0.0),
    "nv-mistralai/mistral-nemo-12b-instruct": ModelPricing(0.0, 0.0),
    "mistralai/mistral-7b-instruct-v0.3": ModelPricing(0.0, 0.0),
    "mistralai/mistral-large": ModelPricing(0.0, 0.0),
    "mistralai/mistral-large-2-instruct": ModelPricing(0.0, 0.0),
    "mistralai/mistral-large-3-675b-instruct-2512": ModelPricing(0.0, 0.0),
    "mistralai/mistral-medium-3-instruct": ModelPricing(0.0, 0.0),
    "mistralai/mistral-medium-3.5-128b": ModelPricing(0.0, 0.0),
    "mistralai/mistral-small-4-119b-2603": ModelPricing(0.0, 0.0),
    "mistralai/ministral-14b-instruct-2512": ModelPricing(0.0, 0.0),
    "mistralai/magistral-small-2506": ModelPricing(0.0, 0.0),
    "mistralai/devstral-2-123b-instruct-2512": ModelPricing(0.0, 0.0),
    "mistralai/codestral-22b-instruct-v0.1": ModelPricing(0.0, 0.0),
    "mistralai/mistral-nemotron": ModelPricing(0.0, 0.0),
    "mistralai/mixtral-8x7b-instruct-v0.1": ModelPricing(0.0, 0.0),
    "mistralai/mixtral-8x22b-instruct-v0.1": ModelPricing(0.0, 0.0),
    "qwen/qwen2.5-coder-32b-instruct": ModelPricing(0.0, 0.0),
    "qwen/qwen3-coder-480b-a35b-instruct": ModelPricing(0.0, 0.0),
    "qwen/qwen3-next-80b-a3b-instruct": ModelPricing(0.0, 0.0),
    "qwen/qwen3-next-80b-a3b-thinking": ModelPricing(0.0, 0.0),
    "qwen/qwen3.5-122b-a10b": ModelPricing(0.0, 0.0),
    "qwen/qwen3.5-397b-a17b": ModelPricing(0.0, 0.0),
    "deepseek-ai/deepseek-coder-6.7b-instruct": ModelPricing(0.0, 0.0),
    "deepseek-ai/deepseek-v4-flash": ModelPricing(0.0, 0.0),
    "deepseek-ai/deepseek-v4-pro": ModelPricing(0.0, 0.0),
    "microsoft/phi-3-vision-128k-instruct": ModelPricing(0.0, 0.0),
    "microsoft/phi-3.5-moe-instruct": ModelPricing(0.0, 0.0),
    "microsoft/phi-4-mini-instruct": ModelPricing(0.0, 0.0),
    "microsoft/phi-4-multimodal-instruct": ModelPricing(0.0, 0.0),
    "google/gemma-2-2b-it": ModelPricing(0.0, 0.0),
    "google/gemma-3-4b-it": ModelPricing(0.0, 0.0),
    "google/gemma-3-12b-it": ModelPricing(0.0, 0.0),
    "google/gemma-3-27b-it": ModelPricing(0.0, 0.0),
    "google/gemma-3n-e2b-it": ModelPricing(0.0, 0.0),
    "google/gemma-3n-e4b-it": ModelPricing(0.0, 0.0),
    "google/gemma-4-31b-it": ModelPricing(0.0, 0.0),
    "ibm/granite-3.0-3b-a800m-instruct": ModelPricing(0.0, 0.0),
    "ibm/granite-3.0-8b-instruct": ModelPricing(0.0, 0.0),
    "ibm/granite-8b-code-instruct": ModelPricing(0.0, 0.0),
    "ibm/granite-34b-code-instruct": ModelPricing(0.0, 0.0),
    "openai/gpt-oss-20b": ModelPricing(0.0, 0.0),
    "openai/gpt-oss-120b": ModelPricing(0.0, 0.0),
    "moonshotai/kimi-k2-instruct": ModelPricing(0.0, 0.0),
    "moonshotai/kimi-k2-instruct-0905": ModelPricing(0.0, 0.0),
    "moonshotai/kimi-k2-thinking": ModelPricing(0.0, 0.0),
    "moonshotai/kimi-k2.6": ModelPricing(0.0, 0.0),
    "minimaxai/minimax-m2.5": ModelPricing(0.0, 0.0),
    "minimaxai/minimax-m2.7": ModelPricing(0.0, 0.0),
    "z-ai/glm-5.1": ModelPricing(0.0, 0.0),
    "z-ai/glm4.7": ModelPricing(0.0, 0.0),
    "z-ai/glm5": ModelPricing(0.0, 0.0),
    "bytedance/seed-oss-36b-instruct": ModelPricing(0.0, 0.0),
    "stepfun-ai/step-3.5-flash": ModelPricing(0.0, 0.0),
    "01-ai/yi-large": ModelPricing(0.0, 0.0),
    "ai21labs/jamba-1.5-large-instruct": ModelPricing(0.0, 0.0),
    "abacusai/dracarys-llama-3.1-70b-instruct": ModelPricing(0.0, 0.0),
    "databricks/dbrx-instruct": ModelPricing(0.0, 0.0),
    "upstage/solar-10.7b-instruct": ModelPricing(0.0, 0.0),
    "writer/palmyra-creative-122b": ModelPricing(0.0, 0.0),
    "writer/palmyra-fin-70b-32k": ModelPricing(0.0, 0.0),
    "writer/palmyra-med-70b": ModelPricing(0.0, 0.0),
    "writer/palmyra-med-70b-32k": ModelPricing(0.0, 0.0),
    "sarvamai/sarvam-m": ModelPricing(0.0, 0.0),
    "stockmark/stockmark-2-100b-instruct": ModelPricing(0.0, 0.0),
    "zyphra/zamba2-7b-instruct": ModelPricing(0.0, 0.0),
    "aisingapore/sea-lion-7b-instruct": ModelPricing(0.0, 0.0),
}

# =============================================================================
# cerebras
# =============================================================================
CEREBRAS_PRICING: dict[str, ModelPricing] = {
    "llama3.1-8b": ModelPricing(0.1, 0.1),
    "gpt-oss-120b": ModelPricing(0.35, 0.75),
    "qwen-3-235b-a22b-instruct-2507": ModelPricing(0.6, 1.2),
    "zai-glm-4.7": ModelPricing(2.25, 2.75),
}

# =============================================================================
# sambanova
# =============================================================================
SAMBANOVA_PRICING: dict[str, ModelPricing] = {
    "Meta-Llama-3.3-70B-Instruct": ModelPricing(0.6, 1.2),
    "Llama-4-Maverick-17B-128E-Instruct": ModelPricing(0.63, 1.8),
    "DeepSeek-V3.1": ModelPricing(3.0, 4.5),
    "DeepSeek-V3.1-cb": ModelPricing(0.15, 0.75),
    "DeepSeek-V3.2": ModelPricing(3.0, 4.5),
    "gpt-oss-120b": ModelPricing(0.22, 0.59),
    "gemma-3-12b-it": ModelPricing(0.2, 0.35),
    "MiniMax-M2.5": ModelPricing(0.3, 1.2),
    "MiniMax-M2.7": ModelPricing(0.6, 2.4),
}

# =============================================================================
# deepinfra
# =============================================================================
DEEPINFRA_PRICING: dict[str, ModelPricing] = {
    "meta-llama/Llama-3.2-11B-Vision-Instruct": ModelPricing(0.245, 0.245),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelPricing(0.1, 0.32),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": ModelPricing(0.15, 0.6),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": ModelPricing(0.08, 0.3),
    "meta-llama/Llama-Guard-4-12B": ModelPricing(0.18, 0.18),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": ModelPricing(0.4, 0.4),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelPricing(0.4, 0.4),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ModelPricing(0.02, 0.05),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelPricing(0.02, 0.03),
    "Qwen/Qwen2.5-72B-Instruct": ModelPricing(0.36, 0.4),
    "Qwen/Qwen3-14B": ModelPricing(0.12, 0.24),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": ModelPricing(0.071, 0.1),
    "Qwen/Qwen3-235B-A22B-Thinking-2507": ModelPricing(0.23, 2.3),
    "Qwen/Qwen3-30B-A3B": ModelPricing(0.09, 0.45),
    "Qwen/Qwen3-32B": ModelPricing(0.08, 0.28),
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo": ModelPricing(0.3, 1.0),
    "Qwen/Qwen3-Max": ModelPricing(1.2, 6.0),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": ModelPricing(0.09, 1.1),
    "Qwen/Qwen3-VL-235B-A22B-Instruct": ModelPricing(0.2, 0.88),
    "Qwen/Qwen3-VL-30B-A3B-Instruct": ModelPricing(0.15, 0.6),
    "deepseek-ai/DeepSeek-R1-0528": ModelPricing(0.5, 2.15),
    "deepseek-ai/DeepSeek-R1-0528-Turbo": ModelPricing(1.0, 3.0),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ModelPricing(0.7, 0.8),
    "deepseek-ai/DeepSeek-V3": ModelPricing(0.32, 0.89),
    "deepseek-ai/DeepSeek-V3-0324": ModelPricing(0.2, 0.77),
    "deepseek-ai/DeepSeek-V3.1": ModelPricing(0.21, 0.79),
    "deepseek-ai/DeepSeek-V3.1-Terminus": ModelPricing(0.27, 0.95),
    "deepseek-ai/DeepSeek-V3.2": ModelPricing(0.26, 0.38),
    "google/gemini-2.5-flash": ModelPricing(0.3, 2.5),
    "google/gemini-2.5-pro": ModelPricing(1.25, 10.0),
    "google/gemma-3-12b-it": ModelPricing(0.04, 0.13),
    "google/gemma-3-27b-it": ModelPricing(0.08, 0.16),
    "google/gemma-3-4b-it": ModelPricing(0.04, 0.08),
    "mistralai/Mistral-Nemo-Instruct-2407": ModelPricing(0.02, 0.04),
    "mistralai/Mistral-Small-24B-Instruct-2501": ModelPricing(0.05, 0.08),
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": ModelPricing(0.075, 0.2),
    "moonshotai/Kimi-K2.5": ModelPricing(0.45, 2.25),
    "moonshotai/Kimi-K2.6": ModelPricing(0.75, 3.5),
    "openai/gpt-oss-120b": ModelPricing(0.039, 0.19),
    "openai/gpt-oss-120b-Turbo": ModelPricing(0.15, 0.6),
    "openai/gpt-oss-20b": ModelPricing(0.03, 0.14),
    "microsoft/phi-4": ModelPricing(0.07, 0.14),
    "nvidia/Llama-3.3-Nemotron-Super-49B-v1.5": ModelPricing(0.1, 0.4),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": ModelPricing(0.1, 0.5),
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": ModelPricing(0.04, 0.16),
    "nvidia/Nemotron-3-Nano-30B-A3B": ModelPricing(0.05, 0.2),
    "zai-org/GLM-4.6": ModelPricing(0.43, 1.74),
    "zai-org/GLM-4.7": ModelPricing(0.4, 1.75),
    "zai-org/GLM-4.7-Flash": ModelPricing(0.06, 0.4),
}

# =============================================================================
# huggingface
# =============================================================================
HUGGINGFACE_PRICING: dict[str, ModelPricing] = {
    # No models in manifest
}

# =============================================================================
# watsonx
# =============================================================================
WATSONX_PRICING: dict[str, ModelPricing] = {
    "ibm/granite-13b-chat-v2": ModelPricing(0.0, 0.0),
    "ibm/granite-3-8b-instruct": ModelPricing(0.0, 0.0),
    "meta-llama/llama-3-3-70b-instruct": ModelPricing(0.0, 0.0),
}

# =============================================================================
# ai21
# =============================================================================
AI21_PRICING: dict[str, ModelPricing] = {
    "jamba-1.5-large": ModelPricing(2.0, 8.0),
    "jamba-1.5-mini": ModelPricing(0.2, 0.4),
}

# =============================================================================
# nebius
# =============================================================================
NEBIUS_PRICING: dict[str, ModelPricing] = {
    "deepseek-ai/DeepSeek-R1": ModelPricing(0.8, 2.4),
    "deepseek-ai/DeepSeek-R1-0528": ModelPricing(0.8, 2.4),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ModelPricing(0.25, 0.75),
    "deepseek-ai/DeepSeek-V3": ModelPricing(0.5, 1.5),
    "deepseek-ai/DeepSeek-V3-0324": ModelPricing(0.5, 1.5),
    "google/gemma-3-27b-it": ModelPricing(0.06, 0.2),
    "meta-llama/Llama-3.3-70B-Instruct": ModelPricing(0.13, 0.4),
    "meta-llama/Llama-Guard-3-8B": ModelPricing(0.02, 0.06),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ModelPricing(0.02, 0.06),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": ModelPricing(0.13, 0.4),
    "meta-llama/Meta-Llama-3.1-405B-Instruct": ModelPricing(1.0, 3.0),
    "mistralai/Mistral-Nemo-Instruct-2407": ModelPricing(0.04, 0.12),
    "NousResearch/Hermes-3-Llama-3.1-405B": ModelPricing(1.0, 3.0),
    "nvidia/Llama-3.1-Nemotron-Ultra-253B-v1": ModelPricing(0.6, 1.8),
    "nvidia/Llama-3.3-Nemotron-Super-49B-v1": ModelPricing(0.1, 0.4),
    "Qwen/Qwen3-235B-A22B": ModelPricing(0.2, 0.6),
    "Qwen/Qwen3-32B": ModelPricing(0.1, 0.3),
    "Qwen/Qwen3-30B-A3B": ModelPricing(0.1, 0.3),
    "Qwen/Qwen3-14B": ModelPricing(0.08, 0.24),
    "Qwen/Qwen3-4B": ModelPricing(0.08, 0.24),
    "Qwen/QwQ-32B": ModelPricing(0.15, 0.45),
    "Qwen/Qwen2.5-72B-Instruct": ModelPricing(0.13, 0.4),
    "Qwen/Qwen2.5-32B-Instruct": ModelPricing(0.06, 0.2),
    "Qwen/Qwen2.5-Coder-7B": ModelPricing(0.01, 0.03),
    "Qwen/Qwen2.5-VL-72B-Instruct": ModelPricing(0.13, 0.4),
    "Qwen/Qwen2-VL-72B-Instruct": ModelPricing(0.13, 0.4),
    "Qwen/Qwen2-VL-7B-Instruct": ModelPricing(0.02, 0.06),
    "BAAI/bge-en-icl": ModelPricing(0.01, 0.0),
    "BAAI/bge-multilingual-gemma2": ModelPricing(0.01, 0.0),
    "intfloat/e5-mistral-7b-instruct": ModelPricing(0.01, 0.0),
}

# =============================================================================
# ovhcloud
# =============================================================================
OVHCLOUD_PRICING: dict[str, ModelPricing] = {
    "Meta-Llama-3_3-70B-Instruct": ModelPricing(0.67, 0.67),
    "Llama-3.1-8B-Instruct": ModelPricing(0.1, 0.1),
    "DeepSeek-R1-Distill-Llama-70B": ModelPricing(0.67, 0.67),
    "Mistral-Nemo-Instruct-2407": ModelPricing(0.13, 0.13),
    "Mistral-Small-3.2-24B-Instruct-2506": ModelPricing(0.09, 0.28),
    "Mistral-7B-Instruct-v0.3": ModelPricing(0.1, 0.1),
    "Mixtral-8x7B-Instruct-v0.1": ModelPricing(0.63, 0.63),
    "gpt-oss-120b": ModelPricing(0.08, 0.4),
    "gpt-oss-20b": ModelPricing(0.04, 0.15),
    "Qwen3-32B": ModelPricing(0.08, 0.23),
    "Qwen3.5-9B": ModelPricing(0.1, 0.15),
    "Qwen3-Coder-30B-A3B-Instruct": ModelPricing(0.06, 0.22),
    "Qwen2.5-VL-72B-Instruct": ModelPricing(0.91, 0.91),
    "Meta-Llama-3_1-70B-Instruct": ModelPricing(0.67, 0.67),
    "Qwen2.5-Coder-32B-Instruct": ModelPricing(0.87, 0.87),
    "llava-v1.6-mistral-7b-hf": ModelPricing(0.29, 0.29),
    "mamba-codestral-7B-v0.1": ModelPricing(0.19, 0.19),
    "bge-multilingual-gemma2": ModelPricing(0.01, 0.0),
    "bge-m3": ModelPricing(0.01, 0.0),
    "Qwen3-Embedding-8B": ModelPricing(0.1, 0.0),
}

# =============================================================================
# zai
# =============================================================================
ZAI_PRICING: dict[str, ModelPricing] = {
    "glm-5": ModelPricing(1.0, 3.2, 0.2),
    "glm-5-code": ModelPricing(1.2, 5.0, 0.3),
    "glm-4.7": ModelPricing(0.6, 2.2, 0.11),
    "glm-4.6": ModelPricing(0.6, 2.2, 0.11),
    "glm-4.5": ModelPricing(0.6, 2.2),
    "glm-4.5v": ModelPricing(0.6, 1.8),
    "glm-4.5-x": ModelPricing(2.2, 8.9),
    "glm-4.5-air": ModelPricing(0.2, 1.1),
    "glm-4.5-airx": ModelPricing(1.1, 4.5),
    "glm-4.5-flash": ModelPricing(0.0, 0.0),
    "glm-4-32b-0414-128k": ModelPricing(0.1, 0.1),
}

# =============================================================================
# moonshot
# =============================================================================
MOONSHOT_PRICING: dict[str, ModelPricing] = {
    "kimi-k2.5": ModelPricing(0.6, 3.0, 0.1),
    "kimi-k2.6": ModelPricing(0.95, 4.0, 0.16),
    "kimi-latest": ModelPricing(2.0, 5.0, 0.15),
    "kimi-latest-128k": ModelPricing(2.0, 5.0, 0.15),
    "kimi-latest-32k": ModelPricing(1.0, 3.0, 0.15),
    "kimi-latest-8k": ModelPricing(0.2, 2.0, 0.15),
    "kimi-thinking-preview": ModelPricing(0.6, 2.5, 0.15),
    "kimi-k2-thinking": ModelPricing(0.6, 2.5, 0.15),
    "kimi-k2-thinking-turbo": ModelPricing(1.15, 8.0, 0.15),
    "kimi-k2-0711-preview": ModelPricing(0.6, 2.5, 0.15),
    "kimi-k2-0905-preview": ModelPricing(0.6, 2.5, 0.15),
    "kimi-k2-turbo-preview": ModelPricing(1.15, 8.0, 0.15),
    "moonshot-v1-8k": ModelPricing(0.2, 2.0),
    "moonshot-v1-32k": ModelPricing(1.0, 3.0),
    "moonshot-v1-128k": ModelPricing(2.0, 5.0),
    "moonshot-v1-8k-0430": ModelPricing(0.2, 2.0),
    "moonshot-v1-32k-0430": ModelPricing(1.0, 3.0),
    "moonshot-v1-128k-0430": ModelPricing(2.0, 5.0),
    "moonshot-v1-8k-vision-preview": ModelPricing(0.2, 2.0),
    "moonshot-v1-32k-vision-preview": ModelPricing(1.0, 3.0),
    "moonshot-v1-128k-vision-preview": ModelPricing(2.0, 5.0),
    "moonshot-v1-auto": ModelPricing(2.0, 5.0),
}


ALL_PRICING: dict[str, dict[str, ModelPricing]] = {
    "openai": OPENAI_PRICING,
    "azure": AZURE_PRICING,
    "anthropic": ANTHROPIC_PRICING,
    "gemini": GEMINI_PRICING,
    "vertex_ai": VERTEX_AI_PRICING,
    "bedrock": BEDROCK_PRICING,
    "mistral": MISTRAL_PRICING,
    "cohere": COHERE_PRICING,
    "groq": GROQ_PRICING,
    "together_ai": TOGETHER_PRICING,
    "fireworks_ai": FIREWORKS_PRICING,
    "deepseek": DEEPSEEK_PRICING,
    "perplexity": PERPLEXITY_PRICING,
    "databricks": DATABRICKS_PRICING,
    "ollama": OLLAMA_PRICING,
    "xai": XAI_PRICING,
    "openrouter": OPENROUTER_PRICING,
    "nvidia_nim": NVIDIA_NIM_PRICING,
    "cerebras": CEREBRAS_PRICING,
    "sambanova": SAMBANOVA_PRICING,
    "deepinfra": DEEPINFRA_PRICING,
    "huggingface": HUGGINGFACE_PRICING,
    "watsonx": WATSONX_PRICING,
    "ai21": AI21_PRICING,
    "nebius": NEBIUS_PRICING,
    "ovhcloud": OVHCLOUD_PRICING,
    "zai": ZAI_PRICING,
    "moonshot": MOONSHOT_PRICING,
}


class UnknownModelPricingError(ArcLLMError):
    """Raised when pricing is not available for a model."""


def _normalize_model_name(model: str) -> tuple[str | None, str]:
    """Split a model string into ``(provider, model_id)``.

    Accepts ``provider/model`` or bare ``model``. Provider keys are matched
    against ``ALL_PRICING`` after normalising kebab-case to snake_case so that
    e.g. ``vertex-ai/gemini-2.5-pro`` resolves to ``vertex_ai``.
    """
    provider: str | None = None
    model_name = model

    if "/" in model:
        head, tail = model.split("/", 1)
        head_norm = head.lower().replace("-", "_")
        if head_norm in ALL_PRICING:
            provider = head_norm
            model_name = tail
        # Otherwise the slash belongs to the model id (e.g. Together's
        # ``meta-llama/Llama-4-...``); fall through and search every provider.

    return provider, model_name


def get_model_pricing(model: str) -> ModelPricing:
    """Return pricing for ``model``.

    Raises ``UnknownModelPricingError`` if no entry is found.
    """
    provider, model_name = _normalize_model_name(model)

    if provider is not None:
        pricing_table = ALL_PRICING.get(provider, {})
        if model_name in pricing_table:
            return pricing_table[model_name]
        raise UnknownModelPricingError(
            f"No pricing available for model {model_name!r} from provider {provider!r}",
            model=model,
            provider=provider,
        )

    for pricing_table in ALL_PRICING.values():
        if model_name in pricing_table:
            return pricing_table[model_name]

    raise UnknownModelPricingError(
        f"No pricing available for model {model!r}",
        model=model,
    )


def cost_per_token(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    *,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> tuple[float, float]:
    """Return ``(prompt_cost, completion_cost)`` in USD.

    Prompt-side cost is split into three slices when the provider supports
    prompt caching:

    - ``cache_read_input_tokens``: billed at
      ``cached_input_cost_per_million`` (typically 10% of base).
    - ``cache_creation_input_tokens``: billed at 1.25x the base input rate
      (Anthropic's documented cache-write surcharge). Falls back to base
      when the model has no cached pricing entry.
    - The remainder (``prompt_tokens - cache_read - cache_creation``):
      billed at ``input_cost_per_million``.

    Callers that don't track cache state simply omit the cache args; cost
    falls back to the simple ``prompt_tokens * input_rate`` calculation.
    """
    pricing = get_model_pricing(model)
    input_rate = pricing.input_cost_per_million
    cached_rate = (
        pricing.cached_input_cost_per_million
        if pricing.cached_input_cost_per_million is not None
        else input_rate
    )
    creation_rate = input_rate * 1.25  # Anthropic cache-write surcharge

    base_prompt = max(0, prompt_tokens - cache_read_input_tokens - cache_creation_input_tokens)
    prompt_cost = (
        (base_prompt / 1_000_000) * input_rate
        + (cache_read_input_tokens / 1_000_000) * cached_rate
        + (cache_creation_input_tokens / 1_000_000) * creation_rate
    )
    completion_cost = (completion_tokens / 1_000_000) * pricing.output_cost_per_million
    return (prompt_cost, completion_cost)


def completion_cost(
    response: ModelResponse,
    model: str | None = None,
) -> float:
    """Return the total USD cost for ``response``.

    ``model`` overrides ``response.model`` when given. Returns ``0.0`` when
    ``response.usage`` is ``None`` (provider didn't report usage). When the
    response carries cache token counts (Anthropic-family providers), they
    are factored into the prompt-side cost at the cached rate.
    """
    model_name = model or response.model
    if not model_name:
        raise ValueError("Model name required but not provided")

    usage = response.usage
    if usage is None:
        return 0.0

    prompt_cost, comp_cost = cost_per_token(
        model_name,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
    )
    return prompt_cost + comp_cost
