"""
Pricing tables for supported models.

Prices are in USD per 1 million tokens.
Last updated: 2025-01-08

To update prices:
1. Check provider pricing pages
2. Update the relevant dict in this file
3. Update PRICING_VERSION
4. Run tests to ensure format is valid
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


# Version for tracking pricing table updates
PRICING_VERSION = "2026.01.08"


@dataclass(slots=True, frozen=True)
class ModelPricing:
    """Pricing information for a model."""

    input_cost_per_million: float
    output_cost_per_million: float
    # Optional: cached input pricing (for prompt caching)
    cached_input_cost_per_million: float | None = None


# =============================================================================
# OpenAI Pricing (USD per 1M tokens)
# =============================================================================
#
# Official Pricing Page: https://openai.com/pricing
# API Pricing Docs: https://platform.openai.com/docs/pricing
#
# HOW TO UPDATE:
# 1. Check https://openai.com/pricing for current prices
# 2. Update the dict below
# 3. Update PRICING_VERSION at top of file
# 4. Run: pytest tests/test_pricing.py
#
# MODEL NAMING:
# - Models use exact API model IDs
# - Date-suffixed versions (e.g., gpt-4o-2024-11-20) have specific pricing
# - Aliases (e.g., "gpt-4o") point to latest version
#
# PRICE FORMAT: ModelPricing(input_cost_per_million, output_cost_per_million)
#

OPENAI_PRICING: dict[str, ModelPricing] = {
    # =========================================================================
    # GPT-5.2 series (Latest flagship - December 2025)
    # https://platform.openai.com/docs/models/gpt-5
    # =========================================================================
    "gpt-5.2": ModelPricing(5.00, 15.00),
    "gpt-5.2-2025-12-11": ModelPricing(5.00, 15.00),
    "gpt-5.2-chat-latest": ModelPricing(5.00, 15.00),
    "gpt-5.2-pro": ModelPricing(20.00, 80.00),
    "gpt-5.2-pro-2025-12-11": ModelPricing(20.00, 80.00),
    # =========================================================================
    # GPT-5.1 series (November 2025)
    # =========================================================================
    "gpt-5.1": ModelPricing(4.00, 12.00),
    "gpt-5.1-2025-11-13": ModelPricing(4.00, 12.00),
    "gpt-5.1-chat-latest": ModelPricing(4.00, 12.00),
    "gpt-5.1-codex": ModelPricing(4.00, 12.00),
    "gpt-5.1-codex-max": ModelPricing(8.00, 24.00),
    "gpt-5.1-codex-mini": ModelPricing(2.00, 6.00),
    # =========================================================================
    # GPT-5 series (August 2025)
    # =========================================================================
    "gpt-5": ModelPricing(3.00, 10.00),
    "gpt-5-2025-08-07": ModelPricing(3.00, 10.00),
    "gpt-5-chat-latest": ModelPricing(3.00, 10.00),
    "gpt-5-mini": ModelPricing(0.50, 2.00),
    "gpt-5-mini-2025-08-07": ModelPricing(0.50, 2.00),
    "gpt-5-nano": ModelPricing(0.15, 0.60),
    "gpt-5-nano-2025-08-07": ModelPricing(0.15, 0.60),
    "gpt-5-pro": ModelPricing(15.00, 60.00),
    "gpt-5-pro-2025-10-06": ModelPricing(15.00, 60.00),
    "gpt-5-codex": ModelPricing(3.00, 10.00),
    "gpt-5-search-api": ModelPricing(3.00, 10.00),
    "gpt-5-search-api-2025-10-14": ModelPricing(3.00, 10.00),
    # =========================================================================
    # GPT-4.1 series
    # =========================================================================
    "gpt-4.1": ModelPricing(2.00, 8.00),
    "gpt-4.1-2025-04-14": ModelPricing(2.00, 8.00),
    "gpt-4.1-mini": ModelPricing(0.10, 0.40),
    "gpt-4.1-mini-2025-04-14": ModelPricing(0.10, 0.40),
    "gpt-4.1-nano": ModelPricing(0.05, 0.20),
    "gpt-4.1-nano-2025-04-14": ModelPricing(0.05, 0.20),
    # =========================================================================
    # o3 series (reasoning - 2025)
    # https://platform.openai.com/docs/models/o3
    # =========================================================================
    "o3": ModelPricing(20.00, 80.00),
    "o3-2025-04-16": ModelPricing(20.00, 80.00),
    "o3-mini": ModelPricing(5.00, 20.00),
    "o3-mini-2025-01-31": ModelPricing(5.00, 20.00),
    "o3-pro": ModelPricing(40.00, 160.00),
    "o3-pro-2025-06-10": ModelPricing(40.00, 160.00),
    "o3-deep-research": ModelPricing(30.00, 120.00),
    "o3-deep-research-2025-06-26": ModelPricing(30.00, 120.00),
    # =========================================================================
    # o1 series (reasoning)
    # https://platform.openai.com/docs/models/o1
    # =========================================================================
    "o1": ModelPricing(15.00, 60.00),
    "o1-2024-12-17": ModelPricing(15.00, 60.00),
    "o1-mini": ModelPricing(3.00, 12.00),
    "o1-mini-2024-09-12": ModelPricing(3.00, 12.00),
    "o1-pro": ModelPricing(30.00, 120.00),
    "o1-pro-2025-03-19": ModelPricing(30.00, 120.00),
    # =========================================================================
    # GPT-4o series (Legacy but still active)
    # =========================================================================
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4o-2024-11-20": ModelPricing(2.50, 10.00),
    "gpt-4o-2024-08-06": ModelPricing(2.50, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4o-mini-2024-07-18": ModelPricing(0.15, 0.60),
    "chatgpt-4o-latest": ModelPricing(5.00, 15.00),
    # =========================================================================
    # Legacy models (older generations)
    # =========================================================================
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-2024-04-09": ModelPricing(10.00, 30.00),
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-4-0613": ModelPricing(30.00, 60.00),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-0125": ModelPricing(0.50, 1.50),
    # =========================================================================
    # Embeddings
    # =========================================================================
    "text-embedding-3-small": ModelPricing(0.02, 0.0),
    "text-embedding-3-large": ModelPricing(0.13, 0.0),
    "text-embedding-ada-002": ModelPricing(0.10, 0.0),
}


# =============================================================================
# Anthropic Pricing (USD per 1M tokens)
# Official pricing page: https://www.anthropic.com/pricing
# API docs: https://docs.anthropic.com/en/docs/about-claude/models
#
# HOW TO UPDATE:
# 1. Visit https://www.anthropic.com/pricing for current prices
# 2. Update the dict below
# 3. Update PRICING_VERSION at top of file
# 4. Run: pytest tests/test_pricing.py
#
# PRICE FORMAT: ModelPricing(input_cost, output_cost, cached_input_cost)
# - cached_input_cost is for prompt caching feature
# - See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
# =============================================================================

ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    # =========================================================================
    # Claude 4.5 series (Current flagship - released November 2025)
    # https://docs.anthropic.com/en/docs/about-claude/models#claude-4-5-family
    # =========================================================================
    # Claude 4.5 Opus - Most powerful, best for complex reasoning
    "claude-4-5-opus-20251120": ModelPricing(20.00, 100.00, 2.00),
    "claude-4-5-opus-latest": ModelPricing(20.00, 100.00, 2.00),
    # Claude 4.5 Sonnet - Balanced performance and cost
    "claude-4-5-sonnet-20251015": ModelPricing(5.00, 25.00, 0.50),
    "claude-4-5-sonnet-latest": ModelPricing(5.00, 25.00, 0.50),
    # Claude 4.5 Haiku - Fast and affordable
    "claude-4-5-haiku-20251201": ModelPricing(1.00, 5.00, 0.10),
    "claude-4-5-haiku-latest": ModelPricing(1.00, 5.00, 0.10),
    # =========================================================================
    # Claude 4 series (Released June 2025)
    # =========================================================================
    # Claude 4 Opus
    "claude-4-opus-20250615": ModelPricing(18.00, 90.00, 1.80),
    "claude-4-opus-latest": ModelPricing(18.00, 90.00, 1.80),
    # Claude 4 Sonnet
    "claude-4-sonnet-20250601": ModelPricing(4.00, 20.00, 0.40),
    "claude-4-sonnet-latest": ModelPricing(4.00, 20.00, 0.40),
    # Claude 4 Haiku
    "claude-4-haiku-20250701": ModelPricing(0.80, 4.00, 0.08),
    "claude-4-haiku-latest": ModelPricing(0.80, 4.00, 0.08),
    # =========================================================================
    # Claude 3.5 series (DEPRECATED - retiring March 2026)
    # Still functional but migrate to Claude 4 series
    # =========================================================================
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-sonnet-latest": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00, 0.08),
    "claude-3-5-haiku-latest": ModelPricing(0.80, 4.00, 0.08),
    # =========================================================================
    # Claude 3 series (DEPRECATED - for backwards compatibility)
    # =========================================================================
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00, 1.50),
    "claude-3-opus-latest": ModelPricing(15.00, 75.00, 1.50),
    "claude-3-sonnet-20240229": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25, 0.03),
    # =========================================================================
    # Legacy Claude 2 (DEPRECATED - end of life)
    # =========================================================================
    "claude-2.1": ModelPricing(8.00, 24.00),
}


# =============================================================================
# Google Gemini Pricing (USD per 1M tokens)
# https://ai.google.dev/pricing
# =============================================================================

GEMINI_PRICING: dict[str, ModelPricing] = {
    # Gemini 2.0
    "gemini-2.0-flash-exp": ModelPricing(0.0, 0.0),  # Free preview
    "gemini-2.0-flash-thinking-exp": ModelPricing(0.0, 0.0),  # Free preview
    # Gemini 1.5 Pro
    "gemini-1.5-pro": ModelPricing(1.25, 5.00),  # <=128k
    "gemini-1.5-pro-latest": ModelPricing(1.25, 5.00),
    "gemini-1.5-pro-001": ModelPricing(1.25, 5.00),
    "gemini-1.5-pro-002": ModelPricing(1.25, 5.00),
    # Gemini 1.5 Flash
    "gemini-1.5-flash": ModelPricing(0.075, 0.30),  # <=128k
    "gemini-1.5-flash-latest": ModelPricing(0.075, 0.30),
    "gemini-1.5-flash-001": ModelPricing(0.075, 0.30),
    "gemini-1.5-flash-002": ModelPricing(0.075, 0.30),
    "gemini-1.5-flash-8b": ModelPricing(0.0375, 0.15),  # <=128k
    "gemini-1.5-flash-8b-001": ModelPricing(0.0375, 0.15),
    # Gemini 1.0 Pro
    "gemini-1.0-pro": ModelPricing(0.50, 1.50),
    "gemini-1.0-pro-latest": ModelPricing(0.50, 1.50),
    "gemini-1.0-pro-001": ModelPricing(0.50, 1.50),
    "gemini-pro": ModelPricing(0.50, 1.50),  # Alias
    # Embeddings
    "text-embedding-004": ModelPricing(0.00, 0.0),  # Free tier
    "embedding-001": ModelPricing(0.00, 0.0),
}


# =============================================================================
# Mistral Pricing (USD per 1M tokens)
# https://mistral.ai/technology/#pricing
# =============================================================================

MISTRAL_PRICING: dict[str, ModelPricing] = {
    # Premier models
    "mistral-large-latest": ModelPricing(2.00, 6.00),
    "mistral-large-2411": ModelPricing(2.00, 6.00),
    "mistral-large-2407": ModelPricing(2.00, 6.00),
    "pixtral-large-latest": ModelPricing(2.00, 6.00),
    "pixtral-large-2411": ModelPricing(2.00, 6.00),
    # Free models
    "mistral-small-latest": ModelPricing(0.20, 0.60),
    "mistral-small-2409": ModelPricing(0.20, 0.60),
    "pixtral-12b-2409": ModelPricing(0.15, 0.15),
    "mistral-nemo-latest": ModelPricing(0.15, 0.15),
    "mistral-nemo-2407": ModelPricing(0.15, 0.15),
    # Codestral
    "codestral-latest": ModelPricing(0.20, 0.60),
    "codestral-2405": ModelPricing(0.20, 0.60),
    # Ministral
    "ministral-3b-latest": ModelPricing(0.04, 0.04),
    "ministral-3b-2410": ModelPricing(0.04, 0.04),
    "ministral-8b-latest": ModelPricing(0.10, 0.10),
    "ministral-8b-2410": ModelPricing(0.10, 0.10),
    # Legacy
    "mistral-medium-latest": ModelPricing(2.70, 8.10),
    "mistral-tiny": ModelPricing(0.25, 0.25),
    "open-mistral-7b": ModelPricing(0.25, 0.25),
    "open-mixtral-8x7b": ModelPricing(0.70, 0.70),
    "open-mixtral-8x22b": ModelPricing(2.00, 6.00),
    # Embeddings
    "mistral-embed": ModelPricing(0.10, 0.0),
}


# =============================================================================
# Cohere Pricing (USD per 1M tokens)
# https://cohere.com/pricing
# =============================================================================

COHERE_PRICING: dict[str, ModelPricing] = {
    # Command R+
    "command-r-plus": ModelPricing(2.50, 10.00),
    "command-r-plus-08-2024": ModelPricing(2.50, 10.00),
    "command-r-plus-04-2024": ModelPricing(3.00, 15.00),
    # Command R
    "command-r": ModelPricing(0.15, 0.60),
    "command-r-08-2024": ModelPricing(0.15, 0.60),
    "command-r-03-2024": ModelPricing(0.50, 1.50),
    # Command
    "command": ModelPricing(1.00, 2.00),
    "command-light": ModelPricing(0.30, 0.60),
    "command-nightly": ModelPricing(1.00, 2.00),
    "command-light-nightly": ModelPricing(0.30, 0.60),
    # Embeddings
    "embed-english-v3.0": ModelPricing(0.10, 0.0),
    "embed-multilingual-v3.0": ModelPricing(0.10, 0.0),
    "embed-english-light-v3.0": ModelPricing(0.10, 0.0),
    "embed-multilingual-light-v3.0": ModelPricing(0.10, 0.0),
    "embed-english-v2.0": ModelPricing(0.10, 0.0),
    "embed-multilingual-v2.0": ModelPricing(0.10, 0.0),
}


# =============================================================================
# Groq Pricing (USD per 1M tokens)
# https://groq.com/pricing/
# =============================================================================

# =============================================================================
# Groq Pricing (USD per 1M tokens)
# https://groq.com/pricing/
# https://console.groq.com/docs/models
#
# Last updated: 2026-01-08
# To update: Check https://console.groq.com/docs/models for current pricing
# =============================================================================

GROQ_PRICING: dict[str, ModelPricing] = {
    # =========================================================================
    # Llama 4 series (Latest - January 2026)
    # =========================================================================
    "meta-llama/llama-4-maverick-17b-128e-instruct": ModelPricing(0.20, 0.60),
    "meta-llama/llama-4-scout-17b-16e-instruct": ModelPricing(0.11, 0.34),
    # =========================================================================
    # OpenAI GPT-OSS (Open-weight models on Groq)
    # =========================================================================
    "openai/gpt-oss-120b": ModelPricing(0.30, 0.90),
    "openai/gpt-oss-20b": ModelPricing(0.05, 0.15),
    "openai/gpt-oss-safeguard-20b": ModelPricing(0.05, 0.15),
    # =========================================================================
    # Moonshot Kimi K2
    # =========================================================================
    "moonshotai/kimi-k2-instruct": ModelPricing(0.15, 0.45),
    "moonshotai/kimi-k2-instruct-0905": ModelPricing(0.15, 0.45),
    # =========================================================================
    # Qwen 3 series
    # =========================================================================
    "qwen/qwen3-32b": ModelPricing(0.12, 0.36),
    # =========================================================================
    # Groq Compound (agentic models)
    # =========================================================================
    "groq/compound": ModelPricing(0.00, 0.00),  # Free tier
    "groq/compound-mini": ModelPricing(0.00, 0.00),  # Free tier
    # =========================================================================
    # Llama 3.3
    # =========================================================================
    "llama-3.3-70b-versatile": ModelPricing(0.59, 0.79),
    # =========================================================================
    # Llama 3.1
    # =========================================================================
    "llama-3.1-8b-instant": ModelPricing(0.05, 0.08),
    # =========================================================================
    # Llama Guard (safety models)
    # =========================================================================
    "meta-llama/llama-guard-4-12b": ModelPricing(0.20, 0.20),
    "meta-llama/llama-prompt-guard-2-86m": ModelPricing(0.02, 0.02),
    "meta-llama/llama-prompt-guard-2-22m": ModelPricing(0.02, 0.02),
    # =========================================================================
    # Other models
    # =========================================================================
    "allam-2-7b": ModelPricing(0.05, 0.08),  # SDAIA Arabic model
    # =========================================================================
    # Whisper (audio transcription - per minute pricing converted to tokens)
    # =========================================================================
    "whisper-large-v3": ModelPricing(0.111, 0.00),  # Audio input only
    "whisper-large-v3-turbo": ModelPricing(0.04, 0.00),  # Audio input only
    # =========================================================================
    # Canopy Labs Orpheus (TTS models - free during preview)
    # =========================================================================
    "canopylabs/orpheus-v1-english": ModelPricing(0.00, 0.00),
    "canopylabs/orpheus-arabic-saudi": ModelPricing(0.00, 0.00),
}


# =============================================================================
# Together AI Pricing (USD per 1M tokens)
# https://www.together.ai/pricing
# =============================================================================

TOGETHER_PRICING: dict[str, ModelPricing] = {
    # =========================================================================
    # Llama 4 series (Latest - January 2026)
    # =========================================================================
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": ModelPricing(0.27, 0.85),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": ModelPricing(0.18, 0.59),
    # =========================================================================
    # Llama 3.3
    # =========================================================================
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelPricing(0.88, 0.88),
    # =========================================================================
    # Llama 3.2 (Vision models require dedicated endpoint)
    # =========================================================================
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": ModelPricing(1.20, 1.20),
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": ModelPricing(0.18, 0.18),
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": ModelPricing(0.06, 0.06),
    # =========================================================================
    # Llama 3.1
    # =========================================================================
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": ModelPricing(3.50, 3.50),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelPricing(0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelPricing(0.18, 0.18),
    # =========================================================================
    # Qwen 2.5
    # =========================================================================
    "Qwen/Qwen2.5-72B-Instruct-Turbo": ModelPricing(1.20, 1.20),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": ModelPricing(0.30, 0.30),
    # =========================================================================
    # Mixtral (may require dedicated endpoint)
    # =========================================================================
    "mistralai/Mixtral-8x22B-Instruct-v0.1": ModelPricing(1.20, 1.20),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelPricing(0.60, 0.60),
    # =========================================================================
    # DeepSeek
    # =========================================================================
    "deepseek-ai/DeepSeek-R1": ModelPricing(3.00, 7.00),  # Reasoning model
    "deepseek-ai/DeepSeek-V3": ModelPricing(0.90, 0.90),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ModelPricing(0.90, 0.90),
}


# =============================================================================
# Fireworks AI Pricing (USD per 1M tokens)
# https://fireworks.ai/pricing
# Pricing tiers (serverless):
#   - <4B params: $0.10/1M tokens
#   - 4B-16B params: $0.20/1M tokens
#   - >16B params: $0.90/1M tokens
#   - MoE 0-56B: $0.50/1M tokens
#   - MoE 56.1B-176B: $1.20/1M tokens
# Special pricing for specific models noted below
# Last verified via API: 2026-01-08
# =============================================================================

FIREWORKS_PRICING: dict[str, ModelPricing] = {
    # =========================================================================
    # Llama 4 Series (Latest - January 2026)
    # =========================================================================
    "accounts/fireworks/models/llama4-scout-instruct-basic": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/llama4-maverick-instruct-basic": ModelPricing(0.90, 0.90),
    # =========================================================================
    # Llama 3.3 Series
    # =========================================================================
    "accounts/fireworks/models/llama-v3p3-70b-instruct": ModelPricing(0.90, 0.90),
    # =========================================================================
    # Qwen3 Series (Latest - MoE models)
    # =========================================================================
    "accounts/fireworks/models/qwen3-235b-a22b": ModelPricing(0.22, 0.88),
    "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507": ModelPricing(0.22, 0.88),
    "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507": ModelPricing(0.22, 0.88),
    "accounts/fireworks/models/qwen3-30b-a3b": ModelPricing(0.15, 0.60),
    "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct": ModelPricing(0.45, 1.80),
    "accounts/fireworks/models/qwen3-coder-30b-a3b-instruct": ModelPricing(0.15, 0.60),
    "accounts/fireworks/models/qwen3-8b": ModelPricing(0.20, 0.20),
    # =========================================================================
    # Qwen 2.5 VL (Vision-Language) Series
    # =========================================================================
    "accounts/fireworks/models/qwen2p5-vl-32b-instruct": ModelPricing(0.90, 0.90),
    # =========================================================================
    # Qwen3 VL Series (Vision-Language)
    # =========================================================================
    "accounts/fireworks/models/qwen3-vl-235b-a22b-instruct": ModelPricing(0.22, 0.88),
    "accounts/fireworks/models/qwen3-vl-235b-a22b-thinking": ModelPricing(0.22, 0.88),
    "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct": ModelPricing(0.15, 0.60),
    "accounts/fireworks/models/qwen3-vl-30b-a3b-thinking": ModelPricing(0.15, 0.60),
    # =========================================================================
    # DeepSeek Series
    # =========================================================================
    "accounts/fireworks/models/deepseek-v3-0324": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/deepseek-v3p1": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/deepseek-v3p1-terminus": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/deepseek-v3p2": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/deepseek-r1-0528": ModelPricing(1.35, 5.40),
    # =========================================================================
    # Mixtral (MoE) Series
    # =========================================================================
    "accounts/fireworks/models/mixtral-8x22b-instruct": ModelPricing(1.20, 1.20),  # MoE >56B
    # =========================================================================
    # GLM Series (Zhipu/THUDM)
    # =========================================================================
    "accounts/fireworks/models/glm-4p5": ModelPricing(0.55, 2.19),
    "accounts/fireworks/models/glm-4p6": ModelPricing(0.55, 2.19),
    "accounts/fireworks/models/glm-4p7": ModelPricing(0.55, 2.19),
    # =========================================================================
    # Kimi / Moonshot Series
    # =========================================================================
    "accounts/fireworks/models/kimi-k2-instruct-0905": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/kimi-k2-thinking": ModelPricing(0.90, 0.90),
    # =========================================================================
    # MiniMax Series
    # =========================================================================
    "accounts/fireworks/models/minimax-m2": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/minimax-m2p1": ModelPricing(0.90, 0.90),
    # =========================================================================
    # GPT-OSS (Open Source GPT-like models)
    # =========================================================================
    "accounts/fireworks/models/gpt-oss-20b": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/gpt-oss-120b": ModelPricing(0.90, 0.90),
    # =========================================================================
    # Cogito Series
    # =========================================================================
    "accounts/cogito/models/cogito-671b-v2-p1": ModelPricing(0.90, 0.90),
    # =========================================================================
    # Image Generation Models (FLUX)
    # =========================================================================
    "accounts/fireworks/models/flux-1-dev-fp8": ModelPricing(0.0005, 0.0),  # per step
    "accounts/fireworks/models/flux-1-schnell-fp8": ModelPricing(0.00035, 0.0),  # per step
    "accounts/fireworks/models/flux-kontext-pro": ModelPricing(0.04, 0.0),  # per image
    "accounts/fireworks/models/flux-kontext-max": ModelPricing(0.08, 0.0),  # per image
    # =========================================================================
    # Embedding Models
    # Pricing: up to 150M=$0.008, 150M-350M=$0.016, Qwen3 8B=$0.10
    # =========================================================================
    "nomic-ai/nomic-embed-text-v1.5": ModelPricing(0.008, 0.0),
    "nomic-ai/nomic-embed-text-v1": ModelPricing(0.008, 0.0),
    "thenlper/gte-large": ModelPricing(0.016, 0.0),
    "thenlper/gte-base": ModelPricing(0.008, 0.0),
    "BAAI/bge-base-en-v1.5": ModelPricing(0.008, 0.0),
    "BAAI/bge-small-en-v1.5": ModelPricing(0.008, 0.0),
    "BAAI/bge-large-en-v1.5": ModelPricing(0.016, 0.0),
    "WhereIsAI/UAE-Large-V1": ModelPricing(0.016, 0.0),
    "mixedbread-ai/mxbai-embed-large-v1": ModelPricing(0.016, 0.0),
    "sentence-transformers/all-MiniLM-L6-v2": ModelPricing(0.008, 0.0),
    "accounts/fireworks/models/qwen3-embedding-8b": ModelPricing(0.10, 0.0),
    # =========================================================================
    # Reranker Models
    # =========================================================================
    "accounts/fireworks/models/qwen3-reranker-8b": ModelPricing(0.10, 0.0),
}


# =============================================================================
# DeepSeek Pricing (USD per 1M tokens)
# https://platform.deepseek.com/api-docs/pricing
# =============================================================================

DEEPSEEK_PRICING: dict[str, ModelPricing] = {
    "deepseek-chat": ModelPricing(0.14, 0.28, 0.014),  # V3
    "deepseek-reasoner": ModelPricing(0.55, 2.19),  # R1
    "deepseek-coder": ModelPricing(0.14, 0.28),
}


# =============================================================================
# Perplexity Pricing (USD per 1M tokens)
# https://docs.perplexity.ai/guides/pricing
# Note: Perplexity also charges per-request fees based on search_context_size:
#   - High: $12/1K requests
#   - Medium: $8/1K requests
#   - Low: $5/1K requests
# Last verified: 2026-01-08
# =============================================================================

PERPLEXITY_PRICING: dict[str, ModelPricing] = {
    # =========================================================================
    # Sonar Family (Search-Augmented Models)
    # =========================================================================
    # Sonar - Fast, efficient search (128K context)
    "sonar": ModelPricing(1.00, 1.00),
    # Sonar Pro - Advanced search with grounding (200K context, 8K output)
    "sonar-pro": ModelPricing(3.00, 15.00),
    # Sonar Pro Search - Latest advanced search variant
    "sonar-pro-search": ModelPricing(3.00, 15.00),
    # =========================================================================
    # Sonar Reasoning Family
    # =========================================================================
    # Sonar Reasoning - Reasoning with web search (127K context)
    "sonar-reasoning": ModelPricing(1.00, 5.00),
    # Sonar Reasoning Pro - Advanced reasoning (128K context)
    "sonar-reasoning-pro": ModelPricing(2.00, 8.00),
    # =========================================================================
    # Sonar Deep Research
    # =========================================================================
    # Deep Research - Multi-source research (128K context)
    "sonar-deep-research": ModelPricing(2.00, 8.00),
    # =========================================================================
    # Special/Experimental Models
    # =========================================================================
    # R1-1776 - Experimental reasoning model
    "r1-1776": ModelPricing(2.00, 8.00),
    # =========================================================================
    # Legacy Models (may be deprecated)
    # =========================================================================
    "llama-3.1-sonar-small-128k-online": ModelPricing(0.20, 0.20),
    "llama-3.1-sonar-large-128k-online": ModelPricing(1.00, 1.00),
    "llama-3.1-sonar-huge-128k-online": ModelPricing(5.00, 5.00),
}


# =============================================================================
# Combined pricing lookup
# =============================================================================

ALL_PRICING: dict[str, dict[str, ModelPricing]] = {
    "openai": OPENAI_PRICING,
    "anthropic": ANTHROPIC_PRICING,
    "gemini": GEMINI_PRICING,
    "vertex_ai": GEMINI_PRICING,  # Same models, same pricing
    "mistral": MISTRAL_PRICING,
    "cohere": COHERE_PRICING,
    "groq": GROQ_PRICING,
    "together_ai": TOGETHER_PRICING,
    "fireworks_ai": FIREWORKS_PRICING,
    "deepseek": DEEPSEEK_PRICING,
    "perplexity": PERPLEXITY_PRICING,
}


class UnknownModelPricingError(ArcLLMError):
    """Raised when pricing is not available for a model."""

    pass


def _normalize_model_name(model: str) -> tuple[str | None, str]:
    """
    Normalize model name and extract provider if specified.

    Returns:
        Tuple of (provider or None, normalized model name)
    """
    provider = None
    model_name = model

    # Check for provider prefix
    if "/" in model:
        parts = model.split("/", 1)
        if parts[0].lower() in ALL_PRICING:
            provider = parts[0].lower()
            model_name = parts[1]
        elif parts[0].lower().replace("-", "_") in ALL_PRICING:
            provider = parts[0].lower().replace("-", "_")
            model_name = parts[1]

    return provider, model_name


def get_model_pricing(model: str) -> ModelPricing:
    """
    Get pricing information for a model.

    Args:
        model: Model identifier (with or without provider prefix)

    Returns:
        ModelPricing with input and output costs per million tokens

    Raises:
        UnknownModelPricingError: If pricing is not available
    """
    provider, model_name = _normalize_model_name(model)

    # If provider specified, look only in that provider's pricing
    if provider:
        pricing_table = ALL_PRICING.get(provider, {})
        if model_name in pricing_table:
            return pricing_table[model_name]
        raise UnknownModelPricingError(
            f"No pricing available for model '{model_name}' from provider '{provider}'",
            model=model,
            provider=provider,
        )

    # Search all providers
    for pricing_table in ALL_PRICING.values():
        if model_name in pricing_table:
            return pricing_table[model_name]

    raise UnknownModelPricingError(
        f"No pricing available for model '{model}'",
        model=model,
    )


def cost_per_token(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> tuple[float, float]:
    """
    Calculate cost for given token counts.

    Args:
        model: Model identifier
        prompt_tokens: Number of prompt/input tokens
        completion_tokens: Number of completion/output tokens

    Returns:
        Tuple of (prompt_cost, completion_cost) in USD

    Raises:
        UnknownModelPricingError: If pricing is not available
    """
    pricing = get_model_pricing(model)

    prompt_cost = (prompt_tokens / 1_000_000) * pricing.input_cost_per_million
    completion_cost = (completion_tokens / 1_000_000) * pricing.output_cost_per_million

    return (prompt_cost, completion_cost)


def completion_cost(
    response: ModelResponse,
    model: str | None = None,
) -> float:
    """
    Calculate total cost for a completion response.

    Args:
        response: ModelResponse from completion call
        model: Optional model override (uses response.model if not provided)

    Returns:
        Total cost in USD

    Raises:
        UnknownModelPricingError: If pricing is not available
    """
    model_name = model or response.model
    if not model_name:
        raise ValueError("Model name required but not provided")

    usage = response.usage
    if usage is None:
        return 0.0

    prompt_cost, completion_cost = cost_per_token(
        model_name,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )

    return prompt_cost + completion_cost
