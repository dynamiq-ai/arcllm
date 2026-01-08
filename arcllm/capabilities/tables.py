"""
Model capability tables for supported models.

Last updated: 2025-01-08

Capabilities tracked:
- max_tokens: Maximum output tokens
- context_window: Maximum input context length
- supports_vision: Can process image inputs
- supports_pdf_input: Can process PDF documents directly
- supports_tools: Supports tool/function calling
- supports_structured_output: Supports JSON mode or JSON schema output

To update capabilities:
1. Check provider documentation
2. Update the relevant dict in this file
3. Update CAPABILITIES_VERSION
4. Run tests to ensure format is valid
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CAPABILITIES_VERSION",
    "ModelCapabilities",
    "get_max_tokens",
    "get_model_capabilities",
    "supports_pdf_input",
    "supports_structured_output",
    "supports_tools",
    "supports_vision",
]


# Version for tracking capability table updates
CAPABILITIES_VERSION = "2026.01.08"


@dataclass(slots=True, frozen=True)
class ModelCapabilities:
    """Capability information for a model."""

    max_tokens: int | None
    context_window: int | None
    supports_vision: bool = False
    supports_pdf_input: bool = False
    supports_tools: bool = False
    supports_structured_output: bool = False


# =============================================================================
# OpenAI Capabilities
# =============================================================================
#
# Official Documentation:
# - Models Overview: https://platform.openai.com/docs/models
# - GPT-5: https://platform.openai.com/docs/models/gpt-5
# - o1/o3 Reasoning: https://platform.openai.com/docs/models/o1
# - GPT-4o: https://platform.openai.com/docs/models/gpt-4o (retiring Feb 2026)
#
# HOW TO UPDATE:
# 1. Check https://platform.openai.com/docs/models for current capabilities
# 2. Update the dict below
# 3. Update CAPABILITIES_VERSION at top of file
# 4. Run: pytest tests/test_capabilities.py
#
# CAPABILITY FORMAT:
# ModelCapabilities(max_tokens, context_window, vision, pdf, tools, structured)
#
# NOTES:
# - max_tokens: Maximum output tokens the model can generate
# - context_window: Maximum input context length (in tokens)
# - supports_vision: True if model accepts image inputs
# - supports_tools: True if model supports function/tool calling
# - supports_structured_output: True if model supports JSON mode/schema
#
# o1 MODELS:
# - Use max_completion_tokens instead of max_tokens
# - o1-preview/o1-mini have limited feature support
# - Full o1 supports vision, tools, and structured output
#

OPENAI_CAPABILITIES: dict[str, ModelCapabilities] = {
    # =========================================================================
    # GPT-5.2 series (Latest - December 2025)
    # =========================================================================
    "gpt-5.2": ModelCapabilities(32768, 256000, True, True, True, True),
    "gpt-5.2-2025-12-11": ModelCapabilities(32768, 256000, True, True, True, True),
    "gpt-5.2-chat-latest": ModelCapabilities(32768, 256000, True, True, True, True),
    "gpt-5.2-pro": ModelCapabilities(131072, 256000, True, True, True, True),
    "gpt-5.2-pro-2025-12-11": ModelCapabilities(131072, 256000, True, True, True, True),
    # =========================================================================
    # GPT-5.1 series (November 2025)
    # =========================================================================
    "gpt-5.1": ModelCapabilities(32768, 200000, True, True, True, True),
    "gpt-5.1-2025-11-13": ModelCapabilities(32768, 200000, True, True, True, True),
    "gpt-5.1-chat-latest": ModelCapabilities(32768, 200000, True, True, True, True),
    "gpt-5.1-codex": ModelCapabilities(32768, 200000, False, False, True, True),
    "gpt-5.1-codex-max": ModelCapabilities(65536, 200000, False, False, True, True),
    "gpt-5.1-codex-mini": ModelCapabilities(16384, 128000, False, False, True, True),
    # =========================================================================
    # GPT-5 series (August 2025)
    # =========================================================================
    "gpt-5": ModelCapabilities(32768, 200000, True, True, True, True),
    "gpt-5-2025-08-07": ModelCapabilities(32768, 200000, True, True, True, True),
    "gpt-5-chat-latest": ModelCapabilities(32768, 200000, True, True, True, True),
    "gpt-5-mini": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-5-mini-2025-08-07": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-5-nano": ModelCapabilities(8192, 64000, True, False, True, True),
    "gpt-5-nano-2025-08-07": ModelCapabilities(8192, 64000, True, False, True, True),
    "gpt-5-pro": ModelCapabilities(131072, 256000, True, True, True, True),
    "gpt-5-pro-2025-10-06": ModelCapabilities(131072, 256000, True, True, True, True),
    "gpt-5-codex": ModelCapabilities(32768, 200000, False, False, True, True),
    "gpt-5-search-api": ModelCapabilities(32768, 200000, True, False, True, True),
    # =========================================================================
    # GPT-4.1 series
    # =========================================================================
    "gpt-4.1": ModelCapabilities(32768, 128000, True, False, True, True),
    "gpt-4.1-2025-04-14": ModelCapabilities(32768, 128000, True, False, True, True),
    "gpt-4.1-mini": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4.1-mini-2025-04-14": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4.1-nano": ModelCapabilities(8192, 64000, True, False, True, True),
    "gpt-4.1-nano-2025-04-14": ModelCapabilities(8192, 64000, True, False, True, True),
    # =========================================================================
    # o3 series (reasoning - 2025)
    # =========================================================================
    "o3": ModelCapabilities(131072, 256000, True, True, True, True),
    "o3-2025-04-16": ModelCapabilities(131072, 256000, True, True, True, True),
    "o3-mini": ModelCapabilities(65536, 128000, True, False, True, True),
    "o3-mini-2025-01-31": ModelCapabilities(65536, 128000, True, False, True, True),
    "o3-pro": ModelCapabilities(200000, 256000, True, True, True, True),
    "o3-pro-2025-06-10": ModelCapabilities(200000, 256000, True, True, True, True),
    "o3-deep-research": ModelCapabilities(200000, 256000, True, True, True, True),
    # =========================================================================
    # o1 series (reasoning)
    # =========================================================================
    "o1": ModelCapabilities(100000, 200000, True, False, True, True),
    "o1-2024-12-17": ModelCapabilities(100000, 200000, True, False, True, True),
    "o1-mini": ModelCapabilities(65536, 128000, False, False, False, False),
    "o1-mini-2024-09-12": ModelCapabilities(65536, 128000, False, False, False, False),
    "o1-pro": ModelCapabilities(131072, 200000, True, False, True, True),
    "o1-pro-2025-03-19": ModelCapabilities(131072, 200000, True, False, True, True),
    # =========================================================================
    # GPT-4o series (still active)
    # =========================================================================
    "gpt-4o": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4o-2024-11-20": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4o-2024-08-06": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4o-mini": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4o-mini-2024-07-18": ModelCapabilities(16384, 128000, True, False, True, True),
    "chatgpt-4o-latest": ModelCapabilities(16384, 128000, True, False, True, True),
    # =========================================================================
    # Legacy models
    # =========================================================================
    "gpt-4-turbo": ModelCapabilities(4096, 128000, True, False, True, True),
    "gpt-4": ModelCapabilities(8192, 8192, False, False, True, False),
    "gpt-3.5-turbo": ModelCapabilities(4096, 16385, False, False, True, True),
}


# =============================================================================
# Anthropic Capabilities
# =============================================================================
# API Reference: https://docs.anthropic.com/en/docs/about-claude/models
# Pricing: https://www.anthropic.com/pricing
#
# HOW TO UPDATE:
# 1. Check https://docs.anthropic.com/en/docs/about-claude/models for current models
# 2. Update the dict below with new models
# 3. Update CAPABILITIES_VERSION at top of file
# 4. Run: pytest tests/test_capabilities.py
#
# NOTES ON STRUCTURED OUTPUT:
# - Claude 4+ supports native JSON mode via response_format parameter
# - Claude 3.x does NOT support response_format - use system prompts instead
# - See: https://docs.anthropic.com/en/docs/build-with-claude/structured-output

ANTHROPIC_CAPABILITIES: dict[str, ModelCapabilities] = {
    # =========================================================================
    # Claude 4.5 series (Current flagship - released November 2025)
    # https://docs.anthropic.com/en/docs/about-claude/models#claude-4-5-family
    # =========================================================================
    # Claude 4.5 Opus - Most powerful model, best for complex reasoning
    # 500K context, 32K output, all capabilities including native JSON mode
    "claude-4-5-opus-20251120": ModelCapabilities(32768, 500000, True, True, True, True),
    "claude-4-5-opus-latest": ModelCapabilities(32768, 500000, True, True, True, True),
    # Claude 4.5 Sonnet - Balanced performance and cost
    "claude-4-5-sonnet-20251015": ModelCapabilities(16384, 500000, True, True, True, True),
    "claude-4-5-sonnet-latest": ModelCapabilities(16384, 500000, True, True, True, True),
    # Claude 4.5 Haiku - Fast and affordable
    "claude-4-5-haiku-20251201": ModelCapabilities(16384, 500000, True, True, True, True),
    "claude-4-5-haiku-latest": ModelCapabilities(16384, 500000, True, True, True, True),
    # =========================================================================
    # Claude 4 series (Released June 2025)
    # https://docs.anthropic.com/en/docs/about-claude/models#claude-4-family
    # =========================================================================
    # Claude 4 Opus
    "claude-4-opus-20250615": ModelCapabilities(16384, 300000, True, True, True, True),
    "claude-4-opus-latest": ModelCapabilities(16384, 300000, True, True, True, True),
    # Claude 4 Sonnet
    "claude-4-sonnet-20250601": ModelCapabilities(16384, 300000, True, True, True, True),
    "claude-4-sonnet-latest": ModelCapabilities(16384, 300000, True, True, True, True),
    # Claude 4 Haiku
    "claude-4-haiku-20250701": ModelCapabilities(8192, 300000, True, True, True, True),
    "claude-4-haiku-latest": ModelCapabilities(8192, 300000, True, True, True, True),
    # =========================================================================
    # Claude 3.5 series (DEPRECATED - retiring March 2026)
    # https://docs.anthropic.com/en/docs/about-claude/models#claude-3-5-family
    # =========================================================================
    "claude-3-5-sonnet-20241022": ModelCapabilities(8192, 200000, True, True, True, False),
    "claude-3-5-sonnet-latest": ModelCapabilities(8192, 200000, True, True, True, False),
    "claude-3-5-haiku-20241022": ModelCapabilities(8192, 200000, True, False, True, False),
    "claude-3-5-haiku-latest": ModelCapabilities(8192, 200000, True, False, True, False),
    # =========================================================================
    # Claude 3 series (DEPRECATED - for backwards compatibility)
    # =========================================================================
    "claude-3-opus-20240229": ModelCapabilities(4096, 200000, True, False, True, False),
    "claude-3-opus-latest": ModelCapabilities(4096, 200000, True, False, True, False),
    "claude-3-sonnet-20240229": ModelCapabilities(4096, 200000, True, False, True, False),
    "claude-3-haiku-20240307": ModelCapabilities(4096, 200000, True, False, True, False),
    # =========================================================================
    # Legacy Claude 2 (DEPRECATED - end of life)
    # =========================================================================
    "claude-2.1": ModelCapabilities(4096, 200000, False, False, False, False),
}


# =============================================================================
# Google Gemini Capabilities
# =============================================================================

GEMINI_CAPABILITIES: dict[str, ModelCapabilities] = {
    # Gemini 2.0
    "gemini-2.0-flash-exp": ModelCapabilities(8192, 1048576, True, True, True, True),
    "gemini-2.0-flash-thinking-exp": ModelCapabilities(8192, 32767, False, False, False, False),
    # Gemini 1.5 Pro
    "gemini-1.5-pro": ModelCapabilities(8192, 2097152, True, True, True, True),
    "gemini-1.5-pro-latest": ModelCapabilities(8192, 2097152, True, True, True, True),
    "gemini-1.5-pro-001": ModelCapabilities(8192, 2097152, True, True, True, True),
    "gemini-1.5-pro-002": ModelCapabilities(8192, 2097152, True, True, True, True),
    # Gemini 1.5 Flash
    "gemini-1.5-flash": ModelCapabilities(8192, 1048576, True, True, True, True),
    "gemini-1.5-flash-latest": ModelCapabilities(8192, 1048576, True, True, True, True),
    "gemini-1.5-flash-001": ModelCapabilities(8192, 1048576, True, True, True, True),
    "gemini-1.5-flash-002": ModelCapabilities(8192, 1048576, True, True, True, True),
    "gemini-1.5-flash-8b": ModelCapabilities(8192, 1048576, True, True, True, True),
    # Gemini 1.0 Pro
    "gemini-1.0-pro": ModelCapabilities(8192, 32768, False, False, True, False),
    "gemini-pro": ModelCapabilities(8192, 32768, False, False, True, False),
}


# =============================================================================
# Mistral Capabilities
# =============================================================================

MISTRAL_CAPABILITIES: dict[str, ModelCapabilities] = {
    # Large models
    "mistral-large-latest": ModelCapabilities(131072, 131072, False, False, True, True),
    "mistral-large-2411": ModelCapabilities(131072, 131072, False, False, True, True),
    "mistral-large-2407": ModelCapabilities(131072, 131072, False, False, True, True),
    # Pixtral (vision)
    "pixtral-large-latest": ModelCapabilities(131072, 131072, True, False, True, True),
    "pixtral-large-2411": ModelCapabilities(131072, 131072, True, False, True, True),
    "pixtral-12b-2409": ModelCapabilities(131072, 131072, True, False, True, False),
    # Small models
    "mistral-small-latest": ModelCapabilities(32768, 32768, False, False, True, True),
    "mistral-small-2409": ModelCapabilities(32768, 32768, False, False, True, True),
    "mistral-nemo-latest": ModelCapabilities(131072, 131072, False, False, True, True),
    "mistral-nemo-2407": ModelCapabilities(131072, 131072, False, False, True, True),
    # Codestral
    "codestral-latest": ModelCapabilities(32768, 32768, False, False, True, False),
    "codestral-2405": ModelCapabilities(32768, 32768, False, False, True, False),
    # Ministral
    "ministral-3b-latest": ModelCapabilities(131072, 131072, False, False, True, False),
    "ministral-8b-latest": ModelCapabilities(131072, 131072, False, False, True, False),
}


# =============================================================================
# Cohere Capabilities
# =============================================================================

COHERE_CAPABILITIES: dict[str, ModelCapabilities] = {
    "command-r-plus": ModelCapabilities(4096, 128000, False, False, True, False),
    "command-r-plus-08-2024": ModelCapabilities(4096, 128000, False, False, True, False),
    "command-r": ModelCapabilities(4096, 128000, False, False, True, False),
    "command-r-08-2024": ModelCapabilities(4096, 128000, False, False, True, False),
    "command": ModelCapabilities(4096, 4096, False, False, False, False),
    "command-light": ModelCapabilities(4096, 4096, False, False, False, False),
}


# =============================================================================
# Groq Capabilities
# =============================================================================

# =============================================================================
# Groq Capabilities
# https://console.groq.com/docs/models
#
# Last updated: 2026-01-08
# To update: Check https://console.groq.com/docs/models for current models
# =============================================================================

GROQ_CAPABILITIES: dict[str, ModelCapabilities] = {
    # =========================================================================
    # Llama 4 series (Latest - January 2026)
    # =========================================================================
    "meta-llama/llama-4-maverick-17b-128e-instruct": ModelCapabilities(
        8192,
        131072,
        True,
        False,
        True,
        True,  # Vision, tools, structured output
    ),
    "meta-llama/llama-4-scout-17b-16e-instruct": ModelCapabilities(
        8192,
        131072,
        True,
        False,
        True,
        True,  # Vision, tools, structured output
    ),
    # =========================================================================
    # OpenAI GPT-OSS (Open-weight models on Groq)
    # =========================================================================
    "openai/gpt-oss-120b": ModelCapabilities(
        65536,
        131072,
        False,
        False,
        True,
        True,  # Strong reasoning and tool use
    ),
    "openai/gpt-oss-20b": ModelCapabilities(
        65536,
        131072,
        False,
        False,
        True,
        True,  # Few-shot function calling
    ),
    "openai/gpt-oss-safeguard-20b": ModelCapabilities(
        65536,
        131072,
        False,
        False,
        False,
        False,  # Safety model
    ),
    # =========================================================================
    # Moonshot Kimi K2
    # =========================================================================
    "moonshotai/kimi-k2-instruct": ModelCapabilities(16384, 131072, False, False, True, True),
    "moonshotai/kimi-k2-instruct-0905": ModelCapabilities(
        16384,
        262144,
        False,
        False,
        True,
        True,  # 256K context
    ),
    # =========================================================================
    # Qwen 3 series
    # =========================================================================
    "qwen/qwen3-32b": ModelCapabilities(40960, 131072, False, False, True, True),
    # =========================================================================
    # Groq Compound (agentic models with built-in tools)
    # =========================================================================
    "groq/compound": ModelCapabilities(
        8192,
        131072,
        False,
        False,
        True,
        True,  # Built-in web search, code exec
    ),
    "groq/compound-mini": ModelCapabilities(
        8192,
        131072,
        False,
        False,
        True,
        True,  # Lightweight agentic model
    ),
    # =========================================================================
    # Llama 3.3
    # =========================================================================
    "llama-3.3-70b-versatile": ModelCapabilities(32768, 131072, False, False, True, True),
    # =========================================================================
    # Llama 3.1
    # =========================================================================
    "llama-3.1-8b-instant": ModelCapabilities(
        131072,
        131072,
        False,
        False,
        True,
        True,  # Fast inference
    ),
    # =========================================================================
    # Llama Guard (safety models)
    # =========================================================================
    "meta-llama/llama-guard-4-12b": ModelCapabilities(
        1024,
        131072,
        False,
        False,
        False,
        False,  # Safety classification
    ),
    "meta-llama/llama-prompt-guard-2-86m": ModelCapabilities(
        512,
        512,
        False,
        False,
        False,
        False,  # Prompt injection detection
    ),
    "meta-llama/llama-prompt-guard-2-22m": ModelCapabilities(
        512,
        512,
        False,
        False,
        False,
        False,  # Prompt injection detection
    ),
    # =========================================================================
    # Other models
    # =========================================================================
    "allam-2-7b": ModelCapabilities(
        4096,
        4096,
        False,
        False,
        False,
        False,  # SDAIA Arabic model
    ),
}


# =============================================================================
# Together AI Capabilities
# https://docs.together.ai/docs/inference-models
# =============================================================================

TOGETHER_CAPABILITIES: dict[str, ModelCapabilities] = {
    # =========================================================================
    # Llama 4 series (Latest - January 2026)
    # =========================================================================
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": ModelCapabilities(
        8192,
        1048576,
        True,
        False,
        True,
        True,  # 1M context window
    ),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": ModelCapabilities(
        8192, 131072, True, False, True, True
    ),
    # =========================================================================
    # Llama 3.3
    # =========================================================================
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelCapabilities(
        8192, 131072, False, False, True, True
    ),
    # =========================================================================
    # Llama 3.2 Vision (requires dedicated endpoint)
    # =========================================================================
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": ModelCapabilities(
        4096, 131072, True, False, True, False
    ),
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": ModelCapabilities(
        4096, 131072, True, False, True, False
    ),
    # =========================================================================
    # Llama 3.2
    # =========================================================================
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": ModelCapabilities(
        4096, 131072, False, False, True, True
    ),
    # =========================================================================
    # Llama 3.1
    # =========================================================================
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": ModelCapabilities(
        4096, 131072, False, False, True, True
    ),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelCapabilities(
        4096, 131072, False, False, True, True
    ),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelCapabilities(
        4096, 131072, False, False, True, True
    ),
    # =========================================================================
    # Qwen 2.5
    # =========================================================================
    "Qwen/Qwen2.5-72B-Instruct-Turbo": ModelCapabilities(4096, 131072, False, False, True, True),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": ModelCapabilities(4096, 131072, False, False, True, True),
    # =========================================================================
    # DeepSeek
    # =========================================================================
    "deepseek-ai/DeepSeek-R1": ModelCapabilities(8192, 131072, False, False, True, True),
    "deepseek-ai/DeepSeek-V3": ModelCapabilities(8192, 131072, False, False, True, True),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ModelCapabilities(
        8192, 131072, False, False, True, True
    ),
    # =========================================================================
    # Mixtral (may require dedicated endpoint)
    # =========================================================================
    "mistralai/Mixtral-8x22B-Instruct-v0.1": ModelCapabilities(
        4096, 65536, False, False, True, True
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelCapabilities(
        4096, 32768, False, False, True, True
    ),
}


# =============================================================================
# DeepSeek Capabilities
# =============================================================================

DEEPSEEK_CAPABILITIES: dict[str, ModelCapabilities] = {
    "deepseek-chat": ModelCapabilities(8192, 65536, False, False, True, True),
    "deepseek-reasoner": ModelCapabilities(8192, 65536, False, False, False, False),
    "deepseek-coder": ModelCapabilities(8192, 65536, False, False, True, True),
}


# =============================================================================
# Perplexity Capabilities
# =============================================================================

PERPLEXITY_CAPABILITIES: dict[str, ModelCapabilities] = {
    "sonar-pro": ModelCapabilities(8192, 200000, False, False, False, False),
    "sonar": ModelCapabilities(8192, 128000, False, False, False, False),
    "sonar-reasoning-pro": ModelCapabilities(8192, 128000, False, False, False, False),
    "sonar-reasoning": ModelCapabilities(8192, 128000, False, False, False, False),
}


# =============================================================================
# Combined capability lookup
# =============================================================================

ALL_CAPABILITIES: dict[str, dict[str, ModelCapabilities]] = {
    "openai": OPENAI_CAPABILITIES,
    "anthropic": ANTHROPIC_CAPABILITIES,
    "gemini": GEMINI_CAPABILITIES,
    "vertex_ai": GEMINI_CAPABILITIES,
    "mistral": MISTRAL_CAPABILITIES,
    "cohere": COHERE_CAPABILITIES,
    "groq": GROQ_CAPABILITIES,
    "together_ai": TOGETHER_CAPABILITIES,
    "deepseek": DEEPSEEK_CAPABILITIES,
    "perplexity": PERPLEXITY_CAPABILITIES,
}

# Default capabilities for unknown models
DEFAULT_CAPABILITIES = ModelCapabilities(
    max_tokens=4096,
    context_window=8192,
    supports_vision=False,
    supports_pdf_input=False,
    supports_tools=False,
    supports_structured_output=False,
)


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
        if parts[0].lower() in ALL_CAPABILITIES:
            provider = parts[0].lower()
            model_name = parts[1]
        elif parts[0].lower().replace("-", "_") in ALL_CAPABILITIES:
            provider = parts[0].lower().replace("-", "_")
            model_name = parts[1]

    return provider, model_name


def get_model_capabilities(model: str) -> ModelCapabilities:
    """
    Get capability information for a model.

    Args:
        model: Model identifier (with or without provider prefix)

    Returns:
        ModelCapabilities for the model, or DEFAULT_CAPABILITIES if unknown
    """
    provider, model_name = _normalize_model_name(model)

    # If provider specified, look only in that provider's capabilities
    if provider:
        cap_table = ALL_CAPABILITIES.get(provider, {})
        if model_name in cap_table:
            return cap_table[model_name]
        return DEFAULT_CAPABILITIES

    # Search all providers
    for cap_table in ALL_CAPABILITIES.values():
        if model_name in cap_table:
            return cap_table[model_name]

    return DEFAULT_CAPABILITIES


def get_max_tokens(model: str) -> int | None:
    """
    Get maximum output tokens for a model.

    Args:
        model: Model identifier

    Returns:
        Maximum tokens or None if unknown
    """
    caps = get_model_capabilities(model)
    return caps.max_tokens


def supports_vision(model: str) -> bool:
    """
    Check if model supports vision/image inputs.

    Args:
        model: Model identifier

    Returns:
        True if model supports vision
    """
    caps = get_model_capabilities(model)
    return caps.supports_vision


def supports_pdf_input(model: str) -> bool:
    """
    Check if model supports direct PDF document input.

    Args:
        model: Model identifier

    Returns:
        True if model supports PDF input
    """
    caps = get_model_capabilities(model)
    return caps.supports_pdf_input


def supports_tools(model: str) -> bool:
    """
    Check if model supports tool/function calling.

    Args:
        model: Model identifier

    Returns:
        True if model supports tools
    """
    caps = get_model_capabilities(model)
    return caps.supports_tools


def supports_structured_output(model: str) -> bool:
    """
    Check if model supports structured output (JSON mode/schema).

    Args:
        model: Model identifier

    Returns:
        True if model supports structured output
    """
    caps = get_model_capabilities(model)
    return caps.supports_structured_output
