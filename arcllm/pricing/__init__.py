"""
Pricing tables and cost calculation for arcllm.

Provides:
- cost_per_token(model, prompt_tokens, completion_tokens, ...)
- completion_cost(response, model)
- get_model_pricing(model)
- image_cost(model, n)
- audio_cost(model, characters=..., seconds=...)
- rerank_cost(model, queries)
"""

from arcllm.pricing.tables import (
    PRICING_VERSION,
    ModelPricing,
    audio_cost,
    completion_cost,
    cost_per_token,
    get_model_pricing,
    image_cost,
    rerank_cost,
)

__all__ = [
    "PRICING_VERSION",
    "ModelPricing",
    "audio_cost",
    "completion_cost",
    "cost_per_token",
    "get_model_pricing",
    "image_cost",
    "rerank_cost",
]
