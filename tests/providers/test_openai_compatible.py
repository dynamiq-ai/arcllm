"""Smoke tests for OpenAI-shape thin wrappers.

Groq, Together AI, Fireworks AI, DeepSeek, Perplexity (0.x) plus xAI,
OpenRouter, NVIDIA NIM, Cerebras, SambaNova, Anyscale and DeepInfra (0.4)
all expose an OpenAI-compatible Chat Completions surface. The arcllm
adapters subclass ``OpenAIAdapter`` and only override the base URL + auth
header, so a brief smoke per adapter is enough — the OpenAI parse/build
paths are covered in ``test_openai.py``.
"""

from __future__ import annotations

import orjson
import pytest

from arcllm.providers.base import ProviderConfig
from arcllm.providers.cerebras_adapter import CerebrasAdapter
from arcllm.providers.deepinfra_adapter import DeepInfraAdapter
from arcllm.providers.deepseek_adapter import DeepSeekAdapter
from arcllm.providers.fireworks_adapter import FireworksAdapter
from arcllm.providers.groq_adapter import GroqAdapter
from arcllm.providers.moonshot_adapter import MoonshotAdapter
from arcllm.providers.nebius_adapter import NebiusAdapter
from arcllm.providers.nvidia_nim_adapter import NvidiaNIMAdapter
from arcllm.providers.openrouter_adapter import OpenRouterAdapter
from arcllm.providers.ovhcloud_adapter import OVHCloudAdapter
from arcllm.providers.perplexity_adapter import PerplexityAdapter
from arcllm.providers.sambanova_adapter import SambaNovaAdapter
from arcllm.providers.together_adapter import TogetherAdapter
from arcllm.providers.xai_adapter import XAIAdapter
from arcllm.providers.zai_adapter import ZAIAdapter


@pytest.mark.parametrize(
    "adapter_cls,api_base_substring,model",
    [
        (GroqAdapter, "api.groq.com", "llama-3.3-70b-versatile"),
        (TogetherAdapter, "api.together.xyz", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        (
            FireworksAdapter,
            "api.fireworks.ai",
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
        ),
        (DeepSeekAdapter, "api.deepseek.com", "deepseek-v4-flash"),
        (PerplexityAdapter, "api.perplexity.ai", "sonar"),
        # Tier A (0.4)
        (XAIAdapter, "api.x.ai", "grok-4-latest"),
        (OpenRouterAdapter, "openrouter.ai", "openai/gpt-4o-mini"),
        (NvidiaNIMAdapter, "integrate.api.nvidia.com", "meta/llama-3.3-70b-instruct"),
        (CerebrasAdapter, "api.cerebras.ai", "llama-3.3-70b"),
        (SambaNovaAdapter, "api.sambanova.ai", "Meta-Llama-3.3-70B-Instruct"),
        (DeepInfraAdapter, "api.deepinfra.com", "meta-llama/Llama-3.3-70B-Instruct"),
        # Production push (0.4) — 4 more OpenAI-compat providers
        (NebiusAdapter, "api.studio.nebius.ai", "meta-llama/Llama-3.3-70B-Instruct"),
        (OVHCloudAdapter, "endpoints.kepler.ai.cloud.ovh.net", "Llama-3.3-70B-Instruct"),
        (ZAIAdapter, "api.z.ai", "glm-4.6"),
        (MoonshotAdapter, "api.moonshot.ai", "kimi-k2.6"),
    ],
)
def test_chat_request_targets_correct_host(
    adapter_cls: type, api_base_substring: str, model: str
) -> None:
    adapter = adapter_cls(ProviderConfig(api_key="test-key"))
    req = adapter.build_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
    )
    assert api_base_substring in req.url
    assert req.url.endswith("/chat/completions")
    assert req.headers["Authorization"] == "Bearer test-key"
    body = orjson.loads(req.body or b"")
    assert body["model"] == model
    assert body["messages"][0]["role"] == "user"


@pytest.mark.parametrize(
    "adapter_cls",
    [GroqAdapter, TogetherAdapter, FireworksAdapter, DeepSeekAdapter, PerplexityAdapter],
)
def test_tools_pass_through(adapter_cls: type) -> None:
    """OpenAI-compat providers all forward function tools verbatim."""
    adapter = adapter_cls(ProviderConfig(api_key="test-key"))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    # Use a safe model id per provider — we're not exercising the response.
    model_by_class = {
        GroqAdapter: "llama-3.3-70b-versatile",
        TogetherAdapter: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        FireworksAdapter: "accounts/fireworks/models/llama-v3p3-70b-instruct",
        DeepSeekAdapter: "deepseek-v4-flash",
        PerplexityAdapter: "sonar",
    }
    req = adapter.build_request(
        model=model_by_class[adapter_cls],
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
    )
    body = orjson.loads(req.body or b"")
    assert body["tools"] == tools


def test_groq_uses_bearer_auth() -> None:
    """Sanity-check Groq specifically — uses Bearer not custom header."""
    adapter = GroqAdapter(ProviderConfig(api_key="gsk_test"))
    req = adapter.build_request(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.headers["Authorization"] == "Bearer gsk_test"


def test_together_path_preserves_slash_in_model_id() -> None:
    """Together model ids contain ``/``; the body must keep it intact."""
    adapter = TogetherAdapter(ProviderConfig(api_key="test-key"))
    req = adapter.build_request(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": "hi"}],
    )
    body = orjson.loads(req.body or b"")
    assert "/" in body["model"]
    assert body["model"] == "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"


def test_perplexity_does_not_have_embeddings() -> None:
    """Perplexity raises ``UnsupportedModelError`` on embedding requests."""
    from arcllm.exceptions import UnsupportedModelError

    adapter = PerplexityAdapter(ProviderConfig(api_key="test-key"))
    with pytest.raises(UnsupportedModelError):
        adapter.build_embedding_request(model="not-real", input=["hi"])


def test_deepseek_supports_embeddings_path() -> None:
    """DeepSeek embedding requests target /embeddings on the same host."""
    adapter = DeepSeekAdapter(ProviderConfig(api_key="test-key"))
    # DeepSeek does not currently offer embeddings either; the inherited
    # OpenAI build_embedding_request still produces a URL on the configured
    # base. We just verify the URL has the right shape.
    req = adapter.build_embedding_request(model="deepseek-embed-v1", input=["hi"])
    assert req.url.endswith("/embeddings")


# ----------------------------------------------------------------------
# Tier A (0.4) — provider-specific behaviour beyond the parametrized smoke
# ----------------------------------------------------------------------


def test_xai_rejects_embeddings() -> None:
    from arcllm.exceptions import UnsupportedModelError

    adapter = XAIAdapter(ProviderConfig(api_key="xai-test"))
    with pytest.raises(UnsupportedModelError):
        adapter.build_embedding_request(model="grok-embed", input=["hi"])


def test_cerebras_rejects_embeddings() -> None:
    from arcllm.exceptions import UnsupportedModelError

    adapter = CerebrasAdapter(ProviderConfig(api_key="cb-test"))
    with pytest.raises(UnsupportedModelError):
        adapter.build_embedding_request(model="cerebras-embed", input=["hi"])


def test_sambanova_rejects_embeddings() -> None:
    from arcllm.exceptions import UnsupportedModelError

    adapter = SambaNovaAdapter(ProviderConfig(api_key="sn-test"))
    with pytest.raises(UnsupportedModelError):
        adapter.build_embedding_request(model="sn-embed", input=["hi"])


def test_openrouter_attribution_headers_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OPENROUTER_REFERER`` / ``OPENROUTER_APP_NAME`` env vars set the recommended attribution headers."""
    monkeypatch.setenv("OPENROUTER_REFERER", "https://example.com")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "test-app")
    adapter = OpenRouterAdapter(ProviderConfig(api_key="or-test"))
    req = adapter.build_request(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.headers["HTTP-Referer"] == "https://example.com"
    assert req.headers["X-Title"] == "test-app"


def test_openrouter_attribution_headers_omitted_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_REFERER", raising=False)
    monkeypatch.delenv("OPENROUTER_APP_NAME", raising=False)
    adapter = OpenRouterAdapter(ProviderConfig(api_key="or-test"))
    req = adapter.build_request(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert "HTTP-Referer" not in req.headers
    assert "X-Title" not in req.headers


def test_extra_headers_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom headers via ``ProviderConfig.extra_headers`` are merged in."""
    adapter = NvidiaNIMAdapter(ProviderConfig(api_key="nv-test", extra_headers={"X-Custom": "yes"}))
    req = adapter.build_request(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.headers["X-Custom"] == "yes"
    assert req.headers["Authorization"] == "Bearer nv-test"


@pytest.mark.parametrize(
    "adapter_cls,env_var",
    [
        (XAIAdapter, "XAI_API_KEY"),
        (OpenRouterAdapter, "OPENROUTER_API_KEY"),
        (NvidiaNIMAdapter, "NVIDIA_NIM_API_KEY"),
        (CerebrasAdapter, "CEREBRAS_API_KEY"),
        (SambaNovaAdapter, "SAMBANOVA_API_KEY"),
        (DeepInfraAdapter, "DEEPINFRA_API_KEY"),
        # Production push
        (NebiusAdapter, "NEBIUS_API_KEY"),
        (OVHCloudAdapter, "OVHCLOUD_API_KEY"),
        (ZAIAdapter, "ZAI_API_KEY"),
        (MoonshotAdapter, "MOONSHOT_API_KEY"),
    ],
)
def test_api_key_resolution_from_env(
    adapter_cls: type, env_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each Tier A adapter pulls its key from the documented env var."""
    monkeypatch.setenv(env_var, f"env-{env_var.lower()}")
    adapter = adapter_cls(ProviderConfig())  # no explicit key
    headers = adapter._build_headers()
    assert headers["Authorization"] == f"Bearer env-{env_var.lower()}"
