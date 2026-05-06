"""Per-family dispatch tests for the Azure adapter.

Azure exposes two surfaces:

- Azure OpenAI Service: ``{resource}.openai.azure.com/openai/deployments/{deployment}/...``
- Azure AI Foundry serverless: ``{endpoint}.services.ai.azure.com/models/...``

Dispatch is **host-first** (parses the configured ``api_base``), with a
model-name fallback for unfamiliar hosts.
"""

from __future__ import annotations

import orjson
import pytest

from arcllm.providers.azure_adapter import (
    AzureOpenAIAdapter,
    _is_azure_openai_model,
    _is_foundry_host,
)
from arcllm.providers.base import ProviderConfig


def _make_adapter() -> AzureOpenAIAdapter:
    return AzureOpenAIAdapter(
        ProviderConfig(
            api_key="azure-key",
            api_base="https://my-resource.openai.azure.com",
            azure_deployment="my-deployment",
        )
    )


class TestAzureFamilyDetection:
    @pytest.mark.parametrize(
        "model,is_openai",
        [
            ("gpt-4o", True),
            ("gpt-4o-mini", True),
            ("gpt-5-mini", True),
            ("o1", True),
            ("o3-mini", True),
            ("o4-mini", True),
            ("text-embedding-3-small", True),
            ("chatgpt-4o-latest", True),
            ("Phi-4", False),
            ("Llama-3.3-70B-Instruct", False),
            ("Mistral-large-2411", False),
            ("Cohere-command-r-plus", False),
        ],
    )
    def test_dispatch(self, model: str, is_openai: bool) -> None:
        assert _is_azure_openai_model(model) is is_openai


class TestAzureOpenAIPath:
    def test_chat_url_uses_deployment(self) -> None:
        adapter = _make_adapter()
        req = adapter.build_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "/openai/deployments/my-deployment/chat/completions" in req.url
        assert "api-version=" in req.url
        # OpenAI-on-Azure does NOT need the model in the body — the deployment determines it.
        body = orjson.loads(req.body or b"")
        assert "model" not in body
        assert body["messages"][0]["role"] == "user"

    def test_embedding_url_uses_deployment(self) -> None:
        adapter = _make_adapter()
        req = adapter.build_embedding_request(
            model="text-embedding-3-small",
            input=["hi"],
        )
        assert "/openai/deployments/my-deployment/embeddings" in req.url


def _make_foundry_adapter() -> AzureOpenAIAdapter:
    """Adapter pointed at an Azure AI Foundry serverless endpoint."""
    return AzureOpenAIAdapter(
        ProviderConfig(
            api_key="azure-key",
            api_base="https://my-foundry.services.ai.azure.com",
        )
    )


class TestAzureFoundryPath:
    def test_chat_url_uses_models_endpoint(self) -> None:
        """Foundry host → /models/chat/completions, model in body."""
        adapter = _make_foundry_adapter()
        req = adapter.build_request(
            model="Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "/models/chat/completions" in req.url
        assert "/openai/deployments/" not in req.url
        body = orjson.loads(req.body or b"")
        assert body["model"] == "Llama-3.3-70B-Instruct"

    def test_embedding_url_uses_models_endpoint(self) -> None:
        adapter = _make_foundry_adapter()
        req = adapter.build_embedding_request(
            model="Cohere-embed-v3-multilingual",
            input=["hi"],
        )
        assert "/models/embeddings" in req.url
        body = orjson.loads(req.body or b"")
        assert body["model"] == "Cohere-embed-v3-multilingual"


class TestAzureHostBasedDispatch:
    """Regression: host detection must override model-name sniffing.

    Previously a custom-named model (e.g. a fine-tune called
    ``my-llama``) on an ``*.openai.azure.com`` host was routed to the
    Foundry ``/models/chat/completions`` URL because it didn't match the
    OpenAI prefix list. That misroute hits the wrong endpoint.
    """

    def test_custom_model_on_openai_service_routes_to_deployment(self) -> None:
        adapter = _make_adapter()  # host: my-resource.openai.azure.com
        req = adapter.build_request(
            model="my-fine-tuned-llama",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "/openai/deployments/my-deployment/chat/completions" in req.url
        assert "/models/chat/completions" not in req.url

    def test_foundry_host_overrides_openai_named_model(self) -> None:
        """Even a model called 'gpt-foo' on a Foundry host goes to /models."""
        adapter = _make_foundry_adapter()
        req = adapter.build_request(
            model="gpt-foo-deployed-on-foundry",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "/models/chat/completions" in req.url

    def test_is_foundry_host_helper(self) -> None:
        assert _is_foundry_host("https://x.services.ai.azure.com")
        assert _is_foundry_host("https://x.services.ai.azure.com/models")
        assert not _is_foundry_host("https://x.openai.azure.com")
        assert not _is_foundry_host("https://localhost:1234")
        assert not _is_foundry_host(None)
