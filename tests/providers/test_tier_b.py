"""Wire-shape tests for Tier B providers (0.4.0).

Covers:
- HuggingFace (OpenAI-shape on Hub Inference / Inference Endpoints)
- AI21 Studio (OpenAI-shape with custom auth env var)
- WatsonX (OpenAI-shape body rewritten to ``model_id`` + ``project_id``)
- AzureAI alias (re-points the existing Azure adapter under ``azure_ai``)
- CustomLLM (user-supplied base URL; OpenAI-shape pass-through)
"""

from __future__ import annotations

import orjson
import pytest

from arcllm.exceptions import ArcLLMError, InvalidRequestError
from arcllm.providers.ai21_adapter import AI21Adapter
from arcllm.providers.base import ProviderConfig, get_provider
from arcllm.providers.custom_adapter import CustomAdapter
from arcllm.providers.huggingface_adapter import HuggingFaceAdapter
from arcllm.providers.watsonx_adapter import WatsonXAdapter

# --- HuggingFace ------------------------------------------------------------


def test_huggingface_targets_router_v1_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_test_key")
    adapter = HuggingFaceAdapter(ProviderConfig())
    req = adapter.build_request(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert "router.huggingface.co" in req.url
    assert req.url.endswith("/v1/chat/completions")
    assert req.headers["Authorization"] == "Bearer hf_test_key"


def test_huggingface_dedicated_endpoint_via_api_base() -> None:
    """Inference-Endpoints deployments override base URL via api_base."""
    custom_url = "https://abc.us-east-1.aws.endpoints.huggingface.cloud/v1"
    adapter = HuggingFaceAdapter(ProviderConfig(api_key="hf_test", api_base=custom_url))
    req = adapter.build_request(
        model="custom-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.url == f"{custom_url}/chat/completions"


# --- AI21 -------------------------------------------------------------------


def test_ai21_studio_endpoint() -> None:
    adapter = AI21Adapter(ProviderConfig(api_key="ai21_test"))
    req = adapter.build_request(
        model="jamba-1.5-large",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert "api.ai21.com/studio/v1" in req.url
    assert req.url.endswith("/chat/completions")
    assert req.headers["Authorization"] == "Bearer ai21_test"
    body = orjson.loads(req.body or b"")
    assert body["model"] == "jamba-1.5-large"


def test_ai21_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI21_API_KEY", "env-ai21")
    adapter = AI21Adapter(ProviderConfig())
    headers = adapter._build_headers()
    assert headers["Authorization"] == "Bearer env-ai21"


# --- WatsonX ----------------------------------------------------------------


def test_watsonx_chat_request_renames_model_to_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project")
    adapter = WatsonXAdapter(
        ProviderConfig(api_key="eyJ.fake-iam.token", api_base="https://us-south.ml.cloud.ibm.com")
    )
    req = adapter.build_request(
        model="ibm/granite-13b-chat-v2",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        max_tokens=128,
    )
    assert "us-south.ml.cloud.ibm.com" in req.url
    # Non-streaming request hits /chat (not /chat_stream)
    assert "/ml/v1/text/chat?" in req.url
    assert "/chat_stream" not in req.url
    assert "version=2024-08-01" in req.url
    body = orjson.loads(req.body or b"")
    assert "model" not in body
    assert body["model_id"] == "ibm/granite-13b-chat-v2"
    assert body["project_id"] == "test-project"
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 128


def test_watsonx_streaming_routes_to_chat_stream_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """watsonx.ai exposes /chat_stream as a separate URL for SSE streaming.

    Sending stream=true to /chat returns a buffered response — the
    server does *not* honor stream=true on the non-stream path.
    """
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project")
    adapter = WatsonXAdapter(
        ProviderConfig(api_key="eyJ.fake-iam.token", api_base="https://us-south.ml.cloud.ibm.com")
    )
    req = adapter.build_request(
        model="ibm/granite-13b-chat-v2",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert "/ml/v1/text/chat_stream?" in req.url
    # Body still carries stream:true per OpenAI shape — both fields are
    # required by watsonx.
    body = orjson.loads(req.body or b"")
    assert body.get("stream") is True


def test_watsonx_explicit_kwarg_project_id_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATSONX_PROJECT_ID", "env-project")
    adapter = WatsonXAdapter(
        ProviderConfig(api_key="eyJ.fake-iam.token", api_base="https://us-south.ml.cloud.ibm.com")
    )
    req = adapter.build_request(
        model="ibm/granite-13b-chat-v2",
        messages=[{"role": "user", "content": "hi"}],
        project_id="explicit-project",
    )
    body = orjson.loads(req.body or b"")
    assert body["project_id"] == "explicit-project"


def test_watsonx_space_id_alternative(monkeypatch: pytest.MonkeyPatch) -> None:
    """``space_id`` may be supplied instead of ``project_id``."""
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    adapter = WatsonXAdapter(
        ProviderConfig(api_key="eyJ.fake-iam.token", api_base="https://us-south.ml.cloud.ibm.com")
    )
    req = adapter.build_request(
        model="ibm/granite-13b-chat-v2",
        messages=[{"role": "user", "content": "hi"}],
        space_id="space-xyz",
    )
    body = orjson.loads(req.body or b"")
    assert body["space_id"] == "space-xyz"
    assert "project_id" not in body


def test_watsonx_missing_project_or_space_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    monkeypatch.delenv("WATSONX_SPACE_ID", raising=False)
    adapter = WatsonXAdapter(
        ProviderConfig(api_key="eyJ.fake-iam.token", api_base="https://us-south.ml.cloud.ibm.com")
    )
    with pytest.raises(InvalidRequestError, match="project_id"):
        adapter.build_request(
            model="ibm/granite-13b-chat-v2",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_watsonx_api_version_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATSONX_PROJECT_ID", "p")
    monkeypatch.setenv("WATSONX_API_VERSION", "2025-01-01")
    adapter = WatsonXAdapter(
        ProviderConfig(api_key="eyJ.fake-iam.token", api_base="https://us-south.ml.cloud.ibm.com")
    )
    req = adapter.build_request(
        model="ibm/granite-13b-chat-v2",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert "version=2025-01-01" in req.url


def test_watsonx_jwt_token_bypasses_iam_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-exchanged IAM access token (JWT-shaped) is used verbatim."""
    monkeypatch.setenv("WATSONX_PROJECT_ID", "p")
    adapter = WatsonXAdapter(
        ProviderConfig(api_key="eyJ.real.iamtoken", api_base="https://us-south.ml.cloud.ibm.com")
    )

    def boom(*args: object, **kwargs: object) -> None:  # pragma: no cover
        raise AssertionError("IAM exchange should NOT be called for JWT tokens")

    monkeypatch.setattr(adapter, "_exchange_apikey_for_iam_token", boom)
    headers = adapter._build_headers()
    assert headers["Authorization"] == "Bearer eyJ.real.iamtoken"


def test_watsonx_raw_apikey_triggers_iam_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw IBM Cloud API keys are exchanged for an IAM token, then cached."""
    import time as time_mod

    monkeypatch.setenv("WATSONX_PROJECT_ID", "p")
    adapter = WatsonXAdapter(
        ProviderConfig(api_key="raw-apikey-abc", api_base="https://us-south.ml.cloud.ibm.com")
    )

    calls: list[str] = []

    def fake_exchange(apikey: str) -> tuple[str, float]:
        calls.append(apikey)
        return "eyJ.exchanged.token", time_mod.time() + 3600

    monkeypatch.setattr(adapter, "_exchange_apikey_for_iam_token", fake_exchange)
    h1 = adapter._build_headers()
    h2 = adapter._build_headers()
    assert h1["Authorization"] == "Bearer eyJ.exchanged.token"
    assert h2["Authorization"] == "Bearer eyJ.exchanged.token"
    # Token cache: only one exchange call even across multiple header builds.
    assert len(calls) == 1
    assert calls[0] == "raw-apikey-abc"


def test_watsonx_iam_token_refreshes_on_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the cached IAM token has expired, the adapter re-exchanges."""
    import time as time_mod

    monkeypatch.setenv("WATSONX_PROJECT_ID", "p")
    adapter = WatsonXAdapter(
        ProviderConfig(api_key="raw-apikey-x", api_base="https://us-south.ml.cloud.ibm.com")
    )

    counter = {"n": 0}

    def fake_exchange(apikey: str) -> tuple[str, float]:
        counter["n"] += 1
        # Token already past its refresh window so the next call re-exchanges.
        return f"eyJ.token{counter['n']}", time_mod.time() - 1

    monkeypatch.setattr(adapter, "_exchange_apikey_for_iam_token", fake_exchange)
    h1 = adapter._build_headers()
    h2 = adapter._build_headers()
    assert h1["Authorization"] == "Bearer eyJ.token1"
    assert h2["Authorization"] == "Bearer eyJ.token2"
    assert counter["n"] == 2


# --- AzureAI alias ----------------------------------------------------------


def test_azure_ai_alias_resolves_to_azure_adapter() -> None:
    """The ``azure_ai`` provider name routes to the same adapter as ``azure``."""
    cfg = ProviderConfig(
        api_key="tk",
        api_base="https://my.openai.azure.com",
        api_version="2024-08-01-preview",
    )
    a = get_provider("azure_ai", cfg)
    b = get_provider("azure", cfg)
    assert a.__class__ is b.__class__  # same AzureOpenAIAdapter class
    assert a.provider_name == "azure"


# --- CustomLLM --------------------------------------------------------------


def test_custom_requires_api_base() -> None:
    with pytest.raises(InvalidRequestError, match="api_base"):
        CustomAdapter(ProviderConfig(api_key="tk"))


def test_custom_uses_supplied_base_url_and_extra_headers() -> None:
    adapter = CustomAdapter(
        ProviderConfig(
            api_key="my-token",
            api_base="https://internal-llm.example.com/v1",
            extra_headers={"X-Tenant": "acme"},
        )
    )
    req = adapter.build_request(
        model="self-hosted-llama",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.url == "https://internal-llm.example.com/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer my-token"
    assert req.headers["X-Tenant"] == "acme"


def test_custom_works_without_api_key() -> None:
    """Self-hosted endpoints often have no auth — empty key is fine."""
    adapter = CustomAdapter(
        ProviderConfig(api_base="http://localhost:8080/v1"),
    )
    req = adapter.build_request(
        model="local",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert "Authorization" not in req.headers
    assert req.url.endswith("/chat/completions")


# --- Registry parity --------------------------------------------------------


@pytest.mark.parametrize(
    "name,cls_name",
    [
        ("huggingface", "HuggingFaceAdapter"),
        ("ai21", "AI21Adapter"),
        ("watsonx", "WatsonXAdapter"),
        ("azure_ai", "AzureOpenAIAdapter"),
        ("custom", "CustomAdapter"),
    ],
)
def test_provider_resolves_via_registry(name: str, cls_name: str) -> None:
    """All Tier B adapters are reachable through ``get_provider``."""
    # custom + watsonx + azure_ai need extra config; skip resolution-deep tests here.
    cfg = ProviderConfig(
        api_key="tk",
        api_base="https://example/v1" if name == "custom" else None,
        api_version="2024-08-01-preview" if name == "azure_ai" else None,
    )
    try:
        a = get_provider(name, cfg)
    except ArcLLMError:
        pytest.skip(f"adapter {name} requires more config")
    assert a.__class__.__name__ == cls_name
