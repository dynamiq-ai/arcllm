"""litellm-compat import-surface tests.

These tests pin the public symbols, exception classes, module-level
attributes, and submodule paths that real-world litellm consumers import.
Together they protect the arcllm drop-in claim from the README.

Failures here mean the drop-in story regressed — each test corresponds
to a real call site in production code that imports from ``litellm.*``.
Do not delete tests without checking the downstream impact first.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from arcllm.http.client import HTTPResponse

# ---------------------------------------------------------------------------
# Type aliases (arcllm.types)
# ---------------------------------------------------------------------------


def test_choices_alias_is_choice():
    """Pins ``Choices`` (plural). litellm reuses one class for both
    streaming and non-streaming choice items; arcllm names it ``Choice``
    (singular). The alias keeps ``from litellm import Choices`` resolving
    after an import-path swap."""
    from arcllm import Choices
    from arcllm.types import Choice

    assert Choices is Choice


def test_chat_completion_delta_tool_call_alias_is_dict():
    """Pins ``litellm.types.utils.ChatCompletionDeltaToolCall``.

    Used by callers as a type annotation only. arcllm emits streaming
    tool-call deltas as plain dicts (see ``ChunkDelta.tool_calls``), so
    the alias resolves to ``dict`` — sufficient for annotation
    compatibility with ``List[ChatCompletionDeltaToolCall]`` hints.
    """
    from arcllm.types import ChatCompletionDeltaToolCall

    assert ChatCompletionDeltaToolCall is dict


# ---------------------------------------------------------------------------
# Exceptions (arcllm.exceptions)
# ---------------------------------------------------------------------------


def test_context_window_exceeded_error_is_bad_request_subclass():
    """Pins ``ContextWindowExceededError`` as a ``BadRequestError`` subclass.
    Callers ``except`` it explicitly to surface their own context-length
    error to the user (vs. retrying or aborting the whole request)."""
    from arcllm import ContextWindowExceededError
    from arcllm.exceptions import BadRequestError

    assert issubclass(ContextWindowExceededError, BadRequestError)


def test_context_window_exceeded_error_constructible():
    """The class must be raise-able with the standard litellm shape."""
    from arcllm import ContextWindowExceededError

    exc = ContextWindowExceededError("prompt is too long")
    assert "too long" in str(exc)


# ---------------------------------------------------------------------------
# Capability helpers (arcllm.capabilities)
# ---------------------------------------------------------------------------


def test_supports_response_schema_matches_supports_structured_output():
    """Pins ``supports_response_schema``. litellm exposes this as the
    "does this model accept ``response_format``?" capability helper; arcllm
    names it ``supports_structured_output`` and exposes a litellm-named
    alias."""
    from arcllm import supports_response_schema, supports_structured_output

    # Identity not required — equivalence is. Probe a couple of well-known
    # models so we're testing real behavior, not the alias trivially returning
    # the same constant.
    for model in ("openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-latest"):
        assert supports_response_schema(model) == supports_structured_output(model)


# ---------------------------------------------------------------------------
# Custom logger stub (arcllm.integrations.custom_logger)
# ---------------------------------------------------------------------------


def test_custom_logger_importable_and_subclassable():
    """Pins ``litellm.integrations.custom_logger.CustomLogger`` as a
    subclassable base. arcllm exposes a no-op stub at the same path so
    callers subclassing it (typically as observability hooks) keep
    loading after an import-path swap.
    """
    from arcllm.integrations.custom_logger import CustomLogger

    class _Handler(CustomLogger):
        def __init__(self):
            super().__init__()
            self.called = False

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.called = True

    handler = _Handler()
    # Sub-class override is honored.
    handler.log_success_event({}, None, 0.0, 0.0)
    assert handler.called is True


def test_custom_logger_base_hooks_are_no_op():
    """Base hooks must accept the canonical signature and return None silently
    — arcllm never invokes them, but downstream code may chain super() calls.
    """
    from arcllm.integrations.custom_logger import CustomLogger

    logger = CustomLogger()
    # Each of these must be callable with the litellm-shape signature without
    # raising. Return value is unused.
    logger.log_pre_api_call("model", [], {})
    logger.log_post_api_call({}, None, 0.0, 0.0)
    logger.log_success_event({}, None, 0.0, 0.0)
    logger.log_failure_event({}, None, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Module-level passive attributes (litellm-compat surface on `arcllm`)
# ---------------------------------------------------------------------------


def test_module_level_callback_lists_present_and_mutable():
    """Pins ``success_callback`` / ``_async_success_callback`` /
    ``failure_callback`` / ``callbacks`` as mutable module-level lists.
    litellm-compat callers mutate these (``.append(...)``, direct
    assignment) at import time; arcllm exposes them as passive lists so
    those mutations don't AttributeError. arcllm itself never invokes
    them — the SDK has no callback subsystem."""
    import arcllm

    for name in ("success_callback", "_async_success_callback", "failure_callback", "callbacks"):
        attr = getattr(arcllm, name)
        assert isinstance(attr, list), f"{name} should be a list, got {type(attr)!r}"
        # Mutation must round-trip.
        attr.append("sentinel")
        assert getattr(arcllm, name)[-1] == "sentinel"
        attr.remove("sentinel")


def test_module_level_drop_params_default_false():
    import arcllm

    # Snapshot then restore to keep test hermetic.
    prior = arcllm.drop_params
    try:
        # The module ships with the flag off.
        assert prior is False
    finally:
        arcllm.drop_params = prior


# ---------------------------------------------------------------------------
# Behavioral compat: `arcllm.drop_params = True` must affect completion()
# ---------------------------------------------------------------------------


@pytest.fixture
def _mock_openai_response():
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _patched_http(mock_response: dict):
    """Helper: patch the sync HTTP client to return ``mock_response``."""
    body = json.dumps(mock_response).encode("utf-8")
    response = HTTPResponse(status_code=200, headers={}, body=body)
    cm = patch("arcllm.core._get_http_client")
    return cm, response


def test_module_drop_params_false_raises_on_unknown_param(_mock_openai_response):
    """Baseline: with ``arcllm.drop_params=False`` and no per-call override,
    an unsupported param must still raise. This is the existing behavior —
    the test exists so we notice if it ever regresses."""
    import arcllm
    from arcllm import completion
    from arcllm.exceptions import UnsupportedParameterError

    cm, http_response = _patched_http(_mock_openai_response)
    prior = arcllm.drop_params
    try:
        arcllm.drop_params = False
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), cm as mock_client:
            mock_client.return_value.request.return_value = http_response
            with pytest.raises(UnsupportedParameterError):
                completion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    not_a_real_param=42,
                )
    finally:
        arcllm.drop_params = prior


def test_module_drop_params_true_tolerates_unknown_param(_mock_openai_response):
    """Setting ``arcllm.drop_params = True`` at module level must propagate
    to ``completion()`` so callers don't have to thread the flag per call.
    This mirrors litellm's ergonomics: setting ``litellm.drop_params =
    True`` once at import time turns on the silent-drop behavior for every
    subsequent call.
    """
    import arcllm
    from arcllm import completion

    cm, http_response = _patched_http(_mock_openai_response)
    prior = arcllm.drop_params
    try:
        arcllm.drop_params = True
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), cm as mock_client:
            mock_client.return_value.request.return_value = http_response
            response = completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                not_a_real_param=42,
            )
        assert response.choices[0].message.content == "ok"
    finally:
        arcllm.drop_params = prior


def test_per_call_drop_params_overrides_module_default(_mock_openai_response):
    """Per-call ``drop_params=False`` must beat module-level ``True``."""
    import arcllm
    from arcllm import completion
    from arcllm.exceptions import UnsupportedParameterError

    cm, http_response = _patched_http(_mock_openai_response)
    prior = arcllm.drop_params
    try:
        arcllm.drop_params = True
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), cm as mock_client:
            mock_client.return_value.request.return_value = http_response
            with pytest.raises(UnsupportedParameterError):
                completion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    not_a_real_param=42,
                    drop_params=False,
                )
    finally:
        arcllm.drop_params = prior


# ---------------------------------------------------------------------------
# Typed request-message factories + ``ModelResponseStream`` +
# ``StreamingChoices`` + ``OpenAIMessageContent`` + ``arcllm.types.utils``
# path + ``add_function_to_prompt`` passive attr + ``acreate_file`` stub.
# ---------------------------------------------------------------------------


# Each name below is a litellm symbol that callers import to construct
# OpenAI-shape request messages — ``ChatCompletionAssistantMessage(
# role="assistant", content="hi")`` returns a plain ``dict`` that
# ``arcllm.acompletion`` consumes natively via its ``list[dict]`` messages
# interface.
_LITELLM_TYPED_MESSAGE_CLASSES = (
    "ChatCompletionAssistantMessage",
    "ChatCompletionAssistantToolCall",
    "ChatCompletionMessageToolCall",
    "ChatCompletionSystemMessage",
    "ChatCompletionToolMessage",
    "ChatCompletionUserMessage",
    "Function",
)


def test_adk_message_classes_resolve_to_dict_factory():
    """Pins each typed message / tool-call class as a dict-subclass factory
    with attribute access. litellm's Pydantic-ish models support both
    ``obj.field`` and ``obj["field"]``; the alias must preserve that dual
    interface."""
    import arcllm

    for name in _LITELLM_TYPED_MESSAGE_CLASSES:
        cls = getattr(arcllm, name)
        # Must be a dict subclass — guarantees JSON serialization,
        # ``**unpacking``, and ``dict(instance)`` all work.
        assert issubclass(cls, dict), (
            f"{name} should alias to a dict subclass, got {cls!r}"
        )
        # Must support both styles of access.
        instance = cls(probe="value")
        assert instance["probe"] == "value"
        assert instance.probe == "value"


def test_adk_message_factory_constructs_openai_shape_dict():
    """End-to-end factory check: kwargs → dict → OpenAI request shape."""
    from arcllm import (
        ChatCompletionAssistantMessage,
        ChatCompletionMessageToolCall,
        ChatCompletionUserMessage,
        Function,
    )

    user_msg = ChatCompletionUserMessage(role="user", content="hi")
    assert user_msg == {"role": "user", "content": "hi"}

    fn = Function(name="lookup", arguments='{"q":"x"}')
    tool_call = ChatCompletionMessageToolCall(type="function", id="call_1", function=fn)
    assert tool_call == {
        "type": "function",
        "id": "call_1",
        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
    }

    asst = ChatCompletionAssistantMessage(
        role="assistant", content=None, tool_calls=[tool_call]
    )
    assert asst["tool_calls"][0]["function"]["name"] == "lookup"


def test_openai_message_content_alias_is_list():
    """``OpenAIMessageContent`` is a type-annotation alias. Aliasing to
    ``list`` keeps it honest — OpenAI multimodal content is a list of
    content blocks."""
    from arcllm import OpenAIMessageContent

    assert OpenAIMessageContent is list


def test_streaming_choices_alias_is_chunk_choice():
    """Pins ``litellm.types.utils.StreamingChoices``. Callers (typically in
    test fixtures) construct mock streaming responses with this name;
    arcllm names the equivalent type ``ChunkChoice``."""
    from arcllm.types import ChunkChoice, StreamingChoices

    assert StreamingChoices is ChunkChoice


def test_model_response_stream_alias_is_stream_chunk():
    """Pins ``ModelResponseStream``. litellm exposes the streaming chunk
    type under this name at both ``litellm`` top level and
    ``litellm.types.utils``; arcllm names the equivalent type
    ``StreamChunk``."""
    from arcllm.types import ModelResponseStream, StreamChunk

    assert ModelResponseStream is StreamChunk


def test_arcllm_types_utils_submodule_resolves_to_arcllm_types():
    """Pins the ``arcllm.types.utils`` submodule path. Callers that
    ``from litellm.types.utils import …`` should resolve through the
    equivalent ``arcllm.types.utils`` after an import swap; arcllm
    exposes the same surface via a one-line sys.modules registration."""
    import arcllm  # noqa: F401  (triggers the registration)
    import arcllm.types as canonical
    import arcllm.types.utils as alias
    from arcllm.types.utils import (  # noqa: F401
        ChatCompletionDeltaToolCall,
        Choices,
        Delta,
        ModelResponse,
        ModelResponseStream,
        StreamingChoices,
    )

    assert alias is canonical


def test_module_add_function_to_prompt_default_false():
    """Pins ``add_function_to_prompt`` as a passive module attribute.
    litellm-compat callers set ``litellm.add_function_to_prompt = True``
    at import time; arcllm exposes the attribute so that assignment
    doesn't AttributeError. arcllm does NOT implement the underlying
    behavior (auto-injecting function defs into the system prompt for
    providers without native tool support). Setting it is a no-op;
    documented gap.
    """
    import arcllm

    prior = arcllm.add_function_to_prompt
    try:
        assert prior is False
        arcllm.add_function_to_prompt = True
        assert arcllm.add_function_to_prompt is True
    finally:
        arcllm.add_function_to_prompt = prior


async def test_acreate_file_stub_raises_notimplemented():
    """Pins ``acreate_file`` as a NotImplementedError stub. litellm-compat
    callers invoke ``litellm.acreate_file(...)`` to upload PDF/docx to a
    provider's Files API for multimodal flows routed through OpenAI or
    Azure. arcllm does not implement file upload — the stub exists so
    the failure mode is obvious and local. Text-only and inline-data
    multimodal paths are unaffected.
    """
    import arcllm

    with pytest.raises(NotImplementedError, match="acreate_file"):
        await arcllm.acreate_file(file=b"x", purpose="assistants", custom_llm_provider="openai")


def test_adk_style_message_build_round_trips_through_arcllm(_mock_openai_response):
    """End-to-end: build a full multi-turn conversation (system / user /
    assistant + tool_calls / tool / user) using the typed-class aliases
    and push it through a mocked HTTP layer. The captured request body
    must contain valid OpenAI-shape JSON for every message — proves the
    aliases serialize through arcllm's adapter unchanged.
    """
    from arcllm import (
        ChatCompletionAssistantMessage,
        ChatCompletionMessageToolCall,
        ChatCompletionSystemMessage,
        ChatCompletionToolMessage,
        ChatCompletionUserMessage,
        Function,
        completion,
    )

    body_bytes = json.dumps(_mock_openai_response).encode("utf-8")
    http_response = HTTPResponse(status_code=200, headers={}, body=body_bytes)

    captured = {}

    class _FakeClient:
        def request(self, method, url, *, headers, body, timeout, stream):
            captured["body"] = json.loads(body)
            return http_response

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        patch("arcllm.core._get_http_client", return_value=_FakeClient()),
    ):
        completion(
            model="gpt-4o-mini",
            messages=[
                ChatCompletionSystemMessage(role="system", content="be brief"),
                ChatCompletionUserMessage(role="user", content="what's 1+1?"),
                ChatCompletionAssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            type="function",
                            id="call_42",
                            function=Function(name="calculator", arguments='{"a":1,"b":1}'),
                        )
                    ],
                ),
                ChatCompletionToolMessage(role="tool", tool_call_id="call_42", content="2"),
                ChatCompletionUserMessage(role="user", content="thanks"),
            ],
        )

    sent = captured["body"]["messages"]
    assert len(sent) == 5
    assert sent[0]["role"] == "system"
    assert sent[2]["tool_calls"][0]["function"]["name"] == "calculator"
    assert sent[3]["role"] == "tool"
    assert sent[3]["tool_call_id"] == "call_42"


# ---------------------------------------------------------------------------
# ``arcllm.utils`` path alias + ``Router`` stub class.
# ---------------------------------------------------------------------------


def test_arcllm_utils_alias_exposes_usage():
    """Pins ``Usage`` reachable via the ``arcllm.utils`` path. litellm-compat
    callers import ``from litellm.utils import Usage`` (and friends);
    after the litellm-shim swap this becomes ``from arcllm.utils import
    Usage``. arcllm keeps ``Usage`` defined in ``arcllm.types``;
    registering ``arcllm.utils`` as a sys.modules alias makes the import
    path resolve without duplicating definitions.
    """
    import arcllm  # noqa: F401  (triggers types module + sys.modules setup)
    from arcllm.types import Usage as CanonicalUsage
    from arcllm.utils import Usage

    assert Usage is CanonicalUsage


def test_arcllm_utils_alias_module_identity():
    """``arcllm.utils`` resolves to the same module object as ``arcllm.types``
    — no duplicate class hierarchies, no drift risk."""
    import arcllm  # noqa: F401
    import arcllm.types as canonical
    import arcllm.utils as alias

    assert alias is canonical


def test_router_stub_class_importable():
    """Pins ``litellm.Router`` as an importable name. litellm-compat
    callers accept a Router instance as a constructor argument or
    reference it for type annotations; arcllm does not yet implement
    load-balanced routing, so the stub class exists to keep the import
    path resolving. Any attempted construction fails locally with a
    clear NotImplementedError rather than an AttributeError far from
    the root cause.
    """
    from arcllm import Router

    assert isinstance(Router, type)


def test_router_stub_raises_on_construction():
    """Constructing the stub must raise NotImplementedError with a message
    that names the class so users know where to look."""
    from arcllm import Router

    with pytest.raises(NotImplementedError, match=r"Router"):
        Router(
            model_list=[
                {"model_name": "foo", "litellm_params": {"model": "openai/gpt-4o-mini"}}
            ]
        )
