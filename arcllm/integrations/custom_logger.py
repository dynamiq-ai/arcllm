"""litellm-compat ``CustomLogger`` stub.

Mirrors the public hook surface of
``litellm.integrations.custom_logger.CustomLogger`` so downstream
subclasses load without modification. arcllm does NOT invoke these
hooks — they are defined only so ``class X(CustomLogger): ...`` keeps
working after a ``litellm → arcllm`` import-path swap.

Migration note: callers that want per-call observability should read
``response.usage`` and ``response.choices[0].message`` directly from
the return value of :func:`arcllm.completion`. The SDK does not
maintain a callback registry.
"""

from __future__ import annotations

from typing import Any


class CustomLogger:
    """No-op base class. arcllm never invokes these methods.

    The signatures match litellm so existing subclasses that override one or
    more hooks keep type-checking and runtime behavior unchanged.
    """

    def log_pre_api_call(self, model: str, messages: list[Any], kwargs: dict[str, Any]) -> None:
        """Hook fired by litellm *before* the API call. arcllm never calls it."""

    def log_post_api_call(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Hook fired by litellm *after* the API call. arcllm never calls it."""

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Hook fired by litellm on successful completion. arcllm never calls it."""

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Hook fired by litellm on failure. arcllm never calls it."""

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Async variant of :meth:`log_success_event`. arcllm never calls it."""

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Async variant of :meth:`log_failure_event`. arcllm never calls it."""
