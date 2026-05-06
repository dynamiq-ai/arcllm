"""
Exception classes for arcllm.

All exceptions inherit from ArcLLMError for easy catching.
Provider-specific errors are mapped to these stable exception types.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "APIConnectionError",
    "APIError",
    "ArcLLMError",
    "AuthenticationError",
    "BadRequestError",
    "BudgetExceededError",
    "ConnectionError",
    "ContentFilterError",
    "InternalServerError",
    "InvalidRequestError",
    "ProviderAPIError",
    "RateLimitError",
    "ResponseParseError",
    "ServiceUnavailableError",
    "Timeout",
    "TimeoutError",
    "UnsupportedModelError",
    "UnsupportedParameterError",
]


class ArcLLMError(Exception):
    """Base exception for all arcllm errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        raw_response: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.request_id = request_id
        self.raw_response = raw_response

    def __str__(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.status_code:
            parts.append(f"status_code={self.status_code}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"status_code={self.status_code!r}, "
            f"request_id={self.request_id!r})"
        )


class AuthenticationError(ArcLLMError):
    """
    Raised when authentication fails.

    Common causes:
    - Invalid API key
    - Expired API key
    - Missing API key
    - Insufficient permissions
    """

    pass


class RateLimitError(ArcLLMError):
    """
    Raised when rate limits are exceeded.

    Check the retry_after attribute for suggested wait time if available.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class TimeoutError(ArcLLMError):
    """
    Raised when a request times out.

    This can be either a connection timeout or a read timeout.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_type: str | None = None,  # "connect" or "read"
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.timeout_type = timeout_type
        self.timeout_seconds = timeout_seconds


class ConnectionError(ArcLLMError):
    """
    Raised when connection to provider fails.

    Common causes:
    - Network issues
    - DNS resolution failure
    - SSL/TLS errors
    - Provider endpoint unreachable
    """

    pass


class ProviderAPIError(ArcLLMError):
    """
    Raised when the provider returns an error response.

    This is used for provider-specific errors that don't map to
    other more specific exception types.
    """

    def __init__(
        self,
        message: str,
        *,
        error_type: str | None = None,
        error_code: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.error_type = error_type
        self.error_code = error_code


class UnsupportedModelError(ArcLLMError):
    """
    Raised when the requested model is not supported.

    Common causes:
    - Model name is invalid
    - Model is not available for the provider
    - Model requires different API endpoint
    """

    pass


class UnsupportedParameterError(ArcLLMError):
    """
    Raised when unsupported parameters are passed (and drop_params=False).

    The unsupported_params attribute contains the list of parameter names
    that are not supported by the provider/model.
    """

    def __init__(
        self,
        message: str,
        *,
        unsupported_params: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.unsupported_params = unsupported_params or []


class ResponseParseError(ArcLLMError):
    """
    Raised when the provider response cannot be parsed.

    Common causes:
    - Invalid JSON response
    - Unexpected response structure
    - Incomplete streaming response
    """

    def __init__(
        self,
        message: str,
        *,
        raw_data: str | bytes | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.raw_data = raw_data


class ContentFilterError(ArcLLMError):
    """
    Raised when content is blocked by safety filters.

    This typically happens when:
    - Input violates content policy
    - Output was filtered due to safety concerns
    """

    def __init__(
        self,
        message: str,
        *,
        filter_reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.filter_reason = filter_reason


class InvalidRequestError(ArcLLMError):
    """
    Raised when the request is malformed or invalid.

    Common causes:
    - Missing required parameters
    - Invalid parameter values
    - Malformed message format
    """

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.param = param


class BudgetExceededError(ProviderAPIError):
    """Raised when the account has exhausted its quota or billing limit.

    Distinct from :class:`RateLimitError` (which is recoverable by retrying
    after a delay) — a budget exception requires human intervention. Maps
    most commonly to HTTP 402 (Payment Required) and to provider-specific
    quota-exceeded payloads on 429 / 403.
    """


class ServiceUnavailableError(ProviderAPIError):
    """Raised when the provider returns 503 (or equivalent transient outage).

    Often retryable but separate from rate limits — providers expose this
    when a region / model variant is temporarily down.
    """


class InternalServerError(ProviderAPIError):
    """Raised on 5xx responses other than 503.

    Indicates a provider-side bug or partial outage. Usually retryable.
    """


def map_status_code_to_exception(
    status_code: int,
    message: str,
    **kwargs: Any,
) -> ArcLLMError:
    """Map HTTP status code to the right :class:`ArcLLMError` subclass.

    Provider adapters call this from ``parse_error`` so the error surface
    stays consistent across providers. Distinguishes:

    - 401/403 → :class:`AuthenticationError`
    - 402 → :class:`BudgetExceededError` (some providers signal quota here)
    - 404 → :class:`UnsupportedModelError`
    - 408 → :class:`TimeoutError`
    - 429 → :class:`RateLimitError` (or :class:`BudgetExceededError` if the
      message obviously signals quota — providers like OpenAI, Anthropic,
      and Bedrock mix these).
    - 503 → :class:`ServiceUnavailableError`
    - other 5xx → :class:`InternalServerError`
    - 400 → :class:`InvalidRequestError`
    """
    if status_code == 401:
        return AuthenticationError(message, status_code=status_code, **kwargs)
    if status_code == 403:
        return AuthenticationError(
            f"Permission denied: {message}", status_code=status_code, **kwargs
        )
    if status_code == 402:
        return BudgetExceededError(message, status_code=status_code, **kwargs)
    if status_code == 429:
        # Some providers conflate quota exhaustion with rate limiting on 429.
        # If the message clearly signals quota/billing, route to the more
        # accurate exception; otherwise treat as a recoverable rate limit.
        lowered = message.lower()
        if any(token in lowered for token in ("quota", "billing", "credit", "budget")):
            return BudgetExceededError(message, status_code=status_code, **kwargs)
        return RateLimitError(message, status_code=status_code, **kwargs)
    if status_code == 404:
        return UnsupportedModelError(message, status_code=status_code, **kwargs)
    if status_code == 400:
        return InvalidRequestError(message, status_code=status_code, **kwargs)
    if status_code == 408:
        return TimeoutError(message, status_code=status_code, **kwargs)
    if status_code == 503:
        return ServiceUnavailableError(message, status_code=status_code, **kwargs)
    if status_code >= 500:
        return InternalServerError(f"Server error: {message}", status_code=status_code, **kwargs)
    return ProviderAPIError(message, status_code=status_code, **kwargs)


# Litellm-compatibility aliases. Kept for downstream callers (notably
# dynamiq) that import the legacy names from litellm.exceptions. These add
# zero runtime overhead — they are simply alternative bindings to the same
# class objects.
Timeout = TimeoutError
APIConnectionError = ConnectionError
# litellm names ``APIError`` and ``BadRequestError`` map to arcllm's
# ``ProviderAPIError`` (the broader provider-error base) and
# ``InvalidRequestError`` (400-class semantics) respectively.
APIError = ProviderAPIError
BadRequestError = InvalidRequestError
