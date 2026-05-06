"""
Error handling example for arcllm.

arcllm raises a focused hierarchy of exceptions that you can catch
selectively. The mapping from provider HTTP status codes is:

- 401 → ``AuthenticationError``
- 402 / 429+quota-words → ``BudgetExceededError``
- 408 → ``TimeoutError``
- 429 → ``RateLimitError`` (with ``retry_after`` if the provider sent it)
- 503 → ``ServiceUnavailableError``
- 5xx (other) → ``InternalServerError``
- 400 → ``InvalidRequestError``
- 404 (model not found) → ``UnsupportedModelError``

All inherit from ``ArcLLMError``.

For callers migrating from litellm, two compatibility aliases live in
``arcllm.exceptions``: ``Timeout = TimeoutError`` and
``APIConnectionError = ConnectionError``.

Run::

    export OPENAI_API_KEY="sk-..."
    python examples/error_handling.py
"""

from __future__ import annotations

import arcllm
from arcllm.exceptions import (
    APIConnectionError,
    ArcLLMError,
    AuthenticationError,
    BudgetExceededError,
    InternalServerError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,  # alias of TimeoutError
    UnsupportedModelError,
)


def safe_completion() -> None:
    """Catch each error class with the right granularity."""
    try:
        response = arcllm.completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=8,
            timeout=30.0,
        )
        print(response.choices[0].message.content)
    except AuthenticationError as e:
        # Wrong/missing key. Don't retry.
        print(f"Auth failed: {e.message}")
    except BudgetExceededError as e:
        # Quota / billing exhausted. Don't retry on the same key.
        print(f"Budget exceeded: {e.message}")
    except RateLimitError as e:
        # Recoverable — back off and retry.
        retry_after = e.retry_after or 1.0
        print(f"Rate limited. Retry after {retry_after}s")
    except (Timeout, APIConnectionError) as e:
        # Network blip — retry with backoff.
        print(f"Network: {type(e).__name__}: {e.message}")
    except (ServiceUnavailableError, InternalServerError) as e:
        # 5xx from provider — retry with backoff, but cap attempts.
        print(f"Provider down: {type(e).__name__}: {e.message}")
    except UnsupportedModelError as e:
        # Wrong model id — fix the call site, don't retry.
        print(f"Unknown model: {e.message}")
    except InvalidRequestError as e:
        # Malformed request — fix the call site, don't retry.
        print(f"Bad request: {e.message}")
    except ArcLLMError as e:
        # Last-resort net for anything new in the hierarchy.
        print(f"arcllm error ({type(e).__name__}): {e.message}")


def demonstrate_unsupported_model() -> None:
    """Trigger ``UnsupportedModelError`` deterministically."""
    try:
        arcllm.completion(
            model="openai/this-model-definitely-does-not-exist",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4,
        )
    except UnsupportedModelError as e:
        print(f"Got expected UnsupportedModelError: {e.message}")


if __name__ == "__main__":
    safe_completion()
    demonstrate_unsupported_model()
