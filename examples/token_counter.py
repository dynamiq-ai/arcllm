"""
Token counter example for arcllm.

Without extras, ``token_counter`` falls back to a chars/4 heuristic and
warns once per process. For exact counts on OpenAI-family models (and
DeepSeek / Perplexity / xAI which all share the cl100k_base encoder),
install with the ``tokenize`` extra::

    pip install "arcllm-sdk[tokenize]"

The same call shape works either way — the extra just upgrades the
backend from heuristic to tiktoken-precise.

Run::

    python examples/token_counter.py
"""

from __future__ import annotations

import warnings

import arcllm


def example_text_only() -> None:
    """Count tokens in a single string."""
    n = arcllm.token_counter(model="gpt-4o", text="The arc connecting you to every LLM.")
    print(f"text-only count: {n}")


def example_messages() -> None:
    """Count tokens in a chat-completion-shaped messages list."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    n = arcllm.token_counter(model="gpt-4o", messages=messages)
    print(f"messages count: {n}")


def example_unknown_model_falls_back() -> None:
    """Heuristic path: any model the encoder doesn't recognise → chars/4.

    Ignore the one-time warning so the demo is quiet — production
    callers should let it surface.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = arcllm.token_counter(model="some-vendor/custom-model", text="hello world")
    print(f"unknown-model heuristic count: {n} (chars/4 of 'hello world')")


if __name__ == "__main__":
    example_text_only()
    example_messages()
    example_unknown_model_falls_back()
