"""
Simple completion example for arcllm.

This example demonstrates basic chat completion usage.

Run::

    export OPENAI_API_KEY="sk-..."
    python examples/simple_completion.py
"""

import arcllm


def main():
    """Run a simple completion."""
    # Simple completion
    response = arcllm.completion(
        model="gpt-4o-mini",  # or "openai/gpt-4o-mini"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        temperature=0.7,
        max_tokens=100,
    )

    # Access the response
    print("Response ID:", response.id)
    print("Model:", response.model)
    print("Content:", response.choices[0].message.content)
    print("Finish reason:", response.choices[0].finish_reason)

    # Access usage
    if response.usage:
        print("\nUsage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")

    # Calculate cost
    try:
        cost = arcllm.completion_cost(response)
        print(f"\nEstimated cost: ${cost:.6f}")
    except Exception as e:
        print(f"\nCould not calculate cost: {e}")


def example_with_different_providers():
    """Sketches of model strings for other providers.

    Each block is commented out — uncomment after exporting the matching
    API key (see README "Authentication" table).
    """
    # OpenAI
    # arcllm.completion(model="openai/gpt-4o-mini", messages=msgs)

    # Anthropic — set ANTHROPIC_API_KEY
    # arcllm.completion(model="anthropic/claude-haiku-4-5", messages=msgs)

    # Gemini — set GEMINI_API_KEY
    # arcllm.completion(model="gemini/gemini-2.5-flash-lite", messages=msgs)

    # Groq — set GROQ_API_KEY
    # arcllm.completion(model="groq/llama-3.3-70b-versatile", messages=msgs)

    # xAI — set XAI_API_KEY
    # arcllm.completion(model="xai/grok-3-mini", messages=msgs)

    # Cohere — set COHERE_API_KEY
    # arcllm.completion(model="cohere/command-r-08-2024", messages=msgs)
    return


if __name__ == "__main__":
    main()
