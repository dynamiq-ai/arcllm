"""
Streaming completion example for arcllm.

This example demonstrates streaming responses with usage tracking.
"""

import arcllm


def main():
    """Run a streaming completion."""
    print("Streaming response:\n")

    # Stream with usage included
    stream = arcllm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Write a short poem about coding."},
        ],
        stream=True,
        stream_options={"include_usage": True},  # Get usage in final chunk
    )

    # Collect chunks for building final response
    chunks = []

    # Iterate over stream chunks
    for chunk in stream:
        chunks.append(chunk)

        # Print content as it arrives
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

        # Check for finish reason
        if chunk.choices and chunk.choices[0].finish_reason:
            print(f"\n\nFinish reason: {chunk.choices[0].finish_reason}")

        # Print usage if present (final chunk with include_usage=True)
        if chunk.usage:
            print("\nUsage:")
            print(f"  Prompt tokens: {chunk.usage.prompt_tokens}")
            print(f"  Completion tokens: {chunk.usage.completion_tokens}")
            print(f"  Total tokens: {chunk.usage.total_tokens}")

    # Build complete response from chunks
    print("\n--- Building complete response from chunks ---")
    complete_response = arcllm.stream_chunk_builder(chunks)
    print(f"Complete content: {complete_response.choices[0].message.content[:100]}...")


async def async_streaming():
    """Async streaming example."""
    print("Async streaming response:\n")

    stream = await arcllm.acompletion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Count from 1 to 5."},
        ],
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()


if __name__ == "__main__":
    main()

    print("\n\n=== Async streaming ===\n")
    import asyncio

    asyncio.run(async_streaming())
