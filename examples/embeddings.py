"""
Embeddings example for arcllm.

This example demonstrates creating text embeddings.
"""

import arcllm


def main():
    """Run embeddings example."""
    print("=== Embeddings Example ===\n")

    # Create embeddings for multiple texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries.",
        "Python is a versatile programming language.",
    ]

    response = arcllm.embedding(
        model="text-embedding-3-small",
        input=texts,
    )

    print(f"Model: {response.model}")
    print(f"Number of embeddings: {len(response.data)}")
    print("\nUsage:")
    print(f"  Prompt tokens: {response.usage.prompt_tokens}")
    print(f"  Total tokens: {response.usage.total_tokens}")

    # Show embedding dimensions and first few values
    for i, embedding_data in enumerate(response.data):
        embedding = embedding_data.embedding
        print(f"\nText {i + 1}: '{texts[i][:50]}...'")
        print(f"  Dimensions: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")


def single_embedding():
    """Create a single embedding."""
    print("\n=== Single Embedding ===\n")

    response = arcllm.embedding(
        model="text-embedding-3-small",
        input="Hello, world!",
    )

    embedding = response.data[0].embedding
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def similarity_example():
    """Calculate similarity between texts."""
    print("\n=== Similarity Example ===\n")

    texts = [
        "I love programming in Python",
        "Python is my favorite language",
        "The weather is nice today",
    ]

    response = arcllm.embedding(
        model="text-embedding-3-small",
        input=texts,
    )

    embeddings = [d.embedding for d in response.data]

    print("Cosine similarities:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  '{texts[i][:30]}...' <-> '{texts[j][:30]}...': {sim:.4f}")


async def async_embedding():
    """Async embedding example."""
    print("\n=== Async Embedding ===\n")

    response = await arcllm.aembedding(
        model="text-embedding-3-small",
        input=["Async embeddings are cool!"],
    )

    print(f"Embedding dimensions: {len(response.data[0].embedding)}")


if __name__ == "__main__":
    main()
    single_embedding()
    similarity_example()

    # To run async example:
    # import asyncio
    # asyncio.run(async_embedding())
