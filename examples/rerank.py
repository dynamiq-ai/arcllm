"""
Reranking example for arcllm.

Cohere is the reference rerank provider in 0.4 (Voyage / Bedrock /
Jina rerank land in 0.5). The same call shape works for any provider
that exposes ``arcllm.rerank``.

Run::

    export COHERE_API_KEY="cohere-..."
    python examples/rerank.py
"""

from __future__ import annotations

import arcllm


def main() -> None:
    query = "Who created the Python programming language?"
    documents = [
        "Linus Torvalds created the Linux kernel in 1991.",
        "Guido van Rossum created the Python programming language in 1991.",
        "Dennis Ritchie designed the C programming language at Bell Labs.",
        "Bjarne Stroustrup designed C++ as an extension of C.",
        "James Gosling created the Java programming language at Sun Microsystems.",
    ]

    response = arcllm.rerank(
        model="cohere/rerank-v3.5",
        query=query,
        documents=documents,
        top_n=3,
    )

    print(f"Query: {query}\n")
    print(f"Top {len(response.results)} of {len(documents)} documents:")
    for rank, r in enumerate(response.results, 1):
        print(f"  {rank}. [#{r.index}] score={r.relevance_score:.3f}")
        print(f"     {r.document}")


async def async_main() -> None:
    """Same call, async — useful inside an event loop."""
    response = await arcllm.arerank(
        model="cohere/rerank-v3.5",
        query="best small open-weights models",
        documents=[
            "Llama 3.2 1B is a tiny instruction-tuned model.",
            "GPT-4o is OpenAI's flagship multimodal model.",
            "Phi-4 mini is a 4B-parameter Microsoft model.",
        ],
        top_n=2,
    )
    for r in response.results:
        print(f"#{r.index} {r.relevance_score:.3f} {r.document}")


if __name__ == "__main__":
    main()

    print("\n=== Async ===\n")
    import asyncio

    asyncio.run(async_main())
