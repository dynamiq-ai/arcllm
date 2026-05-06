"""
Image generation / variation / edit example for arcllm.

OpenAI is the reference image provider in 0.4 (DALL-E 3, gpt-image-1,
DALL-E 2 for variations). The call shape mirrors the OpenAI Images API.

Run::

    export OPENAI_API_KEY="sk-..."
    python examples/image_generation.py
"""

from __future__ import annotations

import base64
from pathlib import Path

import arcllm


def example_generation() -> None:
    """Text-to-image with DALL-E 3."""
    response = arcllm.image_generation(
        model="openai/dall-e-3",
        prompt="A teal arc connecting two glowing endpoints, vector art, minimal style",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image = response.data[0]
    print(f"Generated image URL: {image.url}")
    if image.revised_prompt:
        print(f"DALL-E 3 revised prompt to: {image.revised_prompt!r}")


def example_b64_response() -> None:
    """Same call, but ask for the bytes inline as base64."""
    response = arcllm.image_generation(
        model="openai/dall-e-3",
        prompt="abstract circuit board pattern, cyan and slate",
        response_format="b64_json",
        size="1024x1024",
    )
    if response.data[0].b64_json:
        out = Path("/tmp/arcllm_example_circuit.png")
        out.write_bytes(base64.b64decode(response.data[0].b64_json))
        print(f"Wrote {out} ({out.stat().st_size} bytes)")


def example_variation_and_edit() -> None:
    """Variation + edit (multipart). Requires a starter image on disk.

    Skipped here — the multipart flow needs an actual PNG. See the README
    for the call shape; everything routes through the same adapter.
    """
    # arcllm.image_variation(
    #     model="openai/dall-e-2",
    #     image=Path("starter.png").read_bytes(),
    #     n=1,
    #     size="512x512",
    # )
    # arcllm.image_edit(
    #     model="openai/gpt-image-1",
    #     image=Path("starter.png").read_bytes(),
    #     mask=Path("mask.png").read_bytes(),
    #     prompt="replace the sky with a starfield",
    # )
    return


async def async_main() -> None:
    """Same call, async."""
    response = await arcllm.aimage_generation(
        model="openai/dall-e-3",
        prompt="a single sharp arc carved into stone, monochrome",
        size="1024x1024",
    )
    print(f"async URL: {response.data[0].url}")


if __name__ == "__main__":
    example_generation()
    example_b64_response()
    example_variation_and_edit()

    print("\n=== Async ===\n")
    import asyncio

    asyncio.run(async_main())
