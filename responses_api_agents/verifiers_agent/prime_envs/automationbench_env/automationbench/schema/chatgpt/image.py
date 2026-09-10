# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT Generated Image model."""

from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.chatgpt.base import (
    ChatGPTRecord,
    generate_chatgpt_image_id,
    generate_unix_timestamp,
)


class GeneratedImage(ChatGPTRecord):
    """An image produced by DALL-E or a GPT Image model."""

    id: str = Field(default_factory=generate_chatgpt_image_id)

    # Generation parameters
    prompt: str
    model: Literal["dall-e-2", "dall-e-3", "gpt-image-1", "gpt-image-1.5"] = "dall-e-3"
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024"
    quality: Literal["standard", "hd"] = "standard"
    style: Literal["vivid", "natural"] = "vivid"

    # Output
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None

    # User tracking
    user: Optional[str] = None

    # Timestamps
    created_at: int = Field(default_factory=generate_unix_timestamp)
