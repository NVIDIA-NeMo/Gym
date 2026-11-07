# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Images that a tool returns next to its JSON reply.

A Resources Server tool route can list images under ``tool_output_images`` in its JSON reply. Each entry is a
``ToolOutputImage``: a base64 ``data:`` URL and an optional label, which the model gets as text before the image.
``simple_agent`` sends the rest of the reply as the tool output and each image in a user message after the turn's
tool outputs. The MCP endpoint returns the images as image content after the tool result's text. An agent that
does not read the key, such as ``simple_agent_with_compaction``, passes the whole reply to the model as text,
base64 data included.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


TOOL_OUTPUT_IMAGES_KEY = "tool_output_images"


class ToolOutputImage(BaseModel):
    """One image in a tool reply."""

    model_config = ConfigDict(extra="forbid")

    image_url: str = Field(pattern=r"^data:image/(png|jpeg|gif|webp);base64,")
    label: Optional[str] = None

    @property
    def media_type(self) -> str:
        """The image's media type, such as ``image/png``."""
        return self.image_url[len("data:") : self.image_url.index(";")]

    @property
    def data(self) -> str:
        """The image's base64 data, without the ``data:`` prefix."""
        return self.image_url.partition(",")[2]


_TOOL_OUTPUT_IMAGES = TypeAdapter(list[ToolOutputImage])


def split_tool_output_images(payload: object) -> tuple[object, list[ToolOutputImage]]:
    """Return a tool reply without its ``tool_output_images``, and those images.

    A reply that is not a JSON object, or has no ``tool_output_images`` key, comes back unchanged with no images.
    A null image list means no images. A malformed image list raises ``pydantic.ValidationError``: it is a fault of
    the tool, not of the model.
    """
    if not isinstance(payload, dict) or TOOL_OUTPUT_IMAGES_KEY not in payload:
        return payload, []
    rest = {key: value for key, value in payload.items() if key != TOOL_OUTPUT_IMAGES_KEY}
    images = payload[TOOL_OUTPUT_IMAGES_KEY]
    # A typed reply model with an optional image list sends null when the reply has no images.
    return rest, [] if images is None else _TOOL_OUTPUT_IMAGES.validate_python(images)
