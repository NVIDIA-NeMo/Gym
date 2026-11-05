# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Images that a tool returns next to its JSON reply.

A Resources Server tool route can list images under ``tool_output_images`` in its JSON reply. Each entry is a
``ToolOutputImage``: a base64 ``data:`` URL and an optional label that the model sees right before the image.
An agent sends the rest of the reply as the tool output and the images to the model as images: the MCP endpoint
returns them as image content in the tool result, and ``simple_agent`` sends them in user messages after the turn's
tool outputs.
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
    A malformed image list raises ``pydantic.ValidationError``: it is a fault of the tool, not of the model.
    """
    if not isinstance(payload, dict) or TOOL_OUTPUT_IMAGES_KEY not in payload:
        return payload, []
    rest = {key: value for key, value in payload.items() if key != TOOL_OUTPUT_IMAGES_KEY}
    return rest, _TOOL_OUTPUT_IMAGES.validate_python(payload[TOOL_OUTPUT_IMAGES_KEY])
