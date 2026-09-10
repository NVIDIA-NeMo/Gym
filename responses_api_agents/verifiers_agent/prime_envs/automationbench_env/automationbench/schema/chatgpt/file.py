# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT File model."""

from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.chatgpt.base import (
    ChatGPTRecord,
    generate_chatgpt_file_id,
    generate_unix_timestamp,
)


class File(ChatGPTRecord):
    """An uploaded file managed by the OpenAI platform."""

    id: str = Field(default_factory=generate_chatgpt_file_id)
    object: Literal["file"] = "file"

    # File info
    filename: str
    bytes: int = 0
    purpose: Literal["assistants", "fine-tune", "batch", "user_data"] = "assistants"

    # Content
    content_type: Optional[str] = None
    url: Optional[str] = None

    # Timestamps
    created_at: int = Field(default_factory=generate_unix_timestamp)

    # Status
    status: Literal["uploaded", "processed", "error"] = "uploaded"
    status_details: Optional[str] = None

    # State
    is_deleted: bool = False
