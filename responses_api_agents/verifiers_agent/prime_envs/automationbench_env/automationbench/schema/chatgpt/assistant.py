# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT Assistant model."""

from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.chatgpt.base import (
    ChatGPTRecord,
    generate_chatgpt_assistant_id,
    generate_unix_timestamp,
)


class Assistant(ChatGPTRecord):
    """An OpenAI Assistant configuration."""

    id: str = Field(default_factory=generate_chatgpt_assistant_id)
    object: Literal["assistant"] = "assistant"

    # Basic info
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    model: str = "gpt-5-mini"

    # Tools available to the assistant
    tools: list[str] = Field(
        default_factory=list,
        description="Capabilities granted to the assistant, such as code_interpreter or file_search",
    )

    # File associations
    file_ids: list[str] = Field(default_factory=list)
    vector_store_id: Optional[str] = None

    # Metadata
    metadata: dict = Field(default_factory=dict)

    # Timestamps (Unix epoch)
    created_at: int = Field(default_factory=generate_unix_timestamp)

    # State
    is_deleted: bool = False
