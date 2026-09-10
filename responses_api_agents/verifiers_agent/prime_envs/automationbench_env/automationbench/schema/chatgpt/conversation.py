# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT Conversation (Thread) and Message models."""

from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.chatgpt.base import (
    ChatGPTRecord,
    generate_chatgpt_message_id,
    generate_chatgpt_thread_id,
    generate_unix_timestamp,
)


class Message(ChatGPTRecord):
    """An individual message within a conversation thread."""

    id: str = Field(default_factory=generate_chatgpt_message_id)
    object: Literal["thread.message"] = "thread.message"

    # Thread association
    thread_id: str

    # Message content
    role: Literal["user", "assistant", "system"] = "user"
    content: str

    # Assistant association (for assistant messages)
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None

    # Attachments
    file_ids: list[str] = Field(default_factory=list)

    # Metadata
    metadata: dict = Field(default_factory=dict)

    # Timestamps
    created_at: int = Field(default_factory=generate_unix_timestamp)

    # State
    is_deleted: bool = False


class Conversation(ChatGPTRecord):
    """A conversation thread grouping related messages."""

    id: str = Field(default_factory=generate_chatgpt_thread_id)
    object: Literal["thread"] = "thread"

    # Memory key for continuation
    memory_key: Optional[str] = None

    # Assistant association
    assistant_id: Optional[str] = None

    # Metadata
    metadata: dict = Field(default_factory=dict)

    # Timestamps
    created_at: int = Field(default_factory=generate_unix_timestamp)

    # State
    is_deleted: bool = False
