# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT Vector Store model."""

from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.chatgpt.base import (
    ChatGPTRecord,
    generate_chatgpt_vector_store_id,
    generate_unix_timestamp,
)


class VectorStoreFileCounts(ChatGPTRecord):
    """Breakdown of file processing states within a vector store."""

    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    total: int = 0


class VectorStore(ChatGPTRecord):
    """An OpenAI Vector Store used to enable semantic file search."""

    id: str = Field(default_factory=generate_chatgpt_vector_store_id)
    object: Literal["vector_store"] = "vector_store"

    # Basic info
    name: Optional[str] = None

    # File associations
    file_ids: list[str] = Field(default_factory=list)
    file_counts: VectorStoreFileCounts = Field(default_factory=VectorStoreFileCounts)

    # Usage
    usage_bytes: int = 0

    # Status
    status: Literal["expired", "in_progress", "completed"] = "completed"

    # Expiration policy
    expires_after_days: Optional[int] = None
    expires_at: Optional[int] = None
    last_active_at: Optional[int] = None

    # Timestamps
    created_at: int = Field(default_factory=generate_unix_timestamp)

    # Metadata
    metadata: dict = Field(default_factory=dict)

    # State
    is_deleted: bool = False
