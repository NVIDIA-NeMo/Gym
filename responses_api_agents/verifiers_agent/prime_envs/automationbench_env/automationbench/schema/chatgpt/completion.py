# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT Completion/Response model."""

from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.chatgpt.base import (
    ChatGPTRecord,
    generate_chatgpt_response_id,
    generate_unix_timestamp,
)


class CompletionUsage(ChatGPTRecord):
    """Tracks how many tokens were consumed by a completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Completion(ChatGPTRecord):
    """A single ChatGPT completion or model response."""

    id: str = Field(default_factory=generate_chatgpt_response_id)
    object: Literal["chat.completion", "text_completion", "response"] = "chat.completion"

    # Model used
    model: str = "gpt-5-mini"

    # Input
    prompt: Optional[str] = None
    user_message: Optional[str] = None
    instructions: Optional[str] = None

    # Response content
    response_text: str = ""
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", None] = "stop"

    # For structured outputs
    response_format: Literal["text", "json_schema", "json_object"] = "text"

    # Tool usage
    tools_used: list[str] = Field(default_factory=list)
    tool_calls_count: int = 0

    # Token usage
    usage: CompletionUsage = Field(default_factory=CompletionUsage)

    # Timestamps
    created_at: int = Field(default_factory=generate_unix_timestamp)

    # Cancellation
    was_cancelled: bool = False

    # Metadata
    metadata: dict = Field(default_factory=dict)
