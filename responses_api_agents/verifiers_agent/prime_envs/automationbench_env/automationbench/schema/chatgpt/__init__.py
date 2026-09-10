# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Schema models for ChatGPT."""

from automationbench.schema.chatgpt.assistant import Assistant
from automationbench.schema.chatgpt.base import (
    ChatGPTRecord,
    ChatGPTState,
    generate_chatgpt_assistant_id,
    generate_chatgpt_file_id,
    generate_chatgpt_image_id,
    generate_chatgpt_message_id,
    generate_chatgpt_response_id,
    generate_chatgpt_run_id,
    generate_chatgpt_thread_id,
    generate_chatgpt_vector_store_id,
    generate_unix_timestamp,
)
from automationbench.schema.chatgpt.completion import Completion, CompletionUsage
from automationbench.schema.chatgpt.conversation import Conversation, Message
from automationbench.schema.chatgpt.file import File
from automationbench.schema.chatgpt.image import GeneratedImage
from automationbench.schema.chatgpt.transcription import (
    Transcription,
    TranscriptionSegment,
    TranscriptionWord,
)
from automationbench.schema.chatgpt.vector_store import VectorStore, VectorStoreFileCounts

# Resolve forward references once all imports are complete
ChatGPTState.model_rebuild()

__all__ = [
    "ChatGPTRecord",
    "ChatGPTState",
    "generate_chatgpt_assistant_id",
    "generate_chatgpt_file_id",
    "generate_chatgpt_image_id",
    "generate_chatgpt_message_id",
    "generate_chatgpt_response_id",
    "generate_chatgpt_run_id",
    "generate_chatgpt_thread_id",
    "generate_chatgpt_vector_store_id",
    "generate_unix_timestamp",
    "Assistant",
    "Completion",
    "CompletionUsage",
    "Conversation",
    "Message",
    "File",
    "GeneratedImage",
    "Transcription",
    "TranscriptionSegment",
    "TranscriptionWord",
    "VectorStore",
    "VectorStoreFileCounts",
]
