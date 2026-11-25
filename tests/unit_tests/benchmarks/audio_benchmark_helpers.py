# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the audio benchmark tests (``test_<name>_benchmark.py``)."""

import io
import json
import wave
from pathlib import Path
from types import ModuleType

import numpy as np


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def sine_wav_bytes(rate: int = 16000, seconds: float = 0.5, channels: int = 1) -> bytes:
    """A 440 Hz 16-bit PCM WAV (written with ``wave``, no ``soundfile``); two channels are the tone and its inverse."""
    mono = 0.25 * np.sin(2 * np.pi * 440 * np.arange(int(rate * seconds)) / rate).astype(np.float32)
    samples = mono if channels == 1 else np.stack([mono, -mono], axis=1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def pcm_wav_bytes(frames: int, rate: int = 16000, channels: int = 1, sample: bytes = b"\x01\x00") -> bytes:
    """A 16-bit PCM WAV of ``frames`` frames, every sample set to ``sample``."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(sample * frames * channels)
    return buffer.getvalue()


def assert_capped_and_remapped_rows(
    module: ModuleType, output: Path, *, rows: int, audio_path: str, index: int = 0
) -> list[dict]:
    """A ``prepare(audio_root=..., max_samples_per_profile=...)`` run keeps ``rows`` rows, prefixes the audio path
    of row ``index`` with the root, and is never mistaken for the default prepared data."""
    loaded = read_rows(output)
    assert len(loaded) == rows
    assert loaded[index]["responses_create_params"]["metadata"]["audio_path"] == audio_path
    assert not module.is_prepared_data_current(output), "capped or remapped data must be prepared again by default"
    return loaded
