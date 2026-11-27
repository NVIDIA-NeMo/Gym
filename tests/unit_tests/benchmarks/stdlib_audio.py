# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard-library stand-ins for ``soundfile`` and ``soxr``, for the benchmark ``prepare()`` tests.

The audio benchmarks' ``prepare.py`` modules decode and write audio with ``soundfile`` (and resample with ``soxr``),
which the benchmark runtime installs through ``prepare_dependencies`` and the core test environment does not contain.
The offline tests only ever handle 16-bit PCM WAV, so ``wave`` and NumPy are enough to run them without those
packages. The stand-ins are installed only when the real package cannot be imported, so an environment that has
``soundfile`` still exercises it. They are not general codecs: they read and write PCM16 WAV, whatever the requested
``format`` (a ``.flac`` name or ``format="FLAC"`` still gets WAV bytes), and ``soxr.resample`` is polyphase resampling.
"""

from __future__ import annotations

import importlib
import io
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from scipy.signal import resample_poly


class LibsndfileError(RuntimeError):
    """What ``soundfile`` raises for audio it cannot open or decode."""


@dataclass(frozen=True)
class Info:
    frames: int
    samplerate: int
    channels: int
    subtype: str = "PCM_16"
    format: str = "WAV"

    @property
    def duration(self) -> float:
        return self.frames / self.samplerate


def _to_float(raw: bytes, channels: int, dtype: str) -> np.ndarray:
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
    return samples.astype(dtype) / 32768.0 if np.dtype(dtype).kind == "f" else samples.astype(dtype)


def _to_pcm16(data: Any, channels: int) -> bytes:
    array = np.asarray(data)
    if array.dtype.kind == "f":
        array = np.clip(np.rint(array.astype("float64") * 32767.0), -32768, 32767)
    return array.reshape(-1, channels).astype("<i2").tobytes()


class SoundFile:
    """Reads or writes a PCM16 WAV file or file object."""

    def __init__(
        self,
        file: Any,
        mode: str = "r",
        samplerate: int | None = None,
        channels: int | None = None,
        subtype: str | None = None,
        format: str | None = None,
        **_: Any,
    ) -> None:
        self._writing = "w" in mode
        target = io.BytesIO(file) if isinstance(file, bytes) else file
        try:
            self._wave = wave.open(
                str(target) if isinstance(target, Path) else target, "wb" if self._writing else "rb"
            )
        except (wave.Error, EOFError) as error:
            raise LibsndfileError(f"Error opening {file!r}: {error}") from error
        if self._writing:
            self.samplerate, self.channels, self.frames = int(samplerate), int(channels), 0
            self._wave.setnchannels(self.channels)
            self._wave.setsampwidth(2)
            self._wave.setframerate(self.samplerate)
        else:
            if self._wave.getsampwidth() != 2:
                self._wave.close()
                raise LibsndfileError("The stand-in reads 16-bit PCM WAV only")
            self.samplerate, self.channels, self.frames = (
                self._wave.getframerate(),
                self._wave.getnchannels(),
                self._wave.getnframes(),
            )
        self.subtype, self.format = "PCM_16", "WAV"

    def __len__(self) -> int:
        return self.frames

    def read(self, frames: int = -1, dtype: str = "float64", always_2d: bool = False) -> np.ndarray:
        raw = self._wave.readframes(self.frames if frames < 0 else frames)
        samples = _to_float(raw, self.channels, dtype)
        return samples if always_2d or self.channels > 1 else samples[:, 0]

    def seek(self, frames: int) -> int:
        self._wave.setpos(frames)
        return frames

    def write(self, data: Any) -> None:
        self._wave.writeframes(_to_pcm16(data, self.channels))
        self.frames += len(np.asarray(data))

    def close(self) -> None:
        self._wave.close()

    def __enter__(self) -> "SoundFile":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def info(file: Any) -> Info:
    with SoundFile(file) as stream:
        return Info(stream.frames, stream.samplerate, stream.channels)


def read(file: Any, frames: int = -1, dtype: str = "float64", always_2d: bool = False, **_: Any):
    with SoundFile(file) as stream:
        return stream.read(frames, dtype, always_2d), stream.samplerate


def write(file: Any, data: Any, samplerate: int, **_: Any) -> None:
    array = np.asarray(data)
    with SoundFile(file, "w", samplerate=samplerate, channels=1 if array.ndim == 1 else array.shape[1]) as stream:
        stream.write(array)


def resample(samples: np.ndarray, in_rate: float, out_rate: float, **_: Any) -> np.ndarray:
    divisor = int(np.gcd(int(in_rate), int(out_rate)))
    return resample_poly(samples, int(out_rate) // divisor, int(in_rate) // divisor).astype(samples.dtype)


def _module(name: str, **members: Any) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(members)
    return module


def stand_ins() -> dict[str, ModuleType]:
    """The stand-in module for each of ``soundfile`` and ``soxr`` that the environment cannot import."""
    candidates = {
        "soundfile": _module(
            "soundfile",
            SoundFile=SoundFile,
            LibsndfileError=LibsndfileError,
            info=info,
            read=read,
            write=write,
        ),
        "soxr": _module("soxr", resample=resample),
    }
    missing = {}
    for name, module in candidates.items():
        try:
            importlib.import_module(name)
        except ImportError:
            missing[name] = module
    return missing


def install(monkeypatch: Any) -> list[str]:
    """Put the missing stand-ins into ``sys.modules`` for one test (undone by ``monkeypatch``)."""
    modules = stand_ins()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return sorted(modules)
