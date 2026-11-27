# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The standard-library audio stand-ins and the WAV helpers the audio benchmark tests rely on."""

import io
import sys

import numpy as np
import pytest

from tests.unit_tests.benchmarks import stdlib_audio
from tests.unit_tests.benchmarks.audio_benchmark_helpers import pcm_wav_bytes, sine_wav_bytes


def test_a_wav_round_trips_through_the_stand_in_reader_and_writer() -> None:
    samples = np.array([0.0, 0.5, -0.5, 0.25], dtype="float64")
    buffer = io.BytesIO()
    stdlib_audio.write(buffer, samples, 16000)
    data, rate = stdlib_audio.read(io.BytesIO(buffer.getvalue()))
    assert rate == 16000 and np.allclose(data, samples, atol=1e-4)
    details = stdlib_audio.info(io.BytesIO(buffer.getvalue()))
    assert (details.frames, details.samplerate, details.channels) == (4, 16000, 1)
    assert details.duration == pytest.approx(4 / 16000)


def test_stereo_audio_is_read_as_two_columns_and_bytes_are_accepted() -> None:
    data, rate = stdlib_audio.read(sine_wav_bytes(rate=8000, seconds=0.1, channels=2))
    assert rate == 8000 and data.shape == (800, 2)
    assert np.allclose(data[:, 0], -data[:, 1], atol=1e-3)
    assert stdlib_audio.info(pcm_wav_bytes(frames=10)).frames == 10


def test_audio_the_stand_in_cannot_decode_raises_the_soundfile_error() -> None:
    with pytest.raises(stdlib_audio.LibsndfileError):
        stdlib_audio.info(io.BytesIO(b"not audio"))


def test_resampling_changes_the_length_by_the_rate_ratio() -> None:
    samples = np.sin(np.linspace(0, 20, 1600)).astype("float32")
    assert len(stdlib_audio.resample(samples, 16000, 8000)) == 800
    assert stdlib_audio.resample(samples, 16000, 16000).dtype == samples.dtype


def test_stand_ins_are_installed_only_for_a_package_that_cannot_be_imported(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "soundfile", None)  # an import of it now fails
    monkeypatch.setitem(sys.modules, "soxr", object())  # an importable package is left alone
    installed = stdlib_audio.install(monkeypatch)
    assert installed == ["soundfile"]
    assert sys.modules["soundfile"].read is stdlib_audio.read
    assert not hasattr(sys.modules["soxr"], "resample")
