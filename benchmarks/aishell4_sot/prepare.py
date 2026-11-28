# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the AISHELL-4 test meetings (speaker-attributed Mandarin transcription) for Gym.

The 20 test meetings and their TextGrid annotations come from the pinned ``AISHELL/AISHELL-4`` revision (a Hugging
Face copy of OpenSLR SLR111). Each meeting is one 8-channel FLAC; channel zero is written as a 16 kHz mono 16-bit WAV
to ``data/audio``, the deliberate single-distant-microphone condition.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download

from benchmarks import audio_manifest
from benchmarks import mandarin_meeting as meeting


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "aishell4_sot_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), Path(meeting.__file__), PROFILES, Path(audio_manifest.__file__))

HF_REPO_ID = "AISHELL/AISHELL-4"
HF_REVISION = "aada72727856313b19d4a030383c426364931dbf"  # pragma: allowlist secret
SOURCE_SPLIT = "test"
# Revision of the corpus' released baseline repository, whose text normalization the evaluator follows.
OFFICIAL_REPOSITORY_REVISION = "bad82b77c3753df1b232c5c6491cd3e2f2e32d24"  # pragma: allowlist secret
SESSION_IDS = (
    "L_R003S01C02",
    "L_R003S02C02",
    "L_R003S03C02",
    "L_R003S04C02",
    "L_R004S01C01",
    "L_R004S02C01",
    "L_R004S03C01",
    "L_R004S06C01",
    "M_R003S01C01",
    "M_R003S02C01",
    "M_R003S04C01",
    "M_R003S05C01",
    "S_R003S01C01",
    "S_R003S02C01",
    "S_R003S03C01",
    "S_R003S04C01",
    "S_R004S01C01",
    "S_R004S02C01",
    "S_R004S03C01",
    "S_R004S04C01",
)
SOURCE_METADATA = {
    "source_repo": HF_REPO_ID,
    "source_revision": HF_REVISION,
    "source_split": SOURCE_SPLIT,
    "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
    "license": "CC-BY-SA-4.0",
}


def _flac_name(session_id: str) -> str:
    return f"{SOURCE_SPLIT}/wav/{session_id}.flac"


def _textgrid_name(session_id: str) -> str:
    return f"{SOURCE_SPLIT}/TextGrid/{session_id}.TextGrid"


def _source_info(session_id: str) -> dict:
    """Read the FLAC header (channels, frames, sample rate) with a range request instead of downloading the file."""
    import soundfile as sf

    with HfFileSystem().open(f"datasets/{HF_REPO_ID}@{HF_REVISION}/{_flac_name(session_id)}", "rb") as stream:
        info = sf.info(stream)
    return {"channels": info.channels, "frames": info.frames, "sample_rate": info.samplerate}


def convert_audio(session_id: str, destination: Path) -> None:
    """Download one meeting, write channel zero to ``destination`` and delete the download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".download-") as staging:
        flac = hf_hub_download(
            HF_REPO_ID, _flac_name(session_id), repo_type="dataset", revision=HF_REVISION, local_dir=staging
        )
        meeting.write_first_channel_wav(Path(flac), destination)


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(*, audio_root: str | None = None, max_samples_per_profile: int | None = None) -> Path:
    """Download AISHELL-4 and write the combined benchmark JSONL.

    Args:
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N meetings (smoke runs); a positive integer or ``None``.
    """
    if max_samples_per_profile is not None and (
        isinstance(max_samples_per_profile, bool)
        or not isinstance(max_samples_per_profile, int)
        or max_samples_per_profile < 1
    ):
        raise ValueError(
            f"max_samples_per_profile must be a positive integer or None, got {max_samples_per_profile!r}"
        )
    audio_dir = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"
    sources = []
    for session_id in SESSION_IDS[:max_samples_per_profile]:
        wav = DATA_DIR / "audio" / f"{session_id}.wav"
        if not wav.is_file():
            convert_audio(session_id, wav)
        textgrid = Path(
            hf_hub_download(HF_REPO_ID, _textgrid_name(session_id), repo_type="dataset", revision=HF_REVISION)
        )
        sources.append(
            meeting.format_source_row(
                corpus="aishell4-sot",
                session_id=session_id,
                audio_path=f"{audio_dir}/{session_id}.wav",
                duration=meeting.wav_duration(wav),
                textgrid=textgrid.read_text(encoding="utf-8-sig"),
                source_metadata=SOURCE_METADATA,
                source_audio=_flac_name(session_id),
                source_textgrid=_textgrid_name(session_id),
                source_info=_source_info(session_id),
            )
        )

    prepared = audio_manifest.profile_rows(PROFILES, sources)
    options = {"audio_root": audio_root, "max_samples_per_profile": max_samples_per_profile}
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
