# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Coval STT v1 (50 LibriSpeech test-clean utterances) for Gym.

The dataset definition is the pinned ``stt-v1`` manifest of the ``coval-ai/benchmarks`` repository (checked by
SHA-256). Each utterance is a public 16 kHz mono PCM16 WAV object whose SHA-256 the manifest lists.
"""

from __future__ import annotations

import json
import wave
from decimal import Decimal
from pathlib import Path, PurePosixPath

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "coval_stt_v1_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

SOURCE_REPO = "https://github.com/coval-ai/benchmarks"
SOURCE_REVISION = "db1e3d4c731b9051292fa36e73d9d2dc859372d3"  # pragma: allowlist secret
SOURCE_MANIFEST_PATH = "runner/src/coval_bench/datasets/manifests/stt-v1.json"
SOURCE_MANIFEST_URL = f"https://raw.githubusercontent.com/coval-ai/benchmarks/{SOURCE_REVISION}/{SOURCE_MANIFEST_PATH}"
SOURCE_MANIFEST_SHA256 = "8c35e1d080e5482bb64eb8b9f39fd870ca334f9f5ca6d188680750715b6605e3"  # pragma: allowlist secret
SOURCE_OBJECT_ROOT = "https://storage.googleapis.com/coval-benchmarks-datasets/stt-v1"
SOURCE_DATASET_ID = "stt-v1"
SOURCE_DATASET_VERSION = "1.0.0"
SOURCE_DATASET_LICENSE = "CC-BY-4.0"
EXPECTED_ROWS = 50
EXPECTED_DURATION_SECONDS = Decimal("393.265063")
EXPECTED_SPEAKERS = 40
SAMPLE_RATE = 16000
# The prompt of the source row; the profile's prompt_override replaces it in the request.
PROMPT = "Transcribe the following:"


def load_items() -> list[dict]:
    """Download the pinned manifest and check it against the released census."""
    path = audio_manifest.download(SOURCE_MANIFEST_URL, DATA_DIR / "source" / "stt-v1.json", SOURCE_MANIFEST_SHA256)
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = {"id": SOURCE_DATASET_ID, "version": SOURCE_DATASET_VERSION, "license": SOURCE_DATASET_LICENSE}
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"Coval manifest {key} drift: expected {value!r}, got {document.get(key)!r}")
    items = document["items"]
    if len(items) != EXPECTED_ROWS:
        raise ValueError(f"Coval row-count drift: expected {EXPECTED_ROWS}, got {len(items)}")
    for number, item in enumerate(items, 1):
        if PurePosixPath(item["path"]) != PurePosixPath("audio") / f"{number:04d}.wav":
            raise ValueError(f"Coval manifest row {number} has a noncanonical path: {item['path']!r}")
        if not item["transcript"].strip() or not item["utterance_id"].startswith(
            f"{item['speaker_id']}-{item['chapter_id']}-"
        ):
            raise ValueError(f"Coval manifest row {number} is not a usable LibriSpeech utterance")
    if len({item["utterance_id"] for item in items}) != EXPECTED_ROWS:
        raise ValueError("Coval manifest has duplicate utterance ids")
    duration = sum(Decimal(str(item["duration_sec"])) for item in items)
    if duration != EXPECTED_DURATION_SECONDS:
        raise ValueError(f"Coval duration drift: expected {EXPECTED_DURATION_SECONDS}, got {duration}")
    if len({item["speaker_id"] for item in items}) != EXPECTED_SPEAKERS:
        raise ValueError(f"Coval speaker-count drift: expected {EXPECTED_SPEAKERS}")
    return items


def format_source_row(item: dict, audio_prefix: str) -> dict:
    audio_path = f"{audio_prefix}/{item['path']}"
    duration = float(item["duration_sec"])
    return {
        "id": item["utterance_id"],
        "messages": [{"role": "user", "content": PROMPT, "audio": {"path": audio_path, "duration": duration}}],
        "expected_answer": item["transcript"],
        "audio_filepath": audio_path,
        "audio_duration": duration,
        "speaker_id": item["speaker_id"],
        "task_type": "Coval-STT-v1-ASR",
        "extra_fields": {
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_manifest_path": SOURCE_MANIFEST_PATH,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_dataset_version": SOURCE_DATASET_VERSION,
            "source_license": SOURCE_DATASET_LICENSE,
            "source_object_path": f"{SOURCE_DATASET_ID}/{item['path']}",
            "source_audio_sha256": item["sha256"],
            "source_chapter_id": item["chapter_id"],
            "source_utterance_id": item["utterance_id"],
        },
    }


def download_audio(items: list[dict], data_dir: Path) -> None:
    """Fetch each utterance (verified against its manifest SHA-256) and check the WAV contract."""
    for item in items:
        path = audio_manifest.download(f"{SOURCE_OBJECT_ROOT}/{item['path']}", data_dir / item["path"], item["sha256"])
        try:
            with wave.open(str(path), "rb") as stream:
                header = (stream.getnchannels(), stream.getsampwidth(), stream.getframerate())
                duration = stream.getnframes() / stream.getframerate()
        except (OSError, EOFError, wave.Error) as error:
            raise ValueError(f"Coval audio is not a WAV file: {path}") from error
        if header != (1, 2, SAMPLE_RATE) or abs(duration - float(item["duration_sec"])) > 1.0 / SAMPLE_RATE:
            raise ValueError(f"Coval audio contract failed for {path}: {header}, {duration}s")


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(*, audio_root: str | None = None, max_samples_per_profile: int | None = None) -> Path:
    """Download Coval STT v1 and write the benchmark JSONL.

    Args:
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N utterances (smoke runs).
    """
    if max_samples_per_profile is not None and max_samples_per_profile < 1:
        raise ValueError(f"max_samples_per_profile must be at least 1, got {max_samples_per_profile}")
    items = load_items()[:max_samples_per_profile]
    download_audio(items, DATA_DIR)

    prefix = audio_manifest.audio_prefix(DATA_DIR, audio_root)
    prepared = audio_manifest.profile_rows(PROFILES, [format_source_row(item, prefix) for item in items])
    options = {"audio_root": audio_root, "max_samples_per_profile": max_samples_per_profile}
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
