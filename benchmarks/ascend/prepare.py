# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare ASCEND (Mandarin-English code-switched conversations) for Gym.

The 1,315 test utterances come from the pinned ``CAiRE/ASCEND`` revision. Preparation authenticates the test parquet
(size and SHA-256), decodes each clip to mono float32 and writes it as a WAV file to ``data/audio``.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "ascend_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

HF_REPO_ID = "CAiRE/ASCEND"
HF_REVISION = "737e9800ae31be9932ba8464c80366559bd28424"  # pragma: allowlist secret
SPLIT = "test"
TEST_PARQUET = "main/test-00000-of-00001.parquet"
TEST_PARQUET_BYTES = 105756434
TEST_PARQUET_SHA256 = "a4c81d2b5ed6124f052089a695972808c16e0ce0c365ec9773c5d1a8fcf043a7"  # pragma: allowlist secret
EXPECTED_ROWS = 1315
# The prompt of the source row; each profile's prompt_override replaces it in the request.
PROMPT = "Transcribe the following audio."


def download_test_parquet() -> Path:
    """Download the pinned test parquet and check it is the released file."""
    path = Path(hf_hub_download(HF_REPO_ID, TEST_PARQUET, repo_type="dataset", revision=HF_REVISION))
    if path.stat().st_size != TEST_PARQUET_BYTES:
        raise ValueError(
            f"ASCEND test parquet size mismatch: expected {TEST_PARQUET_BYTES}, found {path.stat().st_size}"
        )
    if (checksum := audio_manifest.sha256_file(path)) != TEST_PARQUET_SHA256:
        raise ValueError(f"ASCEND test parquet SHA-256 mismatch: expected {TEST_PARQUET_SHA256}, found {checksum}")
    return path


def read_rows(parquet_path: Path, max_samples: int | None = None):
    """Yield the test rows in file order (at most ``max_samples``), checking the released row count first."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    if parquet.metadata.num_rows != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} ASCEND test rows, found {parquet.metadata.num_rows}")
    remaining = EXPECTED_ROWS if max_samples is None else max_samples
    for batch in parquet.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            if remaining <= 0:
                return
            remaining -= 1
            yield row


def format_source_row(row: dict, index: int, audio_dir: str) -> dict:
    """Build the source row of one utterance; ``index`` is its 0-based position in the split."""
    sample_id = str(row["id"] if row["id"] is not None else f"ascend-test-{index}").strip()
    sample_id = sample_id.replace("/", "_").replace(":", "_")
    duration = float(row["duration"] or 0.0)
    if duration <= 0:
        raise ValueError(f"ASCEND row {sample_id} has no duration")
    audio_path = f"{audio_dir}/{sample_id}.wav"
    return {
        "id": sample_id,
        "expected_answer": str(row["transcription"]),
        "audio_path": audio_path,
        "duration": duration,
        "messages": [{"role": "user", "content": PROMPT, "audio": {"path": audio_path, "duration": duration}}],
        "subset_for_metrics": SPLIT,
        "task_type": "Code-Switched-ASR",
        "extra_fields": {
            "language": row["language"],
            "speaker_id": row["original_speaker_id"],
            "session_id": row["session_id"],
            "topic": row["topic"],
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "source_split": SPLIT,
        },
    }


def write_audio(audio: dict, path: Path) -> None:
    """Decode a parquet audio cell (mono float32, channels averaged) and write it atomically as a WAV file."""
    import soundfile as sf

    samples, sampling_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    samples = np.asarray(samples)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    staged = path.with_name(path.name + ".tmp")
    sf.write(staged, samples, int(sampling_rate), format="WAV")
    staged.replace(path)


def check_audio(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as stream:
            frames = stream.getnframes()
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"ASCEND audio is missing or not a WAV file: {path}") from error
    if frames == 0:
        raise ValueError(f"ASCEND audio is empty: {path}")


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(*, audio_root: str | None = None, max_samples_per_profile: int | None = None) -> Path:
    """Download ASCEND and write the benchmark JSONL.

    Args:
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N rows (smoke runs).
    """
    if max_samples_per_profile is not None and (
        isinstance(max_samples_per_profile, bool)
        or not isinstance(max_samples_per_profile, int)
        or max_samples_per_profile < 1
    ):
        raise ValueError(f"max_samples_per_profile must be a positive integer, got {max_samples_per_profile!r}")
    audio_dir = DATA_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    row_audio_dir = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"

    sources = []
    for index, row in enumerate(read_rows(download_test_parquet(), max_samples_per_profile)):
        source = format_source_row(row, index, row_audio_dir)
        local_path = audio_dir / f"{source['id']}.wav"
        if not local_path.is_file():
            write_audio(row["audio"], local_path)
        check_audio(local_path)
        sources.append(source)

    prepared = audio_manifest.profile_rows(PROFILES, sources)
    options = {"audio_root": audio_root, "max_samples_per_profile": max_samples_per_profile}
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
