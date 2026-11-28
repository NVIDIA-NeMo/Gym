# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Contextual Earnings-22 (test split) for Gym.

The 772 test clips of the pinned ``argmaxinc/contextual-earnings22`` revision are written as 16 kHz mono WAV files
to ``data/audio``. Each clip is used in three conditions (profiles): ``contextless`` (no context), ``local`` (the
clip's own keyword dictionary in the prompt) and ``global`` (the keyword list of the whole call, which may
include words absent from the clip). All three are prepared by default; ``modes`` narrows the set.
"""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "contextual_earnings22_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

HF_REPO_ID = "argmaxinc/contextual-earnings22"
HF_REVISION = "59d7f94771cc197984d8038e494c83abeff87282"  # pragma: allowlist secret
PARQUET_FILE = "data/test-00000-of-00001.parquet"
PARQUET_SHA256 = "4d43471525d8df72930629da0b3dfe20d03b5442f04fd7abedf9df9d301661ec"  # pragma: allowlist secret
SPLIT = "test"
EXPECTED_ROWS = 772
SAMPLING_RATE = 16000
MIN_SAMPLES = 320
PREPARATION_SCHEMA_VERSION = 2
DURATION_POLICY = "decoded_wav_frames_divided_by_16000"

MODES = ("contextless", "local", "global")
PROFILE_PREFIX = "contextual-earnings22."

PROMPT_CONTEXTLESS = "Transcribe the English audio into text, ensuring all punctuation marks are included."
PROMPT_LOCAL = (
    "This audio may contain the following words or phrases: {keyword_list}. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)
PROMPT_GLOBAL = (
    "This audio may contain the following words or phrases (some may not be present in the clip): "
    "{keyword_list}. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)


def select_modes(modes: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize the ``modes`` argument (a list or a comma-separated string; default all) to ``MODES`` order."""
    return audio_manifest.select(modes, MODES, "Contextual Earnings-22 modes")


def audio_filename(file_id: str, audio_start: float, index: int) -> str:
    """Unique clip name: ``file_id`` and start time identify a window, the dataset row index breaks ties."""
    return f"{file_id}__{audio_start:09.2f}__{index}.wav"


def decode_audio(encoded: bytes) -> np.ndarray:
    """Decode one embedded clip to mono float32; the source is already 16 kHz."""
    import soundfile as sf

    samples, rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=False)
    if samples.ndim == 2:
        if samples.shape[1] != 1:
            raise ValueError(f"Contextual Earnings-22 audio must be mono, found shape {samples.shape}")
        samples = samples[:, 0]
    if samples.ndim != 1 or samples.size < MIN_SAMPLES or not np.isfinite(samples).all():
        raise ValueError(f"Contextual Earnings-22 audio is empty, short or not finite: shape={samples.shape}")
    if int(rate) != SAMPLING_RATE:
        raise ValueError(f"Contextual Earnings-22 sample-rate drift: expected {SAMPLING_RATE}, found {rate}")
    return samples


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    staged = path.with_name(path.name + ".tmp")
    with wave.open(str(staged), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLING_RATE)
        stream.writeframes(pcm.tobytes())
    os.replace(staged, path)


def wav_frames(path: Path) -> int:
    """Frame count of a prepared clip, after checking its format."""
    try:
        with wave.open(str(path), "rb") as stream:
            header = (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"Contextual Earnings-22 audio is missing or not a WAV file: {path}") from error
    if header[:3] != (1, 2, SAMPLING_RATE) or header[3] < MIN_SAMPLES:
        raise ValueError(f"Contextual Earnings-22 audio contract failed for {path}: {header}")
    return header[3]


def load_samples(max_samples: int | None, audio_dir: Path) -> list[dict]:
    """Read the pinned test split, write the first ``max_samples`` clips, and return their metadata."""
    import pyarrow.parquet as pq

    path = Path(hf_hub_download(HF_REPO_ID, PARQUET_FILE, repo_type="dataset", revision=HF_REVISION))
    if (actual := audio_manifest.sha256_file(path)) != PARQUET_SHA256:
        raise ValueError(f"Contextual Earnings-22 parquet hash drift: expected {PARQUET_SHA256}, found {actual}")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Contextual Earnings-22 row-count drift: expected {EXPECTED_ROWS}, found {parquet.metadata.num_rows}"
        )
    audio_dir.mkdir(parents=True, exist_ok=True)
    columns = [name for name in parquet.schema_arrow.names if name != "transcript"]
    samples: list[dict] = []
    seen: set[str] = set()
    for batch in parquet.iter_batches(batch_size=16, columns=columns):
        for row in batch.to_pylist():
            index = len(samples)
            if index == max_samples:
                return samples
            audio_start = float(row["audio_start"])
            name = audio_filename(row["file_id"], audio_start, index)
            if name in seen:
                raise ValueError(f"Contextual Earnings-22 duplicate clip name: {name}")
            seen.add(name)
            wav = audio_dir / name
            if not wav.is_file():
                write_wav(wav, decode_audio(row["audio"]["bytes"]))
            frames = wav_frames(wav)
            source_duration = float(row["duration"])
            if not np.isfinite(source_duration) or source_duration <= 0:
                raise ValueError(f"Contextual Earnings-22 row {index} has an invalid duration")
            samples.append(
                {
                    "file_id": row["file_id"],
                    "audio_filename": name,
                    "audio_start": audio_start,
                    "audio_end": float(row["audio_end"]),
                    "duration": frames / SAMPLING_RATE,
                    "source_duration": source_duration,
                    "text": row["text"],
                    "norm_text": row["norm_text"],
                    "local_keywords": list(row["dictionary"]),
                    "global_keywords": list(row["keywords"]),
                    "source_row_id": f"{row['file_id']}:{audio_start:.2f}:{index}",
                }
            )
    return samples


def format_source_row(sample: dict, mode: str, audio_dir: str) -> dict:
    """Skills-style source row of one clip in one condition."""
    audio_path = f"{audio_dir}/{sample['audio_filename']}"
    local_keywords, global_keywords = sample["local_keywords"], sample["global_keywords"]
    if mode == "contextless":
        prompt = PROMPT_CONTEXTLESS
    elif mode == "local":
        prompt = PROMPT_LOCAL.format(keyword_list=", ".join(local_keywords))
    elif mode == "global":
        prompt = PROMPT_GLOBAL.format(keyword_list=", ".join(global_keywords))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return {
        "id": f"contextual-earnings22-{mode}-{sample['source_row_id']}",
        "messages": [
            {"role": "user", "content": prompt, "audio": {"path": audio_path, "duration": float(sample["duration"])}}
        ],
        "expected_answer": sample["norm_text"],
        "expected_answer_cased": sample["text"],
        "keyword_list": local_keywords,
        "local_keywords": local_keywords,
        "global_keywords": global_keywords,
        "subset_for_metrics": sample["file_id"],
        "file_id": sample["file_id"],
        "audio_start": sample["audio_start"],
        "duration": float(sample["duration"]),
        "source_duration": float(sample["source_duration"]),
        "duration_policy": DURATION_POLICY,
        "audio_filepath": audio_path,
        "source_provenance": {
            "repo_id": HF_REPO_ID,
            "revision": HF_REVISION,
            "split": SPLIT,
            "expected_rows": EXPECTED_ROWS,
            "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
            "duration_policy": DURATION_POLICY,
        },
    }


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    modes: str | list[str] | tuple[str, ...] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
) -> Path:
    """Download Contextual Earnings-22 and write the benchmark JSONL.

    Args:
        modes: Conditions to keep (``contextless``, ``local``, ``global``), as a list or a comma-separated
            string. Default: all three.
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N clips of each condition (smoke runs).
    """
    selected = select_modes(modes)
    if max_samples_per_profile is not None and max_samples_per_profile < 1:
        raise ValueError(f"max_samples_per_profile must be a positive integer, got {max_samples_per_profile}")
    samples = load_samples(max_samples_per_profile, DATA_DIR / "audio")
    audio_dir = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"
    prepared = audio_manifest.profile_rows(
        PROFILES,
        {
            PROFILE_PREFIX + mode: [format_source_row(sample, mode, audio_dir) for sample in samples]
            for mode in selected
        },
    )
    options = {
        "modes": None if selected == MODES else list(selected),
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
