# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare HALAS (hallucination-annotated Earnings-22 test segments) for Gym.

The 745 test rows come from the pinned ``MatBar99/HALAS`` revision. Their audio is cut from the pinned
``distil-whisper/earnings22`` ``chunked`` test split: only the referenced clips are read (parquet row groups
are fetched on demand), converted to 16 kHz mono 16-bit WAV and written to ``data/audio``.
"""

from __future__ import annotations

import csv
import io
import wave
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
from huggingface_hub import HfFileSystem, hf_hub_download
from scipy.signal import resample_poly

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "halas_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

HF_REPO_ID = "MatBar99/HALAS"
HF_REVISION = "317cef3a10d1097edc37e8eb1c007f415e3d0c55"  # pragma: allowlist secret
EARNINGS22_REPO_ID = "distil-whisper/earnings22"
EARNINGS22_REVISION = "0a034f9ed86d33a3859d9025d3e621cf243773ab"  # pragma: allowlist secret
EXPECTED_ROWS = 745
EXPECTED_DUPLICATE_AUDIO_COUNTS = {"23_4484146.wav": 2}
SAMPLE_RATE = 16000
MIN_SAMPLES = 320
# The prompt of the source row; each profile's prompt_override replaces it in the request.
PROMPT = (
    "Transcribe the English audio into text. Return only the words spoken in the audio; "
    "do not add explanations or continue beyond the audible speech."
)


def _audio_id(row: dict) -> str:
    return Path(row["audio_id"].strip()).name


def _reference_text(row: dict) -> str:
    return row["corrected_reference_text"].strip() or row["e22_reference_text"].strip()


def load_halas_rows(max_samples: int | None = None) -> list[dict]:
    """Read the pinned HALAS test split, check it against the released census, then keep the first rows."""
    path = hf_hub_download(HF_REPO_ID, "test.csv", repo_type="dataset", revision=HF_REVISION)
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"HALAS test row-count drift: expected {EXPECTED_ROWS}, found {len(rows)}")
    audio_counts = Counter(_audio_id(row) for row in rows)
    duplicates = {audio_id: count for audio_id, count in audio_counts.items() if count > 1}
    if duplicates != EXPECTED_DUPLICATE_AUDIO_COUNTS:
        raise ValueError(f"HALAS audio census drift: expected {EXPECTED_DUPLICATE_AUDIO_COUNTS}, found {duplicates}")
    for index, row in enumerate(rows):
        if row["split"].strip().lower() != "test" or float(row["audio_duration"]) <= 0 or not _reference_text(row):
            raise ValueError(f"HALAS row {index} is not a usable test row")
        row["source_row_index"] = index
    return rows[:max_samples]


def _model_label_counts(row: dict) -> dict[str, int]:
    """Count the released systems' per-segment labels: hallucination or looping vs everything else."""
    counts = {"hallucination_or_looping": 0, "no_hallucination": 0}
    for key in (key for key in row if key.endswith("_label")):
        label = (row[key] or "").strip().lower()
        if not label:
            continue
        flagged = label != "no hallucination" and ("hallucination" in label or "looping" in label)
        counts["hallucination_or_looping" if flagged else "no_hallucination"] += 1
    return counts


def format_source_row(row: dict, audio_dir: str) -> dict:
    audio_id = _audio_id(row)
    audio_path = f"{audio_dir}/{audio_id}"
    duration = float(row["audio_duration"])
    counts = _model_label_counts(row)
    return {
        "id": f"halas-test-{row['source_row_index']:04d}-{audio_id}",
        "messages": [{"role": "user", "content": PROMPT, "audio": {"path": audio_path, "duration": duration}}],
        "expected_answer": _reference_text(row),
        "expected_answer_cased": row["e22_reference_text"].strip(),
        "audio_id": audio_id,
        "audio_duration": duration,
        "audio_filepath": audio_path,
        "source_row_index": row["source_row_index"],
        "subset_for_metrics": "test",
        "halas_released_system_hallucination_count": counts["hallucination_or_looping"],
        "halas_released_system_no_hallucination_count": counts["no_hallucination"],
        "extra_fields": {
            "source_repo": HF_REPO_ID,
            "source_revision": HF_REVISION,
            "source_split": "test",
            "expected_rows": EXPECTED_ROWS,
            "audio_source_repo": EARNINGS22_REPO_ID,
            "audio_source_revision": EARNINGS22_REVISION,
        },
        "task_type": "HALAS-ASR",
    }


def canonical_samples(encoded: bytes) -> np.ndarray:
    """Decode one clip to mono float32 at 16 kHz (deterministic Kaiser-windowed polyphase resampling)."""
    import soundfile as sf

    samples, rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or rate <= 0:
        raise ValueError(f"Earnings22 clip is empty or malformed: shape={samples.shape}, rate={rate}")
    if rate != SAMPLE_RATE:
        divisor = gcd(rate, SAMPLE_RATE)
        samples = resample_poly(
            samples, up=SAMPLE_RATE // divisor, down=rate // divisor, window=("kaiser", 5.0), padtype="constant"
        )
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < MIN_SAMPLES or not np.isfinite(samples).all():
        raise ValueError(f"Earnings22 canonical audio is too short or not finite: {samples.size} samples")
    return samples


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    staged = path.with_name(path.name + ".tmp")
    with wave.open(str(staged), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(pcm.tobytes())
    staged.replace(path)


def extract_audio(audio_ids: set[str], audio_dir: Path) -> None:
    """Write the wanted Earnings-22 clips to ``audio_dir``, reading only the parquet row groups that hold them."""
    import pyarrow.parquet as pq

    audio_dir.mkdir(parents=True, exist_ok=True)
    pending = {audio_id for audio_id in audio_ids if not (audio_dir / audio_id).is_file()}
    filesystem = HfFileSystem()
    shards = sorted(filesystem.glob(f"datasets/{EARNINGS22_REPO_ID}@{EARNINGS22_REVISION}/chunked/test-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No chunked test shards in {EARNINGS22_REPO_ID}@{EARNINGS22_REVISION}")
    for shard in shards:
        if not pending:
            break
        with filesystem.open(shard, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            ids = parquet.read(columns=["segment_id", "file_id"]).to_pydict()
            clip_ids = [f"{s.strip()}_{f.strip()}.wav" for s, f in zip(ids["segment_id"], ids["file_id"])]
            start = 0
            for group in range(parquet.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                wanted = [i for i in range(size) if clip_ids[start + i] in pending]
                if wanted:
                    audio = parquet.read_row_group(group, columns=["audio"]).column("audio")
                    for i in wanted:
                        clip_id = clip_ids[start + i]
                        write_wav(audio_dir / clip_id, canonical_samples(audio[i].as_py()["bytes"]))
                        pending.discard(clip_id)
                start += size
    if pending:
        raise FileNotFoundError(
            f"{len(pending)} HALAS clips are missing from Earnings22; first 10: {sorted(pending)[:10]}"
        )


def check_audio(audio_ids: set[str], audio_dir: Path) -> None:
    for audio_id in sorted(audio_ids):
        try:
            with wave.open(str(audio_dir / audio_id), "rb") as stream:
                header = (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes())
        except (OSError, EOFError, wave.Error) as error:
            raise ValueError(f"HALAS audio is missing or not a WAV file: {audio_dir / audio_id}") from error
        if header[:3] != (1, 2, SAMPLE_RATE) or header[3] < MIN_SAMPLES:
            raise ValueError(f"HALAS audio contract failed for {audio_id}: {header}")


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(*, audio_root: str | None = None, max_samples_per_profile: int | None = None) -> Path:
    """Download HALAS and its audio and write the combined benchmark JSONL.

    Args:
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N rows (smoke runs).
    """
    rows = load_halas_rows(max_samples_per_profile)
    audio_ids = {_audio_id(row) for row in rows}
    extract_audio(audio_ids, DATA_DIR / "audio")
    check_audio(audio_ids, DATA_DIR / "audio")

    audio_dir = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"
    prepared = audio_manifest.profile_rows(PROFILES, [format_source_row(row, audio_dir) for row in rows])
    options = {"audio_root": audio_root, "max_samples_per_profile": max_samples_per_profile}
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
