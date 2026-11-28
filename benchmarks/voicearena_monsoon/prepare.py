# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the public VoiceArena Monsoon test splits (Indian English and Hindi) for Gym.

Each profile reads one pinned Hugging Face dataset revision. Only the clips a run needs are fetched (parquet row
groups are read on demand), converted to 16 kHz mono 16-bit WAV and written to ``data/<language>/public/audio``.
The speaker-disjoint private halves and the separately licensed hosted VoiceArena corpus are not part of this
benchmark.
"""

from __future__ import annotations

import io
import os
import re
import wave
from math import gcd
from pathlib import Path
from typing import NamedTuple

import numpy as np
from huggingface_hub import HfFileSystem
from scipy.signal import resample_poly

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "voicearena_monsoon_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

SAMPLE_RATE = 16000
MIN_SAMPLES = 320
# The prompt of the source row; each profile's prompt_override replaces it in the request.
PROMPT = "Transcribe the following:"
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")


class Source(NamedTuple):
    language: str  # profile language, also the directory name under data/
    source_language: str  # value of the dataset's ``language`` column
    repo: str
    revision: str
    expected_rows: int
    expected_speakers: int
    reference_field: str  # "text" (plain reference) or "lattice" (accepted spelling variants per span)


SOURCES = {
    "voicearena-monsoon.en-in.public": Source(
        "en-in",
        "en_in",
        "VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN",
        "bc1da7b42ef6e2853123c97bf6d22067e4802d11",
        2102,
        1444,
        "text",
    ),
    "voicearena-monsoon.hi-in.public": Source(
        "hi-in",
        "hi_in",
        "VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN",
        "a4b7375e9adea16c16100295976cdb0aabd9fb6d",
        753,
        468,
        "lattice",
    ),
}


def _shards(filesystem: HfFileSystem, source: Source) -> list[str]:
    shards = sorted(filesystem.glob(f"datasets/{source.repo}@{source.revision}/data/test-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {source.repo}@{source.revision}")
    return shards


def _check_lattice(value: list, number: int) -> list[list[str]]:
    lattice = []
    for slot in value or []:
        variants = [variant.strip() for variant in slot or [] if variant and variant.strip()]
        if not variants or len(variants) != len(slot):
            raise ValueError(f"Monsoon hi-in row {number} has an invalid lattice slot: {slot!r}")
        lattice.append(variants)
    if not lattice:
        raise ValueError(f"Monsoon hi-in row {number} has an empty lattice")
    return lattice


def load_rows(source: Source, max_samples: int | None = None) -> list[dict]:
    """Read the pinned split's metadata, check it against the released census, then keep the first rows."""
    import pyarrow.parquet as pq

    columns = ["id", source.reference_field, "speaker_id", "language", "audio_length_s"]
    filesystem = HfFileSystem()
    rows = []
    for shard in _shards(filesystem, source):
        with filesystem.open(shard, "rb") as stream:
            table = pq.ParquetFile(stream).read(columns=columns).to_pydict()
        rows.extend(dict(zip(table, values)) for values in zip(*table.values()))
    if len(rows) != source.expected_rows:
        raise ValueError(
            f"Monsoon {source.language} row-count drift: expected {source.expected_rows}, got {len(rows)}"
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Monsoon {source.language} has duplicate ids")
    if len({row["speaker_id"] for row in rows}) != source.expected_speakers:
        raise ValueError(f"Monsoon {source.language} speaker-count drift: expected {source.expected_speakers}")
    for number, row in enumerate(rows, 1):
        if _SAFE_ID.fullmatch(row["id"]) is None or row["language"] != source.source_language:
            raise ValueError(f"Monsoon {source.language} row {number} has an unexpected id or language")
        if not row["audio_length_s"] > 0:
            raise ValueError(f"Monsoon {source.language} row {number} has an invalid duration")
        if source.reference_field == "text":
            if not row["text"].strip():
                raise ValueError(f"Monsoon {source.language} row {number} has an empty reference")
        else:
            row["lattice"] = _check_lattice(row["lattice"], number)
    return rows[:max_samples]


def format_source_row(row: dict, profile: str, source: Source, audio_prefix: str) -> dict:
    audio_path = f"{audio_prefix}/{source.language}/public/audio/{row['id']}.wav"
    duration = float(row["audio_length_s"])
    lattice = row.get("lattice")
    expected = " ".join(slot[0] for slot in lattice) if lattice else row["text"].strip()
    source_row = {
        "id": f"voicearena-monsoon-{profile.removeprefix('voicearena-monsoon.')}-{row['id']}",
        "messages": [{"role": "user", "content": PROMPT, "audio": {"path": audio_path, "duration": duration}}],
        "expected_answer": expected,
        "audio_filepath": audio_path,
        "audio_duration": duration,
        "monsoon_language": source.language,
        "speaker_id": row["speaker_id"],
        "extra_fields": {
            "source_repo": source.repo,
            "source_revision": source.revision,
            "source_split": "test",
            "source_license": "CC-BY-4.0",
            "source_id": row["id"],
        },
    }
    if lattice:
        source_row["reference_lattice"] = lattice
    return source_row


def canonical_samples(encoded: bytes) -> np.ndarray:
    """Decode one clip to mono float32 at 16 kHz (deterministic Kaiser-windowed polyphase resampling)."""
    import soundfile as sf

    try:
        samples, rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=False)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Monsoon raw audio could not be decoded by libsndfile") from error
    if samples.ndim == 2 and samples.shape[1] == 1:
        samples = samples[:, 0]
    if samples.ndim != 1 or samples.size == 0 or rate <= 0 or not np.isfinite(samples).all():
        raise ValueError(f"Monsoon source audio must be finite nonempty mono; got shape={samples.shape}, rate={rate}")
    expected = samples.size
    if rate != SAMPLE_RATE:
        divisor = gcd(rate, SAMPLE_RATE)
        samples = resample_poly(
            samples, up=SAMPLE_RATE // divisor, down=rate // divisor, window=("kaiser", 5.0), padtype="constant"
        )
        expected = -(-expected * SAMPLE_RATE // rate)  # ceil
    samples = np.asarray(samples, dtype=np.float32)
    if samples.shape != (expected,) or samples.size < MIN_SAMPLES or not np.isfinite(samples).all():
        raise ValueError(f"Monsoon canonical audio is invalid: shape={samples.shape}, expected {expected} frames")
    return samples


def write_wav(path: Path, samples: np.ndarray) -> None:
    import soundfile as sf

    staged = path.with_name(path.name + ".tmp")
    sf.write(staged, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    os.replace(staged, path)


def extract_audio(source: Source, rows: list[dict], audio_dir: Path) -> None:
    """Write the wanted clips to ``audio_dir``, reading only the parquet row groups that hold them."""
    import pyarrow.parquet as pq

    audio_dir.mkdir(parents=True, exist_ok=True)
    pending = {row["id"] for row in rows if not (audio_dir / f"{row['id']}.wav").is_file()}
    filesystem = HfFileSystem()
    for shard in _shards(filesystem, source):
        if not pending:
            break
        with filesystem.open(shard, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            ids = parquet.read(columns=["id"]).column("id").to_pylist()
            start = 0
            for group in range(parquet.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                wanted = [i for i in range(size) if ids[start + i] in pending]
                if wanted:
                    audio = parquet.read_row_group(group, columns=["audio"]).column("audio")
                    for i in wanted:
                        write_wav(audio_dir / f"{ids[start + i]}.wav", canonical_samples(audio[i].as_py()["bytes"]))
                        pending.discard(ids[start + i])
                start += size
    if pending:
        raise FileNotFoundError(f"{len(pending)} Monsoon clips are missing from {source.repo}: {sorted(pending)[:10]}")


def check_audio(rows: list[dict], audio_dir: Path) -> None:
    for row in rows:
        path = audio_dir / f"{row['id']}.wav"
        try:
            with wave.open(str(path), "rb") as stream:
                header = (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes())
        except (OSError, EOFError, wave.Error) as error:
            raise ValueError(f"Monsoon audio is missing or not a WAV file: {path}") from error
        if header[:3] != (1, 2, SAMPLE_RATE) or header[3] < MIN_SAMPLES:
            raise ValueError(f"Monsoon audio contract failed for {path}: {header}")
        if abs(header[3] / SAMPLE_RATE - float(row["audio_length_s"])) > 0.11:
            raise ValueError(f"Monsoon audio duration drift for {path}: metadata={row['audio_length_s']}, {header}")


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(*, audio_root: str | None = None, max_samples_per_profile: int | None = None) -> Path:
    """Download the public Monsoon splits and write the combined benchmark JSONL.

    Args:
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N rows of each profile (smoke runs).
    """
    if max_samples_per_profile is not None and (
        isinstance(max_samples_per_profile, bool)
        or not isinstance(max_samples_per_profile, int)
        or max_samples_per_profile < 1
    ):
        raise ValueError(
            f"max_samples_per_profile must be a positive integer or None, got {max_samples_per_profile!r}"
        )
    profiles = audio_manifest.load_profiles(PROFILES)["profiles"]
    rows = {name: load_rows(SOURCES[name], max_samples_per_profile) for name in profiles}
    for name, profile_rows in rows.items():
        audio_dir = DATA_DIR / SOURCES[name].language / "public" / "audio"
        extract_audio(SOURCES[name], profile_rows, audio_dir)
        check_audio(profile_rows, audio_dir)

    prefix = audio_manifest.audio_prefix(DATA_DIR, audio_root)
    prepared = audio_manifest.profile_rows(
        PROFILES,
        {name: [format_source_row(row, name, SOURCES[name], prefix) for row in rows[name]] for name in profiles},
    )
    options = {"audio_root": audio_root, "max_samples_per_profile": max_samples_per_profile}
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
