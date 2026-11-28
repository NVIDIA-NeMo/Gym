# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMAR (multiple-choice audio reasoning) for Gym.

The rows are the 1,000 test questions of the pinned ``BoJack/MMAR`` revision (MMAR publishes every label and may
change its single test set in place, hence the pin). By default all seven modalities are prepared; ``modalities``
narrows the set, for example to ``SPEECH_MODALITIES`` (the 294 ``speech`` questions). The
matching clips are read from the dataset's audio archive, which is streamed rather than stored, and written
unchanged to ``data/audio``.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mmar_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

HF_REPO_ID = "BoJack/MMAR"
HF_REVISION = "3bd051123480e80d273ae9e8e9f1653f49010ac7"
# Revision of https://github.com/ddlBoJack/MMAR whose code/evaluation.py the scorer follows.
OFFICIAL_SCORER_REVISION = "3bce090a967db576c8ad433b290a8da582d6d2a0"
METADATA_FILENAME = "MMAR-meta.json"
AUDIO_ARCHIVE_FILENAME = "mmar-audio.tar.gz"
# Rows per ``modality`` value in the pinned metadata (1,000 in total).
EXPECTED_MODALITY_COUNTS = {
    "speech": 294,
    "sound": 165,
    "music": 206,
    "mix-sound-speech": 218,
    "mix-music-speech": 82,
    "mix-sound-music-speech": 24,
    "mix-sound-music": 11,
}
# The speech-only subset: 294 questions.
SPEECH_MODALITIES = ("speech",)
# The prompt of the source row; the profile's prompt_override only adds the system prompt.
ANSWER_INSTRUCTION = "Respond with only the text of the correct option."


def select_modalities(modalities: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize the ``modalities`` argument (a list or a comma-separated string; default all) to known values."""
    return audio_manifest.select(modalities, EXPECTED_MODALITY_COUNTS, "MMAR modalities", keep_order=True)


def load_mmar_rows(modalities: tuple[str, ...] | None = None, max_samples: int | None = None) -> list[dict]:
    """Read the pinned metadata, check its per-modality census, keep the selected modalities (file order)."""
    path = hf_hub_download(HF_REPO_ID, METADATA_FILENAME, repo_type="dataset", revision=HF_REVISION)
    with open(path, encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in {METADATA_FILENAME}, got {type(rows).__name__}")
    counts = Counter(row.get("modality") for row in rows)
    if counts != EXPECTED_MODALITY_COUNTS:
        raise ValueError(f"MMAR modality census drift: expected {EXPECTED_MODALITY_COUNTS}, found {dict(counts)}")
    selected = select_modalities(modalities)
    return [dict(row) for row in rows if row["modality"] in selected][:max_samples]


def _audio_filename(row: dict) -> str:
    name = Path(str(row["audio_path"])).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid MMAR audio_path: {row['audio_path']!r}")
    return name


def extract_audio(rows: list[dict], audio_dir: Path) -> None:
    """Write the wanted clips to ``audio_dir``, streaming the archive and stopping once all are found.

    The archive (about 3 GB) is read from its start on every call that has clips to fetch, so all 1,000 clips
    cost one full pass; clips already on disk are skipped, which makes an interrupted run cheap to repeat.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    missing = {name for name in map(_audio_filename, rows) if not (audio_dir / name).is_file()}
    if not missing:
        return

    found: set[str] = set()
    archive_path = f"datasets/{HF_REPO_ID}@{HF_REVISION}/{AUDIO_ARCHIVE_FILENAME}"
    with HfFileSystem().open(archive_path, "rb") as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or name not in missing or name in found:
                continue
            destination = audio_dir / name
            staged = destination.with_name(destination.name + ".tmp")
            with archive.extractfile(member) as source, staged.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            staged.replace(destination)
            found.add(name)
            if found == missing:
                break
    if missing - found:
        unresolved = sorted(missing - found)
        raise FileNotFoundError(f"MMAR archive is missing {len(unresolved)} clips; first 10: {unresolved[:10]}")


def _seconds(value: str) -> float:
    hours, minutes, seconds = value.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _duration_seconds(timestamp: object) -> float | None:
    """Clip length from the ``"HH:MM:SS,HH:MM:SS"`` timestamp of the source video, if it parses and is positive."""
    try:
        start, end = str(timestamp).split(",", maxsplit=1)
        duration = _seconds(end) - _seconds(start)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
    return (
        "Listen to the audio and answer the multiple-choice question.\n\n"
        f"Question: {question}\n\nOptions:\n{options}\n\n{ANSWER_INSTRUCTION}"
    )


def format_source_row(row: dict, audio_dir: str) -> dict:
    """Keep the official fields and add the audio, the neutral prompt, the answer and the metric subset."""
    choices = list(row["choices"])
    audio_path = f"{audio_dir}/{_audio_filename(row)}"
    audio = {"path": audio_path}
    if (duration := _duration_seconds(row.get("timestamp"))) is not None:
        audio["duration"] = duration
    return {
        **row,
        "source_audio_path": row["audio_path"],
        "audio_path": audio_path,
        "audio_filepath": audio_path,
        "messages": [{"role": "user", "content": _prompt(row["question"], choices), "audio": audio}],
        "expected_answer": row["answer"],
        "choices": choices,
        "subset_for_metrics": row["category"],
        "dataset_revision": HF_REVISION,
        "official_scorer_revision": OFFICIAL_SCORER_REVISION,
    }


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    modalities: str | list[str] | tuple[str, ...] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
) -> Path:
    """Download the MMAR questions and clips and write the combined benchmark JSONL.

    Args:
        modalities: ``modality`` values to keep, as a list or a comma-separated string (default: all seven).
            ``SPEECH_MODALITIES`` keeps the 294 speech-only questions.
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N rows (smoke runs).
    """
    selected = select_modalities(modalities)
    rows = load_mmar_rows(selected, max_samples_per_profile)
    extract_audio(rows, DATA_DIR / "audio")

    audio_dir = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"
    prepared = audio_manifest.profile_rows(PROFILES, [format_source_row(row, audio_dir) for row in rows])
    options = {
        "modalities": None if set(selected) == set(EXPECTED_MODALITY_COUNTS) else list(selected),
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
