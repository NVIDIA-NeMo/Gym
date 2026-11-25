# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the Numb3rs TN/ITN benchmark for NeMo Gym.

Numb3rs (`nvidia/Numb3rs`) is a TN/ITN ASR benchmark: each clip has a written reference (``text_tn``, e.g.
``"$100"``) and a spoken reference (``text_itn``, e.g. ``"one hundred dollars"``). The transcripts come from the
pinned ``manifest.jsonl`` (the dataset's default Hub configuration exposes only audio and a category label) and the
audio from the same dataset revision. Only data files are downloaded; no dataset code is run.

One combined JSONL holds three profiles over the same clips, told apart by ``benchmark_profile``:

* ``numb3rs``: neutral prompt, scored against ``text_itn``;
* ``numb3rs.tn``: written-form prompt, scored against ``text_tn``;
* ``numb3rs.itn``: spoken-form prompt, scored against ``text_itn``.

A row carries its own prompt text in ``prompt``; ``prompts/default.yaml`` (the dataset's ``prompt_config``) turns it
into the chat messages at rollout time. Every row also carries the other form as a diagnostic reference
(``reference_fields``), and ``normalization: numb3rs`` selects the category-based normalizer of ``asr_with_pc``.
"""

import json
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download, snapshot_download

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "numb3rs_benchmark.jsonl"
CODE_FILES = (Path(__file__), Path(audio_manifest.__file__))

HF_REPO_ID = "nvidia/Numb3rs"
HF_REVISION = "192908075e1bd293914cc7b508f4a183ba6ef2b8"  # pragma: allowlist secret
MANIFEST_FILENAME = "manifest.jsonl"
EXPECTED_ROWS = 10131

PROFILES = {
    "numb3rs": {"prompt": "Transcribe the audio file into English text.", "reference": "text_itn"},
    "numb3rs.tn": {
        "prompt": "Transcribe the audio into written form with numbers as digits (e.g., '$100', '3.14', 'Jan 1').",
        "reference": "text_tn",
    },
    "numb3rs.itn": {
        "prompt": "Transcribe the audio into spoken form with numbers spelled out "
        "(e.g., 'one hundred dollars', 'three point one four', 'january first').",
        "reference": "text_itn",
    },
}


def _relative(row: dict) -> str:
    """The clip path inside the dataset repository (the manifest writes it as ``./CATEGORY/name.wav``)."""
    return Path(row["audio_filepath"]).as_posix()


def load_manifest(max_samples: int | None = None) -> list[dict]:
    """Read the pinned manifest, check its census and keep the first ``max_samples`` rows."""
    path = hf_hub_download(HF_REPO_ID, MANIFEST_FILENAME, repo_type="dataset", revision=HF_REVISION)
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Numb3rs manifest row-count drift: expected {EXPECTED_ROWS}, found {len(rows)}")
    for index, row in enumerate(rows):
        relative = Path(row["audio_filepath"])
        category = row["info"].upper()
        if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0].upper() != category:
            raise ValueError(f"Numb3rs manifest row {index} has an unexpected audio path: {row['audio_filepath']!r}")
        if not row["original_text"].strip() or not row["text"].strip() or float(row["duration"]) <= 0:
            raise ValueError(f"Numb3rs manifest row {index} is not usable")
    # Category by category, each in manifest order; a row cap therefore keeps the first categories only.
    return sorted(rows, key=lambda row: row["info"].upper())[:max_samples]


def download_audio(rows: list[dict]) -> Path:
    """Download exactly the referenced clips into ``data/source`` and return that directory."""
    source = DATA_DIR / "source"
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=HF_REVISION,
        allow_patterns=sorted({_relative(row) for row in rows}),
        local_dir=source,
    )
    return source


def iter_rows(rows: list[dict], audio_dir: str, profiles: list[str]) -> Iterator[dict]:
    """The benchmark rows of ``profiles`` (in that order) over manifest ``rows`` whose clips are under ``audio_dir``."""
    for profile in profiles:
        spec = PROFILES[profile]
        for row in rows:
            category = row["info"].upper()
            tn, itn = row["original_text"].strip(), row["text"].strip()
            references = {"text_tn": tn, "text_itn": itn}
            yield {
                "responses_create_params": {"metadata": {"audio_path": f"{audio_dir}/{_relative(row)}"}},
                "prompt": spec["prompt"],
                "expected_answer": references[spec["reference"]],
                "text_tn": tn,
                "text_itn": itn,
                "audio_duration": float(row["duration"]),
                "category": category,
                "sample_id": f"{profile}:{Path(row['audio_filepath']).stem}",
                "subset_for_metrics": f"{profile}_{category}",
                "task_type": "ASR_LEADERBOARD",
                "reference_fields": ["text_tn", "text_itn"],
                "normalization": "numb3rs",
                "benchmark_profile": profile,
                "extra_fields": {"source_repo": HF_REPO_ID, "source_revision": HF_REVISION},
            }


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    audio_root: str | None = None,
    profiles: str | list[str] | tuple[str, ...] | None = None,
    max_samples_per_profile: int | None = None,
) -> Path:
    """Download Numb3rs and write the combined benchmark JSONL.

    Args:
        audio_root: Directory rows use for ``data/`` when the model server sees the data directory somewhere other
            than its local path (for example ``/data/numb3rs`` in a container that mounts it there). Default: the
            absolute local path.
        profiles: The profiles to write, as a list or a comma-separated string. Default: all of them.
        max_samples_per_profile: Keep only the first N clips, in category order (smoke runs).
    """
    if max_samples_per_profile is not None and (
        isinstance(max_samples_per_profile, bool)
        or not isinstance(max_samples_per_profile, int)
        or max_samples_per_profile < 1
    ):
        raise ValueError(f"max_samples_per_profile must be a positive integer, got {max_samples_per_profile!r}")
    selected = audio_manifest.select(profiles or None, PROFILES, "Numb3rs profiles", keep_order=True)
    rows = load_manifest(max_samples_per_profile)
    source = download_audio(rows)
    for row in rows:
        if not (source / _relative(row)).is_file():
            raise FileNotFoundError(f"Numb3rs audio was not downloaded: {_relative(row)}")
    audio_dir = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/source"
    options = {"audio_root": audio_root, "profiles": profiles, "max_samples_per_profile": max_samples_per_profile}
    return audio_manifest.write_rows(
        OUTPUT_FPATH, iter_rows(rows, audio_dir, selected), code_files=CODE_FILES, **options
    )
