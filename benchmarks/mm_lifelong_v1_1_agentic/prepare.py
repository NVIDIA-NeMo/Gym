# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare agentic MM-Lifelong v1.1 rows: the 1,023 questions of day_test, week_test and month_val, each answered by
an agent in a sandbox that holds the original videos of the question's split.

The questions are the single-turn benchmark's (benchmarks/mm_lifelong_v1_1: the annotations at a pinned revision,
downloaded into ``data_dir`` and checked against the release's checksums). The videos are too large to upload for
every episode (8.1 GiB for the day, 95.7 GiB in 6,266 clips for the week, 85.3 GiB for the month), so each split's
videos live in a sandbox image that holds them, built with sandbox_image/build.py from the user's own copy of the
videos. ``media_dir`` is that script's output: for each domain, ``catalog.json`` (the image's catalog, whose SHA-256
each row pins) and ``video_catalog.json`` (the files as the model reads them: path, playable duration, and the
recording's place in time). Each row names its domain's image and catalog (``preloaded_media``) and has the seed copy
``video_catalog.json`` to the sandbox's media root.

The prompt is the dataset agentic runner's description of the inputs: VIDEO INPUTS (the catalog's records, or for
more than 100 files the clips per day folder and where the catalog is), then its TIME COORDINATES note, then the
question. The note is copied from src/eval_codex_multivideo.py of the dataset (revision
b8a87afff4e24e76f6333937fb7b1c77dd5036fc, MIT License, Copyright (c) 2026 MM-Lifelong contributors; see
benchmarks/mm_lifelong_v1_1/third-party-notices.md). The runner's other instructions (its rules, programs and answer
format) belong to its own harness and are left out, as is its renaming of the week clips.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from benchmarks.mm_lifelong_v1_1.prepare import SPLITS, TASK_PREFIX, interleave, load_release
from resources_servers.video_qa_rubric.task import prepare_agentic


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "mm_lifelong_v1_1_agentic_benchmark.jsonl"
# Where the images built by sandbox_image/build.py hold each domain's videos and their catalog.
MEDIA_ROOT = PurePosixPath("/testbed")
CATALOG = MEDIA_ROOT / "videos" / "catalog.json"
# Where the seed puts the catalog the model reads, relative to the media root.
MODEL_CATALOG = "video_catalog.json"
# Files per domain in the release.
VIDEO_FILES = {"day": 1, "week": 6266, "month": 23}
# The runner lists up to this many files in the prompt; for more it counts the clips of each day folder.
MAX_LISTED = 100
TIME_COORDINATES = """\
Month physical_start/physical_end are real livestream UTC calendar times copied
from the paper, including unobserved gaps between recordings. The year is not
specified by the paper table; do not invent one. Week videos have Day X labels
and the paper's time-of-day envelope for that day's recording session. A day's
envelope is not the start/end time of each short clip. Day uses the paper's
relative Day 1 timeline. duration_seconds is only the playable length of a file.
Physical calendar time and clip-local playback seconds are different coordinates."""


def image_name(domain: str) -> str:
    """The name a row gives its domain's image; the config maps it to an image reference."""
    return f"{TASK_PREFIX}_{domain}"


def video_inputs(records: list[dict[str, Any]]) -> str:
    """The VIDEO INPUTS block: one JSON line per file, or for many files the clips per day folder."""
    if len(records) <= MAX_LISTED:
        return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    folders = Counter((record["day"], str(PurePosixPath(record["path"]).parent)) for record in records)
    if len({day for day, _ in folders}) != len(folders):
        raise ValueError("Each day's clips must be in one folder")
    lines = [f"Day {day}: {count} clips under {folder}/" for (day, folder), count in sorted(folders.items())]
    lines.append(f"Read {MEDIA_ROOT / MODEL_CATALOG} for each individual path and exact recorded-time range.")
    return "\n".join(lines)


def prompt(question: str, inputs: str) -> str:
    return f"VIDEO INPUTS\n{inputs}\n\nTIME COORDINATES\n{TIME_COORDINATES}\n\nQUESTION\n{question}"


def media(media_dir: Path, domain: str) -> tuple[dict[str, Any], str]:
    """The domain's preloaded_media binding and its VIDEO INPUTS, after checking that both catalogs name the same files
    inside the image's media root."""
    folder = media_dir / domain
    catalog_bytes = (folder / "catalog.json").read_bytes()
    model_catalog = folder / MODEL_CATALOG
    model_bytes = model_catalog.read_bytes()
    files = [video["path"] for video in json.loads(catalog_bytes)["videos"]]
    records = json.loads(model_bytes)
    if len(files) != VIDEO_FILES[domain] or sorted(files) != sorted(record["path"] for record in records):
        raise ValueError(
            f"{folder}: catalog.json and {MODEL_CATALOG} must name the {VIDEO_FILES[domain]} {domain} files"
        )
    if not all(PurePosixPath(path).is_relative_to(MEDIA_ROOT / "videos" / domain) for path in files):
        raise ValueError(f"{folder}: the {domain} files must be under {MEDIA_ROOT / 'videos' / domain}")
    binding = {
        "image": image_name(domain),
        "catalog": str(CATALOG),
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "files": [
            {
                "source": str(model_catalog),
                "sandbox_path": MODEL_CATALOG,
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
            }
        ],
    }
    return binding, video_inputs(records)


def prepare(data_dir: str | None = None, media_dir: str | None = None) -> Path:
    """Write the agentic rows to OUTPUT_FPATH and return it.

    ``data_dir`` receives the annotation files (as for the single-turn benchmark; no frames are read). ``media_dir`` is
    sandbox_image/build.py's output for the images the run uses; the resources server reads its ``video_catalog.json``
    files, so it must be one of the server's video_roots.
    """
    if data_dir is None or media_dir is None:
        raise ValueError("Set mm_lifelong_v1_1_agentic_data_dir and mm_lifelong_v1_1_agentic_media_dir")
    _, _, questions = load_release(Path(data_dir).absolute())
    rows_by_split = []
    for split, items in questions.items():
        binding, inputs = media(Path(media_dir).absolute(), SPLITS[split][0])
        rows = []
        for item in items:
            row = dict(
                prepare_agentic(
                    f"{TASK_PREFIX}/{item['id']}",
                    item["question"],
                    item["rubric"],
                    text=prompt(item["question"], inputs),
                    grading="reference_levels",
                    reference_answer=item["answer"],
                    subset=split,
                )
            )
            row["preloaded_media"] = binding
            rows.append(row)
        rows_by_split.append(rows)
    rows = interleave(rows_by_split)
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FPATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return OUTPUT_FPATH
