# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare agentic Video-MME (w/o subtitles) rows and their frozen cohort.

The questions are the authors' annotation, ``videomme/test-00000-of-00001.parquet`` of the Hugging Face dataset
lmms-lab/Video-MME (2,700 questions over 900 videos) at a pinned revision, downloaded and checked by SHA-256;
$VIDEOMME_V1_PARQUET reads a local copy of that file instead, for a host without internet access. The videos are not
read: the source directory holds ``video/<videoID>.mp4`` and ``FILES.sha256``, their SHA-256 as ``sha256sum`` writes
it (paths relative to the directory). ``video_mcqa/prepare.py``'s ``prepare_dataset`` builds the rows and the cohort:
each row binds its video by path and by that SHA-256, and the resources server checks the hash in the sandbox before
the episode starts.

Each row asks in the benchmark's letter wording after one line that names the video in the sandbox (prompt
``videomme_v1``), so a judge model reads its answer (llm_extraction_v1). Task IDs are ``videomme_v1:<question_id>``.
"""

import hashlib
import json
import os
from pathlib import Path

from resources_servers.video_mcqa.dataset import VideoSource
from resources_servers.video_mcqa.prepare import prepare_dataset


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "videomme_v1_agentic_benchmark.jsonl"
COHORT_FPATH = BENCHMARK_DIR / "data" / "videomme_v1_agentic_cohort.json"
REPO_ID = "lmms-lab/Video-MME"
DATASET_REVISION = "ead1408f75b618502df9a1d8e0950166bf0a2a0b"  # pragma: allowlist secret
PARQUET_FILENAME = "videomme/test-00000-of-00001.parquet"
PARQUET_SHA256 = "7fffab8ed38ecc2f9f0eca4c44d8a11636f0ee96116ede83580c8b9ae0faf986"  # pragma: allowlist secret
NUM_QUESTIONS = 2700


def parquet_path() -> Path:
    """The annotation file: $VIDEOMME_V1_PARQUET, or the pinned revision's from Hugging Face."""
    if os.environ.get("VIDEOMME_V1_PARQUET"):
        return Path(os.environ["VIDEOMME_V1_PARQUET"])
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, PARQUET_FILENAME, repo_type="dataset", revision=DATASET_REVISION))


def load_records(path: Path) -> list[dict]:
    """The annotation's rows, in its order, after checking the file's SHA-256."""
    import pyarrow.parquet as pq

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != PARQUET_SHA256:
        raise ValueError(f"{path} has SHA-256 {sha256}, expected {PARQUET_SHA256} ({REPO_ID}@{DATASET_REVISION})")
    records = pq.read_table(path).to_pylist()
    if len(records) != NUM_QUESTIONS:
        raise ValueError(f"{path} has {len(records)} questions, expected {NUM_QUESTIONS}")
    return records


def read_checksums(path: Path) -> dict[str, str]:
    """``FILES.sha256`` lines (``<sha256>  [./]<path>``) by path."""
    checksums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name.removeprefix("./")] = digest
    return checksums


def media_bindings(records: list[dict], *, checksums: dict[str, str], video_dir: Path) -> dict[str, VideoSource]:
    """Each annotated video at ``videos/<videoID>.mp4`` in the sandbox, read from ``video_dir`` on this host."""
    media = {}
    for video in dict.fromkeys(str(record["videoID"]) for record in records):
        sha256 = checksums.get(f"video/{video}.mp4")
        if sha256 is None:
            raise ValueError(f"FILES.sha256 has no SHA-256 for video/{video}.mp4")
        sandbox_path = f"videos/{video}.mp4"
        media[video] = VideoSource(
            source_url=sandbox_path, sha256=sha256, source=str(video_dir / f"{video}.mp4"), sandbox_path=sandbox_path
        )
    return media


def prepare(source_dir: str | None = None, video_dir: str | None = None, sandbox_root: str = "/testbed") -> Path:
    """Write the agentic rows to OUTPUT_FPATH and the frozen cohort to COHORT_FPATH; return OUTPUT_FPATH.

    ``source_dir`` holds ``FILES.sha256`` and ``video/``; ``video_dir`` is where the resources server reads that
    ``video/`` directory (default ``<source_dir>/video``), so it must be one of the server's ``video_roots``.
    """
    if source_dir is None:
        raise ValueError("Set videomme_v1_agentic_source_dir to the directory that holds FILES.sha256 and video/")
    source = Path(source_dir)
    checksums = read_checksums(source / "FILES.sha256")
    records = load_records(parquet_path())
    videos = Path(video_dir) if video_dir is not None else source / "video"
    rows, cohort = prepare_dataset(
        records,
        benchmark="videomme_v1",
        dataset_revision=DATASET_REVISION,
        media=media_bindings(records, checksums=checksums, video_dir=videos),
        setting="agentic",
        answer_policy="llm_extraction_v1",
        prompt="videomme_v1",
        sandbox_root=sandbox_root,
    )
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FPATH.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    COHORT_FPATH.write_text(cohort.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return OUTPUT_FPATH
