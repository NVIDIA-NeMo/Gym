# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare single-turn Video-MME (w/o subtitles) rows and their frozen cohort.

The questions are the authors' annotation, ``videomme/test-00000-of-00001.parquet`` of the Hugging Face dataset
lmms-lab/Video-MME (2,700 questions over 900 videos) at a pinned revision, downloaded and checked by SHA-256;
$VIDEOMME_V1_PARQUET reads a local copy of that file instead, for a host without internet access. The videos are not
read or decoded: the source directory holds ``video/<videoID>.mp4`` and ``FILES.sha256``, their SHA-256 as
``sha256sum`` writes it (paths relative to the directory). ``video_mcqa/prepare.py``'s ``prepare_dataset`` builds the
rows and the cohort: each row sends the video as ``file://<video_url_dir>/<videoID>.mp4`` so the model server chooses
the frames (for example vLLM's ``--allowed-local-media-path`` and ``--media-io-kwargs``).

Each row asks in the benchmark's letter wording (prompt ``videomme_v1``), so a judge model reads its answer
(llm_extraction_v1). Task IDs are ``videomme_v1:<question_id>``.
"""

import hashlib
import json
import os
from pathlib import Path

from resources_servers.video_mcqa.dataset import VideoSource
from resources_servers.video_mcqa.prepare import prepare_dataset


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "videomme_v1_benchmark.jsonl"
COHORT_FPATH = BENCHMARK_DIR / "data" / "videomme_v1_cohort.json"
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
        if line.strip():
            digest, name = line.split(maxsplit=1)
            checksums[name.strip().removeprefix("./")] = digest
    return checksums


def media_bindings(records: list[dict], *, checksums: dict[str, str], video_url_dir: str) -> dict[str, VideoSource]:
    """Each annotated video as a ``file://`` URI under ``video_url_dir`` with its SHA-256 from ``FILES.sha256``."""
    base_dir = Path(video_url_dir)
    if not base_dir.is_absolute():
        raise ValueError("video_url_dir must be an absolute path")
    media = {}
    for video in dict.fromkeys(str(record["videoID"]) for record in records):
        relpath = f"video/{video}.mp4"
        sha256 = checksums.get(relpath)
        if sha256 is None:
            raise ValueError(f"FILES.sha256 has no SHA-256 for {relpath}")
        media[video] = VideoSource(
            source_url=relpath,
            sha256=sha256,
            video_url=(base_dir / f"{video}.mp4").as_uri(),
        )
    return media


def prepare(source_dir: str | None = None, video_url_dir: str | None = None) -> Path:
    """Write the single-turn rows to OUTPUT_FPATH and the frozen cohort to COHORT_FPATH; return OUTPUT_FPATH.

    ``source_dir`` holds ``FILES.sha256`` and ``video/``; ``video_url_dir`` is where the model server reads that
    ``video/`` directory (default ``<source_dir>/video``), so the rows' ``file://`` URLs point there.
    """
    if source_dir is None:
        raise ValueError("Set videomme_v1_source_dir to the directory that holds FILES.sha256 and video/")
    source = Path(source_dir).resolve()
    url_dir = video_url_dir if video_url_dir is not None else str(source / "video")
    if not Path(url_dir).is_absolute():
        raise ValueError("video_url_dir must be an absolute path")
    checksums = read_checksums(source / "FILES.sha256")
    records = load_records(parquet_path())
    rows, cohort = prepare_dataset(
        records,
        benchmark="videomme_v1",
        dataset_revision=DATASET_REVISION,
        media=media_bindings(records, checksums=checksums, video_url_dir=url_dir),
        setting="single_turn",
        answer_policy="llm_extraction_v1",
        prompt="videomme_v1",
    )
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FPATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    COHORT_FPATH.write_text(cohort.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return OUTPUT_FPATH
