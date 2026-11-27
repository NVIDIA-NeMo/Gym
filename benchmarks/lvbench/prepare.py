# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare single-turn LVBench rows and their frozen cohort.

The questions are the authors' annotation, ``video_info.meta.jsonl`` of the Hugging Face dataset THUDM/LVBench
(1,549 questions over 103 long videos; not LongVideoBench) at a pinned revision, downloaded and checked by SHA-256;
$LVBENCH_ANNOTATIONS reads a local copy of that file instead, for a host without internet access. The videos are not
read or decoded: the source directory holds ``video/<key>.mp4`` and ``FILES.sha256``, their SHA-256 as ``sha256sum``
writes it (paths relative to the directory). ``video_mcqa/prepare.py``'s ``prepare_dataset`` builds the rows and the
cohort: each row sends the video as ``file://<video_url_dir>/<key>.mp4`` so the model server chooses the frames (for
example vLLM's ``--allowed-local-media-path`` and ``--media-io-kwargs``).

Each row asks in the benchmark's letter wording (prompt ``lvbench``), so a judge model reads its answer
(llm_extraction_v1). Task IDs are ``lvbench:<uid>``.
"""

import hashlib
import json
import os
from pathlib import Path

from resources_servers.video_mcqa.dataset import VideoSource
from resources_servers.video_mcqa.prepare import prepare_dataset


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "lvbench_benchmark.jsonl"
COHORT_FPATH = BENCHMARK_DIR / "data" / "lvbench_cohort.json"
REPO_ID = "THUDM/LVBench"
DATASET_REVISION = "0caedb92002cc268bad486449e551c76f0485670"  # pragma: allowlist secret
ANNOTATIONS_FILENAME = "video_info.meta.jsonl"
ANNOTATIONS_SHA256 = "a3042f766f568d457e2a91a30cdcf32a886c5421226eefa98a006ffaae038d8f"  # pragma: allowlist secret
NUM_VIDEOS = 103
NUM_QUESTIONS = 1549


def annotations_path() -> Path:
    """The annotation file: $LVBENCH_ANNOTATIONS, or the pinned revision's from Hugging Face."""
    if os.environ.get("LVBENCH_ANNOTATIONS"):
        return Path(os.environ["LVBENCH_ANNOTATIONS"])
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, ANNOTATIONS_FILENAME, repo_type="dataset", revision=DATASET_REVISION))


def load_records(path: Path) -> list[dict]:
    """The annotation's per-video records, in file order, after checking the file's SHA-256."""
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != ANNOTATIONS_SHA256:
        raise ValueError(f"{path} has SHA-256 {sha256}, expected {ANNOTATIONS_SHA256} ({REPO_ID}@{DATASET_REVISION})")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != NUM_VIDEOS:
        raise ValueError(f"{path} has {len(records)} videos, expected {NUM_VIDEOS}")
    questions = sum(len(record.get("qa", ())) for record in records)
    if questions != NUM_QUESTIONS:
        raise ValueError(f"{path} has {questions} questions, expected {NUM_QUESTIONS}")
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
    for video in dict.fromkeys(str(record["key"]) for record in records):
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
        raise ValueError("Set lvbench_source_dir to the directory that holds FILES.sha256 and video/")
    source = Path(source_dir).resolve()
    url_dir = video_url_dir if video_url_dir is not None else str(source / "video")
    if not Path(url_dir).is_absolute():
        raise ValueError("video_url_dir must be an absolute path")
    checksums = read_checksums(source / "FILES.sha256")
    records = load_records(annotations_path())
    rows, cohort = prepare_dataset(
        records,
        benchmark="lvbench",
        dataset_revision=DATASET_REVISION,
        media=media_bindings(records, checksums=checksums, video_url_dir=url_dir),
        setting="single_turn",
        answer_policy="llm_extraction_v1",
        prompt="lvbench",
    )
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FPATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    COHORT_FPATH.write_text(cohort.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return OUTPUT_FPATH
