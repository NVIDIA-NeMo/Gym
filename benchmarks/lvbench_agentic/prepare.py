# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare agentic LVBench rows and their frozen cohort.

The questions are the authors' annotation, ``video_info.meta.jsonl`` of the Hugging Face dataset THUDM/LVBench
(1,549 questions over 103 long videos; not LongVideoBench) at a pinned revision, downloaded and checked by SHA-256 via
``benchmarks/lvbench/prepare.py`` ($LVBENCH_ANNOTATIONS reads a local copy instead, for a host without internet
access). The videos are not read: the source directory holds ``video/<key>.mp4`` and ``FILES.sha256``, their SHA-256
as ``sha256sum`` writes it (paths relative to the directory). ``video_mcqa/prepare.py``'s ``prepare_dataset`` builds
the rows and the cohort: each row binds its video by path and by that SHA-256, and the resources server checks the hash
in the sandbox before the episode starts.

Each row asks in the benchmark's letter wording after one line that names the video in the sandbox (prompt ``lvbench``),
so a judge model reads its answer (llm_extraction_v1). Task IDs are ``lvbench:<uid>``.
"""

import json
from pathlib import Path

from benchmarks.lvbench.prepare import DATASET_REVISION, annotations_path, load_records, read_checksums
from resources_servers.video_mcqa.dataset import VideoSource
from resources_servers.video_mcqa.prepare import prepare_dataset


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "lvbench_agentic_benchmark.jsonl"
COHORT_FPATH = BENCHMARK_DIR / "data" / "lvbench_agentic_cohort.json"


def media_bindings(records: list[dict], *, checksums: dict[str, str], video_dir: Path) -> dict[str, VideoSource]:
    """Each annotated video at ``videos/<key>.mp4`` in the sandbox, read from ``video_dir`` on this host."""
    media = {}
    for video in dict.fromkeys(str(record["key"]) for record in records):
        relpath = f"video/{video}.mp4"
        sha256 = checksums.get(relpath)
        if sha256 is None:
            raise ValueError(f"FILES.sha256 has no SHA-256 for {relpath}")
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
        raise ValueError("Set lvbench_agentic_source_dir to the directory that holds FILES.sha256 and video/")
    source = Path(source_dir).resolve()
    checksums = read_checksums(source / "FILES.sha256")
    records = load_records(annotations_path())
    videos = (Path(video_dir) if video_dir is not None else source / "video").resolve()
    rows, cohort = prepare_dataset(
        records,
        benchmark="lvbench",
        dataset_revision=DATASET_REVISION,
        media=media_bindings(records, checksums=checksums, video_dir=videos),
        setting="agentic",
        answer_policy="llm_extraction_v1",
        prompt="lvbench",
        sandbox_root=sandbox_root,
    )
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FPATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    COHORT_FPATH.write_text(cohort.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return OUTPUT_FPATH
