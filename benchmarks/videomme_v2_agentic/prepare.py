# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare agentic Video-MME v2 rows and their frozen cohort.

The questions are the authors' annotation, ``test.parquet`` of the Hugging Face dataset MME-Benchmarks/Video-MME-v2
(3,200 questions over 800 videos) at a pinned revision, downloaded and checked by SHA-256; $VIDEOMME_V2_PARQUET reads
a local copy of that file instead, for a host without internet access. The videos are not read: the source directory
holds ``video/<video_id>.mp4`` and ``FILES.sha256``, their SHA-256 as ``sha256sum`` writes it (paths relative to the
directory). Each row binds its video by path and by that SHA-256, and the resources server checks the hash in the
sandbox before the episode starts.

Each row asks in the benchmark's letter wording after one line that names the video in the sandbox
(``prepare_agentic(..., prompt="videomme_v2")``), so a judge model reads its answer (llm_extraction_v1). Task IDs are
``videommev2-L<level>/<question_id>``. The cohort groups the four questions of each video in their official order
(the group's level is its fourth question's) and keeps every question's own level.
"""

import hashlib
import json
import os
from pathlib import Path

from resources_servers.video_mcqa.task import LABELS, VideoQuestion, prepare_agentic
from resources_servers.videomme_v2.app import CohortManifest


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "videomme_v2_agentic_benchmark.jsonl"
COHORT_FPATH = BENCHMARK_DIR / "data" / "videomme_v2_agentic_cohort.json"
DATASET_SOURCE = "MME-Benchmarks/Video-MME-v2"
DATASET_REVISION = "6e4bebb03202e1ddbf3d37703e560e51c5aa2d64"  # pragma: allowlist secret
PARQUET_FILENAME = "test.parquet"
PARQUET_SHA256 = "8dc7f8c8830aa49dd08a82592f8276899472a145155dde3bea5dd6914a65a9b4"  # pragma: allowlist secret
NUM_QUESTIONS = 3200


def parquet_path() -> Path:
    """The annotation file: $VIDEOMME_V2_PARQUET, or the pinned revision's from Hugging Face."""
    if os.environ.get("VIDEOMME_V2_PARQUET"):
        return Path(os.environ["VIDEOMME_V2_PARQUET"])
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATASET_SOURCE, PARQUET_FILENAME, repo_type="dataset", revision=DATASET_REVISION))


def load_records(path: Path) -> list[dict]:
    """The annotation's rows, in its order, after checking the file's SHA-256."""
    import pyarrow.parquet as pq

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != PARQUET_SHA256:
        raise ValueError(
            f"{path} has SHA-256 {sha256}, expected {PARQUET_SHA256} ({DATASET_SOURCE}@{DATASET_REVISION})"
        )
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


def parse_options(text: str) -> tuple[str, ...]:
    """The annotation's options, one ``<label>. <text>`` line each in label order, without their labels."""
    lines = text.split("\n")
    if len(lines) > len(LABELS) or any(not line.startswith(f"{label}. ") for label, line in zip(LABELS, lines)):
        raise ValueError(f"Options must be one '<label>. <text>' line per choice, labelled in order: {text!r}")
    return tuple(line[len("A. ") :] for line in lines)


def agentic_row(record: dict, *, checksums: dict[str, str], video_dir: Path, sandbox_root: str) -> dict:
    """One annotation row as an agentic row with its video binding."""
    level = int(record["level"])
    video = str(record["video_id"])
    sandbox_path = f"videos/{video}.mp4"
    sha256 = checksums.get(f"video/{video}.mp4")
    if sha256 is None:
        raise ValueError(f"FILES.sha256 has no SHA-256 for video/{video}.mp4")
    options = parse_options(record["options"])
    question = VideoQuestion(
        task_id=f"videommev2-L{level}/{record['question_id']}",
        dataset_source=DATASET_SOURCE,
        dataset_revision=DATASET_REVISION,
        video_id=video,
        video_source=sandbox_path,
        video_sha256=sha256,
        question=record["question"],
        options=options,
        level=level,
        group_id=video,
        # The annotation repeats an option text in a few questions; keep them, as the shared loader does.
        allow_duplicate_options=len({option.strip() for option in options}) != len(options),
    )
    row = dict(
        prepare_agentic(
            question,
            expected_answer=record["answer"],
            video_path=f"{sandbox_root}/{sandbox_path}",
            prompt="videomme_v2",
        )
    )
    row["video_binding"] = {"source": str(video_dir / f"{video}.mp4"), "sandbox_path": sandbox_path, "sha256": sha256}
    if question.allow_duplicate_options:
        row["annotation_warnings"] = ["duplicate_option_text"]
    return row


def build_cohort(records: list[dict], rows: list[dict]) -> CohortManifest:
    """Four questions per video in annotation order; a group's level is its fourth question's, as the official
    script's."""
    groups: dict[str, list[tuple[dict, dict]]] = {}
    for record, row in zip(records, rows):
        groups.setdefault(str(record["video_id"]), []).append((record, row))
    entries = []
    for video, members in groups.items():
        if [record["question_id"] for record, _ in members] != [f"{video}-{position}" for position in range(1, 5)]:
            raise ValueError(f"Video {video}: expected its four questions in order")
        last = members[-1][0]
        entries.append(
            {
                "group_id": video,
                "task_ids": [row["task_id"] for _, row in members],
                "group_type": last["group_type"],
                "level": int(last["level"]),
                "group_structure": last["group_structure"] if last["group_type"] == "logic" else None,
            }
        )
    return CohortManifest(
        schema_version=2,
        dataset_revision=DATASET_REVISION,
        setting="agentic",
        answer_policy="llm_extraction_v1",
        task_ids=tuple(row["task_id"] for row in rows),
        groups=entries,
        task_levels={row["task_id"]: row["verifier_metadata"]["level"] for row in rows},
    )


def prepare(source_dir: str | None = None, video_dir: str | None = None, sandbox_root: str = "/testbed") -> Path:
    """Write the agentic rows to OUTPUT_FPATH and the frozen cohort to COHORT_FPATH; return OUTPUT_FPATH.

    ``source_dir`` holds ``FILES.sha256`` and ``video/``; ``video_dir`` is where the resources server reads that
    ``video/`` directory (default ``<source_dir>/video``), so it must be one of the server's ``video_roots``.
    """
    if source_dir is None:
        raise ValueError("Set videomme_v2_agentic_source_dir to the directory that holds FILES.sha256 and video/")
    source = Path(source_dir)
    checksums = read_checksums(source / "FILES.sha256")
    records = load_records(parquet_path())
    videos = Path(video_dir) if video_dir is not None else source / "video"
    rows = [
        agentic_row(record, checksums=checksums, video_dir=videos, sandbox_root=sandbox_root) for record in records
    ]
    cohort = build_cohort(records, rows)
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FPATH.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    COHORT_FPATH.write_text(cohort.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return OUTPUT_FPATH
