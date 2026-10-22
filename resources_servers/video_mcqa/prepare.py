# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare local evaluation annotations without selecting frames or calling a model."""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from resources_servers.video_mcqa.app import AccuracyCohort
from resources_servers.video_mcqa.dataset import (
    SOURCES,
    Benchmark,
    VideoSource,
    load_lvbench,
    load_videomme_v1,
)
from resources_servers.video_mcqa.scoring import AnswerPolicy
from resources_servers.video_mcqa.task import Setting, prepare_agentic, prepare_single_turn


def sha256_file(path: Path) -> str:
    """Hash original media and annotation bytes without loading them into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_annotations(path: Path) -> list[dict[str, object]]:
    """Load local JSON, JSONL or optional Parquet annotations; no network access."""
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ValueError("Parquet annotations require the optional pyarrow package") from error
        records = parquet.read_table(path).to_pylist()
    elif path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Annotations must use .json, .jsonl or .parquet")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Annotations must contain a list of record objects")
    return records


def prepare_dataset(
    records: list[dict[str, object]],
    *,
    benchmark: Benchmark,
    dataset_revision: str,
    media: Mapping[str, VideoSource],
    setting: Setting,
    answer_policy: AnswerPolicy = "boxed_only_v1",
    num_repeats: int = 1,
) -> tuple[list[dict[str, object]], AccuracyCohort]:
    """Build Gym rows and a frozen cohort for the single-turn or agentic media path.

    Single-turn requires an already-prepared VideoAnyRes video_url; it does not
    invent a fixed frame count or resize policy. Agentic requires an original
    source file and a safe sandbox-relative path. Gold, category labels and
    temporal evidence never enter responses_create_params.
    """
    if benchmark not in SOURCES or setting not in ("single_turn", "agentic"):
        raise ValueError("Unknown benchmark or setting")
    loader = load_videomme_v1 if benchmark == "videomme_v1" else load_lvbench
    annotations = loader(records, dataset_revision=dataset_revision, media=media)
    rows = []
    for annotation in annotations:
        question = annotation.question
        binding = media[question.video_id]
        if setting == "single_turn":
            if not binding.video_url:
                raise ValueError(f"Video {question.video_id!r} needs a prepared VideoAnyRes video_url")
            prepared = prepare_single_turn(
                question, expected_answer=annotation.expected_answer, video_url=binding.video_url
            )
            row = cast(dict[str, object], dict(prepared))
        else:
            path = PurePosixPath(binding.sandbox_path or "")
            if not binding.source or not path.parts or path.is_absolute() or ".." in path.parts:
                raise ValueError("Agentic media needs source and a safe sandbox-relative path")
            prepared = prepare_agentic(question, expected_answer=annotation.expected_answer, video_path=str(path))
            row = cast(dict[str, object], dict(prepared))
            row["video_binding"] = {"source": binding.source, "sandbox_path": str(path), "sha256": binding.sha256}
        if annotation.annotation_warnings:
            row["annotation_warnings"] = list(annotation.annotation_warnings)
        rows.append(row)
    cohort = AccuracyCohort(
        benchmark=benchmark,
        dataset_source=SOURCES[benchmark],
        dataset_revision=dataset_revision,
        setting=setting,
        answer_policy=answer_policy,
        task_ids=tuple(annotation.question.task_id for annotation in annotations),
        task_subsets={annotation.question.task_id: annotation.subsets for annotation in annotations},
        num_repeats=num_repeats,
    )
    return rows, cohort


def prepare_files(
    *,
    annotations: Path,
    media_manifest: Path,
    output_dir: Path,
    benchmark: Benchmark,
    dataset_revision: str,
    setting: Setting,
    answer_policy: AnswerPolicy = "boxed_only_v1",
    num_repeats: int = 1,
) -> dict[str, object]:
    """Validate complete inputs, verify bound source hashes, then write a fresh directory.

    Existing outputs are never overwritten. A caller can reuse a completed
    directory or choose a new directory for a changed annotation/media revision.
    Any partial write is visibly incomplete because receipt.json is written last.
    """
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output_dir}")
    payload = json.loads(media_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The media manifest must map original video IDs to explicit bindings")
    media = {identity: VideoSource.model_validate(value) for identity, value in payload.items()}
    rows, cohort = prepare_dataset(
        read_annotations(annotations),
        benchmark=benchmark,
        dataset_revision=dataset_revision,
        media=media,
        setting=setting,
        answer_policy=answer_policy,
        num_repeats=num_repeats,
    )
    checked: dict[Path, str] = {}
    for row in rows:
        metadata = cast(dict[str, object], row["verifier_metadata"])
        binding = media[cast(str, metadata["video_id"])]
        if binding.source is not None:
            path = Path(binding.source).resolve(strict=True)
            if path not in checked:
                checked[path] = sha256_file(path)
            if checked[path] != binding.sha256:
                raise ValueError(f"Source video SHA256 differs from its frozen binding: {path}")
            if setting == "agentic":
                cast(dict[str, object], row["video_binding"])["source"] = str(path)
    receipt: dict[str, object] = {
        "benchmark": benchmark,
        "dataset_source": SOURCES[benchmark],
        "dataset_revision": dataset_revision,
        "setting": setting,
        "questions": len(rows),
        "videos": len({row["verifier_metadata"]["video_id"] for row in rows}),
        "annotations_sha256": sha256_file(annotations),
        "media_manifest_sha256": sha256_file(media_manifest),
        "original_videos_hashed": len(checked),
        "audio": False,
        "subtitles": False,
        "evaluation_only": True,
        "annotation_warning_counts": dict(
            Counter(warning for row in rows for warning in row.get("annotation_warnings", []))
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "tasks.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "cohort.json").open("x", encoding="utf-8") as stream:
        stream.write(cohort.model_dump_json(indent=2) + "\n")
    receipt["tasks_sha256"] = sha256_file(output_dir / "tasks.jsonl")
    receipt["cohort_sha256"] = sha256_file(output_dir / "cohort.json")
    with (output_dir / "receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main() -> None:
    """CLI for user-provided, licensed local evaluation inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(SOURCES), required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--setting", choices=("single_turn", "agentic"), required=True)
    parser.add_argument("--answer-policy", choices=("boxed_only_v1", "boxed_or_letter_v1"), default="boxed_only_v1")
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(prepare_files(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
