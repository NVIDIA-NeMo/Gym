#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download stable, minimal LabBench2 question snapshots from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.labbench2_pdf.data_utils import (
    DEFAULT_QUESTIONS_DIR,
    QUESTION_FIELDS,
    SUPPORTED_BENCHMARKS,
    load_questions,
)


DEFAULT_HF_REPO_ID = "EdisonScientific/labbench2"
# The revision used by the standalone direct-PDF benchmark snapshot.
DEFAULT_HF_REVISION = "27d12d72af24e3f70db8a99df63e567366cbdb80"


@dataclass(frozen=True, slots=True)
class QuestionDownloadOptions:
    output_dir: Path = DEFAULT_QUESTIONS_DIR
    benchmarks: tuple[str, ...] = SUPPORTED_BENCHMARKS
    repo_id: str = DEFAULT_HF_REPO_ID
    revision: str | None = DEFAULT_HF_REVISION
    overwrite: bool = False


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _can_reuse(options: QuestionDownloadOptions, output_dir: Path) -> bool:
    manifest = _read_manifest(output_dir / "source_manifest.json")
    if manifest is None:
        return False
    if manifest.get("repo_id") != options.repo_id or manifest.get("revision") != options.revision:
        return False
    try:
        rows = load_questions(output_dir, options.benchmarks)
    except (FileNotFoundError, OSError, ValueError):
        return False
    expected = manifest.get("configs") or {}
    return all(expected.get(benchmark) == len(rows[benchmark]) for benchmark in options.benchmarks)


def _load_hf_rows(repo_id: str, benchmark: str, revision: str | None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - root Gym environment includes datasets
        raise RuntimeError("question download requires the Gym `datasets` dependency") from exc
    return load_dataset(repo_id, benchmark, split="train", revision=revision)


def _write_snapshot(path: Path, rows: Any) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for source_row in rows:
            missing = [field for field in QUESTION_FIELDS if field not in source_row]
            if missing:
                raise ValueError(f"Hugging Face row is missing fields {missing}")
            row = {field: source_row[field] for field in QUESTION_FIELDS}
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def download_questions(options: QuestionDownloadOptions) -> dict[str, Any]:
    output_dir = options.output_dir.expanduser().resolve()
    if len(set(options.benchmarks)) != len(options.benchmarks):
        raise ValueError("benchmarks must be unique")
    unsupported = sorted(set(options.benchmarks) - set(SUPPORTED_BENCHMARKS))
    if unsupported:
        raise ValueError(f"unsupported benchmarks: {unsupported}")
    if not options.overwrite and _can_reuse(options, output_dir):
        manifest = _read_manifest(output_dir / "source_manifest.json")
        assert manifest is not None
        print(f"Reusing question snapshot at {output_dir}", flush=True)
        return manifest

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".labbench2-questions-", dir=output_dir.parent))
    try:
        counts: dict[str, int] = {}
        for benchmark in options.benchmarks:
            print(f"[{benchmark}] Downloading questions from {options.repo_id}...", flush=True)
            rows = _load_hf_rows(options.repo_id, benchmark, options.revision)
            counts[benchmark] = _write_snapshot(temporary_dir / f"{benchmark}.jsonl", rows)

        # Validate all snapshots together before replacing a previously usable cache.
        load_questions(temporary_dir, options.benchmarks)
        manifest = {
            "schema_version": "labbench2_question_snapshot.v1",
            "repo_id": options.repo_id,
            "revision": options.revision,
            "exported_at": datetime.now(UTC).isoformat(),
            "fields": list(QUESTION_FIELDS),
            "configs": counts,
        }
        (temporary_dir / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for benchmark in options.benchmarks:
            os.replace(temporary_dir / f"{benchmark}.jsonl", output_dir / f"{benchmark}.jsonl")
        os.replace(temporary_dir / "source_manifest.json", output_dir / "source_manifest.json")
        return manifest
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_QUESTIONS_DIR)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=SUPPORTED_BENCHMARKS,
        default=list(SUPPORTED_BENCHMARKS),
    )
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = download_questions(
            QuestionDownloadOptions(
                output_dir=args.output_dir,
                benchmarks=tuple(args.benchmarks),
                repo_id=args.repo_id,
                revision=args.revision,
                overwrite=args.overwrite,
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Prepared {sum(manifest['configs'].values())} questions at {args.output_dir}")


if __name__ == "__main__":
    main()
