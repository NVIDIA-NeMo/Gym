# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare HealthBench Professional for direct NeMo Gym evaluation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.healthbench.prepare_common import iter_local_records, write_gym_jsonl


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "healthbench_professional_benchmark.jsonl"

HF_DATASET = "openai/healthbench-professional"
DEFAULT_SOURCE_FILENAME = "healthbench_professional_eval.jsonl"
HF_REVISION = "349962fd46dd02343a0d8a606491baf59154ea1a"
BENCHMARK_NAME = "healthbench_professional"


def _iter_source_records(data_dir: str | Path, source_filename: str, revision: str) -> Iterator[dict[str, Any]]:
    local_records = iter_local_records(data_dir, source_filename)
    if local_records is not None:
        yield from local_records
        return

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            f"{data_dir!r} is not a local path. Install datasets or pass a local {source_filename}/assets.zip path."
        ) from error

    dataset = load_dataset(str(data_dir), split="test", revision=revision)
    for record in dataset:
        yield dict(record)


def prepare(
    data_dir: str | Path = HF_DATASET,
    output: Path = OUTPUT_FPATH,
    limit: int | None = None,
    *,
    revision: str = HF_REVISION,
) -> Path:
    return write_gym_jsonl(
        _iter_source_records(data_dir, DEFAULT_SOURCE_FILENAME, revision),
        output=output,
        benchmark_name=BENCHMARK_NAME,
        limit=limit,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        "-d",
        default=HF_DATASET,
        help="HF dataset id, local dataset directory, source JSONL, or assets.zip path.",
    )
    parser.add_argument("--output", "-o", type=Path, default=OUTPUT_FPATH)
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--revision", default=HF_REVISION, help="Pinned Hugging Face dataset revision")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(data_dir=args.data_dir, output=args.output, limit=args.limit, revision=args.revision)
