# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare chembench directly from pinned upstream sources."""

from collections.abc import Iterator
from pathlib import Path

from benchmarks.chembench.data_utils import materialize, write_rows
from resources_servers.chembench.task_data import TaskData


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "test.jsonl"

DATA_REVISION = "6e1d25748952393f44e35b8e85bbe567246a6430"  # pragma: allowlist secret


def build_rows() -> Iterator[dict[str, object]]:
    from datasets import load_dataset

    from benchmarks.chembench.conversion import TOPICS, format_entry

    for topic in TOPICS:
        print(f"Downloading/preparing ChemBench: {topic}", flush=True)
        for source in load_dataset("jablonkagroup/ChemBench", topic, split="train", revision=DATA_REVISION):
            entry = format_entry(source, topic, use_cot=False)
            yield {
                "responses_create_params": {"input": [{"role": "user", "content": entry.pop("problem")}]},
                "verifier_metadata": entry,
            }


def prepare(input_path: str | Path | None = None, output_path: str | Path = OUTPUT_FPATH) -> Path:
    """Download and convert raw sources, or validate an explicitly supplied Gym JSONL."""
    options = dict(task_schema=TaskData, identity_key="uuid", benchmark="chembench")
    if input_path is not None:
        return materialize(Path(input_path), Path(output_path), **options)
    return write_rows(build_rows(), Path(output_path), **options)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional already converted Gym JSONL; default downloads raw data")
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH)
    args = parser.parse_args()
    print(prepare(args.input, args.output))
