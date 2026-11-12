# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare chemcotbench directly from pinned upstream sources."""

from collections.abc import Iterator
from pathlib import Path

from benchmarks.chemcotbench.data_utils import materialize, write_rows
from resources_servers.chemcotbench.task_data import TaskData


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "test.jsonl"


def build_rows() -> Iterator[dict[str, object]]:
    import json
    import subprocess
    from tempfile import TemporaryDirectory

    from resources_servers.chemcotbench.setup_molopt import ensure_molopt_runtime
    from resources_servers.chemcotbench.setup_upstream import ensure_data, ensure_repository

    repo = ensure_repository(None)
    data = ensure_data(None)
    python, oracle_dir = ensure_molopt_runtime()
    # All upstream chemistry imports run in the isolated, pinned Python 3.11 runtime.
    with TemporaryDirectory(prefix="chemcot-prepare-") as temporary:
        converted = Path(temporary) / "rows.jsonl"
        subprocess.run(
            [
                str(python),
                str(BENCHMARK_DIR / "prepare_worker.py"),
                "--repo",
                str(repo),
                "--data-dir",
                str(data),
                "--output",
                str(converted),
            ],
            cwd=oracle_dir,
            check=True,
            timeout=3600,
        )
        with converted.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def prepare(input_path: str | Path | None = None, output_path: str | Path = OUTPUT_FPATH) -> Path:
    """Download and convert raw sources, or validate an explicitly supplied Gym JSONL."""
    options = dict(task_schema=TaskData, identity_key="id", benchmark="chemcotbench")
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
