# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare SWE-bench Pro v2 data: HuggingFace metadata plus each task's pinned upstream verifier files."""

import json
import tarfile
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from datasets import load_dataset

from nemo_gym.config import get_hf_token


UPSTREAM_COMMIT = "66f92766bba642462d4bbe5479e83f91f9211862"  # pragma: allowlist secret
UPSTREAM_ARCHIVE_URL = f"https://github.com/scaleapi/SWE-bench_Pro-os/archive/{UPSTREAM_COMMIT}.tar.gz"
DATASET_REVISION = "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"  # pragma: allowlist secret
EXPECTED_INSTANCE_COUNT = 642
BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "swebench_pro_v2_benchmark.jsonl"
UPSTREAM_CACHE_DIR = DATA_DIR / "swebench_pro_v2_upstream"


def fetch_upstream_assets(cache_dir: Path = UPSTREAM_CACHE_DIR) -> Path:
    """Download and extract the pinned upstream repository once."""
    root = cache_dir / f"SWE-bench_Pro-os-{UPSTREAM_COMMIT}"
    if root.is_dir():
        return root

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"{UPSTREAM_COMMIT}.tar.gz"
    urllib.request.urlretrieve(UPSTREAM_ARCHIVE_URL, archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(cache_dir, filter="data")
    if not root.is_dir():
        raise FileNotFoundError(f"Expected extracted upstream directory at {root}")
    return root


def enrich_row(row: Mapping[str, Any], upstream_root: Path) -> dict[str, Any]:
    """Attach one task's upstream verifier files and its instruction to a public dataset row."""
    task_dir = upstream_root / "v2" / "tasks" / str(row["instance_id"])
    tests_dir = task_dir / "tests"
    test_config = (tests_dir / "config.json").read_text(encoding="utf-8")
    config = json.loads(test_config)
    enriched = dict(row)
    enriched.update(
        {
            # The verifier grades with config.json, so take the test lists from the same file.
            "fail_to_pass": config["fail_to_pass"],
            "pass_to_pass": config["pass_to_pass"],
            "selected_test_files_to_run": config["selected_test_files_to_run"],
            "test_config": test_config,
            "test_sh": (tests_dir / "test.sh").read_text(encoding="utf-8"),
            "run_script": (tests_dir / "run_script.sh").read_text(encoding="utf-8"),
            "parser_script": (tests_dir / "parser.py").read_text(encoding="utf-8"),
            "test_patch": (tests_dir / "test_patch.patch").read_text(encoding="utf-8"),
            "responses_create_params": {
                "input": [{"role": "user", "content": (task_dir / "instruction.md").read_text(encoding="utf-8")}],
            },
            "subset": "pro_v2",
            "split": "test",
            "evaluator_commit": UPSTREAM_COMMIT,
            "dataset_revision": DATASET_REVISION,
        }
    )
    return enriched


def prepare(
    dataset: Iterable[Mapping[str, Any]] | None = None,
    upstream_root: Path | None = None,
    output_fpath: Path = OUTPUT_FPATH,
) -> Path:
    """Materialize the SWE-bench Pro v2 split as self-contained NeMo Gym JSONL."""
    if dataset is None:
        dataset = load_dataset(
            "ScaleAI/SWE-bench_Pro",
            "default",
            split="test",
            revision=DATASET_REVISION,
            token=get_hf_token(),
        )
    if upstream_root is None:
        upstream_root = fetch_upstream_assets()

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_tmp_fpath = output_fpath.with_suffix(output_fpath.suffix + ".tmp")
    count = 0
    with output_tmp_fpath.open("w", encoding="utf-8") as output:
        for row in dataset:
            output.write(json.dumps(enrich_row(row, upstream_root)) + "\n")
            count += 1
    if count == 0:
        raise ValueError("SWE-bench Pro v2 preparation produced no rows")
    if hasattr(dataset, "num_rows") and count != EXPECTED_INSTANCE_COUNT:
        raise ValueError(f"Expected {EXPECTED_INSTANCE_COUNT} SWE-bench Pro v2 rows, got {count}")

    output_tmp_fpath.replace(output_fpath)
    print(f"Wrote {count} SWE-bench Pro v2 problems to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    prepare()
