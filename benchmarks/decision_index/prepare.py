# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the Decision Index 0.3 public rows.

The rows are converted from the Decision Index kit's frozen 0.3 suite by the benchmark owner's conversion script
(not in this repository). Some source datasets do not allow redistribution, so the rows come from two places:

- ``registry``: the benchmarks whose sources carry a permissive licence, from Gym's GitLab dataset registry
  (dataset ``decision_index``, pinned version and sha256 below);
- ``fetch_only``: everything else. Each user rebuilds the suite with the public kit and converts it locally, then
  points ``decision_index_fetch_only_jsonl`` at the converted file.

A user who has converted the whole suite locally sets ``decision_index_jsonl`` to that file instead; nothing is
downloaded then. Without either local file the prepared data covers only the registry benchmarks, and the run's
``decision_index_full_suite`` metric says so.
"""

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from nemo_gym.config import DownloadJsonlDatasetGitlabConfig, get_global_config_dict
from nemo_gym.gitlab_utils import download_jsonl_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "decision_index_benchmark.jsonl"

DATASET_NAME = "decision_index"
VERSION = "0.3.0"
ARTIFACT = "di03_public_registry.jsonl"
# The registry download neither checks the HTTP status nor verifies content, so the digest guards both.
REGISTRY_SHA256 = "bef38544049ff641df151336df49a67d787dffe35647499694e7afafd04e24ab"  # pragma: allowlist secret
REGISTRY_ROWS = 65_346
FULL_ROWS = 140_178

FULL_KEY = "decision_index_jsonl"
FETCH_ONLY_KEY = "decision_index_fetch_only_jsonl"


def _check_rows(path: Path) -> int:
    """Count rows and check each is a converted Decision Index row."""
    count = 0
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("di_request"), str) or "di_row" not in (row.get("verifier_metadata") or {}):
                raise ValueError(f"{path}:{line_number} is not a converted Decision Index row")
            count += 1
    return count


def _download_registry(target: Path) -> None:
    download_jsonl_dataset(
        DownloadJsonlDatasetGitlabConfig(
            dataset_name=DATASET_NAME, version=VERSION, artifact_fpath=ARTIFACT, output_fpath=str(target)
        )
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != REGISTRY_SHA256:
        raise ValueError(
            f"{DATASET_NAME} {VERSION} downloaded with sha256 {digest}, not the pinned {REGISTRY_SHA256}; "
            "check mlflow_tracking_uri and mlflow_tracking_token"
        )


def prepare(full_jsonl: Optional[str] = None, fetch_only_jsonl: Optional[str] = None) -> Path:
    """Write the benchmark rows to OUTPUT_FPATH and return the path."""
    config = get_global_config_dict()
    full_jsonl = full_jsonl or config.get(FULL_KEY)
    fetch_only_jsonl = fetch_only_jsonl or config.get(FETCH_ONLY_KEY)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=DATA_DIR) as temporary_dir:
        staged = Path(temporary_dir) / OUTPUT_FPATH.name
        if full_jsonl:
            shutil.copyfile(full_jsonl, staged)
            rows = _check_rows(staged)
            print(f"{rows} Decision Index rows from {full_jsonl} (full suite: {rows == FULL_ROWS})")
        else:
            _download_registry(staged)
            rows = _check_rows(staged)
            if rows != REGISTRY_ROWS:
                raise ValueError(f"{DATASET_NAME} {VERSION} has {rows} rows, not {REGISTRY_ROWS}")
            if fetch_only_jsonl:
                with staged.open("a", encoding="utf-8") as out, open(fetch_only_jsonl, encoding="utf-8") as extra:
                    shutil.copyfileobj(extra, out)
                rows = _check_rows(staged)
            else:
                print(
                    f"Only the {REGISTRY_ROWS} registry rows are prepared; set {FETCH_ONLY_KEY} to your locally "
                    "converted rows for the full suite."
                )
            print(f"{rows} Decision Index rows (full suite: {rows == FULL_ROWS})")
        staged.replace(OUTPUT_FPATH)
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
