# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the agentic IF instruction-following benchmark.

The sealed rows are private. They live in Gym's GitLab dataset registry as two artifacts of one dataset version: the
rows, and the ``tests/`` files of the terminal task folders those rows reference. Preparation downloads both with the
configured ``mlflow_tracking_uri`` and ``mlflow_tracking_token``, checks their pinned SHA-256 digests and every row's
identity and receipts, writes the task tests at the Gym-relative paths the rows name (under this benchmark's ``data/``,
as Terminal-Bench 2.1's prepared task folders are), and writes the rows last.
"""

import base64
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import DownloadJsonlDatasetGitlabConfig
from nemo_gym.gitlab_utils import download_jsonl_dataset
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.runtime import validate_runtime


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "agentic_if_benchmark.jsonl"

DATASET_NAME = "agentic_if"
DATASET_VERSION = "2.0.0"
ROWS_ARTIFACT = "agentic_if_benchmark.jsonl"
ROWS_SHA256 = "6167213c22ee48a23b7e783d6ea8b6ec8998f4c5108c4f3cdb1129b1360196a0"  # pragma: allowlist secret
TESTS_ARTIFACT = "terminal-bench-tests.jsonl"
TESTS_SHA256 = "947225ae43b18428f570d86eb81835ce337a16e371e6c0c269bf12f7e2ac97b2"  # pragma: allowlist secret

# Terminal rows name their task folders by Gym-relative paths under this prefix; the path is part of each sealed row.
TASKS_PREFIX = PurePosixPath("benchmarks/agentic_if/data/terminal-bench-tasks")
TASKS_DIR = PARENT_DIR / TASKS_PREFIX
# Written beside the prepared task tests, so a cached preparation can be checked without downloading again.
TESTS_MARKER = ".tests-sha256"


def _download(artifact: str, expected_sha256: str, directory: Path) -> Path:
    """Download one artifact of the pinned dataset version and check its digest."""
    path = directory / artifact
    download_jsonl_dataset(
        DownloadJsonlDatasetGitlabConfig(
            dataset_name=DATASET_NAME,
            version=DATASET_VERSION,
            artifact_fpath=artifact,
            output_fpath=str(path),
        )
    )
    with path.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != expected_sha256:
        # The download helper writes any response body, so a failed request surfaces here.
        raise RuntimeError(
            f"{DATASET_NAME} {DATASET_VERSION} {artifact} has SHA-256 {actual}, expected {expected_sha256}; "
            "check mlflow_tracking_uri and mlflow_tracking_token"
        )
    return path


def _task_folders(rows: list[dict]) -> set[str]:
    """Check each row's identity and receipts; return the task folders the terminal rows reference."""
    task_ids, folders = set(), set()
    for row in rows:
        content = {key: value for key, value in row.items() if key != "task_id"}
        if row.get("task_id") != digest(content):
            raise ValueError(f"Row {row.get('task_id')!r} does not carry its content digest as task_id")
        if row["task_id"] in task_ids:
            raise ValueError(f"Duplicate task_id {row['task_id']}")
        task_ids.add(row["task_id"])
        validate_runtime(content)
        if (folder := content.get("task_folder")) is None:
            continue
        path = PurePosixPath(folder)
        if path.is_absolute() or ".." in path.parts or not path.is_relative_to(TASKS_PREFIX):
            raise ValueError(f"task_folder {folder!r} is not a folder of {TASKS_PREFIX}")
        folders.add(folder)
    return folders


def _write_tests(tests_path: Path, folders: set[str]) -> None:
    """Write the task tests at their Gym-relative paths, replacing an earlier preparation in one rename."""
    staging = Path(tempfile.mkdtemp(prefix=".terminal-bench-tasks-", dir=TASKS_DIR.parent))
    try:
        with tests_path.open() as stream:
            for line in stream:
                record = json.loads(line)
                relative = PurePosixPath(record["path"])
                if ".." in relative.parts or not relative.is_relative_to(TASKS_PREFIX):
                    raise ValueError(f"Test file {record['path']!r} is not under {TASKS_PREFIX}")
                target = staging / relative.relative_to(TASKS_PREFIX)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(record["content_b64"]))
                target.chmod(record["mode"])
        missing = sorted(
            folder
            for folder in folders
            if not (staging / PurePosixPath(folder).relative_to(TASKS_PREFIX) / "tests/test.sh").is_file()
        )
        if missing:
            raise RuntimeError(f"{len(missing)} referenced task folders have no tests/test.sh: {missing[:3]}")
        (staging / TESTS_MARKER).write_text(TESTS_SHA256)
        if TASKS_DIR.exists():
            shutil.rmtree(TASKS_DIR)
        os.replace(staging, TASKS_DIR)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def is_prepared_data_current(fpath: Path) -> bool:
    """Whether the prepared rows and task tests match the pinned dataset version."""
    try:
        with Path(fpath).open("rb") as file:
            rows_current = hashlib.file_digest(file, "sha256").hexdigest() == ROWS_SHA256
        return rows_current and (TASKS_DIR / TESTS_MARKER).read_text() == TESTS_SHA256
    except OSError:
        return False


def prepare() -> Path:
    """Download and check the pinned rows and task tests; the rows are written last."""
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".agentic_if_benchmark-", dir=OUTPUT_FPATH.parent) as directory:
        rows_path = _download(ROWS_ARTIFACT, ROWS_SHA256, Path(directory))
        tests_path = _download(TESTS_ARTIFACT, TESTS_SHA256, Path(directory))
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        _write_tests(tests_path, _task_folders(rows))
        os.replace(rows_path, OUTPUT_FPATH)
    print(f"Prepared {len(rows)} agentic IF tasks at {OUTPUT_FPATH}")
    return OUTPUT_FPATH.absolute()


if __name__ == "__main__":
    prepare()
