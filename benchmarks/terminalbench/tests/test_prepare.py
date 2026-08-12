# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.terminalbench.prepare import _task_row, prepare
from nemo_gym.benchmarks import BenchmarkConfig


def _task(root: Path, name: str) -> Path:
    task = root / "terminal-bench" / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        """
[environment]
docker_image = "task-image:latest"
cpus = 4
memory = "8G"
storage = "12G"

[agent]
timeout_sec = 120

[verifier]
timeout_sec = 60
""".strip()
        + "\n"
    )
    (task / "instruction.md").write_text("solve the task\n")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\nWORKDIR /workspace\n")
    (task / "tests" / "test.sh").write_text("echo 1\n")
    return task


def test_task_row_carries_sandbox_metadata_without_tests(tmp_path) -> None:
    row = _task_row(_task(tmp_path, "one"))
    metadata = row["responses_create_params"]["metadata"]
    assert row["responses_create_params"]["input"][0]["content"] == "solve the task\n"
    assert metadata["docker_image"] == "task-image:latest"
    assert metadata["workdir"] == "/workspace"
    assert metadata["memory_mb"] == "8192"
    assert metadata["storage_mb"] == "12288"
    assert "tests" not in json.dumps(row["responses_create_params"])


def test_prepare_selects_requested_tasks(tmp_path) -> None:
    _task(tmp_path, "one")
    _task(tmp_path, "two")
    output = tmp_path / "out.jsonl"
    with patch("benchmarks.terminalbench.prepare._download"):
        prepare(output=output, tasks_cache=tmp_path, task_names=["two"])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_name"] for row in rows] == ["two"]


def test_config_resolves_claude_code_and_one_dataset() -> None:
    config = BenchmarkConfig.from_config_path(Path("benchmarks/terminalbench/config.yaml"))
    assert config.agent_name == "terminalbench_claude_code"
    assert config.name == "terminalbench"
    assert config.num_repeats == 1
