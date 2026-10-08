# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


@pytest.fixture
def task_assets(tmp_path: Path) -> Path:
    task_id = "example-task"
    task_dir = tmp_path / "tasks" / task_id
    for directory in (task_dir / "tests", task_dir / "solution"):
        directory.mkdir(parents=True)

    (task_dir / "task.toml").write_text(
        """schema_version = "1.3"
[task]
name = "datacurve/example-task"
description = ""
authors = []
keywords = []
[metadata]
task_id = "example-task"
repository_url = "https://github.com/example/repo"
base_commit_hash = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
language = "python"
[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 1800.0
[verifier.environment]
cpus = 3
memory_mb = 12288
storage_mb = 25600
[[verifier.collect]]
command = "cd /app && mkdir -p /logs/artifacts && git config --global --add safe.directory /app && git diff --binary 0123456789abcdef0123456789abcdef01234567 HEAD > /logs/artifacts/model.patch"
timeout_sec = 300.0
[agent]
network_mode = "no-network"
timeout_sec = 5400.0
[environment]
docker_image = "public.example/project/example-task:v1.1"
cpus = 2
memory_mb = 8192
storage_mb = 20480
""",
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Implement the feature and commit it.\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (task_dir / "tests" / "test.patch").write_text("", encoding="utf-8")
    (task_dir / "tests" / "grader.py").write_text("", encoding="utf-8")
    (task_dir / "tests" / "config.json").write_text("{}\n", encoding="utf-8")
    (task_dir / "solution" / "solution.patch").write_text("golden patch\n", encoding="utf-8")

    return task_dir.parent
