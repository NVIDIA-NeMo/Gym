# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from resources_servers.deepswe.task_store import (
    DeepSWETaskStore,
    Task,
    task_collect_hook,
    task_id,
    task_image,
    task_sandbox_resources,
    task_verifier_files,
)


def test_load_task_store(task_assets: Path) -> None:
    store = DeepSWETaskStore(task_assets, expected_task_count=1)
    task = store.get("example-task")

    assert len(store) == 1
    assert isinstance(task, Task)
    assert task_id(task) == "example-task"
    assert task_image(task) == "public.example/project/example-task:v1.1"
    assert (
        task.config.metadata["base_commit_hash"]
        == "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
    )
    collect = task_collect_hook(task)
    assert collect.command.endswith(
        "git diff --binary 0123456789abcdef0123456789abcdef01234567 HEAD > /logs/artifacts/model.patch"
    )
    assert collect.timeout_sec == 300
    assert task_sandbox_resources(task, phase="agent") == {"cpu": 2, "memory_mib": 8192, "disk_gib": 20}
    assert task_sandbox_resources(task, phase="verifier") == {"cpu": 3, "memory_mib": 12288, "disk_gib": 25}
    assert set(task_verifier_files(task)) == {"test.sh", "test.patch", "grader.py", "config.json"}
