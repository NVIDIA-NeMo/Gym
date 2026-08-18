# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from resources_servers.deepswe.task_store import DeepSWETaskStore


def test_load_task_store(task_assets: Path) -> None:
    store = DeepSWETaskStore(task_assets, expected_task_count=1)
    task = store.get("example-task")

    assert len(store) == 1
    assert task.image == "public.example/project/example-task:v1.1"
    assert task.base_commit == "0123456789abcdef0123456789abcdef01234567"
    assert task.memory_mib == 8192
    assert task.disk_gib == 20
    assert set(task.verifier_files) == {"test.sh", "test.patch", "grader.py", "config.json"}
