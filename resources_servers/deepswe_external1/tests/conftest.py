# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from resources_servers.deepswe_external1.prepare_examples import describe_assets, materialize_task
from resources_servers.deepswe_external1.task_store import ASSET_PATHS, PhaseLimits, PreparedTask, TaskDefinition


@pytest.fixture
def task(tmp_path: Path) -> PreparedTask:
    source = tmp_path / "source"
    for name in ASSET_PATHS:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original asset\n", encoding="utf-8")
    base = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
    (source / "tests/config.json").write_text(json.dumps({"base_commit": base}), encoding="utf-8")
    limits = PhaseLimits(cpus=1, memory_mb=1024, storage_mb=2048, timeout_sec=120)
    definition = TaskDefinition(
        task_id="example-task",
        image="public.example/tasks/base@sha256:" + "a" * 64,
        verifier_image="public.example/tasks/verifier@sha256:" + "b" * 64,
        base_commit=base,
        agent=limits,
        verifier=limits,
        assets=describe_assets(source),
    )
    return materialize_task(source, tmp_path / "tasks", definition)
