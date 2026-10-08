# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from resources_servers.deepswe_external1.inline_task import InlineTask
from resources_servers.deepswe_external1.task_data import PhaseLimits, TaskData, TaskFiles


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def task() -> InlineTask:
    base = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
    limits = PhaseLimits(cpus=1, memory_mb=1024, storage_mb=2048, timeout_sec=120)
    return InlineTask(
        TaskData(
            task_id="example-task",
            image="public.example/tasks/base@sha256:" + "a" * 64,
            verifier_image="public.example/tasks/verifier@sha256:" + "b" * 64,
            base_commit=base,
            agent=limits,
            verifier=limits,
            files=TaskFiles(
                test_script="#!/bin/bash\nprintf 'held out test'\n",
                test_patch="held out patch\n",
                grader="original grader\n",
                grader_config=json.dumps({"base_commit": base}),
                solve_script="#!/bin/bash\nprintf 'golden solution'\n",
                solution_patch="golden patch\n",
            ),
        )
    )
