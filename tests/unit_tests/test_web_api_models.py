# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.web.api_models import WebSeedSessionRequest, WebStepRequest
from nemo_gym.web.models import WebAction, WebBenchmark, WebTask


def test_backend_neutral_requests_preserve_task_and_operation_identity() -> None:
    task = WebTask(benchmark=WebBenchmark.WEBARENA, task_id=42)
    seed = WebSeedSessionRequest(task=task)
    step = WebStepRequest(
        operation_id="task-42-step-1",
        action=WebAction(name="noop", script="noop()"),
    )

    assert seed.task.task_id == "42"
    assert step.operation_id == "task-42-step-1"
