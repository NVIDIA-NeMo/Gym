# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tasks the harness graded without ever running.

The in-sandbox harness still emits a result when it fails to set the task up, so
a rollout can carry a real-looking zero while no transcript was produced. Scored
as-is it is indistinguishable from a genuine zero, and the dispatcher caches it,
so resume never retries.
"""

from unittest.mock import AsyncMock, patch

import pytest

from responses_api_agents.pinchbench.app import (
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    SandboxKilledError,
    TaskNotExecutedError,
    _classify_task_failure,
)
from responses_api_agents.pinchbench.tests.test_app import make_agent


GRADED_ZERO = {
    "reward": 0.0,
    "grading_type": "hybrid",
    "breakdown": {},
    "notes": "Skipped: task execution failed (error), no transcript to evaluate",
    "status": "success",
}
GRADED_ONE = {**GRADED_ZERO, "reward": 1.0, "notes": ""}


async def _run(agent, transcript_events, result):
    record = {"responses_create_params": {"input": "do the task"}, "verifier_metadata": {"task_id": "task_x"}}
    body = type("Body", (), {"model_dump": lambda self: dict(record)})()
    with (
        patch.object(agent, "_run_in_sandbox", AsyncMock(return_value=None)),
        patch.object(agent, "_parse_result", return_value=result),
        patch.object(agent, "_response_from_transcript", return_value=agent._empty_response("task_x")),
        patch.object(agent, "_collect_transcript", return_value=(transcript_events, "/archive")),
    ):
        return await agent.run(body)


@pytest.mark.asyncio
async def test_grade_without_transcript_is_not_reported_as_a_score():
    response = await _run(make_agent(), [], GRADED_ZERO)

    assert getattr(response, NG_FAILURE_CLASS_KEY) == "not_executed"
    assert response.status != "success"


@pytest.mark.asyncio
async def test_grade_without_transcript_is_not_persisted_so_resume_retries():
    response = await _run(make_agent(), [], GRADED_ZERO)

    assert getattr(response, NG_NO_PERSIST_KEY) is True


@pytest.mark.asyncio
async def test_graded_run_with_a_transcript_is_reported_normally():
    response = await _run(make_agent(), [{"type": "message"}], GRADED_ONE)

    assert response.reward == 1.0
    assert response.status == "success"
    assert not hasattr(response, NG_FAILURE_CLASS_KEY)


def test_not_executed_routes_apart_from_the_other_failures():
    assert _classify_task_failure(TaskNotExecutedError("x")) == "not_executed"
    assert _classify_task_failure(SandboxKilledError("x")) == "kill_shaped"
    assert _classify_task_failure(TimeoutError("x")) == "timeout_exceeded"
    assert _classify_task_failure(RuntimeError("x")) == "legitimate"
