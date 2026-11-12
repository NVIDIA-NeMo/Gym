# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A verifier's sandbox lifecycle record is evidence of sandbox use, not a coverage condition.

A trial whose grading is otherwise bound stays observed when its verifier recorded the sandboxes it used (outcome
completed or unreported, teardown confirmed, unconfirmed or failed, the agent's sandbox included). Only a grading
sandbox that reports a failure, or a malformed record, leaves the trial unobserved.
"""

from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents.utils.sandbox_observation import lifecycle_observation
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec
from tests.unit_tests.orchestration.test_evaluation_health import _run
from tests.unit_tests.orchestration.test_grading_artifacts import _observed, _ungraded


def _sandbox(role: str, **changes) -> dict:
    """One sandbox as the collector projects it: Terminal-Bench 4 records the agent's and the verifier's."""
    return {
        "role": role,
        "provider": "docker",
        "sandbox_id": f"{role}-sandbox",
        "outcome": "unknown",
        "exit_code": None,
        "teardown_confirmed": True,
        "error_type": None,
    } | changes


def _with_sandboxes(row: dict, *sandboxes: dict) -> None:
    row["ng_trial_observations"]["verifier_invocations"][0]["sandbox_observations"] = list(sandboxes)


def _test_execution(row: dict) -> None:
    """Graded by a saved test run, as a sandboxed verifier saves it."""
    _ungraded(row, verifier_exit_code=1)
    row["evaluation_completed"] = True


@pytest.mark.parametrize("artifact", [False, True], ids=["grading-operation", "test-execution"])
@pytest.mark.parametrize(
    "sandboxes",
    [
        [_sandbox("agent"), _sandbox("verifier")],
        [_sandbox("verifier", outcome="completed", exit_code=1)],
        [_sandbox("verifier", teardown_confirmed=False)],
        [_sandbox("agent", outcome="timeout"), _sandbox("verifier")],
        # As the producer records a close or a client release that failed after the grading.
        [
            _sandbox(
                "verifier",
                outcome="sandbox_error",
                error_type="close:TimeoutError",
                teardown_confirmed=False,
                cleanup_errors=["close:TimeoutError"],
            )
        ],
        [_sandbox("verifier", error_type="aclose:OSError", cleanup_errors=["aclose:OSError"])],
    ],
    ids=[
        "unreported",
        "completed-tests-failed",
        "teardown-unconfirmed",
        "agent-timed-out",
        "close-failed-after-grade",
        "client-release-failed",
    ],
)
def test_a_recorded_sandbox_use_beside_bound_grading_is_observed(artifact, sandboxes):
    collected = _run(20)
    for row in collected.results:
        if artifact:
            _test_execution(row)
        _with_sandboxes(row, *sandboxes)
    passed, reason, report, task = _observed(collected)
    assert passed is True and report["exact"] is True, reason
    assert task["trial_counts"]["unobserved"] == 0


@pytest.mark.parametrize(
    "sandbox",
    [
        _sandbox("verifier", outcome="failed"),
        _sandbox("verifier", outcome="sandbox_error"),
        _sandbox("verifier", outcome="oom"),
        _sandbox("verifier", error_type="SandboxCreateError"),
        # A failed grading stays a failure when its teardown failed too.
        _sandbox(
            "verifier",
            outcome="failed",
            error_type="close:TimeoutError",
            teardown_confirmed=False,
            cleanup_errors=["close:TimeoutError"],
        ),
        _sandbox("verifier", outcome="sandbox_error", error_type="SandboxExecError", cleanup_errors=["close:OSError"]),
        "malformed",
    ],
    ids=[
        "failed",
        "sandbox-error",
        "oom",
        "error-type",
        "failed-and-close-failed",
        "error-besides-close",
        "malformed",
    ],
)
def test_a_grading_sandbox_failure_leaves_the_trial_unobserved(sandbox):
    collected = _run(20)
    for row in collected.results:
        _test_execution(row)
    _with_sandboxes(collected.results[0], _sandbox("agent"), sandbox)
    passed, reason, report, _ = _observed(collected)
    assert passed is None and report["status"] == "UNVERIFIED", reason


@pytest.mark.sandbox
@pytest.mark.parametrize("error", [TimeoutError(), ConnectionError("refused")], ids=["timeout", "connection"])
async def test_a_verifier_sandbox_whose_close_failed_after_grading_keeps_the_trial_observed(error):
    """End to end from the producer: a real sandbox whose provider close fails, as lifecycle_observation records it."""
    from tests.unit_tests.agents.test_sandbox_lifecycle_observation import FakeSandboxProvider

    provider = FakeSandboxProvider()
    provider.close = AsyncMock(side_effect=error)
    sandbox = AsyncSandbox(provider)
    await sandbox.start(SandboxSpec(image="task"))
    with pytest.raises(type(error)):
        await sandbox.stop()
    observation = lifecycle_observation(sandbox, role="verifier")
    assert observation["cleanup_errors"] and observation["error_type"] == observation["cleanup_errors"][0]
    collected = _run(20)
    for row in collected.results:
        _test_execution(row)
    _with_sandboxes(collected.results[0], observation)
    passed, reason, report, task = _observed(collected)
    assert passed is True and task["trial_counts"]["unobserved"] == 0, reason
