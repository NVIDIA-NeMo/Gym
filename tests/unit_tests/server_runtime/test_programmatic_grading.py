# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A grading observation executes its reviewed scorer once and cannot erase its result."""

import pytest

from nemo_gym.core.episode import rollout_context
from nemo_gym.server_runtime.verifier_observability import observe_programmatic_grading, observe_verifier
from nemo_gym.trajectory.trajectory_record import grading_result_binding


def result(reward=1):
    return {
        "response": {"id": "synthetic-response", "output": [{"type": "message", "content": "saved input"}]},
        "reward": reward,
    }


@pytest.mark.asyncio
async def test_calculation_executes_once_and_inbound_observation_is_not_trusted():
    calls = []
    original = result()
    original["ng_verifier_invocations"] = [{"invocation_id": "forged", "status": "completed"}]

    def calculate():
        calls.append("executed")
        return original

    async def verify():
        return observe_programmatic_grading(calculate)

    with rollout_context("synthetic-rollout"):
        actual = await observe_verifier(verify, enabled=True)()
    assert calls == ["executed"]
    assert actual["reward"] == original["reward"] and actual["response"] == original["response"]
    invocation = actual["ng_verifier_invocations"][0]
    assert invocation["invocation_id"] != "forged"
    operation = invocation["grading_operations"][0]
    assert operation["outcome"] == "completed" and operation["dependencies"] == []
    assert operation["result_binding_sha256"] == grading_result_binding(original)


@pytest.mark.asyncio
async def test_caught_calculation_error_remains_observed_failed():
    def calculate():
        raise ValueError("synthetic grading failure")

    async def verify():
        try:
            return observe_programmatic_grading(calculate)
        except ValueError:
            return result(0)

    with rollout_context("synthetic-rollout"):
        actual = await observe_verifier(verify, enabled=True)()
    operation = actual["ng_verifier_invocations"][0]["grading_operations"][0]
    assert actual["reward"] == 0
    assert operation["outcome"] == "failed" and operation["result_binding_sha256"] is None


@pytest.mark.asyncio
async def test_unobserved_caught_dependency_is_not_programmatic():
    async def verify():
        try:
            raise ConnectionError("unobserved external judge")
        except ConnectionError:
            return result(0)

    with rollout_context("synthetic-rollout"):
        actual = await observe_verifier(verify, enabled=True)()
    assert actual["ng_verifier_invocations"][0].get("grading_operations") is None
    assert actual["reward"] == 0


@pytest.mark.asyncio
async def test_binding_failure_cannot_erase_an_already_calculated_grade(monkeypatch):
    import nemo_gym.server_runtime.verifier_observability as observation

    def broken(_):
        raise RuntimeError("synthetic metadata encoding failure")

    monkeypatch.setattr(observation, "grading_result_binding", broken)

    async def verify():
        return observe_programmatic_grading(lambda: result(1))

    with rollout_context("synthetic-rollout"):
        actual = await observe_verifier(verify, enabled=True)()
    assert actual["reward"] == 1 and actual["ng_verifier_invocations"][0]["status"] == "completed"
    assert actual["ng_verifier_invocations"][0]["grading_operations"][0]["outcome"] == "unknown"
