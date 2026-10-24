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
"""Exercise live admission waits, shared budgets, and actual nested ATIF conversion."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from environments.skill_trigger.budget import BudgetReached, TrialBudget
from environments.skill_trigger.codex_runtime import CodexRuntime
from environments.skill_trigger.container import TrialContainer
from environments.skill_trigger.tests.test_codex import fixture


def runtime(tmp_path):
    budget = TrialBudget(model="test-model", root_session_id="harbor-root", preserve_payload=False, clock=lambda: 10.0)
    budget.release()
    result = CodexRuntime(container=TrialContainer("a" * 64), budget=budget, evidence_dir=tmp_path, model="test-model")
    result.accept_cli({"type": "thread.started", "thread_id": "root"})
    return result


def append(runtime, path, events):
    for event in events:
        runtime.records.append(path=path, event=event, at=0.0)


def payload(sid, model="test-model"):
    return {"model": model, "client_metadata": {"thread_id": sid}}


async def test_child_admission_waits_for_parent_proof(tmp_path):
    run = runtime(tmp_path)
    root, child = fixture()
    append(run, "root", root[:-1])
    append(run, "child", child)
    task = asyncio.create_task(run.resolve_request(payload("child")))
    try:
        await asyncio.sleep(0.03)
        assert not task.done() and run.budget.sessions == {"root"}
        append(run, "root", root[-1:])
        assert await asyncio.wait_for(task, 1) == "child"
        assert run.budget.sessions == {"root", "child"}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_model_selection_boundary_is_not_forwarded(tmp_path):
    run = runtime(tmp_path)
    root, child = fixture(selected_model="other")
    append(run, "root", root)
    append(run, "child", child)
    with pytest.raises(BudgetReached, match="routing_boundary"):
        await run.resolve_request(payload("child", "other"))
    assert run.budget.requests == [] and run.budget.sessions == {"root"}


async def test_children_share_budget_and_are_embedded_in_actual_atif(tmp_path):
    run = runtime(tmp_path)
    root, child = fixture(fork=True)
    append(run, "root", root)
    append(run, "child", child)
    for sid in ["root", "child", "root"]:
        resolved = await run.resolve_request(payload(sid))
        async with run.budget.forward(session_id=resolved, endpoint="/responses", payload=payload(sid)):
            pass
    with pytest.raises(BudgetReached, match="routing_horizon"):
        async with run.budget.forward(session_id="child", endpoint="/responses", payload=payload("child")):
            pytest.fail("Fourth shared request must not forward")
    root_id, sessions, spawns, outputs = run.evidence(skill_paths={})
    assert root_id == "root" and {s.id for s in sessions} == {"root", "child"}
    assert len(spawns) == 1
    assert {(o.session_id, o.call_id) for o in outputs} == {("root", "old"), ("child", "new")}
    trajectory = run.trajectory()
    assert trajectory.trajectory_id == "root"
    assert [t.trajectory_id for t in trajectory.subagent_trajectories] == ["child"]
    refs = [
        ref.trajectory_id
        for step in trajectory.steps
        if step.observation
        for output in step.observation.results
        for ref in output.subagent_trajectory_ref or []
    ]
    assert refs == ["child"]
    child_calls = [
        call.tool_call_id for step in trajectory.subagent_trajectories[0].steps for call in step.tool_calls or []
    ]
    assert child_calls == ["new"]


async def test_failed_reader_preserves_diagnostics_and_closes_once(tmp_path):
    run = runtime(tmp_path)
    process = SimpleNamespace(returncode=0, wait=AsyncMock(return_value=0), stderr=asyncio.StreamReader())
    process.stderr.feed_data(b"collector diagnostics")
    process.stderr.feed_eof()
    run.process = process

    async def fail():
        raise ValueError("altered rollout")

    run.reader = asyncio.create_task(fail())
    with pytest.raises(ValueError, match="altered rollout"):
        await run.stop()
    assert (tmp_path / "codex-watch.stderr").read_bytes() == b"collector diagnostics"
    await run.stop()  # cleanup must not replace the original capture error
