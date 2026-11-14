# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The environment owns request deduplication and suite aggregation."""

import asyncio
import sys

import pytest
from fastapi import HTTPException

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from responses_api_agents.wildclawbench_agent.tests.helpers import make_pipeline


async def test_saved_reply_conflicts_and_interrupted_attempts_are_not_reexecuted(tmp_path, monkeypatch):
    environment, _, _, _ = make_pipeline(tmp_path, sys.executable, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def episode(self, row):
        calls.append(row)
        entered.set()
        await release.wait()
        return {"reward": 0.0}

    monkeypatch.setattr(SingleAgentTurnLegacyEnvironmentServer, "run_legacy", episode)
    row = dict(_ng_task_index=1, _ng_rollout_index=0, _ng_run_id="one", responses_create_params={"input": "task"})
    first = asyncio.create_task(environment.run_legacy(row))
    await entered.wait()
    second = asyncio.create_task(environment.run_legacy(row))
    with pytest.raises(HTTPException) as conflict:
        await environment.run_legacy(dict(row, verifier_metadata={"different": True}))
    assert conflict.value.status_code == 409
    release.set()
    assert await first == await second == await environment.run_legacy(row) == {"reward": 0.0}
    assert len(calls) == 1
    # A restarted server reads the saved result; no policy stage is invoked twice.
    other, _, _, _ = make_pipeline(tmp_path, sys.executable, monkeypatch)
    assert await other.run_legacy(row) == {"reward": 0.0}
    result = next((tmp_path / "episodes").glob("*/environment_result.json"))
    result.unlink()
    with pytest.raises(HTTPException) as incomplete:
        await other.run_legacy(row)
    assert incomplete.value.status_code == 409
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field", ["agent_context", "wildclaw_execution_key", "episode_id", "task", "_ng_model_sampling"]
)
async def test_private_controls_are_rejected(tmp_path, monkeypatch, field):
    environment, _, _, _ = make_pipeline(tmp_path, sys.executable, monkeypatch)
    with pytest.raises(HTTPException) as rejected:
        await environment.run_legacy(dict(_ng_task_index=0, _ng_rollout_index=0, **{field: {"value": True}}))
    assert rejected.value.status_code == 422
    assert not (tmp_path / "episodes").exists()
