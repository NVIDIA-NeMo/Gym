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
"""Whitebox agent executions park and restore at typed turn boundaries."""

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI

from nemo_gym._checkpoint import (
    AGENT_MANIFEST_NAME,
    AGENT_STATE_SUBDIR,
    AgentBoundaryRecord,
    AgentCheckpointError,
    AgentCheckpointParticipant,
    AgentStaleAttemptError,
    CheckpointPhase,
    ControlCapabilities,
    ControlFence,
    DuplicateExecutionError,
    MultiProcessCapability,
    commit_agent_state,
    install_agent_checkpoint,
    install_control_plane,
    restore_agent_state,
)


def _boundary(*, attempt_index: int = 0, boundary_index: int = 1) -> AgentBoundaryRecord:
    return AgentBoundaryRecord(
        rollout_id="rollout-a",
        attempt_index=attempt_index,
        boundary_index=boundary_index,
        output_items=[
            {"type": "function_call", "call_id": "tool-1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "tool-1", "output": "result"},
        ],
        usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        last_committed_model_call_id="call-1",
        resource_state_revisions={"resources": 3},
    )


@pytest.mark.asyncio
async def test_timed_out_prepare_can_retire_and_retry() -> None:
    participant = AgentCheckpointParticipant()
    await participant.begin("rollout-a", 0, task=None)
    fence = ControlFence()
    app = FastAPI()
    install_control_plane(
        app,
        capabilities=ControlCapabilities(
            component="responses_api_agents",
            name="agent",
            multi_process=MultiProcessCapability(mode="single_worker", num_workers=1),
        ),
        fence=fence,
    )
    install_agent_checkpoint(app, participant=participant, fence=fence, auth_token="secret")
    headers = {"authorization": "Bearer secret"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        failed = await client.post(
            "/ng-control/v1/agent-checkpoint/prepare",
            json={"checkpoint_id": "checkpoint-1", "deadline_ts": time.time() + 0.01},
            headers=headers,
        )
        assert failed.status_code == 409
        status = await client.get(
            "/ng-control/v1/agent-checkpoint/status",
            params={"checkpoint_id": "checkpoint-1"},
            headers=headers,
        )
        assert status.status_code == 200
        assert status.json()["blocking_attempts"] == [
            {
                "rollout_id": "rollout-a",
                "attempt_index": 0,
                "generation": 1,
                "state": "running",
                "parked_boundary_state": None,
                "boundary_index": None,
                "age_seconds": status.json()["blocking_attempts"][0]["age_seconds"],
            }
        ]
        retired = await client.post(
            "/ng-control/v1/agent-checkpoint/retire",
            json={
                "checkpoint_id": "checkpoint-1",
                "deadline_ts": time.time() + 2,
                "rollout_id": "rollout-a",
                "attempt_index": 0,
            },
            headers=headers,
        )
        assert retired.status_code == 200
        retried = await client.post(
            "/ng-control/v1/agent-checkpoint/prepare",
            json={"checkpoint_id": "checkpoint-1", "deadline_ts": time.time() + 2},
            headers=headers,
        )
        for phase in (CheckpointPhase.COMMITTED_PAUSED, CheckpointPhase.RESTORED_PAUSED):
            fence.phase = phase
            status = await client.get(
                "/ng-control/v1/agent-checkpoint/status",
                params={"checkpoint_id": "checkpoint-1"},
                headers=headers,
            )
            assert status.status_code == 200
        conflict = await client.get(
            "/ng-control/v1/agent-checkpoint/status",
            params={"checkpoint_id": "checkpoint-2"},
            headers=headers,
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "checkpoint_conflict"
    assert retried.status_code == 200
    assert retried.json()["running"] == 0


@pytest.mark.asyncio
async def test_one_run_per_attempt() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
    with pytest.raises(DuplicateExecutionError):
        await participant.begin("rollout-a", 0, task=asyncio.current_task())
    await participant.finish(execution, outcome="completed")
    replacement = await participant.begin("rollout-a", 1, task=asyncio.current_task())
    await participant.finish(replacement, outcome="completed")


@pytest.mark.asyncio
async def test_completed_attempt_replays_retained_terminal_result() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
    terminal_result = {"reward": 1.0}
    await participant.finish(execution, outcome="completed", result=terminal_result)

    replay = await participant.begin("rollout-a", 0, task=asyncio.current_task())

    assert replay is execution
    assert replay.terminal_result is terminal_result
    assert await participant.retire("rollout-a", 0) == {
        "retired": False,
        "tombstoned": False,
        "completed_unacknowledged": True,
    }
    status = participant.status()
    assert status["completed_unacknowledged"] == 1
    assert status["ready_to_commit"] is False
    assert participant.resolve("rollout-a", 0) is execution


@pytest.mark.asyncio
async def test_retire_tombstones_attempt_without_execution() -> None:
    participant = AgentCheckpointParticipant()

    assert await participant.retire("rollout-a", 7) == {"retired": False, "tombstoned": True}
    with pytest.raises(AgentStaleAttemptError):
        await participant.begin("rollout-a", 7, task=None)


@pytest.mark.asyncio
async def test_prepare_waits_for_boundary_and_resume_is_explicit() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())

    prepare = asyncio.create_task(participant.prepare(time.time() + 2))
    await asyncio.sleep(0)
    boundary = asyncio.create_task(participant.commit_boundary(execution, _boundary()))
    report = await prepare
    assert report["running"] == 0
    assert report["parked"] == 1
    assert not boundary.done()

    await asyncio.sleep(0)
    assert not boundary.done()
    assert (await participant.resume())["released"] == 1
    await boundary
    await participant.finish(execution, outcome="completed")


@pytest.mark.asyncio
async def test_failed_prepare_rolls_back_park_request() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())

    report = await participant.prepare(time.time() + 0.01)

    assert report["running"] == 1
    assert participant.status()["executions"][0]["state"] == "running"
    await participant.commit_boundary(execution, _boundary())
    assert participant.status()["executions"][0]["state"] == "running"
    await participant.resume()


@pytest.mark.asyncio
async def test_prepare_reports_completed_unacknowledged_result() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
    await participant.finish(execution, outcome="completed", result={"reward": 1.0})

    report = await participant.prepare(time.time() + 2)

    assert report["ready_to_commit"] is False
    assert report["completed_unacknowledged"] == 1
    assert report["completed_unacknowledged_attempts"][0]["rollout_id"] == "rollout-a"
    with pytest.raises(AgentCheckpointError):
        participant.records_for_commit()


@pytest.mark.asyncio
async def test_prepare_route_rejects_completed_unacknowledged_result() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
    await participant.finish(execution, outcome="completed", result={"reward": 1.0})
    fence = ControlFence()
    app = FastAPI()
    install_control_plane(
        app,
        capabilities=ControlCapabilities(
            component="responses_api_agents",
            name="agent",
            multi_process=MultiProcessCapability(mode="single_worker", num_workers=1),
        ),
        fence=fence,
    )
    install_agent_checkpoint(app, participant=participant, fence=fence, auth_token="secret")
    headers = {"authorization": "Bearer secret"}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        prepare = await client.post(
            "/ng-control/v1/agent-checkpoint/prepare",
            json={"checkpoint_id": "checkpoint-1", "deadline_ts": time.time() + 2},
            headers=headers,
        )
        status = await client.get(
            "/ng-control/v1/agent-checkpoint/status",
            params={"checkpoint_id": "checkpoint-1"},
            headers=headers,
        )

    assert prepare.status_code == 409
    assert "completed_unacknowledged=1" in prepare.json()["error"]["detail"]
    assert fence.phase == CheckpointPhase.IDLE
    assert status.json()["completed_unacknowledged"] == 1


@pytest.mark.asyncio
async def test_prepare_distinguishes_parked_execution_without_boundary() -> None:
    participant = AgentCheckpointParticipant()
    outer = asyncio.create_task(asyncio.Event().wait())
    execution = await participant.begin("rollout-a", 0, task=outer)
    parked = asyncio.create_task(participant.park(execution))
    while participant.status()["parked"] != 1:
        await asyncio.sleep(0)

    report = await participant.prepare(time.time() + 2)

    assert report["ready_to_commit"] is False
    assert report["parked_with_boundary"] == 0
    assert report["parked_without_boundary"] == 1
    assert report["executions"][0]["parked_boundary_state"] == "parked_without_boundary"
    await participant.retire("rollout-a", 0)
    with pytest.raises(asyncio.CancelledError):
        await parked
    with pytest.raises(asyncio.CancelledError):
        await outer


@pytest.mark.asyncio
async def test_retire_cancels_outer_and_actual_parked_tasks() -> None:
    participant = AgentCheckpointParticipant()
    outer = asyncio.create_task(asyncio.Event().wait())
    execution = await participant.begin("rollout-a", 0, task=outer)
    prepare = asyncio.create_task(participant.prepare(time.time() + 2))
    await asyncio.sleep(0)
    parked = asyncio.create_task(participant.commit_boundary(execution, _boundary()))
    await prepare

    assert participant.status()["parked"] == 1
    await participant.retire("rollout-a", 0)
    with pytest.raises(asyncio.CancelledError):
        await outer
    with pytest.raises(asyncio.CancelledError):
        await parked
    with pytest.raises(AgentStaleAttemptError):
        await participant.commit_boundary(execution, _boundary(boundary_index=2))


@pytest.mark.asyncio
async def test_failed_execution_is_tombstoned_and_released() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
    await participant.commit_boundary(execution, _boundary())

    await participant.finish(execution, outcome="failed")

    assert participant.resolve("rollout-a", 0) is None
    assert execution.boundary is None
    with pytest.raises(AgentStaleAttemptError):
        await participant.commit_boundary(execution, _boundary(boundary_index=2))
    with pytest.raises(AgentStaleAttemptError):
        await participant.begin("rollout-a", 0, task=None)


@pytest.mark.asyncio
async def test_cancelled_parked_run_keeps_boundary_until_commit(tmp_path) -> None:
    participant = AgentCheckpointParticipant()
    begun = asyncio.Event()
    proceed = asyncio.Event()

    async def run() -> None:
        execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
        begun.set()
        await proceed.wait()
        try:
            await participant.commit_boundary(execution, _boundary())
        except asyncio.CancelledError:
            await participant.finish(execution, outcome="cancelled")
            raise

    task = asyncio.create_task(run())
    await begun.wait()
    prepare = asyncio.create_task(participant.prepare(time.time() + 2))
    await asyncio.sleep(0)
    proceed.set()
    report = await prepare
    assert report["parked"] == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert commit_agent_state(participant, tmp_path, checkpoint_id="checkpoint-1")["records"] == 1
    assert (await participant.resume())["state"] == "accepting"


@pytest.mark.asyncio
async def test_instance_namespaces_do_not_overwrite_each_other(tmp_path) -> None:
    first = AgentCheckpointParticipant("agent/one")
    second = AgentCheckpointParticipant("agent two")
    first_execution = await first.begin("rollout-a", 0, task=asyncio.current_task())
    second_execution = await second.begin("rollout-a", 0, task=asyncio.current_task())
    first_prepare = asyncio.create_task(first.prepare(time.time() + 2))
    second_prepare = asyncio.create_task(second.prepare(time.time() + 2))
    await asyncio.sleep(0)
    first_park = asyncio.create_task(first.commit_boundary(first_execution, _boundary()))
    second_park = asyncio.create_task(second.commit_boundary(second_execution, _boundary()))
    await asyncio.gather(first_prepare, second_prepare)

    assert commit_agent_state(first, tmp_path, checkpoint_id="checkpoint-1")["records"] == 1
    assert commit_agent_state(second, tmp_path, checkpoint_id="checkpoint-1")["records"] == 1
    restored_first = AgentCheckpointParticipant("agent/one")
    restored_second = AgentCheckpointParticipant("agent two")
    assert restore_agent_state(restored_first, tmp_path)["records"] == 1
    assert restore_agent_state(restored_second, tmp_path)["records"] == 1

    await first.resume()
    await second.resume()
    await asyncio.gather(first_park, second_park)


def test_restore_rejects_wrong_instance_manifest(tmp_path) -> None:
    participant = AgentCheckpointParticipant()
    directory = tmp_path / AGENT_STATE_SUBDIR
    directory.mkdir()
    (directory / AGENT_MANIFEST_NAME).write_text(
        '{"schema_version": 1, "checkpoint_id": "checkpoint-1", "instance_name": "other", "files": {}}'
    )

    with pytest.raises(AgentCheckpointError, match="belongs to instance"):
        restore_agent_state(participant, tmp_path)


@pytest.mark.parametrize("instance_name", ["", "line\nbreak", "x" * 513])
def test_invalid_instance_names_are_rejected(instance_name) -> None:
    with pytest.raises(ValueError):
        AgentCheckpointParticipant(instance_name)


@pytest.mark.asyncio
async def test_prepare_deadline_reports_execution_between_boundaries() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=None)
    report = await participant.prepare(time.time() + 0.01)
    assert report["running"] == 1
    assert report["executions"][0]["state"] == "park_requested"
    await participant.retire("rollout-a", 0)
    await participant.finish(execution, outcome="completed")


@pytest.mark.asyncio
async def test_boundary_indices_are_monotonic_and_idempotent() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
    record = _boundary()
    await participant.commit_boundary(execution, record)
    await participant.commit_boundary(execution, record)
    with pytest.raises(AgentCheckpointError):
        await participant.commit_boundary(
            execution,
            record.model_copy(update={"last_committed_model_call_id": "other"}),
        )
    await participant.finish(execution, outcome="completed")


@pytest.mark.asyncio
async def test_commit_restore_maps_source_attempt_to_replacement(tmp_path) -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 2, task=asyncio.current_task())
    prepare = asyncio.create_task(participant.prepare(time.time() + 2))
    await asyncio.sleep(0)
    park = asyncio.create_task(participant.commit_boundary(execution, _boundary(attempt_index=2, boundary_index=4)))
    await prepare

    summary = commit_agent_state(participant, tmp_path, checkpoint_id="checkpoint-1")
    assert summary["records"] == 1
    restored = AgentCheckpointParticipant()
    assert restore_agent_state(restored, tmp_path) == {
        "records": 1,
        "source_checkpoint_id": "checkpoint-1",
    }
    await restored.resume()
    replacement = await restored.begin("rollout-a", 3, task=None)
    continuation = restored.continuation(replacement)
    assert continuation is not None
    assert continuation.boundary_index == 4
    assert continuation.last_committed_model_call_id == "call-1"
    assert continuation.resource_state_revisions == {"resources": 3}
    with pytest.raises(AgentStaleAttemptError):
        await restored.begin("rollout-a", 2, task=None)

    await participant.resume()
    await park
    await participant.finish(execution, outcome="completed")


@pytest.mark.asyncio
async def test_durable_agent_commit_retry_returns_original_result(tmp_path) -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
    prepare = asyncio.create_task(participant.prepare(time.time() + 2))
    await asyncio.sleep(0)
    park = asyncio.create_task(participant.commit_boundary(execution, _boundary()))
    await prepare
    first = commit_agent_state(participant, tmp_path, checkpoint_id="checkpoint-1")
    second = commit_agent_state(participant, tmp_path, checkpoint_id="checkpoint-1")
    assert second == first
    await participant.resume()
    await park
    await participant.finish(execution, outcome="completed")


def test_restore_rejects_corrupted_boundary_before_activation(tmp_path) -> None:
    directory = tmp_path / AGENT_STATE_SUBDIR
    directory.mkdir()
    (directory / AGENT_MANIFEST_NAME).write_text(
        '{"schema_version": 1, "checkpoint_id": "checkpoint-1", "files": {"missing.json": "bad"}}'
    )
    participant = AgentCheckpointParticipant()
    with pytest.raises(AgentCheckpointError):
        restore_agent_state(participant, tmp_path)
    assert participant.resolve("rollout-a", 1) is None
