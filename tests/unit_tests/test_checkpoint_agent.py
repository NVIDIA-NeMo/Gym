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

import pytest

from nemo_gym._checkpoint import (
    AGENT_MANIFEST_NAME,
    AGENT_STATE_SUBDIR,
    AgentBoundaryRecord,
    AgentCheckpointError,
    AgentCheckpointParticipant,
    DuplicateExecutionError,
    commit_agent_state,
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
async def test_one_run_per_attempt() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
    with pytest.raises(DuplicateExecutionError):
        await participant.begin("rollout-a", 0, task=asyncio.current_task())
    await participant.finish(execution)
    replacement = await participant.begin("rollout-a", 1, task=asyncio.current_task())
    await participant.finish(replacement)


@pytest.mark.asyncio
async def test_prepare_waits_for_boundary_and_resume_is_explicit() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())

    prepare = asyncio.create_task(participant.prepare(time.time() + 2))
    await asyncio.sleep(0)
    boundary = asyncio.create_task(participant.commit_boundary(_boundary()))
    report = await prepare
    assert report["running"] == 0
    assert report["parked"] == 1
    assert not boundary.done()

    await asyncio.sleep(0)
    assert not boundary.done()
    assert (await participant.resume())["released"] == 1
    await boundary
    await participant.finish(execution)


@pytest.mark.asyncio
async def test_prepare_deadline_reports_execution_between_boundaries() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=None)
    report = await participant.prepare(time.time() + 0.01)
    assert report["running"] == 1
    assert report["executions"][0]["state"] == "park_requested"
    await participant.retire("rollout-a", 0)
    await participant.finish(execution)


@pytest.mark.asyncio
async def test_boundary_indices_are_monotonic_and_idempotent() -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
    record = _boundary()
    await participant.commit_boundary(record)
    await participant.commit_boundary(record)
    with pytest.raises(AgentCheckpointError):
        await participant.commit_boundary(record.model_copy(update={"last_committed_model_call_id": "other"}))
    await participant.finish(execution)


@pytest.mark.asyncio
async def test_commit_restore_maps_source_attempt_to_replacement(tmp_path) -> None:
    participant = AgentCheckpointParticipant()
    execution = await participant.begin("rollout-a", 2, task=asyncio.current_task())
    prepare = asyncio.create_task(participant.prepare(time.time() + 2))
    await asyncio.sleep(0)
    park = asyncio.create_task(participant.commit_boundary(_boundary(attempt_index=2, boundary_index=4)))
    await prepare

    summary = commit_agent_state(participant, tmp_path, checkpoint_id="checkpoint-1")
    assert summary["records"] == 1
    restored = AgentCheckpointParticipant()
    assert restore_agent_state(restored, tmp_path) == {
        "records": 1,
        "source_checkpoint_id": "checkpoint-1",
    }
    continuation = restored.continuation("rollout-a", 3)
    assert continuation is not None
    assert continuation.boundary_index == 4
    assert continuation.last_committed_model_call_id == "call-1"
    assert continuation.resource_state_revisions == {"resources": 3}

    await participant.resume()
    await park
    await participant.finish(execution)


def test_restore_rejects_corrupted_boundary_before_activation(tmp_path) -> None:
    directory = tmp_path / AGENT_STATE_SUBDIR
    directory.mkdir()
    (directory / AGENT_MANIFEST_NAME).write_text(
        '{"schema_version": 1, "checkpoint_id": "checkpoint-1", "files": {"missing.json": "bad"}}'
    )
    participant = AgentCheckpointParticipant()
    with pytest.raises(AgentCheckpointError):
        restore_agent_state(participant, tmp_path)
    assert participant.continuation("rollout-a", 1) is None
