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
"""Stateful resources sessions use one lock and one atomic restore cut."""

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI, Response

from nemo_gym.checkpoint import (
    RESOURCE_STATE_REVISION_HEADER,
    ResourcesCheckpointError,
    ResourcesCheckpointParticipant,
    ResourceSnapshot,
    ResourcesSessionMiddleware,
    commit_resources_state,
    load_resources_state,
)
from nemo_gym.rollout_correlation import ATTEMPT_INDEX_HEADER, ROLLOUT_ID_HEADER


def _participant(state):
    async def export(rollout_id: str, attempt_index: int):
        return dict(state[(rollout_id, attempt_index)])

    async def restore(snapshots: list[ResourceSnapshot]):
        replacement = dict(state)
        for snapshot in snapshots:
            replacement[(snapshot.rollout_id, snapshot.attempt_index)] = dict(snapshot.state)
        state.clear()
        state.update(replacement)

    return ResourcesCheckpointParticipant(export_state=export, restore_states=restore)


@pytest.mark.asyncio
async def test_mutations_serialize_and_prepare_rejects_waiter() -> None:
    state = {("rollout-a", 0): {"value": 0}}
    participant = _participant(state)
    app = FastAPI()
    entered = asyncio.Event()
    release = asyncio.Event()

    @app.post("/mutate")
    async def mutate():
        entered.set()
        await release.wait()
        state[("rollout-a", 0)]["value"] += 1
        return state[("rollout-a", 0)]

    app.add_middleware(ResourcesSessionMiddleware, participant=participant)
    headers = {ROLLOUT_ID_HEADER: "rollout-a", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = asyncio.create_task(client.post("/mutate", headers=headers))
        await entered.wait()
        second = asyncio.create_task(client.post("/mutate", headers=headers))
        prepare = asyncio.create_task(participant.prepare(time.time() + 2))
        await asyncio.sleep(0)
        release.set()
        first_response, second_response, report = await asyncio.gather(first, second, prepare)

    assert first_response.status_code == 200
    assert first_response.headers[RESOURCE_STATE_REVISION_HEADER] == "1"
    assert second_response.status_code == 409
    assert report == {"sessions": 1, "state": "prepared"}
    assert participant.prepared_snapshots()[0].state == {"value": 1}


@pytest.mark.asyncio
async def test_restore_activates_replacement_attempt_as_one_set(tmp_path) -> None:
    state = {
        ("rollout-a", 0): {"value": 1},
        ("rollout-b", 2): {"value": 9},
    }
    source = _participant(state)
    source.register("rollout-a", 0)
    source.register("rollout-b", 2)
    source.record_mutation("rollout-a", 0)
    source.record_mutation("rollout-b", 2)
    await source.prepare(time.time() + 2)
    first = commit_resources_state(source, tmp_path, checkpoint_id="checkpoint-1", server_name="resources")
    second = commit_resources_state(source, tmp_path, checkpoint_id="checkpoint-1", server_name="resources")
    assert second == first

    checkpoint_id, snapshots = load_resources_state(tmp_path, server_name="resources")
    assert checkpoint_id == "checkpoint-1"
    restored_state = {}
    restored = _participant(restored_state)
    await restored.restore(snapshots)
    assert restored_state == {
        ("rollout-a", 1): {"value": 1},
        ("rollout-b", 3): {"value": 9},
    }
    assert restored.status()["state"] == "paused"

    app = FastAPI()

    @app.post("/mutate")
    async def mutate():
        return {"ok": True}

    app.add_middleware(ResourcesSessionMiddleware, participant=restored)
    stale_headers = {ROLLOUT_ID_HEADER: "rollout-a", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        stale = await client.post("/mutate", headers=stale_headers)
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_attempt"


@pytest.mark.asyncio
async def test_verify_retires_completed_session_before_prepare() -> None:
    state = {("rollout-a", 0): {"value": 1}}
    participant = _participant(state)
    app = FastAPI()

    @app.post("/verify")
    async def verify():
        state.pop(("rollout-a", 0))
        return {"reward": 1.0}

    app.add_middleware(ResourcesSessionMiddleware, participant=participant)
    headers = {ROLLOUT_ID_HEADER: "rollout-a", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/verify", headers=headers)
    assert response.status_code == 200
    assert await participant.prepare(time.time() + 2) == {"sessions": 0, "state": "prepared"}


@pytest.mark.asyncio
async def test_failed_verify_keeps_live_session_checkpointable() -> None:
    state = {("rollout-a", 0): {"value": 1}}
    participant = _participant(state)
    app = FastAPI()

    @app.post("/verify")
    async def verify():
        return Response(status_code=503)

    app.add_middleware(ResourcesSessionMiddleware, participant=participant)
    headers = {ROLLOUT_ID_HEADER: "rollout-a", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/verify", headers=headers)
    assert response.status_code == 503
    assert await participant.prepare(time.time() + 2) == {"sessions": 1, "state": "prepared"}
    assert participant.prepared_snapshots()[0].state == {"value": 1}


@pytest.mark.asyncio
async def test_failed_bulk_restore_does_not_activate_partial_state() -> None:
    state = {("existing", 0): {"value": 4}}

    async def export(rollout_id: str, attempt_index: int):
        return {}

    async def fail_restore(snapshots: list[ResourceSnapshot]):
        replacement = dict(state)
        replacement[(snapshots[0].rollout_id, snapshots[0].attempt_index)] = snapshots[0].state
        raise RuntimeError("validation failed")

    participant = ResourcesCheckpointParticipant(export_state=export, restore_states=fail_restore)
    snapshots = [
        ResourceSnapshot(rollout_id="rollout-a", attempt_index=0, state_revision=1, state={"value": 1}),
        ResourceSnapshot(rollout_id="rollout-b", attempt_index=0, state_revision=1, state={"value": 2}),
    ]
    with pytest.raises(RuntimeError, match="validation failed"):
        await participant.restore(snapshots)
    assert state == {("existing", 0): {"value": 4}}


@pytest.mark.asyncio
async def test_prepare_failure_keeps_admission_closed() -> None:
    async def export(rollout_id: str, attempt_index: int):
        raise RuntimeError("cannot serialize")

    async def restore(snapshots: list[ResourceSnapshot]):
        return None

    participant = ResourcesCheckpointParticipant(export_state=export, restore_states=restore)
    participant.register("rollout-a", 0)
    with pytest.raises(RuntimeError, match="cannot serialize"):
        await participant.prepare(time.time() + 2)
    assert not participant.accepting


def test_load_rejects_missing_manifest(tmp_path) -> None:
    with pytest.raises(ResourcesCheckpointError):
        load_resources_state(tmp_path, server_name="resources")
