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
import json
import time

import httpx
import pytest
from fastapi import FastAPI, Response

from nemo_gym._checkpoint import (
    RESOURCE_STATE_REVISION_HEADER,
    CheckpointPhase,
    ControlError,
    ControlFence,
    ResourcesCheckpointError,
    ResourcesCheckpointParticipant,
    ResourceSnapshot,
    ResourcesSessionMiddleware,
    ResourcesUnsafeRestoreError,
    commit_resources_state,
    install_resources_checkpoint,
    load_resources_state,
)
from nemo_gym.rollout_correlation import ATTEMPT_INDEX_HEADER, ROLLOUT_ID_HEADER, RolloutContextMiddleware


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
    participant.bind("rollout-a", 0)
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
    assert participant.prepared_snapshots()[0].state_revision == 1


@pytest.mark.asyncio
async def test_nonmutation_and_legacy_routes_are_not_globally_refused() -> None:
    participant = _participant({})
    app = FastAPI()

    @app.get("/")
    async def liveness():
        return {"ok": True}

    @app.post("/verify")
    async def verify():
        return {"reward": 1.0}

    route_kind = lambda path, method: "terminal" if method == "POST" and path == "/verify" else None
    app.add_middleware(ResourcesSessionMiddleware, participant=participant, route_kind=route_kind)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/")).status_code == 200
        assert (await client.get("/docs")).status_code == 200
        assert (await client.post("/verify")).status_code == 200


@pytest.mark.asyncio
async def test_identityless_state_creation_prevents_silent_checkpoint_omission() -> None:
    participant = _participant({})
    app = FastAPI()

    @app.post("/seed_session")
    async def seed():
        return {"ok": True}

    app.add_middleware(
        ResourcesSessionMiddleware,
        participant=participant,
        route_kind=lambda path, method: "start" if method == "POST" and path == "/seed_session" else None,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/seed_session")).status_code == 200
    with pytest.raises(ResourcesCheckpointError, match="without execution identity"):
        await participant.prepare(time.time() + 2)


@pytest.mark.asyncio
async def test_prefixed_verify_uses_context_and_retires_session() -> None:
    state = {("rollout-a", 0): {"value": 1}}
    participant = _participant(state)
    participant.bind("rollout-a", 0)
    app = FastAPI()

    @app.post("/verify")
    async def verify():
        state.pop(("rollout-a", 0))
        return {"reward": 1.0}

    route_kind = lambda path, method: "terminal" if method == "POST" and path == "/verify" else None
    app.add_middleware(ResourcesSessionMiddleware, participant=participant, route_kind=route_kind)
    app.add_middleware(RolloutContextMiddleware)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ng-rollout/rollout-a/verify")
    assert response.status_code == 200
    assert participant.status()["sessions"] == 0
    assert participant.status()["locked_sessions"] == 0


@pytest.mark.asyncio
async def test_failed_start_does_not_bind_and_unbound_mutation_is_rejected() -> None:
    state = {}
    participant = _participant(state)
    app = FastAPI()

    @app.post("/seed_session")
    async def seed():
        return Response(status_code=422)

    @app.post("/mutate")
    async def mutate():
        state[("rollout-a", 0)] = {"value": 1}
        return {"ok": True}

    def route_kind(path, method):
        if method != "POST":
            return None
        return "start" if path == "/seed_session" else "mutation" if path == "/mutate" else None

    app.add_middleware(ResourcesSessionMiddleware, participant=participant, route_kind=route_kind)
    headers = {ROLLOUT_ID_HEADER: "rollout-a", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/seed_session", headers=headers)).status_code == 422
        unbound = await client.post("/mutate", headers=headers)
    assert unbound.status_code == 409
    assert unbound.json()["error"]["code"] == "resources_session_unbound"
    assert state == {}
    assert await participant.prepare(time.time() + 2) == {"sessions": 0, "state": "prepared"}


@pytest.mark.asyncio
async def test_declared_read_is_serialized_without_advancing_revision() -> None:
    participant = _participant({("rollout-a", 0): {"value": 1}})
    participant.bind("rollout-a", 0)
    participant.record_mutation("rollout-a", 0)
    app = FastAPI()

    @app.post("/read")
    async def read():
        return {"value": 1}

    app.add_middleware(
        ResourcesSessionMiddleware,
        participant=participant,
        route_kind=lambda path, method: "read" if method == "POST" and path == "/read" else None,
    )
    headers = {ROLLOUT_ID_HEADER: "rollout-a", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/read", headers=headers)
    assert response.status_code == 200
    assert RESOURCE_STATE_REVISION_HEADER not in response.headers
    assert participant.revision_for("rollout-a", 0) == 1


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
    participant.bind("rollout-a", 0)
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
    participant.bind("rollout-a", 0)
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
async def test_prepare_skips_session_retired_while_waiting_for_lock() -> None:
    state = {("rollout-a", 0): {"value": 1}}
    participant = _participant(state)
    participant.bind("rollout-a", 0)
    app = FastAPI()
    entered = asyncio.Event()
    release = asyncio.Event()

    @app.post("/verify")
    async def verify():
        entered.set()
        await release.wait()
        state.pop(("rollout-a", 0))
        return {"reward": 1.0}

    app.add_middleware(ResourcesSessionMiddleware, participant=participant)
    headers = {ROLLOUT_ID_HEADER: "rollout-a", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        verify_task = asyncio.create_task(client.post("/verify", headers=headers))
        await entered.wait()
        prepare_task = asyncio.create_task(participant.prepare(time.time() + 2))
        await asyncio.sleep(0)
        release.set()
        response, report = await asyncio.gather(verify_task, prepare_task)
    assert response.status_code == 200
    assert report == {"sessions": 0, "state": "prepared"}


@pytest.mark.asyncio
async def test_restore_expected_starts_paused_and_restore_requires_fresh_process() -> None:
    snapshots = [ResourceSnapshot(rollout_id="rollout-a", attempt_index=0, state_revision=3, state={"value": 7})]
    state = {}

    async def export(rollout_id: str, attempt_index: int):
        return dict(state[(rollout_id, attempt_index)])

    async def restore(restored_snapshots: list[ResourceSnapshot]):
        for snapshot in restored_snapshots:
            state[(snapshot.rollout_id, snapshot.attempt_index)] = dict(snapshot.state)

    participant = ResourcesCheckpointParticipant(
        export_state=export,
        restore_states=restore,
        restore_expected=True,
    )
    assert not participant.accepting
    with pytest.raises(ResourcesCheckpointError, match="has not restored"):
        participant.resume()
    await participant.restore(snapshots)
    assert not participant.accepting
    assert participant.resume() == {"state": "accepting"}
    assert state[("rollout-a", 1)] == {"value": 7}

    served = _participant({("other", 0): {"value": 1}})
    served.bind("other", 0)
    with pytest.raises(ResourcesUnsafeRestoreError, match="fresh process"):
        await served.restore(snapshots)


@pytest.mark.asyncio
async def test_lock_pruned_after_terminal_release_while_tombstone_remains() -> None:
    source_state = {("rollout-a", 0): {"value": 1}}
    source = _participant(source_state)
    source.bind("rollout-a", 0)
    source.record_mutation("rollout-a", 0)
    await source.prepare(time.time() + 2)

    restored = _participant({})
    await restored.restore(source.prepared_snapshots())
    assert restored.is_tombstoned("rollout-a", 0)
    restored.retire("rollout-a", 1)
    assert restored.status()["sessions"] == 0
    assert restored.status()["lock_entries"] == 0
    assert restored.status()["tombstones"] == 1


@pytest.mark.asyncio
async def test_retire_after_prepare_removes_session_from_prepared_cut() -> None:
    participant = _participant({("rollout-a", 0): {"value": 1}})
    participant.bind("rollout-a", 0)
    participant.record_mutation("rollout-a", 0)
    await participant.prepare(time.time() + 2)
    assert len(participant.prepared_snapshots()) == 1

    await participant.retire_execution("rollout-a", 0)

    assert participant.prepared_snapshots() == []
    assert participant.is_tombstoned("rollout-a", 0)
    assert participant.status()["lock_entries"] == 0


@pytest.mark.asyncio
async def test_status_and_retire_routes_require_active_checkpoint() -> None:
    participant = _participant({("rollout-a", 0): {"value": 1}})
    participant.bind("rollout-a", 0)
    participant.record_mutation("rollout-a", 0)
    fence = ControlFence()

    async def begin_prepare():
        return {"state": "preparing"}

    await fence.run_operation(
        "checkpoint-active",
        "test/begin-prepare",
        allowed_phases=frozenset({CheckpointPhase.IDLE}),
        phase_during=CheckpointPhase.PREPARING,
        phase_after=CheckpointPhase.PREPARING,
        run=begin_prepare,
    )
    app = FastAPI()

    @app.exception_handler(ControlError)
    async def control_error_handler(_request, exc):
        return Response(
            content=json.dumps({"error": {"code": exc.code, "detail": exc.detail}}),
            media_type="application/json",
            status_code=exc.status_code,
        )

    install_resources_checkpoint(
        app,
        participant=participant,
        fence=fence,
        auth_token="secret",
        server_name="resources",
        route_kind=lambda path, method: None,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        unauthenticated = await client.get(
            "/ng-control/v1/resources-checkpoint/status",
            params={"checkpoint_id": "checkpoint-active", "deadline_ts": time.time() + 2},
        )
        async with participant.mutation_lock("rollout-a", 0):
            status = await client.get(
                "/ng-control/v1/resources-checkpoint/status",
                headers={"authorization": "Bearer secret"},
                params={"checkpoint_id": "checkpoint-active", "deadline_ts": time.time() + 2},
            )
        wrong_checkpoint = await client.post(
            "/ng-control/v1/resources-checkpoint/retire",
            headers={"authorization": "Bearer secret"},
            json={
                "checkpoint_id": "checkpoint-other",
                "deadline_ts": time.time() + 2,
                "rollout_id": "rollout-a",
                "attempt_index": 0,
            },
        )
        assert participant.is_bound("rollout-a", 0)
        retire = await client.post(
            "/ng-control/v1/resources-checkpoint/retire",
            headers={"authorization": "Bearer secret"},
            json={
                "checkpoint_id": "checkpoint-active",
                "deadline_ts": time.time() + 2,
                "rollout_id": "rollout-a",
                "attempt_index": 0,
            },
        )
    assert unauthenticated.status_code == 401
    assert status.status_code == 200
    assert status.json()["state"] == "accepting"
    assert status.json()["per_session"] == [
        {"rollout_id": "rollout-a", "attempt_index": 0, "revision": 1, "locked": True}
    ]
    assert wrong_checkpoint.status_code == 409
    assert wrong_checkpoint.json()["error"]["code"] == "checkpoint_conflict"
    assert retire.status_code == 200
    assert retire.json()["retired"] is True
    assert participant.status()["lock_entries"] == 0
    assert participant.status()["tombstones"] == 1


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
