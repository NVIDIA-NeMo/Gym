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
    EXPECTED_RESOURCE_STATE_REVISION_HEADER,
    RESOURCE_REQUEST_ID_HEADER,
    RESOURCE_STATE_REVISION_HEADER,
    AgentBoundaryRecord,
    AgentCheckpointParticipant,
    CheckpointPhase,
    ControlError,
    ControlFence,
    ResourceRevisionMismatchError,
    ResourcesCheckpointError,
    ResourcesCheckpointParticipant,
    ResourceSnapshot,
    ResourcesSessionMiddleware,
    ResourcesUncertainMutationError,
    ResourcesUnsafeRestoreError,
    commit_agent_state,
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
    headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: "mutation-1",
    }
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
    assert report == {
        "sessions": 1,
        "state": "prepared",
        "inventory": [{"rollout_id": "rollout-a", "attempt_index": 0, "revision": 1, "mutation_receipts": 1}],
    }
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
        response = await client.post(
            "/ng-rollout/rollout-a/verify",
            headers={RESOURCE_REQUEST_ID_HEADER: "verify-prefixed-1"},
        )
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
    headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: "start-or-mutation-1",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/seed_session", headers=headers)).status_code == 422
        unbound = await client.post(
            "/mutate",
            headers={**headers, RESOURCE_REQUEST_ID_HEADER: "unbound-mutation-1"},
        )
    assert unbound.status_code == 409
    assert unbound.json()["error"]["code"] == "resources_session_unbound"
    assert state == {}
    assert await participant.prepare(time.time() + 2) == {
        "sessions": 0,
        "state": "prepared",
        "inventory": [],
    }


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
    headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: "read-1",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/read", headers=headers)
    assert response.status_code == 200
    assert RESOURCE_STATE_REVISION_HEADER not in response.headers
    assert participant.revision_for("rollout-a", 0) == 1


@pytest.mark.asyncio
async def test_expected_revision_rejects_stale_mutation_before_handler() -> None:
    state = {("rollout-a", 0): {"value": 1}}
    participant = _participant(state)
    participant.bind("rollout-a", 0)
    participant.record_mutation("rollout-a", 0)
    calls = 0
    app = FastAPI()

    @app.post("/mutate")
    async def mutate():
        nonlocal calls
        calls += 1
        state[("rollout-a", 0)]["value"] += 1
        return state[("rollout-a", 0)]

    app.add_middleware(ResourcesSessionMiddleware, participant=participant)
    identity = {ROLLOUT_ID_HEADER: "rollout-a", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        stale = await client.post(
            "/mutate",
            headers={
                **identity,
                EXPECTED_RESOURCE_STATE_REVISION_HEADER: "0",
                RESOURCE_REQUEST_ID_HEADER: "request-stale",
            },
        )
        accepted = await client.post(
            "/mutate",
            headers={
                **identity,
                EXPECTED_RESOURCE_STATE_REVISION_HEADER: "1",
                RESOURCE_REQUEST_ID_HEADER: "request-current",
            },
        )

    assert stale.status_code == 409
    assert stale.json()["error"] | {"message": stale.json()["error"]["message"]} == {
        "code": "resource_revision_mismatch",
        "message": stale.json()["error"]["message"],
        "expected_revision": 0,
        "actual_revision": 1,
    }
    assert accepted.status_code == 200
    assert accepted.headers[RESOURCE_STATE_REVISION_HEADER] == "2"
    assert calls == 1


@pytest.mark.asyncio
async def test_checkpoint_managed_mutation_requires_request_id() -> None:
    participant = _participant({("rollout-a", 0): {"value": 0}})
    participant.bind("rollout-a", 0)
    calls = 0
    app = FastAPI()

    @app.post("/mutate")
    async def mutate():
        nonlocal calls
        calls += 1
        return {"ok": True}

    app.add_middleware(ResourcesSessionMiddleware, participant=participant)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/mutate",
            headers={ROLLOUT_ID_HEADER: "rollout-a", ATTEMPT_INDEX_HEADER: "0"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "resource_request_receipt_error"
    assert calls == 0


@pytest.mark.asyncio
async def test_uncertain_mutation_response_replays_exact_receipt_after_commit_restore(tmp_path) -> None:
    source_state = {("rollout-a", 0): {"value": 0}}
    source = _participant(source_state)
    source.bind("rollout-a", 0)
    calls = 0

    def build_app(participant, state, key):
        app = FastAPI()

        @app.post("/mutate")
        async def mutate():
            nonlocal calls
            calls += 1
            state[key]["value"] += 1
            response = Response(
                content=json.dumps({"value": state[key]["value"]}),
                media_type="application/json",
                status_code=201,
                headers={"x-original-response": "retained"},
            )
            response.set_cookie("resource-session", "cookie-1")
            return response

        app.add_middleware(ResourcesSessionMiddleware, participant=participant)
        return app

    source_app = build_app(source, source_state, ("rollout-a", 0))
    request_headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        EXPECTED_RESOURCE_STATE_REVISION_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: "simple-agent-action-1",
        "cookie": "b=2; a=1",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=source_app), base_url="http://test") as client:
        original = await client.post("/mutate", json={"action": "increment"}, headers=request_headers)

    assert original.status_code == 201
    assert original.headers[RESOURCE_STATE_REVISION_HEADER] == "1"
    await source.prepare(time.time() + 2)
    snapshot = source.prepared_snapshots()[0]
    assert snapshot.mutation_receipts[0].request_id == "simple-agent-action-1"
    commit_resources_state(source, tmp_path, checkpoint_id="checkpoint-1", server_name="resources")

    _, snapshots = load_resources_state(tmp_path, server_name="resources")
    restored_state = {}
    restored = _participant(restored_state)
    await restored.restore(snapshots)
    restored.resume()
    restored_app = build_app(restored, restored_state, ("rollout-a", 1))
    replay_headers = {**request_headers, ATTEMPT_INDEX_HEADER: "1", "cookie": "a=1; b=2"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restored_app), base_url="http://test") as client:
        replay = await client.post("/mutate", json={"action": "increment"}, headers=replay_headers)
        wrong_cookie = await client.post(
            "/mutate",
            json={"action": "increment"},
            headers={**replay_headers, "cookie": "a=changed; b=2"},
        )
        wrong_input = await client.post("/mutate", json={"action": "decrement"}, headers=replay_headers)
        wrong_revision = await client.post(
            "/mutate",
            json={"action": "increment"},
            headers={**replay_headers, EXPECTED_RESOURCE_STATE_REVISION_HEADER: "1"},
        )

    assert calls == 1
    assert restored_state[("rollout-a", 1)] == {"value": 1}
    assert replay.status_code == original.status_code
    assert replay.content == original.content
    assert replay.headers[RESOURCE_STATE_REVISION_HEADER] == original.headers[RESOURCE_STATE_REVISION_HEADER]
    assert replay.headers["x-original-response"] == "retained"
    assert replay.cookies["resource-session"] == "cookie-1"
    assert wrong_cookie.status_code == 409
    assert wrong_cookie.json()["error"]["code"] == "resource_request_receipt_error"
    assert wrong_input.status_code == 409
    assert wrong_input.json()["error"]["code"] == "resource_request_receipt_error"
    assert wrong_revision.status_code == 409
    assert wrong_revision.json()["error"]["code"] == "resource_request_receipt_error"


@pytest.mark.asyncio
async def test_receipt_capacity_fails_closed_until_session_retirement() -> None:
    state = {("rollout-a", 0): {"value": 0}}

    async def export(rollout_id: str, attempt_index: int):
        return dict(state[(rollout_id, attempt_index)])

    async def restore(snapshots: list[ResourceSnapshot]):
        return None

    participant = ResourcesCheckpointParticipant(
        export_state=export,
        restore_states=restore,
        max_receipts_per_session=1,
    )
    participant.bind("rollout-a", 0)
    calls = 0
    app = FastAPI()

    @app.post("/mutate")
    async def mutate():
        nonlocal calls
        calls += 1
        state[("rollout-a", 0)]["value"] += 1
        return state[("rollout-a", 0)]

    app.add_middleware(ResourcesSessionMiddleware, participant=participant)
    identity = {ROLLOUT_ID_HEADER: "rollout-a", ATTEMPT_INDEX_HEADER: "0"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/mutate",
            headers={
                **identity,
                EXPECTED_RESOURCE_STATE_REVISION_HEADER: "0",
                RESOURCE_REQUEST_ID_HEADER: "request-1",
            },
        )
        exhausted = await client.post(
            "/mutate",
            headers={
                **identity,
                EXPECTED_RESOURCE_STATE_REVISION_HEADER: "1",
                RESOURCE_REQUEST_ID_HEADER: "request-2",
            },
        )

    assert first.status_code == 200
    assert exhausted.status_code == 409
    assert exhausted.json()["error"]["code"] == "resource_receipt_capacity_exhausted"
    assert calls == 1
    assert participant.status()["mutation_receipts"] == 1
    await participant.retire_execution("rollout-a", 0)
    assert participant.status()["mutation_receipts"] == 0


@pytest.mark.asyncio
async def test_evicted_reset_receipt_leaves_fail_closed_tombstone() -> None:
    state = {}

    async def export(rollout_id: str, attempt_index: int):
        return dict(state[(rollout_id, attempt_index)])

    async def restore(snapshots: list[ResourceSnapshot]):
        return None

    participant = ResourcesCheckpointParticipant(
        export_state=export,
        restore_states=restore,
        max_retired_receipt_sessions=0,
    )
    calls = 0
    app = FastAPI()

    @app.post("/reset")
    async def reset():
        nonlocal calls
        calls += 1
        state[("rollout-a", 0)] = {"value": calls}
        return state[("rollout-a", 0)]

    app.add_middleware(
        ResourcesSessionMiddleware,
        participant=participant,
        route_kind=lambda path, method: "start" if path == "/reset" and method == "POST" else None,
    )
    headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: "reset-1",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        original = await client.post("/reset", json={"seed": 1}, headers=headers)
        participant.retire("rollout-a", 0, preserve_receipts=True)
        replay = await client.post("/reset", json={"seed": 1}, headers=headers)

    assert original.status_code == 200
    assert replay.status_code == 409
    assert replay.json()["error"]["code"] == "resource_receipt_evicted"
    assert calls == 1
    assert participant.status()["mutation_receipts"] == 0
    assert participant.status()["evicted_receipt_tombstones"] == 1


@pytest.mark.parametrize(("path", "raises"), [("/mutate", True), ("/verify", False)])
@pytest.mark.asyncio
async def test_incomplete_streaming_mutation_poisoned_before_metadata_publish(path: str, raises: bool) -> None:
    state = {("rollout-a", 0): {"value": 0}}
    participant = _participant(state)
    participant.bind("rollout-a", 0)
    calls = 0

    async def incomplete_app(scope, receive, send):
        nonlocal calls
        calls += 1
        state[("rollout-a", 0)]["value"] += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"partial":', "more_body": True})
        if raises:
            raise RuntimeError("stream failed")

    app = ResourcesSessionMiddleware(incomplete_app, participant)
    headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        EXPECTED_RESOURCE_STATE_REVISION_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: f"uncertain-{path}",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        if raises:
            with pytest.raises(RuntimeError, match="stream failed"):
                await client.post(path, headers=headers)
        else:
            incomplete = await client.post(path, headers=headers)
            assert incomplete.status_code == 409
            assert incomplete.json()["error"]["code"] == "resource_mutation_uncertain"
        retry = await client.post(path, headers=headers)

    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "resource_mutation_uncertain"
    assert calls == 1
    assert participant.revision_for("rollout-a", 0) == 0
    assert participant.status()["mutation_receipts"] == 0
    assert len(participant.status()["uncertain_sessions"]) == 1
    with pytest.raises(ResourcesUncertainMutationError):
        await participant.prepare(time.time() + 2)


@pytest.mark.asyncio
async def test_egress_failure_after_receipt_replays_without_mutation() -> None:
    state = {("rollout-a", 0): {"value": 0}}
    participant = _participant(state)
    participant.bind("rollout-a", 0)
    calls = 0
    inner = FastAPI()

    @inner.post("/mutate")
    async def mutate():
        nonlocal calls
        calls += 1
        state[("rollout-a", 0)]["value"] += 1
        return state[("rollout-a", 0)]

    inner.add_middleware(ResourcesSessionMiddleware, participant=participant)
    fail_egress = True

    async def app(scope, receive, send):
        async def maybe_fail(message):
            nonlocal fail_egress
            if fail_egress:
                fail_egress = False
                raise RuntimeError("transport lost after durable receipt")
            await send(message)

        await inner(scope, receive, maybe_fail)

    headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        EXPECTED_RESOURCE_STATE_REVISION_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: "egress-failure-1",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        with pytest.raises(RuntimeError, match="transport lost"):
            await client.post("/mutate", json={"action": "increment"}, headers=headers)
        replay = await client.post("/mutate", json={"action": "increment"}, headers=headers)

    assert replay.status_code == 200
    assert replay.json() == {"value": 1}
    assert replay.headers[RESOURCE_STATE_REVISION_HEADER] == "1"
    assert calls == 1
    assert participant.revision_for("rollout-a", 0) == 1
    assert participant.status()["uncertain_sessions"] == []


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
    stale_headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: "stale-1",
    }
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
    headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: "verify-1",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/verify", headers=headers)
    assert response.status_code == 200
    assert await participant.prepare(time.time() + 2) == {
        "sessions": 0,
        "state": "prepared",
        "inventory": [],
    }


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
    headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: "verify-failed-1",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/verify", headers=headers)
    assert response.status_code == 503
    assert await participant.prepare(time.time() + 2) == {
        "sessions": 1,
        "state": "prepared",
        "inventory": [{"rollout_id": "rollout-a", "attempt_index": 0, "revision": 0, "mutation_receipts": 1}],
    }
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
    headers = {
        ROLLOUT_ID_HEADER: "rollout-a",
        ATTEMPT_INDEX_HEADER: "0",
        RESOURCE_REQUEST_ID_HEADER: "verify-waiting-1",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        verify_task = asyncio.create_task(client.post("/verify", headers=headers))
        await entered.wait()
        prepare_task = asyncio.create_task(participant.prepare(time.time() + 2))
        await asyncio.sleep(0)
        release.set()
        response, report = await asyncio.gather(verify_task, prepare_task)
    assert response.status_code == 200
    assert report == {"sessions": 0, "state": "prepared", "inventory": []}


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


@pytest.mark.asyncio
async def test_restore_rejects_agent_resources_revision_mismatch(tmp_path) -> None:
    resources = _participant({("rollout-a", 0): {"value": 1}})
    resources.bind("rollout-a", 0)
    resources.record_mutation("rollout-a", 0)
    await resources.prepare(time.time() + 2)
    commit_resources_state(resources, tmp_path, checkpoint_id="checkpoint-1", server_name="resources")

    agent = AgentCheckpointParticipant("agent")
    execution = await agent.begin("rollout-a", 0, task=asyncio.current_task())
    prepare = asyncio.create_task(agent.prepare(time.time() + 2))
    await asyncio.sleep(0)
    park = asyncio.create_task(
        agent.commit_boundary(
            execution,
            AgentBoundaryRecord(
                rollout_id="rollout-a",
                attempt_index=0,
                boundary_index=1,
                turn_index=1,
                output_items=[],
                resource_state_revisions={"resources": 2},
            ),
        )
    )
    await prepare
    commit_agent_state(agent, tmp_path, checkpoint_id="checkpoint-1")

    with pytest.raises(ResourceRevisionMismatchError, match="does not match"):
        load_resources_state(tmp_path, server_name="resources")

    await agent.resume()
    await park
