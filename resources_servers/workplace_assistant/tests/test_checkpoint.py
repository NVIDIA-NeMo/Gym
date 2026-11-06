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
"""Workplace Assistant sessions checkpoint through the real control routes and continue in a fresh server."""

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request
from omegaconf import DictConfig

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
)
from resources_servers.workplace_assistant import app as workplace_app
from resources_servers.workplace_assistant.app import (
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
    _export_tool_env,
)


AUTH = {"authorization": "Bearer t"}
SEED = {"responses_create_params": {"input": "hi"}}
# Every restore below continues the episodes its source checkpoint exported.
SCOPE = [{"rollout_id": "r"}, {"rollout_id": "q"}]

# Steps before the checkpoint mutate three different tables, one of which (the CRM) holds empty CSV cells.
BEFORE = [
    (
        "calendar_create_event",
        {
            "event_name": "Checkpoint sync",
            "participant_email": "raj.patel@atlas.com",
            "event_start": "2023-12-01 10:00:00",
            "duration": "30",
        },
    ),
    (
        "customer_relationship_manager_add_customer",
        {"customer_name": "Checkpoint Corp", "assigned_to_email": "raj.patel@atlas.com", "status": "Lead"},
    ),
    ("email_send_email", {"recipient": "raj.patel@atlas.com", "subject": "Checkpoint", "body": "Before the cut"}),
]
# Steps after the checkpoint only give the same answers if the earlier mutations survived.
AFTER = [
    (
        "calendar_create_event",
        {
            "event_name": "Checkpoint sync",
            "participant_email": "raj.patel@atlas.com",
            "event_start": "2023-12-02 10:00:00",
            "duration": "45",
        },
    ),
    ("calendar_search_events", {"query": "Checkpoint sync"}),
    ("customer_relationship_manager_search_customers", {"customer_name": "Checkpoint Corp"}),
    ("email_search_emails", {"query": "Before the cut"}),
]


def make_server() -> tuple[WorkbenchResourcesServer, FastAPI]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = WorkbenchResourcesServerConfig(host="", port=0, entrypoint="", name="workplace_assistant")
    server = WorkbenchResourcesServer(config=config, server_client=server_client)
    return server, server.setup_webserver()


def client_for(app: FastAPI, cookies: dict[str, str] | None = None) -> httpx.AsyncClient:
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r")
    client.cookies.update(cookies or {})
    return client


def control(checkpoint_id: str = "c1", **extra: Any) -> dict[str, Any]:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


async def call_tools(
    client: httpx.AsyncClient, steps: list[tuple[str, dict[str, str]]], capture_key: str
) -> list[Any]:
    """Call the tools as the attempt ``capture_key``: with checkpointing on, each call names its attempt."""
    outputs = []
    for name, arguments in steps:
        response = await client.post(f"/ng-rollout/{capture_key}/{name}", json=arguments)
        assert response.status_code == 200, response.text
        outputs.append(response.json()["output"])
    return outputs


def verify_body() -> dict[str, Any]:
    calls = [
        {"type": "function_call", "call_id": f"c{i}", "name": name, "arguments": json.dumps(arguments)}
        for i, (name, arguments) in enumerate(BEFORE + AFTER[:1])
    ]
    return {
        "responses_create_params": {"input": "hi"},
        "response": {
            "id": "resp",
            "created_at": 0.0,
            "model": "m",
            "object": "response",
            "output": calls,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
        "ground_truth": [{"name": name, "arguments": json.dumps(arguments)} for name, arguments in BEFORE + AFTER[:1]],
        "id": 0,
        "category": "workplace_assistant_calendar",
        "environment_name": "workplace_assistant",
    }


async def checkpoint(client: httpx.AsyncClient, checkpoint_dir: Path) -> dict[str, Any]:
    prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
    assert prepared.json()["phase"] == "prepared"
    commit = await client.post(
        "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(checkpoint_dir)), headers=AUTH
    )
    assert commit.status_code == 200, commit.text
    return commit.json()


async def restore(client: httpx.AsyncClient, checkpoint_dir: Path) -> httpx.Response:
    return await client.post(
        "/ng-control/v1/checkpoint/restore",
        json=control("r1", checkpoint_dir=str(checkpoint_dir), episode_ids=SCOPE),
        headers=AUTH,
    )


async def test_a_restored_session_continues_exactly_like_an_uncheckpointed_run(tmp_path: Path) -> None:
    reference, reference_app = make_server()
    async with client_for(reference_app) as client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await call_tools(client, BEFORE, "r")
        expected_outputs = await call_tools(client, AFTER, "r")
        [expected_tables] = [_export_tool_env(env) for env in reference.session_id_to_tool_env.values()]
        expected_verify = (await client.post("/ng-rollout/r/verify", json=verify_body())).json()

    _, source_app = make_server()
    async with client_for(source_app) as client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await call_tools(client, BEFORE, "r")
        commit = await checkpoint(client, tmp_path)
        cookies = dict(client.cookies)

    restored, restored_app = make_server()
    async with client_for(restored_app, cookies) as client:
        assert (await restore(client, tmp_path)).json()["phase"] == "restored"
        await client.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        outputs = await call_tools(client, AFTER, "r-a1")
        [tables] = [_export_tool_env(env) for env in restored.session_id_to_tool_env.values()]
        # The replacement attempt verifies, as the restored session's owner.
        verified = (await client.post("/ng-rollout/r-a1/verify", json=verify_body())).json()
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()

    assert commit["manifest"]["record_count"] == 1
    # The second event's ID follows the first one's, and the searches find what was written before the cut.
    assert outputs == expected_outputs
    assert outputs[0] == "00000301"
    assert outputs[1]["pagination"]["total_events"] == 2
    assert outputs[2]["pagination"]["total_customers"] == 1
    assert tables == expected_tables
    assert verified["reward"] == expected_verify["reward"] == 1.0
    # Verify ends the session and discards its tool environment.
    assert restored.session_id_to_tool_env == {}
    assert status["report"]["counts"]["sessions"] == 0


async def test_invalid_restored_state_installs_nothing(tmp_path: Path) -> None:
    source, source_app = make_server()
    async with client_for(source_app) as healthy, client_for(source_app) as broken:
        await healthy.post("/ng-rollout/r/seed_session", json=SEED)
        await broken.post("/ng-rollout/q/seed_session", json=SEED)
        await call_tools(healthy, BEFORE, "r")
        broken_env = source.session_id_to_tool_env[next(reversed(source.session_id_to_tool_env))]
        del broken_env["containers"]["analytics"]._plots_data
        commit = await checkpoint(healthy, tmp_path)
        cookies = dict(healthy.cookies)

    restored, restored_app = make_server()
    async with client_for(restored_app, cookies) as client:
        with pytest.raises(ValueError, match="invalid workplace assistant state.*analytics tables"):
            await restore(client, tmp_path)
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()

    assert commit["manifest"]["record_count"] == 2
    assert restored.session_id_to_tool_env == {}
    assert status["phase"] == "idle" and status["report"]["counts"]["sessions"] == 0


@pytest.mark.parametrize("reseed", [False, True])
async def test_restored_typed_session_keeps_its_binding_and_verify_lifetime(tmp_path: Path, reseed: bool) -> None:
    task_data = {
        key: value for key, value in verify_body().items() if key not in ("response", "responses_create_params")
    }
    seed = ResourcesSeedSessionRequest(
        resources_session_id="typed-session",
        episode_id=EpisodeId(rollout_id="r"),
        task_id=TaskId(taskset="workplace_assistant", task_id="0"),
        task_data=task_data,
    )
    _, source_app = make_server()
    async with client_for(source_app) as client:
        assert (await client.post("/ng-rollout/r/seed_session", json=seed.model_dump(mode="json"))).status_code == 200
        await call_tools(client, BEFORE, "r")
        await checkpoint(client, tmp_path)
        cookies = dict(client.cookies)

    restored, restored_app = make_server()
    replacement = seed.model_copy(update={"episode_id": EpisodeId(rollout_id="r", attempt=1)})
    async with client_for(restored_app, cookies) as client:
        assert (await restore(client, tmp_path)).json()["phase"] == "restored"
        await client.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        if reseed:
            # Restoring gives ownership to the replacement attempt, never to a different task or the old attempt.
            for conflicting in (
                seed,
                replacement.model_copy(update={"task_id": TaskId(taskset="workplace_assistant", task_id="other")}),
            ):
                refused = await client.post("/ng-rollout/r-a1/seed_session", json=conflicting.model_dump(mode="json"))
                assert refused.status_code == 409, refused.text
            wrong_close = await client.post(
                "/ng-rollout/r-a1/close_session",
                json={"resources_session_id": seed.resources_session_id, "episode_id": seed.episode_id.model_dump()},
            )
            assert wrong_close.status_code == 409, wrong_close.text
            repeated = await client.post("/ng-rollout/r-a1/seed_session", json=replacement.model_dump(mode="json"))
            assert repeated.status_code == 200, repeated.text
        assert await call_tools(client, AFTER[:1], "r-a1") == ["00000301"]
        for _ in range(2):
            verified = (await client.post("/ng-rollout/r-a1/verify", json=verify_body())).json()
            assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
        assert restored.session_id_to_tool_env == {} and restored._session_episodes == {}
        client.cookies.clear()
        closed = await client.post(
            "/ng-rollout/r-a1/close_session",
            json={
                "resources_session_id": seed.resources_session_id,
                "episode_id": replacement.episode_id.model_dump(),
            },
        )
        assert closed.status_code == 200, closed.text
    assert restored.session_id_to_tool_env == {}
    assert restored._session_episodes == {}


async def test_a_retired_session_is_gone(tmp_path: Path) -> None:
    server, app = make_server()
    async with client_for(app) as client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await call_tools(client, BEFORE, "r")
        await client.post(
            "/ng-control/v1/checkpoint/retire", json=control(episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        commit = await checkpoint(client, tmp_path)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        stale = await client.post("/calendar_search_events", json={"query": "Checkpoint sync"})

    assert commit["manifest"]["record_count"] == 0
    # The retire stopped and released the session; a late call is not served from it and does not recreate it.
    assert stale.status_code >= 400
    assert server.session_id_to_tool_env == {}


async def test_a_session_dropped_by_a_failed_verify_is_left_out_of_the_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> bool:
        raise RuntimeError("verifier failed")

    source, source_app = make_server()
    async with client_for(source_app) as live, client_for(source_app) as failed:
        await live.post("/ng-rollout/r/seed_session", json=SEED)
        await failed.post("/ng-rollout/q/seed_session", json=SEED)
        await call_tools(live, BEFORE, "r")
        await call_tools(failed, BEFORE, "q")
        with monkeypatch.context() as patch:
            patch.setattr(workplace_app, "is_correct", fail)
            with pytest.raises(RuntimeError, match="verifier failed"):
                await failed.post("/ng-rollout/q/verify", json=verify_body())
        commit = await checkpoint(live, tmp_path)
        live_cookies, failed_cookies = dict(live.cookies), dict(failed.cookies)

    restored, restored_app = make_server()
    async with client_for(restored_app, live_cookies) as live, client_for(restored_app, failed_cookies) as failed:
        await restore(live, tmp_path)
        await live.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        continued = await call_tools(live, AFTER[:1], "r-a1")
        gone = await failed.post("/calendar_search_events", json={"query": "Checkpoint sync"})

    assert len(source.session_id_to_tool_env) == 1
    assert commit["episode_ids"] == ["r"]
    assert commit["manifest"]["record_count"] == 1
    assert len(restored.session_id_to_tool_env) == 1
    assert continued == ["00000301"]
    assert gone.status_code == 404


def typed_seed(session_id: str = "typed-session", rollout_id: str = "r") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=rollout_id),
        task_id=TaskId(taskset="workplace_assistant", task_id="0"),
        task_data={
            key: value for key, value in verify_body().items() if key not in ("response", "responses_create_params")
        },
    )


async def test_retire_releases_typed_binding() -> None:
    server, webserver = make_server()
    async with client_for(webserver) as client:
        assert (
            await client.post("/ng-rollout/r/seed_session", json=typed_seed().model_dump(mode="json"))
        ).status_code == 200
        await call_tools(client, BEFORE, "r")
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire", json=control(episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        assert retired.status_code == 200, retired.text
    assert server.session_id_to_tool_env == {} and server._session_episodes == {}


@pytest.mark.parametrize("typed", [False, True])
async def test_verify_failure_releases_tables_and_can_retry(monkeypatch: pytest.MonkeyPatch, typed: bool) -> None:
    server, webserver = make_server()
    async with client_for(webserver) as client:
        await client.post("/ng-rollout/r/seed_session", json=typed_seed().model_dump(mode="json") if typed else SEED)
        await call_tools(client, BEFORE + AFTER[:1], "r")
        with monkeypatch.context() as patch:
            patch.setattr(workplace_app, "is_correct", MagicMock(side_effect=RuntimeError("verifier failed")))
            with pytest.raises(RuntimeError, match="verifier failed"):
                await client.post("/ng-rollout/r/verify", json=verify_body())
        assert server.session_id_to_tool_env == {} and server._session_episodes == {}
        retried = await client.post("/ng-rollout/r/verify", json=verify_body())
        assert (retried.json()["reward"], retried.json()["mask_sample"]) == (1.0, False)


async def test_verify_releases_typed_state_before_checkpoint_ownership_ends() -> None:
    server, webserver = make_server()
    async with client_for(webserver) as client:
        await client.post("/ng-rollout/r/seed_session", json=typed_seed().model_dump(mode="json"))
        await call_tools(client, BEFORE + AFTER[:1], "r")
        verified = await client.post(
            "/ng-rollout/r/verify", json=verify_body(), headers={"x-ng-checkpoint-close-follows": "1"}
        )
        assert verified.json()["reward"] == 1.0
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()
        assert status["report"]["counts"]["sessions"] == 0
        # There is no later retire callback once wait-verify unbinds the session.
        assert server.session_id_to_tool_env == {} and server._session_episodes == {}


async def test_mixed_checkpoint_restores_both_seed_protocols(tmp_path: Path) -> None:
    source, source_app = make_server()
    seed = typed_seed(rollout_id="q")
    async with client_for(source_app) as legacy, client_for(source_app) as typed:
        await legacy.post("/ng-rollout/r/seed_session", json=SEED)
        await typed.post("/ng-rollout/q/seed_session", json=seed.model_dump(mode="json"))
        await call_tools(legacy, BEFORE, "r")
        await call_tools(typed, BEFORE, "q")
        states = await source.export_session_states(list(source.session_id_to_tool_env))
        assert all(set(state) == {"tool_env", "episode_id", "task_id"} for state in states.values())
        assert sum(state["episode_id"] is None for state in states.values()) == 1
        assert (await checkpoint(legacy, tmp_path))["manifest"]["record_count"] == 2
        legacy_cookies, typed_cookies = dict(legacy.cookies), dict(typed.cookies)
    restored, restored_app = make_server()
    async with client_for(restored_app, legacy_cookies) as legacy, client_for(restored_app, typed_cookies) as typed:
        assert (await restore(legacy, tmp_path)).json()["phase"] == "restored"
        await legacy.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        assert restored._session_episodes == {
            seed.resources_session_id: (EpisodeId(rollout_id="q", attempt=1), seed.task_id)
        }
        for client, attempt in ((legacy, "r-a1"), (typed, "q-a1")):
            assert await call_tools(client, AFTER[:1], attempt) == ["00000301"]
            assert (await client.post(f"/ng-rollout/{attempt}/verify", json=verify_body())).json()["reward"] == 1.0
        assert restored.session_id_to_tool_env == {} and restored._session_episodes == {}
        closed = await typed.post(
            "/ng-rollout/q-a1/close_session",
            json={
                "resources_session_id": seed.resources_session_id,
                "episode_id": {"rollout_id": "q", "attempt": 1},
            },
        )
        assert closed.status_code == 200


# "flat" is the tables-only snapshot main wrote; "wrapped" is a cookie session's snapshot with a null identity.
@pytest.mark.parametrize("snapshot", ["flat", "wrapped"])
async def test_an_unused_restored_cookie_session_can_be_adopted_without_resetting_tables(snapshot: str) -> None:
    source, source_app = make_server()
    async with client_for(source_app) as client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await call_tools(client, BEFORE, "r")
    [(session_id, tool_env)] = source.session_id_to_tool_env.items()
    if snapshot == "flat":
        state = _export_tool_env(tool_env)
    else:
        state = (await source.export_session_states([session_id]))[session_id]
        assert state["episode_id"] is None and state["task_id"] is None
    restored, _ = make_server()
    await restored.restore_session_states({"typed-session": state})
    from fastapi import Request

    request = Request({"type": "http", "session": {}})
    await restored.seed_session(request, typed_seed())
    assert _export_tool_env(restored.session_id_to_tool_env["typed-session"]) == _export_tool_env(tool_env)
    assert restored._session_episodes["typed-session"] == (typed_seed().episode_id, typed_seed().task_id)


async def test_a_typed_seed_with_invalid_task_data_cannot_adopt_a_restored_session() -> None:
    source, source_app = make_server()
    async with client_for(source_app) as client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await call_tools(client, BEFORE, "r")
    [(session_id, tool_env)] = source.session_id_to_tool_env.items()
    restored, _ = make_server()
    await restored.restore_session_states({"typed-session": _export_tool_env(tool_env)})

    request = Request({"type": "http", "session": {}})
    invalid = typed_seed().model_copy(update={"task_data": {"ground_truth": []}})
    with pytest.raises(ResourcesSeedRejectedError):
        await restored.seed_session(request, invalid)

    assert restored.session_identity("typed-session") is None
    assert "typed-session" in restored._restored_unbound_sessions


async def test_a_restored_cookie_session_in_use_cannot_be_adopted(tmp_path: Path) -> None:
    _, source_app = make_server()
    async with client_for(source_app) as client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await call_tools(client, BEFORE, "r")
        await checkpoint(client, tmp_path)
        cookies = dict(client.cookies)

    restored, restored_app = make_server()
    async with client_for(restored_app, cookies) as cookie_run, client_for(restored_app) as other:
        assert (await restore(cookie_run, tmp_path)).json()["phase"] == "restored"
        await cookie_run.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        [session_id] = restored.session_id_to_tool_env
        # The restored /run continues, so its tables are no longer unbound.
        assert await call_tools(cookie_run, AFTER[:1], "r-a1") == ["00000301"]
        taken = await other.post(
            "/ng-rollout/x/seed_session", json=typed_seed(session_id, rollout_id="x").model_dump(mode="json")
        )
        assert taken.status_code == 409, taken.text
        assert restored._session_episodes == {}
        # The refused seed left the cookie run's tables as they were, and the run's verify scores.
        [found] = await call_tools(cookie_run, AFTER[3:], "r-a1")
        assert found["pagination"]["total_emails"] == 1
        verified = (await cookie_run.post("/ng-rollout/r-a1/verify", json=verify_body())).json()
        assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert restored.session_id_to_tool_env == {}


async def test_an_undone_restore_can_be_installed_again() -> None:
    # A restore that failed on another worker is undone here with delete_restored, then retried in this process.
    seed = typed_seed()
    continuation = seed.model_copy(update={"episode_id": EpisodeId(rollout_id="r", attempt=1)})
    source, source_app = make_server()
    async with client_for(source_app) as client:
        assert (await client.post("/ng-rollout/r/seed_session", json=seed.model_dump(mode="json"))).status_code == 200
        await call_tools(client, BEFORE, "r")
        cookies = dict(client.cookies)
    records = await source.checkpoint_participant.export([seed.episode_id])

    restored, restored_app = make_server()
    participant = restored.checkpoint_participant
    await participant.install(records, [seed.episode_id])
    await participant.delete_restored(continuation.episode_id)
    assert restored.session_id_to_tool_env == {} and restored._session_episodes == {}
    await participant.install(records, [seed.episode_id])
    async with client_for(restored_app, cookies) as client:
        # Undoing the restore refused nothing: the continuation reuses its session and the checkpoint's tables.
        reused = await client.post("/ng-rollout/r-a1/seed_session", json=continuation.model_dump(mode="json"))
        assert reused.status_code == 200, reused.text
        assert await call_tools(client, AFTER[:1], "r-a1") == ["00000301"]
        verified = (await client.post("/ng-rollout/r-a1/verify", json=verify_body())).json()
        assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert restored.session_id_to_tool_env == {} and restored._session_episodes == {}


async def test_restore_conflict_installs_neither_tables_nor_other_identities() -> None:
    source, _ = make_server()
    request = Request({"type": "http", "session": {}})
    for session_id in ("fresh", "occupied"):
        await source.seed_session(request, typed_seed(session_id))
    states = await source.export_session_states(["fresh", "occupied"])

    destination, _ = make_server()
    await destination.seed_session(request, typed_seed("occupied", rollout_id="different"))
    previous = destination.session_id_to_tool_env["occupied"]
    previous_identity = destination.session_identity("occupied")
    with pytest.raises(ResourcesSessionConflictError):
        await destination.restore_session_states(states)
    assert destination.session_id_to_tool_env == {"occupied": previous}
    assert destination.session_identity("occupied") == previous_identity
    assert destination.session_identity("fresh") is None


async def test_unused_unbound_checkpoint_tables_can_verify_and_release() -> None:
    source, _ = make_server()
    request = Request({"type": "http", "session": {}})
    await source.seed_session(request, typed_seed())
    flat = _export_tool_env(source.session_id_to_tool_env["typed-session"])
    destination, _ = make_server()
    await destination.restore_session_states({"restored-cookie": flat})
    request.session[SESSION_ID_KEY] = "restored-cookie"
    response = await destination.verify(request, workplace_app.WorkbenchVerifyRequest.model_validate(verify_body()))
    assert response.reward == 1.0 and response.mask_sample is False
    assert destination.session_id_to_tool_env == {}
    retry = await destination.verify(request, workplace_app.WorkbenchVerifyRequest.model_validate(verify_body()))
    assert retry.reward == 1.0 and retry.mask_sample is False
