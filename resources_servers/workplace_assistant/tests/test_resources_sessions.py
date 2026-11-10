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
"""Workplace Assistant serves the typed resources sessions API through its real routes, next to an Agent's /run seed."""

import asyncio
import base64
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import SESSION_LOST, EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import (
    NEMO_GYM_MCP_SESSION_TOKEN_HEADER,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
    check_resources_session_contract,
    maybe_auto_expose,
)
from resources_servers.workplace_assistant import app
from resources_servers.workplace_assistant.app import (
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
    WorkbenchVerifyRequest,
)


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
TASK = TaskId(taskset="workplace_assistant", task_id="0")
SEND = {"recipient": "raj.patel@atlas.com", "subject": "Session check", "body": "Sent before the repeated seed"}
GROUND_TRUTH = [{"name": "email_send_email", "arguments": json.dumps(SEND)}]
TASK_DATA = {
    "ground_truth": GROUND_TRUTH,
    "id": 0,
    "category": "workplace_assistant_email",
    "environment_name": "workplace_assistant",
}


def make_server(**config: Any) -> WorkbenchResourcesServer:
    return WorkbenchResourcesServer(
        config=WorkbenchResourcesServerConfig(host="", port=0, entrypoint="", name="workplace_assistant", **config),
        server_client=MagicMock(spec=ServerClient),
    )


def cookie_session_id(client: TestClient, server: WorkbenchResourcesServer) -> str:
    # TimestampSigner can refresh the signature while the session identity remains unchanged.
    payload = client.cookies[server.get_session_middleware_key()].split(".", 1)[0]
    return json.loads(base64.b64decode(payload))[SESSION_ID_KEY]


def seed_json(resources_session_id: str = "resources-session", **update: Any) -> dict[str, Any]:
    seed = ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id, episode_id=EPISODE, task_id=TASK, task_data=TASK_DATA
    )
    return seed.model_copy(update=update).model_dump(mode="json")


def close_json(resources_session_id: str = "resources-session") -> dict[str, Any]:
    return {"resources_session_id": resources_session_id, "episode_id": EPISODE.model_dump(mode="json")}


def verify_json(arguments: dict[str, str]) -> dict[str, Any]:
    """The flat verify body the single_agent_turn Environment Server builds from the task's fields."""
    response = NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="model",
        object="response",
        output=[
            {
                "type": "function_call",
                "name": "email_send_email",
                "arguments": json.dumps(arguments),
                "call_id": "call",
                "id": "fc",
                "status": "completed",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return TASK_DATA | {
        "responses_create_params": {"input": "Send the email."},
        "response": response.model_dump(mode="json"),
    }


def test_follows_the_session_contract() -> None:
    server = make_server()
    check_resources_session_contract(
        server.setup_webserver(),
        ResourcesSeedSessionRequest(
            resources_session_id="contract-session", episode_id=EPISODE, task_id=TASK, task_data=TASK_DATA
        ),
        keeps_state=True,
        holds_state=lambda resources_session_id: resources_session_id in server.session_id_to_tool_env,
        verify_body=verify_json(SEND),
    )


def test_a_typed_verify_releases_tables_and_keeps_only_its_retry_identity() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=seed_json())
    assert seeded.status_code == 200, seeded.text
    assert seeded.json() == {
        "resources_session_id": "resources-session",
        "resources_tools": None,
        "sandbox_access": None,
    }
    assert client.post("/email_send_email", json=SEND).status_code == 200

    # A retried seed returns the same session, with the email the agent already sent.
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    found = client.post("/email_search_emails", json={"query": "Sent before the repeated seed"}).json()
    assert found["output"]["pagination"]["total_emails"] == 1

    verified = client.post("/verify", json=verify_json(SEND)).json()
    assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    # Verify ends checkpoint ownership, so the databases and active binding are already released.
    assert server.session_id_to_tool_env == {} and server._session_episodes == {}
    assert client.post("/verify", json=verify_json(SEND)).json()["reward"] == 1.0

    # The close names the session, so it releases it without the seed's cookie.
    client.cookies.clear()
    assert client.post("/close_session", json=close_json()).status_code == 200
    assert "resources-session" not in server.session_id_to_tool_env


def test_tool_calls_and_verify_for_a_closed_session_are_lost() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=seed_json())
    client.post("/close_session", json=close_json())

    called = client.post("/email_send_email", json=SEND)
    assert called.status_code == 404
    assert called.json()["detail"]["failure_kind"] == SESSION_LOST

    verified = client.post("/verify", json=verify_json(SEND)).json()
    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)


def test_a_seed_whose_task_data_is_not_a_task_is_refused_and_records_nothing() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())

    response = client.post("/seed_session", json=seed_json(task_data={"ground_truth": GROUND_TRUTH}))

    assert response.status_code == 422
    assert "task_data is not a valid WorkbenchSeedSessionRequest" in response.json()["detail"]
    assert server.session_id_to_tool_env == {} and server._session_episodes == {}
    # The refusal records nothing, so a corrected seed for the same ID succeeds.
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    assert "resources-session" in server.session_id_to_tool_env


def test_a_seed_whose_tables_fail_to_build_leaves_no_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    server = make_server()
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    real_get_tools = app.get_tools
    calls = []

    def get_tools_failing_once(toolkits):
        calls.append(toolkits)
        if len(calls) == 1:
            raise OSError("Too many open files")
        return real_get_tools(toolkits)

    monkeypatch.setattr(app, "get_tools", get_tools_failing_once)

    assert client.post("/seed_session", json=seed_json()).status_code == 500
    assert server._session_episodes == {}
    # The caller's retry builds the tables, so the session's tool calls reach them.
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    assert client.post("/email_send_email", json=SEND).status_code == 200


def test_an_agents_run_seed_still_gets_a_session_that_verify_releases() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json={"responses_create_params": {"input": "hi"}, **TASK_DATA})
    assert seeded.status_code == 200, seeded.text
    assert seeded.json() == {}
    assert client.post("/email_send_email", json=SEND).status_code == 200
    assert len(server.session_id_to_tool_env) == 1

    verified = client.post("/verify", json=verify_json(SEND)).json()

    assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert server.session_id_to_tool_env == {}
    # The HTTP client sends a verify again when its connection drops after the server answered.
    resent = client.post("/verify", json=verify_json(SEND)).json()
    assert (resent["reward"], resent["mask_sample"]) == (1.0, False)


def test_an_agents_run_verify_without_a_seed_is_masked() -> None:
    client = TestClient(make_server().setup_webserver())

    verified = client.post("/verify", json=verify_json(SEND)).json()

    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)


def test_mcp_tool_calls_reach_the_tables_seeded_under_the_callers_id() -> None:
    server = make_server(expose_tools_over_mcp=True)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    rpc_headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    with TestClient(app) as client:
        seeded = client.post("/seed_session", json=seed_json())
        assert seeded.status_code == 200, seeded.text
        token = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        client.post(
            "/mcp",
            headers=rpc_headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "t", "version": "0"},
                },
            },
        )
        client.post("/mcp", headers=rpc_headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        called = client.post(
            "/mcp",
            headers={**rpc_headers, NEMO_GYM_MCP_SESSION_TOKEN_HEADER: token},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "email_send_email", "arguments": SEND},
            },
        ).json()

    assert called["result"].get("isError") is not True, called
    emails = server.session_id_to_tool_env["resources-session"]["containers"]["email"]._emails
    assert (emails["subject"] == "Session check").sum() == 1


@pytest.mark.parametrize(
    "payload",
    [
        seed_json() | {"extra": True},
        {key: value for key, value in seed_json().items() if key != "episode_id"},
        seed_json(task_data=[]),
    ],
)
@pytest.mark.parametrize("over_mcp", [False, True])
def test_malformed_typed_seed_cannot_fall_back_to_cookie_seed(payload: dict[str, Any], over_mcp: bool) -> None:
    server = make_server(expose_tools_over_mcp=over_mcp)
    webserver = server.setup_webserver()
    maybe_auto_expose(server, webserver)
    client = TestClient(webserver)
    assert client.post("/seed_session", json=payload).status_code == 422
    assert server.session_id_to_tool_env == {} and server._session_episodes == {}
    assert client.post("/seed_session", json=seed_json()).status_code == 200


def test_a_run_seed_keeps_a_row_with_its_own_task_id() -> None:
    # A dataset row's scalar task_id is a row field, not a typed seed's TaskId.
    server = make_server()
    client = TestClient(server.setup_webserver())
    body = {"responses_create_params": {"input": "hi"}, **TASK_DATA, "task_id": "wb-0007"}
    assert client.post("/seed_session", json=body).status_code == 200
    assert client.post("/email_send_email", json=SEND).status_code == 200
    assert server._session_episodes == {}


@pytest.mark.parametrize("completed", ["active", "verified", "closed"])
def test_cookie_seed_cannot_replace_a_typed_session(completed: str) -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    assert client.post("/email_send_email", json=SEND).status_code == 200
    if completed == "verified":
        assert client.post("/verify", json=verify_json(SEND)).status_code == 200
    elif completed == "closed":
        assert client.post("/close_session", json=close_json()).status_code == 200
    assert client.post("/seed_session", json={"responses_create_params": {"input": "hi"}}).status_code == 409
    if completed == "active":
        found = client.post("/email_search_emails", json={"query": SEND["body"]}).json()
        assert found["output"]["pagination"]["total_emails"] == 1
    else:
        assert server.session_id_to_tool_env == {} and server._session_episodes == {}


def test_refused_seed_preserves_the_callers_existing_cookie() -> None:
    server = make_server()
    webserver = server.setup_webserver()
    owner, other = TestClient(webserver), TestClient(webserver)
    assert owner.post("/seed_session", json=seed_json()).status_code == 200
    other_episode = EpisodeId(rollout_id="other").model_dump(mode="json")
    assert other.post("/seed_session", json=seed_json("own-session", episode_id=other_episode)).status_code == 200
    assert other.post("/seed_session", json=seed_json(episode_id=other_episode)).status_code == 409
    assert other.post("/email_send_email", json=SEND).status_code == 200
    own = server.session_id_to_tool_env["own-session"]["containers"]["email"]._emails
    bound = server.session_id_to_tool_env["resources-session"]["containers"]["email"]._emails
    assert (own["subject"] == SEND["subject"]).sum() == 1
    assert (bound["subject"] == SEND["subject"]).sum() == 0


def test_verified_session_keeps_close_ownership_but_not_tools() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    assert client.post("/verify", json=verify_json(SEND)).json()["reward"] == 1.0
    assert client.post("/email_send_email", json=SEND).status_code == 404
    assert client.post("/seed_session", json=seed_json()).status_code == 409
    wrong_close = close_json() | {"episode_id": EpisodeId(rollout_id="other").model_dump(mode="json")}
    assert client.post("/close_session", json=wrong_close).status_code == 409
    assert client.post("/verify", json=verify_json(SEND)).json()["reward"] == 1.0
    assert client.post("/close_session", json=close_json()).status_code == 200
    assert client.post("/verify", json=verify_json(SEND)).json()["mask_sample"] is True


@pytest.mark.parametrize("typed", [False, True])
def test_verify_retry_record_expires_without_extending_on_retry(monkeypatch: pytest.MonkeyPatch, typed: bool) -> None:
    now = 0.0
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: now)
    server = make_server(session_close_retry_window_seconds=10)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=seed_json() if typed else {}).status_code == 200
    [session_id] = server.session_id_to_tool_env
    assert client.post("/verify", json=verify_json(SEND)).json()["reward"] == 1.0
    now = 9.0
    assert client.post("/verify", json=verify_json(SEND)).json()["reward"] == 1.0
    now = 11.0
    verified = client.post("/verify", json=verify_json(SEND)).json()
    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)
    assert not server.session_is_verified(session_id)
    assert server.session_id_to_tool_env == {} and server._session_episodes == {}


def test_task_data_and_verify_required_fields_agree() -> None:
    from pydantic import TypeAdapter

    from nemo_gym.server_runtime import BaseVerifyRequest
    from resources_servers.workplace_assistant.app import WorkbenchVerifyRequest
    from resources_servers.workplace_assistant.task_data import TaskData

    task_fields = TaskData.model_fields
    verify_fields = {
        name: field
        for name, field in WorkbenchVerifyRequest.model_fields.items()
        if name not in BaseVerifyRequest.model_fields
    }
    assert task_fields.keys() == verify_fields.keys()
    for name, field in task_fields.items():
        assert field.is_required() == verify_fields[name].is_required()
        assert TypeAdapter(field.annotation).json_schema() == TypeAdapter(verify_fields[name].annotation).json_schema()


def test_reverify_without_a_session_is_explicitly_unsupported() -> None:
    client = TestClient(make_server().setup_webserver())
    assert client.get("/reverify_mode").json() == "unsupported"
    assert client.post("/verify", json=verify_json(SEND)).json()["mask_sample"] is True


def test_next_cookie_run_keeps_its_cookie_and_replaces_the_finished_run() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json={}).status_code == 200
    [session_id] = server.session_id_to_tool_env
    assert client.post("/email_send_email", json=SEND).status_code == 200
    assert client.post("/verify", json=verify_json(SEND)).json()["reward"] == 1.0
    # The next /run on the same cookie keeps the ID and replaces the finished run, retry record included.
    assert client.post("/seed_session", json={}).status_code == 200
    assert list(server.session_id_to_tool_env) == [session_id] and cookie_session_id(client, server) == session_id
    assert not server.session_is_verified(session_id)
    emails = server.session_id_to_tool_env[session_id]["containers"]["email"]._emails
    assert (emails["subject"] == "Session check").sum() == 0
    assert client.post("/email_send_email", json=SEND).status_code == 200
    assert client.post("/verify", json=verify_json(SEND)).json()["reward"] == 1.0
    assert server.session_id_to_tool_env == {}


def test_typed_seed_cannot_adopt_a_live_cookie_session() -> None:
    server = make_server()
    webserver = server.setup_webserver()
    legacy, typed = TestClient(webserver), TestClient(webserver)
    assert legacy.post("/seed_session", json={}).status_code == 200
    [session_id] = server.session_id_to_tool_env
    assert legacy.post("/email_send_email", json=SEND).status_code == 200
    assert typed.post("/seed_session", json=seed_json(session_id)).status_code == 409
    assert server._session_episodes == {}
    found = legacy.post("/email_search_emails", json={"query": SEND["body"]}).json()
    assert found["output"]["pagination"]["total_emails"] == 1


def test_failed_cookie_reseed_preserves_verification_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    server = make_server()
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json={}).status_code == 200
    [session_id] = server.session_id_to_tool_env
    assert client.post("/verify", json=verify_json(SEND)).json()["reward"] == 1.0
    assert server.session_is_verified(session_id)
    with monkeypatch.context() as patch:
        patch.setattr(app, "get_tools", MagicMock(side_effect=OSError("tables unavailable")))
        assert client.post("/seed_session", json={}).status_code == 500
    assert server.session_is_verified(session_id)
    assert client.post("/verify", json=verify_json(SEND)).json()["reward"] == 1.0
    assert client.post("/seed_session", json={}).status_code == 200
    assert list(server.session_id_to_tool_env) == [session_id]
    assert not server.session_is_verified(session_id)


async def test_a_seed_cannot_interleave_with_a_verify_whose_release_suspends(monkeypatch: pytest.MonkeyPatch) -> None:
    release = WorkbenchResourcesServer.release_session_state

    async def suspending_release(self: WorkbenchResourcesServer, session_id: str) -> None:
        await release(self, session_id)
        await asyncio.sleep(0)

    monkeypatch.setattr(WorkbenchResourcesServer, "release_session_state", suspending_release)
    server = make_server()
    seed = ResourcesSeedSessionRequest.model_validate(seed_json())
    await server.seed_session(Request({"type": "http", "session": {}}), seed)
    verify_request = Request({"type": "http", "session": {SESSION_ID_KEY: seed.resources_session_id}})
    verified, retried_seed = await asyncio.gather(
        server.verify(verify_request, WorkbenchVerifyRequest.model_validate(verify_json(SEND))),
        server.seed_session(Request({"type": "http", "session": {}}), seed),
        return_exceptions=True,
    )
    assert verified.reward == 1.0
    # The retried seed waits for the verify's retry record instead of reusing a binding whose tables are gone.
    assert isinstance(retried_seed, ResourcesSessionConflictError), retried_seed
    assert server.session_id_to_tool_env == {} and server._session_episodes == {}
