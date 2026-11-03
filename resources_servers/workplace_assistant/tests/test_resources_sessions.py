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

import json
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import SESSION_LOST, EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import NEMO_GYM_MCP_SESSION_TOKEN_HEADER, ResourcesSeedSessionRequest, maybe_auto_expose
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.workplace_assistant.app import WorkbenchResourcesServer, WorkbenchResourcesServerConfig


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


def test_a_typed_session_keeps_its_tables_until_close() -> None:
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
    # The Environment Server closes after verifying, so verify leaves the session for that close.
    assert "resources-session" in server.session_id_to_tool_env
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


def test_a_seed_whose_task_data_is_not_a_task_is_refused_for_good() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())

    response = client.post("/seed_session", json=seed_json(task_data={"ground_truth": GROUND_TRUTH}))

    assert response.status_code == 422
    assert "task_data is not a workplace_assistant task" in response.json()["detail"]
    assert server.session_id_to_tool_env == {}
    # The refusal records nothing, so a corrected seed for the same ID succeeds.
    assert client.post("/seed_session", json=seed_json()).status_code == 200


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
