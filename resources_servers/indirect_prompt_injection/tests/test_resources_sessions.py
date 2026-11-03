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
"""The IPI server serves the typed resources sessions API through its real routes, next to an Agent's /run seed."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import SESSION_LOST, EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import ResourcesSeedSessionRequest
from nemo_gym.server_runtime.testing.session_conformance import check_resources_session_contract
from resources_servers.indirect_prompt_injection.app import IPIResourcesServer, IPIResourcesServerConfig


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
TASK = TaskId(taskset="indirect_prompt_injection", task_id="0")
ROW = json.loads((Path(__file__).parents[1] / "data" / "example.jsonl").read_text().splitlines()[0])
# The task's own fields, as collation puts them in task_data.
TASK_DATA = {key: value for key, value in ROW.items() if key not in ("responses_create_params", "agent_ref")}
MESSAGE = {"recipient": "records@clinic.example", "subject": "Session check", "body": "Sent before the repeated seed"}


def make_server() -> IPIResourcesServer:
    return IPIResourcesServer(
        config=IPIResourcesServerConfig(host="", port=0, entrypoint="", name="indirect_prompt_injection"),
        server_client=MagicMock(spec=ServerClient),
    )


def seed_json(resources_session_id: str = "resources-session", **update: Any) -> dict[str, Any]:
    seed = ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id, episode_id=EPISODE, task_id=TASK, task_data=TASK_DATA
    )
    return seed.model_copy(update=update).model_dump(mode="json")


def close_json(resources_session_id: str = "resources-session") -> dict[str, Any]:
    return {"resources_session_id": resources_session_id, "episode_id": EPISODE.model_dump(mode="json")}


def verify_json() -> dict[str, Any]:
    """The flat verify body the single_agent_turn Environment Server builds: the agent called every required tool."""
    response = NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="model",
        object="response",
        output=[
            {
                "type": "function_call",
                "name": tool,
                "arguments": json.dumps({"patient_id": "P001"}),
                "call_id": f"call-{tool}",
                "id": f"fc-{tool}",
                "status": "completed",
            }
            for tool in TASK_DATA["required_tools"]
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return TASK_DATA | {
        "responses_create_params": ROW["responses_create_params"],
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
        holds_state=lambda resources_session_id: resources_session_id in server.session_id_to_env,
        verify_body=verify_json(),
    )


def test_a_typed_session_keeps_its_environment_until_close() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=seed_json())
    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["resources_session_id"] == "resources-session"
    assert "Message sent" in client.post("/send_message", json=MESSAGE).json()["output"]

    # A retried seed returns the same session, with the message the agent already sent.
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    assert server.session_id_to_env["resources-session"]["messages"][-1]["to"] == MESSAGE["recipient"]

    verified = client.post("/verify", json=verify_json()).json()
    assert (verified["reward"], verified["mask_sample"], verified["injection_followed"]) == (1.0, False, False)
    # The Environment Server closes after verifying, so verify leaves the session for that close.
    assert "resources-session" in server.session_id_to_env

    # The close names the session, so it releases it without the seed's cookie.
    client.cookies.clear()
    assert client.post("/close_session", json=close_json()).status_code == 200
    assert "resources-session" not in server.session_id_to_env


def test_tool_calls_and_verify_for_a_closed_session_are_lost() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=seed_json())
    client.post("/close_session", json=close_json())

    called = client.post("/send_message", json=MESSAGE)
    assert called.status_code == 404
    assert called.json()["detail"]["failure_kind"] == SESSION_LOST

    verified = client.post("/verify", json=verify_json()).json()
    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)
    # A lost session never showed the agent the injection, so the sample claims no resisted attack.
    assert verified["reward_safety"] == 0.0


def test_a_seed_without_an_environment_is_refused_for_good() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())
    task_data = {key: value for key, value in TASK_DATA.items() if key != "environment"}

    response = client.post("/seed_session", json=seed_json(task_data=task_data))

    assert response.status_code == 422
    assert "task_data is not an indirect_prompt_injection task" in response.json()["detail"]
    assert server.session_id_to_env == {}
    # The refusal records nothing, so a corrected seed for the same ID succeeds.
    assert client.post("/seed_session", json=seed_json()).status_code == 200


def test_an_agents_run_seed_still_gets_a_session_that_verify_releases() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=ROW)
    assert seeded.status_code == 200, seeded.text
    assert seeded.json() == {}
    assert "Message sent" in client.post("/send_message", json=MESSAGE).json()["output"]
    assert len(server.session_id_to_env) == 1

    verified = client.post("/verify", json=verify_json()).json()

    assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert server.session_id_to_env == {}
    # The HTTP client sends a verify again when its connection drops after the server answered.
    resent = client.post("/verify", json=verify_json()).json()
    assert (resent["reward"], resent["mask_sample"]) == (1.0, False)


def test_an_agents_run_verify_without_a_seed_is_masked() -> None:
    client = TestClient(make_server().setup_webserver())

    verified = client.post("/verify", json=verify_json()).json()

    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)
