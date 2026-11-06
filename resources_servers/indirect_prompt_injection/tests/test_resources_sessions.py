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

import asyncio
import base64
import copy
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from omegaconf import DictConfig

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import SESSION_LOST, EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import (
    CHECKPOINT_CLOSE_FOLLOWS_HEADER,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
    maybe_auto_expose,
)
from resources_servers.indirect_prompt_injection.app import (
    IPIResourcesServer,
    IPIResourcesServerConfig,
    IPISeedSessionRequest,
    IPIVerifyRequest,
    ToolCallRequest,
)


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
TASK = TaskId(taskset="indirect_prompt_injection", task_id="0")
ROW = json.loads((Path(__file__).parents[1] / "data" / "example.jsonl").read_text().splitlines()[0])
# The task's own fields, as collation puts them in task_data.
TASK_DATA = {key: value for key, value in ROW.items() if key not in ("responses_create_params", "agent_ref")}
MESSAGE = {"recipient": "records@clinic.example", "subject": "Session check", "body": "Sent before the repeated seed"}


def make_server(*, checkpoint: bool = False, expose_tools_over_mcp: bool = False) -> IPIResourcesServer:
    server_client = MagicMock(spec=ServerClient)
    if checkpoint:
        server_client.global_config_dict = DictConfig(
            {"checkpoint": {"enabled": True, "control_auth_token": "test-token"}}
        )
    return IPIResourcesServer(
        config=IPIResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="indirect_prompt_injection",
            expose_tools_over_mcp=expose_tools_over_mcp,
        ),
        server_client=server_client,
    )


def make_client(server: IPIResourcesServer) -> TestClient:
    app = server.setup_webserver()
    if server.config.expose_tools_over_mcp:
        maybe_auto_expose(server, app)
    return TestClient(app)


def cookie_session_id(client: TestClient, server: IPIResourcesServer) -> str:
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


def test_a_typed_verify_releases_the_environment_and_preserves_its_retry() -> None:
    server = make_server()
    client = make_client(server)

    seeded = client.post("/seed_session", json=seed_json())
    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["resources_session_id"] == "resources-session"
    assert "Message sent" in client.post("/send_message", json=MESSAGE).json()["output"]

    # A retried seed returns the same session, with the message the agent already sent.
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    assert server.session_id_to_env["resources-session"]["messages"][-1]["to"] == MESSAGE["recipient"]

    verified = client.post("/verify", json=verify_json()).json()
    assert (verified["reward"], verified["mask_sample"], verified["injection_followed"]) == (1.0, False, False)
    assert "resources-session" not in server.session_id_to_env
    assert "resources-session" not in server._session_episodes
    resent = client.post("/verify", json=verify_json()).json()
    assert (resent["reward"], resent["mask_sample"]) == (1.0, False)
    # A delayed seed acknowledgement cannot recreate the released environment.
    assert client.post("/seed_session", json=seed_json()).status_code == 409
    assert "resources-session" not in server.session_id_to_env
    assert client.post("/send_message", json=MESSAGE).status_code == 404

    # The close names the session, so it releases it without the seed's cookie.
    client.cookies.clear()
    assert client.post("/close_session", json=close_json()).status_code == 200
    assert "resources-session" not in server.session_id_to_env


@pytest.mark.parametrize("typed", [False, True])
def test_scoring_failure_releases_the_environment_and_preserves_request_retry(typed: bool) -> None:
    server = make_server()
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=seed_json() if typed else ROW).status_code == 200
    with patch(
        "resources_servers.indirect_prompt_injection.app.check_injection_followed",
        side_effect=RuntimeError("scoring unavailable"),
    ) as scorer:
        assert client.post("/verify", json=verify_json()).status_code == 500
        scorer.assert_called_once()
    assert server.session_id_to_env == {} and server._session_episodes == {}

    retried = client.post("/verify", json=verify_json())
    assert retried.status_code == 200
    assert (retried.json()["reward"], retried.json()["mask_sample"]) == (1.0, False)


def test_tool_calls_and_verify_for_a_closed_session_are_lost() -> None:
    server = make_server()
    client = make_client(server)
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
    client = make_client(server)
    task_data = {key: value for key, value in TASK_DATA.items() if key != "environment"}

    response = client.post("/seed_session", json=seed_json(task_data=task_data))

    assert response.status_code == 422
    assert "task_data is not a valid IPITypedSeedSessionRequest" in response.json()["detail"]
    assert server.session_id_to_env == {}
    # The refusal records nothing, so a corrected seed for the same ID succeeds.
    assert client.post("/seed_session", json=seed_json()).status_code == 200


def test_an_agents_run_seed_still_gets_a_session_that_verify_releases() -> None:
    server = make_server()
    client = make_client(server)

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


def test_offline_reverify_is_declared_unsupported() -> None:
    client = TestClient(make_server().setup_webserver())

    assert client.get("/reverify_mode").json() == "unsupported"


@pytest.mark.parametrize("state", ["active", "verified", "closed"])
def test_a_run_seed_cannot_overwrite_a_typed_session(state: str) -> None:
    server = make_server()
    client = make_client(server)
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    if state == "verified":
        assert client.post("/verify", json=verify_json()).status_code == 200
    elif state == "closed":
        assert client.post("/close_session", json=close_json()).status_code == 200
    original = server.session_id_to_env.get("resources-session")
    assert client.post("/seed_session", json=ROW | {"environment": {}}).status_code == 409
    assert server.session_id_to_env.get("resources-session") is original


def test_a_typed_seed_cannot_take_over_a_run_session() -> None:
    server = make_server()
    client = make_client(server)
    assert client.post("/seed_session", json=ROW).status_code == 200
    [session_id] = server.session_id_to_env
    original = server.session_id_to_env[session_id]
    assert client.post("/seed_session", json=seed_json(session_id)).status_code == 409
    assert server.session_id_to_env[session_id] is original


def test_a_failed_environment_copy_leaves_no_binding_and_can_be_retried() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    with patch("resources_servers.indirect_prompt_injection.app.copy.deepcopy", side_effect=MemoryError):
        assert client.post("/seed_session", json=seed_json()).status_code == 500
    assert server._session_episodes == {}
    assert server.session_id_to_env == {}
    assert client.post("/seed_session", json=seed_json()).status_code == 200
    assert client.post("/send_message", json=MESSAGE).status_code == 200


async def test_typed_sessions_seeded_from_one_task_data_object_share_no_environment() -> None:
    # Over HTTP every seed parses its own task_data. In process, seeds copied from one request share one task_data.
    server = make_server()
    seed_a = ResourcesSeedSessionRequest(
        resources_session_id="session-a", episode_id=EPISODE, task_id=TASK, task_data=TASK_DATA
    )
    seed_b = seed_a.model_copy(
        update={"resources_session_id": "session-b", "episode_id": EpisodeId(rollout_id="other", attempt=0)}
    )
    assert seed_b.task_data is seed_a.task_data
    original = copy.deepcopy(seed_a.task_data)
    request_a = Request({"type": "http", "session": {}})
    await server.seed_session(request_a, seed_a)
    await server.seed_session(Request({"type": "http", "session": {}}), seed_b)

    await server.route_tool_call("send_message", ToolCallRequest(**MESSAGE), request_a)
    environment = server.session_id_to_env["session-a"]
    assert environment["messages"][-1]["to"] == MESSAGE["recipient"]
    # Tool handlers change nested records in place, so change every nested dict and list of the session too.
    for value in environment.values():
        if isinstance(value, dict):
            value["changed"] = True
        elif isinstance(value, list):
            value.append("changed")

    assert server.session_id_to_env["session-b"] == original["environment"]
    assert seed_a.task_data == original


def test_checkpoint_wait_verify_leaves_no_untracked_environment() -> None:
    server = make_server(checkpoint=True)
    client = make_client(server)
    assert client.post("/ng-rollout/rollout/seed_session", json=seed_json()).status_code == 200
    assert server._checkpoint.readiness().counts["sessions"] == 1
    verified = client.post(
        "/ng-rollout/rollout/verify", json=verify_json(), headers={CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"}
    )
    assert verified.status_code == 200
    assert server._checkpoint.readiness().counts["sessions"] == 0
    assert server.session_id_to_env == {}
    assert server._session_episodes == {}
    assert client.post("/ng-rollout/rollout/verify", json=verify_json()).json()["mask_sample"] is False


@pytest.mark.parametrize("typed", [False, True])
def test_checkpoint_retirement_releases_an_unverified_environment(typed: bool) -> None:
    server = make_server(checkpoint=True)
    client = make_client(server)
    assert client.post("/ng-rollout/rollout/seed_session", json=seed_json() if typed else ROW).status_code == 200
    retired = client.post(
        "/ng-control/v1/checkpoint/retire",
        json={"checkpoint_id": "retire", "deadline_ts": time.time() + 5, "episode_ids": [EPISODE.model_dump()]},
        headers={"authorization": "Bearer test-token"},
    )
    assert retired.status_code == 200, retired.text
    assert server.session_id_to_env == {}
    assert server._session_episodes == {}


def test_verified_identity_expires_and_close_still_checks_the_episode() -> None:
    server = make_server()
    client = make_client(server)
    with patch("nemo_gym.server_runtime.base_resources_server.monotonic", return_value=0):
        assert client.post("/seed_session", json=seed_json()).status_code == 200
        assert client.post("/verify", json=verify_json()).json()["reward"] == 1.0
        other = EpisodeId(rollout_id="other", attempt=0)
        assert client.post("/seed_session", json=seed_json(episode_id=other)).status_code == 409
        assert client.post("/close_session", json=close_json() | {"episode_id": other.model_dump()}).status_code == 409
    with patch("nemo_gym.server_runtime.base_resources_server.monotonic", return_value=301):
        assert client.post("/verify", json=verify_json()).json()["mask_sample"] is True
        assert client.post("/seed_session", json=seed_json(episode_id=other)).status_code == 200
        assert client.post("/send_message", json=MESSAGE).status_code == 200


@pytest.mark.parametrize("field", ["resources_session_id", "episode_id", "task_id", "task_data"])
@pytest.mark.parametrize("mcp", [False, True])
def test_a_malformed_typed_seed_cannot_fall_back_to_a_run_seed(field: str, mcp: bool) -> None:
    server = make_server(expose_tools_over_mcp=mcp)
    client = make_client(server)
    body = seed_json()
    body.pop(field)
    # An extra top-level environment also must not let a malformed typed seed use the /run arm.
    body["environment"] = {}
    assert client.post("/seed_session", json=body).status_code == 422
    assert server._session_episodes == {}


@pytest.mark.parametrize("mcp", [False, True])
def test_a_run_seed_keeps_a_row_with_its_own_task_id(mcp: bool) -> None:
    # Dataset task_id/task_data fields do not make a row a typed seed.
    server = make_server(expose_tools_over_mcp=mcp)
    client = make_client(server)
    assert (
        client.post("/seed_session", json=ROW | {"task_id": "ipi-0001", "task_data": {"custom": "legacy"}}).status_code
        == 200
    )
    assert client.post("/send_message", json=MESSAGE).status_code == 200
    assert server._session_episodes == {}


def test_next_run_keeps_its_cookie_and_replaces_the_finished_environment() -> None:
    server = make_server()
    client = make_client(server)
    assert client.post("/seed_session", json=ROW).status_code == 200
    [session_id] = server.session_id_to_env
    assert client.post("/send_message", json=MESSAGE).status_code == 200
    first = client.post("/verify", json=verify_json()).json()
    # The next /run on the same cookie keeps the ID and replaces the finished run, retry record included.
    assert client.post("/seed_session", json=ROW).status_code == 200
    assert list(server.session_id_to_env) == [session_id] and cookie_session_id(client, server) == session_id
    assert not server.session_is_verified(session_id)
    assert server.session_id_to_env[session_id] == ROW["environment"]
    second = client.post("/verify", json=verify_json()).json()
    assert (first["reward"], second["reward"]) == (1.0, 1.0)
    assert server.session_id_to_env == {}


async def test_a_run_seed_racing_a_verify_waits_for_the_release_and_the_retry_record() -> None:
    server = make_server()
    request = Request({"type": "http", "session": {SESSION_ID_KEY: "run-session"}})
    run_seed = IPISeedSessionRequest.model_validate(ROW)
    await server.seed_session(request, run_seed)
    await server.route_tool_call("send_message", ToolCallRequest(**MESSAGE), request)
    released = []
    release = IPIResourcesServer.release_session_state

    async def release_after_io(self: IPIResourcesServer, session_id: str) -> None:
        # The server's own release never waits. One that waits on I/O lets the next run's seed run in mid-verify.
        await asyncio.sleep(0)
        released.append(session_id)
        await release(self, session_id)

    with patch.object(IPIResourcesServer, "release_session_state", release_after_io):
        verified, _ = await asyncio.gather(
            server.verify(request, IPIVerifyRequest.model_validate(verify_json())),
            server.seed_session(request, run_seed),
        )

    assert (verified.reward, verified.mask_sample) == (1.0, False)
    # The verify released the finished run through the hook, and the next run's seed replaced it only afterwards.
    assert released == ["run-session"]
    assert server.session_id_to_env["run-session"] == ROW["environment"]
    assert not server.session_is_verified("run-session")
    # The next run's environment then goes away without a verify: its verify is masked, not answered from the record.
    await server.retire_session_state("run-session")
    assert (await server.verify(request, IPIVerifyRequest.model_validate(verify_json()))).mask_sample is True


@pytest.mark.parametrize("mcp", [False, True])
def test_minimal_cookie_seed_remains_valid_but_typed_seed_requires_the_injection(mcp: bool) -> None:
    server = make_server(expose_tools_over_mcp=mcp)
    client = make_client(server)
    legacy = {"environment": ROW["environment"]}
    initial_messages = list(ROW["environment"].get("messages", []))
    assert client.post("/seed_session", json=legacy).status_code == 200
    [session_id] = server.session_id_to_env
    assert client.post("/send_message", json=MESSAGE).status_code == 200
    assert ROW["environment"].get("messages", []) == initial_messages
    rejected = client.post("/seed_session", json=seed_json(task_data=legacy))
    assert rejected.status_code == 422 and "injection" in rejected.json()["detail"]
    assert list(server.session_id_to_env) == [session_id]
    assert server._session_episodes == {}


def test_failed_cookie_reseed_keeps_retry_evidence_until_initialization_succeeds() -> None:
    server = make_server()
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=ROW).status_code == 200
    [session_id] = server.session_id_to_env
    assert client.post("/verify", json=verify_json()).json()["reward"] == 1.0
    with patch("resources_servers.indirect_prompt_injection.app.copy.deepcopy", side_effect=MemoryError):
        assert client.post("/seed_session", json=ROW).status_code == 500
    assert cookie_session_id(client, server) == session_id
    assert server.session_is_verified(session_id) and not server.session_id_to_env
    assert client.post("/verify", json=verify_json()).json()["mask_sample"] is False
    assert client.post("/seed_session", json=ROW).status_code == 200
    assert list(server.session_id_to_env) == [session_id]
    assert not server.session_is_verified(session_id)
