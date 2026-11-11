# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The typed sessions an Environment Server seeds and closes, next to the seed of the agent's /run."""

import json
from typing import Any

import httpx
from fastapi.testclient import TestClient
from omegaconf import DictConfig

from nemo_gym.core import SESSION_LOST, EpisodeId, NeMoGymResponse, TaskId, failure_kinds
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.agentic_turn_simulation.app import AgenticTurnSimulationResourcesServer
from resources_servers.agentic_turn_simulation.tests.support import VERDICT, judge_reply, make_server, script


_EPISODE = EpisodeId(rollout_id="0-0", attempt=0)
_METADATA = {"harness": "sandbox", "categories": ["web_dev"]}
_HISTORY = [
    {"type": "message", "role": "user", "content": "Write the page."},
    {
        "type": "function_call",
        "call_id": "h1",
        "name": "write_file",
        "arguments": json.dumps({"path": "/home/user/site/index.html", "content": "<h1>Hi</h1>"}),
    },
    {"type": "function_call_output", "call_id": "h1", "output": json.dumps({"status": "success"})},
    {"type": "message", "role": "assistant", "content": "Done."},
    {"type": "message", "role": "user", "content": "Make the heading blue."},
]
_ROW = {"input": _HISTORY, "tools": [], "instructions": "You are a sandbox agent."}
_READ = {"call_id": "c1", "name": "read_file", "arguments": json.dumps({"path": "/home/user/site/index.html"})}


def _seed(resources_session_id: str = "resources-session", episode: EpisodeId = _EPISODE) -> dict[str, Any]:
    # As single_agent_turn seeds: task_data is the row without responses_create_params, which ride beside it.
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=episode,
        task_id=TaskId(taskset="agent_arena", task_id="0"),
        task_data={"verifier_metadata": _METADATA},
        responses_create_params=_ROW,
    ).model_dump(mode="json")


def _close(resources_session_id: str = "resources-session") -> dict[str, Any]:
    return ResourcesCloseSessionRequest(resources_session_id=resources_session_id, episode_id=_EPISODE).model_dump(
        mode="json"
    )


def _verify_body(stop_reason: str = "max_steps") -> dict[str, Any]:
    """The flat verify body single_agent_turn sends: task_data, responses_create_params and the response. The default
    stop reason ends in a code gate, so no judge is called."""
    response = NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy",
        object="response",
        output=[
            {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        metadata={"stop_reason": stop_reason, "steps": "1"},
    )
    return {
        "verifier_metadata": _METADATA,
        "responses_create_params": _ROW,
        "response": response.model_dump(mode="json"),
    }


def _client(server: AgenticTurnSimulationResourcesServer | None = None) -> tuple[Any, TestClient]:
    server = server or make_server()
    return server, TestClient(server.setup_webserver(), raise_server_exceptions=False)


def test_typed_sessions_follow_the_session_contract() -> None:
    server = make_server()
    check_resources_session_contract(
        server.setup_webserver(),
        ResourcesSeedSessionRequest.model_validate(_seed("contract-session")),
        keeps_state=True,
        holds_state=server.has_session_state,
        verify_body=_verify_body(),
    )


def test_the_typed_seed_builds_the_session_from_the_rows_params() -> None:
    server, client = _client()
    seed = client.post("/seed_session", json=_seed())
    assert seed.status_code == 200 and seed.json()["resources_session_id"] == "resources-session"
    state = server._sessions["resources-session"]
    assert state.workspace.files == {"/home/user/site/index.html": "<h1>Hi</h1>"}

    # Any agent can drive the session: /harness takes no body, and the history is already there.
    assert client.post("/harness").json() == {"stop_tools": ["ask_user"], "protocol": "native"}
    read = client.post("/simulate_tool_call", json=_READ).json()
    assert read["source"] == "exact" and json.loads(read["output"])["content"] == "<h1>Hi</h1>"


def test_a_typed_seed_without_the_rows_params_is_refused() -> None:
    server, client = _client()
    seed = _seed()
    del seed["responses_create_params"]  # an Environment Server that does not send them
    refused = client.post("/seed_session", json=seed)
    assert refused.status_code == 422 and "responses_create_params" in refused.text
    assert server._sessions == {}


def test_a_resent_verify_returns_the_first_result_without_judging_again() -> None:
    server, client = _client()
    client.post("/seed_session", json=_seed())
    client.post("/simulate_tool_call", json=_READ)
    sent = script(server, [judge_reply(VERDICT)])  # one judge reply: a second judge call would fail the test

    first = client.post("/verify", json=_verify_body("final_message"))
    assert first.status_code == 200, first.text
    assert (first.json()["reward"], first.json()["simulated_calls"]) == (1.0, {"exact": 1})
    assert not server.has_session_state("resources-session")
    again = client.post("/verify", json=_verify_body("final_message"))
    assert again.status_code == 200 and again.json() == first.json()
    assert len(sent) == 1
    # Another request for the session is scored, with the counts the session left.
    other = client.post("/verify", json=_verify_body("max_steps")).json()
    assert (other["code_gate"], other["simulated_calls"]) == ("unfinished_turn", {"exact": 1})


def test_a_resent_verify_whose_judge_failed_is_judged_again() -> None:
    server, client = _client(make_server(judge_max_attempts=1))
    client.post("/seed_session", json=_seed())
    client.post("/simulate_tool_call", json=_READ)
    sent = script(server, [judge_reply(text="no idea"), judge_reply(VERDICT)])

    failed = client.post("/verify", json=_verify_body("final_message")).json()
    assert (failed["mask_sample"], failed["failure_kind"]) == (True, failure_kinds.JUDGE_FAILED)
    # The masked result was not kept: the same request sent again gets a judge of its own, with the session's counts.
    again = client.post("/verify", json=_verify_body("final_message")).json()
    assert (again["reward"], again["simulated_calls"]) == (1.0, {"exact": 1}) and not again.get("mask_sample")
    assert len(sent) == 2


def test_close_releases_the_session_and_refuses_later_calls() -> None:
    server, client = _client()
    client.post("/seed_session", json=_seed())
    assert client.post("/close_session", json=_close()).status_code == 200
    assert client.post("/close_session", json=_close()).status_code == 200  # idempotent
    assert not server.has_session_state("resources-session")

    for path, body in (("/simulate_tool_call", _READ), ("/harness", None)):
        late = client.post(path, json=body)
        assert late.status_code == 404, late.text
        assert late.json()["detail"]["failure_kind"] == SESSION_LOST
    assert not server.has_session_state("resources-session")

    verified = client.post("/verify", json=_verify_body()).json()
    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)


async def test_with_checkpointing_an_interrupted_episode_restarts_with_a_new_session() -> None:
    """The server declares no checkpoint_mode, so it is restart-only: its seed reply tells single_agent_turn that the
    episode starts over from its input after a crash, and the new attempt seeds a session of its own."""
    assert AgenticTurnSimulationResourcesServer.checkpoint_mode == "restart_only"
    server = make_server()
    server.server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    retry = EpisodeId(rollout_id="0-0", attempt=1)
    transport = httpx.ASGITransport(app=server.setup_webserver())
    async with httpx.AsyncClient(transport=transport, base_url="http://resources") as client:
        first = await client.post(f"/ng-rollout/{_EPISODE.capture_key}/seed_session", json=_seed("s-a0"))
        restarted = await client.post(
            f"/ng-rollout/{retry.capture_key}/seed_session", json=_seed("s-a1", episode=retry)
        )
    assert first.status_code == restarted.status_code == 200, (first.text, restarted.text)
    assert first.headers["x-ng-checkpoint-restart"] == restarted.headers["x-ng-checkpoint-restart"] == "1"
    # The new attempt's session is built from the row again, not taken over from the interrupted one.
    assert server._sessions["s-a1"] is not server._sessions["s-a0"]
    assert server._sessions["s-a1"].workspace.files == {"/home/user/site/index.html": "<h1>Hi</h1>"}
    assert not server._sessions["s-a1"].answered_by


def test_the_runs_seed_holds_the_row_and_its_verify_releases_it() -> None:
    """The agent's /run (behind the legacy_agent relay) seeds with the flat row and never closes."""
    server, client = _client()
    row = {"responses_create_params": _ROW, "verifier_metadata": _METADATA}
    seed = client.post("/seed_session", json=row)
    assert seed.status_code == 200 and seed.json() == {}
    [sid] = server._sessions
    assert client.post("/harness").json() == {"stop_tools": ["ask_user"], "protocol": "native"}
    assert client.post("/simulate_tool_call", json=_READ).json()["source"] == "exact"

    verified = client.post("/verify", json=_verify_body())
    assert verified.status_code == 200 and verified.json()["simulated_calls"] == {"exact": 1}
    assert not server.has_session_state(sid)
