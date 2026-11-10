# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The typed sessions an Environment Server seeds and closes, next to the seed of the agent's /run."""

import json
from typing import Any

from fastapi.testclient import TestClient

from nemo_gym.core import SESSION_LOST, EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.agentic_turn_simulation.tests.support import make_server


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


def _seed(resources_session_id: str = "resources-session") -> dict[str, Any]:
    # The Environment Server's task_data is the row without responses_create_params.
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=_EPISODE,
        task_id=TaskId(taskset="agent_arena", task_id="0"),
        task_data={"verifier_metadata": _METADATA},
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
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        metadata={"stop_reason": stop_reason, "steps": "50"},
    )
    return {
        "verifier_metadata": _METADATA,
        "responses_create_params": _ROW,
        "response": response.model_dump(mode="json"),
    }


def _client(server=None) -> tuple[Any, TestClient]:
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


def test_the_first_harness_call_brings_the_rows_history() -> None:
    server, client = _client()
    seed = client.post("/seed_session", json=_seed())
    assert seed.status_code == 200 and seed.json()["resources_session_id"] == "resources-session"
    assert set(server._awaiting_row) == {"resources-session"} and server._sessions == {}

    # Before the row arrives the session has no history to simulate against: the turn would be masked.
    early = client.post("/simulate_tool_call", json=_READ).json()
    assert early["source"] == "failed" and "call /harness first" in early["output"]
    missing = client.post("/harness")
    assert missing.status_code == 422 and "responses_create_params" in missing.json()["detail"]

    harness = client.post("/harness", json={"responses_create_params": _ROW})
    assert harness.status_code == 200 and harness.json() == {"stop_tools": ["ask_user"], "protocol": "native"}
    state = server._sessions["resources-session"]
    assert state.workspace.files == {"/home/user/site/index.html": "<h1>Hi</h1>"}
    assert server._awaiting_row == {}

    # A later /harness call (another activation of the turn) keeps the session as it is.
    assert client.post("/harness", json={"responses_create_params": {"input": "other"}}).status_code == 200
    assert server._sessions["resources-session"] is state
    read = client.post("/simulate_tool_call", json=_READ).json()
    assert read["source"] == "exact" and json.loads(read["output"])["content"] == "<h1>Hi</h1>"


def test_verify_counts_the_sessions_calls_and_a_retry_reports_the_same() -> None:
    server, client = _client()
    client.post("/seed_session", json=_seed())
    client.post("/harness", json={"responses_create_params": _ROW})
    client.post("/simulate_tool_call", json=_READ)

    first = client.post("/verify", json=_verify_body())
    assert first.status_code == 200, first.text
    assert (first.json()["code_gate"], first.json()["simulated_calls"]) == ("unfinished_turn", {"exact": 1})
    assert not server.has_session_state("resources-session")
    # The client resends a verify whose reply it lost: the retained counts answer it.
    retry = client.post("/verify", json=_verify_body())
    assert retry.json()["simulated_calls"] == {"exact": 1} and not retry.json()["mask_sample"]


def test_close_releases_the_session_and_refuses_later_calls() -> None:
    server, client = _client()
    client.post("/seed_session", json=_seed())
    client.post("/harness", json={"responses_create_params": _ROW})
    assert client.post("/close_session", json=_close()).status_code == 200
    assert client.post("/close_session", json=_close()).status_code == 200  # idempotent
    assert not server.has_session_state("resources-session")

    for path, body in (("/simulate_tool_call", _READ), ("/harness", {"responses_create_params": _ROW})):
        late = client.post(path, json=body)
        assert late.status_code == 404, late.text
        assert late.json()["detail"]["failure_kind"] == SESSION_LOST
    assert not server.has_session_state("resources-session")

    verified = client.post("/verify", json=_verify_body()).json()
    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)


def test_a_session_whose_agent_never_called_harness_is_released_by_close() -> None:
    server, client = _client()
    client.post("/seed_session", json=_seed())
    assert client.post("/close_session", json=_close()).status_code == 200
    assert server._awaiting_row == {}


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
