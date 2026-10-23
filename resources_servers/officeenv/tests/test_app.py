# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The HTTP wiring of the OfficeEnv resources server, driven the way Gym's agents and Environment
Servers drive it.

What an episode does (apps, faults, checks, reward) is OfficeEnv's and tested there; these tests
cover the routes, statuses, session cookies, the Environment Server session contract, the MCP
exposure and Gym's response fields, and that the committed example rows still match the pinned
OfficeEnv release."""

import asyncio
import json
from http.cookiejar import CookieJar
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from officeenv import __version__ as officeenv_version
from officeenv.apps import available
from officeenv.emit import gym_row
from pydantic import ValidationError

from nemo_gym.base_resources_server import NEMO_GYM_MCP_SESSION_TOKEN_HEADER, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.failure_kinds import SESSION_LOST
from nemo_gym.mcp_auto_exposure import maybe_auto_expose
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.officeenv.app import (
    LengthPenaltyConfig,
    OfficeEnvResourcesServer,
    OfficeEnvResourcesServerConfig,
    length_penalty_fields,
    model_turns,
    tools_exposure_catalog,
)


ME, AVA = "alex.morgan@example.com", "ava.rossi@example.com"
SPEC = {
    "task_id": "gym-smoke",
    "seed": 3,
    "actor": {"email": ME},
    "apps": {
        "slack": {
            "data": {
                "users": [
                    {"id": "U001", "name": "alex.morgan", "real_name": "Alex Morgan", "email": ME},
                    {"id": "U002", "name": "ava.rossi", "real_name": "Ava Rossi", "email": AVA},
                ],
                "channels": [
                    {"id": "C001", "name": "sales", "members": ["U001", "U002"]},
                    {"id": "C002", "name": "general", "members": ["U001", "U002"]},
                ],
            }
        }
    },
    "exposure": {"mode": "tools"},
    "faults": [{"kind": "transient_error", "match": {"op": "slack.post_message"}}],
    "checks": [
        {"check": "slack.message_posted", "args": {"channel": "#sales", "contains": "renewal"}},
        {"check": "slack.no_message_posted", "args": {"channel": "#general"}, "guard": True},
    ],
}
PROMPT = "Post 'renewal signed' in #sales"
POST = {"channel": "#sales", "text": "renewal signed"}
RESPONSE = {
    "id": "resp_1",
    "created_at": 0.0,
    "model": "x",
    "object": "response",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}
EXAMPLES = Path(__file__).parents[1] / "data" / "example.jsonl"
RPC_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class StatelessCookieJar(CookieJar):
    def extract_cookies(self, response: object, request: object) -> None:
        # Sessions must come from explicit request cookies, as the agent sends them.
        pass


def make_server(**config: object) -> OfficeEnvResourcesServer:
    settings = {"host": "0.0.0.0", "port": 8080, "entrypoint": "", "name": "officeenv", **config}
    return OfficeEnvResourcesServer(
        config=OfficeEnvResourcesServerConfig(**settings), server_client=MagicMock(spec=ServerClient)
    )


@pytest.fixture
def server() -> OfficeEnvResourcesServer:
    return make_server()


@pytest.fixture
def client(server: OfficeEnvResourcesServer) -> TestClient:
    return TestClient(server.setup_webserver(), cookies=StatelessCookieJar())


def seed_request(
    spec: dict = SPEC, *, session_id: str = "rs-1", rollout_id: str = "r1"
) -> ResourcesSeedSessionRequest:
    """The seed an Environment Server sends for ``spec``'s row: every row field besides the model's request."""
    row = gym_row(spec, PROMPT)
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=rollout_id),
        task_id=TaskId(taskset="officeenv", task_id=row["task_id"]),
        task_data={key: value for key, value in row.items() if key != "responses_create_params"},
    )


def close_body(seed: ResourcesSeedSessionRequest, *, rollout_id: str | None = None) -> dict:
    episode_id = EpisodeId(rollout_id=rollout_id) if rollout_id else seed.episode_id
    return {"resources_session_id": seed.resources_session_id, "episode_id": episode_id.model_dump(mode="json")}


def environment_verify_body(seed: ResourcesSeedSessionRequest, response: dict = RESPONSE) -> dict:
    """The verify body an Environment Server sends: the task's fields, the model's request and its response."""
    row = gym_row(SPEC, PROMPT)
    return seed.task_data | {"responses_create_params": row["responses_create_params"], "response": response}


def test_a_row_runs_through_seed_tool_calls_and_verify(client: TestClient) -> None:
    row = gym_row(SPEC, PROMPT)
    seeded = client.post("/seed_session", json=row)
    assert seeded.status_code == 200, seeded.text
    cookies = seeded.cookies
    body = seeded.json()
    assert body["task_id"] == "gym-smoke" and body["apps"] == ["slack"]
    assert "slack_post_message" in body["tools"] and body["tool_drift"] == []
    assert body["officeenv_server_version"] == officeenv_version

    faulted = client.post("/slack_post_message", json=POST, cookies=cookies)
    assert faulted.status_code == 200 and faulted.json()["error"] == "service_unavailable"
    assert client.post("/slack_post_message", json=POST, cookies=cookies).json()["ok"] is True

    verified = client.post("/verify", json=dict(row, response=RESPONSE), cookies=cookies)
    assert verified.status_code == 200, verified.text
    out = verified.json()
    assert out["reward"] == 1.0 and out["mask_sample"] is False
    assert out["checks_passed"] == 2.0 and out["faults"] == 1.0 and out["policy_calls"] == 2.0
    assert [c["passed"] for c in out["officeenv_checks"]] == [True, True]
    assert out["officeenv_server_version"] == officeenv_version
    assert out["officeenv_version"] == row["officeenv_version"]  # the row round-trips
    assert out["officeenv"]["task_id"] == "gym-smoke"
    assert "reward_components" not in out and "length_penalty" not in out  # off by default


def _user_op(mode: str, op: str, params: dict) -> tuple[str, dict]:
    """The tool call that runs the user app's ``op`` under exposure ``mode``."""
    if mode == "tools":
        return f"user_{op}", params
    if mode == "api_search":
        return "api_call", {"endpoint": f"user.{op}", "params": params}
    if mode == "cli":
        flags = " ".join(f"--{k} {v}" for k, v in params.items())
        return "run_command", {"command": f"user {op.replace('_', '-')} {flags}".strip()}
    return "call_tool", {"server": "user", "name": op, "arguments": params}


@pytest.mark.parametrize("mode", ["tools", "api_search", "cli", "mcp"])
def test_every_exposure_reaches_the_apps_through_the_tool_route(client: TestClient, mode: str) -> None:
    spec = {
        "task_id": "gym-done",
        "seed": 1,
        "actor": {"email": ME},
        "apps": {"user": {}},
        "exposure": {"mode": mode},
        "checks": [{"check": "user.task_completed", "args": {}}],
    }
    row = gym_row(spec, "Tell me when you are done")
    seeded = client.post("/seed_session", json=row)
    assert seeded.json()["tools"] == [tool["name"] for tool in row["responses_create_params"]["tools"]]

    tool, args = _user_op(mode, "complete_task", {"answer": "done"})
    completed = client.post(f"/{tool}", json=args, cookies=seeded.cookies)
    assert completed.status_code == 200, completed.text
    verified = client.post("/verify", json=dict(row, response=RESPONSE), cookies=seeded.cookies).json()
    assert verified["reward"] == 1.0 and verified["policy_calls"] == 1.0, verified["officeenv_checks"]


@pytest.mark.parametrize("forbidden", [False, True])
def test_structured_reports_round_trip_through_http(client: TestClient, forbidden: bool) -> None:
    content = {"fields": ["answer"], "path": "rows", "key": "id", "expected": [{"id": "x", "tags": ["a", "b"]}]}
    spec = {
        "task_id": "gym-structured",
        "actor": {"email": ME},
        "apps": {"user": {}},
        "checks": [
            {"check": "world.records_match", "args": {"app": "user", "source": "reply", "content": content}},
            {
                "check": "world.records_match",
                "args": {
                    "app": "user",
                    "source": "reply",
                    "content": {
                        "fields": ["answer"],
                        "path": "rows",
                        "mode": "not_contains",
                        "expected": [{"citations": "bad"}],
                    },
                },
            },
            {
                "check": "world.records_match",
                "args": {
                    "app": "user",
                    "source": "reply",
                    "content": {
                        "fields": ["answer"],
                        "path": "totals",
                        "mode": "value",
                        "expected": {"net_amount": 1200},
                    },
                },
            },
        ],
    }
    row = gym_row(spec, "Return a structured review with rows and totals.")
    seeded = client.post("/seed_session", json=row)
    assert seeded.status_code == 200 and seeded.json()["tool_drift"] == []
    answer = {
        "Rows": [{"ID": "x", "tags": "B; A", "citations": ["good", "bad"] if forbidden else ["good"]}],
        "Totals": {"Net Amount": "$1,200", "extra": 1},
    }
    completed = client.post("/user_complete_task", json={"answer": answer}, cookies=seeded.cookies)
    assert completed.status_code == 200, completed.text
    verified = client.post("/verify", json=dict(row, response=RESPONSE), cookies=seeded.cookies)
    assert verified.status_code == 200
    assert [c["passed"] for c in verified.json()["officeenv_checks"]] == [True, not forbidden, True]


def test_integration_failures_keep_their_statuses(client: TestClient, server: OfficeEnvResourcesServer) -> None:
    row = gym_row(SPEC, "x")
    before_seed = client.post("/slack_list_channels", json={})
    assert before_seed.status_code == 409 and before_seed.json()["error"]["code"] == "no_session"

    cookies = client.post("/seed_session", json=row).cookies
    malformed = client.post(
        "/slack_list_channels", content=b"{not json", headers={"content-type": "application/json"}, cookies=cookies
    )
    assert malformed.status_code == 400 and malformed.json()["error"]["code"] == "invalid_arguments"
    unknown = client.post("/slack_fly", json={}, cookies=cookies)  # a model-visible tool output
    assert unknown.status_code == 200 and unknown.json()["error"]["code"] == "unknown_tool"

    stored = len(server.service.store)
    no_actor = json.loads(json.dumps(row))
    no_actor["officeenv"].pop("actor")
    invalid = client.post("/seed_session", json=no_actor)
    assert invalid.status_code == 400 and invalid.json()["detail"]["error"]["code"] == "invalid_task"
    other_major = dict(row, officeenv_version="99.0.0")
    refused = client.post("/seed_session", json=other_major)
    assert refused.status_code == 400 and refused.json()["detail"]["error"]["code"] == "incompatible_row"
    assert len(server.service.store) == stored  # a refused seed stores nothing


def test_verifying_a_lost_session_masks_the_sample(client: TestClient) -> None:
    row = gym_row(SPEC, "x")
    out = client.post("/verify", json=dict(row, response=RESPONSE)).json()
    assert out["reward"] == 0.0 and out["mask_sample"] is True
    assert out["failure_kind"] == SESSION_LOST and "no officeenv session" in out["failure_reason"]


def test_the_app_catalog_is_browsable(client: TestClient) -> None:
    apps = client.get("/officeenv/apps").json()["apps"]
    assert "slack" in apps and "post_message" in apps["slack"]["ops"]


def test_the_example_rows_match_the_pinned_officeenv(client: TestClient) -> None:
    rows = [json.loads(line) for line in EXAMPLES.read_text().splitlines()]
    assert len(rows) == 5
    for row in rows:
        seeded = client.post("/seed_session", json=row)
        assert seeded.status_code == 200, (row["task_id"], seeded.text)
        assert seeded.json()["tool_drift"] == [], (row["task_id"], "re-emit data/example.jsonl")
        assert row["officeenv_version"] == officeenv_version, "re-emit data/example.jsonl"


# Environment Server episodes: typed seed and close.


def test_environment_server_sessions_keep_the_resources_session_contract(server: OfficeEnvResourcesServer) -> None:
    check_resources_session_contract(server.setup_webserver(), seed_request(), keeps_state=True)


def test_an_environment_server_episode_runs_under_its_resources_session_id(
    client: TestClient, server: OfficeEnvResourcesServer
) -> None:
    seed = seed_request()
    seeded = client.post("/seed_session", json=seed.model_dump(mode="json"))
    assert seeded.status_code == 200, seeded.text
    assert seeded.json() == {"resources_session_id": "rs-1", "resources_tools": None, "sandbox_access": None}
    assert server.service.store.get("rs-1") is not None

    cookies = seeded.cookies
    assert client.post("/slack_post_message", json=POST, cookies=cookies).json()["error"] == "service_unavailable"
    assert client.post("/slack_post_message", json=POST, cookies=cookies).json()["ok"] is True
    verified = client.post("/verify", json=environment_verify_body(seed), cookies=cookies)
    assert verified.status_code == 200 and verified.json()["reward"] == 1.0, verified.text
    assert verified.json()["officeenv"]["task_id"] == "gym-smoke"

    assert client.post("/close_session", json=close_body(seed), cookies=cookies).status_code == 200
    assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 409


def test_a_seed_binds_its_id_to_one_episode_and_close_frees_an_unverified_world(
    client: TestClient, server: OfficeEnvResourcesServer
) -> None:
    seed = seed_request()
    assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 200
    other = seed_request(rollout_id="r2")
    assert client.post("/seed_session", json=other.model_dump(mode="json")).status_code == 409
    assert client.post("/close_session", json=close_body(seed, rollout_id="r2")).status_code == 409
    assert len(server.service.store) == 1

    assert client.post("/close_session", json=close_body(seed)).status_code == 200
    assert len(server.service.store) == 0 and server.service.notes == {}
    assert client.post("/close_session", json=close_body(seed, rollout_id="r2")).status_code == 409


def test_a_refused_typed_seed_keeps_officeenvs_code_and_leaves_the_id_free(
    client: TestClient, server: OfficeEnvResourcesServer
) -> None:
    seed = seed_request()
    unknown_app = seed.model_copy(update={"task_data": seed.task_data | {"officeenv": {**SPEC, "apps": {"nope": {}}}}})
    refused = client.post("/seed_session", json=unknown_app.model_dump(mode="json"))
    assert refused.status_code == 400 and refused.json()["detail"]["error"]["code"] == "unknown_app"
    assert len(server.service.store) == 0
    assert client.post("/seed_session", json=seed_request().model_dump(mode="json")).status_code == 200


def test_retiring_a_session_frees_its_world(client: TestClient, server: OfficeEnvResourcesServer) -> None:
    cookies = client.post("/seed_session", json=gym_row(SPEC, PROMPT)).cookies
    (session_id,) = server.service.store.sessions
    asyncio.run(server.retire_session_state(session_id))
    assert len(server.service.store) == 0
    assert client.post("/slack_list_channels", json={}, cookies=cookies).status_code == 409


# The MCP exposure.


class MCPClient(TestClient):
    """A test client whose lifespan starts the MCP session manager, then the MCP handshake."""

    def __enter__(self) -> "MCPClient":
        super().__enter__()
        params = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}
        initialize = {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": params}
        self.post("/mcp", headers=RPC_HEADERS, json=initialize)
        self.post("/mcp", headers=RPC_HEADERS, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self


def mcp_client(server: OfficeEnvResourcesServer) -> MCPClient:
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    return MCPClient(app, cookies=StatelessCookieJar())


def rpc(client: TestClient, method: str, params: dict, token: str) -> dict:
    headers = {**RPC_HEADERS, NEMO_GYM_MCP_SESSION_TOKEN_HEADER: token}
    return client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})


def test_the_mcp_catalog_serves_each_tool_as_rows_show_it_to_the_model() -> None:
    rows = [json.loads(line) for line in EXAMPLES.read_text().splitlines()] + [gym_row(SPEC, PROMPT)]
    shown = [
        tool
        for row in rows
        if row["officeenv"].get("exposure", {}).get("mode", "tools") == "tools"
        for tool in row["responses_create_params"]["tools"]
    ]
    catalog = tools_exposure_catalog()
    assert len(shown) > 80 and all(catalog[tool["name"]] == tool for tool in shown)
    assert {name for name in available() if not any(tool.startswith(f"{name}_") for tool in catalog)} == set()


def test_an_episode_token_admits_only_the_episodes_tools_over_mcp() -> None:
    server = make_server(expose_tools_over_mcp=True)
    with mcp_client(server) as client:
        seed = seed_request()
        seeded = client.post("/seed_session", json=seed.model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        access = seeded.json()["resources_tools"]
        assert access["server_name"] == "officeenv" and access["url_path"] == "/mcp"
        token = access["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]

        listed = rpc(client, "tools/list", {}, token).json()["result"]["tools"]
        shown = {tool["name"]: tool for tool in gym_row(SPEC, PROMPT)["responses_create_params"]["tools"]}
        assert {tool["name"] for tool in listed} == set(shown)
        for tool in listed:
            assert (tool["description"], tool["inputSchema"]) == (
                shown[tool["name"]]["description"],
                shown[tool["name"]]["parameters"],
            )

        email = {"to": [AVA], "subject": "renewal", "body": "signed"}
        refused = rpc(client, "tools/call", {"name": "gmail_send_message", "arguments": email}, token).json()
        assert "not allowed" in json.dumps(refused)
        faulted = rpc(client, "tools/call", {"name": "slack_post_message", "arguments": POST}, token).json()
        assert faulted["result"]["structuredContent"]["error"] == "service_unavailable"
        posted = rpc(client, "tools/call", {"name": "slack_post_message", "arguments": POST}, token).json()
        assert posted["result"]["structuredContent"]["ok"] is True

        verified = client.post("/verify", json=environment_verify_body(seed), cookies=seeded.cookies)
        assert verified.status_code == 200 and verified.json()["reward"] == 1.0, verified.text


def test_an_agent_seed_carries_its_mcp_token() -> None:
    server = make_server(expose_tools_over_mcp=True)
    with mcp_client(server) as client:
        row = gym_row(SPEC, PROMPT)
        seeded = client.post("/seed_session", json=row)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["tools"] == [tool["name"] for tool in row["responses_create_params"]["tools"]]
        token = seeded.json()["mcp"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        listed = rpc(client, "tools/list", {}, token).json()["result"]["tools"]
        assert {tool["name"] for tool in listed} == set(seeded.json()["tools"])


@pytest.mark.parametrize("exposure", [{"mode": "api_search"}, {"mode": "tools", "options": {"separator": "__"}}])
def test_the_mcp_exposure_refuses_tools_it_does_not_serve(exposure: dict) -> None:
    server = make_server(expose_tools_over_mcp=True)
    with mcp_client(server) as client:
        refused = client.post(
            "/seed_session", json=seed_request({**SPEC, "exposure": exposure}).model_dump(mode="json")
        )
        assert refused.status_code == 400 and refused.json()["detail"]["error"]["code"] == "unsupported_exposure"
        assert len(server.service.store) == 0 and "rs-1" not in server._episodes


# The training length penalty.


def _reasoning(n: int) -> dict:
    return {"id": f"rs-{n}", "summary": [{"text": "plan", "type": "summary_text"}], "type": "reasoning"}


def _call(n: int) -> dict:
    return {
        "id": f"fc-{n}",
        "call_id": f"call-{n}",
        "name": "slack_post_message",
        "arguments": "{}",
        "type": "function_call",
    }


def _call_output(n: int) -> dict:
    return {"call_id": f"call-{n}", "output": "{}", "type": "function_call_output"}


# Three model responses: two calls at once, one call, then the closing message.
EPISODE = [
    _reasoning(1),
    _call(1),
    _call(2),
    _call_output(1),
    _call_output(2),
    _reasoning(2),
    _call(3),
    _call_output(3),
    _reasoning(3),
    {
        "id": "msg-1",
        "content": [{"annotations": [], "text": "done", "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    },
]


def _usage(output_tokens: int) -> dict:
    return {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 1000 + output_tokens,
    }


def test_the_length_penalty_is_reported_as_its_own_reward_term() -> None:
    server = make_server(length_penalty={"enabled": True, "turn_scale": 20, "token_scale": 2000})
    client = TestClient(server.setup_webserver(), cookies=StatelessCookieJar())
    row = gym_row(SPEC, PROMPT)
    cookies = client.post("/seed_session", json=row).cookies
    for _ in range(2):  # the first post meets the injected fault
        client.post("/slack_post_message", json=POST, cookies=cookies)

    response = RESPONSE | {"output": EPISODE, "usage": _usage(500)}
    out = client.post("/verify", json=dict(row, response=response), cookies=cookies).json()
    # 3 turns / 20 + 500 tokens / 2000 = 0.4 off OfficeEnv's 1.0
    assert out["task_reward"] == 1.0 and out["length_penalty"] == pytest.approx(0.4)
    assert out["model_turns"] == 3.0 and out["generated_tokens"] == 500.0
    assert out["reward_components"] == pytest.approx({"task_reward": 1.0, "length_penalty": -0.4})
    assert sum(out["reward_components"].values()) == out["reward"] == pytest.approx(0.6)


def test_the_length_penalty_spares_low_rewards_and_is_capped() -> None:
    config = LengthPenaltyConfig(enabled=True, turn_scale=20, token_scale=2000)
    response = NeMoGymResponse.model_validate(RESPONSE | {"output": EPISODE, "usage": _usage(5000)})
    capped = length_penalty_fields(0.9, response, config)
    assert capped["length_penalty"] == 0.5 and capped["reward"] == pytest.approx(0.4)
    spared = length_penalty_fields(0.5, response, config)  # at the threshold: unchanged
    assert spared["reward"] == 0.5 and spared["reward_components"] == {"task_reward": 0.5, "length_penalty": 0.0}

    no_usage = NeMoGymResponse.model_validate(RESPONSE | {"output": EPISODE})
    failed = length_penalty_fields(0.0, no_usage, config)  # e.g. a lost session: nothing to deduct
    assert failed["reward"] == 0.0 and "generated_tokens" not in failed
    with pytest.raises(ValueError, match="usage"):
        length_penalty_fields(1.0, no_usage, config)
    with pytest.raises(ValidationError):
        LengthPenaltyConfig(enabled=True, turn_scal=400)


def test_model_turns_count_the_model_responses() -> None:
    assert model_turns(NeMoGymResponse.model_validate(RESPONSE | {"output": EPISODE})) == 3
    assert model_turns(NeMoGymResponse.model_validate(RESPONSE)) == 0
