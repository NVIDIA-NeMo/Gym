# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEED-Bench replay through the agent sessions API."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import orjson
import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.agents import AGENT_ACTIVATION_HEADER, AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.config import AgentServerRef, BaseServerConfig, ModelServerRef, ResourcesServerRef
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord
from resources_servers.speed_bench.app import SpeedBenchResourcesServer, SpeedBenchResourcesServerConfig
from responses_api_agents.speed_bench_agent.app import SpeedBenchAgent, SpeedBenchAgentConfig


TWO_TURNS = {"input": [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]}
# The seed's episode is attempt 1 of "rollout", whose capture key the Environment Server activates on.
ACTIVATION_PATH = "/ng-rollout/rollout-a1/v1/responses"
SPEC = {
    "num_spec_steps": 2,
    "num_draft_tokens": 4,
    "num_accepted_draft_tokens": 3,
    "num_spec_tokens": 2,
    "acceptance_histogram": [0, 1, 1],
}


def model_reply(text: str) -> dict:
    return {
        "id": f"resp-{text}",
        "created_at": 0.0,
        "model": "policy",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "id": f"msg-{text}",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 8,
        },
        "metrics": {"speculative_decoding": SPEC},
    }


class _Reply:
    """A server reply in the shape ServerClient callers read."""

    def __init__(self, body: bytes, *, status: int = 200, cookies: dict[str, str] | None = None, headers=None):
        self.status = status
        self.ok = status < 400
        self.headers = headers or {}
        self.cookies = {name: SimpleNamespace(value=value) for name, value in (cookies or {}).items()}
        self._body = body
        self.content = SimpleNamespace(read=self.read)

    async def read(self) -> bytes:
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ClientResponseError(MagicMock(), (), status=self.status, message="error")


class _Transport(ServerClient):
    """Serves the policy model from canned replies and every other server from an in-process app."""

    model_config = {"arbitrary_types_allowed": True}
    apps: dict = {}
    model_calls: list = []
    model_headers: list = []
    # A user turn whose model call fails with HTTP 500.
    failing_turn: str | None = None
    # Seconds the agent seed's reply takes to arrive after the seed finished, and when it finished.
    session_delay: float = 0.0
    events: dict = {}

    async def post(self, server_name: str, url_path: str, json=None, cookies=None, headers=None, **kwargs):
        body = json.model_dump(mode="json", exclude_unset=True) if hasattr(json, "model_dump") else json
        if server_name == "policy_model":
            self.model_calls.append((url_path, body))
            self.model_headers.append(headers)
            turn = body["input"][-1]["content"]
            if turn == self.failing_turn:
                return _Reply(b'{"detail": "model unavailable"}', status=500)
            return _Reply(orjson.dumps(model_reply(f"reply to {turn}")), cookies={"model": "m"})
        if url_path == "/verify":
            self.events["verify_body"] = body
        client = self.apps[server_name]
        client.cookies.clear()
        client.cookies.update(dict(cookies or {}))
        reply = client.post(url_path, json=body, headers=headers)
        if url_path == "/v1/agent_sessions":
            # A slow seed reply: a window that started at the seed would include this delay.
            self.events["agent_seed_done"] = time.time()
            await asyncio.sleep(self.session_delay)
        return _Reply(reply.content, status=reply.status_code, cookies=dict(reply.cookies), headers=reply.headers)


def make_transport(*, capture: bool = False, **apps) -> _Transport:
    return _Transport(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(
            {
                "observability_enabled": capture,
                "speed_bench": {"resources_servers": {"speed_bench": {"host": "resources", "port": 9000}}},
                "agent": {"responses_api_agents": {"speed_bench_agent": {"host": "agent", "port": 8080}}},
            }
        ),
        apps=apps,
        model_calls=[],
        model_headers=[],
        events={},
    )


def make_agent(transport: ServerClient) -> SpeedBenchAgent:
    return SpeedBenchAgent(
        config=SpeedBenchAgentConfig(
            host="agent",
            port=8080,
            entrypoint="app.py",
            name="agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="speed_bench"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        ),
        server_client=transport,
    )


def seed_request(**overrides) -> AgentSeedSessionRequest:
    fields = {
        "agent_session_id": "agent-session",
        "episode_id": EpisodeId(rollout_id="rollout", attempt=1),
        "task_id": TaskId(taskset="speed_bench", task_id="task"),
    }
    return AgentSeedSessionRequest(**(fields | overrides))


def test_a_session_activation_replays_each_turn_once() -> None:
    transport = make_transport()
    client = TestClient(make_agent(transport).setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json")).status_code == 200

    reply = client.post(ACTIVATION_PATH, json=TWO_TURNS, headers={AGENT_ACTIVATION_HEADER: "1"})

    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert [item["content"][0]["text"] for item in body["output"]] == ["reply to first", "reply to second"]
    assert body["usage"]["output_tokens"] == 6
    assert len(body["metrics"]["requests"]) == 2
    assert body["metrics"]["generation_duration"] >= 0
    # Model calls keep the rollout's capture prefix; the model's cookie is not handed back.
    assert [path for path, _ in transport.model_calls] == [ACTIVATION_PATH] * 2
    assert "model" not in reply.cookies
    # Without capture the reply carries no trajectory and the model calls name no invocation.
    assert "_ng_trajectory" not in body
    assert transport.model_headers == [None, None]

    # A retried activation gets the replay it already ran, not new timed model calls.
    again = client.post(ACTIVATION_PATH, json=TWO_TURNS, headers={AGENT_ACTIVATION_HEADER: "1"})
    assert again.json() == body
    assert len(transport.model_calls) == 2

    close = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "agent-session", "episode_id": seed_request().episode_id.model_dump()},
    )
    assert close.status_code == 200, close.text
    # With model-call capture off there is no evidence to return, as on /run.
    assert close.json()["agent_observations"] is None
    assert "trajectory" not in close.json()


def _close(client: TestClient) -> dict:
    close = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "agent-session", "episode_id": seed_request().episode_id.model_dump()},
    )
    assert close.status_code == 200, close.text
    return close.json()


def test_a_captured_session_returns_the_replay_evidence_at_close() -> None:
    transport = make_transport(capture=True)
    client = TestClient(make_agent(transport).setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json")).status_code == 200

    reply = client.post(ACTIVATION_PATH, json=TWO_TURNS, headers={AGENT_ACTIVATION_HEADER: "1"})

    assert reply.status_code == 200, reply.text
    # The Environment Server takes the evidence at close; the reply it verifies carries none.
    assert "_ng_trajectory" not in reply.json()
    assert transport.model_headers == [{"x-session-id": "root"}] * 2
    closed = _close(client)
    observations = AgentObservationBundle.model_validate(closed["agent_observations"])
    assert observations.source == "speed_bench_agent"
    assert not observations.gaps
    [invocation] = observations.records
    assert (invocation.invocation_id, invocation.in_session, invocation.status) == ("root", True, "completed")
    assert [ref.response_id for ref in invocation.model_calls] == ["resp-reply to first", "resp-reply to second"]
    # The trajectory is bound to the seeded task and the episode's capture key, as rollout collection reads it.
    trajectory = TrajectoryRecord.model_validate(closed["trajectory"])
    assert (trajectory.task_id, trajectory.rollout_id) == ("task", "rollout-a1")
    assert [(turn.task_id, turn.rollout_id, turn.turn_no) for turn in trajectory.turns] == [
        ("task", "rollout-a1", 1),
        ("task", "rollout-a1", 2),
    ]
    assert [len(turn.question) for turn in trajectory.turns] == [1, 3]
    assert trajectory.invocations[0].in_session is True


def test_a_failed_activation_keeps_its_observations_for_close() -> None:
    transport = make_transport(capture=True)
    transport.failing_turn = "second"
    app = make_agent(transport).setup_webserver()
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json")).status_code == 200

    reply = client.post(ACTIVATION_PATH, json=TWO_TURNS, headers={AGENT_ACTIVATION_HEADER: "1"})

    assert reply.status_code == 500, reply.text
    closed = _close(client)
    [invocation] = AgentObservationBundle.model_validate(closed["agent_observations"]).records
    assert (invocation.status, invocation.error_type) == ("failed", "ClientResponseError")
    # The first turn's model call stays owned by the invocation; no trajectory claims a finished replay.
    assert [ref.response_id for ref in invocation.model_calls] == ["resp-reply to first"]
    assert "trajectory" not in closed


def test_a_captured_session_without_an_activation_reports_the_gap() -> None:
    client = TestClient(make_agent(make_transport(capture=True)).setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json")).status_code == 200

    observations = AgentObservationBundle.model_validate(_close(client)["agent_observations"])

    # Capture is on, so close says why it returns no replay instead of returning no evidence.
    assert [gap.code for gap in observations.gaps] == ["observation_capture_failed"]
    assert not observations.records


def test_an_activation_must_keep_the_seeded_rollout_route() -> None:
    transport = make_transport()
    client = TestClient(make_agent(transport).setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json")).status_code == 200

    # Another rollout, or no rollout at all, would put the replay's model calls under another capture key.
    assert client.post("/ng-rollout/rollout/v1/responses", json=TWO_TURNS).status_code == 409
    assert client.post("/v1/responses", json=TWO_TURNS).status_code == 409
    assert client.post(ACTIVATION_PATH, json=TWO_TURNS).status_code == 200
    # A retry on another route of the same rollout is refused rather than joined to the first replay.
    tagged = "/ng-rollout/rollout-a1/training-token-capture/v1/responses"
    assert client.post(tagged, json=TWO_TURNS).status_code == 409
    assert len(transport.model_calls) == 2


def test_a_call_without_a_session_still_propagates_cookies() -> None:
    client = TestClient(make_agent(make_transport()).setup_webserver())
    client.cookies.set("resources", "r")

    reply = client.post("/v1/responses", json=TWO_TURNS)

    assert reply.status_code == 200, reply.text
    assert reply.cookies.get("resources") == "r"
    assert "model" in reply.cookies


def test_optional_tool_access_is_ignored() -> None:
    optional = DirectHTTPToolAccess(name="tools", required=False, base_url="http://tools.test")
    client = TestClient(make_agent(make_transport()).setup_webserver())
    reply = client.post("/v1/agent_sessions", json=seed_request(tool_accesses=[optional]).model_dump(mode="json"))
    assert reply.status_code == 200, reply.text


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {
                "sandbox_access": SandboxAccess(
                    connection=DirectSandboxConnection(provider_config_ref="docker", descriptor={}), workdir="/w"
                )
            },
            "does not support sandbox access",
        ),
        (
            {"tool_accesses": [DirectHTTPToolAccess(name="tools", required=True, base_url="http://tools.test")]},
            "required tool access: tools",
        ),
        (
            {
                "tool_accesses": [
                    MCPToolAccess(
                        name="search", required=True, connection=MCPStreamableHTTPConnection(url="http://mcp.test")
                    )
                ]
            },
            "required tool access: search",
        ),
    ],
)
def test_unsupported_seeds_are_refused(overrides: dict, reason: str) -> None:
    detail = check_agent_seed_rejected(make_agent(make_transport()).setup_webserver(), seed_request(**overrides))
    assert reason in detail


@pytest.mark.parametrize("capture", [False, True])
async def test_a_flat_row_episode_keeps_the_replay_window_and_metadata(capture: bool) -> None:
    """A shipped-config episode: a flat row through single_agent_turn_legacy and the real Resources Server."""
    resources = SpeedBenchResourcesServer(
        config=SpeedBenchResourcesServerConfig(host="resources", port=9000, entrypoint="", name="speed_bench"),
        server_client=MagicMock(spec=ServerClient),
    )
    transport = make_transport(capture=capture, speed_bench=TestClient(resources.setup_webserver()))
    transport.apps["agent"] = TestClient(make_agent(transport).setup_webserver())
    # The agent seed's reply arrives this long after the seed; the replay's window must not include it.
    transport.session_delay = 0.3
    environment = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnLegacyEnvironmentServerConfig(
            name="environment",
            host="environment",
            port=8002,
            entrypoint="app.py",
            resources_server=ResourcesServerRef(type="resources_servers", name="speed_bench"),
            agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
            cleanup_timeout_seconds=10,
        ),
        server_client=transport,
    )

    result = await environment.run_legacy(
        {
            "_ng_task_index": 4,
            "_ng_rollout_index": 0,
            "responses_create_params": TWO_TURNS,
            "verifier_metadata": {"src_id": "qualitative-7", "speed_config": "qualitative", "num_turns": 2},
        }
    )

    assert "_ng_failure_class" not in result, result
    timing = result["response"]["metrics"]
    # The window starts after the agent seed: it covers only the replay's model calls.
    assert timing["generation_started_at"] >= transport.events["agent_seed_done"]
    assert timing["generation_duration"] < transport.session_delay
    assert result["gen_seconds"] == timing["generation_duration"]
    assert result["num_generated_tokens"] == 6
    assert result["acceptance_length"] == pytest.approx(2.5)
    assert result["spec_decode_unavailable"] is False
    # The verifier gets the row's metadata and the replay's response, without the trajectory carrier.
    verify_body = transport.events["verify_body"]
    assert verify_body["verifier_metadata"]["src_id"] == "qualitative-7"
    assert verify_body["response"]["metrics"] == timing
    assert "_ng_trajectory" not in verify_body["response"]
    if capture:
        trajectory = TrajectoryRecord.model_validate(result["ng_trajectory"])
        assert trajectory.rollout_id == "4-0"
        assert [turn.turn_no for turn in trajectory.turns] == [1, 2]
        assert trajectory.invocations[0].in_session is True
        observations = AgentObservationBundle.model_validate(result["ng_agent_observations"])
        assert not observations.gaps
    else:
        assert "ng_trajectory" not in result
