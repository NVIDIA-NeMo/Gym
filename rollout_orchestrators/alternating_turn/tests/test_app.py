# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.config_types import AgentServerRef, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from rollout_orchestrators.alternating_turn.app import (
    AlternatingTurnOrchestrator,
    AlternatingTurnOrchestratorConfig,
    AlternatingTurnRunRequest,
)


class _FakeHttpResponse:
    def __init__(self, payload: dict, cookies: dict | None = None) -> None:
        self._payload = payload
        self.cookies = cookies or {}
        self.status = 200
        self.ok = True

    async def read(self):
        return json.dumps(self._payload).encode()


def _orchestrator(max_turns: int = 8) -> AlternatingTurnOrchestrator:
    config = AlternatingTurnOrchestratorConfig(
        host="",
        port=0,
        entrypoint="",
        name="alternating_turn",
        resources_server=ResourcesServerRef(type="resources_servers", name="environment"),
        agents={
            "player0": AgentServerRef(type="responses_api_agents", name="player0_server"),
            "player1": AgentServerRef(type="responses_api_agents", name="player1_server"),
        },
        focal_agent="player0",
        max_turns=max_turns,
    )
    return AlternatingTurnOrchestrator(config=config, server_client=MagicMock(spec=ServerClient))


def test_run_route_belongs_to_orchestrator() -> None:
    paths = {route.path for route in _orchestrator().setup_webserver().routes}
    assert "/run" in paths
    assert "/v1/responses" not in paths


def test_http_run_route_executes_rollout_wrapper() -> None:
    orchestrator = _orchestrator()
    orchestrator.server_client.post = AsyncMock(
        side_effect=[
            _FakeHttpResponse({"active_agent": "player0", "observation": "P0", "info": {}}),
            _FakeHttpResponse({"action": "[bet]"}),
            _FakeHttpResponse(
                {
                    "rewards": {"player0": 1.0, "player1": -1.0},
                    "terminated": True,
                    "truncated": False,
                    "info": {"history": "bet-fold"},
                }
            ),
        ]
    )

    response = TestClient(orchestrator.setup_webserver()).post(
        "/run",
        json={"responses_create_params": {"input": "play"}},
    )

    assert response.status_code == 200
    assert response.json()["reward"] == 1.0


@pytest.mark.asyncio
async def test_default_aggregate_metrics() -> None:
    result = await _orchestrator().aggregate_metrics(
        AggregateMetricsRequest(
            verify_responses=[
                {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0},
            ]
        )
    )
    assert result.key_metrics["mean/reward"] == 1.0


@pytest.mark.asyncio
async def test_aggregate_metrics_reports_each_agent() -> None:
    result = await _orchestrator().aggregate_metrics(
        AggregateMetricsRequest(
            verify_responses=[
                {
                    "_ng_task_index": 0,
                    "_ng_rollout_index": 0,
                    "reward": -2.0,
                    "agent_rewards": {"player0": -2.0, "player1": 2.0},
                    "terminated": True,
                    "truncated": False,
                },
                {
                    "_ng_task_index": 0,
                    "_ng_rollout_index": 1,
                    "reward": -1.0,
                    "agent_rewards": {"player0": -1.0, "player1": 1.0},
                    "terminated": True,
                    "truncated": False,
                },
            ]
        )
    )

    assert result.key_metrics["mean/reward"] == -1.5
    assert result.per_agent_metrics["player0"].key_metrics["mean/reward"] == -1.5
    assert result.per_agent_metrics["player1"].key_metrics["mean/reward"] == 1.5


def test_focal_agent_must_have_a_server() -> None:
    with pytest.raises(ValidationError, match="focal_agent"):
        AlternatingTurnOrchestratorConfig(
            host="",
            port=0,
            entrypoint="",
            name="alternating_turn",
            resources_server=ResourcesServerRef(type="resources_servers", name="environment"),
            agents={
                "player0": AgentServerRef(type="responses_api_agents", name="player0_server"),
                "player1": AgentServerRef(type="responses_api_agents", name="player1_server"),
            },
            focal_agent="player2",
        )


@pytest.mark.asyncio
async def test_orchestrator_routes_private_observations_and_preserves_cookies() -> None:
    orchestrator = _orchestrator()
    calls = []
    responses = [
        _FakeHttpResponse(
            {"active_agent": "player0", "observation": "P0 private", "info": {"seed": 0}},
            {"session": "reset"},
        ),
        _FakeHttpResponse({"action": "[check]"}, {"player0": "cookie"}),
        _FakeHttpResponse(
            {
                "active_agent": "player1",
                "observation": "P1 private",
                "rewards": {},
                "terminated": False,
                "truncated": False,
                "info": {},
            },
            {"session": "step-1"},
        ),
        _FakeHttpResponse({"action": "[check]"}, {"player1": "cookie"}),
        _FakeHttpResponse(
            {
                "active_agent": None,
                "observation": None,
                "rewards": {"player0": 1.0, "player1": -1.0},
                "terminated": True,
                "truncated": False,
                "info": {"history": "check-check"},
            },
            {"session": "step-2"},
        ),
    ]

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        calls.append((server_name, url_path, json, cookies))
        return responses.pop(0)

    orchestrator.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock()
    request.cookies = {"incoming": "cookie"}
    body = AlternatingTurnRunRequest(responses_create_params={"input": [{"role": "user", "content": "play"}]})

    result = await orchestrator.run(request, body)

    assert result.agent_rewards == {"player0": 1.0, "player1": -1.0}
    assert result.reward == 1.0
    assert result.terminated is True
    assert [(server, path) for server, path, _, _ in calls] == [
        ("environment", "/reset"),
        ("player0_server", "/act"),
        ("environment", "/step"),
        ("player1_server", "/act"),
        ("environment", "/step"),
    ]
    assert calls[0][3] == {"incoming": "cookie"}
    assert calls[2][3] == {"session": "reset"}
    assert calls[4][3] == {"session": "step-1"}
    assert calls[1][2]["observation"] == "P0 private"
    assert calls[3][2]["observation"] == "P1 private"
    assert "P1 private" not in json.dumps(calls[1][2])
    assert "P0 private" not in json.dumps(calls[3][2])


@pytest.mark.asyncio
async def test_turn_limit_truncates_and_closes_environment() -> None:
    orchestrator = _orchestrator(max_turns=1)
    paths = []
    responses = [
        _FakeHttpResponse({"active_agent": "player0", "observation": "P0", "info": {}}),
        _FakeHttpResponse({"action": "[check]"}),
        _FakeHttpResponse(
            {
                "active_agent": "player1",
                "observation": "P1",
                "rewards": {},
                "terminated": False,
                "truncated": False,
                "info": {},
            }
        ),
        _FakeHttpResponse({"closed": True}),
    ]

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        paths.append(url_path)
        return responses.pop(0)

    orchestrator.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock()
    request.cookies = {}
    body = AlternatingTurnRunRequest(responses_create_params={"input": "play"})

    result = await orchestrator.run(request, body)

    assert result.truncated is True
    assert result.terminated is False
    assert paths == ["/reset", "/act", "/step", "/close"]
