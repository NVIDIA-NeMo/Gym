# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import AgentServerRef, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.processors.single_agent_turn import (
    SingleAgentTurnProcessor,
    SingleAgentTurnProcessorConfig,
    SingleAgentTurnRunRequest,
)
from nemo_gym.server_utils import ServerClient


def _response(payload: dict, *, cookies: dict | None = None) -> MagicMock:
    response = MagicMock(status=200, ok=True, cookies=cookies or {})
    response.content.read = AsyncMock(return_value=json.dumps(payload).encode())
    response.read = AsyncMock(return_value=json.dumps(payload))
    return response


def _processor(*, skip_verification: bool = False) -> SingleAgentTurnProcessor:
    config = SingleAgentTurnProcessorConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="policy__processor",
        agent_server=AgentServerRef(type="responses_api_agents", name="policy"),
        resources_server=ResourcesServerRef(type="resources_servers", name="environment"),
        skip_verification=skip_verification,
        skip_verification_reward=0.25,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    return SingleAgentTurnProcessor(config=config, server_client=client)


def test_processor_owns_run_route() -> None:
    paths = {route.path for route in _processor().setup_webserver().routes}
    assert "/run" in paths
    assert "/aggregate_metrics" in paths
    assert "/v1/responses" not in paths


@pytest.mark.asyncio
async def test_run_seeds_calls_policy_and_verifies() -> None:
    processor = _processor()
    model_response = {
        "id": "response",
        "created_at": 1,
        "model": "model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    processor.server_client.post = AsyncMock(
        side_effect=[
            _response({}, cookies={"session": "seeded"}),
            _response(model_response, cookies={"session": "policy"}),
            _response(
                {
                    "responses_create_params": {"input": "question"},
                    "response": model_response,
                    "reward": 1.0,
                }
            ),
        ]
    )

    result = await processor.run(
        MagicMock(cookies={}),
        SingleAgentTurnRunRequest(responses_create_params={"input": "question"}),
    )

    assert result.reward == 1.0
    calls = processor.server_client.post.await_args_list
    assert [(call.kwargs["server_name"], call.kwargs["url_path"]) for call in calls] == [
        ("environment", "/seed_session"),
        ("policy", "/v1/responses"),
        ("environment", "/verify"),
    ]
    assert calls[1].kwargs["cookies"] == {"session": "seeded"}
    assert calls[2].kwargs["cookies"] == {"session": "policy"}


def test_http_run_supports_skipped_verification() -> None:
    processor = _processor(skip_verification=True)
    model_response = {
        "id": "response",
        "created_at": 1,
        "model": "model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    processor.server_client.post = AsyncMock(
        side_effect=[
            _response({}, cookies={"session": "seeded"}),
            _response(model_response, cookies={"session": "policy"}),
        ]
    )

    response = TestClient(processor.setup_webserver()).post(
        "/run",
        json={"responses_create_params": {"input": "question"}},
    )

    assert response.status_code == 200
    assert response.json()["reward"] == 0.25
    assert response.json()["verification_skipped"] is True


@pytest.mark.asyncio
async def test_aggregate_metrics_proxies_to_environment() -> None:
    processor = _processor()
    processor.server_client.post = AsyncMock(return_value=_response({"agent_metrics": {"mean/reward": 1.0}}))

    result = await processor.aggregate_metrics(AggregateMetricsRequest(verify_responses=[{"reward": 1.0}]))

    assert result.agent_metrics == {"mean/reward": 1.0}
    call = processor.server_client.post.await_args
    assert call.kwargs["server_name"] == "environment"
    assert call.kwargs["url_path"] == "/aggregate_metrics"
