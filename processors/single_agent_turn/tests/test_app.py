# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import AgentServerRef, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.processors.single_agent_turn import (
    SingleAgentTurnProcessor,
    SingleAgentTurnProcessorConfig,
    SingleAgentTurnRunRequest,
)
from nemo_gym.server_utils import ServerClient


def _response(payload: dict, *, cookies: dict | None = None, status: int = 200) -> MagicMock:
    response = MagicMock(status=status, ok=True, cookies=cookies or {})
    response.content.read = AsyncMock(return_value=json.dumps(payload).encode())
    response.read = AsyncMock(return_value=json.dumps(payload))
    return response


def _no_runtime() -> MagicMock:
    """A `/sandbox_spec` answer from an environment that needs no sandbox."""
    return _response({}, status=204)


def _processor(*, skip_verification: bool = False, sandbox_provider: dict | None = None) -> SingleAgentTurnProcessor:
    config = SingleAgentTurnProcessorConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="policy__processor",
        agent_server=AgentServerRef(type="responses_api_agents", name="policy"),
        resources_server=ResourcesServerRef(type="resources_servers", name="environment"),
        skip_verification=skip_verification,
        skip_verification_reward=0.25,
        sandbox_provider=sandbox_provider,
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
            _no_runtime(),
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
        ("environment", "/sandbox_spec"),
        ("environment", "/seed_session"),
        ("policy", "/v1/responses"),
        ("environment", "/verify"),
    ]
    assert calls[2].kwargs["cookies"] == {"session": "seeded"}
    assert calls[3].kwargs["cookies"] == {"session": "policy"}


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
            _no_runtime(),
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


@pytest.mark.asyncio
async def test_episode_context_reaches_environment_and_harness() -> None:
    """The processor names the environment, so the harness needs none in its own config."""
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
            _no_runtime(),
            _response({}),
            _response(model_response),
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

    calls = processor.server_client.post.await_args_list
    seeded = calls[1].kwargs["json"]["episode_context"]
    assert seeded["env"] == {"type": "resources_servers", "name": "environment"}
    assert seeded["sandbox"] is None

    harness_params = calls[2].kwargs["json"]
    assert harness_params.episode_context.env.name == "environment"

    verified = calls[3].kwargs["json"]["episode_context"]
    assert verified == seeded

    # Scaffolding, not a result: the descriptor must not ride back out into the rollouts file.
    assert result.episode_context is None


@pytest.mark.asyncio
async def test_processor_owns_the_sandbox_lifecycle() -> None:
    """The box is started before seeding and stopped in `finally`, including on failure."""
    processor = _processor(sandbox_provider={"docker": {}})
    started: list[dict] = []
    stopped: list[str] = []

    class _FakeSandbox:
        def __init__(self, provider, spec):
            self.spec = spec

        async def start(self):
            started.append({"image": self.spec.image})
            return self

        async def serialize(self):
            return {"sandbox_id": "box-1", "provider": "fake"}

        async def stop(self):
            stopped.append("box-1")

    with patch("nemo_gym.processors.single_agent_turn.AsyncSandbox", _FakeSandbox):
        processor.server_client.post = AsyncMock(
            side_effect=[
                _response({"image": "python:3.13-slim"}),
                _response({}),
                RuntimeError("harness blew up"),
            ]
        )
        with pytest.raises(RuntimeError, match="harness blew up"):
            await processor.run(
                MagicMock(cookies={}),
                SingleAgentTurnRunRequest(responses_create_params={"input": "question"}),
            )

    assert started == [{"image": "python:3.13-slim"}]
    # The environment never provisioned or tore down anything; the episode's owner did both,
    # and did the teardown on the path where the episode failed.
    assert stopped == ["box-1"]


@pytest.mark.asyncio
async def test_sandbox_descriptor_is_handed_to_both_sides() -> None:
    processor = _processor(sandbox_provider={"docker": {}})
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

    class _FakeSandbox:
        def __init__(self, provider, spec):
            pass

        async def start(self):
            return self

        async def serialize(self):
            return {"sandbox_id": "box-1", "provider": "fake"}

        async def stop(self):
            return None

    with patch("nemo_gym.processors.single_agent_turn.AsyncSandbox", _FakeSandbox):
        processor.server_client.post = AsyncMock(
            side_effect=[
                _response({"image": "python:3.13-slim"}),
                _response({}),
                _response(model_response),
                _response(
                    {
                        "responses_create_params": {"input": "question"},
                        "response": model_response,
                        "reward": 1.0,
                    }
                ),
            ]
        )
        await processor.run(
            MagicMock(cookies={}),
            SingleAgentTurnRunRequest(responses_create_params={"input": "question"}),
        )

    calls = processor.server_client.post.await_args_list
    descriptor = {"sandbox_id": "box-1", "provider": "fake"}
    assert calls[1].kwargs["json"]["episode_context"]["sandbox"] == descriptor
    assert calls[2].kwargs["json"].episode_context.sandbox == descriptor
    assert calls[3].kwargs["json"]["episode_context"]["sandbox"] == descriptor


@pytest.mark.asyncio
async def test_non_connectable_provider_names_the_constraint() -> None:
    processor = _processor(sandbox_provider={"docker": {}})

    class _UnsharableSandbox:
        def __init__(self, provider, spec):
            pass

        async def start(self):
            return self

        async def serialize(self):
            raise RuntimeError("provider 'docker' does not support serialize()/connect()")

        async def stop(self):
            return None

    with patch("nemo_gym.processors.single_agent_turn.AsyncSandbox", _UnsharableSandbox):
        processor.server_client.post = AsyncMock(side_effect=[_response({"image": "python:3.13-slim"})])
        with pytest.raises(RuntimeError, match="sandbox server"):
            await processor.run(
                MagicMock(cookies={}),
                SingleAgentTurnRunRequest(responses_create_params={"input": "question"}),
            )
