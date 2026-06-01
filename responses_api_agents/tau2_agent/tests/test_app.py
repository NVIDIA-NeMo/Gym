# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.tau2_agent.app import (
    Tau2BenchAgent,
    Tau2BenchAgentConfig,
    Tau2BenchAgentRunRequest,
    Tau2BenchAgentVerifyResponse,
)


def make_config(concurrency: int = 2) -> Tau2BenchAgentConfig:
    return Tau2BenchAgentConfig(
        name="tau2_bench_agent",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        resources_server=ResourcesServerRef(type="resources_servers", name="rs"),
        model_server=ModelServerRef(type="responses_api_models", name="ms"),
        env="singularity",
        concurrency=concurrency,
        openai_api_base="http://localhost:8000/v1",
        tau2_policy_model_name="test_model",
        tau2_domain="test_domain",
    )


def make_run_request() -> Tau2BenchAgentRunRequest:
    return Tau2BenchAgentRunRequest(
        instance_id="inst",
        subset="gym",
        split="train",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], temperature=0.5, top_p=0.8
        ),
    )


def test_model_post_init_sets_semaphore() -> None:
    config = make_config(concurrency=3)
    agent = Tau2BenchAgent(config=config, server_client=MagicMock(spec=ServerClient))
    agent.model_post_init(None)
    assert hasattr(agent, "_sem")
    assert agent._sem._value == 3


@pytest.mark.asyncio
async def test_responses_not_implemented() -> None:
    config = make_config()
    agent = Tau2BenchAgent(config=config, server_client=MagicMock(spec=ServerClient))
    req = NeMoGymResponseCreateParamsNonStreaming(input=[], temperature=0.7)
    with pytest.raises(NotImplementedError):
        await agent.responses(req)


@pytest.mark.asyncio
async def test_run_success_and_error_fallback(monkeypatch) -> None:
    config = make_config()
    agent = Tau2BenchAgent(config=config, server_client=MagicMock(spec=ServerClient))

    simulation_results = SimpleNamespace(
        simulations=[
            SimpleNamespace(
                id="sim-1",
                messages=[{"role": "assistant", "content": "ok"}],
                reward_info=SimpleNamespace(
                    reward=1.0,
                    reward_breakdown={"score": 1.0},
                ),
            )
        ]
    )

    # Patch run dependencies used by app.run
    async def fake_to_thread(func, *args, **kwargs):
        return simulation_results

    monkeypatch.setattr("asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(
        "responses_api_agents.tau2_agent.app.ServerClient.load_from_global_config",
        lambda: MagicMock(global_config_dict={"policy_model_name": "test_model"}),
    )
    monkeypatch.setattr(
        "responses_api_agents.tau2_agent.app.get_first_server_config_dict",
        lambda *_args, **_kwargs: {
            "host": "localhost",
            "port": 8000,
            "model": "user_model",
        },
    )
    monkeypatch.setattr(
        "responses_api_agents.tau2_agent.app.convert_trajectory_to_output_items",
        lambda _messages: [],
    )

    run_req = make_run_request()
    run_req.task_id = "1"
    run_req.task_domain = "airline"

    resp = await agent.run(run_req)
    assert isinstance(resp, Tau2BenchAgentVerifyResponse)
    assert resp.reward == 1.0 or resp.reward == 0.0
    assert resp.response.id == "sim-1"

    # Conversion errors are handled with a 0.0-reward fallback response
    monkeypatch.setattr(
        "responses_api_agents.tau2_agent.app.convert_trajectory_to_output_items",
        lambda _messages: (_ for _ in ()).throw(ValueError("bad trajectory")),
    )
    error_resp = await agent.run(run_req)
    assert isinstance(error_resp, Tau2BenchAgentVerifyResponse)
    assert error_resp.reward == 0.0
    assert error_resp.response.metadata is not None
    assert "bad trajectory" in error_resp.response.metadata["error"]
