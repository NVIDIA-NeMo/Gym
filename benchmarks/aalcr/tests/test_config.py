# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import NeMoGymAsyncOpenAI, SimpleModelServer, SimpleModelServerConfig


BENCHMARK_DIR = Path(__file__).parents[1]
V1_1_CONFIG = BENCHMARK_DIR / "config_v1_1.yaml"


def _resolved_v1_1_config() -> dict:
    return OmegaConf.to_container(OmegaConf.load(V1_1_CONFIG), resolve=True)


def _judge_model_config(config: dict) -> dict:
    return config["aalcr_v1_1_judge_model"]["responses_api_models"]["openai_model"]


def _runtime_judge_model_config(config: dict) -> dict:
    return {
        "name": "aalcr_v1_1_judge_model",
        "host": "127.0.0.1",
        "port": 8081,
        **_judge_model_config(config),
    }


def _judge_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0.0,
        model="openai/openai/gpt-5.6-luna",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
                content=[NeMoGymResponseOutputText(annotations=[], text='{"verdict":"CORRECT"}', type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def test_v1_1_uses_official_luna_medium_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", "not-a-secret")

    config = _resolved_v1_1_config()
    judge_model = SimpleModelServerConfig.model_validate(_runtime_judge_model_config(config))
    judge_ref = config["aalcr_benchmark_resources_server"]["resources_servers"]["aalcr"]["judge_model_server"]

    assert judge_ref == {"type": "responses_api_models", "name": "aalcr_v1_1_judge_model"}
    assert judge_model.openai_base_url == "https://inference-api.nvidia.com/v1"
    assert judge_model.openai_model == "openai/openai/gpt-5.6-luna"
    assert judge_model.extra_body == {"reasoning_effort": "medium"}


async def test_v1_1_forwards_luna_model_and_medium_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", "not-a-secret")
    judge_model = SimpleModelServerConfig.model_validate(_runtime_judge_model_config(_resolved_v1_1_config()))
    server = SimpleModelServer(
        config=judge_model,
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_response = AsyncMock(return_value=_judge_response().model_dump(mode="json"))

    await server.responses(NeMoGymResponseCreateParamsNonStreaming(input="judge input"))

    server._client.create_response.assert_awaited_once_with(
        reasoning_effort="medium",
        input="judge input",
        model="openai/openai/gpt-5.6-luna",
    )


def test_v1_1_allows_preparation_but_rejects_judge_startup_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)

    config = _resolved_v1_1_config()
    judge_model = _judge_model_config(config)
    dataset = config["aalcr_benchmark_simple_agent"]["responses_api_agents"]["simple_agent"]["datasets"][0]

    assert judge_model["openai_api_key"] is None
    assert dataset["prepare_script"] == "benchmarks/aalcr/prepare_v1_1.py"
    with pytest.raises(ValidationError, match="openai_api_key"):
        SimpleModelServerConfig.model_validate(_runtime_judge_model_config(config))


def test_v1_1_does_not_reuse_policy_model_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "policy-model-key")

    judge_model = _judge_model_config(_resolved_v1_1_config())

    assert judge_model["openai_base_url"] == "https://inference-api.nvidia.com/v1"
    assert judge_model["openai_api_key"] is None
