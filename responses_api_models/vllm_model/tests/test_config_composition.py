# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import DictConfig, OmegaConf

from nemo_gym import PARENT_DIR
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def test_benchmark_extra_body_survives_model_type_config_and_reaches_vllm(tmp_path: Path) -> None:
    # `gym eval run --benchmark <b> --model-type vllm_model` merges the vllm_model config after the benchmark
    # config; the per-request budget and template kwargs a benchmark sets must still reach the engine.
    benchmark_settings = {
        "extra_body": {"thinking_token_budget": 128},
        "chat_template_kwargs": {"enable_thinking": True},
        "sampling_overrides": {"min_tokens": 5},
    }
    benchmark_config = tmp_path / "config.yaml"
    benchmark_config.write_text(
        json.dumps({"policy_model": {"responses_api_models": {"vllm_model": benchmark_settings}}})
    )
    model_type_config = PARENT_DIR / "responses_api_models/vllm_model/configs/vllm_model.yaml"
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=DictConfig(
                {
                    "config_paths": [str(benchmark_config), str(model_type_config)],
                    "policy_base_url": "http://api.openai.com/v1",
                    "policy_api_key": "dummy_key",  # pragma: allowlist secret
                    "policy_model_name": "dummy_model",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    block = OmegaConf.to_container(resolved["policy_model"]["responses_api_models"]["vllm_model"], resolve=True)
    config = VLLMModelConfig.model_validate({"name": "policy_model", **block})  # the parser assigns host/port
    server = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    app = server.setup_webserver()

    mock_chat_completion = NeMoGymChatCompletion(
        id="chtcmpl",
        object="chat.completion",
        created=0,
        model="dummy_model",
        choices=[
            NeMoGymChoice(
                index=0,
                finish_reason="stop",
                message=NeMoGymChatCompletionMessage(role="assistant", content="response", tool_calls=[]),
            )
        ],
    )
    captured_kwargs = {}

    async def mock_create_chat_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_chat_completion.model_dump()

    mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
    mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
    server._clients = [mock_client]

    request = NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(type="message", role="user", content="hello")]
    )
    response = TestClient(app).post("/v1/responses", json=request.model_dump(exclude_unset=True, mode="json"))

    assert response.status_code == 200
    assert captured_kwargs["thinking_token_budget"] == 128
    assert captured_kwargs["chat_template_kwargs"] == {"enable_thinking": True}

    assert captured_kwargs["min_tokens"] == 5


@pytest.mark.parametrize("filename", ["vllm_model.yaml", "vllm_model_completions.yaml"])
def test_model_config_keeps_request_settings_and_explicit_null(filename: str, tmp_path: Path) -> None:
    values = {
        "extra_body": {"min_p": 0.1},
        "chat_template_kwargs": {"enable_thinking": True},
        "sampling_overrides": {"min_tokens": 5},
    }
    override = tmp_path / "request.yaml"
    override.write_text(json.dumps({"policy_model": {"responses_api_models": {"vllm_model": values}}}))
    clear = tmp_path / "clear.yaml"
    clear.write_text(json.dumps({"policy_model": {"responses_api_models": {"vllm_model": {"extra_body": None}}}}))
    model_path = PARENT_DIR / "responses_api_models/vllm_model/configs" / filename
    for clear_explicitly in [False, True]:
        paths = [str(override), str(model_path)] + ([str(clear)] if clear_explicitly else [])
        resolved = GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=DictConfig(
                    {
                        "config_paths": paths,
                        "policy_base_url": "https://example.test/v1",
                        "policy_api_key": "dummy",
                        "policy_model_name": "example-model",
                    }
                ),  # pragma: allowlist secret
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
        block = resolved["policy_model"]["responses_api_models"]["vllm_model"]
        for field, value in values.items():
            assert block[field] == (None if field == "extra_body" and clear_explicitly else value)
