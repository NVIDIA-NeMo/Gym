# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request

from benchmarks.indic.milu_1_5.prepare import DIRECTORY
from nemo_gym.client import ServerClient
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config
from responses_api_agents.simple_agent.app import SimpleAgentRunRequest
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


@pytest.mark.parametrize("overrides", [{}, {"temperature": 0.4, "top_p": 0.8, "max_output_tokens": 4096}])
def test_milu_prompt_and_stop_strings_reach_chat_completion_request(overrides):
    row = json.loads((DIRECTORY / "data/example.jsonl").read_text().splitlines()[0])
    prompt = load_prompt_config(str(DIRECTORY / "prompts/default.yaml"))
    rendered = apply_prompt_to_row(row, prompt)
    rendered["responses_create_params"].update(overrides)
    request = SimpleAgentRunRequest.model_validate(rendered)
    model = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="test",
            base_url="http://unused/v1",
            api_key="dummy",  # pragma: allowlist secret
            model="test",
            return_token_id_information=False,
            uses_reasoning_parser=True,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    chat = model.get_converter().responses_to_chat_completion_create_params(request.responses_create_params)
    body = model._preprocess_chat_completion_create_params(
        Request({"type": "http", "headers": []}), chat.model_dump(exclude_unset=True)
    )
    assert body["stop"] == ["</s>", "<|im_end|>", "<eos>"]
    assert body["messages"] == rendered["responses_create_params"]["input"]
    if overrides:
        assert (body["temperature"], body["top_p"], body["max_tokens"]) == (0.4, 0.8, 4096)
