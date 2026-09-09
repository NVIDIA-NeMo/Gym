# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import Request
from pytest import mark

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.deepsearchqa.app import DeepSearchQAAgent, DeepSearchQAConfig


@mark.parametrize("task_index", [0, 1])
async def test_smoke_rollout_runs_harness_through_sandbox_api(monkeypatch, task_index: int) -> None:
    lines = (Path(__file__).parents[1] / "data/example.jsonl").read_text().splitlines()
    task = json.loads(lines[task_index])
    monkeypatch.setenv("MOCK_EXA_SEARCH_RESULT", task["answer"])
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy_model": {"responses_api_models": {"model": {}}}}
    client._build_server_base_url.return_value = "http://model"
    config = DeepSearchQAConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        name="test",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        harness_module="responses_api_agents.deepsearchqa.tests.fake_agent",
        harness_class="FakeAgent",
        harness_config_class="FakeAgentConfig",
        image="unused-by-local-provider",
        sandbox_provider={"local": {}},
        sandbox_spec={"env": {"MOCK_EXA_SEARCH_RESULT": task["answer"]}},
        exa_api_key="temporary-test-key",
    )
    agent = DeepSearchQAAgent(config=config, server_client=client)
    result = await agent.responses(
        Request({"type": "http", "path": "/v1/responses", "path_params": {}, "headers": []}),
        NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content=task["problem"])], model="model"
        ),
    )
    assert result.output[0].content[0].text == task["answer"]
    assert result.model_dump()["object"] == "response"
    assert "temporary-test-key" not in config.model_dump_json()
