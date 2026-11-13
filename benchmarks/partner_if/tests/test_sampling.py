# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check replay's effective requests through the configured Model Server."""

import copy
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from benchmarks.partner_if.tests.test_agent_sessions import activation, make_agent, provider, request_for, seed
from benchmarks.partner_if.tests.test_replay import call, probe, response
from nemo_gym.client import ServerClient
from nemo_gym.config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    ModelServerRef,
    get_first_server_config_dict,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def resolved_config(path):
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {
                    "config_paths": [path],
                    "policy_api_key": "synthetic",
                    "policy_model": {
                        "responses_api_models": {
                            "vllm_model": {
                                "sampling_overrides": {
                                    "temperature": 1.0,
                                    "top_p": 0.8,
                                    "max_tokens": 262144,
                                    "max_completion_tokens": 262144,
                                }
                            }
                        }
                    },
                },
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


def model_for(config, name):
    block = get_first_server_config_dict(config, name)
    return VLLMModel(
        config=VLLMModelConfig(
            name=name,
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            model="test",
            base_url="http://model.example/v1",
            api_key="synthetic",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            uses_interleaved_reasoning=False,
            sampling_overrides=OmegaConf.to_container(block.get("sampling_overrides"), resolve=True)
            if block.get("sampling_overrides") is not None
            else None,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["benchmarks/partner_if/config.yaml", "benchmarks/partner_if/config_ngc_replay.yaml"])
async def test_parent_and_explore_backend_sampling_isolated_from_cohort_overrides(path, tmp_path, monkeypatch):
    config = resolved_config(path)
    agent = make_agent(tmp_path)
    reference = get_first_server_config_dict(config, "partner_if_ngc_replay_agent")["model_server"]
    agent.config.model_server = ModelServerRef.model_validate(OmegaConf.to_container(reference))
    backend = model_for(config, reference["name"])
    seen = []
    provider(
        monkeypatch,
        [
            response(call("Agent", {"prompt": "Read skill", "subagent_type": "Explore"}, "parent")),
            response(call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"}, "child")),
            response(content="Read the skill."),
            response(probe()),
        ],
        seen,
    )
    state = await agent._seed_agent_session_state(seed())
    await agent._execute_step(request_for(), state, activation(2))
    assert len(seen) == 4
    assert any(":delegate-" in item[2]["headers"]["x-session-id"] for item in seen)
    for _, _, request in seen:
        outbound = backend._preprocess_chat_completion_create_params(MagicMock(), copy.deepcopy(request["json"]))
        assert (outbound["temperature"], outbound["top_p"], outbound["max_completion_tokens"]) == (0.7, 0.95, 8192)
        assert "max_tokens" not in outbound
    if path.endswith("config.yaml"):
        text = get_first_server_config_dict(config, "partner_if_policy_model")["sampling_overrides"]
        assert text["temperature"] == 1.0 and text["top_p"] == 0.8
        prime = model_for(config, "partner_if_prime_model")
        outbound = prime._preprocess_chat_completion_create_params(
            MagicMock(),
            {
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.2,
                "max_tokens": 12345,
            },
        )
        assert outbound["temperature"] == 0.2 and outbound["max_tokens"] == 12345
