# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stock parent Responses calls and Explore Chat calls keep the recorded-context sampling contract."""

import copy
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from benchmarks.partner_if.tests.fixture_data import PARAMS
from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.gymnasium_agent.app import GymnasiumAgent, GymnasiumAgentConfig
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def resolved_config(path: str):
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {
                    "config_paths": [path],
                    "policy_model_name": "test",
                    "policy_base_url": "http://model.example/v1",
                    "policy_api_key": "synthetic",
                    "judge_model_name": "test-judge",
                    "judge_base_url": "http://judge.example/v1",
                    "judge_api_key": "synthetic",
                    "partner_if_judge": {
                        "responses_api_models": {
                            "vllm_model": {
                                "entrypoint": "app.py",
                                "base_url": "http://judge.example/v1",
                                "api_key": "synthetic",
                                "model": "test-judge",
                            }
                        }
                    },
                    "policy_model": {
                        "responses_api_models": {
                            "vllm_model": {
                                "entrypoint": "app.py",
                                "sampling_overrides": {
                                    "temperature": 1.0,
                                    "top_p": 0.8,
                                    "max_tokens": 262144,
                                    "max_completion_tokens": 262144,
                                },
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


def model_for(config, name: str) -> VLLMModel:
    block = OmegaConf.to_container(get_first_server_config_dict(config, name), resolve=True)
    return VLLMModel(
        config=VLLMModelConfig.model_validate(
            block
            | {
                "name": name,
                "host": "127.0.0.1",
                "port": 12345,
                "entrypoint": "app.py",
                "model": "test",
                "base_url": "http://model.example/v1",
                "api_key": "synthetic",
                "return_token_id_information": False,
                "uses_reasoning_parser": False,
                "uses_interleaved_reasoning": False,
            }
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.parametrize(
    "path", ["benchmarks/partner_if/config.yaml", "benchmarks/partner_if/config_recorded_context.yaml"]
)
def test_stock_parent_and_explore_sampling_are_isolated_from_the_cohort(path):
    config = resolved_config(path)
    agent_block = OmegaConf.to_container(
        get_first_server_config_dict(config, "partner_if_recorded_context_agent"), resolve=True
    )
    agent_config = GymnasiumAgentConfig.model_validate(agent_block | {"name": "partner_if_recorded_context_agent"})
    assert set(config[agent_config.name]["responses_api_agents"]) == {"gymnasium_agent"}
    assert agent_config.resources_server.name == "partner_if_recorded_context"
    assert agent_config.model_server.name == "partner_if_recorded_context_model"
    assert agent_config.model_server_transport_max_attempts == 1
    agent = GymnasiumAgent(config=agent_config, server_client=MagicMock(spec=ServerClient))
    assert agent._model_server_request_kwargs() == {"_max_num_tries": 1}

    resources = get_first_server_config_dict(config, agent_config.resources_server.name)
    assert resources["policy_model_server"]["name"] == agent_config.model_server.name
    replay = resources["replay"]
    expected = (replay["temperature"], replay["top_p"], replay["max_tokens"])
    assert expected == (0.7, 0.95, 8192)
    model = model_for(config, agent_config.model_server.name)
    assert model.config.max_num_tries == model.config.max_http_attempts == 1
    assert model.config.request_timeout_seconds == replay["timeout_s"] == 180

    original = copy.deepcopy(PARAMS)
    parent = NeMoGymResponseCreateParamsNonStreaming.model_validate(original)
    converted = model._responses_to_chat_request(parent)
    parent_outbound = model._preprocess_chat_completion_create_params(
        MagicMock(), converted.model_dump(exclude_unset=True)
    )
    child_outbound = model._preprocess_chat_completion_create_params(
        MagicMock(),
        {
            "model": "test",
            "messages": [{"role": "user", "content": "Inspect the frozen source."}],
            "temperature": 1.0,
            "top_p": 0.8,
            "max_tokens": 262144,
        },
    )
    assert PARAMS == original
    for outbound in (parent_outbound, child_outbound):
        assert (outbound["temperature"], outbound["top_p"], outbound["max_completion_tokens"]) == expected
        assert "max_tokens" not in outbound
    if path.endswith("config.yaml"):
        cohort = get_first_server_config_dict(config, "partner_if_policy_model")["sampling_overrides"]
        assert (cohort["temperature"], cohort["top_p"]) == (1.0, 0.8)
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
