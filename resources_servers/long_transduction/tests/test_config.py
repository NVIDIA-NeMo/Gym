# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for collector routing through the environment adapter."""

from pathlib import Path

from omegaconf import OmegaConf
from pytest import MonkeyPatch

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import _environment_server_for_agent, _environment_servers_by_agent


def test_benchmark_agent_has_environment_server(monkeypatch: MonkeyPatch) -> None:
    gym_root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(gym_root)
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "benchmarks/long_transduction/config.yaml",
                        "responses_api_models/vllm_model/configs/vllm_model.yaml",
                    ],
                    "policy_base_url": "http://127.0.0.1:8000/v1",
                    "policy_api_key": "test",
                    "policy_model_name": "test",
                }
            ),
        )
    )
    environment = _environment_server_for_agent("long_transduction_agent", _environment_servers_by_agent(config))
    assert environment == "long_transduction_benchmark_environment_server"
    assert config[environment].environment_servers.legacy_agent.agent_server.name == "long_transduction_agent"
