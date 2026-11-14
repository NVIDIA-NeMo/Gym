# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


BENCHMARK_DIR = Path(__file__).resolve().parents[1]


def test_default_uses_shared_harbor_and_standalone_verifier() -> None:
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [str(BENCHMARK_DIR / "config.yaml")],
                    "policy_base_url": "http://policy/v1",
                    "policy_api_key": "TEST_PLACEHOLDER",
                    "policy_model_name": "vision-model",
                    "judge_base_url": "http://judge/v1",
                    "judge_api_key": "TEST_PLACEHOLDER",
                    "judge_model_name": "gpt-5-mini",
                    "vials_max_output_tokens": 131072,
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    from nemo_gym.cli.env import _configured_servers

    assert len(list(_configured_servers(config))) == 2
    agent = config.vials_agent.responses_api_agents.harbor_agent_general
    assert agent.harbor_agent.import_path == "benchmarks.vials.harbor.agent:VialsVisionAgent"
    assert agent.harbor_agent.kwargs.max_output_tokens == 131072
    assert agent.harbor_worker_env.VIALS_POLICY_BASE_URL == "http://policy/v1"
    assert agent.harbor_worker_env.VIALS_JUDGE_BASE_URL == "http://judge/v1"
    assert not agent.harbor_verifier.disable
    assert "vials_resources_server" not in config
    assert "policy_model" not in config
