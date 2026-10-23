# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from resources_servers.arena_judge.app import ArenaJudgeConfig


@pytest.mark.parametrize("preset", ["config.yaml", "gemini_pointwise.yaml"])
def test_config_uses_custom_pointwise_scoring_without_changing_arena(
    monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-placeholder")
    paths = [
        "benchmarks/indic_arena_hard/config.yaml",
        "benchmarks/indic_alpaca_eval/" + preset,
        "responses_api_models/inference_provider/configs/inference_provider.yaml",
    ]
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": paths,
                    "policy_base_url": "http://localhost:8000/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "candidate",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    actual = config.indic_alpaca_eval_resources_server.resources_servers.arena_judge
    parsed = ArenaJudgeConfig.model_validate({"name": "indic_alpaca_eval_resources_server", **actual})
    assert parsed.evaluation_mode == "pointwise"
    assert not actual.verified
    assert (
        config.indic_arena_hard_arena_judge_resources_server.resources_servers.arena_judge.evaluation_mode
        == "pairwise"
    )
    agent = config.indic_alpaca_eval_simple_agent.responses_api_agents.simple_agent
    assert agent.resources_server.name == "indic_alpaca_eval_resources_server"
    assert agent.datasets[0].license == "CC-BY-NC-4.0"
    arena_environment = config.indic_arena_hard_arena_judge_environment_server.environment_servers.legacy_agent
    alpaca_environment = config.indic_alpaca_eval_environment_server.environment_servers.legacy_agent
    assert arena_environment.agent_server.name == "indic_arena_hard_arena_judge_simple_agent"
    assert alpaca_environment.agent_server.name == "indic_alpaca_eval_simple_agent"
    if preset == "gemini_pointwise.yaml":
        judge = config.arena_judge_judge_model.responses_api_models.openai_model
        assert judge.openai_base_url == "https://inference-api.nvidia.com/v1"
        assert judge.openai_model == "gcp/google/gemini-3.8-flash"
