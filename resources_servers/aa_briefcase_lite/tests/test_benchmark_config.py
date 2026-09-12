# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.benchmarks import _benchmark_config_paths
from nemo_gym.cli.main import _asset_config_path
from nemo_gym.global_config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
    resolve_dataset_agent,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "benchmarks/aa_briefcase_lite/config.yaml"
AGENT_NAME = "aa_briefcase_lite_stirrup_agent"
RESOURCES_NAME = "aa_briefcase_lite_resources_server"
AGENT_KEY = f"{AGENT_NAME}.responses_api_agents.stirrup_agent"
RESOURCES_KEY = f"{RESOURCES_NAME}.resources_servers.aa_briefcase_lite"


@pytest.fixture
def resolve_config(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
    monkeypatch.setenv("AA_BRIEFCASE_LITE_DATASET_DIR", str(tmp_path / "dataset"))
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.invalid/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "dummy")
    # Scoring and execution modes must come from Gym config, even in an old shell.
    monkeypatch.setenv("AA_BRIEFCASE_REWARD_MODE", "binary")
    monkeypatch.setenv("EXECUTE_ONLY", "true")
    monkeypatch.setenv("JUDGE_ONLY", "true")

    def resolve(*overrides):
        initial = OmegaConf.merge(
            GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
            {"config_paths": [_asset_config_path("benchmark", "aa_briefcase_lite")]},
            OmegaConf.from_dotlist(overrides),
        )
        return GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=initial,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )

    return resolve


def test_benchmark_discovery_resolves_full_local_profile(resolve_config) -> None:
    assert _benchmark_config_paths(CONFIG_PATH.parent) == [CONFIG_PATH]
    config = resolve_config()
    agent = get_first_server_config_dict(config, AGENT_NAME)
    resources = get_first_server_config_dict(config, RESOURCES_NAME)

    assert resolve_dataset_agent(config, AGENT_NAME) == AGENT_NAME
    assert agent.datasets[0].name == "aa_briefcase_lite"
    assert agent.datasets[0].type == "benchmark"
    assert agent.datasets[0].prepare_script == "benchmarks/aa_briefcase_lite/prepare.py"
    assert agent.datasets[0].num_repeats == 1
    assert agent.task == "aa_briefcase_lite"
    assert agent.agent_max_turns == 500
    assert agent.execute_only is False
    assert agent.judge_only is False
    assert resources.reward_mode == "all"
    assert resources.pairwise_reference_ids == ["gpt-5-5"]
    assert resources.pairwise_num_trials == 2
    assert resources.verified is False
    assert [judge.name for judge in resources.judge_panel] == ["gpt-5.5", "gemini-3.1-pro", "claude-opus-4.8"]
    for judge in resources.judge_panel:
        adapter = get_first_server_config_dict(config, judge.model_server.name)
        assert adapter.openai_model == judge.model


@pytest.mark.parametrize("reward_mode", ["binary", "pairwise", "all"])
def test_judge_only_overrides_keep_reference_profile(resolve_config, reward_mode) -> None:
    config = resolve_config(
        f"{RESOURCES_KEY}.reward_mode={reward_mode}",
        f"{AGENT_KEY}.execute_only=false",
        f"{AGENT_KEY}.judge_only=true",
        f"{AGENT_KEY}.rerun_incomplete=false",
    )
    agent = get_first_server_config_dict(config, AGENT_NAME)
    resources = get_first_server_config_dict(config, RESOURCES_NAME)

    assert agent.execute_only is False
    assert agent.judge_only is True
    assert agent.rerun_incomplete is False
    assert resources.reward_mode == reward_mode
    assert resources.pairwise_reference_ids == ["gpt-5-5"]
    assert resources.pairwise_num_trials == 2


def test_generation_only_override(resolve_config) -> None:
    config = resolve_config(f"{AGENT_KEY}.execute_only=true")
    agent = get_first_server_config_dict(config, AGENT_NAME)

    assert agent.execute_only is True
    assert agent.judge_only is False
