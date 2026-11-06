# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
GYM_ROOT = BENCHMARK_DIR.parents[1]
JUDGE_CONFIG = GYM_ROOT / "resources_servers/vials/configs/judge_model.yaml"


@pytest.mark.parametrize("adapter", ["openai_model", "vllm_model"])
def test_policy_adapter_and_independent_judge(adapter: str) -> None:
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        str(BENCHMARK_DIR / "config.yaml"),
                        str(GYM_ROOT / f"responses_api_models/{adapter}/configs/{adapter}.yaml"),
                    ],
                    "policy_base_url": "https://policy.example/v1",
                    "policy_api_key": "TEST_PLACEHOLDER",
                    "policy_model_name": "nemotron-vl-3.5",
                    "judge_base_url": "https://judge.example/v1",
                    "judge_api_key": "TEST_PLACEHOLDER",
                    "judge_model_name": "openai/openai/gpt-5-mini",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert set(config.policy_model.responses_api_models) == {adapter}
    judge = config.vials_judge_model.responses_api_models.openai_model
    assert judge.openai_base_url == "https://judge.example/v1"
    assert judge.openai_api_key == "TEST_PLACEHOLDER"
    assert judge.openai_model == "openai/openai/gpt-5-mini"
    policy = config.policy_model.responses_api_models[adapter]
    assert policy["model" if adapter == "vllm_model" else "openai_model"] == "nemotron-vl-3.5"


def test_judge_default_does_not_resolve_to_its_own_server_instance() -> None:
    config = OmegaConf.load(JUDGE_CONFIG)
    assert config.vials_judge_model.responses_api_models.openai_model.openai_model == "gpt-5-mini"
    # Force recursive resolution to detect interpolation cycles, not just a truthy DictConfig.
    OmegaConf.to_container(config, resolve=True)


def test_namespaced_judge_override_takes_precedence() -> None:
    config = OmegaConf.merge(
        OmegaConf.load(JUDGE_CONFIG),
        {"judge_model_name": "generic-judge", "vials_judge_model_name": "override-judge"},
    )
    assert config.vials_judge_model.responses_api_models.openai_model.openai_model == "override-judge"
