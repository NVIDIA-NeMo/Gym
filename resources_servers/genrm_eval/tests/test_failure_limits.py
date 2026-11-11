# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All GenRM benchmark variants inherit sidecar-aware failure limits."""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = sorted(
    str(path.relative_to(ROOT))
    for path in (ROOT / "benchmarks").glob("genrm*/*.yaml")
    if path.name in {"config.yaml", "genrm.yaml", "llm_as_a_judge.yaml"}
)


def resolve(path: str, **overrides: float) -> DictConfig:
    model = {}
    if Path(path).name == "config.yaml":
        model = {
            "policy_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "entrypoint": "app.py",
                        "base_url": "https://example.com/v1",
                        "api_key": "dummy",
                        "model": "policy",
                    }
                }
            }
        }
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [path],
                    "policy_base_url": "https://example.com/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "policy",
                    **model,
                    **overrides,
                }
            ),
        )
    )


@pytest.mark.parametrize("path", CONFIGS)
def test_all_genrm_configs_have_one_percent_failure_limits(path: str) -> None:
    config = resolve(path)
    assert config.health_check_max_final_failure_fraction == 0.01
    assert config.health_check_max_not_final_failure_fraction == 0.01


def test_failure_limits_can_be_overridden() -> None:
    config = resolve(
        "benchmarks/genrm_hs4/llm_as_a_judge.yaml",
        health_check_max_final_failure_fraction=0.005,
        health_check_max_not_final_failure_fraction=0.02,
    )
    assert config.health_check_max_final_failure_fraction == 0.005
    assert config.health_check_max_not_final_failure_fraction == 0.02
