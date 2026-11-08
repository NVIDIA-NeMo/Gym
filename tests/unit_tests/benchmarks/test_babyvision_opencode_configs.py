# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BabyVision's configs start opencode_agent in a sandbox session behind an Environment Server, with the manifest's rows."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


ROOT = Path(__file__).resolve().parents[3]


def resolve(config_path: str) -> tuple[GlobalConfigDictParser, dict]:
    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [str(ROOT / config_path)]},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return parser, config


@pytest.mark.parametrize("benchmark", ["babyvision_opencode", "babyvision_opencode_question_only"])
def test_benchmark_config_routes_the_rows_through_an_agent_session(benchmark):
    parser, config = resolve(f"benchmarks/{benchmark}/config.yaml")
    started = {instance.name for instance in parser.filter_for_server_instance_configs(config)}
    assert started == {
        "policy_model",
        "visual_coding_opencode",
        "visual_coding_opencode_judge_model",
        "visual_coding_opencode_agent",
        "visual_coding_opencode_environment_server",
    }

    environment = config["visual_coding_opencode_environment_server"]["environment_servers"]
    pairing = environment["single_agent_turn_legacy"]
    assert pairing["resources_server"]["name"] == "visual_coding_opencode"
    assert pairing["agent_server"]["name"] == "visual_coding_opencode_agent"

    agent = config["visual_coding_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    assert agent["sandbox_provider"] in config
    assert agent["network_access"] == "model_only"
    assert agent["opencode_config"]["compaction"]["auto"] is False
    assert "agent" not in agent["opencode_config"]  # no step limit, so token capture can join every call

    # The benchmark's rows replace the example rows, and match its manifest.
    manifest = yaml.safe_load((ROOT / "benchmarks" / benchmark / "manifest.yaml").read_text())
    assert manifest["agent_server"] == "opencode_agent"
    assert [OmegaConf.to_container(dataset) for dataset in agent["datasets"]] == manifest["datasets"]
