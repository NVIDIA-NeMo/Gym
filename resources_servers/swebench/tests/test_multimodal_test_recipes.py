# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWE-bench Multimodal test recipes evaluate each row with its own evaluation; the dev recipes keep the
harness's."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser
from resources_servers.swebench.app import SwebenchResourcesServerConfig


BENCHMARKS = Path(__file__).parents[3] / "benchmarks/swebench"


def load(recipe: str):
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(BENCHMARKS / recipe)])
    config = OmegaConf.merge(*configs, {"policy_model_name": "served-policy-model"})
    parser._recursively_swap_keys(config)
    return config


def resources_of(config, environment_name: str) -> tuple[SwebenchResourcesServerConfig, dict]:
    environment = SingleAgentTurnEnvironmentServerConfig(
        name=environment_name,
        host="localhost",
        port=8000,
        **OmegaConf.to_container(config[environment_name].environment_servers.single_agent_turn, resolve=True),
    )
    block = OmegaConf.to_container(config[environment.resources_server.name].resources_servers.swebench, resolve=True)
    return SwebenchResourcesServerConfig(name=environment.resources_server.name, host="h", port=1, **block), block


@pytest.mark.parametrize(
    ("recipe", "environment_name", "agent"),
    [
        ("multimodal_test/opencode.yaml", "swebench_multimodal_test_opencode_single_agent_turn", "opencode_agent"),
        ("multimodal_test/hermes.yaml", "swebench_multimodal_test_hermes", "hermes_agent"),
    ],
)
def test_a_test_recipe_routes_its_own_taskset_to_a_row_evaluating_server(recipe, environment_name, agent) -> None:
    config = load(recipe)
    assert OmegaConf.to_container(config.environment_server_routes) == {"swebench:multimodal_test": environment_name}
    resources, block = resources_of(config, environment_name)
    assert resources.evaluation_spec_source == "row" and block["allowed_agents"] == [agent]
    [dataset] = block["datasets"]
    assert (dataset["name"], dataset["taskset"], dataset["prepare_script"]) == (
        "swebench_multimodal_test",
        "swebench:multimodal_test",
        "benchmarks/swebench/multimodal_test/prepare.py",
    )
    assert resources.sandbox_config["ttl_s"] == 25200


@pytest.mark.parametrize(
    ("recipe", "environment_name"),
    [
        ("multimodal/opencode.yaml", "swebench_multimodal_opencode_single_agent_turn"),
        ("multimodal/hermes.yaml", "swebench_multimodal_hermes"),
    ],
)
def test_the_dev_recipes_keep_the_harness_evaluation(recipe, environment_name) -> None:
    resources, block = resources_of(load(recipe), environment_name)
    assert resources.evaluation_spec_source == "swebench_harness"
    assert [dataset["taskset"] for dataset in block["datasets"]] == ["swebench:multimodal"]
