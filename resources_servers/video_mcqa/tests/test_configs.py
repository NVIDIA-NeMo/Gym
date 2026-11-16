# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config_types import DatasetConfig
from resources_servers.video_mcqa.app import VideoMCQAResourcesServerConfig
from resources_servers.video_mcqa.tests.test_app import make_server
from resources_servers.videomme_v2.app import VideoMMEV2ResourcesServerConfig
from responses_api_agents.simple_agent.app import SimpleAgentConfig


@pytest.mark.parametrize("server_name", ["videomme_v1", "lvbench", "videomme_v2", "video_mcqa"])
def test_actual_composed_single_turn_configs_resolve_and_validate(tmp_path, server_name) -> None:
    server, _, _ = make_server(tmp_path, benchmark="lvbench" if server_name == "lvbench" else "videomme_v1")
    path = Path(__file__).parents[2] / server_name / "configs" / f"{server_name}_single_turn.yaml"
    values = {f"{server_name}_cohort_manifest": str(server.config.cohort_manifest)}
    config = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(path), OmegaConf.create(values)), resolve=True)
    config_type = VideoMMEV2ResourcesServerConfig if server_name == "videomme_v2" else VideoMCQAResourcesServerConfig
    resource = config_type(
        name=server_name, host="127.0.0.1", port=8001, **config[server_name]["resources_servers"][server_name]
    )
    assert resource.cohort_manifest == server.config.cohort_manifest
    agent_name = f"{server_name}_single_turn"
    (component,) = config[agent_name]["responses_api_agents"]
    assert component == "simple_agent"
    agent_config = config[agent_name]["responses_api_agents"]["simple_agent"]
    agent = SimpleAgentConfig(name=agent_name, host="127.0.0.1", port=8002, **agent_config)
    assert agent.resources_server.name == server_name and agent.max_steps == 1
    (example,) = [DatasetConfig.model_validate(item) for item in agent_config["datasets"]]
    assert (example.type, example.license) == ("example", "Apache 2.0")
    assert example.jsonl_fpath == f"resources_servers/{server_name}/data/example.jsonl"
    (environment_type,) = config[agent_name + "_environment"]["environment_servers"]
    assert environment_type == "single_agent_turn_legacy"
    environment = SingleAgentTurnLegacyEnvironmentServerConfig(
        name=agent_name + "_environment",
        host="127.0.0.1",
        port=8003,
        **config[agent_name + "_environment"]["environment_servers"]["single_agent_turn_legacy"],
    )
    assert environment.agent_server.name == agent_name
    assert environment.resources_server.name == server_name
