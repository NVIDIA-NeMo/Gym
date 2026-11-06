# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The R2E-Gym OpenCode session config routes its taskset to a resources server that lends its sandbox."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser
from nemo_gym.core import EpisodeId
from nemo_gym.data import materialize_task
from nemo_gym.server_runtime import SingleAgentTurnRequest
from resources_servers.r2e_gym.app import R2EGymInstanceRequest, R2EGymResourcesServerConfig


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "resources_servers/r2e_gym/configs/r2e_gym_opencode_single_agent_turn.yaml"
EXAMPLE_ROW = json.loads((ROOT / "resources_servers/r2e_gym/data/example.jsonl").read_text().splitlines()[0])


def test_config_seeds_opencode_agent_sessions_in_the_resources_sandbox() -> None:
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(CONFIG)])
    config = OmegaConf.merge(*configs, {"policy_model_name": "served-policy-model"})
    parser._recursively_swap_keys(config)

    assert OmegaConf.to_container(config.environment_server_routes) == {
        "r2e_gym": "r2e_gym_opencode_single_agent_turn"
    }
    environment = SingleAgentTurnEnvironmentServerConfig(
        name="r2e_gym_opencode_single_agent_turn",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(
            config.r2e_gym_opencode_single_agent_turn.environment_servers.single_agent_turn, resolve=True
        ),
    )
    assert environment.resources_server.name == "r2e_gym_resources_server"
    assert not environment.resources_tool_transports
    # Only the agent's resources server starts; golden-patch validation has its own config.
    assert config.r2e_gym_golden_patch_resources_server is None

    block = OmegaConf.to_container(config.r2e_gym_resources_server.resources_servers.r2e_gym, resolve=True)
    resources = R2EGymResourcesServerConfig(name="r2e_gym_resources_server", host="localhost", port=8002, **block)
    assert block["allowed_agents"] == ["opencode_agent"]
    assert resources.apply_anti_cheating is True and resources.is_verifying_golden_patch is False
    # The session lives as long as the sandbox, so it must outlive the episode deadline the agent runs under.
    assert environment.default_episode_timeout_seconds < resources.sandbox_config["ttl_s"]
    assert resources.sandbox_config["resources"]["cpu"] == 8, "the override keeps the inherited sandbox settings"
    [dataset] = block["datasets"]
    assert (dataset["taskset"], dataset["type"]) == ("r2e_gym", "train")

    agents = config[environment.agent_server.name].responses_api_agents
    assert list(agents) == ["opencode_agent"]
    assert agents.opencode_agent.resources_server is None  # the agent borrows the resources server's task sandbox
    assert agents.opencode_agent.token_id_capture is True
    assert not agents.opencode_agent.datasets

    # A row of the taskset reaches the typed seed with the task fields it validates, and no model input.
    task = materialize_task(EXAMPLE_ROW, taskset=dataset["taskset"])
    request = SingleAgentTurnRequest(episode_id=EpisodeId(rollout_id="config-test", attempt=0), task=task)
    assert "responses_create_params" not in request.task.task_input.task_data
    seeded = R2EGymInstanceRequest.model_validate(request.task.task_input.task_data)
    assert seeded.instance_id == request.task.task_id.task_id == EXAMPLE_ROW["instance_id"]
