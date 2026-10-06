# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig, OmegaConf

from nemo_gym.global_config import AGENT_POOL_INDEX_KEY_NAME, GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionHelper


PROFILE = "environments/multi_harness_reasoning_gym/config.yaml"
POOL = [
    "hermes_reasoning_gym_agent",
    "openclaw_reasoning_gym_agent",
    "opencode_reasoning_gym_agent",
    "pi_reasoning_gym_agent",
]


def _resolved_profile() -> DictConfig:
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [PROFILE]},
    )
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


def test_profile_composes_all_four_native_harnesses_with_one_dataset_owner() -> None:
    config = _resolved_profile()

    assert list(config.agent_pool.reasoning_gym) == POOL
    assert {
        name: next(iter(config[name].responses_api_agents))
        for name in config.agent_pool.reasoning_gym
    } == {
        "hermes_reasoning_gym_agent": "hermes_agent",
        "openclaw_reasoning_gym_agent": "openclaw_agent",
        "opencode_reasoning_gym_agent": "opencode_agent",
        "pi_reasoning_gym_agent": "pi_agent",
    }

    dataset_owners = []
    for instance_name, block in config.items():
        if not isinstance(block, DictConfig):
            continue
        for server_type in ("resources_servers", "responses_api_agents"):
            for implementation in (block.get(server_type) or {}).values():
                if implementation.get("datasets"):
                    dataset_owners.append(instance_name)

    assert dataset_owners == ["reasoning_gym"]


def test_profile_routes_real_harness_names_in_round_robin_groups() -> None:
    config = _resolved_profile()
    agent_pool = OmegaConf.to_container(config.agent_pool, resolve=True)
    rows = [
        {
            "task_source": "reasoning_gym",
            AGENT_POOL_INDEX_KEY_NAME: task_index,
            "responses_create_params": {"input": f"task {task_index // 2}"},
        }
        for task_index in (0, 0, 1, 1, 2, 2, 3, 3)
    ]

    RolloutCollectionHelper._validate_agent_pool_destinations(agent_pool, config)
    RolloutCollectionHelper._apply_agent_pool(rows, agent_pool)

    assert [row["agent_ref"]["name"] for row in rows] == [agent for agent in POOL for _ in range(2)]
    assert all(row["_ng_agent_pool_assignment"] == row["agent_ref"]["name"] for row in rows)
