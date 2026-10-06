# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig, OmegaConf

from nemo_gym.global_config import AGENT_POOL_INDEX_KEY_NAME, GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionHelper


PROFILE = "responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness.yaml"
ENROOT_PROFILE = "responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness_enroot.yaml"
SOURCE = "anyterminal_hermes"
POOL = ["anyterminal_opencode", "anyterminal_openclaw", "anyterminal_pi", "anyterminal_hermes"]


def _resolved_profile(profile: str = PROFILE) -> DictConfig:
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [profile]},
    )
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


def test_profile_composes_all_four_terminal_harnesses_with_one_dataset_owner() -> None:
    config = _resolved_profile()

    assert list(config.agent_pool[SOURCE]) == POOL
    assert {name: config[name].responses_api_agents.anyterminal_agent.agent_server_class for name in POOL} == {
        "anyterminal_opencode": "OpenCodeAgent",
        "anyterminal_openclaw": "OpenClawAgent",
        "anyterminal_pi": "PiAgent",
        "anyterminal_hermes": "HermesAgent",
    }

    dataset_owners = []
    for instance_name, block in config.items():
        if not isinstance(block, DictConfig):
            continue
        for implementation in (block.get("responses_api_agents") or {}).values():
            if implementation.get("datasets"):
                dataset_owners.append(instance_name)

    assert dataset_owners == [SOURCE]


def test_profile_routes_each_source_task_to_one_p0_harness() -> None:
    config = _resolved_profile()
    agent_pool = OmegaConf.to_container(config.agent_pool, resolve=True)
    rows = [
        {
            "task_source": SOURCE,
            AGENT_POOL_INDEX_KEY_NAME: task_index,
            "responses_create_params": {"input": f"terminal task {task_index // 2}"},
        }
        for task_index in (0, 0, 1, 1, 2, 2, 3, 3)
    ]

    RolloutCollectionHelper._validate_agent_pool_destinations(agent_pool, config)
    RolloutCollectionHelper._apply_agent_pool(rows, agent_pool)

    assert [row["agent_ref"]["name"] for row in rows] == [agent for agent in POOL for _ in range(2)]
    assert all(row["_ng_agent_pool_assignment"] == row["agent_ref"]["name"] for row in rows)


def test_enroot_profile_selects_named_provider_for_every_harness() -> None:
    config = _resolved_profile(ENROOT_PROFILE)

    assert config.sandbox.enroot
    assert all(config[name].responses_api_agents.anyterminal_agent.sandbox_provider == "sandbox" for name in POOL)
