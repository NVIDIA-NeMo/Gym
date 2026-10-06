# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig, OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionHelper


PROFILE = "responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness.yaml"
ENROOT_PROFILE = "responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness_enroot.yaml"
SOURCE = "anyterminal_multi_harness"
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

    assert config.get("agent_pool") is None
    assert config.get("fan_out") is None
    assert {name: config[name].responses_api_agents.anyterminal_agent.agent_server_class for name in POOL} == {
        "anyterminal_opencode": "OpenCodeAgent",
        "anyterminal_openclaw": "OpenClawAgent",
        "anyterminal_pi": "PiAgent",
        "anyterminal_hermes": "HermesAgent",
    }
    assert all(config[name].responses_api_agents.anyterminal_agent.agent_kwargs.model == "model" for name in POOL)
    assert list(config.anyterminal_hermes.responses_api_agents.anyterminal_agent.agent_request_sampling_fields) == [
        "temperature"
    ]

    dataset_owners = []
    for instance_name, block in config.items():
        if not isinstance(block, DictConfig):
            continue
        for implementation in (block.get("responses_api_agents") or {}).values():
            if implementation.get("datasets"):
                dataset_owners.append(instance_name)

    assert dataset_owners == [SOURCE]


def test_generic_source_fans_each_task_out_to_every_p0_harness() -> None:
    config = _resolved_profile()
    fan_out = {SOURCE: POOL}
    rows = [{"task_source": SOURCE, "responses_create_params": {"input": "terminal task"}}]

    RolloutCollectionHelper._validate_agent_pool_destinations(fan_out, config)
    expanded = RolloutCollectionHelper().preprocess_examples(rows, fan_out=fan_out)

    assert [row["agent_ref"]["name"] for row in expanded] == POOL
    assert all(row["task_source"] == SOURCE for row in expanded)


def test_enroot_profile_selects_named_provider_for_every_harness() -> None:
    config = _resolved_profile(ENROOT_PROFILE)

    assert config.sandbox.enroot
    assert all(
        config[name].responses_api_agents.anyterminal_agent.sandbox_provider == "sandbox" for name in [*POOL, SOURCE]
    )
