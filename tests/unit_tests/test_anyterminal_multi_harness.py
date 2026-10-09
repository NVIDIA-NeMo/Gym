# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig, OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionHelper


PROFILE = "responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness.yaml"
ENROOT_PROFILE = "responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness_enroot.yaml"
OPENSANDBOX_PROFILE = "responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness_opensandbox.yaml"
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
    openclaw_config = config.anyterminal_openclaw.responses_api_agents.anyterminal_agent.agent_kwargs.openclaw_config
    openclaw_defaults = openclaw_config.agents.defaults
    assert openclaw_defaults.workspace == "."
    assert openclaw_defaults.skipBootstrap is True
    assert openclaw_defaults.contextInjection == "never"
    assert openclaw_defaults.startupContext.enabled is False
    assert list(openclaw_defaults.skills) == []
    assert openclaw_defaults.contextLimits.toolResultMaxChars == 4000
    assert openclaw_defaults.compaction.reserveTokens == 4096
    assert openclaw_defaults.compaction.reserveTokensFloor == 4096
    assert openclaw_defaults.compaction.keepRecentTokens == 4096
    assert openclaw_defaults.compaction.memoryFlush.enabled is False
    assert list(openclaw_defaults.compaction.postCompactionSections) == []
    assert openclaw_config.skills.limits.maxSkillsInPrompt == 0
    assert openclaw_config.skills.limits.maxSkillsPromptChars == 0
    openclaw_tools = openclaw_config.tools
    assert openclaw_tools.profile == "minimal"
    assert list(openclaw_tools.alsoAllow) == ["exec"]
    assert list(openclaw_tools.deny) == ["session_status"]
    pi_config = config.anyterminal_pi.responses_api_agents.anyterminal_agent.agent_kwargs
    assert pi_config.context_window == 15872
    assert pi_config.max_output_tokens == 4096
    assert pi_config.output_token_policy == "remaining_context"
    assert pi_config.auto_compaction is True
    assert pi_config.compaction_reserve_tokens == 4096
    assert pi_config.compaction_keep_recent_tokens == 4096

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


def test_opensandbox_profile_selects_kubernetes_provider_for_every_harness(monkeypatch) -> None:
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-key")
    config = _resolved_profile(OPENSANDBOX_PROFILE)

    assert config.sandbox.opensandbox.connection.api_key == "test-key"
    assert config.sandbox.opensandbox.connection.domain
    assert all(
        config[name].responses_api_agents.anyterminal_agent.sandbox_provider == "sandbox" for name in [*POOL, SOURCE]
    )
