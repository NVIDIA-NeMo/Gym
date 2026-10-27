# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-owned resources references survive configuration composition."""

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config_types import AmbiguousResourcesRenameError, ConfigMissingValuesError
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


def _parse(config: DictConfig) -> DictConfig:
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                config,
                {"error_on_agent_without_environment_server": True},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


def _config() -> DictConfig:
    return OmegaConf.create(
        """
resources:
  resources_servers:
    mcqa:
      entrypoint: app.py
      domain: other
agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server: null
environment:
  environment_servers:
    single_agent_turn:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: agent
      resources_server:
        type: resources_servers
        name: resources
"""
    )


@pytest.mark.parametrize("inherit", [{"_inherit_from": "resources"}, "${inherit_from:resources}"])
def test_resources_rename_updates_environment_without_agent_binding(inherit: object) -> None:
    config = _config()
    config["renamed_resources"] = inherit

    resolved = _parse(config)

    assert "resources" not in resolved
    assert resolved.environment.environment_servers.single_agent_turn.resources_server.name == "renamed_resources"
    assert resolved.agent.responses_api_agents.simple_agent.resources_server is None


def test_copied_resources_keep_original_binding_unless_environment_overrides_it() -> None:
    config = _config()
    config["copied_resources"] = {"_copy": "resources"}
    config.agent.responses_api_agents.simple_agent.resources_server = {
        "type": "resources_servers",
        "name": "copied_resources",
    }

    resolved = _parse(config)
    assert "resources" in resolved and "copied_resources" in resolved
    assert resolved.environment.environment_servers.single_agent_turn.resources_server.name == "resources"

    config.environment.environment_servers.single_agent_turn.resources_server.name = "copied_resources"
    resolved = _parse(config)
    assert resolved.environment.environment_servers.single_agent_turn.resources_server.name == "copied_resources"


def test_ambiguous_resources_rename_requires_an_environment_choice() -> None:
    config = _config()
    config["first_resources"] = {"_inherit_from": "resources"}
    config["second_resources"] = {"_inherit_from": "resources"}
    # Even a bound agent must not decide which resources the environment uses.
    config.agent.responses_api_agents.simple_agent.resources_server = {
        "type": "resources_servers",
        "name": "second_resources",
    }

    with pytest.raises(AmbiguousResourcesRenameError) as error:
        _parse(config)

    message = str(error.value)
    assert "first_resources" in message and "second_resources" in message
    assert "environment.environment_servers.single_agent_turn.resources_server.name" in message

    config.environment.environment_servers.single_agent_turn.resources_server.name = "first_resources"
    resolved = _parse(config)
    assert resolved.environment.environment_servers.single_agent_turn.resources_server.name == "first_resources"
    assert resolved.agent.responses_api_agents.simple_agent.resources_server.name == "second_resources"


@pytest.mark.parametrize("field", ["type", "name"])
def test_resources_rename_preserves_missing_reference_diagnostic(field: str) -> None:
    config = _config()
    config["renamed_resources"] = {"_inherit_from": "resources"}
    config.environment.environment_servers.single_agent_turn.resources_server[field] = "???"

    with pytest.raises(ConfigMissingValuesError, match="resources_server"):
        _parse(config)


def test_resources_rename_ignores_non_resources_destination() -> None:
    config = _config()
    config["renamed_resources"] = {"_inherit_from": "resources"}
    config["unrelated"] = {"_inherit_from": "resources", "resources_servers": None}

    resolved = _parse(config)

    assert resolved.environment.environment_servers.single_agent_turn.resources_server.name == "renamed_resources"


def test_indic_math_environment_follows_its_resources_rename() -> None:
    resolved = _parse(OmegaConf.create({"config_paths": ["benchmarks/indic/math_500/config.yaml"]}))
    environment = resolved.math_with_judge_environment_server.environment_servers.single_agent_turn_legacy

    assert "math_with_judge" not in resolved
    assert environment.agent_server.name == "indic_math_500_math_with_judge_simple_agent"
    assert environment.resources_server.name == "indic_math_500_math_with_judge_resources_server"
    resources = resolved[environment.resources_server.name].resources_servers.math_with_judge
    assert resources.format_tolerant_answer_extraction is True


def test_indic_ifeval_explicitly_binds_its_copy_while_english_resources_survive() -> None:
    resolved = _parse(OmegaConf.create({"config_paths": ["benchmarks/indic/ifeval/config.yaml"]}))
    environment = (
        resolved.indic_ifeval_instruction_following_environment_server.environment_servers.single_agent_turn_legacy
    )

    assert "instruction_following" in resolved
    assert environment.resources_server.name == "indic_ifeval_instruction_following"


@pytest.mark.parametrize("benchmarks", [("math-500", "aime24"), ("aime24", "math-500")])
def test_math_benchmark_environments_bind_their_own_resources(benchmarks: tuple[str, str]) -> None:
    resolved = _parse(OmegaConf.create({"config_paths": [f"benchmarks/{name}/config.yaml" for name in benchmarks]}))
    bindings = {
        name: instance.environment_servers.single_agent_turn_legacy.resources_server.name
        for name, instance in resolved.items()
        if isinstance(instance, DictConfig) and "environment_servers" in instance
    }

    assert bindings == {
        "math_500_math_with_judge_environment_server": "math_500_math_with_judge_resources_server",
        "aime24_math_with_judge_environment_server": "aime24_math_with_judge_resources_server",
    }
