# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


@pytest.fixture
def config():
    return OmegaConf.create(
        {
            "policy": {"responses_api_models": {"openai_model": {"entrypoint": "app.py"}}},
            "hermes": {
                "responses_api_agents": {
                    "hermes_agent": {
                        "entrypoint": "app.py",
                        "model_server": {"type": "responses_api_models", "name": "policy"},
                    }
                }
            },
        }
    )


@pytest.mark.parametrize("enabled", [True, "true", False, "false"])
def test_agent_gzip_flag_accepts_native_and_quoted_booleans(config, enabled):
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = enabled

    GlobalConfigDictParser().raise_on_invalid_gzip_configuration(config)


def test_quoted_false_sender_does_not_require_a_model_server(config):
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = "false"
    config.hermes.responses_api_agents.hermes_agent.model_server = None

    GlobalConfigDictParser().raise_on_invalid_gzip_configuration(config)


def test_invalid_gzip_flag_reports_its_configuration_path(config):
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = "invalid"

    with pytest.raises(
        ConfigError, match=r"hermes\.responses_api_agents\.hermes_agent\.gzip_requests_enabled.*boolean"
    ):
        GlobalConfigDictParser().raise_on_invalid_gzip_configuration(config)


@pytest.mark.parametrize(("agent_type", "enabled"), [("claude_code_agent", True), ("pi_sandboxed_agent", False)])
def test_unsupported_agents_reject_the_gzip_flag(config, agent_type, enabled):
    config.hermes.responses_api_agents = {
        agent_type: {
            "entrypoint": "app.py",
            "model_server": {"type": "responses_api_models", "name": "policy"},
            "gzip_requests_enabled": enabled,
        }
    }

    with pytest.raises(ConfigError, match=rf"{agent_type}\.gzip_requests_enabled.*not supported"):
        GlobalConfigDictParser().raise_on_invalid_gzip_configuration(config)


def test_compressing_agent_allows_model_free_preparation(config):
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = True
    config.hermes.responses_api_agents.hermes_agent.model_server.name = "policy_model"
    preparation_config = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"hermes": config.hermes},
    )

    GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=preparation_config,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


def test_uncompressed_agent_can_share_a_model_server_with_a_compressing_agent(config):
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = True
    config.pi = {
        "responses_api_agents": {
            "pi_agent": {
                "entrypoint": "app.py",
                "model_server": {"type": "responses_api_models", "name": "policy"},
            }
        }
    }

    GlobalConfigDictParser().raise_on_invalid_gzip_configuration(config)


def test_compressing_agent_requires_a_model_server_reference(config):
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = True
    config.hermes.responses_api_agents.hermes_agent.model_server = None

    with pytest.raises(ConfigError, match="hermes.*requires a model_server reference"):
        GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=config,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
