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


def test_gzip_disabled_on_both_sides_is_allowed(config):
    GlobalConfigDictParser().raise_on_gzip_mismatch(config)


def test_gzip_enabled_on_both_sides_is_allowed(config):
    config.policy.responses_api_models.openai_model.gzip_requests_enabled = True
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = True

    GlobalConfigDictParser().raise_on_gzip_mismatch(config)


def test_compressing_harness_requires_model_decoding(config):
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = True

    with pytest.raises(ConfigError) as error:
        GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=config,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )

    assert "hermes.responses_api_agents.hermes_agent.gzip_requests_enabled=true" in str(error.value)
    assert "model 'policy' has gzip_requests_enabled=false" in str(error.value)


def test_decoding_model_requires_a_connected_compressing_harness(config):
    config.policy.responses_api_models.openai_model.gzip_requests_enabled = True

    with pytest.raises(ConfigError, match="Model 'policy'.*no connected harness enables gzip"):
        GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=config,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )


def test_uncompressed_harness_can_share_a_gzip_model(config):
    config.policy.responses_api_models.openai_model.gzip_requests_enabled = True
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = True
    config.pi = {
        "responses_api_agents": {
            "pi_agent": {
                "entrypoint": "app.py",
                "model_server": {"type": "responses_api_models", "name": "policy"},
                "gzip_requests_enabled": False,
            }
        }
    }

    GlobalConfigDictParser().raise_on_gzip_mismatch(config)


def test_compressing_harness_on_another_model_does_not_satisfy_the_guard(config):
    config.policy.responses_api_models.openai_model.gzip_requests_enabled = True
    config.other_policy = {
        "responses_api_models": {"openai_model": {"entrypoint": "app.py", "gzip_requests_enabled": True}}
    }
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = True
    config.hermes.responses_api_agents.hermes_agent.model_server.name = "other_policy"

    with pytest.raises(ConfigError, match="Model 'policy'.*no connected harness enables gzip"):
        GlobalConfigDictParser().raise_on_gzip_mismatch(config)


def test_compressing_harness_cannot_use_an_unverifiable_external_receiver(config):
    config.hermes.responses_api_agents.hermes_agent.gzip_requests_enabled = True
    config.hermes.responses_api_agents.hermes_agent.model_server = None

    with pytest.raises(ConfigError, match="hermes.*requires a model_server reference"):
        GlobalConfigDictParser().raise_on_gzip_mismatch(config)
