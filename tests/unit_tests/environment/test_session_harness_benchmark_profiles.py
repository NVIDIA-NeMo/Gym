# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The session harnesses' `configs/profiles/benchmark.yaml`, composed onto HLE as the adapter READMEs describe.

Each adapter's own suite seeds its composed agent through HLE's Environment Server (`check_hle_seed`).
"""

import pytest
from omegaconf import OmegaConf

from nemo_gym.config.config_types import ConfigMissingValuesError
from nemo_gym.config.global_config import get_first_server_config_dict
from nemo_gym.environment.agent_registry import discover_agents
from tests.session_harness_profiles import (
    HARNESSES,
    HLE_ENVIRONMENT,
    IMAGE,
    MODEL_ONLY_PROFILES,
    REPO_ROOT,
    REQUIRED_SETTING_VALUES,
    REQUIRED_SETTINGS,
    compose_profile_on_hle,
)


@pytest.mark.parametrize("agent", HARNESSES)
def test_profile_is_not_a_standalone_agent_variant(agent: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    assert list(discover_agents()[agent].variants) == [agent]


@pytest.mark.parametrize("agent", HARNESSES)
def test_profile_composes_onto_hle_with_its_limits_and_a_loopback_head_server(
    agent: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = compose_profile_on_hle(agent, monkeypatch, cli_args=("+default_host=192.0.2.30",))
    composed = f"hle_equivalence_llm_judge_{agent}"
    assert OmegaConf.missing_keys(config) == set()
    assert list(config[composed].responses_api_agents) == [agent]
    settings = get_first_server_config_dict(config, composed)
    assert settings.resources_server.name == "hle_equivalence_llm_judge_resources_server"
    assert settings.sandbox_provider == "sandbox"
    assert settings.sandbox_config.image == IMAGE
    assert settings.sandbox_config.ready_timeout_s == 1200
    # Graded on what the model knows: the sandbox reaches only the model server, from a runtime in the image.
    assert settings.get("network_access", "inherit") == ("model_only" if agent in MODEL_ONLY_PROFILES else "inherit")
    for key in REQUIRED_SETTINGS[agent]:
        assert settings[key] == REQUIRED_SETTING_VALUES[key]
    # Hermes derives its TTL from its own timeouts; the others would get OpenSandbox's 10-minute default.
    assert settings.sandbox_config.get("ttl_s") == (None if agent == "hermes_agent" else 14400)
    if agent == "claude_code_agent":
        assert settings.claude_code_version == "2.1.108"
        assert settings.model_server.name == "policy_model"
    environment = get_first_server_config_dict(config, HLE_ENVIRONMENT)
    assert environment.agent_server.name == composed
    # HLE's judge serves no tools, so its Environment Server sets no transports and grants none.
    assert list(environment.get("resources_tool_transports", [])) == []
    # Sandboxes reach the model server at the caller's address; the head server and the agent stay on loopback.
    assert config.head_server.host == "127.0.0.1"
    assert settings.host == "127.0.0.1"
    # The agent runs the harness on the Gym host for unseeded requests, so only the model server takes the address.
    assert get_first_server_config_dict(config, "policy_model").host == "192.0.2.30"


@pytest.mark.parametrize("agent", HARNESSES)
def test_profile_without_an_image_fails_at_resolve(agent: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigMissingValuesError, match=rf"hle_equivalence_llm_judge_{agent}\..*sandbox_config\.image"):
        compose_profile_on_hle(agent, monkeypatch, image=False)


@pytest.mark.parametrize(
    ("agent", "key"), [(agent, key) for agent in sorted(REQUIRED_SETTINGS) for key in REQUIRED_SETTINGS[agent]]
)
def test_restricted_profile_without_its_required_setting_fails_at_resolve(
    agent: str, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A model_only sandbox cannot download the harness, so its runtime or binary is required before any episode.
    supplied = [
        f"++hle_equivalence_llm_judge_{agent}.responses_api_agents.{agent}.{other}={REQUIRED_SETTING_VALUES[other]}"
        for other in REQUIRED_SETTINGS[agent]
        if other != key
    ]
    with pytest.raises(ConfigMissingValuesError, match=rf"hle_equivalence_llm_judge_{agent}\..*{key}"):
        compose_profile_on_hle(agent, monkeypatch, required=False, cli_args=tuple(supplied))
