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

"""Environment migration: agent environment langgraph."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.rollouts.rollout_collection import _environment_servers_by_agent
from tests.helpers.agent_environment_migration import (
    SCRIPT,
    _parse,
    _server_fronting,
    load_migration,
)


migration = load_migration()


@pytest.mark.parametrize("flavor", ["reflection", "parallel_thinking", "orchestrator", "rewoo"])
def test_langgraph_flavor_composes_with_one_session_environment(tmp_path: Path, flavor: str) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / f"responses_api_agents/langgraph_agent/configs/{flavor}_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting(f"{flavor}_agent", name="session_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {f"{flavor}_agent": ["session_environment"]}
    agent = resolved[f"{flavor}_agent"].responses_api_agents.langgraph_agent
    assert agent.resources_server is None


@pytest.mark.parametrize("flavor", ["reflection", "parallel_thinking", "orchestrator", "rewoo"])
@pytest.mark.parametrize("pairing", ["math", "reasoning_gym", "resources_reasoning_gym"])
def test_langgraph_pairings_use_sessions_without_grants(tmp_path: Path, flavor: str, pairing: str) -> None:
    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig

    root = SCRIPT.parents[1]
    path = (
        root / f"resources_servers/reasoning_gym/configs/{flavor}_agent.yaml"
        if pairing == "resources_reasoning_gym"
        else root / f"environments/langgraph_{flavor}_{pairing}/config.yaml"
    )
    model = tmp_path / "model.yaml"
    model.write_text("policy_model_name: test-model\n")
    resolved = _parse(path, model, strict=True)
    fronting = _environment_servers_by_agent(resolved)
    assert len(fronting) == 1
    agent_name, environments = next(iter(fronting.items()))
    assert len(environments) == 1
    environment = resolved[environments[0]].environment_servers
    assert list(environment) == ["single_agent_turn_legacy"]
    server = environment.single_agent_turn_legacy

    SingleAgentTurnEnvironmentServerConfig.model_validate(
        {**OmegaConf.to_container(server, resolve=True), "name": environments[0]}
    )
    assert server.cleanup_timeout_seconds == 180
    assert server.resources_tool_transports == []
    assert (
        server.resources_server.name == resolved[agent_name].responses_api_agents.langgraph_agent.resources_server.name
    )
