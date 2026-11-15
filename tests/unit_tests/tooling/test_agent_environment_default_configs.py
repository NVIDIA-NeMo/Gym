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

"""Environment migration: agent environment default configs."""

from pathlib import Path

import pytest
import yaml

from nemo_gym.rollouts.rollout_collection import _environment_servers_by_agent
from tests.helpers.agent_environment_migration import (
    AGENT_CONFIG,
    SCRIPT,
    _environment_servers,
    _parse,
    _server_fronting,
    load_migration,
)


migration = load_migration()


@pytest.mark.parametrize("agent_type", ["pi_agent", "pool_agent", "crewai_agent", "cline_agent"])
def test_default_agent_composes_with_exactly_one_session_environment(tmp_path: Path, agent_type: str) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / f"responses_api_agents/{agent_type}/configs/{agent_type}.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting(agent_type, name="session_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {agent_type: ["session_environment"]}
    assert resolved[agent_type].responses_api_agents[agent_type].resources_server is None


def test_mini_swe_session_template_composes_with_exactly_one_session_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    template = SCRIPT.parents[1] / "responses_api_agents/mini_swe_agent/configs/mini_swe_agent_sandboxed.yaml"
    config.write_text(template.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("mini_swe_agent", name="session_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"mini_swe_agent": ["session_environment"]}
    assert resolved.mini_swe_agent.responses_api_agents.mini_swe_agent.resources_server is None


def test_mini_swe_run_config_keeps_its_one_legacy_relay(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text(
        (SCRIPT.parents[1] / "responses_api_agents/mini_swe_agent/configs/mini_swe_agent.yaml").read_text()
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    # The SWE-Gym /run config already declares its relay; listing the agent for sessions adds no second server.
    assert config.read_text() == before
    assert _environment_servers(yaml.safe_load(before)) == {
        "mini_swe_simple_agent": ["mini_swe_simple_environment_server"]
    }


def test_default_codex_composes_with_exactly_one_session_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/codex_agent/configs/codex_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("codex_agent", name="session_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"codex_agent": ["session_environment"]}
    assert resolved.codex_agent.responses_api_agents.codex_agent.resources_server is None


def test_default_kilocode_composes_with_exactly_one_session_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/kilocode_agent/configs/kilocode_sandboxed.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("kilocode_agent", name="session_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"kilocode_agent": ["session_environment"]}
    assert resolved.kilocode_agent.responses_api_agents.kilocode_agent.resources_server is None


def test_default_claude_composes_with_exactly_one_session_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/claude_code_agent/configs/claude_code_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "anthropic_api_key: test-key\n"
        + _server_fronting("claude_code_agent", name="session_environment", server_type="single_agent_turn")
    )
    before = config.read_text()

    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"claude_code_agent": ["session_environment"]}
    assert resolved.claude_code_agent.responses_api_agents.claude_code_agent.resources_server is None


def test_default_openhands_composes_with_exactly_one_session_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/openhands_agent/configs/openhands_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("openhands_agent", name="session_environment", server_type="single_agent_turn")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"openhands_agent": ["session_environment"]}
    assert resolved.openhands_agent.responses_api_agents.openhands_agent.resources_server is None


@pytest.mark.parametrize("resources", [None, {"type": "resources_servers", "name": "my_resources"}])
def test_openhands_gets_no_legacy_relay_whatever_its_resources_binding(
    tmp_path: Path, resources: dict[str, str] | None
) -> None:
    # OpenHands answers /run with 422, so a legacy relay in front of it could never run an episode.
    config = tmp_path / "agent.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "my_agent": {
                    "responses_api_agents": {
                        "openhands_agent": {"entrypoint": "app.py", "resources_server": resources}
                    }
                }
            }
        )
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before


def test_default_droid_composes_with_exactly_one_session_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/droid_agent/configs/droid_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("droid_agent", name="session_environment", server_type="single_agent_turn")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"droid_agent": ["session_environment"]}
    assert resolved.droid_agent.responses_api_agents.droid_agent.resources_server is None


def test_droid_bound_to_resources_gets_no_legacy_relay(tmp_path: Path) -> None:
    # Droid has no /run for a legacy relay to call, whatever its Resources binding.
    config = tmp_path / "agent.yaml"
    config.write_text(AGENT_CONFIG.replace("simple_agent", "droid_agent"))
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before


def test_opencode_session_template_composes_with_one_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/opencode_agent/configs/opencode_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("opencode_agent", name="session_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"opencode_agent": ["session_environment"]}
    assert resolved.opencode_agent.responses_api_agents.opencode_agent.resources_server is None


def test_deepagents_sandbox_template_composes_with_one_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = (
        SCRIPT.parents[1]
        / "responses_api_agents/langchain_deepagents_agent/configs/langchain_deepagents_sandbox_agent.yaml"
    )
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting(
            "langchain_deepagents_sandbox_agent", name="session_environment", server_type="single_agent_turn_legacy"
        )
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"langchain_deepagents_sandbox_agent": ["session_environment"]}
    agent = resolved.langchain_deepagents_sandbox_agent.responses_api_agents.langchain_deepagents_agent
    assert (agent.entrypoint, agent.resources_server, agent.tool_profile) == ("app.py", None, "builtin")


def test_session_openclaw_template_composes_with_one_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("openclaw_agent", name="session_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"openclaw_agent": ["session_environment"]}
    assert resolved.openclaw_agent.responses_api_agents.openclaw_agent.resources_server is None


def test_hermes_overlay_keeps_its_inherited_resources_binding(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG.replace("simple_agent", "hermes_agent"))
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "renamed_agent:\n"
        "  _inherit_from: my_hermes_agent\n"
        "  responses_api_agents:\n"
        "    hermes_agent:\n"
        "      max_turns: 7\n"
    )
    before = overlay.read_text()

    assert migration.main([str(base), str(overlay)]) == 0

    resolved = _parse(base, overlay, strict=True)
    assert overlay.read_text() == before
    assert _environment_servers_by_agent(resolved) == {"renamed_agent": ["my_environment_server"]}
    assert resolved.renamed_agent.responses_api_agents.hermes_agent.resources_server.name == "my_resources"


def test_default_hermes_composes_with_exactly_one_session_environment(tmp_path: Path) -> None:
    config = tmp_path / "hermes.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/hermes_agent/configs/hermes_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("hermes_agent", name="session_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()

    # Migrating the standalone harness must not install a second, legacy route.
    assert migration.main([str(config)]) == 0

    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"hermes_agent": ["session_environment"]}
    assert resolved.hermes_agent.responses_api_agents.hermes_agent.resources_server is None


@pytest.mark.parametrize(
    "agent_type",
    [
        "hermes_agent",
        "pi_agent",
        "pool_agent",
        "prime_agent",
        "simple_strands_agent",
        "codex_agent",
        "openclaw_agent",
        "opencode_agent",
        "kilocode_agent",
        "claude_code_agent",
        "cline_agent",
        "harbor_oracle_agent",
        "terminus_2_sandboxed_agent",
        "mini_swe_agent",
        "miniswe_sandboxed_agent",
    ],
)
def test_agent_session_capable_overlay_keeps_inherited_resources(tmp_path: Path, agent_type: str) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG.replace("simple_agent", agent_type))
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"renamed_agent:\n  _inherit_from: my_{agent_type}\n  responses_api_agents:\n    {agent_type}: {{}}\n"
    )
    before = overlay.read_text()
    assert migration.main([str(base), str(overlay)]) == 0
    # Gym follows the agent rename while preserving the source server's identity (#4141).
    assert overlay.read_text() == before
    assert _environment_servers(yaml.safe_load(base.read_text())) == {f"my_{agent_type}": ["my_environment_server"]}
    resolved = _parse(base, overlay, strict=True)
    assert _environment_servers_by_agent(resolved) == {"renamed_agent": ["my_environment_server"]}
    assert resolved.renamed_agent.responses_api_agents[agent_type].resources_server.name == "my_resources"


def test_default_copilot_composes_with_exactly_one_session_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/copilot_agent/configs/copilot_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("copilot_agent", name="session_environment", server_type="single_agent_turn")
    )
    before = config.read_text()

    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"copilot_agent": ["session_environment"]}
    assert resolved.copilot_agent.responses_api_agents.copilot_agent.resources_server is None
