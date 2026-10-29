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
"""The environment server migration script, run on configs outside the repository."""

import importlib.util
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.config.global_config import environment_server_agent_refs, legacy_environment_server_name
from nemo_gym.rollouts.rollout_collection import _environment_servers_by_agent
from nemo_gym.server_runtime.ray_support import DictConfig


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "add_legacy_agent_environment_servers.py"
_spec = importlib.util.spec_from_file_location("add_legacy_agent_environment_servers", SCRIPT)
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)

AGENT_CONFIG = """\
my_resources:
  resources_servers:
    mcqa:
      entrypoint: app.py
      domain: other
my_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources
"""


def _environment_servers(document: dict) -> dict[str, str]:
    """Map each agent to the environment servers that name it."""
    fronting: dict[str, list[str]] = {}
    for name, instance in document.items():
        for server in (instance.get("environment_servers") or {}).values() if isinstance(instance, dict) else ():
            fronting.setdefault(server["agent_server"]["name"], []).append(name)
    return fronting


def test_migrates_a_config_outside_the_repository(tmp_path: Path) -> None:
    config = tmp_path / "my_run.yaml"
    config.write_text(AGENT_CONFIG)

    assert migration.main([str(tmp_path)]) == 0

    document = yaml.safe_load(config.read_text())
    assert _environment_servers(document) == {"my_simple_agent": ["my_environment_server"]}
    assert config.read_text().startswith(AGENT_CONFIG)  # existing content, comments included, is untouched


@pytest.mark.parametrize("resources", [None, {"type": "resources_servers", "name": "my_resources"}])
@pytest.mark.parametrize(
    "agent_type",
    [
        "hermes_agent",
        "pi_agent",
        "codex_agent",
        "openclaw_agent",
        "opencode_agent",
        "kilocode_agent",
        "claude_code_agent",
        "osworld_agent",
        "harbor_oracle_agent",
        "terminus_2_sandboxed_agent",
        "miniswe_sandboxed_agent",
    ],
)
def test_migration_respects_agent_session_templates(
    tmp_path: Path, resources: dict[str, str] | None, agent_type: str
) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "my_agent": {
                    "responses_api_agents": {agent_type: {"entrypoint": "app.py", "resources_server": resources}}
                }
            }
        )
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    if agent_type != "osworld_agent" and resources is None:
        assert config.read_text() == before
    else:
        assert _environment_servers(yaml.safe_load(config.read_text())) == {"my_agent": ["my_environment_server"]}


@pytest.mark.parametrize(
    "agent_type",
    [
        "hermes_agent",
        "pi_agent",
        "codex_agent",
        "openclaw_agent",
        "opencode_agent",
        "kilocode_agent",
        "claude_code_agent",
        "harbor_oracle_agent",
        "terminus_2_sandboxed_agent",
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


def test_default_pi_composes_with_exactly_one_session_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/pi_agent/configs/pi_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("pi_agent", name="session_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"pi_agent": ["session_environment"]}
    assert resolved.pi_agent.responses_api_agents.pi_agent.resources_server is None


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


def test_check_reports_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    config = tmp_path / "my_run.yaml"
    config.write_text(AGENT_CONFIG)

    assert migration.main(["--check", str(config)]) == 1

    assert config.read_text() == AGENT_CONFIG
    assert f"would add 1 to {config}" in capsys.readouterr().out


def test_leaves_an_agent_that_any_environment_server_fronts(tmp_path: Path) -> None:
    # A second server in front of the agent would make agent-routed rows ambiguous.
    config = tmp_path / "my_run.yaml"
    config.write_text(
        AGENT_CONFIG
        + """\
my_episodes:
  environment_servers:
    single_agent_turn_legacy:
      agent_server:
        type: responses_api_agents
        name: my_simple_agent
"""
    )
    before = config.read_text()

    assert migration.main([str(config)]) == 0

    assert config.read_text() == before


MULTI_AGENT_CONFIG = (
    AGENT_CONFIG
    + """\
my_user_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources
my_conversation:
  environment_servers:
    conversation:
      entrypoint: app.py
      user_agent:
        type: responses_api_agents
        name: my_user_agent
      assistant_agent:
        type: responses_api_agents
        name: my_simple_agent
"""
)


def _server_fronting(agent: str, *, name: str, server_type: str) -> str:
    return f"""\
{name}:
  environment_servers:
    {server_type}:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: {agent}
"""


def _parse(*configs: Path, strict: bool) -> DictConfig:
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                *(OmegaConf.load(config) for config in configs),
                {"error_on_agent_without_environment_server": strict},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


def test_leaves_an_agent_fronted_by_another_config(tmp_path: Path) -> None:
    agent = tmp_path / "agent.yaml"
    agent.write_text(AGENT_CONFIG)
    server = tmp_path / "server.yaml"
    server_text = _server_fronting("my_simple_agent", name="custom_relay", server_type="legacy_agent")
    server.write_text(server_text)

    assert migration.main([str(agent), str(server)]) == 0

    assert agent.read_text() == AGENT_CONFIG
    assert server.read_text() == server_text
    assert _environment_servers_by_agent(_parse(agent, server, strict=True)) == {"my_simple_agent": ["custom_relay"]}


def test_leaves_a_rename_of_a_repository_agent(tmp_path: Path) -> None:
    # The server keeps its name, so routes that name it still resolve.
    config = tmp_path / "my_run.yaml"
    config.write_text("renamed_agent:\n  _inherit_from: workplace_assistant_simple_agent\n")
    before = config.read_text()

    assert migration.main([str(config)]) == 0

    assert config.read_text() == before
    base = SCRIPT.parents[1] / "resources_servers/workplace_assistant/configs/workplace_assistant.yaml"
    resolved = _parse(base, config, strict=True)
    assert _environment_servers_by_agent(resolved) == {"renamed_agent": ["workplace_assistant_environment_server"]}


@pytest.mark.parametrize(
    ("source_server", "expected_server", "expected_type"),
    [
        # The script declares the source's server; Gym points it at the renamed agent.
        ("", "my_environment_server", "legacy_agent"),
        # The source's server has a name the script would not generate.
        (_server_fronting("my_simple_agent", name="my_relay", server_type="legacy_agent"), "my_relay", "legacy_agent"),
        # The source's server is not a legacy_agent relay.
        (
            _server_fronting("my_simple_agent", name="my_environment_server", server_type="single_agent_turn_legacy"),
            "my_environment_server",
            "single_agent_turn_legacy",
        ),
    ],
    ids=["generated-source-server", "custom-named-source-server", "non-legacy-source-server"],
)
def test_migrated_rename_parses_with_one_server(
    tmp_path: Path, source_server: str, expected_server: str, expected_type: str
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + source_server)
    overlay = tmp_path / "rename.yaml"
    overlay.write_text("renamed_agent:\n  _inherit_from: my_simple_agent\n")
    before = overlay.read_text()

    assert migration.main([str(base), str(overlay)]) == 0

    assert overlay.read_text() == before
    # A generated relay would hide a missing server, so require the migrated one.
    resolved = _parse(base, overlay, strict=True)
    assert _environment_servers_by_agent(resolved) == {"renamed_agent": [expected_server]}
    assert list(resolved[expected_server]["environment_servers"]) == [expected_type]


def test_leaves_the_agents_a_multi_agent_server_references(tmp_path: Path) -> None:
    config = tmp_path / "my_run.yaml"
    config.write_text(MULTI_AGENT_CONFIG)

    assert migration.main([str(config)]) == 0

    assert config.read_text() == MULTI_AGENT_CONFIG


def test_migrated_renames_point_a_multi_agent_server_at_the_new_names(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(MULTI_AGENT_CONFIG)
    overlay = tmp_path / "rename.yaml"
    overlay.write_text(
        "renamed_user:\n  _inherit_from: my_user_agent\nrenamed_assistant:\n  _inherit_from: my_simple_agent\n"
    )
    before = overlay.read_text()

    assert migration.main([str(base), str(overlay)]) == 0

    assert overlay.read_text() == before
    # Parsing fails on a reference to a retired agent, so this checks both fields followed the rename.
    resolved = _parse(base, overlay, strict=True)
    assert _environment_servers_by_agent(resolved) == {
        "renamed_user": ["my_conversation"],
        "renamed_assistant": ["my_conversation"],
    }


@pytest.mark.parametrize("check", [False, True])
def test_leaves_a_rename_beside_its_legacy_wrapper(tmp_path: Path, check: bool) -> None:
    config = tmp_path / "my_run.yaml"
    config.write_text(
        AGENT_CONFIG
        + _server_fronting("my_simple_agent", name="my_relay", server_type="legacy_agent")
        + "renamed_agent:\n  _inherit_from: my_simple_agent\n"
    )
    before = config.read_text()

    assert migration.main((["--check"] if check else []) + [str(config)]) == 0

    assert config.read_text() == before
    resolved = _parse(config, strict=True)
    assert _environment_servers_by_agent(resolved) == {"renamed_agent": ["my_relay"]}


@pytest.mark.parametrize(
    ("source_server", "migrated_overlay", "expected_server", "expected_type"),
    [
        (
            "\n" + _server_fronting("my_simple_agent", name="my_environment_server", server_type="legacy_agent"),
            """\
renamed_agent:
  _inherit_from: my_simple_agent

renamed_environment_server:
  _inherit_from: my_environment_server
  environment_servers:
    legacy_agent:
      agent_server:
        name: renamed_agent
""",
            "renamed_environment_server",
            "legacy_agent",
        ),
        (
            _server_fronting("my_simple_agent", name="my_relay", server_type="legacy_agent"),
            """\
renamed_agent:
  _inherit_from: my_simple_agent

my_relay:
  environment_servers:
    legacy_agent:
      agent_server:
        type: responses_api_agents
        name: renamed_agent
""",
            "my_relay",
            "legacy_agent",
        ),
    ],
    ids=["generated-source-server", "custom-named-source-server"],
)
def test_prior_migration_output_keeps_its_server(
    tmp_path: Path, source_server: str, migrated_overlay: str, expected_server: str, expected_type: str
) -> None:
    # Frozen outputs from #4053's script, captured before changing it in this PR.
    # Do not regenerate them using the current migration implementation.
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + source_server)
    overlay = tmp_path / "rename.yaml"
    overlay.write_text(migrated_overlay)

    resolved = _parse(base, overlay, strict=True)

    assert _environment_servers_by_agent(resolved) == {"renamed_agent": [expected_server]}
    assert list(resolved[expected_server].environment_servers) == [expected_type]
    assert resolved[expected_server].environment_servers[expected_type].entrypoint == "app.py"
    assert resolved.renamed_agent.responses_api_agents.simple_agent.resources_server.name == "my_resources"
    assert migration.main(["--check", str(base), str(overlay)]) == 0
    assert base.read_text() == AGENT_CONFIG + source_server
    assert overlay.read_text() == migrated_overlay


def test_reports_an_unreadable_config(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "broken.yaml").write_text("key: [unclosed\n")

    assert migration.main([str(tmp_path)]) == 2

    assert "broken.yaml" in capsys.readouterr().err


def test_finds_the_agent_references_gym_finds() -> None:
    # The script depends only on PyYAML, so it keeps its own copy of the rule.
    server = {
        "entrypoint": "app.py",
        "agent_server": {"name": "untyped_agent_server"},
        "user_agent": {"type": "responses_api_agents", "name": "typed_field"},
        "helper": {"name": "untyped_other_field"},
        "resources_server": {"type": "resources_servers", "name": "not_an_agent"},
    }

    expected = [reference["name"] for reference in environment_server_agent_refs(OmegaConf.create(server))]
    assert expected == ["untyped_agent_server", "typed_field"]
    assert [agent for _, agent in migration.agent_references(server)] == expected


@pytest.mark.parametrize(
    ("agent_name", "agent_type"),
    [
        ("workplace_assistant_simple_agent", "simple_agent"),
        ("gpqa_mcqa_hermes_agent", "hermes_agent"),
        ("single_step_tool_use_with_argument_comparison_swe", "tool_simulation_agent"),
        ("simple_agent", "simple_agent"),
    ],
)
def test_server_names_match_the_relays_gym_generates(agent_name: str, agent_type: str) -> None:
    # The deprecation warning prints a block to paste; it must match what this script writes.
    assert migration.server_name(agent_name, agent_type) == legacy_environment_server_name(agent_name, agent_type)


def test_repository_configs_declare_every_environment_server(capsys: pytest.CaptureFixture[str]) -> None:
    """Every agent in the repository's configs has an environment server, so no run needs a generated relay."""
    assert migration.main(["--check"]) == 0, capsys.readouterr().out


RELAY = _server_fronting("my_simple_agent", name="my_environment_server", server_type="legacy_agent")
FOLLOWING = """\
# A comment that belongs to the next instance.
my_other_resources:
  resources_servers:
    mcqa:
      entrypoint: app.py
      domain: other
"""


def _switched_server(document: dict, name: str) -> dict:
    servers = document[name]["environment_servers"]
    assert list(servers) == ["single_agent_turn_legacy"]
    return servers["single_agent_turn_legacy"]


def test_switch_replaces_a_simple_agent_relay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig

    config = tmp_path / "my_run.yaml"
    config.write_text(AGENT_CONFIG + RELAY + FOLLOWING)

    assert migration.main(["--switch-simple-agent", str(config)]) == 0

    text = config.read_text()
    assert text.startswith(AGENT_CONFIG) and text.endswith(FOLLOWING)
    server = _switched_server(yaml.safe_load(text), "my_environment_server")
    # Without its own resources_server, the server binds the agent's, as the relay did.
    assert "resources_server" not in server
    assert server["agent_server"] == {"type": "responses_api_agents", "name": "my_simple_agent"}
    # The agent's /run called the Resources Server's tool routes; the grant gives the session the same routes.
    assert server["resources_tool_transports"] == ["direct_http"]
    SingleAgentTurnEnvironmentServerConfig.model_validate(
        {**server, "name": "my_environment_server", "host": "h", "port": 1}
    )
    resolved = _parse(config, strict=True)
    assert _environment_servers_by_agent(resolved) == {"my_simple_agent": ["my_environment_server"]}
    assert "relays switched: 1, staying on the relay: 0" in capsys.readouterr().out


def test_switch_check_reports_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "my_run.yaml"
    config.write_text(AGENT_CONFIG + RELAY)

    assert migration.main(["--switch-simple-agent", "--check", str(config)]) == 1

    assert config.read_text() == AGENT_CONFIG + RELAY
    assert "would switch 1 in" in capsys.readouterr().out


def test_switch_leaves_relays_in_front_of_other_agents(tmp_path: Path) -> None:
    config = tmp_path / "my_run.yaml"
    other_agent = AGENT_CONFIG.replace("simple_agent:\n", "tool_simulation_agent:\n")
    config.write_text(other_agent + RELAY)

    assert migration.main(["--switch-simple-agent", str(config)]) == 0

    assert config.read_text() == other_agent + RELAY


@pytest.mark.parametrize(
    ("agent_fields", "reason"),
    [
        (
            {"tool_accesses": [{"kind": "mcp", "name": "search", "required": True}]},
            "required MCP tool access: search",
        ),
        (
            {"tool_accesses": [{"kind": "direct_http", "name": "search.direct_http", "required": True}]},
            "more than one direct HTTP tool access: my_resources.direct_http, search.direct_http",
        ),
        ({"resources_server": None}, "the agent binds no resources server"),
    ],
    ids=["mcp", "two-direct-http", "unbound"],
)
def test_switch_keeps_a_relay_that_simple_agent_sessions_would_reject(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], agent_fields: dict, reason: str
) -> None:
    document = yaml.safe_load(AGENT_CONFIG)
    document["my_simple_agent"]["responses_api_agents"]["simple_agent"].update(agent_fields)
    config = tmp_path / "my_run.yaml"
    config.write_text(yaml.safe_dump(document) + RELAY)
    before = config.read_text()

    assert migration.main(["--switch-simple-agent", str(config)]) == 0

    assert config.read_text() == before
    assert f"stays on the relay: my_environment_server in {config}: {reason}" in capsys.readouterr().out


def test_switch_keeps_the_relays_of_a_run_that_cannot_load_an_include(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "my_run.yaml"
    config.write_text(f"config_paths:\n  - {tmp_path / 'missing.yaml'}\n" + AGENT_CONFIG + RELAY)
    before = config.read_text()

    assert migration.main(["--switch-simple-agent", str(config)]) == 0

    assert config.read_text() == before
    assert "cannot load included config" in capsys.readouterr().out


CHILD = """\
config_paths:
  - {base}
child_resources:
  _inherit_from: my_resources
child_agent:
  _inherit_from: my_simple_agent
  responses_api_agents:
    simple_agent:
      resources_server:
        name: child_resources
child_environment_server:
  _inherit_from: my_environment_server
  environment_servers:
    legacy_agent:
      agent_server:
        name: child_agent
"""


def _parent_and_child(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + RELAY)
    child = tmp_path / "child.yaml"
    child.write_text(CHILD.format(base=base))
    return base, child


def test_switch_keeps_a_child_relay_inheriting_its_switched_parent(tmp_path: Path) -> None:
    base, child = _parent_and_child(tmp_path)

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 0

    document = yaml.safe_load(child.read_text())
    # Still inheriting, the parent stays a template that composition consumes.
    assert document["child_environment_server"]["_inherit_from"] == "my_environment_server"
    server = _switched_server(document, "child_environment_server")
    assert server == {"agent_server": {"name": "child_agent"}}
    resolved = _parse(child, strict=True)
    assert _environment_servers_by_agent(resolved) == {"child_agent": ["child_environment_server"]}
    merged = resolved["child_environment_server"]["environment_servers"]
    assert list(merged) == ["single_agent_turn_legacy"]
    assert merged["single_agent_turn_legacy"]["entrypoint"] == "app.py"
    assert resolved["child_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]["name"] == (
        "child_resources"
    )


def test_a_switched_relay_follows_an_agent_rename(tmp_path: Path) -> None:
    # A benchmark that only renames the agent and its resources server, and declares no relay of its own,
    # reaches the source config's server through Gym's agent rename; that server must name nothing renamed away.
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + RELAY)
    assert migration.main(["--switch-simple-agent", str(base)]) == 0
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        """\
renamed_resources:
  _inherit_from: my_resources
renamed_agent:
  _inherit_from: my_simple_agent
  responses_api_agents:
    simple_agent:
      resources_server:
        name: renamed_resources
"""
    )

    resolved = _parse(base, overlay, strict=True)

    assert _environment_servers_by_agent(resolved) == {"renamed_agent": ["my_environment_server"]}
    assert list(resolved["my_environment_server"]["environment_servers"]) == ["single_agent_turn_legacy"]


def test_switch_refuses_a_child_relay_without_its_parent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base, child = _parent_and_child(tmp_path)
    before = child.read_text()

    assert migration.main(["--switch-simple-agent", str(child)]) == 2

    assert child.read_text() == before
    assert "child_environment_server" in capsys.readouterr().err


def test_switch_refuses_a_parent_relay_without_its_children(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    base, child = _parent_and_child(tmp_path)
    before = base.read_text()
    # The child lives among the configs the script scans for relays that inherit a switched one.
    monkeypatch.setattr(migration, "repo_config_files", lambda: [child])

    assert migration.main(["--switch-simple-agent", str(base)]) == 2

    assert base.read_text() == before
    assert "child_environment_server" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("seed_session", "switches"),
    [
        (None, True),
        ("async def seed_session(self, body: ResourcesSeedSessionRequest | BaseSeedSessionRequest): ...", True),
        # Reads the raw request and answers a typed seed with its resources_session_id.
        ("async def seed_session(self, request: Request):\n        return request['resources_session_id']", True),
        ("async def seed_session(self, request: Request, body: BaseSeedSessionRequest): ...", False),
    ],
    ids=["base", "typed", "raw-typed", "legacy"],
)
def test_switch_keeps_a_relay_whose_resources_server_cannot_take_a_typed_seed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    seed_session: str | None,
    switches: bool,
) -> None:
    server = tmp_path / "servers" / "mcqa"
    server.mkdir(parents=True)
    method = f"\n    {seed_session}\n" if seed_session else "\n    pass\n"
    server.joinpath("app.py").write_text(f"class Server:{method}")
    monkeypatch.setattr(migration, "RESOURCES_SERVERS_ROOT", tmp_path / "servers")
    config = tmp_path / "my_run.yaml"
    config.write_text(AGENT_CONFIG + RELAY)

    assert migration.main(["--switch-simple-agent", str(config)]) == 0

    servers = yaml.safe_load(config.read_text())["my_environment_server"]["environment_servers"]
    assert list(servers) == (["single_agent_turn_legacy"] if switches else ["legacy_agent"])
    if not switches:
        assert "resources server mcqa seeds only the legacy body" in capsys.readouterr().out


HARNESS_OVERRIDE = """\
config_paths:
  - {base}
my_harness_agent:
  responses_api_agents:
    tool_simulation_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources
my_simple_agent: null
my_environment_server:
  environment_servers:
    legacy_agent:
      agent_server:
        name: my_harness_agent
"""


def test_switch_keeps_a_relay_that_another_config_merges_with_a_legacy_agent_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A harness variant points the same server at an agent without sessions, so it must stay a relay.
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + RELAY)
    variant = tmp_path / "variant.yaml"
    variant.write_text(HARNESS_OVERRIDE.format(base=base))
    monkeypatch.setattr(migration, "repo_config_files", lambda: [variant])

    assert migration.main(["--switch-simple-agent", str(base)]) == 0

    assert base.read_text() == AGENT_CONFIG + RELAY
    resolved = _parse(variant, strict=True)
    assert list(resolved["my_environment_server"]["environment_servers"]) == ["legacy_agent"]
    assert f"stays on the relay: my_environment_server in {base}: {variant} merges it with a legacy_agent server" in (
        capsys.readouterr().out
    )


def test_switch_keeps_every_declaration_of_a_relay_another_config_merges_with_a_legacy_agent_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A middle config overrides the relay under its own name, so two files declare the one instance.
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + RELAY)
    middle = tmp_path / "middle.yaml"
    middle.write_text(
        f"config_paths:\n  - {base}\n"
        "my_environment_server:\n  environment_servers:\n    legacy_agent:\n      agent_server:\n"
        "        name: my_simple_agent\n"
    )
    variant = tmp_path / "variant.yaml"
    variant.write_text(HARNESS_OVERRIDE.format(base=middle))
    monkeypatch.setattr(migration, "repo_config_files", lambda: [variant])
    before = (base.read_text(), middle.read_text())

    assert migration.main(["--switch-simple-agent", str(base), str(middle)]) == 0

    assert (base.read_text(), middle.read_text()) == before
    out = capsys.readouterr().out
    assert f"stays on the relay: my_environment_server in {base}: {variant} merges it" in out
    assert f"stays on the relay: my_environment_server in {middle}: {variant} merges it" in out
    assert list(_parse(variant, strict=True)["my_environment_server"]["environment_servers"]) == ["legacy_agent"]


def test_switch_refuses_a_child_relay_another_config_merges_with_a_legacy_agent_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The child cannot stay a relay while its parent switches, so the variant has to declare its own server.
    base, child = _parent_and_child(tmp_path)
    variant = tmp_path / "variant.yaml"
    variant.write_text(
        HARNESS_OVERRIDE.format(base=child)
        .replace("my_simple_agent: null", "child_agent: null")
        .replace("my_environment_server:", "child_environment_server:")
    )
    monkeypatch.setattr(migration, "repo_config_files", lambda: [variant])
    before = (base.read_text(), child.read_text())

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 2

    assert (base.read_text(), child.read_text()) == before
    error = capsys.readouterr().err
    assert "child_environment_server in a run of" in error and "my_environment_server, which switches" in error


def test_a_harness_with_its_own_relay_lets_the_benchmark_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The variant removes the benchmark's servers and fronts its harness with a relay of its own.
    base, child = _parent_and_child(tmp_path)
    variant = tmp_path / "variant.yaml"
    harness = (
        HARNESS_OVERRIDE.format(base=child)
        .split("my_simple_agent: null\n")[0]
        .replace("name: my_resources", "name: child_resources")
    )
    variant.write_text(
        harness
        + "my_simple_agent: null\nchild_agent: null\nmy_environment_server: null\nchild_environment_server: null\n"
        + _server_fronting("my_harness_agent", name="my_harness_environment_server", server_type="legacy_agent")
    )
    monkeypatch.setattr(migration, "repo_config_files", lambda: [variant])

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 0

    benchmark = _parse(child, strict=True)
    assert list(benchmark["child_environment_server"]["environment_servers"]) == ["single_agent_turn_legacy"]
    assert _environment_servers_by_agent(_parse(variant, strict=True)) == {
        "my_harness_agent": ["my_harness_environment_server"]
    }


def test_a_harness_can_override_a_relay_without_changing_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, child = _parent_and_child(tmp_path)
    variant = tmp_path / "variant.yaml"
    harness = (
        HARNESS_OVERRIDE.format(base=child)
        .split("my_simple_agent: null\n")[0]
        .replace("name: my_resources", "name: child_resources")
    )
    relay = _server_fronting("my_harness_agent", name="child_environment_server", server_type="legacy_agent")
    relay = relay.replace("  environment_servers:\n", "  _delete_key: _inherit_from\n  environment_servers:\n")
    relay = relay.replace("    legacy_agent:\n", "    _delete_key: single_agent_turn_legacy\n    legacy_agent:\n")
    variant.write_text(harness + "my_simple_agent: null\nchild_agent: null\nmy_environment_server: null\n" + relay)
    monkeypatch.setattr(migration, "repo_config_files", lambda: [variant])
    before = variant.read_text()
    routing = {"my_harness_agent": ["child_environment_server"]}
    assert _environment_servers_by_agent(_parse(variant, strict=True)) == routing

    assert migration.main(["--switch-simple-agent", str(base), str(child), str(variant)]) == 0

    assert variant.read_text() == before
    benchmark = _parse(child, strict=True)
    assert list(benchmark["child_environment_server"]["environment_servers"]) == ["single_agent_turn_legacy"]
    parsed = _parse(variant, strict=True)
    assert _environment_servers_by_agent(parsed) == routing
    assert list(parsed["child_environment_server"]["environment_servers"]) == ["legacy_agent"]


@pytest.mark.parametrize("present_before_inheritance", [False, True])
def test_resolved_instances_matches_gym_deletion_order(present_before_inheritance: bool) -> None:
    document = {
        "base": {"setting": "inherited", "retained": True},
        "child": {"_inherit_from": "base", "_delete_key": "setting"},
    }
    if present_before_inheritance:
        document["child"]["setting"] = "overridden"
    runtime = OmegaConf.create(document)
    GlobalConfigDictParser()._recursively_swap_keys(runtime)

    expected = {"retained": True}
    if present_before_inheritance:
        # Gym removes the local setting first, so inheritance supplies it again.
        expected["setting"] = "inherited"
    assert OmegaConf.to_container(runtime.child) == expected
    assert migration.resolved_instances([document])["child"] == expected


def test_resolved_instances_deletes_after_all_overlays_merge() -> None:
    documents = [
        {"instance": {"_delete_key": "obsolete", "retained": True}},
        {"instance": {"obsolete": {"setting": "later overlay"}}},
    ]
    runtime = OmegaConf.merge(*documents)
    GlobalConfigDictParser()._recursively_swap_keys(runtime)

    assert OmegaConf.to_container(runtime.instance) == {"retained": True}
    assert migration.resolved_instances(documents)["instance"] == {"retained": True}


@pytest.mark.parametrize(
    ("benchmark", "recipe", "resources", "dataset"),
    [
        ("hle", "pi", "equivalence_llm_judge", "hle"),
        ("hle", "opencode", "equivalence_llm_judge", "hle"),
        ("hle", "pi_search", "equivalence_llm_judge", "hle"),
        ("hle", "opencode_search", "equivalence_llm_judge", "hle"),
        ("hle_diamond", "pi", "equivalence_llm_judge", "hle_diamond_text"),
        ("hle_diamond", "opencode", "equivalence_llm_judge", "hle_diamond_text"),
        ("hle_diamond", "pi_search", "equivalence_llm_judge", "hle_diamond_text"),
        ("hle_diamond", "opencode_search", "equivalence_llm_judge", "hle_diamond_text"),
        ("apex_shortlist", "pi", "math_with_judge", "apex_shortlist"),
        ("apex_shortlist", "opencode", "math_with_judge", "apex_shortlist"),
    ],
)
def test_harness_recipes_preserve_routing_and_bindings_across_the_switch(
    benchmark: str, recipe: str, resources: str, dataset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable in ("PI_SANDBOX_IMAGE", "OPENCODE_SANDBOX_IMAGE", "OPENSANDBOX_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.setenv(variable, "test-only")
    parser = GlobalConfigDictParser()
    _, documents = parser.load_extra_config_paths([f"benchmarks/{benchmark}/{recipe}.yaml"])
    agent_name = f"{benchmark}_{recipe.removesuffix('_search')}_agent"
    server_name = f"{benchmark}_{resources}_environment_server"
    resources_name = f"{benchmark}_{resources}_resources_server"
    resolved = []
    for switched in (False, True):
        if switched:
            for document in documents:
                for name, instance in list(document.items_ex(resolve=False)):
                    if not isinstance(instance, DictConfig):
                        continue
                    relay = (instance.get("environment_servers") or {}).get("legacy_agent")
                    agent = (relay.get("agent_server") or {}).get("name", "") if relay else ""
                    if agent.endswith("_simple_agent"):
                        document[name] = OmegaConf.create(
                            migration.stanza(name, agent, instance.get("_inherit_from"))
                        )[name]
        parsed = parser.parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.merge(
                    GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                    *documents,
                    {"config_paths": [], "error_on_agent_without_environment_server": True},
                ),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
        assert _environment_servers_by_agent(parsed) == {agent_name: [server_name]}
        assert list(parsed[server_name].environment_servers) == ["legacy_agent"]
        agent = next(iter(parsed[agent_name].responses_api_agents.values()))
        assert agent.resources_server.name == resources_name
        assert agent.datasets[0].name == dataset
        if recipe.endswith("_search"):
            assert agent.tool_servers == [{"type": "resources_servers", "name": "tavily"}]
        participants = {}
        for server in parser.filter_for_server_instance_configs(parsed):
            config = OmegaConf.to_container(server.get_inner_run_server_config_dict(), resolve=True)
            participants[server.name] = {key: value for key, value in config.items() if key not in {"host", "port"}}
        resolved.append(participants)
    assert resolved[0] == resolved[1]


def test_switch_keeps_a_relay_inheriting_one_that_stays(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The child has no environment_servers of its own, so only the merged run shows it is a relay.
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + RELAY)
    child = tmp_path / "child.yaml"
    child.write_text(f"config_paths:\n  - {base}\nchild_environment_server:\n  _inherit_from: my_environment_server\n")
    before = child.read_text()

    assert migration.main(["--switch-simple-agent", str(child)]) == 0

    assert child.read_text() == before
    assert "stays on the relay: child_environment_server" in capsys.readouterr().out


def test_switch_keeps_a_relay_with_settings_the_switched_server_would_drop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "my_run.yaml"
    config.write_text(
        AGENT_CONFIG + RELAY.replace("      entrypoint: app.py\n", "      entrypoint: app.py\n      port: 8000\n")
    )
    before = config.read_text()

    assert migration.main(["--switch-simple-agent", str(config)]) == 0

    assert config.read_text() == before
    assert "the relay sets legacy_agent.port, which the switched server would not keep" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (
            "class Verify(BaseModel):\n    rollout_index: int = Field(alias=ROLLOUT_INDEX_KEY_NAME)\n",
            "resources server mcqa reads the row's rollout coordinates, which are not forwarded",
        ),
        (None, "cannot find resources server mcqa under"),
    ],
    ids=["rollout-coordinates", "missing-implementation"],
)
def test_switch_keeps_a_relay_whose_resources_server_it_cannot_drive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    source: str | None,
    reason: str,
) -> None:
    if source is not None:
        server = tmp_path / "servers" / "mcqa"
        server.mkdir(parents=True)
        server.joinpath("app.py").write_text(source)
    monkeypatch.setattr(migration, "RESOURCES_SERVERS_ROOT", tmp_path / "servers")
    config = tmp_path / "my_run.yaml"
    config.write_text(AGENT_CONFIG + RELAY)

    assert migration.main(["--switch-simple-agent", str(config)]) == 0

    assert config.read_text() == AGENT_CONFIG + RELAY
    assert reason in capsys.readouterr().out


def test_switch_writes_nothing_when_a_relay_cannot_be_rewritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "a_first.yaml"
    first.write_text(AGENT_CONFIG + RELAY)
    # A quoted key parses to the same name, but its stanza cannot be found by name.
    second = tmp_path / "b_second.yaml"
    second.write_text(
        AGENT_CONFIG.replace("my_", "other_")
        + RELAY.replace("my_", "other_").replace("other_environment_server:", '"other_environment_server":')
    )
    before = (first.read_text(), second.read_text())

    assert migration.main(["--switch-simple-agent", str(first), str(second)]) == 2

    assert (first.read_text(), second.read_text()) == before
    assert "cannot find the line that starts 'other_environment_server'" in capsys.readouterr().err
