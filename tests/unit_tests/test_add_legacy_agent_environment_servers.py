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

from nemo_gym.global_config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    environment_server_agent_refs,
    legacy_environment_server_name,
)
from nemo_gym.rollout_collection import _environment_servers_by_agent
from nemo_gym.server_utils import DictConfig


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
    "agent_type", ["hermes_agent", "pi_agent", "codex_agent", "openclaw_agent", "opencode_agent", "osworld_agent"]
)
def test_migration_respects_native_session_templates(
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


@pytest.mark.parametrize("agent_type", ["hermes_agent", "pi_agent", "codex_agent", "openclaw_agent", "opencode_agent"])
def test_native_capable_overlay_keeps_inherited_resources(tmp_path: Path, agent_type: str) -> None:
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


def test_default_pi_composes_with_exactly_one_native_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/pi_agent/configs/pi_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("pi_agent", name="native_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"pi_agent": ["native_environment"]}
    assert resolved.pi_agent.responses_api_agents.pi_agent.resources_server is None


def test_default_codex_composes_with_exactly_one_native_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/codex_agent/configs/codex_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("codex_agent", name="native_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"codex_agent": ["native_environment"]}
    assert resolved.codex_agent.responses_api_agents.codex_agent.resources_server is None


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


def test_native_openclaw_template_composes_with_one_environment(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("openclaw_agent", name="native_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"openclaw_agent": ["native_environment"]}
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


def test_default_hermes_composes_with_exactly_one_native_environment(tmp_path: Path) -> None:
    config = tmp_path / "hermes.yaml"
    default = SCRIPT.parents[1] / "responses_api_agents/hermes_agent/configs/hermes_agent.yaml"
    config.write_text(default.read_text())
    composition = tmp_path / "run.yaml"
    composition.write_text(
        "policy_model_name: test-model\n"
        + _server_fronting("hermes_agent", name="native_environment", server_type="single_agent_turn_legacy")
    )
    before = config.read_text()

    # Migrating the standalone harness must not install a second, legacy route.
    assert migration.main([str(config)]) == 0

    assert config.read_text() == before
    resolved = _parse(config, composition, strict=True)
    assert _environment_servers_by_agent(resolved) == {"hermes_agent": ["native_environment"]}
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
