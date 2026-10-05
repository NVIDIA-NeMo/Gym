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
    legacy_environment_server_name,
)
from nemo_gym.rollout_collection import _environment_servers_by_agent


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


def test_rename_inherits_the_renamed_agents_server(tmp_path: Path) -> None:
    # `_inherit_from` moves the source agent, so the source's server must move with it.
    config = tmp_path / "my_run.yaml"
    config.write_text("renamed_agent:\n  _inherit_from: workplace_assistant_simple_agent\n")

    assert migration.main([str(config)]) == 0

    server = yaml.safe_load(config.read_text())["renamed_environment_server"]
    assert server["_inherit_from"] == "workplace_assistant_environment_server"
    assert server["environment_servers"]["legacy_agent"]["agent_server"]["name"] == "renamed_agent"


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


@pytest.mark.parametrize(
    ("source_server", "rename", "expected_server", "expected_type"),
    [
        # The script declares the source's server, so the rename inherits the server it generates.
        ("", "renamed_agent", "renamed_environment_server", "legacy_agent"),
        # The source's server has a name the script would not generate.
        (
            _server_fronting("my_simple_agent", name="my_relay", server_type="legacy_agent"),
            "renamed_agent",
            "renamed_environment_server",
            "legacy_agent",
        ),
        # The source's server is not a legacy_agent relay; the renamed server must keep its type.
        (
            _server_fronting("my_simple_agent", name="my_environment_server", server_type="single_agent_turn_legacy"),
            "renamed_agent",
            "renamed_environment_server",
            "single_agent_turn_legacy",
        ),
        # The rename's server name is the source server's name, so that server is pointed at the new name.
        (
            _server_fronting("my_simple_agent", name="my_environment_server", server_type="single_agent_turn_legacy"),
            "my_agent",
            "my_environment_server",
            "single_agent_turn_legacy",
        ),
    ],
    ids=["generated-source-server", "custom-named-source-server", "non-legacy-source-server", "same-server-name"],
)
def test_migrated_rename_parses_with_one_server_of_the_sources_type(
    tmp_path: Path, source_server: str, rename: str, expected_server: str, expected_type: str
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + source_server)
    overlay = tmp_path / "rename.yaml"
    overlay.write_text(f"{rename}:\n  _inherit_from: my_simple_agent\n")

    assert migration.main([str(base), str(overlay)]) == 0

    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                OmegaConf.load(base),
                OmegaConf.load(overlay),
                # A generated relay would hide a missing server, so require the migrated one.
                {"error_on_agent_without_environment_server": True},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    assert _environment_servers_by_agent(resolved) == {rename: [expected_server]}
    assert list(resolved[expected_server]["environment_servers"]) == [expected_type]


def test_reports_an_unreadable_config(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "broken.yaml").write_text("key: [unclosed\n")

    assert migration.main([str(tmp_path)]) == 2

    assert "broken.yaml" in capsys.readouterr().err


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
