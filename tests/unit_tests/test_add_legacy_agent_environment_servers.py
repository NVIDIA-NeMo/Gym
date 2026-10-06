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

from nemo_gym.global_config import legacy_environment_server_name


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
