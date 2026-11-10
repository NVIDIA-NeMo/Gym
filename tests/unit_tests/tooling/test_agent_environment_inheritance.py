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

"""Environment migration: agent environment inheritance."""

from pathlib import Path

import pytest
import yaml

from nemo_gym.rollouts.rollout_collection import _environment_servers_by_agent
from tests.helpers.agent_environment_migration import (
    AGENT_CONFIG,
    RELAY,
    _parent_and_child,
    _parse,
    _switched_server,
    load_migration,
)


migration = load_migration()


def test_switch_keeps_a_child_relay_inheriting_its_switched_parent(tmp_path: Path) -> None:
    base, child = _parent_and_child(tmp_path)

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 0

    document = yaml.safe_load(child.read_text())
    # Still inheriting, the parent stays a template that composition consumes.
    assert document["child_environment_server"]["_inherit_from"] == "my_environment_server"
    server = _switched_server(document, "child_environment_server")
    assert server == {
        "agent_server": {"name": "child_agent"},
        "resources_server": {"type": "resources_servers", "name": "child_resources"},
    }
    resolved = _parse(child, strict=True)
    assert _environment_servers_by_agent(resolved) == {"child_agent": ["child_environment_server"]}
    merged = resolved["child_environment_server"]["environment_servers"]
    assert list(merged) == ["single_agent_turn_legacy"]
    assert merged["single_agent_turn_legacy"]["entrypoint"] == "app.py"
    assert merged["single_agent_turn_legacy"]["resources_server"]["name"] == "child_resources"
    assert resolved["child_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]["name"] == (
        "child_resources"
    )


def test_a_switched_relay_follows_agent_and_resources_renames(tmp_path: Path) -> None:
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
    server = resolved["my_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    assert server["resources_server"]["name"] == "renamed_resources"


def test_switch_keeps_a_child_resources_copy_explicit(tmp_path: Path) -> None:
    base, child = _parent_and_child(tmp_path)
    child.write_text(child.read_text().replace("_inherit_from: my_resources", "_copy: my_resources"))

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 0

    resolved = _parse(child, strict=True)
    assert "my_resources" in resolved and "child_resources" in resolved
    server = resolved["child_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    assert server["resources_server"]["name"] == "child_resources"


@pytest.mark.parametrize("rename_agent", [False, True], ids=["same-agent", "renamed-agent"])
def test_switch_refuses_an_agent_resources_override_without_an_environment_binding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, rename_agent: bool
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + RELAY)
    child = tmp_path / "child.yaml"
    agent = "child_agent:\n  _inherit_from: my_simple_agent\n" if rename_agent else "my_simple_agent:\n"
    child.write_text(
        f"config_paths:\n  - {base}\n"
        "child_resources:\n  _copy: my_resources\n"
        + agent
        + "  responses_api_agents:\n    simple_agent:\n      resources_server:\n        name: child_resources\n"
    )
    monkeypatch.setattr(migration, "repo_config_files", lambda: [base, child])
    before = base.read_text(), child.read_text()

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 2

    assert (base.read_text(), child.read_text()) == before
    error = capsys.readouterr().err
    assert "my_environment_server.environment_servers.single_agent_turn_legacy.resources_server.name" in error
    assert "child_resources" in error


def test_switch_allows_unambiguous_resources_renames_in_including_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, child = _parent_and_child(tmp_path)
    child.write_text(child.read_text().split("child_environment_server:")[0])
    monkeypatch.setattr(migration, "repo_config_files", lambda: [base, child])

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 0

    resolved = _parse(child, strict=True)
    server = resolved["my_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    assert server["agent_server"]["name"] == "child_agent"
    assert server["resources_server"]["name"] == "child_resources"


@pytest.mark.parametrize("deleted_key", ["responses_api_agents", " responses_api_agents "])
def test_switch_refuses_rebinding_when_only_one_inherited_destination_is_an_agent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, deleted_key: str
) -> None:
    base, child = _parent_and_child(tmp_path)
    child.write_text(
        child.read_text()
        .split("child_environment_server:")[0]
        .replace("_inherit_from: my_resources", "_copy: my_resources")
        + f"holder:\n  _inherit_from: my_simple_agent\n  _delete_key: '{deleted_key}'\n"
    )
    monkeypatch.setattr(migration, "repo_config_files", lambda: [base, child])
    before = base.read_text(), child.read_text()
    resolved = _parse(child, strict=True)
    assert "responses_api_agents" not in resolved["holder"]
    assert _environment_servers_by_agent(resolved) == {"child_agent": ["my_environment_server"]}

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 2

    assert (base.read_text(), child.read_text()) == before
    assert "resources_server.name: child_resources" in capsys.readouterr().err


@pytest.mark.parametrize("rename", ["string", "non-resources-destination"])
def test_switch_matches_parser_resources_rename_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rename: str
) -> None:
    base, child = _parent_and_child(tmp_path)
    text = child.read_text().split("child_environment_server:")[0]
    if rename == "string":
        text = text.replace(
            "child_resources:\n  _inherit_from: my_resources",
            "child_resources: ${inherit_from:my_resources}",
        )
    else:
        text += "holder:\n  _inherit_from: my_resources\n  _delete_key: resources_servers\n"
    child.write_text(text)
    monkeypatch.setattr(migration, "repo_config_files", lambda: [base, child])

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 0

    resolved = _parse(child, strict=True)
    server = resolved["my_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    assert server["agent_server"]["name"] == "child_agent"
    assert server["resources_server"]["name"] == "child_resources"
    assert "my_resources" not in resolved


def test_switch_preserves_a_string_environment_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base, child = _parent_and_child(tmp_path)
    child.write_text(
        child.read_text().split("child_environment_server:")[0]
        + "child_environment_server: ${inherit_from:my_environment_server}\n"
    )
    monkeypatch.setattr(migration, "repo_config_files", lambda: [base, child])
    before = child.read_text()

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 0

    assert child.read_text() == before
    resolved = _parse(child, strict=True)
    assert _environment_servers_by_agent(resolved) == {"child_agent": ["child_environment_server"]}
    server = resolved["child_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    assert server["resources_server"]["name"] == "child_resources"


def test_switch_preserves_an_including_environments_explicit_resources_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + RELAY)
    child = tmp_path / "child.yaml"
    child.write_text(
        f"config_paths:\n  - {base}\n"
        "child_resources:\n  _copy: my_resources\n"
        "my_environment_server:\n  environment_servers:\n    single_agent_turn_legacy:\n"
        "      resources_server:\n        type: resources_servers\n        name: child_resources\n"
    )
    monkeypatch.setattr(migration, "repo_config_files", lambda: [base, child])
    before = child.read_text()

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 0

    assert child.read_text() == before
    resolved = _parse(child, strict=True)
    server = resolved["my_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    assert server["resources_server"]["name"] == "child_resources"
    assert resolved["my_simple_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]["name"] == (
        "my_resources"
    )


def test_switch_refuses_an_ambiguous_resources_rename_without_an_environment_binding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    base, child = _parent_and_child(tmp_path)
    child.write_text(
        child.read_text().split("child_environment_server:")[0] + "other_resources:\n  _inherit_from: my_resources\n"
    )
    monkeypatch.setattr(migration, "repo_config_files", lambda: [base, child])
    before = base.read_text(), child.read_text()

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 2

    assert (base.read_text(), child.read_text()) == before
    assert "resources_server.name: child_resources" in capsys.readouterr().err


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
