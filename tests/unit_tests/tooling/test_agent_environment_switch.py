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

"""Environment migration: agent environment switch."""

from pathlib import Path

import pytest
import yaml

from nemo_gym.rollouts.rollout_collection import _environment_servers_by_agent
from tests.helpers.agent_environment_migration import (
    AGENT_CONFIG,
    FOLLOWING,
    RELAY,
    _parse,
    _switched_server,
    load_migration,
)


migration = load_migration()


def test_switch_replaces_a_simple_agent_relay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig

    config = tmp_path / "my_run.yaml"
    config.write_text(AGENT_CONFIG + RELAY + FOLLOWING)

    assert migration.main(["--switch-simple-agent", str(config)]) == 0

    text = config.read_text()
    assert text.startswith(AGENT_CONFIG) and text.endswith(FOLLOWING)
    server = _switched_server(yaml.safe_load(text), "my_environment_server")
    assert server["resources_server"] == {"type": "resources_servers", "name": "my_resources"}
    assert server["agent_server"] == {"type": "responses_api_agents", "name": "my_simple_agent"}
    # The agent's /run called the Resources Server's tool routes; the grant gives the session the same routes.
    assert server["resources_tool_transports"] == ["direct_http"]
    assert "default_episode_timeout_seconds" not in server
    assert "cleanup_timeout_seconds" not in server
    server_config = SingleAgentTurnLegacyEnvironmentServerConfig.model_validate(
        {**server, "name": "my_environment_server", "host": "h", "port": 1}
    )
    assert server_config.default_episode_timeout_seconds == 21600
    assert server_config.cleanup_timeout_seconds == 180
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


@pytest.mark.parametrize(
    ("seed_session", "switches"),
    [
        (None, True),
        ("async def seed_session(self, body: ResourcesSeedSessionRequest | BaseSeedSessionRequest): ...", True),
        # Reads the raw request and answers an episode seed with its resources_session_id.
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
