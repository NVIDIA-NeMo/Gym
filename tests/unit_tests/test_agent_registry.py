# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

from nemo_gym.agent_registry import (
    AgentEntry,
    _discover_agents_in_dir,
    discover_agents,
)


def _make_agent(agents_dir: Path, name: str, *, app: bool = True, configs: dict = None) -> Path:
    agent_dir = agents_dir / name
    agent_dir.mkdir(parents=True)
    if app:
        (agent_dir / "app.py").write_text("# app\n")
    if configs:
        configs_dir = agent_dir / "configs"
        configs_dir.mkdir()
        for variant, body in configs.items():
            (configs_dir / f"{variant}.yaml").write_text(body)
    return agent_dir


def _pattern_a(agent_type: str = "simple_agent") -> str:
    # References a separate resources server -> composable.
    return (
        f"some_key:\n  responses_api_agents:\n    {agent_type}:\n      entrypoint: app.py\n"
        "      resources_server:\n        type: resources_servers\n        name: ???\n"
        "      description: A composable agent\n"
    )


def _pattern_b(agent_type: str = "swe_agent") -> str:
    # Self-contained framework agent -> not composable.
    return (
        f"some_key:\n  responses_api_agents:\n    {agent_type}:\n      entrypoint: app.py\n"
        "      agent_framework: openhands\n"
    )


class TestDiscoverAgents:
    def test_discovers_and_classifies_pattern_a(self, tmp_path: Path) -> None:
        _make_agent(tmp_path, "simple_agent", configs={"simple_agent": _pattern_a()})

        agents = _discover_agents_in_dir(tmp_path)

        assert set(agents) == {"simple_agent"}
        entry = agents["simple_agent"]
        assert entry.self_contained is False
        assert entry.description == "A composable agent"
        assert list(entry.variants) == ["simple_agent"]

    def test_classifies_pattern_b_as_not_composable(self, tmp_path: Path) -> None:
        _make_agent(tmp_path, "swe_agents", configs={"swebench": _pattern_b()})

        assert _discover_agents_in_dir(tmp_path)["swe_agents"].self_contained is True

    def test_external_harness_agent_is_not_composable(self, tmp_path: Path) -> None:
        body = (
            "k:\n  responses_api_agents:\n    claude_code_agent:\n      entrypoint: app.py\n"
            "      resources_server:\n        name: ???\n      anthropic_api_key: ???\n"
        )
        _make_agent(tmp_path, "claude_code_agent", configs={"claude_code_agent": body})

        # Has a resources_server but drives an external LLM harness -> not composable.
        assert _discover_agents_in_dir(tmp_path)["claude_code_agent"].self_contained is True

    def test_zero_config_agent_is_discovered_and_defaults_composable(self, tmp_path: Path) -> None:
        _make_agent(tmp_path, "aviary_agent", configs=None)  # app.py only, no configs

        entry = _discover_agents_in_dir(tmp_path)["aviary_agent"]
        assert entry.config_paths == ()
        assert entry.self_contained is False

    def test_multiple_variants_are_all_recorded(self, tmp_path: Path) -> None:
        _make_agent(
            tmp_path,
            "langgraph_agent",
            configs={
                "orchestrator_agent": _pattern_a("langgraph_agent"),
                "rewoo_agent": _pattern_a("langgraph_agent"),
            },
        )

        assert set(_discover_agents_in_dir(tmp_path)["langgraph_agent"].variants) == {
            "orchestrator_agent",
            "rewoo_agent",
        }

    def test_non_agent_yaml_is_filtered_out(self, tmp_path: Path) -> None:
        # A configs/ file that is not a gym agent config (no responses_api_agents) is ignored;
        # the dir still counts as an agent because of app.py.
        _make_agent(tmp_path, "swe_agents", configs={"raw_harness": "agent:\n  type: openhands\n"})

        entry = _discover_agents_in_dir(tmp_path)["swe_agents"]
        assert entry.config_paths == ()

    def test_directory_without_app_or_configs_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "not_an_agent").mkdir()
        (tmp_path / "loose_file.txt").write_text("x")

        assert _discover_agents_in_dir(tmp_path) == {}

    def test_unparseable_config_does_not_crash_discovery(self, tmp_path: Path) -> None:
        _make_agent(tmp_path, "broken", configs={"broken": "responses_api_agents: [unclosed\n"})

        # The bad file is skipped (not an agent config); the dir survives via app.py.
        assert _discover_agents_in_dir(tmp_path)["broken"].config_paths == ()

    def test_missing_directory_yields_no_agents(self, tmp_path: Path) -> None:
        assert _discover_agents_in_dir(tmp_path / "nope") == {}


class TestRealAgents:
    def test_discovers_real_simple_agent_as_composable(self) -> None:
        agents = discover_agents()
        # The repo ships a `simple_agent`; it pairs with a separate resources server.
        if "simple_agent" in agents:
            assert agents["simple_agent"].self_contained is False

    def test_agent_entry_is_hashable(self) -> None:
        entry = AgentEntry(name="a", path=Path("a"), config_paths=(Path("a/configs/a.yaml"),), self_contained=True)
        assert {entry: 1}[entry] == 1
        assert entry.variants == {"a": Path("a/configs/a.yaml")}
