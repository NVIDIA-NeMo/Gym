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
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.agent_registry import AgentEntry
from nemo_gym.cli.agents import list_agents


def _mock_global_config(config: dict = None):
    return OmegaConf.create(config or {})


def _entry(name: str, self_contained: bool, variants=(), description=None) -> AgentEntry:
    path = Path("responses_api_agents") / name
    config_paths = tuple(path / "configs" / f"{v}.yaml" for v in variants)
    return AgentEntry(
        name=name,
        path=path,
        config_paths=config_paths,
        self_contained=self_contained,
        description=description,
    )


_AGENTS = {
    "simple_agent": _entry("simple_agent", self_contained=False, variants=("simple_agent",)),
    "swe_agents": _entry(
        "swe_agents",
        self_contained=True,
        variants=("swebench_openhands",),
        description="Agent for software engineering tasks",
    ),
}


class TestListAgents:
    def test_lists_found_agents(self, capsys) -> None:
        with (
            patch("nemo_gym.cli.agents.get_global_config_dict", return_value=_mock_global_config()),
            patch("nemo_gym.cli.agents.discover_agents", return_value=_AGENTS),
        ):
            list_agents()
        out = capsys.readouterr().out
        assert "simple_agent" in out and "swe_agents" in out
        assert "composable" in out and "self-contained" in out

    def test_no_agents(self, capsys) -> None:
        with (
            patch("nemo_gym.cli.agents.get_global_config_dict", return_value=_mock_global_config()),
            patch("nemo_gym.cli.agents.discover_agents", return_value={}),
        ):
            list_agents()
        assert "No agents found" in capsys.readouterr().out

    def test_json_output(self, capsys) -> None:
        with (
            patch("nemo_gym.cli.agents.get_global_config_dict", return_value=_mock_global_config({"json": True})),
            patch("nemo_gym.cli.agents.discover_agents", return_value=_AGENTS),
        ):
            list_agents()
        payload = json.loads(capsys.readouterr().out)
        by_name = {entry["name"]: entry for entry in payload}
        assert by_name["simple_agent"]["self_contained"] is False
        assert by_name["simple_agent"]["pattern"] == "A (composable)"
        assert by_name["swe_agents"]["self_contained"] is True
        assert by_name["swe_agents"]["variants"] == ["swebench_openhands"]

    def test_query_filters_agents(self, capsys) -> None:
        # `gym search agents <query>` reuses this command via the `query` config key (name + description + variants).
        with (
            patch("nemo_gym.cli.agents.get_global_config_dict", return_value=_mock_global_config({"query": "swe"})),
            patch("nemo_gym.cli.agents.discover_agents", return_value=_AGENTS),
        ):
            list_agents()
        out = capsys.readouterr().out
        assert "swe_agents" in out and "Agents matching" in out
        assert "simple_agent" not in out

    def test_query_matches_description(self, capsys) -> None:
        # "engineering" only appears in swe_agents' description, not its name or variants.
        with (
            patch(
                "nemo_gym.cli.agents.get_global_config_dict",
                return_value=_mock_global_config({"query": "engineering"}),
            ),
            patch("nemo_gym.cli.agents.discover_agents", return_value=_AGENTS),
        ):
            list_agents()
        out = capsys.readouterr().out
        assert "swe_agents" in out and "Agents matching" in out
        assert "simple_agent" not in out

    def test_inspect_agent_by_name(self, capsys) -> None:
        with (
            patch(
                "nemo_gym.cli.agents.get_global_config_dict",
                return_value=_mock_global_config({"component_name": "swe_agents"}),
            ),
            patch("nemo_gym.cli.agents.discover_agents", return_value=_AGENTS),
        ):
            list_agents()
        out = capsys.readouterr().out

        assert "The swe_agents agent" in out
        assert "composition: self-contained (B)" in out
        assert "variants: swebench_openhands" in out
        assert "software engineering tasks" in out  # description
        assert "Usage example:" not in out  # thin view

    def test_inspect_unknown_agent_exits(self, capsys) -> None:
        with (
            patch(
                "nemo_gym.cli.agents.get_global_config_dict",
                return_value=_mock_global_config({"component_name": "swe_agent"}),
            ),
            patch("nemo_gym.cli.agents.discover_agents", return_value=_AGENTS),
        ):
            with pytest.raises(SystemExit):
                list_agents()
        out = capsys.readouterr().out
        assert "Unknown agent 'swe_agent'" in out and "swe_agents" in out

    def test_inspect_shows_absolute_path(self, tmp_path: Path, capsys, monkeypatch) -> None:
        # Real discovery (via an extra root): the path line must be the agent dir's absolute path.
        agent_dir = tmp_path / "responses_api_agents" / "my_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "app.py").write_text("")
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))
        with patch(
            "nemo_gym.cli.agents.get_global_config_dict",
            return_value=_mock_global_config({"component_name": "my_agent"}),
        ):
            list_agents()
        assert f"path: {agent_dir.resolve()}" in capsys.readouterr().out
