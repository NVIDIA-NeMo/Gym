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

from omegaconf import OmegaConf

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
    "swe_agents": _entry("swe_agents", self_contained=True, variants=("swebench_openhands",), description="SWE tasks"),
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
