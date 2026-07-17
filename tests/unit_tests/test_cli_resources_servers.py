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
import json
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from nemo_gym.cli.resources_servers import list_resources_servers
from nemo_gym.resources_server_registry import ResourcesServerEntry


def _mock_global_config(config: dict = None):
    return OmegaConf.create(config or {})


def _entry(name: str, domain: str, description: str) -> ResourcesServerEntry:
    path = Path("resources_servers") / name
    return ResourcesServerEntry(
        name=name, config_path=path / "configs" / f"{name}.yaml", path=path, description=description, domain=domain
    )


_SERVERS = {
    "mcqa": _entry("mcqa", "knowledge", "Multi-choice QA"),
    "aviary": _entry("aviary", "math", "Math tasks"),
}


class TestListResourcesServers:
    def test_lists_servers(self, capsys) -> None:
        with (
            patch("nemo_gym.cli.resources_servers.get_global_config_dict", return_value=_mock_global_config()),
            patch("nemo_gym.cli.resources_servers.discover_resources_servers", return_value=_SERVERS),
        ):
            list_resources_servers()
        out = capsys.readouterr().out
        assert "mcqa" in out and "knowledge" in out and "aviary" in out

    def test_no_servers(self, capsys) -> None:
        with (
            patch("nemo_gym.cli.resources_servers.get_global_config_dict", return_value=_mock_global_config()),
            patch("nemo_gym.cli.resources_servers.discover_resources_servers", return_value={}),
        ):
            list_resources_servers()
        assert "No resources servers found" in capsys.readouterr().out

    def test_json_output(self, capsys) -> None:
        with (
            patch(
                "nemo_gym.cli.resources_servers.get_global_config_dict",
                return_value=_mock_global_config({"json": True}),
            ),
            patch("nemo_gym.cli.resources_servers.discover_resources_servers", return_value=_SERVERS),
        ):
            list_resources_servers()
        payload = json.loads(capsys.readouterr().out)
        expected = [
            {"name": "mcqa", "domain": "knowledge", "description": "Multi-choice QA"},
            {"name": "aviary", "domain": "math", "description": "Math tasks"},
        ]
        assert len(payload) == len(expected)
        for row in expected:
            assert row in payload

    def test_query_filters_servers(self, capsys) -> None:
        # `gym search resources-servers <query>` reuses this command via the `query` config key (name + domain).
        with (
            patch(
                "nemo_gym.cli.resources_servers.get_global_config_dict",
                return_value=_mock_global_config({"query": "math"}),
            ),
            patch("nemo_gym.cli.resources_servers.discover_resources_servers", return_value=_SERVERS),
        ):
            list_resources_servers()
        out = capsys.readouterr().out
        assert "aviary" in out and "Resources servers matching" in out
        assert "mcqa" not in out and "knowledge" not in out
