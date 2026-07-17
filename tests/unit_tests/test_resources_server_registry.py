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
from pathlib import Path

from nemo_gym.resources_server_registry import _discover_resources_servers_in_dir


def _make_resources_server(rs_dir: Path, name: str, *, flavors: dict) -> Path:
    """`flavors` maps config stem -> (domain, description). At least one config makes a resources server."""
    server_dir = rs_dir / name
    configs_dir = server_dir / "configs"
    configs_dir.mkdir(parents=True)
    for stem, (domain, description) in flavors.items():
        (configs_dir / f"{stem}.yaml").write_text(
            f"{name}:\n  resources_servers:\n    {name}:\n"
            f"      entrypoint: app.py\n      domain: {domain}\n      description: {description}\n"
        )
    return server_dir


class TestDiscoverResourcesServers:
    def test_reads_metadata_from_default_flavor(self, tmp_path: Path) -> None:
        # Metadata comes from the `<name>.yaml` flavor (the one `--resources-server <name>` resolves to).
        _make_resources_server(
            tmp_path,
            "my_server",
            flavors={"my_server": ("knowledge", "The default"), "some_other_flavor": ("other", "A flavor")},
        )

        servers = _discover_resources_servers_in_dir(tmp_path)

        assert set(servers) == {"my_server", "my_server/some_other_flavor"}  # one token per flavor
        entry = servers["my_server"]
        assert entry.domain == "knowledge"
        assert entry.description == "The default"
        assert entry.config_path.name == "my_server.yaml"

    def test_flavor_tokens_when_no_default(self, tmp_path: Path) -> None:
        # No `<name>.yaml`, so each flavor is its own `<name>/<flavor>` token (no collapsing to one entry).
        _make_resources_server(
            tmp_path, "my_server", flavors={"beta": ("science", "Beta"), "alpha": ("math", "Alpha")}
        )

        servers = _discover_resources_servers_in_dir(tmp_path)

        assert set(servers) == {"my_server/alpha", "my_server/beta"}
        assert servers["my_server/alpha"].domain == "math"

    def test_dirs_without_a_config_are_skipped(self, tmp_path: Path) -> None:
        # A dir with no config (e.g. a stray .egg-info) is not a resources server.
        (tmp_path / "my_server.egg-info").mkdir()
        _make_resources_server(tmp_path, "real_server", flavors={"real_server": ("other", "Real")})

        assert set(_discover_resources_servers_in_dir(tmp_path)) == {"real_server"}

    def test_helper_configs_are_skipped(self, tmp_path: Path) -> None:
        # A config with no `resources_servers` block (e.g. a judge model helper) is not a flavor.
        _make_resources_server(tmp_path, "my_server", flavors={"my_server": ("knowledge", "Default")})
        (tmp_path / "my_server" / "configs" / "judge_model.yaml").write_text(
            "judge_model:\n  responses_api_models:\n    m:\n      x: 1\n"
        )

        assert set(_discover_resources_servers_in_dir(tmp_path)) == {"my_server"}

    def test_missing_directory_yields_no_servers(self, tmp_path: Path) -> None:
        assert _discover_resources_servers_in_dir(tmp_path / "nope") == {}


class TestReadResourcesServerValue:
    def test_reads_value_from_config(self, tmp_path: Path) -> None:
        from nemo_gym.resources_server_registry import read_resources_server_value

        cfg = tmp_path / "my_server.yaml"
        cfg.write_text("s:\n  resources_servers:\n    my_server:\n      value: The value\n")

        assert read_resources_server_value(cfg) == "The value"

    def test_returns_none_when_no_value(self, tmp_path: Path) -> None:
        from nemo_gym.resources_server_registry import read_resources_server_value

        cfg = tmp_path / "my_server.yaml"
        cfg.write_text("s:\n  resources_servers:\n    my_server:\n      domain: x\n")

        assert read_resources_server_value(cfg) is None
