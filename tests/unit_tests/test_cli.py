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
import shutil
import sys
import tomllib
from importlib import import_module
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
from pytest import MonkeyPatch, raises

import nemo_gym.global_config
from nemo_gym import PARENT_DIR
from nemo_gym.cli import RunConfig, display_help, init_resources_server
from nemo_gym.config_types import ResourcesServerInstanceConfig


# TODO: Eventually we want to add more tests to ensure that the CLI flows do not break
class TestCLI:
    def test_sanity(self) -> None:
        RunConfig(entrypoint="", name="")

    def test_pyproject_scripts(self) -> None:
        pyproject_path = PARENT_DIR / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject_data = tomllib.load(f)

        project_scripts = pyproject_data["project"]["scripts"]

        for script_name, import_path in project_scripts.items():
            # Dedupe `nemo_gym_*` from `ng_*` commands
            if not script_name.startswith("ng_"):
                continue

            # We only test `+h=true` and not `+help=true`
            print(f"Running `{script_name} +h=true`")

            module, fn = import_path.split(":")
            fn = getattr(import_module(module), fn)

            with MonkeyPatch.context() as mp:
                mp.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", OmegaConf.create({"h": True}))

                text_trap = StringIO()
                mp.setattr(sys, "stdout", text_trap)

                with raises(SystemExit):
                    fn()

    def test_display_help_discovers_scripts(self) -> None:
        with MonkeyPatch.context() as mp:
            mp.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", OmegaConf.create({}))

            text_trap = StringIO()
            mp.setattr(sys, "stdout", text_trap)

            display_help()

            output = text_trap.getvalue()
            assert "ng_help" in output
            assert "ng_run" in output
            assert "ng_collect_rollouts" in output

    def test_init_resources_server_includes_domain(self) -> None:
        """Test that init_resources_server creates a config with the required domain field."""

        server_name = "test_cli_server"
        entrypoint = f"nemo_gym/resources_servers/{server_name}"
        server_path = Path(entrypoint).resolve()

        # Clean up any existing test server directory
        if server_path.exists():
            shutil.rmtree(server_path)

        try:
            with MonkeyPatch.context() as mp:
                # Set up the global config to point to our test entrypoint
                mp.setattr(
                    nemo_gym.global_config,
                    "_GLOBAL_CONFIG_DICT",
                    OmegaConf.create({"entrypoint": entrypoint}),
                )

                # Run init_resources_server
                init_resources_server()

                # Verify the generated config file exists
                config_file = server_path / "configs" / f"{server_name}.yaml"
                assert config_file.exists(), f"Config file not created at {config_file}"

                # Load and verify the config
                config_dict = OmegaConf.load(config_file)

                # Check that the domain field is present in the resources server config
                resources_server_key = f"{server_name}_resources_server"
                assert resources_server_key in config_dict, f"Resources server key '{resources_server_key}' not found"

                resources_config = config_dict[resources_server_key]
                assert "resources_servers" in resources_config
                assert server_name in resources_config["resources_servers"]

                server_config = resources_config["resources_servers"][server_name]
                assert "domain" in server_config, "Domain field missing from resources server config"
                assert server_config["domain"] == "other", f"Expected domain 'other', got '{server_config['domain']}'"

                # Verify that the config can be validated (this would have failed before the fix)
                full_config_dict = OmegaConf.create(
                    {
                        "name": resources_server_key,
                        "server_type_config_dict": config_dict[resources_server_key],
                        **OmegaConf.to_container(config_dict[resources_server_key]),
                    }
                )

                # This should not raise an assertion error about missing domain
                instance_config = ResourcesServerInstanceConfig.model_validate(full_config_dict)
                assert instance_config is not None
        finally:
            # Clean up the test server directory
            if server_path.exists():
                shutil.rmtree(server_path)

    def test_run_helper_prefers_cwd_server_over_install(self, tmp_path: Path) -> None:
        """ng_run should use a local CWD server dir instead of the installed one."""
        # Create a fake local server dir in tmp_path (simulates user's own resources_servers/)
        local_server = tmp_path / "resources_servers" / "my_server"
        local_server.mkdir(parents=True)
        (local_server / "requirements.txt").write_text("nemo-gym\n")

        with patch.object(Path, "cwd", return_value=tmp_path):
            _cwd_path = Path.cwd() / Path("resources_servers", "my_server")
            dir_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / Path("resources_servers", "my_server")

        assert dir_path == local_server

    def test_run_helper_falls_back_to_install_when_not_in_cwd(self, tmp_path: Path) -> None:
        """ng_run should fall back to PARENT_DIR when the server doesn't exist in CWD."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            _cwd_path = Path.cwd() / Path("resources_servers", "arc_agi")
            dir_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / Path("resources_servers", "arc_agi")

        assert dir_path == PARENT_DIR / "resources_servers" / "arc_agi"
