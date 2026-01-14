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
import sys
import tomllib
from importlib import import_module
from io import StringIO
from pathlib import Path

from omegaconf import OmegaConf
from pytest import MonkeyPatch, raises

import nemo_gym.global_config
from nemo_gym import PARENT_DIR
from nemo_gym.cli import RunConfig, init_resources_server
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

    def test_init_resources_server_includes_domain(self) -> None:
        """Test that init_resources_server creates a config with the required domain field."""
        import shutil

        # Use a temp directory but stay in the project root for access to template files
        server_name = "test_cli_server"
        entrypoint = f"resources_servers/{server_name}"

        # Clean up any existing test server directory
        if Path(entrypoint).exists():
            shutil.rmtree(entrypoint)

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
                config_file = Path(entrypoint) / "configs" / f"{server_name}.yaml"
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
            if Path(entrypoint).exists():
                shutil.rmtree(entrypoint)
