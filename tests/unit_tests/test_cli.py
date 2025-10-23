# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tomllib
from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock

from omegaconf import OmegaConf
from pytest import MonkeyPatch, raises

import nemo_gym.global_config
from nemo_gym import PARENT_DIR
from nemo_gym.cli import RunConfig
from nemo_gym.config_types import BaseNeMoGymCLIConfig


# TODO: Eventually we want to add more tests to ensure that the CLI flows do not break
class TestCLI:
    def test_sanity(self) -> None:
        RunConfig(entrypoint="", name="")

    def test_pyproject_scripts(self) -> None:
        pyproject_path = Path(PARENT_DIR) / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject_data = tomllib.load(f)

        project_scripts = pyproject_data["project"]["scripts"]

        def _test_single_arg(script_name: str, import_path: str, arg: str):
            print(f"Running `{script_name} +{arg}=true`")

            module, fn = import_path.split(":")
            fn = getattr(import_module(module), fn)

            with MonkeyPatch.context() as mp:
                mp.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", OmegaConf.create({arg: True}))

                pre_process_mock = MagicMock(side_effect=BaseNeMoGymCLIConfig.pre_process)
                mp.setattr(BaseNeMoGymCLIConfig, "pre_process", pre_process_mock)

                with raises(SystemExit):
                    fn()

                pre_process_mock.assert_called_once()

        for script_name, import_path in project_scripts.items():
            _test_single_arg(script_name, import_path, "h")
            _test_single_arg(script_name, import_path, "help")
