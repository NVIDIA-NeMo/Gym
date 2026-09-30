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
import shlex
from pathlib import Path

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.cli.env import _server_launch_command
from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    UV_VENV_DIR_KEY_NAME,
)
from tests.unit_tests.test_global_config import TestGlobalConfig as _TestGlobalConfig


class TestServerLaunchCommand:
    """`gym env start` runs each server with its venv's own interpreter, not the `python` on PATH after
    `source bin/activate`, so a relocated venv whose `bin/activate` names its original prefix cannot win."""

    def _setup_server_dir(self, tmp_path: Path) -> Path:
        # A space in the path catches any unquoted venv or server path in the generated command.
        server_dir = tmp_path / "my checkout" / "resources_servers" / "my_server"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        (tmp_path / "my checkout" / "pyproject.toml").write_text("")
        return server_dir.absolute()

    @pytest.mark.parametrize("uv_venv_dir", [None, "shared venvs"])
    def test_runs_the_entrypoint_with_the_venv_interpreter(self, tmp_path: Path, uv_venv_dir: str | None) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        global_config_dict = _TestGlobalConfig._default_global_config_dict_values.fget(None) | {
            UV_VENV_DIR_KEY_NAME: str(tmp_path / uv_venv_dir) if uv_venv_dir else str(PARENT_DIR)
        }
        expected_venv = (
            tmp_path / uv_venv_dir / "resources_servers" / "my_server" / ".venv"
            if uv_venv_dir
            else server_dir / ".venv"
        )

        command = _server_launch_command(
            server_dir, global_config_dict, "my_server", Path("app.py"), shlex.quote("config: yaml")
        )

        *setup_lines, launch_line = command.splitlines()
        assert shlex.split(launch_line) == [str(expected_venv / "bin" / "python"), "app.py"]
        # The interpreter comes from the same venv that the setup step builds and activates.
        assert f"source {shlex.quote(str(expected_venv / 'bin' / 'activate'))}" in setup_lines[0]
        assert f"{NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME}='config: yaml'" in command
        assert f"{NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME}=my_server" in command
