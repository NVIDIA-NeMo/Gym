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
import os
import stat
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf
from pytest import MonkeyPatch

import nemo_gym
import nemo_gym.config.global_config
from nemo_gym.config.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_DICT_PATH_ENV_VAR_NAME,
    maybe_get_global_config_dict,
)
from nemo_gym.server_runtime.supervisor.server_supervisor import _write_config_dict_file
from nemo_gym.utils.server_process import _has_injected_global_config_env


def _fresh_child(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
    monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_PATH_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(nemo_gym.config.global_config, "_GLOBAL_CONFIG_DICT", None)


def test_child_reads_config_from_file(monkeypatch: MonkeyPatch) -> None:
    _fresh_child(monkeypatch)
    path = _write_config_dict_file(OmegaConf.to_yaml({"my_server": {"port": 1234}}))
    try:
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_PATH_ENV_VAR_NAME, path)

        assert _has_injected_global_config_env()
        assert OmegaConf.to_container(maybe_get_global_config_dict()) == {"my_server": {"port": 1234}}
    finally:
        os.unlink(path)


def test_config_dict_env_var_wins_over_file(monkeypatch: MonkeyPatch) -> None:
    _fresh_child(monkeypatch)
    path = _write_config_dict_file(OmegaConf.to_yaml({"source": "file"}))
    try:
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_PATH_ENV_VAR_NAME, path)
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "source: env")

        assert maybe_get_global_config_dict().source == "env"
    finally:
        os.unlink(path)


def test_config_dict_file_is_private() -> None:
    path = _write_config_dict_file("api_key: synthetic-test-secret\n")
    try:
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    finally:
        os.unlink(path)


def test_config_over_env_string_limit_reaches_child(tmp_path: Path) -> None:
    # Linux caps a single env string at 128 KiB (MAX_ARG_STRLEN); passing this in an env var fails Popen with E2BIG.
    config = {f"server_{i}": {"description": "x" * 1000} for i in range(200)}
    path = _write_config_dict_file(OmegaConf.to_yaml(config))
    try:
        assert os.path.getsize(path) > 128 * 1024
        env = {key: value for key, value in os.environ.items() if not key.startswith("NEMO_GYM_CONFIG_DICT")}
        env[NEMO_GYM_CONFIG_DICT_PATH_ENV_VAR_NAME] = path
        env["PYTHONPATH"] = str(Path(nemo_gym.__file__).parent.parent)
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "from nemo_gym.config.global_config import maybe_get_global_config_dict; "
                "print(len(maybe_get_global_config_dict()))",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert child.returncode == 0, child.stderr
        assert child.stdout.strip() == "200"
    finally:
        os.unlink(path)
