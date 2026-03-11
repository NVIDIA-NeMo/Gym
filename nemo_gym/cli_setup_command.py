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
from os import environ
from pathlib import Path
from subprocess import Popen
from sys import stderr, stdout

from omegaconf import DictConfig

from nemo_gym.global_config import (
    HEAD_SERVER_DEPS_KEY_NAME,
    PIP_INSTALL_VERBOSE_KEY_NAME,
    PYTHON_VERSION_KEY_NAME,
    SKIP_VENV_IF_PRESENT_KEY_NAME,
    UV_CACHE_DIR_KEY_NAME,
    UV_PIP_SET_PYTHON_KEY_NAME,
    UV_VENV_DIR_KEY_NAME,
    get_global_config_dict,
)


def setup_env_command(dir_path: Path, global_config_dict: DictConfig, prefix: str) -> str:
    head_server_deps = global_config_dict[HEAD_SERVER_DEPS_KEY_NAME]

    root_venv_path = global_config_dict[UV_VENV_DIR_KEY_NAME]
    venv_path = Path(root_venv_path, *dir_path.parts[-2:], ".venv").absolute()

    uv_venv_cmd = f"uv venv --seed --allow-existing --python {global_config_dict[PYTHON_VERSION_KEY_NAME]} {venv_path}"

    venv_python_fpath = venv_path / "bin/python"
    venv_activate_fpath = venv_path / "bin/activate"
    skip_venv_if_present = global_config_dict[SKIP_VENV_IF_PRESENT_KEY_NAME]
    should_skip_venv_setup = bool(skip_venv_if_present) and venv_python_fpath.exists() and venv_activate_fpath.exists()

    # explicitly set python path if specified. In Google colab, ng_run fails due to uv pip install falls back to system python (/usr) without this and errors.
    # not needed for most clusters. should be safe in all scenarios, but only minimally tested outside of colab.
    # see discussion and examples here: https://github.com/NVIDIA-NeMo/Gym/pull/526#issuecomment-3676230383
    uv_pip_set_python = global_config_dict.get(UV_PIP_SET_PYTHON_KEY_NAME, False)
    uv_pip_python_flag = f"--python {venv_python_fpath} " if uv_pip_set_python else ""

    verbose_flag = "-v " if global_config_dict.get(PIP_INSTALL_VERBOSE_KEY_NAME) else ""

    is_editable_install = (dir_path / "../../pyproject.toml").exists()

    if should_skip_venv_setup:
        env_setup_cmd = f"source {venv_activate_fpath}"
    else:
        has_pyproject_toml = (dir_path / "pyproject.toml").exists()
        has_requirements_txt = (dir_path / "requirements.txt").exists()
        if has_pyproject_toml and has_requirements_txt:
            raise RuntimeError(
                f"Found both pyproject.toml and requirements.txt for uv venv setup in server dir: {dir_path}. Please only use one or the other!"
            )
        elif has_pyproject_toml:
            if is_editable_install:
                install_cmd = (
                    f"""uv pip install {verbose_flag}{uv_pip_python_flag}'-e .' {" ".join(head_server_deps)}"""
                )
            else:
                # install nemo-gym from pypi instead of relative path in pyproject.toml
                install_cmd = (
                    f"""uv pip install {verbose_flag}{uv_pip_python_flag}nemo-gym && """
                    f"""uv pip install {verbose_flag}{uv_pip_python_flag}--no-sources '-e .' {" ".join(head_server_deps)}"""
                )
        elif has_requirements_txt:
            if is_editable_install:
                install_cmd = f"""uv pip install {verbose_flag}{uv_pip_python_flag}-r requirements.txt {" ".join(head_server_deps)}"""
            else:
                # install nemo-gym from pypi instead of relative path in requirements.txt
                install_cmd = (
                    f"""(echo 'nemo-gym' && grep -v -F '../..' requirements.txt) | """
                    f"""uv pip install {verbose_flag}{uv_pip_python_flag}-r /dev/stdin {" ".join(head_server_deps)}"""
                )
        else:
            raise RuntimeError(
                f"Missing pyproject.toml or requirements.txt for uv venv setup in server dir: {dir_path}"
            )

        prefix_cmd = f" > >(sed 's/^/({prefix}) /') 2> >(sed 's/^/({prefix}) /' >&2)"
        env_setup_cmd = f"{uv_venv_cmd}{prefix_cmd} && source {venv_activate_fpath} && {install_cmd}{prefix_cmd}"

    return f"cd {dir_path} && {env_setup_cmd}"


def run_command(command: str, working_dir_path: Path) -> Popen:
    global_config_dict = get_global_config_dict()
    global_config_dict

    work_dir = f"{working_dir_path.absolute()}"
    custom_env = environ.copy()
    py_path = custom_env.get("PYTHONPATH", None)
    if py_path is not None:
        custom_env["PYTHONPATH"] = f"{work_dir}:{py_path}"
    else:
        custom_env["PYTHONPATH"] = work_dir

    custom_env["UV_CACHE_DIR"] = global_config_dict[UV_CACHE_DIR_KEY_NAME]

    redirect_stdout = stdout
    redirect_stderr = stderr
    return Popen(
        command,
        executable="/bin/bash",
        shell=True,
        env=custom_env,
        stdout=redirect_stdout,
        stderr=redirect_stderr,
    )
