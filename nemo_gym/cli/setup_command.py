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
import importlib.metadata
import os
import re
import shlex
from os import environ
from pathlib import Path
from subprocess import Popen
from sys import stderr, stdout
from typing import IO, Any

from omegaconf import DictConfig

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, PARENT_DIR
from nemo_gym.global_config import (
    HEAD_SERVER_DEPS_KEY_NAME,
    NEMO_GYM_LOG_DIR_KEY_NAME,
    PIP_INSTALL_VERBOSE_KEY_NAME,
    PYTHON_VERSION_KEY_NAME,
    SKIP_VENV_IF_PRESENT_KEY_NAME,
    UV_CACHE_DIR_KEY_NAME,
    UV_PIP_SET_PYTHON_KEY_NAME,
    UV_VENV_DIR_KEY_NAME,
    get_global_config_dict,
)


def _get_nemo_gym_install_flags() -> str:
    """
    Build uv pip install flags for nemo-gym in sub-venvs.

    Supports:
    - Pre-release versions via NEMO_GYM_ALLOW_PRERELEASE=true
    - Custom PyPI indexes via UV_INDEX_URL, UV_EXTRA_INDEX_URL, UV_INDEX_STRATEGY
    - Auto-detection of parent venv version for consistency

    Returns:
        String of flags to add to 'uv pip install nemo-gym'
        Example: "--pre --index-url https://test.pypi.org/simple/ ==0.2.1rc0"
    """
    flags = ""

    # 1. Pre-release flag
    allow_prerelease = os.getenv("NEMO_GYM_ALLOW_PRERELEASE", "").lower() == "true"
    if allow_prerelease:
        flags += "--pre "
        # When pre-releases are enabled, also use unsafe-best-match strategy if not already set
        if not os.getenv("UV_INDEX_STRATEGY"):
            flags += "--index-strategy unsafe-best-match "
        # Pin fastapi<1.0 to avoid broken test.pypi package
        flags += "'fastapi<1.0' "

    # 2. Index URLs (respects uv's standard env vars)
    index_url = os.getenv("UV_INDEX_URL")
    if index_url:
        flags += f"--index-url {index_url} "

    extra_index_url = os.getenv("UV_EXTRA_INDEX_URL")
    if extra_index_url:
        flags += f"--extra-index-url {extra_index_url} "

    # Explicit index strategy (overrides auto-set above)
    index_strategy = os.getenv("UV_INDEX_STRATEGY")
    if index_strategy:
        flags += f"--index-strategy {index_strategy} "

    return flags


def _get_nemo_gym_version_spec(is_editable_install: bool) -> str:
    """
    Detect nemo-gym version from parent venv and return version specifier.

    Args:
        is_editable_install: Whether nemo-gym is installed in editable mode in parent venv

    Returns:
        Version specifier string (e.g., "==0.2.1rc0") or empty string
    """
    # Don't pin version for editable installs (development mode)
    if is_editable_install:
        return ""

    try:
        parent_version = importlib.metadata.version("nemo-gym")
        # Pin to exact version for consistency between parent and sub-venvs
        return f"=={parent_version}"
    except importlib.metadata.PackageNotFoundError:
        # nemo-gym not installed in parent venv (shouldn't happen, but be safe)
        return ""


def get_venv_path(dir_path: Path, global_config_dict: DictConfig) -> Path:
    """Return the server venv path for the configured venv root."""
    root_venv_path = Path(global_config_dict[UV_VENV_DIR_KEY_NAME])
    if root_venv_path.resolve() != PARENT_DIR.resolve():
        return Path(root_venv_path, *dir_path.parts[-2:], ".venv").absolute()
    return (dir_path / ".venv").absolute()


def setup_env_command(dir_path: Path, global_config_dict: DictConfig, prefix: str) -> str:
    head_server_deps = global_config_dict[HEAD_SERVER_DEPS_KEY_NAME]

    venv_path = get_venv_path(dir_path, global_config_dict)

    uv_venv_cmd = f"uv venv --seed --allow-existing --python {global_config_dict[PYTHON_VERSION_KEY_NAME]} {venv_path}"

    venv_python_fpath = venv_path / "bin/python"
    venv_activate_fpath = venv_path / "bin/activate"
    skip_venv_if_present = global_config_dict[SKIP_VENV_IF_PRESENT_KEY_NAME]
    should_skip_venv_setup = bool(skip_venv_if_present) and venv_python_fpath.exists() and venv_activate_fpath.exists()

    # explicitly set python path if specified. In Google colab, gym env start fails due to uv pip install falls back to system python (/usr) without this and errors.
    # not needed for most clusters. should be safe in all scenarios, but only minimally tested outside of colab.
    # see discussion and examples here: https://github.com/NVIDIA-NeMo/Gym/pull/526#issuecomment-3676230383
    uv_pip_set_python = global_config_dict.get(UV_PIP_SET_PYTHON_KEY_NAME, False)
    uv_pip_python_flag = f"--python {venv_python_fpath} " if uv_pip_set_python else ""

    verbose_flag = "-v " if global_config_dict.get(PIP_INSTALL_VERBOSE_KEY_NAME) else ""

    is_editable_install = (dir_path.resolve() / "../../pyproject.toml").exists()
    # Downloaded components can reuse a development checkout even before its version reaches PyPI.
    package_core = ""
    if (dir_path.resolve().parents[1] / "gym-package.json").is_file() and (PARENT_DIR / "pyproject.toml").is_file():
        package_core = f"-e {shlex.quote(str(PARENT_DIR))}"

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
                install_flags = _get_nemo_gym_install_flags()
                version_spec = _get_nemo_gym_version_spec(is_editable_install)
                core_requirement = package_core or f"nemo-gym{version_spec}"
                install_cmd = (
                    f"""uv pip install {verbose_flag}{uv_pip_python_flag}{install_flags}{core_requirement} && """
                    f"""uv pip install {verbose_flag}{uv_pip_python_flag}--no-sources '-e .' {" ".join(head_server_deps)}"""
                )
        elif has_requirements_txt:
            has_overrides_txt = (dir_path / "overrides.txt").exists()
            override_flag = "--override overrides.txt " if has_overrides_txt else ""
            if is_editable_install:
                install_cmd = f"""uv pip install {verbose_flag}{uv_pip_python_flag}{override_flag}-r requirements.txt {" ".join(head_server_deps)}"""
            else:
                install_flags = _get_nemo_gym_install_flags()
                version_spec = _get_nemo_gym_version_spec(is_editable_install)
                local_extras = re.findall(
                    r"(?m)^\s*(?:-e\s+)?nemo[-_]gym\[([^]]+)\]\s*@\s*\.\./\.\.",
                    (dir_path / "requirements.txt").read_text(),
                )
                extras = sorted({extra.strip() for group in local_extras for extra in group.split(",")})
                extras_spec = f"[{','.join(extras)}]" if extras else ""
                if package_core:
                    package_core = f"-e {shlex.quote(f'{PARENT_DIR}{extras_spec}')}"
                requirements_source = (
                    r"grep -v -E '^[[:space:]]*(-e[[:space:]]+)?nemo[-_]gym(\[[^]]+\])?"
                    r"[[:space:]]*@[[:space:]]*\.\./\.\./?([[:space:]]|$)' requirements.txt"
                )
                if not package_core:
                    requirements_source = f"(echo 'nemo-gym{extras_spec}{version_spec}' && {requirements_source})"
                core_flag = f"{package_core} " if package_core else ""
                install_cmd = (
                    f"""{requirements_source} | """
                    f"""uv pip install {verbose_flag}{uv_pip_python_flag}{install_flags}{override_flag}{core_flag}-r /dev/stdin {" ".join(head_server_deps)}"""
                )
        else:
            raise RuntimeError(
                f"Missing pyproject.toml or requirements.txt for uv venv setup in server dir: {dir_path}"
            )

        prefix_cmd = f" > >(sed 's/^/({prefix}) /') 2> >(sed 's/^/({prefix}) /' >&2)"
        env_setup_cmd = f"{uv_venv_cmd}{prefix_cmd} && source {venv_activate_fpath} && {install_cmd}{prefix_cmd}"

    return f"cd {dir_path} && {env_setup_cmd}"


def run_command(
    command: str,
    working_dir_path: Path,
    server_name: str = "",
    project_root: Path | None = None,
    *,
    global_config_dict: DictConfig | None = None,
    stdout_target: IO[Any] | None = None,
    stderr_target: IO[Any] | None = None,
) -> Popen:
    if global_config_dict is None:
        global_config_dict = get_global_config_dict()

    work_dir = f"{working_dir_path.absolute()}"
    custom_env = environ.copy()
    # Entry points can import sibling components before importing nemo_gym. Resolve explicit
    # package roots before the command changes directory so those imports work at process startup.
    py_path_entries = [work_dir]
    py_path_entries.extend(
        str(Path(root).resolve())
        for root in custom_env.get(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, "").split(os.pathsep)
        if root
    )
    if project_root is not None:
        py_path_entries.append(f"{project_root.absolute()}")
    existing_py_path = custom_env.get("PYTHONPATH")
    if existing_py_path:
        py_path_entries.append(existing_py_path)
    custom_env["PYTHONPATH"] = os.pathsep.join(py_path_entries)

    custom_env["UV_CACHE_DIR"] = global_config_dict[UV_CACHE_DIR_KEY_NAME]

    log_dir = global_config_dict.get(NEMO_GYM_LOG_DIR_KEY_NAME)
    if log_dir:
        safe_name = (server_name or working_dir_path.name).replace("/", "_")
        log_path = Path(log_dir) / f"{safe_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = f"set -o pipefail; ({command}) 2>&1 | tee -a {log_path}"

    redirect_stdout = stdout if stdout_target is None else stdout_target
    redirect_stderr = stderr if stderr_target is None else stderr_target
    return Popen(
        command,
        executable="/bin/bash",
        shell=True,
        env=custom_env,
        stdout=redirect_stdout,
        stderr=redirect_stderr,
    )
