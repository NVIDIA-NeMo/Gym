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
import hashlib
import importlib.metadata
import json
import os
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


NEMO_GYM_ALLOWED_COMPONENT_ROOTS_ENV_VAR_NAME = "NEMO_GYM_ALLOWED_COMPONENT_ROOTS"
PARENT_RUNTIME_OVERRIDES_FILENAME = ".nemo-gym-parent-runtime-overrides.txt"
ENVIRONMENT_IDENTITY_FILENAME = ".nemo-gym-environment-identity"
_ENVIRONMENT_INPUT_FILENAMES = (
    "requirements.txt",
    "pyproject.toml",
    "overrides.txt",
    "uv-overrides.txt",
    "uv.toml",
    "uv-torch-backend.txt",
    "uv-python-version.txt",
    "uv-managed-python.txt",
)


def _validate_component_working_dir(
    working_dir_path: Path,
    *,
    server_name: str,
    environment: dict[str, str],
) -> None:
    """Reject a server resolved outside scheduler-approved component roots."""

    configured_roots = environment.get(NEMO_GYM_ALLOWED_COMPONENT_ROOTS_ENV_VAR_NAME, "")
    if not configured_roots:
        return

    working_dir = working_dir_path.resolve()
    allowed_roots = [Path(value).resolve() for value in configured_roots.split(os.pathsep) if value]
    if not any(working_dir.is_relative_to(root) for root in allowed_roots):
        raise RuntimeError(
            "Gym component path is outside scheduler-approved roots; refusing "
            "to start a potentially stale server. "
            f"server={server_name or working_dir_path.name}, "
            f"working_dir={working_dir}, allowed_roots={allowed_roots}"
        )
    print(
        "NEMO_GYM_COMPONENT_PATH_GATE|"
        f"server={server_name or working_dir_path.name}|"
        f"working_dir={working_dir}|allowed_roots={configured_roots}",
        flush=True,
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


def _server_environment_identity(
    dir_path: Path,
    *,
    python_request: str,
    head_server_deps: list[str],
    is_editable_install: bool,
    nemo_gym_version_spec: str,
) -> str:
    """Fingerprint every input that determines a managed server environment."""

    component_dir = dir_path.resolve()
    input_files: dict[str, str] = {}
    for filename in _ENVIRONMENT_INPUT_FILENAMES:
        path = component_dir / filename
        if path.is_file():
            input_files[filename] = hashlib.sha256(path.read_bytes()).hexdigest()

    parent_project = (component_dir / "../../pyproject.toml").resolve()
    if parent_project.is_file():
        input_files["../../pyproject.toml"] = hashlib.sha256(parent_project.read_bytes()).hexdigest()

    payload = {
        "schema_version": 1,
        "component_dir": str(component_dir),
        "head_server_deps": head_server_deps,
        "input_files": input_files,
        "is_editable_install": is_editable_install,
        "nemo_gym_version_spec": nemo_gym_version_spec,
        "python_request": python_request,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def setup_env_command(dir_path: Path, global_config_dict: DictConfig, prefix: str) -> str:
    head_server_deps = [str(dependency) for dependency in global_config_dict[HEAD_SERVER_DEPS_KEY_NAME]]
    head_server_dep_args = shlex.join(head_server_deps)

    venv_path = get_venv_path(dir_path, global_config_dict)
    parent_runtime_overrides_path = venv_path / PARENT_RUNTIME_OVERRIDES_FILENAME
    environment_identity_path = venv_path / ENVIRONMENT_IDENTITY_FILENAME
    parent_runtime_overrides_cmd = (
        f"printf '%s\\n' {head_server_dep_args} > {shlex.quote(str(parent_runtime_overrides_path))}"
    )

    python_request = str(global_config_dict[PYTHON_VERSION_KEY_NAME])
    python_request_arg = python_request
    python_venv_flags = ""
    python_version_path = dir_path / "uv-python-version.txt"
    if python_version_path.exists():
        python_request = python_version_path.read_text(encoding="utf-8").strip()
        if not python_request:
            raise RuntimeError(f"Empty Python version in {python_version_path}")
        python_request_arg = shlex.quote(python_request)
    managed_python_path = dir_path / "uv-managed-python.txt"
    if managed_python_path.exists():
        managed_python = managed_python_path.read_text(encoding="utf-8").strip().lower()
        if managed_python != "true":
            raise RuntimeError(f"Expected 'true' in {managed_python_path}, got {managed_python!r}")
        python_venv_flags = "--managed-python "

    is_editable_install = (dir_path.resolve() / "../../pyproject.toml").exists()
    nemo_gym_version_spec = _get_nemo_gym_version_spec(is_editable_install)
    environment_identity = _server_environment_identity(
        dir_path,
        python_request=python_request,
        head_server_deps=head_server_deps,
        is_editable_install=is_editable_install,
        nemo_gym_version_spec=nemo_gym_version_spec,
    )
    environment_identity_tmp_path = f"{shlex.quote(str(environment_identity_path))}.tmp.$$"
    clear_environment_identity_cmd = f"rm -f {shlex.quote(str(environment_identity_path))}"
    commit_environment_identity_cmd = (
        f"printf '%s\\n' {environment_identity} > {environment_identity_tmp_path} && "
        f"mv {environment_identity_tmp_path} {shlex.quote(str(environment_identity_path))}"
    )

    # `uv pip --project` has no effect. A server-local uv.toml passed via
    # --config-file is the supported way to keep an enclosing workspace (for
    # example /opt/nemo-rl) from contributing unrelated resolver policy.
    uv_config_path = dir_path / "uv.toml"
    uv_config_flag = f"--config-file {shlex.quote(str(uv_config_path.resolve()))} " if uv_config_path.exists() else ""
    uv_venv_cmd = (
        f"uv venv {uv_config_flag}--seed --allow-existing {python_venv_flags}--python {python_request_arg} {venv_path}"
    )

    venv_python_fpath = venv_path / "bin/python"
    venv_activate_fpath = venv_path / "bin/activate"
    skip_venv_if_present = global_config_dict[SKIP_VENV_IF_PRESENT_KEY_NAME]
    try:
        recorded_environment_identity = environment_identity_path.read_text(encoding="utf-8").strip()
    except OSError:
        recorded_environment_identity = ""
    should_skip_venv_setup = (
        bool(skip_venv_if_present)
        and venv_python_fpath.exists()
        and venv_activate_fpath.exists()
        and recorded_environment_identity == environment_identity
    )

    # explicitly set python path if specified. In Google colab, gym env start fails due to uv pip install falls back to system python (/usr) without this and errors.
    # not needed for most clusters. should be safe in all scenarios, but only minimally tested outside of colab.
    # see discussion and examples here: https://github.com/NVIDIA-NeMo/Gym/pull/526#issuecomment-3676230383
    uv_pip_set_python = global_config_dict.get(UV_PIP_SET_PYTHON_KEY_NAME, False)
    uv_pip_python_flag = f"--python {venv_python_fpath} " if uv_pip_set_python else ""

    verbose_flag = "-v " if global_config_dict.get(PIP_INSTALL_VERBOSE_KEY_NAME) else ""

    # A server is a separate process with its own venv. Allow it to select its
    # Python, dependency metadata, and Torch backend without leaking those
    # choices into sibling server venvs through process-wide UV_* variables.
    # Parent-sensitive packages such as Ray are both direct requirements and
    # resolver overrides. A direct requirement alone cannot replace an exact
    # stale transitive pin in a server dependency, which can either make the
    # solve impossible or split one Ray cluster across incompatible versions.
    # Materialize the dynamic authority inside the writable venv rather than
    # baking the current parent version into every server's source tree.
    server_uv_flags = f"{uv_config_flag}--overrides {shlex.quote(str(parent_runtime_overrides_path))} "
    overrides_path = dir_path / "uv-overrides.txt"
    if overrides_path.exists():
        server_uv_flags += f"--overrides {shlex.quote(str(overrides_path.resolve()))} "
    torch_backend_path = dir_path / "uv-torch-backend.txt"
    if torch_backend_path.exists():
        torch_backend = torch_backend_path.read_text(encoding="utf-8").strip()
        valid_torch_backends = {"auto", "cpu", "cu126", "cu128", "cu129", "cu130", "xpu"}
        if torch_backend not in valid_torch_backends:
            raise RuntimeError(f"Invalid Torch backend in {torch_backend_path}: {torch_backend!r}")
        server_uv_flags += f"--torch-backend {torch_backend} "

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
                install_cmd = f"""uv pip install {server_uv_flags}{verbose_flag}{uv_pip_python_flag}'-e .' {head_server_dep_args}"""
            else:
                # install nemo-gym from pypi instead of relative path in pyproject.toml
                # with support for pre-releases, custom indexes, and version pinning
                install_flags = _get_nemo_gym_install_flags()
                version_spec = nemo_gym_version_spec
                install_cmd = (
                    f"""uv pip install {server_uv_flags}{verbose_flag}{uv_pip_python_flag}{install_flags}nemo-gym{version_spec} && """
                    f"""uv pip install {server_uv_flags}{verbose_flag}{uv_pip_python_flag}--no-sources '-e .' {head_server_dep_args}"""
                )
        elif has_requirements_txt:
            has_overrides_txt = (dir_path / "overrides.txt").exists()
            override_flag = "--override overrides.txt " if has_overrides_txt else ""
            if is_editable_install:
                install_cmd = f"""uv pip install {server_uv_flags}{verbose_flag}{uv_pip_python_flag}{override_flag}-r requirements.txt {head_server_dep_args}"""
            else:
                # install nemo-gym from pypi instead of relative path in requirements.txt
                # with support for pre-releases, custom indexes, and version pinning
                install_flags = _get_nemo_gym_install_flags()
                version_spec = nemo_gym_version_spec
                install_cmd = (
                    f"""(echo 'nemo-gym{version_spec}' && grep -v -F '../..' requirements.txt) | """
                    f"""uv pip install {server_uv_flags}{verbose_flag}{uv_pip_python_flag}{install_flags}{override_flag}-r /dev/stdin {head_server_dep_args}"""
                )
        else:
            raise RuntimeError(
                f"Missing pyproject.toml or requirements.txt for uv venv setup in server dir: {dir_path}"
            )

        prefix_cmd = f" > >(sed 's/^/({prefix}) /') 2> >(sed 's/^/({prefix}) /' >&2)"
        env_setup_cmd = (
            f"{clear_environment_identity_cmd} && {uv_venv_cmd}{prefix_cmd} && "
            f"{parent_runtime_overrides_cmd} && source {venv_activate_fpath} && "
            f"{install_cmd}{prefix_cmd} && {commit_environment_identity_cmd}"
        )

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
    _validate_component_working_dir(
        working_dir_path,
        server_name=server_name,
        environment=custom_env,
    )
    # The server dir on PYTHONPATH lets `import app` work. When a caller passes `project_root` (the
    # dir containing resources_servers/, responses_api_agents/, ...), it's added so generated
    # `resources_servers.<name>.app`-style imports resolve from outside a repo checkout — opt-in, so
    # this generic helper doesn't bake a layout assumption in for its other callers.
    py_path_entries = [work_dir]
    if project_root is not None:
        absolute_project_root = f"{project_root.absolute()}"
        py_path_entries.append(absolute_project_root)
        # ``nemo_gym`` normalizes sys.path at import time so component discovery has one
        # precedence contract. Merely prepending PYTHONPATH is therefore insufficient:
        # an older editable Gym install can still win after that normalization. Mark the
        # already-resolved component root as an explicit extra root too; extra roots are
        # intentionally highest priority in both discovery and Python imports.
        existing_extra_roots = custom_env.get(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME)
        extra_roots = [absolute_project_root]
        if existing_extra_roots:
            extra_roots.append(existing_extra_roots)
        custom_env[NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME] = os.pathsep.join(extra_roots)
    existing_py_path = custom_env.get("PYTHONPATH")
    if existing_py_path:
        py_path_entries.append(existing_py_path)
    custom_env["PYTHONPATH"] = ":".join(py_path_entries)

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
