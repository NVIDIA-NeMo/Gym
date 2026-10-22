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
import platform
import shlex
import sys
from os import environ
from pathlib import Path
from subprocess import Popen
from sys import stderr, stdout
from typing import IO, Any, Mapping

from omegaconf import DictConfig, ListConfig
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, PARENT_DIR
from nemo_gym.global_config import (
    COMPONENT_EXTRAS_KEY_NAME,
    HEAD_SERVER_CONSTRAINTS_KEY_NAME,
    HEAD_SERVER_DEPS_KEY_NAME,
    NEMO_GYM_LOG_DIR_KEY_NAME,
    PIP_INSTALL_VERBOSE_KEY_NAME,
    PYTHON_VERSION_KEY_NAME,
    SKIP_VENV_IF_PRESENT_KEY_NAME,
    UV_CACHE_DIR_KEY_NAME,
    UV_LOCK_TIMEOUT_KEY_NAME,
    UV_VENV_DIR_KEY_NAME,
    get_global_config_dict,
)


NEMO_GYM_ALLOWED_COMPONENT_ROOTS_ENV_VAR_NAME = "NEMO_GYM_ALLOWED_COMPONENT_ROOTS"


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


def get_venv_path(dir_path: Path, global_config_dict: DictConfig) -> Path:
    """Resolve the venv a server runs from: ``uv_venv_dir/<type>/<name>/.venv``, else ``<server dir>/.venv``.

    Callers need this to launch a server with the venv's own interpreter rather than trusting whatever
    ``bin/activate`` puts on PATH. A venv copied or moved after creation keeps the *original* prefix
    hard-coded in ``bin/activate``, so sourcing it silently hands the server a different interpreter than
    the one ``uv_venv_dir`` asked for.
    """
    root_venv_path = Path(global_config_dict[UV_VENV_DIR_KEY_NAME])
    if root_venv_path.resolve() != PARENT_DIR.resolve():
        return Path(root_venv_path, *dir_path.parts[-2:], ".venv").absolute()
    return (dir_path / ".venv").absolute()


def _python_request_admits(request: str, version: str) -> bool:
    """Whether a uv Python request admits ``version``.

    The request is a specifier such as ``>=3.13,<3.14`` or a version such as ``3.13``, optionally prefixed with
    ``cpython@`` or ``cpython``. Requests this cannot interpret, such as ``pypy@3.10`` or the free-threaded
    ``3.13+freethreaded``, admit nothing and are left to uv.
    """
    try:
        candidate = Version(version)
    except InvalidVersion:
        return False
    if platform.python_implementation() == "CPython":
        for prefix in ("cpython@", "cpython"):
            if request.startswith(prefix):
                request = request.removeprefix(prefix)
                break
    try:
        return candidate in SpecifierSet(request)
    except InvalidSpecifier:
        pass
    try:
        requested = Version(request)
    except InvalidVersion:
        return False
    # PEP 440 reads uv's variant suffix (``+freethreaded``, ``+debug``) as a local label; a plain parent build
    # does not satisfy a variant or pre-release request.
    if requested.local is not None or requested.is_prerelease:
        return False
    return candidate.release[: len(requested.release)] == requested.release


def _server_python_request(dir_path: Path, parent_python: str) -> str:
    """The interpreter a server venv is built with: the parent's, unless the server's ``.python-version`` excludes it.

    A range such as ``>=3.13.14,<3.14`` would otherwise let uv pick a different patch release than the parent's,
    and Ray refuses to connect processes whose Python patch versions differ.
    """
    python_version_path = dir_path / ".python-version"
    if not python_version_path.exists():
        return parent_python
    # Like uv, skip blank and comment lines and use the first request.
    lines = python_version_path.read_text(encoding="utf-8").splitlines()
    request = next((line.strip() for line in lines if line.strip() and not line.strip().startswith("#")), "")
    if not request:
        raise RuntimeError(f"Empty Python version in {python_version_path}")
    return parent_python if _python_request_admits(request, parent_python) else request


def setup_env_command(dir_path: Path, global_config_dict: DictConfig, prefix: str) -> str:
    venv_path = get_venv_path(dir_path, global_config_dict)
    venv_python_fpath = venv_path / "bin/python"
    venv_activate_fpath = venv_path / "bin/activate"
    skip_venv_if_present = global_config_dict[SKIP_VENV_IF_PRESENT_KEY_NAME]
    activate_cmd = f"source {shlex.quote(str(venv_activate_fpath))}"
    if skip_venv_if_present and venv_python_fpath.exists() and venv_activate_fpath.exists():
        return f"cd {shlex.quote(str(dir_path))} && {activate_cmd}"

    try:
        for filename in ("pyproject.toml", "uv.lock"):
            if not (dir_path / filename).is_file():
                raise RuntimeError(
                    f"Missing {filename} for locked setup in {dir_path}. "
                    "Declare dependencies in pyproject.toml, run `uv lock` in the component directory, "
                    "and commit both files. Setup never resolves unlocked requirements."
                )
        sync_command = [
            sys.executable,
            str(Path(__file__).with_name("_locked_env.py")),
            "--project",
            str(dir_path.absolute()),
            "--venv",
            str(venv_path),
            "--python",
            _server_python_request(dir_path, str(global_config_dict[PYTHON_VERSION_KEY_NAME])),
        ]
        extras_by_component = global_config_dict.get(COMPONENT_EXTRAS_KEY_NAME) or {}
        if not isinstance(extras_by_component, Mapping):
            raise RuntimeError("component_extras must map component paths to lists of extra names.")
        component_path = "/".join(dir_path.parts[-2:])
        extras = extras_by_component.get(component_path, [])
        if not isinstance(extras, (list, ListConfig)) or any(
            not isinstance(extra, str) or not extra for extra in extras
        ):
            raise RuntimeError(f"component_extras[{component_path!r}] must be a list of nonempty extra names.")
        for extra in extras:
            sync_command.extend(["--extra", extra])
        for requirement in global_config_dict[HEAD_SERVER_DEPS_KEY_NAME]:
            sync_command.extend(["--requirement", requirement])
        for constraint in global_config_dict.get(HEAD_SERVER_CONSTRAINTS_KEY_NAME) or []:
            sync_command.extend(["--constraint", constraint])
        if global_config_dict.get(PIP_INSTALL_VERBOSE_KEY_NAME):
            sync_command.append("--verbose")
        prefix_cmd = f" > >(sed 's/^/({prefix}) /') 2> >(sed 's/^/({prefix}) /' >&2)"
        install_command = f"{shlex.join(sync_command)}{prefix_cmd}"
    except RuntimeError as error:
        if not skip_venv_if_present:
            raise
        install_command = f"printf '%s\\n' {shlex.quote(str(error))} >&2; exit 1"

    # When setup is needed, serialize installers and recheck readiness under
    # the lock in case another server completes installation before this one.
    setup_command = [
        sys.executable,
        str(Path(__file__).with_name("_venv_setup.py")),
        "--venv",
        str(venv_path),
        "--command",
        install_command,
    ]
    if skip_venv_if_present:
        setup_command.append("--skip-if-ready")

    return f"cd {shlex.quote(str(dir_path))} && {shlex.join(setup_command)} && {activate_cmd}"


def run_command(
    command: str,
    working_dir_path: Path,
    server_name: str = "",
    project_root: Path | None = None,
    *,
    global_config_dict: DictConfig | None = None,
    stdout_target: IO[Any] | None = None,
    stderr_target: IO[Any] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> Popen:
    if global_config_dict is None:
        global_config_dict = get_global_config_dict()

    work_dir = f"{working_dir_path.absolute()}"
    custom_env = environ.copy()
    if extra_env is not None:
        custom_env.update(extra_env)
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
    # Servers start concurrently and contend for the lock on that shared cache, so the wait has to
    # cover a cold install of the slowest one rather than uv's 300s default.
    uv_lock_timeout = global_config_dict.get(UV_LOCK_TIMEOUT_KEY_NAME)
    if uv_lock_timeout is not None:
        custom_env["UV_LOCK_TIMEOUT"] = str(uv_lock_timeout)

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
