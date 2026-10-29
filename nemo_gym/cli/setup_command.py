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
import importlib.util
import os
import platform
import re
import shlex
import sys
from os import environ
from pathlib import Path
from subprocess import Popen
from sys import stderr, stdout
from typing import IO, Any, Mapping

from omegaconf import DictConfig
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, PARENT_DIR
from nemo_gym.global_config import (
    HEAD_SERVER_CONSTRAINTS_KEY_NAME,
    HEAD_SERVER_DEPS_KEY_NAME,
    NEMO_GYM_LOG_DIR_KEY_NAME,
    PIP_INSTALL_VERBOSE_KEY_NAME,
    PYTHON_VERSION_KEY_NAME,
    SKIP_VENV_IF_PRESENT_KEY_NAME,
    UV_CACHE_DIR_KEY_NAME,
    UV_LOCK_TIMEOUT_KEY_NAME,
    UV_PIP_SET_PYTHON_KEY_NAME,
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


# A requirements line that installs Gym from the checkout root, e.g. `-e nemo-gym[dev,sandbox] @ ../../`.
_LOCAL_NEMO_GYM_REQUIREMENT_RE = re.compile(
    r"^\s*(?:-e\s+)?nemo[-_]gym\s*(?:\[(?P<extras>[^\]]*)\])?\s*@\s*\.\./\.\./?\s*(?:#.*)?$"
)


def _local_nemo_gym_extras(requirements_fpath: Path) -> str:
    """The extras a server requests on its local Gym requirement, formatted as `[a,b]`, or "" for none.

    Installs from PyPI replace the server's `nemo-gym[...] @ ../../` line with a `nemo-gym` requirement, so
    its extras must be carried over or their dependencies are never installed.
    """
    extras: list[str] = []
    for line in requirements_fpath.read_text().splitlines():
        match = _LOCAL_NEMO_GYM_REQUIREMENT_RE.match(line)
        if match is None or not match.group("extras"):
            continue
        for extra in match.group("extras").split(","):
            extra = extra.strip()
            if extra and extra not in extras:
                extras.append(extra)
    return f"[{','.join(extras)}]" if extras else ""


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
    head_server_deps = [str(dependency) for dependency in global_config_dict[HEAD_SERVER_DEPS_KEY_NAME]]
    head_server_dep_args = shlex.join(head_server_deps)
    # Pins applied only to packages a server installs anyway, e.g. the parent's exact Ray version.
    head_server_constraints = [
        str(constraint) for constraint in global_config_dict.get(HEAD_SERVER_CONSTRAINTS_KEY_NAME) or []
    ]
    constraint_args = " ".join(shlex.quote(c) for c in head_server_constraints)
    constraint_flag = f"-c <(printf '%s\\n' {constraint_args}) " if head_server_constraints else ""
    # A constraint cannot replace an exact stale pin in a server dependency (for example
    # another Ray version), which makes the solve impossible or splits one Ray cluster across
    # versions. Passing the same pins as overrides replaces such a pin and, like a constraint,
    # never adds a package the server does not require. head_server_deps stay plain
    # requirements, so a server pin that conflicts with them still fails the install loudly.
    parent_override_flag = f"--overrides <(printf '%s\\n' {constraint_args}) " if head_server_constraints else ""

    venv_path = get_venv_path(dir_path, global_config_dict)
    venv_python_fpath = venv_path / "bin/python"
    venv_activate_fpath = venv_path / "bin/activate"
    skip_venv_if_present = global_config_dict[SKIP_VENV_IF_PRESENT_KEY_NAME]
    should_skip_venv_setup = bool(skip_venv_if_present) and venv_python_fpath.exists() and venv_activate_fpath.exists()
    activate_cmd = f"source {shlex.quote(str(venv_activate_fpath))}"
    if should_skip_venv_setup:
        # Reuse existing environments, including prebuilt venvs without Gym's
        # completion marker or writable setup-lock directories.
        return f"cd {shlex.quote(str(dir_path))} && {activate_cmd}"

    python_request_arg = shlex.quote(
        _server_python_request(dir_path, str(global_config_dict[PYTHON_VERSION_KEY_NAME]))
    )

    # `uv pip --project` has no effect. A server-local uv.toml passed via
    # --config-file is the supported way to keep an enclosing workspace (for
    # example /opt/nemo-rl) from contributing unrelated resolver policy.
    uv_config_path = dir_path / "uv.toml"
    uv_config_flag = f"--config-file {shlex.quote(str(uv_config_path.resolve()))} " if uv_config_path.exists() else ""
    uv_venv_cmd = (
        f"uv venv {uv_config_flag}--seed --allow-existing --python {python_request_arg} {shlex.quote(str(venv_path))}"
    )

    # explicitly set python path if specified. In Google colab, gym env start fails due to uv pip install falls back to system python (/usr) without this and errors.
    # not needed for most clusters. should be safe in all scenarios, but only minimally tested outside of colab.
    # see discussion and examples here: https://github.com/NVIDIA-NeMo/Gym/pull/526#issuecomment-3676230383
    uv_pip_set_python = global_config_dict.get(UV_PIP_SET_PYTHON_KEY_NAME, False)
    uv_pip_python_flag = f"--python {shlex.quote(str(venv_python_fpath))} " if uv_pip_set_python else ""

    verbose_flag = "-v " if global_config_dict.get(PIP_INSTALL_VERBOSE_KEY_NAME) else ""

    # A server is a separate process with its own venv. Allow standard uv
    # configuration in pyproject.toml or uv.toml without leaking those choices
    # into sibling server venvs through process-wide UV_* variables.
    server_uv_flags = f"{uv_config_flag}{parent_override_flag}"

    is_editable_install = (dir_path.resolve() / "../../pyproject.toml").exists()

    try:
        has_pyproject_toml = (dir_path / "pyproject.toml").exists()
        has_requirements_txt = (dir_path / "requirements.txt").exists()
        if has_pyproject_toml and has_requirements_txt:
            raise RuntimeError(
                f"Found both pyproject.toml and requirements.txt for uv venv setup in server dir: {dir_path}. Please only use one or the other!"
            )
        elif has_pyproject_toml:
            if is_editable_install:
                install_cmd = f"""uv pip install {server_uv_flags}{verbose_flag}{uv_pip_python_flag}{constraint_flag}'-e .' {head_server_dep_args}"""
            else:
                # install nemo-gym from pypi instead of relative path in pyproject.toml
                # with support for pre-releases, custom indexes, and version pinning
                install_flags = _get_nemo_gym_install_flags()
                version_spec = _get_nemo_gym_version_spec(is_editable_install)
                install_cmd = (
                    f"""uv pip install {server_uv_flags}{verbose_flag}{uv_pip_python_flag}{install_flags}{constraint_flag}nemo-gym{version_spec} && """
                    f"""uv pip install {server_uv_flags}{verbose_flag}{uv_pip_python_flag}{constraint_flag}--no-sources '-e .' {head_server_dep_args}"""
                )
        elif has_requirements_txt:
            has_overrides_txt = (dir_path / "overrides.txt").exists()
            override_flag = "--override overrides.txt " if has_overrides_txt else ""
            if is_editable_install:
                install_cmd = f"""uv pip install {server_uv_flags}{verbose_flag}{uv_pip_python_flag}{constraint_flag}{override_flag}-r requirements.txt {head_server_dep_args}"""
            else:
                # install nemo-gym from pypi instead of relative path in requirements.txt
                # with support for pre-releases, custom indexes, and version pinning
                install_flags = _get_nemo_gym_install_flags()
                version_spec = _get_nemo_gym_version_spec(is_editable_install)
                extras = _local_nemo_gym_extras(dir_path / "requirements.txt")
                install_cmd = (
                    f"""(echo 'nemo-gym{extras}{version_spec}' && grep -v -F '../..' requirements.txt) | """
                    f"""uv pip install {server_uv_flags}{verbose_flag}{uv_pip_python_flag}{install_flags}{constraint_flag}{override_flag}-r /dev/stdin {head_server_dep_args}"""
                )
        else:
            raise RuntimeError(
                f"Missing pyproject.toml or requirements.txt for uv venv setup in server dir: {dir_path}"
            )

        prefix_cmd = f" > >(sed 's/^/({prefix}) /') 2> >(sed 's/^/({prefix}) /' >&2)"
        install_command = f"{uv_venv_cmd}{prefix_cmd} && {activate_cmd} && {install_cmd}{prefix_cmd}"

    except RuntimeError as error:
        if not skip_venv_if_present:
            raise
        # A ready environment needs no manifest. Fail only if the locked
        # readiness check determines that installation is required.
        install_command = f"printf '%s\\n' {shlex.quote(str(error))} >&2; exit 1"

    # When setup is needed, serialize installers and recheck readiness under
    # the lock in case another server completes installation before this one.
    setup_command = [
        sys.executable,
        # Found by module name rather than next to this file, because the script may live in another package.
        # The script itself runs without importing Gym.
        importlib.util.find_spec("nemo_gym.cli._venv_setup").origin,
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
