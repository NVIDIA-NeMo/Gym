#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare rollout components with Gym's installer and verify local execution."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


LOCAL_ROOT = Path("/raid/scratch")
ROLLOUT_COMPONENTS = (
    "responses_api_models/vllm_model",
    "responses_api_models/openai_model",
    "resources_servers/gdpval",
    "responses_api_agents/stirrup_agent",
)
LOCAL_ENV_PATHS = (
    "TMPDIR",
    "RAY_TMPDIR",
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "XDG_CACHE_HOME",
    "APPTAINER_TMPDIR",
    "APPTAINER_CACHEDIR",
    "PYTHONPYCACHEPREFIX",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "GDPVAL_REF_FILES_DIR",
    "MARS_UV",
    "GDPVAL_CONTAINER_PATH",
    "APPTAINER_BIN",
)


def assert_node_local(path: Path, *, local_root: Path = LOCAL_ROOT) -> Path:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(local_root.resolve(strict=True)) or resolved == local_root.resolve():
        raise ValueError(f"path resolves outside node-local storage: {path} -> {resolved}")
    return resolved


def _editable_path_placeholders(*, local_root: Path) -> set[str]:
    """Audit registered setuptools finders before accepting their synthetic paths."""
    placeholders = set()
    for name, module in tuple(sys.modules.items()):
        if module is None or not name.startswith("__editable__") or not name.endswith("_finder"):
            continue
        namespace = vars(module)
        finder = namespace.get("_EditableFinder")
        if not isinstance(finder, type) or finder.__module__ != name or finder not in sys.meta_path:
            continue
        assert_node_local(Path(namespace["__file__"]), local_root=local_root)
        mapping, namespaces = namespace.get("MAPPING"), namespace.get("NAMESPACES")
        if not isinstance(mapping, dict) or not isinstance(namespaces, dict):
            raise ValueError(f"invalid editable finder mappings: {name}")
        for path in mapping.values():
            assert_node_local(Path(path), local_root=local_root)
        for paths in namespaces.values():
            if not isinstance(paths, (list, tuple)):
                raise ValueError(f"invalid editable namespace paths: {name}")
            for path in paths:
                assert_node_local(Path(path), local_root=local_root)

        namespace_finder = namespace.get("_EditableNamespaceFinder")
        if not isinstance(namespace_finder, type) or namespace_finder.__module__ != name:
            continue
        hook = vars(namespace_finder).get("_path_hook")
        if not isinstance(hook, classmethod) or hook.__get__(None, namespace_finder) not in sys.path_hooks:
            continue
        placeholder = namespace.get("PATH_PLACEHOLDER")
        if not isinstance(placeholder, str) or not re.fullmatch(
            r"__editable__\.[A-Za-z0-9_.-]+\.finder\.__path_hook__", placeholder
        ):
            raise ValueError(f"invalid editable path placeholder: {name}")
        placeholders.add(placeholder)
    return placeholders


def verify_runtime(gym_root: Path, component_venvs: Path, *, local_root: Path = LOCAL_ROOT) -> dict:
    """Check real interpreter/import paths, including copied-venv escape routes."""
    paths = {
        "gym_root": gym_root,
        "component_venvs": component_venvs,
        "executable": Path(sys.executable),
        "base_executable": Path(sys._base_executable),
        "prefix": Path(sys.prefix),
        "base_prefix": Path(sys.base_prefix),
    }
    for name in LOCAL_ENV_PATHS:
        if not os.environ.get(name):
            raise ValueError(f"node-local runtime variable is unset: {name}")
        paths[name] = Path(os.environ[name])
    paths["apptainer"] = Path(os.environ["APPTAINER_BIN"]) / "apptainer"
    paths["uv_on_path"] = Path(shutil.which("uv") or "/missing-uv")
    if paths["uv_on_path"].resolve() != paths["MARS_UV"].resolve():
        raise ValueError("uv on PATH differs from the staged executable")
    editable_placeholders = _editable_path_placeholders(local_root=local_root)
    for index, entry in enumerate(sys.path):
        if entry in editable_placeholders:
            continue
        path = Path(entry or os.getcwd())
        # CPython lists its optional stdlib zip even when that archive is absent.
        if not path.exists() and path.suffix == ".zip":
            path = path.parent
        paths[f"sys.path[{index}]"] = path
    if os.environ.get("NEMO_GYM_EXTRA_ROOTS"):
        for index, entry in enumerate(os.environ["NEMO_GYM_EXTRA_ROOTS"].split(os.pathsep)):
            paths[f"extra_root[{index}]"] = Path(entry)
    report = {name: str(assert_node_local(path, local_root=local_root)) for name, path in paths.items()}
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        namespace = vars(module)
        origin = namespace.get("__file__")
        if origin and not origin.startswith("<"):
            assert_node_local(Path(origin), local_root=local_root)
        for entry in namespace.get("__path__", ()):
            if entry not in editable_placeholders:
                assert_node_local(Path(entry), local_root=local_root)
    for path in component_venvs.rglob(".venv"):
        assert_node_local(path, local_root=local_root)
        assert_node_local(path / "bin/python", local_root=local_root)
    return report


def _component_setup_config(component_venvs: Path):
    from omegaconf import open_dict

    from nemo_gym.global_config import GlobalConfigDictParser

    # Let the selected Gym revision derive its own parent-version constraints,
    # including Ray, instead of maintaining a second dependency policy here.
    cache_dir = os.environ["UV_CACHE_DIR"]
    try:
        config = GlobalConfigDictParser().parse_no_environment()
    finally:
        os.environ["UV_CACHE_DIR"] = cache_dir
    with open_dict(config):
        config.uv_venv_dir = str(component_venvs)
        config.uv_cache_dir = cache_dir
        config.python_version = sys.executable
        config.uv_pip_set_python = True
        config.skip_venv_if_present = False
    return config


def prepare_components(gym_root: Path, component_venvs: Path, *, local_root: Path = LOCAL_ROOT) -> None:
    """Install and probe fresh component environments before launching services."""
    gym_root = assert_node_local(gym_root, local_root=local_root)
    component_venvs = assert_node_local(component_venvs, local_root=local_root)
    assert_node_local(Path(sys.executable), local_root=local_root)
    # Gym's historical setup command interpolates these paths into shell text.
    # Reject unsupported characters before asking it to compose that command.
    for path in (gym_root, component_venvs, Path(sys.executable)):
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path)):
            raise ValueError(f"unsupported component setup path: {path}")

    from nemo_gym.cli import setup_command

    config = _component_setup_config(component_venvs)
    helper = Path(__file__).resolve()
    plans = []
    for component in ROLLOUT_COMPONENTS:
        directory = assert_node_local(gym_root / component, local_root=local_root)
        # The older pinned rollout revision predates get_venv_path. Its setup
        # command uses this same configured <type>/<name>/.venv layout.
        resolver = getattr(setup_command, "get_venv_path", None)
        venv = resolver(directory, config) if resolver else component_venvs / component / ".venv"
        venv.parent.mkdir(parents=True, exist_ok=True)
        assert_node_local(venv.parent, local_root=local_root)
        if venv.exists() or venv.is_symlink():
            raise ValueError(f"refusing an existing component environment: {venv}")
        plans.append((component, directory, venv))

    for component, directory, venv in plans:
        command = setup_command.setup_env_command(directory, config, directory.name)
        environment = {**os.environ, "PYTHONPATH": f"{directory}:{gym_root}", "UV_CACHE_DIR": config.uv_cache_dir}
        subprocess.run(["bash", "-e", "-o", "pipefail", "-c", command], cwd=directory, env=environment, check=True)
        interpreter = venv / "bin/python"
        assert_node_local(interpreter, local_root=local_root)
        subprocess.run(
            [
                str(interpreter),
                str(helper),
                "verify",
                "--gym-root",
                str(gym_root),
                "--component-venvs",
                str(component_venvs),
                "--module",
                "nemo_gym",
                "--module",
                "app",
            ],
            cwd=directory,
            env=environment,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--gym-root", type=Path, required=True)
    verify.add_argument("--component-venvs", type=Path, required=True)
    verify.add_argument("--module", action="append", default=[])
    prepare = commands.add_parser("prepare-components")
    prepare.add_argument("--gym-root", type=Path, required=True)
    prepare.add_argument("--component-venvs", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare-components":
            prepare_components(args.gym_root, args.component_venvs)
        else:
            for name in args.module:
                importlib.import_module(name)
            print("MARS_ROLLOUT_RUNTIME_PASS " + json.dumps(verify_runtime(args.gym_root, args.component_venvs)))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"MARS_ROLLOUT_RUNTIME_FAIL: {error}") from error


if __name__ == "__main__":
    main()
