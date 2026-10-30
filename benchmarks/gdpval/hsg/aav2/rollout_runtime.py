#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the rollout's root interpreter, requested modules, and local paths."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path


LOCAL_ROOT = Path("/raid/scratch")
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
    root = local_root.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"path resolves outside node-local storage: {path} -> {resolved}")
    return resolved


def verify_runtime(
    gym_root: Path,
    component_venvs: Path,
    *,
    modules: list[str] | None = None,
    local_root: Path = LOCAL_ROOT,
    sandbox: bool = True,
) -> dict:
    paths = {
        "gym_root": gym_root,
        "component_venvs": component_venvs,
        "cwd": Path.cwd(),
        "executable": Path(sys.executable),
        "base_executable": Path(sys._base_executable),
        "prefix": Path(sys.prefix),
        "base_prefix": Path(sys.base_prefix),
    }
    for name in LOCAL_ENV_PATHS:
        if not sandbox and name in ("GDPVAL_CONTAINER_PATH", "APPTAINER_BIN"):
            continue
        if not os.environ.get(name):
            raise ValueError(f"node-local runtime variable is unset: {name}")
        paths[name] = Path(os.environ[name])
    if sandbox:
        paths["apptainer"] = paths["APPTAINER_BIN"] / "apptainer"
    paths["uv_on_path"] = Path(shutil.which("uv") or "/missing-uv")
    if paths["uv_on_path"].resolve() != paths["MARS_UV"].resolve():
        raise ValueError("uv on PATH differs from the staged executable")
    report = {name: str(assert_node_local(path, local_root=local_root)) for name, path in paths.items()}
    for name in modules or []:
        origin = getattr(importlib.import_module(name), "__file__", None)
        if not origin:
            raise ValueError(f"rollout module has no source file: {name}")
        source = assert_node_local(Path(origin), local_root=local_root)
        if not source.is_relative_to(Path(report["gym_root"])):
            raise ValueError(f"rollout module is outside the staged Gym source: {name} -> {source}")
        report[f"module:{name}"] = str(source)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--component-venvs", type=Path, required=True)
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--without-sandbox", action="store_true", help="Judge-only execution needs no agent sandbox")
    args = parser.parse_args()
    try:
        report = verify_runtime(
            args.gym_root, args.component_venvs, modules=args.module, sandbox=not args.without_sandbox
        )
        print("MARS_ROLLOUT_RUNTIME_PASS " + json.dumps(report))
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit(f"MARS_ROLLOUT_RUNTIME_FAIL: {error}") from error


if __name__ == "__main__":
    main()
