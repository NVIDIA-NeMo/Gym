# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the OSWorld agent's explicitly installed runtime dependencies."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


OSWORLD_AGENT_RELATIVE_DIR = Path("responses_api_agents/osworld_agent")


@dataclass(frozen=True)
class RuntimeDependency:
    """One dependency excluded from Gym's default managed environments."""

    distribution: str
    import_name: str
    specifier: str

    @property
    def requirement(self) -> str:
        return f"{self.distribution}{self.specifier}"


OPTIONAL_RUNTIME_DEPENDENCIES = (
    # OpenCV 4.8 wheels use NumPy's 1.x ABI. This repeats the normal agent
    # requirement so the opt-in installer can repair a drifted managed venv.
    RuntimeDependency("numpy", "numpy", "<2"),
    RuntimeDependency("cryptography", "cryptography", "~=46.0"),
    RuntimeDependency("opencv-python-headless", "cv2", "~=4.8.1.78"),
    RuntimeDependency("torchvision", "torchvision", "==0.26.0"),
)


def managed_agent_venv_path(gym_root: Path, venv_root: Path | None = None) -> Path:
    """Return Gym's managed venv path for the OSWorld agent server."""

    gym_root = gym_root.expanduser().resolve()
    resolved_venv_root = gym_root if venv_root is None else venv_root.expanduser().resolve()
    agent_dir = gym_root / OSWORLD_AGENT_RELATIVE_DIR
    if resolved_venv_root == gym_root:
        return agent_dir / ".venv"
    return resolved_venv_root / OSWORLD_AGENT_RELATIVE_DIR / ".venv"


def managed_agent_venv_from_env(gym_root: Path, env_file: Path) -> Path:
    """Resolve the OSWorld agent venv from a prepared Gym ``env.yaml``."""

    env_file = env_file.expanduser().resolve()
    payload: Any = yaml.safe_load(env_file.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Gym environment must be a YAML mapping: {env_file}")

    raw_venv_root = payload.get("uv_venv_dir")
    if raw_venv_root is None:
        return managed_agent_venv_path(gym_root)
    if not isinstance(raw_venv_root, str) or not raw_venv_root.strip():
        raise ValueError(f"uv_venv_dir must be a non-empty path string in {env_file}")
    if "${" in raw_venv_root:
        raise ValueError(
            f"uv_venv_dir contains an unresolved interpolation in {env_file}; "
            "set OSWORLD_AGENT_VENV to the resolved agent venv path"
        )

    venv_root = Path(raw_venv_root).expanduser()
    if not venv_root.is_absolute():
        venv_root = env_file.parent / venv_root
    return managed_agent_venv_path(gym_root, venv_root)


def validate_optional_runtime_dependencies(
    dependencies: Sequence[RuntimeDependency] = OPTIONAL_RUNTIME_DEPENDENCIES,
) -> tuple[str, ...]:
    """Return actionable problems in the current Python environment."""

    problems: list[str] = []
    version_ready: list[RuntimeDependency] = []
    invalid_distributions: set[str] = set()
    for dependency in dependencies:
        try:
            installed_version = importlib.metadata.version(dependency.distribution)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{dependency.requirement}: package is not installed")
            invalid_distributions.add(dependency.distribution)
            continue

        try:
            installed = Version(installed_version)
        except InvalidVersion:
            problems.append(
                f"{dependency.requirement}: installed version {installed_version!r} is not a valid version"
            )
            invalid_distributions.add(dependency.distribution)
        else:
            if installed not in SpecifierSet(dependency.specifier):
                problems.append(
                    f"{dependency.requirement}: installed version {installed_version!r} "
                    "does not satisfy the requirement"
                )
                invalid_distributions.add(dependency.distribution)
            else:
                version_ready.append(dependency)

    for dependency in version_ready:
        # Importing OpenCV against NumPy 2 prints a native ABI traceback before
        # Python can catch the ImportError. The NumPy version error is already
        # sufficient and more actionable, so avoid that noisy dependent import.
        if dependency.import_name == "cv2" and "numpy" in invalid_distributions:
            continue
        try:
            importlib.import_module(dependency.import_name)
        except Exception as exc:  # noqa: BLE001 - binary import errors must be reported as readiness failures.
            problems.append(
                f"{dependency.requirement}: import {dependency.import_name!r} failed ({type(exc).__name__}: {exc})"
            )
    return tuple(problems)


def require_optional_runtime_dependencies(
    *,
    venv_path: Path | None = None,
    installer: Path | None = None,
) -> None:
    """Fail an actual OSWorld agent startup with copyable remediation."""

    problems = validate_optional_runtime_dependencies()
    if not problems:
        return

    resolved_venv = Path(sys.prefix).resolve() if venv_path is None else venv_path.expanduser().resolve()
    resolved_installer = (
        Path(__file__).resolve().with_name("install_optional_runtime_deps.sh")
        if installer is None
        else installer.expanduser().resolve()
    )
    details = "\n".join(f"  - {problem}" for problem in problems)
    command = shlex.join(["bash", str(resolved_installer), str(resolved_venv)])
    raise RuntimeError(
        "OSWorld optional runtime dependencies are not ready in this agent venv:\n"
        f"{details}\n"
        "Run the explicit opt-in installer, then start Gym again:\n"
        f"  {command}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve-venv", help="print the managed OSWorld agent venv path")
    resolve_parser.add_argument("--gym-root", type=Path, required=True)
    resolve_parser.add_argument("--env-file", type=Path, required=True)

    check_parser = subparsers.add_parser("check", help="validate this interpreter's OSWorld runtime packages")
    check_parser.add_argument("--quiet", action="store_true", help="suppress output when validation succeeds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "resolve-venv":
        try:
            print(managed_agent_venv_from_env(args.gym_root, args.env_file))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(f"Cannot resolve the managed OSWorld agent venv: {exc}", file=sys.stderr)
            return 2
        return 0

    problems = validate_optional_runtime_dependencies()
    if problems:
        if not args.quiet:
            print(
                "OSWorld optional runtime dependencies are missing, incompatible, or unusable:",
                file=sys.stderr,
            )
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
        return 1
    if not args.quiet:
        versions = ", ".join(
            f"{dependency.distribution}={importlib.metadata.version(dependency.distribution)}"
            for dependency in OPTIONAL_RUNTIME_DEPENDENCIES
        )
        print(f"OSWorld optional runtime dependencies are ready: {versions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
