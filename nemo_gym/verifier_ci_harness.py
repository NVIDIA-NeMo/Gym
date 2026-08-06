# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository-owned offline verifier harness used locally and in CI.

Component tests remain useful to authors, but they are not a trust boundary.
The selected runtime entrypoint must expose the fixed
``create_offline_verifier_app`` contract; this harness imports that entrypoint
and runs Gym's canonical fixture against the returned ASGI application.  It
starts no service, model, Ray cluster, or network listener.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Sequence

from omegaconf import DictConfig, OmegaConf

from nemo_gym.verifier_fixture import (
    DETERMINISM_ENV_VAR,
    HIGHER_IS_BETTER_ENV_VAR,
    REWARD_RANGE_ENV_VAR,
    VerifierFixtureError,
    exercise_verifier_fixture,
    verifier_fixture_environment,
)


if TYPE_CHECKING:
    from fastapi.testclient import TestClient


VERIFIER_HARNESS_MODULE = "nemo_gym.verifier_ci_harness"
OFFLINE_VERIFIER_APP_FACTORY = "create_offline_verifier_app"


class VerifierCIHarnessError(VerifierFixtureError):
    """The selected verifier cannot be exercised by the canonical CI harness."""


@dataclass(frozen=True)
class VerifierHarnessInvocation:
    """A complete, shell-free invocation of the canonical verifier harness."""

    command: tuple[str, ...]
    stdin: str
    environment: dict[str, str]


def _resources_server_runtime_matches(
    config: Mapping[str, Any] | DictConfig,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Return every concrete resources-server runtime in a resolved config."""

    matches: list[tuple[str, str, dict[str, Any]]] = []
    for raw_instance, block in config.items():
        if not isinstance(block, (Mapping, DictConfig)):
            continue
        resources = block.get("resources_servers")
        if not isinstance(resources, (Mapping, DictConfig)):
            continue
        for raw_implementation, server_config in resources.items():
            if not isinstance(server_config, (Mapping, DictConfig)):
                continue
            plain = (
                OmegaConf.to_container(server_config, resolve=True)
                if isinstance(server_config, DictConfig)
                else server_config
            )
            if not isinstance(plain, Mapping):  # pragma: no cover - guarded above
                continue
            matches.append(
                (
                    str(raw_implementation),
                    str(raw_instance),
                    {str(key): value for key, value in plain.items()},
                )
            )
    return matches


def select_resources_server_runtime(
    config: Mapping[str, Any] | DictConfig,
    implementation: str,
) -> tuple[str, dict[str, Any]]:
    """Return the one resolved runtime instance for a verifier implementation."""

    matches = [
        (instance_name, server_config)
        for candidate, instance_name, server_config in _resources_server_runtime_matches(config)
        if candidate == implementation
    ]
    if len(matches) != 1:
        rendered = ", ".join(instance for instance, _server_config in matches) or "none"
        raise VerifierCIHarnessError(
            f"Resources server implementation {implementation!r} must resolve to exactly one runtime instance; "
            f"found: {rendered}."
        )

    return matches[0]


def select_sole_resources_server_runtime(
    config: Mapping[str, Any] | DictConfig,
) -> tuple[str, str, dict[str, Any]]:
    """Return the sole verifier runtime selected by a legacy resolved recipe."""

    matches = _resources_server_runtime_matches(config)
    if len(matches) != 1:
        rendered = ", ".join(f"{instance}/{implementation}" for implementation, instance, _config in matches)
        raise VerifierCIHarnessError(
            "A dependency-only legacy unit must resolve exactly one Resources Server for its local CI check; "
            f"found: {rendered or 'none'}."
        )
    return matches[0]


def build_verifier_harness_invocation(
    *,
    python_executable: str | Path,
    project_root: str | Path,
    component_dir: str | Path,
    entrypoint: str | Path,
    instance_name: str,
    fixture_path: str | Path,
    server_config: Mapping[str, Any],
    reward_range: tuple[int | float, int | float] | None = None,
    higher_is_better: bool = True,
    determinism: str | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> VerifierHarnessInvocation:
    """Build the shared local/CI subprocess contract without executing code."""

    component = Path(component_dir).resolve()
    fixture = Path(fixture_path).resolve()
    relative_entrypoint = Path(entrypoint)
    if relative_entrypoint.is_absolute():
        try:
            relative_entrypoint = relative_entrypoint.resolve().relative_to(component)
        except ValueError as error:
            raise VerifierCIHarnessError(
                f"Resources-server entrypoint '{entrypoint}' is outside component '{component}'."
            ) from error
    if ".." in relative_entrypoint.parts:
        raise VerifierCIHarnessError(f"Resources-server entrypoint must be component-relative, got {entrypoint!r}.")

    environment = dict(os.environ if base_environment is None else base_environment)
    # Optional contracts must not leak in from a developer shell or an earlier
    # in-process check. A legacy unit without a manifest deliberately validates
    # only the expectations carried by its fixture.
    for name in (REWARD_RANGE_ENV_VAR, HIGHER_IS_BETTER_ENV_VAR, DETERMINISM_ENV_VAR):
        environment.pop(name, None)
    environment.update(
        verifier_fixture_environment(
            reward_range=reward_range,
            higher_is_better=higher_is_better,
            determinism=determinism,
            update_expected=False,
        )
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    python_path_entries = [str(Path(project_root).resolve())]
    if existing_python_path := environment.get("PYTHONPATH"):
        python_path_entries.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    command = (
        str(python_executable),
        "-m",
        VERIFIER_HARNESS_MODULE,
        "--component-dir",
        str(component),
        "--entrypoint",
        relative_entrypoint.as_posix(),
        "--instance-name",
        instance_name,
        "--fixture",
        str(fixture),
    )
    return VerifierHarnessInvocation(
        command=command,
        stdin=json.dumps(dict(server_config), sort_keys=True, default=str),
        environment=environment,
    )


def _contained_entrypoint(component_dir: Path, entrypoint: str) -> Path:
    component = component_dir.resolve(strict=True)
    relative = Path(entrypoint)
    if relative.is_absolute() or ".." in relative.parts:
        raise VerifierCIHarnessError(f"Resources-server entrypoint must be component-relative, got {entrypoint!r}.")
    candidate = component / relative
    cursor = component
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise VerifierCIHarnessError(f"Resources-server entrypoint path '{cursor}' must not be a symbolic link.")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(component)
    except (OSError, RuntimeError, ValueError) as error:
        raise VerifierCIHarnessError(
            f"Resources-server entrypoint '{candidate}' does not stay inside '{component}'."
        ) from error
    if not resolved.is_file() or resolved.suffix != ".py":
        raise VerifierCIHarnessError(f"Resources-server entrypoint '{resolved}' must be a Python file.")
    return resolved


def _load_entrypoint(component_dir: Path, entrypoint: Path) -> ModuleType:
    project_root = component_dir.parent.parent
    try:
        relative = entrypoint.relative_to(project_root).with_suffix("")
    except ValueError as error:
        raise VerifierCIHarnessError(
            f"Resources-server entrypoint '{entrypoint}' is outside component root '{project_root}'."
        ) from error
    module_name = ".".join(relative.parts)
    search_paths = (str(component_dir), str(project_root))
    for path in reversed(search_paths):
        sys.path.insert(0, path)
    try:
        importlib.invalidate_caches()
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
    except Exception as error:
        raise VerifierCIHarnessError(
            f"Could not import resources-server entrypoint '{entrypoint}': {error}."
        ) from error
    finally:
        for path in search_paths:
            sys.path.remove(path)
    loaded_path = Path(module.__file__ or "").resolve()
    if loaded_path != entrypoint:
        raise VerifierCIHarnessError(
            f"Resources-server entrypoint import resolved to '{loaded_path}', expected '{entrypoint}'."
        )
    return module


def _offline_app_factory(module: ModuleType, entrypoint: Path):
    factory = getattr(module, OFFLINE_VERIFIER_APP_FACTORY, None)
    if not callable(factory):
        raise VerifierCIHarnessError(
            f"Resources-server entrypoint '{entrypoint}' must expose callable "
            f"{OFFLINE_VERIFIER_APP_FACTORY}(*, server_config, instance_name)."
        )
    return factory


def exercise_selected_verifier(
    *,
    component_dir: str | Path,
    entrypoint: str,
    instance_name: str,
    server_config: dict[str, Any],
    fixture_path: str | Path,
    reward_range: tuple[int | float, int | float] | None = None,
    higher_is_better: bool = True,
    determinism: str | None = None,
) -> None:
    """Construct the selected server and execute the canonical fixture in process."""

    component = Path(component_dir).resolve(strict=True)
    entrypoint_path = _contained_entrypoint(component, entrypoint)
    fixture = Path(fixture_path).resolve(strict=True)
    try:
        fixture.relative_to(component)
    except ValueError as error:
        raise VerifierCIHarnessError(f"Verifier fixture '{fixture}' must stay inside '{component}'.") from error

    module = _load_entrypoint(component, entrypoint_path)
    app_factory = _offline_app_factory(module, entrypoint_path)

    def client_factory() -> TestClient:
        from fastapi.testclient import TestClient

        try:
            app = app_factory(server_config=dict(server_config), instance_name=instance_name)
            return TestClient(app)
        except Exception as error:
            raise VerifierCIHarnessError(
                f"Offline verifier app factory in '{entrypoint_path}' failed: {error}."
            ) from error

    exercise_verifier_fixture(
        client_factory,
        fixture,
        reward_range=reward_range,
        higher_is_better=higher_is_better,
        determinism=determinism,
        update_expected=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gym's repository-owned offline verifier fixture harness.")
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--instance-name", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise VerifierCIHarnessError("Resolved resources-server config on stdin must be a JSON object.")
        raw_range_json = os.environ.get(REWARD_RANGE_ENV_VAR)
        raw_range = json.loads(raw_range_json) if raw_range_json is not None else None
        if raw_range is not None and (not isinstance(raw_range, list) or len(raw_range) != 2):
            raise VerifierCIHarnessError(f"{REWARD_RANGE_ENV_VAR} must contain a two-value JSON array.")
        exercise_selected_verifier(
            component_dir=args.component_dir,
            entrypoint=args.entrypoint,
            instance_name=args.instance_name,
            server_config=payload,
            fixture_path=args.fixture,
            reward_range=(raw_range[0], raw_range[1]) if raw_range is not None else None,
            higher_is_better=os.environ.get(HIGHER_IS_BETTER_ENV_VAR, "true").casefold() == "true",
            determinism=os.environ.get(DETERMINISM_ENV_VAR),
        )
    except (KeyError, OSError, ValueError, VerifierFixtureError) as error:
        print(f"Canonical verifier fixture failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "OFFLINE_VERIFIER_APP_FACTORY",
    "VERIFIER_HARNESS_MODULE",
    "VerifierCIHarnessError",
    "VerifierHarnessInvocation",
    "select_resources_server_runtime",
    "select_sole_resources_server_runtime",
    "build_verifier_harness_invocation",
    "exercise_selected_verifier",
    "main",
]
