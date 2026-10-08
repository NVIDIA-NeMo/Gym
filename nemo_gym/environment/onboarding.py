# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free checks for manifest-backed environment verifiers."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from nemo_gym import PARENT_DIR, component_search_roots
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.environment.validation import EnvironmentValidationReport, ResolvedComponent, validate_environment
from nemo_gym.registry import EnvironmentCatalogEntry
from nemo_gym.verifier_fixture import (
    VerifierFixture,
    VerifierFixtureError,
    VerifierFixtureResult,
    exercise_verifier_fixture,
)


class EnvironmentOnboardingError(ConfigError):
    """A manifest-backed onboarding check could not be completed."""


@dataclass(frozen=True)
class VerifierRunSpec:
    name: str
    kind: str
    resources_server: str
    manifest_path: str
    app_path: str
    component_root: str
    server_dir: str
    bundled_component: bool
    reward_range: tuple[float, float]
    higher_is_better: bool
    determinism: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifierReport:
    """Summary of an in-process verifier-fixture run."""

    name: str
    kind: str
    resources_server: str
    manifest_path: str
    fixture_path: str
    cases: tuple[VerifierFixtureResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VerifierReport":
        cases = tuple(
            VerifierFixtureResult(
                name=case["name"],
                kind=case["kind"],
                observed_rewards=tuple(case.get("observed_rewards", ())),
            )
            for case in value["cases"]
        )
        return cls(
            name=value["name"],
            kind=value["kind"],
            resources_server=value["resources_server"],
            manifest_path=value["manifest_path"],
            fixture_path=value["fixture_path"],
            cases=cases,
        )


def _resources_server_app(component: ResolvedComponent) -> tuple[Path, Path, Path]:
    if component.boundary != "resources_servers" or not component.entrypoint:
        raise EnvironmentOnboardingError("The resolved workload has no runnable resources-server entrypoint.")

    relative_dir = Path(component.boundary) / component.implementation
    entrypoint = Path(component.entrypoint)
    if entrypoint.is_absolute() or ".." in entrypoint.parts:
        raise EnvironmentOnboardingError(
            f"Resources-server entrypoint must stay within '{relative_dir}': {component.entrypoint!r}"
        )

    for root in component_search_roots():
        component_root = root.expanduser().resolve()
        server_dir = component_root / relative_dir
        markers = [path for path in (server_dir / "requirements.txt", server_dir / "pyproject.toml") if path.is_file()]
        if not markers:
            continue
        if len(markers) > 1:
            raise EnvironmentOnboardingError(
                f"Resources server '{server_dir}' must use requirements.txt or pyproject.toml, not both."
            )
        candidate = server_dir / entrypoint
        if not candidate.is_file():
            raise EnvironmentOnboardingError(f"Resources-server entrypoint was not found: {candidate}")
        resolved = candidate.resolve()
        resolved_server_dir = server_dir.resolve()
        if not resolved.is_relative_to(resolved_server_dir):
            raise EnvironmentOnboardingError(
                f"Resources-server entrypoint resolves outside '{server_dir}': {candidate}"
            )
        return resolved, component_root, resolved_server_dir

    raise EnvironmentOnboardingError(
        f"Resources server {component.implementation!r} was not found in a component root."
    )


def _runtime_component_paths(spec: VerifierRunSpec) -> tuple[Path, Path, Path]:
    app_path = Path(spec.app_path)
    component_root = Path(spec.component_root)
    server_dir = Path(spec.server_dir)
    if not spec.bundled_component:
        return app_path, component_root, server_dir
    try:
        relative_app = app_path.relative_to(component_root)
        relative_server = server_dir.relative_to(component_root)
    except ValueError as error:
        raise EnvironmentOnboardingError("Bundled verifier paths must be inside their component root.") from error
    child_root = PARENT_DIR.resolve()
    child_app = child_root / relative_app
    child_server = child_root / relative_server
    if not child_app.is_file():
        raise EnvironmentOnboardingError(f"Bundled resources-server entrypoint was not found: {child_app}")
    return child_app, child_root, child_server


def _import_resources_server(app_path: Path, component_root: Path, server_dir: Path) -> ModuleType:
    relative_module = app_path.relative_to(component_root).with_suffix("")
    module_parts = relative_module.parts
    module_name = (
        ".".join(module_parts)
        if module_parts and all(part.isidentifier() for part in module_parts)
        else "_nemo_gym_verifier_fixture"
    )
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    if spec is None or spec.loader is None:
        raise EnvironmentOnboardingError(f"Could not create an import specification for '{app_path}'.")

    module = importlib.util.module_from_spec(spec)
    module_prefix = module_name.split(".", 1)[0]
    previous_modules = {
        name: loaded
        for name, loaded in sys.modules.items()
        if name == module_prefix or name.startswith(f"{module_prefix}.")
    }
    previous_sys_path = sys.path.copy()
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.modules[module_name] = module
    sys.path[:0] = [str(server_dir), str(component_root)]
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise EnvironmentOnboardingError(
            f"Could not import resources-server entrypoint '{app_path}': {error}"
        ) from error
    finally:
        sys.path[:] = previous_sys_path
        for name in tuple(sys.modules):
            if name == module_prefix or name.startswith(f"{module_prefix}."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
    return module


def prepare_verifier_run(
    entry: EnvironmentCatalogEntry,
    validation: EnvironmentValidationReport | None = None,
) -> VerifierRunSpec:
    """Validate a catalog entry and resolve the scorer process boundary."""
    if entry.manifest_path is None:
        raise EnvironmentOnboardingError(
            f"{entry.kind.capitalize()} {entry.name!r} has no manifest.yaml. "
            f"Add one beside '{entry.config_path}' before running verifier checks."
        )

    manifest_path = entry.manifest_path.expanduser().resolve()
    config_path = entry.config_path.expanduser().resolve()
    validation = validation or validate_environment(manifest_path, config_path)
    manifest = load_manifest(manifest_path)

    resources_component = next(
        (component for component in validation.components if component.role == "resources_server"),
        None,
    )
    if resources_component is None:
        raise EnvironmentOnboardingError("The resolved workload has no resources server.")
    app_path, component_root, server_dir = _resources_server_app(resources_component)
    bundled_component = component_root == PARENT_DIR.resolve() and not (PARENT_DIR / "pyproject.toml").is_file()
    return VerifierRunSpec(
        name=manifest.name,
        kind=manifest.kind.value,
        resources_server=manifest.resources_server,
        manifest_path=str(manifest_path),
        app_path=str(app_path),
        component_root=str(component_root),
        server_dir=str(server_dir),
        bundled_component=bundled_component,
        reward_range=tuple(manifest.reward.range),
        higher_is_better=manifest.reward.higher_is_better,
        determinism=manifest.determinism.value,
    )


async def exercise_verifier_run(spec: VerifierRunSpec, *, update_expected: bool = False) -> VerifierReport:
    """Import and exercise a prepared verifier inside its dependency environment."""
    if spec.bundled_component and update_expected:
        raise EnvironmentOnboardingError(
            "Cannot update a verifier fixture in an installed Gym package; use a source checkout."
        )
    module = _import_resources_server(*_runtime_component_paths(spec))
    fixture = getattr(module, "VERIFIER_FIXTURE", None)
    if not isinstance(fixture, VerifierFixture):
        raise EnvironmentOnboardingError(
            f"Resources server {spec.resources_server!r} must export VERIFIER_FIXTURE as a VerifierFixture."
        )

    try:
        cases = await exercise_verifier_fixture(
            fixture,
            reward_range=spec.reward_range,
            higher_is_better=spec.higher_is_better,
            determinism=spec.determinism,
            update_expected=update_expected,
        )
    except VerifierFixtureError as error:
        raise EnvironmentOnboardingError(f"Verifier fixture for {spec.kind} {spec.name!r} failed: {error}") from error

    return VerifierReport(
        name=spec.name,
        kind=spec.kind,
        resources_server=spec.resources_server,
        manifest_path=spec.manifest_path,
        fixture_path=str(Path(fixture.cases_path).expanduser().resolve()),
        cases=cases,
    )


async def verify_environment(
    entry: EnvironmentCatalogEntry,
    *,
    update_expected: bool = False,
) -> VerifierReport:
    """Validate and exercise a verifier in the current Python environment."""
    return await exercise_verifier_run(prepare_verifier_run(entry), update_expected=update_expected)


__all__ = [
    "EnvironmentOnboardingError",
    "VerifierRunSpec",
    "VerifierReport",
    "exercise_verifier_run",
    "prepare_verifier_run",
    "verify_environment",
]
