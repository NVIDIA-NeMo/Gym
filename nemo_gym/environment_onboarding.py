# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local test and publish gates for manifest-backed environments."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.config_types import ConfigError
from nemo_gym.environment_manifest import IntegrationProfile, load_manifest
from nemo_gym.environment_scaffold import SCAFFOLD_PLACEHOLDER
from nemo_gym.environment_validation import EnvironmentValidationReport, validate_environment
from nemo_gym.registry import (
    EnvironmentCatalogEntry,
)
from nemo_gym.verifier_fixture import VerifierFixture, VerifierFixtureResult, exercise_verifier_fixture


PUBLISH_PLACEHOLDER = SCAFFOLD_PLACEHOLDER
_TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".py", ".txt", ".yaml", ".yml"})
_IGNORED_AUTHORING_DIRS = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "build", "dist", "node_modules"}
)


class EnvironmentOnboardingError(ConfigError):
    """An onboarding command could not complete locally."""


@dataclass(frozen=True)
class VerifierTestReport:
    name: str
    resources_server: str
    fixture_path: str
    cases: tuple[VerifierFixtureResult, ...]
    updated_expected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "resources_server": self.resources_server,
            "fixture_path": self.fixture_path,
            "cases": [case.model_dump(mode="json") for case in self.cases],
            "updated_expected": self.updated_expected,
        }


@dataclass(frozen=True)
class PublishReport:
    name: str
    kind: str
    version: str
    status: str
    validation: EnvironmentValidationReport
    verifier_test: VerifierTestReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "status": self.status,
            "validation": self.validation.to_dict(),
            "verifier_test": self.verifier_test.to_dict(),
        }


@contextmanager
def _import_path(path: Path):
    value = str(path)
    already_present = value in sys.path
    if not already_present:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if not already_present:
            sys.path.remove(value)


def _load_resources_module(resources_server: str, entry: EnvironmentCatalogEntry) -> ModuleType:
    relative = Path("resources_servers") / resources_server / "app.py"
    candidates = [_component_root(entry) / relative]
    candidates.append(_resolve_under_cwd_or_install(relative))
    app_path = next((path.resolve() for path in candidates if path.is_file()), None)
    if app_path is None:
        raise EnvironmentOnboardingError(f"Resources server '{resources_server}' does not have a discoverable app.py.")

    component_root = app_path.parents[2]
    module_name = ".".join(app_path.relative_to(component_root).with_suffix("").parts)
    package_name = module_name.rpartition(".")[0]
    for loaded_name in tuple(sys.modules):
        if loaded_name == package_name or loaded_name.startswith(f"{package_name}."):
            sys.modules.pop(loaded_name)
    try:
        with _import_path(app_path.parent), _import_path(component_root):
            importlib.invalidate_caches()
            module = importlib.import_module(module_name)
    except Exception as error:
        raise EnvironmentOnboardingError(
            f"Could not import resources server '{resources_server}' from '{app_path}': {error}"
        ) from error
    loaded_path = Path(module.__file__).resolve() if module.__file__ else None
    if loaded_path != app_path:
        raise EnvironmentOnboardingError(
            f"Resources server '{resources_server}' resolved to '{loaded_path}', not '{app_path}'."
        )
    return module


def _component_root(entry: EnvironmentCatalogEntry) -> Path:
    return next(
        (parent.parent for parent in entry.path.resolve().parents if parent.name in {"environments", "benchmarks"}),
        entry.path.parent.parent,
    )


def test_environment(
    entry: EnvironmentCatalogEntry,
    *,
    update_expected: bool = False,
    validate_first: bool = True,
) -> VerifierTestReport:
    """Exercise the selected scorer directly, without Ray or server processes."""
    if entry.manifest_path is None:
        raise EnvironmentOnboardingError(
            f"'{entry.name}' has no manifest. Use `gym env test --resources-server NAME` for legacy server tests."
        )
    if validate_first:
        validate_environment(entry)
    manifest = load_manifest(entry.manifest_path)
    module = _load_resources_module(manifest.resources_server, entry)
    fixture = getattr(module, "VERIFIER_FIXTURE", None)
    if not isinstance(fixture, VerifierFixture):
        raise EnvironmentOnboardingError(
            f"Resources server '{manifest.resources_server}' does not export VERIFIER_FIXTURE."
        )
    try:
        results = asyncio.run(
            exercise_verifier_fixture(
                fixture,
                reward_range=manifest.reward.range,
                determinism=manifest.determinism,
                update_expected=update_expected,
            )
        )
    except Exception as error:
        raise EnvironmentOnboardingError(
            f"Verifier fixture for '{manifest.resources_server}' failed: {error}"
        ) from error
    return VerifierTestReport(
        name=entry.name,
        resources_server=manifest.resources_server,
        fixture_path=str(Path(fixture.cases_path).resolve()),
        cases=results,
        updated_expected=update_expected,
    )


def find_publish_placeholders(path: Path) -> tuple[Path, ...]:
    """Return authored text files that still contain the scaffold's explicit marker."""
    matches: list[Path] = []
    for current, directories, filenames in os.walk(path, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in _IGNORED_AUTHORING_DIRS and not (current_path / directory).is_symlink()
        )
        for filename in sorted(filenames):
            candidate = current_path / filename
            if candidate.is_symlink() or candidate.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            try:
                with candidate.open(encoding="utf-8") as stream:
                    contains_placeholder = any(PUBLISH_PLACEHOLDER in line for line in stream)
            except (OSError, UnicodeError) as error:
                raise EnvironmentOnboardingError(f"Could not inspect authored file '{candidate}': {error}") from error
            if contains_placeholder:
                matches.append(candidate)
    return tuple(matches)


def _readiness_paths(entry: EnvironmentCatalogEntry) -> tuple[Path, ...]:
    if entry.manifest_path is None:
        return (entry.path,)
    manifest = load_manifest(entry.manifest_path)
    component_root = _component_root(entry)
    paths = [entry.path]
    paths.append(component_root / "resources_servers" / manifest.resources_server)
    if manifest.integration_profile in {IntegrationProfile.MEASURED_LOOP, IntegrationProfile.EXTERNAL_LOOP}:
        paths.append(component_root / "responses_api_agents" / manifest.agent_server)
    return tuple(dict.fromkeys(path for path in paths if path.exists()))


def publish_environment(
    entry: EnvironmentCatalogEntry,
) -> PublishReport:
    """Run the local readiness gate for a registry entry."""
    placeholders = tuple(
        sorted(
            {match for path in _readiness_paths(entry) for match in find_publish_placeholders(path)},
            key=str,
        )
    )
    if placeholders:
        listed = "\n".join(f"  - {path}" for path in placeholders)
        raise EnvironmentOnboardingError(
            f"'{entry.name}' still contains scaffold placeholders:\n{listed}\n"
            f"Replace every {PUBLISH_PLACEHOLDER} marker before running `gym env publish`."
        )

    validation = validate_environment(entry)
    verifier_test = test_environment(entry, validate_first=False)
    if entry.status != "experimental":
        raise EnvironmentOnboardingError(f"Catalog entry '{entry.name}' has unexpected local status '{entry.status}'.")
    return PublishReport(
        name=entry.name,
        kind=entry.kind,
        version=validation.version,
        status=entry.status,
        validation=validation,
        verifier_test=verifier_test,
    )


__all__ = [
    "EnvironmentOnboardingError",
    "PUBLISH_PLACEHOLDER",
    "PublishReport",
    "VerifierTestReport",
    "find_publish_placeholders",
    "publish_environment",
    "test_environment",
]
