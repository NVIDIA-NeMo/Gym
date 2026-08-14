# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local publication checks for manifest-backed workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from nemo_gym.config_types import ConfigError
from nemo_gym.environment.manifest import EnvironmentKind, load_manifest
from nemo_gym.environment.onboarding import VerifierReport
from nemo_gym.environment.validation import EnvironmentValidationReport
from nemo_gym.registry import EnvironmentCatalogEntry, discover_environment_catalog


class EnvironmentPublicationError(ConfigError):
    """A workload failed a local publication check."""


@dataclass(frozen=True)
class EnvironmentPublicationReport:
    name: str
    version: str
    kind: str
    status: str
    manifest_path: str
    verifier_cases: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_placeholder(value: str | None) -> bool:
    return value is not None and value.strip().casefold().startswith("todo")


def _publication_placeholders(entry: EnvironmentCatalogEntry) -> tuple[str, ...]:
    if entry.manifest_path is None:
        return ("manifest",)
    manifest = load_manifest(entry.manifest_path)
    placeholders: list[str] = []
    if _is_placeholder(manifest.description):
        placeholders.append("description")
    if any(_is_placeholder(author) for author in manifest.authors):
        placeholders.append("authors")
    if manifest.licensing == "unknown":
        placeholders.append("licensing")
    if manifest.kind == EnvironmentKind.BENCHMARK:
        if _is_placeholder(manifest.canonical_split):
            placeholders.append("canonical_split")
        if _is_placeholder(manifest.standard_prompt_config):
            placeholders.append("standard_prompt_config")
    return tuple(placeholders)


def finalize_publication(
    entry: EnvironmentCatalogEntry,
    validation: EnvironmentValidationReport,
    verifier: VerifierReport,
    *,
    catalog_entries: Iterable[EnvironmentCatalogEntry] | None = None,
) -> EnvironmentPublicationReport:
    """Confirm local checks and exact discoverability without starting a full evaluation."""
    if entry.manifest_path is None:
        raise EnvironmentPublicationError(f"{entry.kind.capitalize()} {entry.name!r} has no manifest.yaml.")

    placeholders = _publication_placeholders(entry)
    if placeholders:
        raise EnvironmentPublicationError(
            "Replace scaffold placeholders before publishing: " + ", ".join(placeholders)
        )

    if validation.name != entry.name or validation.kind != entry.kind:
        raise EnvironmentPublicationError("Validation report does not describe the selected catalog entry.")
    if verifier.name != entry.name or verifier.kind != entry.kind or not verifier.cases:
        raise EnvironmentPublicationError("Verifier report does not describe a passing fixture for this workload.")

    entries = tuple(discover_environment_catalog() if catalog_entries is None else catalog_entries)
    selected_manifest = Path(entry.manifest_path).resolve()
    matches = [
        candidate
        for candidate in entries
        if candidate.name == entry.name
        and candidate.kind == entry.kind
        and candidate.manifest_path is not None
        and candidate.manifest_path.resolve() == selected_manifest
    ]
    if len(matches) != 1:
        raise EnvironmentPublicationError(
            f"Catalog did not resolve {entry.kind} {entry.name!r} to its exact manifest after publication checks."
        )
    published = matches[0]
    if published.status != "experimental":
        raise EnvironmentPublicationError(
            f"Newly published workloads must enter as experimental, observed {published.status!r}."
        )

    return EnvironmentPublicationReport(
        name=validation.name,
        version=validation.version,
        kind=validation.kind,
        status=published.status,
        manifest_path=str(selected_manifest),
        verifier_cases=len(verifier.cases),
    )


__all__ = [
    "EnvironmentPublicationError",
    "EnvironmentPublicationReport",
    "finalize_publication",
]
