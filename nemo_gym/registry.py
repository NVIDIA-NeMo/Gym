# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Local registry for manifest-backed and legacy environments and benchmarks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from omegaconf import DictConfig, OmegaConf

from nemo_gym import PARENT_DIR, component_search_roots
from nemo_gym.benchmarks import benchmark_config_name, benchmark_config_paths
from nemo_gym.config_types import ConfigError
from nemo_gym.discovery import iter_server_configs, read_config_metadata
from nemo_gym.environment_manifest import EnvironmentManifest, load_manifest


ENVIRONMENTS_SUBDIR = "environments"
ENVIRONMENTS_DIR = PARENT_DIR / ENVIRONMENTS_SUBDIR
BENCHMARKS_SUBDIR = "benchmarks"
ENVIRONMENT_CONFIG_FILENAME = "config.yaml"
MANIFEST_FILENAME = "manifest.yaml"

CatalogKind = Literal["environment", "benchmark"]
CatalogStatus = Literal["experimental", "validated", "no-manifest"]


class RegistryError(ConfigError):
    """A registry entry is invalid or cannot be selected unambiguously."""


@dataclass(frozen=True)
class EnvironmentCatalogEntry:
    """One runnable unit discovered from a manifest or legacy config."""

    # Keep the first five fields aligned with the historical EnvironmentEntry constructor.
    name: str
    config_path: Path
    path: Path
    description: Optional[str] = None
    domain: Optional[str] = None
    kind: CatalogKind = "environment"
    status: CatalogStatus = "no-manifest"
    manifest_path: Optional[Path] = None
    version: Optional[str] = None
    integration_profile: Optional[str] = None
    modality: Optional[str] = None
    licensing: Optional[str] = None
    lifecycle: Optional[str] = None


@dataclass(frozen=True)
class EnvironmentEntry(EnvironmentCatalogEntry):
    """A discovered environment: its name, where it lives, and lightweight metadata."""

    kind: CatalogKind = field(default="environment", init=False)


def _enum_value(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _path_identity(tree_dir: Path, manifest_path: Path) -> str:
    relative = manifest_path.parent.relative_to(tree_dir)
    if relative == Path("."):
        raise RegistryError(f"Manifest '{manifest_path}' must be inside a named registry directory.")
    return relative.as_posix()


def _manifest_entry(tree_dir: Path, manifest_path: Path, expected_kind: CatalogKind) -> EnvironmentCatalogEntry:
    expected_name = _path_identity(tree_dir, manifest_path)
    manifest: EnvironmentManifest = load_manifest(manifest_path)
    actual_kind = _enum_value(manifest.kind)
    if actual_kind != expected_kind:
        raise RegistryError(
            f"Manifest '{manifest_path}' declares kind '{actual_kind}', but its registry path requires "
            f"'{expected_kind}'."
        )
    if manifest.name != expected_name:
        raise RegistryError(
            f"Manifest '{manifest_path}' declares name '{manifest.name}', but its registry path requires "
            f"'{expected_name}'."
        )

    config_path = manifest_path.with_name(ENVIRONMENT_CONFIG_FILENAME)
    if not config_path.is_file():
        raise RegistryError(f"Manifest '{manifest_path}' requires a sibling config.yaml.")

    values = {
        "name": manifest.name,
        "config_path": config_path,
        "path": manifest_path.parent,
        "description": manifest.description,
        "domain": _enum_value(manifest.domain),
        "status": "experimental",
        "manifest_path": manifest_path,
        "version": manifest.version,
        "integration_profile": _enum_value(manifest.integration_profile),
        "modality": manifest.modality,
        "licensing": manifest.licensing,
        "lifecycle": _enum_value(manifest.lifecycle),
    }
    if expected_kind == "environment":
        return EnvironmentEntry(**values)
    return EnvironmentCatalogEntry(kind="benchmark", **values)


def _legacy_entry(name: str, config_path: Path, kind: CatalogKind) -> EnvironmentCatalogEntry:
    domain, description = read_config_metadata(config_path)
    values = {
        "name": name,
        "config_path": config_path,
        "path": config_path.parent,
        "description": description,
        "domain": domain,
    }
    if kind == "environment":
        return EnvironmentEntry(**values)
    return EnvironmentCatalogEntry(kind="benchmark", **values)


def _legacy_config_paths(tree_dir: Path, kind: CatalogKind) -> Iterable[tuple[str, Path]]:
    if kind == "benchmark":
        for config_path in benchmark_config_paths(tree_dir):
            yield benchmark_config_name(config_path.relative_to(tree_dir)), config_path
        return

    for child in sorted(tree_dir.iterdir()):
        config_path = child / ENVIRONMENT_CONFIG_FILENAME
        if child.is_dir() and config_path.is_file():
            yield child.name, config_path


def _discover_registry_tree(
    tree_dir: Path,
    kind: CatalogKind,
    *,
    claimed: frozenset[tuple[CatalogKind, str]] = frozenset(),
) -> Dict[tuple[CatalogKind, str], EnvironmentCatalogEntry]:
    """Discover one registry tree without resolving or importing runnable configs."""
    if not tree_dir.is_dir():
        return {}

    entries: Dict[tuple[CatalogKind, str], EnvironmentCatalogEntry] = {}
    manifest_configs: set[Path] = set()
    for manifest_path in sorted(tree_dir.rglob(MANIFEST_FILENAME)):
        expected_name = _path_identity(tree_dir, manifest_path)
        key = (kind, expected_name)
        if key in claimed:
            continue
        entry = _manifest_entry(tree_dir, manifest_path, kind)
        entries[key] = entry
        manifest_configs.add(entry.config_path.resolve())

    for name, config_path in _legacy_config_paths(tree_dir, kind):
        key = (kind, name)
        if config_path.resolve() in manifest_configs or key in claimed or key in entries:
            continue
        entries[key] = _legacy_entry(name, config_path, kind)
    return entries


def _discover_environments_in_dir(environments_dir: Path) -> Dict[str, EnvironmentEntry]:
    """Map environment name to its effective manifest or legacy config under one directory."""
    return {
        name: entry
        for (_kind, name), entry in _discover_registry_tree(environments_dir, "environment").items()
        if isinstance(entry, EnvironmentEntry)
    }


def discover_environments() -> Dict[str, EnvironmentEntry]:
    """Discover environments with standard component-root precedence."""
    environments: Dict[str, EnvironmentEntry] = {}
    for root in component_search_roots():
        claimed = frozenset(("environment", name) for name in environments)
        entries = _discover_registry_tree(root / ENVIRONMENTS_SUBDIR, "environment", claimed=claimed)
        environments.update(
            (name, entry) for (_kind, name), entry in entries.items() if isinstance(entry, EnvironmentEntry)
        )
    return environments


def discover_environment_catalog() -> tuple[EnvironmentCatalogEntry, ...]:
    """Discover the manifest/legacy union for environments and benchmarks.

    Earlier component roots win. Within one root, a valid manifest replaces only its sibling legacy
    ``config.yaml``; other benchmark config flavors remain visible as legacy entries.
    """
    entries: Dict[tuple[CatalogKind, str], EnvironmentCatalogEntry] = {}
    for root in component_search_roots():
        for kind, subdir in (("environment", ENVIRONMENTS_SUBDIR), ("benchmark", BENCHMARKS_SUBDIR)):
            entries.update(_discover_registry_tree(root / subdir, kind, claimed=frozenset(entries)))
    return tuple(sorted(entries.values(), key=lambda entry: (entry.name.casefold(), entry.kind)))


def resolve_catalog_entry(
    name: str,
    kind: CatalogKind | str | None = None,
    *,
    entries: Iterable[EnvironmentCatalogEntry] | None = None,
) -> EnvironmentCatalogEntry:
    """Resolve a catalog name, requiring ``kind`` only when both kinds use the same name."""
    selected_kind = _enum_value(kind)
    if selected_kind not in (None, "environment", "benchmark"):
        raise RegistryError(f"Unknown registry kind '{selected_kind}'.")

    matches = [
        entry
        for entry in (discover_environment_catalog() if entries is None else entries)
        if entry.name == name and (selected_kind is None or entry.kind == selected_kind)
    ]
    if not matches:
        suffix = f" with kind '{selected_kind}'" if selected_kind else ""
        raise RegistryError(f"Unknown registry entry '{name}'{suffix}.")
    if len(matches) > 1:
        kinds = ", ".join(sorted(entry.kind for entry in matches))
        raise RegistryError(f"Registry name '{name}' is ambiguous ({kinds}); specify a kind.")
    return matches[0]


def read_environment_details(config_path: Path) -> Dict[str, object]:
    """Deep-parse an environment config for the ``gym list environments <name>`` inspect view.

    Returns ``domain``, ``description`` (via :func:`~nemo_gym.discovery.read_config_metadata`), plus
    ``value``, ``resources_servers`` (names), ``agent`` (the agent type), and dataset ``names`` read from
    the config's server blocks. Never raises: an unreadable config yields empty/None fields.
    """
    domain, description = read_config_metadata(config_path)
    try:
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception:
        raw = None

    value: Optional[str] = None
    resources_servers: List[str] = []
    agent: Optional[str] = None
    datasets: List[str] = []
    for group_key, server_name, server_config in iter_server_configs(raw):
        if group_key == "resources_servers":
            resources_servers.append(server_name)
            if value is None and server_config.get("value"):
                value = str(server_config["value"])
        elif group_key == "responses_api_agents":
            if agent is None:
                agent = server_name
            for dataset in server_config.get("datasets") or []:
                if isinstance(dataset, (dict, DictConfig)) and dataset.get("name"):
                    datasets.append(str(dataset["name"]))

    return {
        "domain": domain,
        "description": description,
        "value": value,
        "resources_servers": resources_servers,
        "agent": agent,
        "datasets": datasets,
    }
