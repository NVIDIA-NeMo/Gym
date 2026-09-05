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
"""Read-only catalog for manifest-backed and legacy environments and benchmarks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import yaml
from omegaconf import DictConfig, OmegaConf

from nemo_gym import PARENT_DIR, component_search_roots
from nemo_gym.benchmarks import _benchmark_config_name, _benchmark_config_paths
from nemo_gym.config_types import ConfigError
from nemo_gym.discovery import iter_server_configs, read_config_metadata
from nemo_gym.environment.manifest import EnvironmentManifest, ManifestError, load_manifest


ENVIRONMENTS_SUBDIR = "environments"
ENVIRONMENTS_DIR = PARENT_DIR / ENVIRONMENTS_SUBDIR
BENCHMARKS_SUBDIR = "benchmarks"
RESOURCES_SERVERS_SUBDIR = "resources_servers"
ENVIRONMENT_CONFIG_FILENAME = "config.yaml"
ENVIRONMENT_TOMBSTONE_FILENAME = ".nemo_gym_tombstone"
MANIFEST_FILENAME = "manifest.yaml"

CatalogKind = Literal["environment", "benchmark"]
CatalogStatus = Literal["experimental", "no-manifest"]


class RegistryError(ConfigError):
    """A catalog entry is invalid or cannot be selected unambiguously."""


@dataclass(frozen=True)
class EnvironmentCatalogEntry:
    """A discovered runnable unit and its lightweight catalog metadata."""

    # Preserve the historical EnvironmentEntry constructor prefix.
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
    resources_server_selector: Optional[str] = None


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
        raise RegistryError(f"Manifest '{manifest_path}' must be inside a named catalog directory.")
    return relative.as_posix()


def _manifest_entry(
    tree_dir: Path,
    manifest_path: Path,
    expected_kind: CatalogKind,
) -> EnvironmentCatalogEntry:
    expected_name = _path_identity(tree_dir, manifest_path)
    manifest: EnvironmentManifest = load_manifest(manifest_path)
    actual_kind = _enum_value(manifest.kind)
    if actual_kind != expected_kind:
        raise RegistryError(
            f"Manifest '{manifest_path}' declares kind '{actual_kind}', "
            f"but its catalog path requires '{expected_kind}'."
        )
    if manifest.name != expected_name:
        raise RegistryError(
            f"Manifest '{manifest_path}' declares name '{manifest.name}', "
            f"but its catalog path requires '{expected_name}'."
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


def _runnable_resource_config(
    config_path: Path,
) -> tuple[CatalogKind, Optional[str], Optional[str]] | None:
    """Read a resource-tree config that carries its own runnable workload."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None

    servers = tuple(iter_server_configs(raw))
    if not any(group == "resources_servers" for group, _name, _config in servers):
        return None
    datasets = [
        dataset
        for group, _name, config in servers
        if group == "responses_api_agents"
        for dataset in (config.get("datasets") or [])
        if isinstance(dataset, (dict, DictConfig))
    ]
    if not datasets:
        return None
    kind: CatalogKind = (
        "benchmark" if any(dataset.get("type") == "benchmark" for dataset in datasets) else "environment"
    )
    domain = next((str(config["domain"]) for _group, _name, config in servers if config.get("domain")), None)
    description = next(
        (str(config["description"]) for _group, _name, config in servers if config.get("description")),
        None,
    )
    return kind, domain, description


def _resource_workload_identity(resources_dir: Path, config_path: Path) -> tuple[str, str]:
    relative = config_path.relative_to(resources_dir)
    component = relative.parts[0]
    flavor = config_path.stem
    name = f"resources_servers/{component}" + (f"/{flavor}" if flavor != component else "")
    selector = component + (f"/{flavor}" if flavor != component else "")
    return name, selector


def _discover_resource_workloads(
    resources_dir: Path,
    *,
    claimed: frozenset[tuple[CatalogKind, str]] = frozenset(),
) -> Dict[tuple[CatalogKind, str], EnvironmentCatalogEntry]:
    """Discover legacy runnable workloads colocated with resources servers."""
    if not resources_dir.is_dir():
        return {}

    entries: Dict[tuple[CatalogKind, str], EnvironmentCatalogEntry] = {}
    for config_path in sorted(resources_dir.glob("*/configs/*.yaml")):
        metadata = _runnable_resource_config(config_path)
        if metadata is None:
            continue
        kind, domain, description = metadata
        name, selector = _resource_workload_identity(resources_dir, config_path)
        canonical_name = selector.rsplit("/", 1)[-1]
        if (kind, canonical_name) in claimed:
            continue
        key = (kind, name)
        if key in claimed or key in entries:
            continue
        values = {
            "name": name,
            "config_path": config_path,
            "path": config_path.parent.parent,
            "description": description,
            "domain": domain,
            "resources_server_selector": selector,
        }
        entries[key] = (
            EnvironmentEntry(**values) if kind == "environment" else EnvironmentCatalogEntry(kind=kind, **values)
        )
    return entries


def _legacy_config_paths(tree_dir: Path, kind: CatalogKind) -> Iterable[tuple[str, Path]]:
    if kind == "benchmark":
        for config_path in _benchmark_config_paths(tree_dir):
            if config_path.name != MANIFEST_FILENAME:
                yield _benchmark_config_name(config_path.relative_to(tree_dir)), config_path
        return

    for child in sorted(tree_dir.iterdir()):
        config_path = child / ENVIRONMENT_CONFIG_FILENAME
        tombstone_path = child / ENVIRONMENT_TOMBSTONE_FILENAME
        if child.is_dir() and config_path.is_file() and not tombstone_path.is_file():
            yield child.name, config_path


def _discover_registry_tree(
    tree_dir: Path,
    kind: CatalogKind,
    *,
    claimed: frozenset[tuple[CatalogKind, str]] = frozenset(),
) -> Dict[tuple[CatalogKind, str], EnvironmentCatalogEntry]:
    """Discover one catalog tree without importing or resolving runnable components."""
    if not tree_dir.is_dir():
        return {}

    entries: Dict[tuple[CatalogKind, str], EnvironmentCatalogEntry] = {}
    manifest_configs: set[Path] = set()
    for manifest_path in sorted(tree_dir.rglob(MANIFEST_FILENAME)):
        relative_path = manifest_path.relative_to(tree_dir)
        if kind == "environment" and (tree_dir / relative_path.parts[0] / ENVIRONMENT_TOMBSTONE_FILENAME).is_file():
            continue
        key = (kind, _path_identity(tree_dir, manifest_path))
        if key in claimed:
            continue
        try:
            entry = _manifest_entry(tree_dir, manifest_path, kind)
        except ManifestError:
            continue
        entries[key] = entry
        manifest_configs.add(entry.config_path.resolve())

    for name, config_path in _legacy_config_paths(tree_dir, kind):
        key = (kind, name)
        if key in claimed or key in entries or config_path.resolve() in manifest_configs:
            continue
        entries[key] = _legacy_entry(name, config_path, kind)
    return entries


def _discover_environments_in_dir(environments_dir: Path) -> Dict[str, EnvironmentEntry]:
    """Map environment name to its effective manifest or legacy entry under one directory."""
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
        discovered = _discover_registry_tree(root / ENVIRONMENTS_SUBDIR, "environment", claimed=claimed)
        environments.update(
            (name, entry) for (_kind, name), entry in discovered.items() if isinstance(entry, EnvironmentEntry)
        )
    return environments


def discover_environment_catalog() -> tuple[EnvironmentCatalogEntry, ...]:
    """Discover manifest and legacy runnable units across all workload locations."""
    entries: Dict[tuple[CatalogKind, str], EnvironmentCatalogEntry] = {}
    for root in component_search_roots():
        for kind, subdir in (("environment", ENVIRONMENTS_SUBDIR), ("benchmark", BENCHMARKS_SUBDIR)):
            discovered = _discover_registry_tree(root / subdir, kind, claimed=frozenset(entries))
            entries.update(discovered)
        entries.update(_discover_resource_workloads(root / RESOURCES_SERVERS_SUBDIR, claimed=frozenset(entries)))
    return tuple(sorted(entries.values(), key=lambda entry: (entry.name.casefold(), entry.name, entry.kind)))


def resolve_catalog_entry(
    name: str,
    kind: CatalogKind | str | None = None,
    *,
    entries: Iterable[EnvironmentCatalogEntry] | None = None,
) -> EnvironmentCatalogEntry:
    """Resolve a catalog name, requiring its kind only for a cross-kind collision."""
    selected_kind = _enum_value(kind)
    if selected_kind not in (None, "environment", "benchmark"):
        raise RegistryError(f"Unknown catalog kind '{selected_kind}'.")

    matches = [
        entry
        for entry in (discover_environment_catalog() if entries is None else entries)
        if entry.name == name and (selected_kind is None or entry.kind == selected_kind)
    ]
    if not matches:
        suffix = f" with kind '{selected_kind}'" if selected_kind else ""
        raise RegistryError(f"Unknown catalog entry '{name}'{suffix}.")
    if len(matches) > 1:
        kinds = ", ".join(sorted(entry.kind for entry in matches))
        raise RegistryError(f"Catalog name '{name}' is ambiguous ({kinds}); specify a kind.")
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
