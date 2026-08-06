# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static environment inspection used by ``gym env validate``.

The helpers in this module intentionally operate on an already-resolved Gym
configuration.  They never import an environment implementation, start Ray, or
contact a service.  This keeps the onboarding feedback loop deterministic and
cheap while still checking the parts that are knowable before execution:
component boundaries, capability compatibility, and dataset shape.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata as importlib_metadata
import inspect as inspect_module
import json
import os
import re
import stat
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym import _resolve_under_cwd_or_install, component_search_roots
from nemo_gym.config_types import ConfigError
from nemo_gym.credential_keys import is_credential_key
from nemo_gym.discovery import SERVER_ROLE_BY_GROUP, component_capability_declaration, iter_server_configs
from nemo_gym.environment_files import is_runtime_source_path, resolve_runtime_local_references
from nemo_gym.environment_manifest import (
    EnvironmentManifest,
    ManifestDataset,
    grading_modes_from_source,
    manifest_implied_capabilities,
    manifest_required_capabilities,
    parse_python_callable_reference,
)
from nemo_gym.prompt import PromptConfig, apply_prompt_to_row, load_prompt_config
from nemo_gym.repository_io import find_repository_root
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers.registry import get_provider_class


_EXTERNAL_LOOP_AGENTS = frozenset({"harbor_agent", "mini_swe_agent", "mini_swe_agent_2", "pinchbench", "tau2"})
# This legacy adapter customizes ``SimpleAgent.run`` only to materialize
# multimodal input; Gym's stock loop still owns the episode.  Keep its existing
# measured-loop classification while source-shape inference is used for newly
# scaffolded SimpleAgent subclasses.
_LEGACY_MEASURED_LOOP_AGENTS = frozenset({"labbench2_vlm_agent"})
_PROFILE_PINS = {
    "stock-loop": frozenset({"agent_server"}),
    "measured-loop": frozenset({"agent_server"}),
    "external-loop": frozenset(),
    "custom-driver": frozenset(),
}


class _DatasetResponseCreateParams(BaseModel):
    """Validate the JSON-level Responses request envelope without runtime imports."""

    model_config = ConfigDict(extra="forbid")

    background: bool | None = None
    include: list[str] | None = None
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: dict[str, str] | None = None
    model: str | None = None
    parallel_tool_calls: bool = True
    previous_response_id: str | None = None
    prompt: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    store: bool | None = None
    temperature: float | None = None
    text: dict[str, Any] | None = None
    tool_choice: str | dict[str, Any] = "auto"
    tools: list[dict[str, Any]] = Field(default_factory=list)
    top_logprobs: int | None = None
    top_p: float | None = None
    truncation: Literal["auto", "disabled"] | None = None
    user: str | None = None
    stream: Literal[False] | None = None


def pinned_component_roles(profile: str) -> frozenset[str]:
    """Return component roles fixed by an integration profile."""

    try:
        return _PROFILE_PINS[str(profile)]
    except KeyError as error:
        raise ConfigError(f"Unknown integration_profile={profile!r}.") from error


@dataclass(frozen=True)
class ComponentInspection:
    """One deployable selected by the resolved workload."""

    role: str
    instance: str
    implementation: str
    version: str | None
    entrypoint: str | None
    boundary: str | None
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "instance": self.instance,
            "implementation": self.implementation,
            "version": self.version,
            "entrypoint": self.entrypoint,
            "boundary": self.boundary,
            "requires": list(self.requires),
            "provides": list(self.provides),
            "pinned": self.pinned,
        }


@dataclass(frozen=True)
class ComponentProvenance:
    """Filesystem provenance for one selected runtime component."""

    role: str
    instance: str
    implementation: str
    source_directory: Path | None
    selected_config_path: Path | None = None
    entrypoint_source_directory: Path | None = None
    dependency_directories: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _SandboxProviderSelection:
    """One registered sandbox provider selected by a launched server."""

    instance: str
    consumer: str
    implementation: str
    config: dict[str, Any]
    default_metadata: dict[str, Any]
    provider_class: type[Any]
    source_file: Path | None


@dataclass(frozen=True)
class RolloutDriverProvenance:
    """Filesystem provenance for a selected custom rollout driver."""

    module_name: str
    function_name: str
    source_file: Path | None
    source_directory: Path | None
    dependency_directories: tuple[Path, ...] = ()


@dataclass(frozen=True)
class DatasetPreparationProvenance:
    """Filesystem provenance for one dataset preparation program."""

    reference: str
    source_file: Path | None
    source_directory: Path | None
    dependency_directories: tuple[Path, ...] = ()


@dataclass(frozen=True)
class DatasetInspection:
    """Static result for one dataset declared by an agent server."""

    name: str
    dataset_type: str
    path: Path
    rows: int | None
    status: str
    detail: str | None = None
    materialized_sample: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.dataset_type,
            "path": str(self.path),
            "rows": self.rows,
            "status": self.status,
            "detail": self.detail,
            "materialized_sample": self.materialized_sample,
        }


ResponsibilityOwnerRole = Literal["dataset", "resources_server", "agent_server", "model_server", "rollout_driver"]


@dataclass(frozen=True)
class ResponsibilityOwner:
    """One concrete owner of an onboarding responsibility."""

    role: ResponsibilityOwnerRole
    instance: str
    implementation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "instance": self.instance,
            "implementation": self.implementation,
        }


@dataclass(frozen=True)
class ResponsibilityMapping:
    """Resolved owners of the workload's authoring and runtime responsibilities."""

    task_preparation: tuple[ResponsibilityOwner, ...] = ()
    model_interaction: tuple[ResponsibilityOwner, ...] = ()
    tools_and_state: tuple[ResponsibilityOwner, ...] = ()
    verification: tuple[ResponsibilityOwner, ...] = ()
    rollout_coordination: tuple[ResponsibilityOwner, ...] = ()

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {
            "task_preparation": [owner.to_dict() for owner in self.task_preparation],
            "model_interaction": [owner.to_dict() for owner in self.model_interaction],
            "tools_and_state": [owner.to_dict() for owner in self.tools_and_state],
            "verification": [owner.to_dict() for owner in self.verification],
            "rollout_coordination": [owner.to_dict() for owner in self.rollout_coordination],
        }


@dataclass(frozen=True)
class WorkloadInspection:
    """Everything ``validate`` can report without provisioning compute."""

    profile: str
    components: tuple[ComponentInspection, ...]
    datasets: tuple[DatasetInspection, ...]
    config_paths: tuple[str, ...]
    composition_hash: str
    responsibilities: ResponsibilityMapping
    overrides: dict[str, Any] = field(default_factory=dict)
    fixed_constraints: tuple[str, ...] = field(default_factory=tuple)
    compatibility_decisions: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "components": [component.to_dict() for component in self.components],
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "config_paths": list(self.config_paths),
            "composition_hash": self.composition_hash,
            "responsibilities": self.responsibilities.to_dict(),
            "overrides": self.overrides,
            "fixed_constraints": list(self.fixed_constraints),
            "compatibility_decisions": list(self.compatibility_decisions),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CompositionMirror:
    """Manifest-owned view projected from the authoritative Hydra config."""

    resources_server: str | None
    agent_server: str | None
    model_server: str | None
    datasets: tuple[ManifestDataset, ...]
    rollout_driver: str | None
    grading_mode: str | None

    def to_manifest_update(self) -> dict[str, Any]:
        return {
            "resources_server": self.resources_server,
            "agent_server": self.agent_server,
            "model_server": self.model_server,
            "datasets": list(self.datasets) or None,
            "rollout_driver": self.rollout_driver,
            "grading_mode": self.grading_mode,
        }


def _as_plain_mapping(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        if not isinstance(value, dict):  # pragma: no cover - DictConfig is mapping-shaped here
            raise ConfigError("Resolved Gym configuration must be a mapping.")
        return value
    return dict(config)


def _string_set(value: object, *, component: str, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise ConfigError(f"Component '{component}' field '{field_name}' must be a string or list of strings.")

    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"Component '{component}' field '{field_name}' contains a non-string or empty value.")
        capability = item.strip()
        if capability not in normalized:
            normalized.append(capability)
    return tuple(normalized)


def _simple_agent_source_profile(
    implementation: str,
    server_config: Mapping[str, Any],
    config_paths: Sequence[Path],
) -> str | None:
    """Infer the profile from a local ``SimpleAgent`` subclass extension point.

    The generated measured-loop scaffold owns the turn-level ``responses``
    method.  The external-loop scaffold leaves that endpoint inherited and owns
    the complete episode through ``run``.  Reading this shape statically keeps
    classification independent of a mutable config label and imports no runtime
    implementation.
    """

    source_directory, _selected_config = resolve_component_source_directory(
        "responses_api_agents",
        implementation,
        config_paths,
    )
    if source_directory is None:
        return None
    entrypoint = _resolve_component_runtime_entrypoint(source_directory, server_config)
    if entrypoint is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(entrypoint.read_text(encoding="utf-8"), filename=str(entrypoint))
    except (OSError, SyntaxError, UnicodeError):
        return None

    inferred: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {
            base.id if isinstance(base, ast.Name) else base.attr
            for base in node.bases
            if isinstance(base, (ast.Name, ast.Attribute))
        }
        if "SimpleAgent" not in base_names:
            continue
        methods = {member.name for member in node.body if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if "responses" in methods:
            inferred.add("measured-loop")
        elif "run" in methods:
            inferred.add("external-loop")

    if len(inferred) > 1:
        raise ConfigError(
            f"Agent implementation {implementation!r} has conflicting measured-loop and external-loop source shapes."
        )
    return next(iter(inferred), None)


def infer_integration_profile(config: Mapping[str, Any] | DictConfig) -> str:
    """Classify the four existing Gym integration shapes from resolved config.

    ``custom-driver`` has a direct config signal. Existing external adapters are
    preserved by implementation identity, while newly scaffolded SimpleAgent
    subclasses are classified from the method they implement. The stock loop is
    identified by ``simple_agent``; remaining legacy in-process agents are
    measured loops. Agent-owned ``integration_profile`` labels are deliberately
    ignored so the declaration being checked cannot attest itself.
    """

    plain = _as_plain_mapping(config)
    if plain.get("rollout_collection_driver"):
        return "custom-driver"

    agent_configs = [
        (implementation, server_config)
        for role, _instance, implementation, server_config in _server_instances(plain)
        if role == "agent_server"
    ]
    agent_implementations = [implementation for implementation, _server_config in agent_configs]
    if any(agent in _EXTERNAL_LOOP_AGENTS for agent in agent_implementations):
        return "external-loop"
    if not agent_implementations or all(agent == "simple_agent" for agent in agent_implementations):
        return "stock-loop"
    config_paths = _resolved_config_paths(plain)
    for implementation, server_config in agent_configs:
        if implementation in {"simple_agent", *_LEGACY_MEASURED_LOOP_AGENTS}:
            continue
        if _simple_agent_source_profile(implementation, server_config, config_paths) == "external-loop":
            return "external-loop"
    return "measured-loop"


def _server_instances(
    config: Mapping[str, Any],
    *,
    strict_launch: bool = False,
) -> Iterable[tuple[str, str, str, Mapping[str, Any]]]:
    """Yield the component each top-level instance actually launches.

    ``RunHelper`` dispatches the first server group and first implementation in
    each instance.  Inspection must use the same rule; walking every nested
    group would validate and hash components that never start.
    """

    for instance, value in config.items():
        if not isinstance(value, Mapping) or not value:
            continue
        group = next(iter(value))
        role = SERVER_ROLE_BY_GROUP.get(str(group))
        implementations = value[group]
        if role is None:
            if strict_launch and isinstance(implementations, Mapping) and implementations:
                implementation = next(iter(implementations))
                server_config = implementations[implementation]
                if isinstance(server_config, Mapping) and "entrypoint" in server_config:
                    raise ConfigError(
                        f"Server instance {str(instance)!r} uses unsupported server group {str(group)!r}."
                    )
            continue
        if not isinstance(implementations, Mapping) or not implementations:
            if strict_launch:
                raise ConfigError(f"Server instance {str(instance)!r} has no configured {str(group)} implementation.")
            continue
        implementation = next(iter(implementations))
        server_config = implementations[implementation]
        if strict_launch and not isinstance(server_config, Mapping):
            raise ConfigError(
                f"Server instance {str(instance)!r} has malformed {str(group)} implementation {str(implementation)!r}."
            )
        if isinstance(server_config, Mapping) and "entrypoint" in server_config:
            yield role, str(instance), str(implementation), server_config


def _sandbox_provider_source(provider_class: type[Any]) -> Path | None:
    try:
        raw_source = inspect_module.getsourcefile(provider_class) or inspect_module.getfile(provider_class)
    except (OSError, TypeError):
        return None
    if not raw_source:
        return None
    source = Path(raw_source)
    return source.resolve() if source.is_file() else None


def _sandbox_provider_package_directory(selection: _SandboxProviderSelection) -> Path | None:
    """Return the smallest import package that owns a selected provider."""

    source_file = selection.source_file
    package = Path(__file__).resolve().parent / "sandbox" / "providers" / selection.implementation
    if source_file is not None and package.is_dir():
        try:
            source_file.resolve().relative_to(package.resolve())
        except ValueError:
            pass
        else:
            return package
    if source_file is None:
        return None
    return _module_source_package_directory(selection.provider_class.__module__, source_file)


def _selected_sandbox_providers(config: Mapping[str, Any]) -> tuple[_SandboxProviderSelection, ...]:
    """Resolve sandbox selections exactly as provider-backed servers do at runtime."""

    selections: list[_SandboxProviderSelection] = []
    seen: set[tuple[str, str]] = set()
    for role, instance, implementation, server_config in _server_instances(config):
        if "sandbox_provider" not in server_config or server_config.get("sandbox_provider") is None:
            continue
        raw_provider = server_config["sandbox_provider"]
        consumer = f"{role}:{instance}/{implementation}"
        try:
            provider_config = resolve_provider_config(raw_provider, config)
            default_metadata = resolve_provider_metadata(raw_provider, config)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{consumer} has an invalid sandbox_provider: {error}") from error

        provider_name = next(iter(provider_config))
        try:
            provider_class = get_provider_class(provider_name)
        except Exception as error:
            raise ConfigError(
                f"Sandbox provider {provider_name!r} selected by {consumer} is not registered or could not be loaded: "
                f"{error}"
            ) from error

        provider_instance = raw_provider if isinstance(raw_provider, str) else f"{instance}.sandbox_provider"
        key = (provider_instance, provider_name)
        if key in seen:
            continue
        seen.add(key)
        selections.append(
            _SandboxProviderSelection(
                instance=provider_instance,
                consumer=consumer,
                implementation=provider_name,
                config=provider_config,
                default_metadata=default_metadata,
                provider_class=provider_class,
                source_file=_sandbox_provider_source(provider_class),
            )
        )
    return tuple(selections)


def _boundary(server_config: Mapping[str, Any]) -> str | None:
    host = server_config.get("host")
    port = server_config.get("port")
    if host is not None and port is not None:
        return f"http://{host}:{port}"
    if host is not None:
        return str(host)
    return None


_NON_SCORE_COMPONENT_KEYS = frozenset(
    {
        "aliases",
        "description",
        "host",
        "integration_profile",
        "license",
        "licensing",
        "modality",
        "num_workers",
        "port",
        "value",
        "verified",
        "verified_url",
    }
)
_COMPONENT_SUBDIRS = {role: group for group, role in SERVER_ROLE_BY_GROUP.items()}
_NON_SCORE_COMPONENT_DIRS = frozenset(
    {
        ".git",
        ".gym",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "cache",
        "results",
        "tests",
        "venv",
    }
)
_NON_SCORE_COMPONENT_FILES = frozenset(
    {
        ".coverage",
        ".ds_store",
        ".gitignore",
        "manifest.yaml",
    }
)


def _bounded_preview(value: Any, *, depth: int = 0) -> Any:
    """Return a small JSON-ready preview without exposing credential values."""

    if depth >= 6:
        return "<max depth>"
    if isinstance(value, Mapping):
        preview: dict[str, Any] = {}
        entries = list(value.items())
        for key, item in entries[:24]:
            name = str(key)
            preview[name] = "<redacted>" if is_credential_key(name) else _bounded_preview(item, depth=depth + 1)
        if len(entries) > 24:
            preview["<omitted>"] = len(entries) - 24
        return preview
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        preview = [_bounded_preview(item, depth=depth + 1) for item in items[:8]]
        if len(items) > 8:
            preview.append(f"<{len(items) - 8} items omitted>")
        return preview
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + f"… <{len(value) - 500} chars omitted>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)[:500]


def _materialized_responses_sample(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    params = rows[0].get("responses_create_params")
    if not isinstance(params, Mapping):
        return None
    preview = _bounded_preview(params)
    assert isinstance(preview, dict)
    return {"responses_create_params": preview}


def is_score_affecting_component_path(path: Path) -> bool:
    """Whether a component-local path can affect the deployed implementation.

    The selected, fully resolved server config is already hashed separately, so
    sibling config flavors are excluded. Tests, generated caches, and repository
    documentation are likewise not part of a deployed component. Everything
    else is included deliberately: helper modules, prompts, vendored code,
    requirements, and data may all change runtime or scoring behavior.
    """

    # Top-level YAML files in ``configs/`` are selectable config flavors; the
    # selected flavor is represented by its fully resolved server config below.
    # Other files below ``configs/`` can still be runtime inputs (templates,
    # scripts, assets) and must remain part of the deployed tree digest.
    if len(path.parts) == 2 and path.parts[0].casefold() == "configs" and path.suffix.casefold() in {".yaml", ".yml"}:
        return False
    parent_parts = path.parts[:-1]
    if any(part.casefold() in _NON_SCORE_COMPONENT_DIRS or part.endswith(".egg-info") for part in parent_parts):
        return False
    name = path.name.casefold()
    if name in _NON_SCORE_COMPONENT_FILES or name.startswith(("license", "readme")):
        return False
    return path.suffix.casefold() not in {".pyc", ".pyo"}


def _resolved_config_paths(config: Mapping[str, Any]) -> tuple[Path, ...]:
    raw_paths = config.get("config_paths") or []
    if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes, bytearray)):
        return ()
    paths: list[Path] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, (str, Path)) or not str(raw_path):
            continue
        path = Path(raw_path)
        resolved = path if path.is_absolute() else _resolve_under_cwd_or_install(path)
        resolved = resolved.resolve()
        if resolved not in paths:
            paths.append(resolved)
    return tuple(paths)


def _component_directory_from_config_path(
    config_path: Path,
    group: str,
    implementation: str,
) -> Path | None:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception:
        return None
    if not any(
        discovered_group == group and str(name) == implementation
        for discovered_group, name, _server_config in iter_server_configs(raw)
    ):
        return None
    for root in component_search_roots():
        try:
            relative = config_path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if len(relative.parts) >= 3 and relative.parts[0] == group:
            return Path(os.path.abspath(root / group / relative.parts[1]))
    return None


def _config_defines_component(config_path: Path, group: str, implementation: str) -> bool:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception:
        return False
    return any(
        discovered_group == group and str(name) == implementation
        for discovered_group, name, _server_config in iter_server_configs(raw)
    )


def _registry_component_directory_for_path(path: Path, *, root: Path | None = None) -> Path | None:
    resolved_path = path.resolve()
    roots = (root,) if root is not None else tuple(component_search_roots())
    for candidate_root in roots:
        resolved_root = candidate_root.resolve()
        try:
            relative = resolved_path.relative_to(resolved_root)
        except ValueError:
            continue
        if len(relative.parts) < 3 or relative.parts[0] not in _COMPONENT_SUBDIRS.values():
            return None
        directory = resolved_root / relative.parts[0] / relative.parts[1]
        return Path(os.path.abspath(directory)) if directory.is_dir() else None
    return None


def resolve_component_source_directory(
    group: str,
    implementation: str,
    config_paths: Sequence[str | Path] = (),
) -> tuple[Path | None, Path | None]:
    """Resolve an implementation to its defining component directory and config.

    Config provenance wins over the conventional ``registry/implementation``
    layout so aliases such as ``langgraph_agent/reflection_agent`` bind to the
    source that actually defines them. Ambiguous aliases fail closed.
    """

    if group not in _COMPONENT_SUBDIRS.values():
        raise ConfigError(f"Unknown component registry {group!r}.")

    declared: list[tuple[Path, Path]] = []
    defining_paths: list[Path] = []
    for raw_path in config_paths:
        path = Path(raw_path)
        path = path if path.is_absolute() else _resolve_under_cwd_or_install(path)
        path = path.resolve()
        if _config_defines_component(path, group, implementation) and path not in defining_paths:
            defining_paths.append(path)
        directory = _component_directory_from_config_path(path, group, implementation)
        if directory is not None and (directory, path) not in declared:
            declared.append((directory, path))

    directories = tuple(dict.fromkeys(directory for directory, _path in declared))
    if len(directories) > 1:
        rendered = ", ".join(str(path) for path in directories)
        raise ConfigError(
            f"Component implementation {implementation!r} is defined by multiple selected {group} directories: "
            f"{rendered}. Select one unambiguous config flavor."
        )
    if directories:
        selected_path = next(path for directory, path in reversed(declared) if directory == directories[0])
        return directories[0], selected_path

    nominal = _resolve_under_cwd_or_install(Path(group) / implementation)
    if nominal.is_dir():
        return Path(os.path.abspath(nominal)), defining_paths[-1] if defining_paths else None
    if not config_paths:
        return None, None

    discovered: list[tuple[Path, Path]] = []
    for root in component_search_roots():
        registry = root / group
        root_matches: list[tuple[Path, Path]] = []
        for pattern in ("*/configs/*.yaml", "*/configs/*.yml"):
            for path in sorted(registry.glob(pattern)):
                directory = _component_directory_from_config_path(path, group, implementation)
                if directory is not None and (directory, path.resolve()) not in root_matches:
                    root_matches.append((directory, path.resolve()))
        if root_matches:
            discovered = root_matches
            break

    directories = tuple(dict.fromkeys(directory for directory, _path in discovered))
    if len(directories) > 1:
        rendered = ", ".join(str(path) for path in directories)
        raise ConfigError(
            f"Component implementation {implementation!r} is declared by multiple {group} directories: "
            f"{rendered}. Pass the intended config explicitly."
        )
    if not directories:
        return None, None
    selected_path = next(path for directory, path in reversed(discovered) if directory == directories[0])
    return directories[0], selected_path


def _registry_imports(source_path: Path) -> set[tuple[str, str]]:
    """Return absolute imports rooted at a Gym component registry."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError, UnicodeError):
        return set()

    imports: set[tuple[str, str]] = set()
    registries = frozenset(_COMPONENT_SUBDIRS.values())

    def add(module: str) -> None:
        normalized = module.replace("\\", "/")
        parts = normalized.split("/") if "/" in normalized else normalized.split(".")
        if len(parts) >= 2 and parts[0] in registries and parts[1]:
            imports.add((parts[0], parts[1]))

    def static_path_parts(node: ast.AST) -> list[str]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return [*static_path_parts(node.left), *static_path_parts(node.right)]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [part for part in Path(node.value).parts if part not in {"", "."}]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
            return [part for argument in node.args for part in static_path_parts(argument)]
        return []

    def add_path(parts: Sequence[str]) -> None:
        for index, part in enumerate(parts[:-1]):
            if part in registries and parts[index + 1]:
                imports.add((part, parts[index + 1]))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            add(module)
            if module in registries:
                for alias in node.names:
                    add(f"{module}.{alias.name}")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            add_path(static_path_parts(node))
    return imports


def _module_source_package_directory(module_name: str, source_file: Path) -> Path | None:
    """Resolve the smallest supported package tree that owns ``source_file``."""

    module_parts = tuple(part for part in module_name.split(".") if part)
    if not module_parts or any(not part.isidentifier() for part in module_parts):
        return None
    source = source_file.resolve()
    import_root: Path | None = None
    for candidate_root in source.parents:
        module_path = candidate_root.joinpath(*module_parts)
        source_candidates = (module_path.with_suffix(".py"), module_path / "__init__.py")
        if any(candidate.resolve() == source for candidate in source_candidates):
            import_root = candidate_root
            break
        if source.parent == module_path.parent and source.name.startswith(module_path.name + "."):
            # Native extension modules include the ABI in their filename.
            import_root = candidate_root
            break
    if import_root is None:
        return None

    registry_roots = {*_COMPONENT_SUBDIRS.values(), "benchmarks", "environments"}
    if len(module_parts) >= 2 and module_parts[0] in registry_roots:
        package = import_root / module_parts[0] / module_parts[1]
    else:
        package = import_root / module_parts[0]
    if not package.is_dir():
        return None
    try:
        source.relative_to(package.resolve())
    except ValueError:
        return None
    return package


def _first_party_package_root() -> Path:
    """Return the installed source root for the ``nemo_gym`` package."""

    return Path(__file__).resolve().parent


def _source_module_name(source_path: Path, package_root: Path) -> str | None:
    try:
        relative = source_path.resolve().relative_to(package_root.resolve())
    except ValueError:
        return None
    if relative.name == "__init__.py":
        parts = relative.parent.parts
    elif relative.suffix == ".py":
        parts = relative.with_suffix("").parts
    else:
        return None
    return ".".join((package_root.name, *parts))


@lru_cache(maxsize=4096)
def _cached_first_party_imports(
    source_path: str,
    package_root: str,
    _size: int,
    _mtime_ns: int,
    _ctime_ns: int,
) -> frozenset[str]:
    source = Path(source_path)
    root = Path(package_root)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError, UnicodeError):
        return frozenset()

    package_name = root.name
    current_module = _source_module_name(source, root)
    current_package = None
    if current_module is not None:
        current_package = current_module if source.name == "__init__.py" else current_module.rpartition(".")[0]
    imports: set[str] = set()

    def add(module: str) -> None:
        if module == package_name or module.startswith(package_name + "."):
            imports.add(module)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                if current_package is None:
                    continue
                package_parts = current_package.split(".")
                parents = node.level - 1
                if parents >= len(package_parts):
                    continue
                base_parts = package_parts[: len(package_parts) - parents]
                if module:
                    base_parts.extend(module.split("."))
                module = ".".join(base_parts)
            add(module)
            if module:
                for alias in node.names:
                    if alias.name != "*":
                        add(f"{module}.{alias.name}")
        elif isinstance(node, ast.Call) and node.args:
            function_name = None
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if function_name in {"__import__", "import_module"}:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    add(argument.value)
    return frozenset(imports)


def _first_party_imports(source_path: Path, package_root: Path) -> set[str]:
    """Return statically discoverable imports under one first-party package."""

    try:
        source_stat = source_path.stat()
    except OSError:
        return set()
    return set(
        _cached_first_party_imports(
            str(source_path.resolve()),
            str(package_root.resolve()),
            source_stat.st_size,
            source_stat.st_mtime_ns,
            source_stat.st_ctime_ns,
        )
    )


def _first_party_module_sources(module_name: str, package_root: Path) -> tuple[Path, ...]:
    """Resolve a first-party module and the package initializers Python executes."""

    package_name = package_root.name
    if module_name == package_name:
        relative_parts: tuple[str, ...] = ()
    elif module_name.startswith(package_name + "."):
        relative_parts = tuple(module_name.removeprefix(package_name + ".").split("."))
    else:
        return ()

    sources: list[Path] = []

    def add(path: Path) -> None:
        if path.is_file() and path not in sources:
            sources.append(path)

    add(package_root / "__init__.py")
    for length in range(1, len(relative_parts)):
        add(package_root.joinpath(*relative_parts[:length]) / "__init__.py")
    if relative_parts:
        target = package_root.joinpath(*relative_parts)
        package_source = target / "__init__.py"
        if package_source.is_file():
            add(package_source)
        else:
            add(target.with_suffix(".py"))
    return tuple(sources)


def _first_party_runtime_dependency_files(source_paths: Iterable[Path]) -> tuple[Path, ...]:
    """Resolve the transitive first-party Python files used by runtime sources."""

    package_root = _first_party_package_root()
    resolved_package_root = package_root.resolve()
    seed_sources = {path.resolve() for path in source_paths if path.is_file() and path.suffix == ".py"}
    queue = sorted(seed_sources, key=str)
    scanned: set[Path] = set()
    dependencies: set[Path] = set()
    while queue:
        source_path = queue.pop(0)
        if source_path in scanned:
            continue
        scanned.add(source_path)
        for module_name in sorted(_first_party_imports(source_path, package_root)):
            for dependency in _first_party_module_sources(module_name, package_root):
                resolved_dependency = dependency.resolve()
                try:
                    resolved_dependency.relative_to(resolved_package_root)
                except ValueError as error:
                    raise ConfigError(
                        f"First-party runtime dependency '{dependency}' resolves outside '{package_root}'."
                    ) from error
                if resolved_dependency not in seed_sources:
                    dependencies.add(resolved_dependency)
                if resolved_dependency not in scanned:
                    queue.append(resolved_dependency)
    return tuple(
        sorted(
            dependencies,
            key=lambda path: path.relative_to(package_root.resolve()).as_posix(),
        )
    )


def _first_party_dependency_digests(source_paths: Iterable[Path], *, key_prefix: str) -> dict[str, str]:
    package_root = _first_party_package_root().resolve()
    digests: dict[str, str] = {}
    for source_path in _first_party_runtime_dependency_files(source_paths):
        digest = _referenced_file_digest(source_path)
        if digest is None:
            raise ConfigError(f"First-party runtime dependency '{source_path}' is missing or unreadable.")
        relative = source_path.relative_to(package_root).as_posix()
        digests[f"{key_prefix}:{package_root.name}/{relative}"] = digest
    return digests


def _configured_registry_imports(value: object) -> set[tuple[str, str]]:
    """Return component modules named as import strings in resolved config."""

    registries = frozenset(_COMPONENT_SUBDIRS.values())
    imports: set[tuple[str, str]] = set()
    if isinstance(value, Mapping):
        for nested in value.values():
            imports.update(_configured_registry_imports(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            imports.update(_configured_registry_imports(nested))
    elif isinstance(value, str):
        module = value.partition(":")[0]
        parts = module.split(".")
        if len(parts) >= 2 and parts[0] in registries and parts[1]:
            imports.add((parts[0], parts[1]))
    return imports


def component_runtime_dependency_directories(
    component_dir: Path,
    *,
    configured_values: object = None,
) -> tuple[Path, ...]:
    """Resolve transitive local component imports from source and config."""

    root = Path(os.path.abspath(component_dir))
    seen = {root.resolve()}
    queue = [root]
    dependencies: list[Path] = []
    while queue:
        current = queue.pop(0)
        imports: set[tuple[str, str]] = set()
        if current == root:
            imports.update(_configured_registry_imports(configured_values))
        for source_path in sorted(current.rglob("*.py")):
            if source_path.is_file() and is_score_affecting_component_path(source_path.relative_to(current)):
                imports.update(_registry_imports(source_path))
        for group, directory_name in sorted(imports):
            dependency = _resolve_under_cwd_or_install(
                Path(group) / directory_name,
                validator=Path.is_dir,
            )
            if not dependency.is_dir():
                continue
            dependency = Path(os.path.abspath(dependency))
            resolved_dependency = dependency.resolve()
            if resolved_dependency in seen:
                continue
            seen.add(resolved_dependency)
            dependencies.append(dependency)
            queue.append(dependency)
    return tuple(dependencies)


def resolve_component_provenance(config: Mapping[str, Any] | DictConfig) -> tuple[ComponentProvenance, ...]:
    """Resolve source and transitive local dependencies for launched components."""

    plain = _as_plain_mapping(config)
    config_paths = _resolved_config_paths(plain)
    provenance: list[ComponentProvenance] = []
    for role, instance, implementation, server_config in _server_instances(plain):
        group = _COMPONENT_SUBDIRS[role]
        source_directory, selected_config_path = resolve_component_source_directory(
            group,
            implementation,
            config_paths,
        )
        entrypoint_source_directory: Path | None = None
        dependencies: list[Path] = []
        if source_directory is not None:
            dependencies.extend(
                component_runtime_dependency_directories(source_directory, configured_values=server_config)
            )
            entrypoint = _resolve_component_runtime_entrypoint(source_directory, server_config)
            entrypoint_owner = _registry_component_directory_for_path(entrypoint) if entrypoint is not None else None
            selected_config_owner = (
                _registry_component_directory_for_path(selected_config_path)
                if selected_config_path is not None
                else None
            )
            if entrypoint_owner == source_directory:
                entrypoint_source_directory = source_directory
            elif entrypoint_owner is not None and entrypoint_owner == selected_config_owner:
                entrypoint_source_directory = entrypoint_owner
            if entrypoint_source_directory is not None and entrypoint_source_directory != source_directory:
                if entrypoint_source_directory not in dependencies:
                    dependencies.append(entrypoint_source_directory)
                for dependency in component_runtime_dependency_directories(
                    entrypoint_source_directory,
                    configured_values=server_config,
                ):
                    if dependency != source_directory and dependency not in dependencies:
                        dependencies.append(dependency)
        provenance.append(
            ComponentProvenance(
                role=role,
                instance=instance,
                implementation=implementation,
                source_directory=source_directory,
                selected_config_path=selected_config_path,
                entrypoint_source_directory=entrypoint_source_directory,
                dependency_directories=tuple(dependencies),
            )
        )
    return tuple(provenance)


def _uses_excluded_component_directory(path: Path) -> bool:
    return any(
        part.casefold() in _NON_SCORE_COMPONENT_DIRS or part.casefold().endswith(".egg-info")
        for part in path.parts[:-1]
    )


def _excluded_test_import(source_path: Path, component_dir: Path) -> str | None:
    """Return a local excluded-tests import used by one deployed Python source.

    Test trees stay outside normal component hashes. Rejecting a runtime import
    from that tree prevents a component from executing unhashed Python while
    preserving the useful property that ordinary test-only edits do not drift a
    published composition lock.
    """

    tests_dir = component_dir / "tests"
    if not tests_dir.is_dir():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError, UnicodeError):
        # Syntax/runtime validation remains the server launcher's job. This
        # check only rejects imports that can be identified safely.
        return None

    registry = component_dir.parent.name
    implementation = component_dir.name

    def references_tests(module: str, *, relative: bool = False) -> bool:
        parts = tuple(part for part in module.split(".") if part)
        if not parts:
            return False
        if (relative or parts[0] == "tests") and parts[0] == "tests":
            return True
        return len(parts) >= 3 and parts[:3] == (registry, implementation, "tests")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if references_tests(alias.name):
                    return alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            candidates = [module]
            candidates.extend(f"{module}.{alias.name}" if module else alias.name for alias in node.names)
            for candidate in candidates:
                if references_tests(candidate, relative=node.level > 0):
                    prefix = "." * node.level
                    return prefix + candidate
    return None


def _resolve_component_runtime_entrypoint(
    component_dir: Path,
    server_config: Mapping[str, Any],
    *,
    required: bool = False,
    component_label: str = "Component",
    entrypoint_source_directory: Path | None = None,
) -> Path | None:
    """Resolve one component entrypoint, optionally requiring local source."""

    if not component_dir.is_dir():
        if required:
            raise ConfigError(f"{component_label} source directory does not exist at '{component_dir}'.")
        return None
    raw_entrypoint = server_config.get("entrypoint")
    if not isinstance(raw_entrypoint, str) or not raw_entrypoint:
        if required:
            raise ConfigError(f"{component_label} does not declare a Python entrypoint.")
        return None
    entrypoint = Path(raw_entrypoint)
    if entrypoint.is_absolute():
        raise ConfigError(f"Component entrypoint '{entrypoint}' must be relative to '{component_dir}'.")
    try:
        resolved_entrypoint = (component_dir / entrypoint).resolve()
    except (OSError, RuntimeError) as error:
        if required:
            raise ConfigError(
                f"Could not resolve {component_label.casefold()} entrypoint '{entrypoint}' below '{component_dir}': "
                f"{error}."
            ) from error
        return None
    if not resolved_entrypoint.is_file():
        if required:
            raise ConfigError(
                f"{component_label} entrypoint '{entrypoint}' does not exist at '{resolved_entrypoint}'."
            )
        return None

    component_root = component_dir.resolve() if required else component_dir.parent.parent.resolve()
    try:
        root_relative = resolved_entrypoint.relative_to(component_root)
    except ValueError as error:
        root_relative = None
        if required and entrypoint_source_directory is not None:
            explicit_root = entrypoint_source_directory.resolve()
            try:
                root_relative = resolved_entrypoint.relative_to(explicit_root)
            except ValueError:
                pass
        if root_relative is None:
            boundary = "component source directory" if required else "component search root"
            raise ConfigError(
                f"Component entrypoint '{resolved_entrypoint}' resolves outside {boundary} '{component_root}'."
            ) from error
    if _uses_excluded_component_directory(root_relative):
        raise ConfigError(
            f"Component entrypoint '{resolved_entrypoint}' is under an excluded tests/cache tree and would not "
            "be covered by the composition lock. Move runtime code into the deployed component source tree."
        )
    return resolved_entrypoint


def _validate_component_runtime_sources(
    component_dir: Path,
    server_config: Mapping[str, Any],
) -> Path | None:
    """Validate that configured runtime Python is covered by the source lock."""

    resolved_entrypoint = _resolve_component_runtime_entrypoint(component_dir, server_config)
    if resolved_entrypoint is None:
        # Legacy inspection permits partial configs whose source is not installed.
        return None

    sources = {
        path
        for path in component_dir.rglob("*.py")
        if path.is_file() and is_score_affecting_component_path(path.relative_to(component_dir))
    }
    sources.add(resolved_entrypoint)
    for source_path in sorted(sources):
        imported = _excluded_test_import(source_path, component_dir)
        if imported is not None:
            raise ConfigError(
                f"Runtime source '{source_path}' imports {imported!r} from the excluded tests tree. "
                "Move the imported runtime helper into the deployed component source tree."
            )
    return resolved_entrypoint


@lru_cache(maxsize=512)
def _cached_component_tree_digest(
    component_dir: str,
    facts: tuple[tuple[str, str, int, int, int, int, str], ...],
) -> str:
    """Hash the score-affecting contents of a component tree.

    File stat facts form the cache key only; the resulting digest contains paths,
    executable bits, and deployed file bytes, never timestamps. Safe in-tree file
    links are normalized to their target bytes. This keeps hashes reproducible across clones while avoiding repeated reads when
    many environments reuse the same component in one catalog/CI process.
    """

    root = Path(component_dir)
    digest = hashlib.sha256(b"nemo-gym-component-tree-v1\0")
    for relative, kind, _size, _mtime_ns, _ctime_ns, mode, link_target in facts:
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(kind.encode())
        digest.update(b"\0")
        digest.update(str(mode).encode())
        digest.update(b"\0")
        try:
            digest.update((root / relative).read_bytes())
        except OSError as error:
            raise ConfigError(f"Could not hash component source '{root / relative}': {error}.") from error
        digest.update(b"\0")
    return digest.hexdigest()


def _component_tree_digest(component_dir: Path, *, environment_package: bool = False) -> str | None:
    """Return a reproducible digest of deployed files below ``component_dir``."""

    if component_dir.is_symlink() or component_dir.parent.is_symlink():
        raise ConfigError(
            f"Component source '{component_dir}' is reached through a symbolic-link component or registry "
            "directory. Keep deployed component sources inside their selected component tree."
        )
    if not component_dir.is_dir():
        return None
    resolved_component_dir = component_dir.resolve()
    facts: list[tuple[str, str, int, int, int, int, str]] = []

    def record(path: Path, relative_path: Path) -> None:
        relative = relative_path.as_posix()
        if path.is_symlink():
            try:
                resolved_target = path.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise ConfigError(
                    f"Component source '{component_dir}' contains an unresolvable deployed symlink "
                    f"'{relative}': {error}."
                ) from error
            try:
                resolved_target.relative_to(resolved_component_dir)
            except ValueError as error:
                raise ConfigError(
                    f"Component source '{component_dir}' contains deployed symlink '{relative}' whose target "
                    f"'{resolved_target}' is outside the component tree. Copy the dependency into the component "
                    "or declare it through a score-affecting config reference."
                ) from error
            if not resolved_target.is_file():
                raise ConfigError(
                    f"Component source '{component_dir}' contains deployed directory symlink '{relative}'. "
                    "Use a real directory so every deployed target is hashed unambiguously."
                )
            target_stat = resolved_target.stat()
            facts.append(
                (
                    relative,
                    "file",
                    target_stat.st_size,
                    target_stat.st_mtime_ns,
                    target_stat.st_ctime_ns,
                    stat.S_IMODE(target_stat.st_mode),
                    "",
                )
            )
        elif path.is_file():
            file_stat = path.stat()
            facts.append(
                (
                    relative,
                    "file",
                    file_stat.st_size,
                    file_stat.st_mtime_ns,
                    file_stat.st_ctime_ns,
                    stat.S_IMODE(file_stat.st_mode),
                    "",
                )
            )

    try:
        for current, directory_names, file_names in os.walk(component_dir, topdown=True, followlinks=False):
            current_path = Path(current)
            relative_root = current_path.relative_to(component_dir)
            retained_directories: list[str] = []
            for directory_name in sorted(directory_names):
                relative_path = relative_root / directory_name
                normalized = directory_name.casefold()
                if normalized in _NON_SCORE_COMPONENT_DIRS or normalized.endswith(".egg-info"):
                    continue
                path = current_path / directory_name
                if path.is_symlink():
                    record(path, relative_path)
                else:
                    retained_directories.append(directory_name)
            directory_names[:] = retained_directories
            for file_name in sorted(file_names):
                relative_path = relative_root / file_name
                if environment_package and relative_path == Path("config.yaml"):
                    continue
                if is_score_affecting_component_path(relative_path):
                    record(current_path / file_name, relative_path)
    except OSError as error:
        raise ConfigError(f"Could not inventory component source '{component_dir}': {error}.") from error
    facts.sort(key=lambda fact: fact[0])
    return _cached_component_tree_digest(str(component_dir.resolve()), tuple(facts))


def _score_affecting_value(value: Any, *, omit_component_metadata: bool = False) -> Any:
    """Return a JSON-ready value without runtime allocation or credential noise."""

    if isinstance(value, Mapping):
        return {
            str(key): _score_affecting_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not (omit_component_metadata and str(key).casefold() in _NON_SCORE_COMPONENT_KEYS)
            and not is_credential_key(key)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_score_affecting_value(item) for item in value]
    return value


def _component_content_version(
    role: str,
    implementation: str,
    server_config: Mapping[str, Any],
    provenance: ComponentProvenance | None = None,
) -> str:
    declared = server_config.get("version") or server_config.get("revision")
    digest = hashlib.sha256(
        json.dumps(
            {
                "role": role,
                "implementation": implementation,
                "declared_version": declared,
                "config": _score_affecting_value(server_config, omit_component_metadata=True),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    )
    subdir = _COMPONENT_SUBDIRS[role]
    component_dir = (
        provenance.source_directory
        if provenance is not None and provenance.source_directory is not None
        else _resolve_under_cwd_or_install(Path(subdir) / implementation)
    )
    runtime_entrypoint = _validate_component_runtime_sources(component_dir, server_config)
    if runtime_entrypoint is not None:
        try:
            digest.update(b"runtime-entrypoint\0")
            digest.update(runtime_entrypoint.read_bytes())
        except OSError as error:
            raise ConfigError(f"Could not hash component entrypoint '{runtime_entrypoint}': {error}.") from error
    tree_digest = _component_tree_digest(component_dir)
    if tree_digest is not None:
        digest.update(f"component-tree\0{component_dir.parent.name}/{component_dir.name}\0".encode())
        digest.update(tree_digest.encode())
    if provenance is not None:
        for dependency_dir in provenance.dependency_directories:
            dependency_digest = _component_tree_digest(dependency_dir)
            if dependency_digest is None:
                continue
            digest.update(f"dependency-tree\0{dependency_dir.parent.name}/{dependency_dir.name}\0".encode())
            digest.update(dependency_digest.encode())
    content_version = f"sha256:{digest.hexdigest()}"
    return f"{declared}+{content_version}" if declared is not None else content_version


def _runtime_python_sources(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in directory.rglob("*.py")
                if path.is_file() and is_score_affecting_component_path(path.relative_to(directory))
            ),
            key=lambda path: path.relative_to(directory).as_posix(),
        )
    )


@lru_cache(maxsize=1)
def _installed_packages_distributions() -> Mapping[str, list[str]]:
    return importlib_metadata.packages_distributions()


def _sandbox_provider_distribution_sources(
    selection: _SandboxProviderSelection,
) -> tuple[str, tuple[tuple[str, Path], ...]] | None:
    """Resolve runtime files from the installed distribution owning a provider."""

    source_file = selection.source_file
    top_level = selection.provider_class.__module__.partition(".")[0]
    if source_file is None or not top_level:
        return None
    try:
        distribution_names = _installed_packages_distributions().get(top_level, ())
    except (OSError, TypeError, ValueError):
        return None
    resolved_source = source_file.resolve()
    for distribution_name in sorted(distribution_names):
        try:
            distribution = importlib_metadata.distribution(distribution_name)
        except (importlib_metadata.PackageNotFoundError, OSError, ValueError):
            continue
        entries: list[tuple[str, Path]] = []
        owns_source = False
        for member in distribution.files or ():
            relative = Path(str(member))
            normalized_parts = tuple(part.casefold() for part in relative.parts)
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or any(part.endswith((".dist-info", ".egg-info")) for part in normalized_parts)
            ):
                continue
            try:
                located = Path(distribution.locate_file(member))
                if not located.is_file():
                    continue
                is_provider_source = located.resolve() == resolved_source
                if is_provider_source:
                    owns_source = True
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            if is_provider_source or is_score_affecting_component_path(relative):
                entries.append((relative.as_posix(), located))
        if owns_source and entries:
            entries.sort(key=lambda item: item[0])
            return distribution_name, tuple(entries)
    return None


def _source_file_set_digest(entries: Iterable[tuple[str, Path]], *, label: str) -> str:
    digest = hashlib.sha256(f"nemo-gym-{label}-v1\0".encode())
    try:
        for relative, source_path in entries:
            file_digest = _referenced_file_digest(source_path)
            if file_digest is None:
                raise ConfigError(f"Could not hash provider source '{source_path}'.")
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(str(stat.S_IMODE(source_path.stat().st_mode)).encode())
            digest.update(b"\0")
            digest.update(file_digest.encode())
            digest.update(b"\0")
    except ConfigError:
        raise
    except OSError as error:
        raise ConfigError(f"Could not hash provider {label} sources: {error}.") from error
    return digest.hexdigest()


def _sandbox_provider_content_version(
    selection: _SandboxProviderSelection,
    *,
    require_source_binding: bool = False,
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "implementation": selection.implementation,
                "provider_class": (f"{selection.provider_class.__module__}.{selection.provider_class.__qualname__}"),
                "config": _score_affecting_value(selection.config),
                "default_metadata": _score_affecting_value(selection.default_metadata),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    )
    source_bound = False
    runtime_sources: tuple[Path, ...] = ()
    package = _sandbox_provider_package_directory(selection)
    if package is not None:
        package_digest = _component_tree_digest(package)
        if package_digest is not None:
            digest.update(b"provider-package\0")
            digest.update(package_digest.encode())
            source_bound = True
            runtime_sources = _runtime_python_sources(package)
    built_in_module = f"nemo_gym.sandbox.providers.{selection.implementation}"
    is_built_in = (
        selection.provider_class.__module__ == built_in_module
        or selection.provider_class.__module__.startswith(built_in_module + ".")
    )
    if not is_built_in:
        distribution_sources = _sandbox_provider_distribution_sources(selection)
        if distribution_sources is not None:
            distribution_name, entries = distribution_sources
            digest.update(f"provider-distribution\0{distribution_name}\0".encode())
            digest.update(_source_file_set_digest(entries, label="provider-distribution").encode())
            source_bound = True
            distribution_python = tuple(path for _relative, path in entries if path.suffix == ".py")
            runtime_sources = (*runtime_sources, *distribution_python)
    if not source_bound and selection.source_file is not None:
        try:
            digest.update(b"provider-source\0")
            digest.update(selection.source_file.read_bytes())
            source_bound = True
            runtime_sources = (selection.source_file,) if selection.source_file.suffix == ".py" else ()
        except OSError as error:
            raise ConfigError(f"Could not hash sandbox provider source '{selection.source_file}': {error}.") from error
    if require_source_binding and not source_bound:
        raise ConfigError(
            f"Sandbox provider {selection.implementation!r} selected by {selection.consumer} has no readable "
            "package, distribution, or source file whose executable contents can be composition-locked."
        )
    for key, value in sorted(
        _first_party_dependency_digests(runtime_sources, key_prefix="provider-first-party").items()
    ):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _referenced_file_digest(reference: object) -> str | None:
    if not isinstance(reference, (str, Path)) or not str(reference):
        return None
    path = _resolve_under_cwd_or_install(str(reference))
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _referenced_directory_digest(reference: object, *, label: str) -> str | None:
    """Hash every runtime-visible file in a referenced directory."""

    if not isinstance(reference, (str, Path)) or not str(reference):
        return None
    path = _resolve_under_cwd_or_install(str(reference))
    if not path.is_dir():
        return None
    digest = hashlib.sha256(f"nemo-gym-{label}-tree-v1\0".encode())
    try:
        entries = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
        for entry in entries:
            relative_path = entry.relative_to(path)
            if not is_runtime_source_path(relative_path):
                continue
            relative = relative_path.as_posix()
            if entry.is_symlink():
                resolved = entry.resolve(strict=True)
                if resolved.is_dir():
                    raise ConfigError(
                        f"Referenced {label} directory '{path}' contains directory symlink '{relative}'. "
                        "Use a real directory so every runtime input is hashed unambiguously."
                    )
                if not resolved.is_file():
                    raise ConfigError(f"Referenced {label} directory '{path}' contains non-file symlink '{relative}'.")
                file_path = resolved
                kind = f"symlink:{os.readlink(entry)}"
            elif entry.is_file():
                file_path = entry
                kind = "file"
            else:
                continue
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(kind.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(str(stat.S_IMODE(file_path.stat().st_mode)).encode())
            digest.update(b"\0")
            digest.update(file_path.read_bytes())
            digest.update(b"\0")
    except ConfigError:
        raise
    except (OSError, RuntimeError) as error:
        raise ConfigError(f"Could not hash referenced {label} directory '{path}': {error}.") from error
    return digest.hexdigest()


def _driver_package_directory(module_name: str) -> Path | None:
    """Resolve a driver module to the smallest repository package that owns it."""

    driver_path = _resolve_under_cwd_or_install(Path(*module_name.split(".")).with_suffix(".py"))
    if not driver_path.is_file():
        return None
    absolute_driver = Path(os.path.abspath(driver_path))
    registry_roots = {*_COMPONENT_SUBDIRS.values(), "benchmarks", "environments"}
    for search_root in component_search_roots():
        absolute_root = Path(os.path.abspath(search_root))
        try:
            relative = absolute_driver.relative_to(absolute_root)
        except ValueError:
            continue
        if len(relative.parts) >= 3 and relative.parts[0] in registry_roots:
            return absolute_root / relative.parts[0] / relative.parts[1]
        if len(relative.parts) >= 2:
            return absolute_root / relative.parts[0]
        return None
    return absolute_driver.parent


def resolve_rollout_driver_provenance(config: Mapping[str, Any] | DictConfig) -> RolloutDriverProvenance | None:
    """Resolve the source package and component dependencies of a custom driver."""

    plain = _as_plain_mapping(config)
    driver = plain.get("rollout_collection_driver")
    if driver is None:
        return None
    if not isinstance(driver, str):
        raise ConfigError("rollout_collection_driver must be a 'module.path:function' string.")
    try:
        module_name, function_name = parse_python_callable_reference(
            driver,
            field_name="rollout_collection_driver",
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error
    relative_source = Path(*module_name.split(".")).with_suffix(".py")
    source_file = _resolve_under_cwd_or_install(relative_source)
    source_file = Path(os.path.abspath(source_file)) if source_file.is_file() else None
    source_directory = _driver_package_directory(module_name)
    dependencies = (
        component_runtime_dependency_directories(
            source_directory,
            configured_values=_rollout_driver_config(plain),
        )
        if source_directory is not None
        else ()
    )
    return RolloutDriverProvenance(
        module_name=module_name,
        function_name=function_name,
        source_file=source_file,
        source_directory=source_directory,
        dependency_directories=dependencies,
    )


def _validate_rollout_driver_sources(provenance: RolloutDriverProvenance) -> None:
    source_file = provenance.source_file
    source_directory = provenance.source_directory
    if source_file is None or source_directory is None:
        return
    if source_directory.is_symlink() or source_directory.parent.is_symlink():
        raise ConfigError(
            f"Rollout driver source '{source_directory}' is reached through a symbolic-link package or registry."
        )
    try:
        source_file.resolve().relative_to(source_directory.resolve())
    except ValueError as error:
        raise ConfigError(
            f"Rollout driver module '{source_file}' resolves outside its hashed package '{source_directory}'."
        ) from error
    relative_source = source_file.relative_to(source_directory)
    if _uses_excluded_component_directory(relative_source):
        raise ConfigError(f"Rollout driver module '{source_file}' is under an excluded tests/cache tree.")
    for path in sorted(source_directory.rglob("*.py")):
        if not path.is_file() or not is_score_affecting_component_path(path.relative_to(source_directory)):
            continue
        imported = _excluded_test_import(path, source_directory)
        if imported is not None:
            raise ConfigError(
                f"Rollout driver source '{path}' imports {imported!r} from the excluded tests tree. "
                "Move the runtime helper into the deployed driver package."
            )


def _bound_target_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.List, ast.Tuple)):
        return tuple(name for item in target.elts for name in _bound_target_names(item))
    return ()


def _module_level_bindings(tree: ast.Module) -> dict[str, ast.AST]:
    """Return unconditional module bindings in their final source order."""

    bindings: dict[str, ast.AST] = {}
    for statement in tree.body:
        if isinstance(statement, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
            bindings[statement.name] = statement
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                for name in _bound_target_names(target):
                    bindings[name] = statement.value if isinstance(target, ast.Name) else statement
        elif isinstance(statement, ast.AnnAssign):
            for name in _bound_target_names(statement.target):
                if statement.value is not None:
                    bindings[name] = statement.value
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                name = alias.asname or alias.name.partition(".")[0]
                bindings[name] = alias
        elif isinstance(statement, ast.Delete):
            for target in statement.targets:
                for name in _bound_target_names(target):
                    bindings.pop(name, None)
    return bindings


def _is_statically_callable(binding: ast.AST, bindings: Mapping[str, ast.AST], seen: set[str]) -> bool:
    if isinstance(binding, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Lambda)):
        return True
    if isinstance(binding, ast.Name):
        if binding.id in seen or binding.id not in bindings:
            return False
        return _is_statically_callable(bindings[binding.id], bindings, {*seen, binding.id})
    return False


def _validate_rollout_driver_symbol(provenance: RolloutDriverProvenance) -> None:
    source_file = provenance.source_file
    if source_file is None:
        return
    reference = f"{provenance.module_name}:{provenance.function_name}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise ConfigError(
            f"rollout_collection_driver {reference!r} cannot be statically inspected in '{source_file}': {error}."
        ) from error
    bindings = _module_level_bindings(tree)
    binding = bindings.get(provenance.function_name)
    if binding is None:
        raise ConfigError(
            f"rollout_collection_driver {reference!r} does not define symbol {provenance.function_name!r} "
            f"in local module '{source_file}'."
        )
    if not _is_statically_callable(binding, bindings, {provenance.function_name}):
        raise ConfigError(
            f"rollout_collection_driver {reference!r} symbol {provenance.function_name!r} in '{source_file}' "
            "is not statically callable. Define a module-level function, async function, class, or callable alias."
        )


def validate_rollout_driver_contract(
    config: Mapping[str, Any] | DictConfig,
    *,
    require_local_source: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate custom-driver syntax and locally knowable source provenance."""

    provenance = resolve_rollout_driver_provenance(config)
    if provenance is None:
        return (), ()
    if provenance.source_file is None:
        local_namespaces = {*_COMPONENT_SUBDIRS.values(), "benchmarks", "environments"}
        if provenance.module_name.partition(".")[0] in local_namespaces:
            raise ConfigError(
                f"rollout_collection_driver module '{provenance.module_name}' does not resolve to local source."
            )
        if require_local_source:
            raise ConfigError(
                f"rollout_collection_driver module '{provenance.module_name}' does not resolve to local, "
                "version-bound source."
            )
        return (), (
            f"Rollout driver module '{provenance.module_name}' is not local; its installed source is checked at runtime.",
        )
    _validate_rollout_driver_sources(provenance)
    _validate_rollout_driver_symbol(provenance)
    return (f"rollout driver {provenance.module_name}: local source is version-bound",), ()


def _agent_dataset_configs(config: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for role, _instance, _implementation, server_config in _server_instances(config):
        if role != "agent_server":
            continue
        for dataset in server_config.get("datasets") or []:
            if isinstance(dataset, Mapping):
                yield dataset


def _registry_package_directory_for_file(source_file: Path) -> Path | None:
    registries = {*_COMPONENT_SUBDIRS.values(), "benchmarks", "environments"}
    absolute_source = Path(os.path.abspath(source_file))
    for search_root in component_search_roots():
        absolute_root = Path(os.path.abspath(search_root))
        try:
            relative = absolute_source.relative_to(absolute_root)
        except ValueError:
            continue
        if len(relative.parts) >= 3 and relative.parts[0] in registries:
            return absolute_root / relative.parts[0] / relative.parts[1]
    return None


def resolve_dataset_preparation_provenance(
    config: Mapping[str, Any] | DictConfig,
) -> tuple[DatasetPreparationProvenance, ...]:
    """Resolve each prepare script, its owning registry package, and local dependencies."""

    plain = _as_plain_mapping(config)
    provenance: list[DatasetPreparationProvenance] = []
    seen: set[str] = set()
    for dataset in _agent_dataset_configs(plain):
        raw_reference = dataset.get("prepare_script")
        if not isinstance(raw_reference, (str, Path)) or not str(raw_reference):
            continue
        reference = str(raw_reference)
        if reference in seen:
            continue
        seen.add(reference)
        candidate = _resolve_under_cwd_or_install(reference)
        source_file = Path(os.path.abspath(candidate)) if candidate.is_file() else None
        source_directory = _registry_package_directory_for_file(source_file) if source_file is not None else None
        dependencies = (
            component_runtime_dependency_directories(source_directory, configured_values=dataset)
            if source_directory is not None
            else ()
        )
        provenance.append(
            DatasetPreparationProvenance(
                reference=reference,
                source_file=source_file,
                source_directory=source_directory,
                dependency_directories=dependencies,
            )
        )
    return tuple(provenance)


def validate_dataset_preparation_sources(
    config: Mapping[str, Any] | DictConfig,
) -> tuple[str, ...]:
    """Require benchmark preparation programs to be local, version-bound Python."""

    decisions: list[str] = []
    for preparation in resolve_dataset_preparation_provenance(config):
        source_file = preparation.source_file
        source_directory = preparation.source_directory
        if source_file is None:
            raise ConfigError(f"Benchmark prepare_script '{preparation.reference}' does not resolve to a file.")
        if source_directory is None:
            raise ConfigError(
                f"Benchmark prepare_script '{source_file}' is outside a supported environment registry package."
            )
        if source_file.is_symlink() or source_directory.is_symlink() or source_directory.parent.is_symlink():
            raise ConfigError(f"Benchmark prepare_script '{source_file}' must not use a symbolic-link source path.")
        try:
            source_file.resolve(strict=True).relative_to(source_directory.resolve(strict=True))
            module = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        except (OSError, RuntimeError, ValueError, SyntaxError) as error:
            raise ConfigError(
                f"Benchmark prepare_script '{source_file}' is not valid version-bound Python: {error}."
            ) from error
        if not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "prepare"
            for node in module.body
        ):
            raise ConfigError(f"Benchmark prepare_script '{source_file}' must define a top-level prepare() function.")
        decisions.append(f"benchmark prepare_script {preparation.reference}: local source is version-bound")
    return tuple(decisions)


def _direct_dataset_file_digests(config: Mapping[str, Any]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for field_name in ("input_jsonl_fpath", "prompt_config"):
        reference = config.get(field_name)
        digest = _referenced_file_digest(reference)
        if digest is not None:
            digests[f"{field_name}:{reference}"] = digest

    for dataset in _agent_dataset_configs(config):
        for field_name in ("jsonl_fpath", "prompt_config"):
            reference = dataset.get(field_name)
            digest = _referenced_file_digest(reference)
            if digest is not None:
                digests[f"{field_name}:{reference}"] = digest
    return digests


def _package_provenance_digests(
    source_directory: Path,
    dependencies: Sequence[Path],
    *,
    package_key: str,
    dependency_key: str,
) -> dict[str, str]:
    digests: dict[str, str] = {}
    package_digest = _component_tree_digest(
        source_directory,
        environment_package=source_directory.parent.name in {"benchmarks", "environments"},
    )
    if package_digest is not None:
        label = f"{source_directory.parent.name}/{source_directory.name}"
        digests[f"{package_key}:{label}"] = package_digest
    for dependency in dependencies:
        dependency_digest = _component_tree_digest(
            dependency,
            environment_package=dependency.parent.name in {"benchmarks", "environments"},
        )
        if dependency_digest is not None:
            label = f"{dependency.parent.name}/{dependency.name}"
            digests[f"{dependency_key}:{label}"] = dependency_digest
    return digests


def _dataset_preparation_digests(config: Mapping[str, Any]) -> dict[str, str]:
    digests: dict[str, str] = {}

    for preparation in resolve_dataset_preparation_provenance(config):
        if preparation.source_directory is None:
            digest = _referenced_file_digest(preparation.source_file)
            if digest is not None:
                digests[f"prepare_script:{preparation.reference}"] = digest
            continue
        digests.update(
            _package_provenance_digests(
                preparation.source_directory,
                preparation.dependency_directories,
                package_key="prepare_package",
                dependency_key="prepare_dependency",
            )
        )
    return digests


def _stock_rollout_driver_source() -> Path:
    return Path(__file__).resolve().with_name("rollout_collection.py")


def _rollout_driver_digests(config: Mapping[str, Any]) -> dict[str, str]:
    driver_provenance = resolve_rollout_driver_provenance(config)
    if driver_provenance is None:
        if infer_integration_profile(config) == "external-loop":
            return {}
        source_file = _stock_rollout_driver_source()
        driver_digest = _referenced_file_digest(source_file)
        if driver_digest is None:
            raise ConfigError(f"Stock rollout driver source '{source_file}' is missing or unreadable.")
        digests = {"stock_rollout_driver:nemo_gym.rollout_collection": driver_digest}
        digests.update(
            _first_party_dependency_digests(
                (source_file,),
                key_prefix="stock_rollout_driver_dependency",
            )
        )
        return digests
    if driver_provenance.source_directory is not None:
        digests = _package_provenance_digests(
            driver_provenance.source_directory,
            driver_provenance.dependency_directories,
            package_key="rollout_driver_package",
            dependency_key="rollout_driver_dependency",
        )
        runtime_sources = [*_runtime_python_sources(driver_provenance.source_directory)]
        for dependency in driver_provenance.dependency_directories:
            runtime_sources.extend(_runtime_python_sources(dependency))
        digests.update(
            _first_party_dependency_digests(
                runtime_sources,
                key_prefix="rollout_driver_first_party_dependency",
            )
        )
        return digests
    if driver_provenance.source_file is None:
        return {}
    driver_digest = _referenced_file_digest(driver_provenance.source_file)
    if driver_digest is None:
        return {}
    digests = {f"rollout_collection_driver:{driver_provenance.source_file.name}": driver_digest}
    digests.update(
        _first_party_dependency_digests(
            (driver_provenance.source_file,),
            key_prefix="rollout_driver_first_party_dependency",
        )
    )
    return digests


def _inspect_rollout_driver(
    config: Mapping[str, Any],
    profile: str,
    *,
    include_content_version: bool,
) -> ComponentInspection | None:
    driver = config.get("rollout_collection_driver")
    if driver is None and profile == "external-loop":
        return None

    provenance = resolve_rollout_driver_provenance(config)
    implementation = str(driver) if driver is not None else "nemo_gym.rollout_collection"
    source_file = provenance.source_file if provenance is not None else _stock_rollout_driver_source()
    version = None
    if include_content_version:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "implementation": implementation,
                    "sources": _rollout_driver_digests(config),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        version = f"sha256:{digest}"
    return ComponentInspection(
        role="rollout_driver",
        instance="custom" if driver is not None else "stock",
        implementation=implementation,
        version=version,
        entrypoint=str(source_file) if source_file is not None else None,
        boundary="in-process",
    )


def _skills_digests(config: Mapping[str, Any]) -> dict[str, str]:
    skills = config.get("skills")
    if not isinstance(skills, Mapping):
        return {}
    skills_path = skills.get("path")
    digest = _referenced_directory_digest(skills_path, label="skills")
    return {f"skills:{skills_path}": digest} if digest is not None else {}


def _runtime_local_reference_digests(config: Mapping[str, Any]) -> dict[str, str]:
    repo_root = find_repository_root(Path.cwd()) or Path.cwd()
    component_bases = {
        component.instance: component.source_directory
        for component in resolve_component_provenance(config)
        if component.source_directory is not None
    }
    digests: dict[str, str] = {}
    for reference in resolve_runtime_local_references(
        config,
        repo_root=repo_root,
        allow_external=True,
        base_directories=component_bases,
    ):
        path = reference.path
        if path is None:
            continue
        if path.is_dir():
            digest = _referenced_directory_digest(path, label="runtime input")
        else:
            digest = _referenced_file_digest(path)
        if digest is not None:
            digests[f"runtime_input:{reference.field}:{reference.reference}"] = digest
    return digests


def validate_runtime_local_references(config: Mapping[str, Any] | DictConfig) -> tuple[str, ...]:
    """Require arbitrary manifest-bound file inputs to be local, present, and link-free."""

    plain = _as_plain_mapping(config)
    repo_root = find_repository_root(Path.cwd()) or Path.cwd()
    component_bases = {
        component.instance: component.source_directory
        for component in resolve_component_provenance(plain)
        if component.source_directory is not None
    }
    references = resolve_runtime_local_references(
        plain,
        repo_root=repo_root,
        require_existing=True,
        base_directories=component_bases,
    )
    return tuple(
        f"runtime input {reference.field} resolves to {reference.path}"
        for reference in references
        if reference.path is not None
    )


def _score_affecting_file_digests(config: Mapping[str, Any]) -> dict[str, str]:
    digests = {
        **_direct_dataset_file_digests(config),
        **_dataset_preparation_digests(config),
        **_skills_digests(config),
        **_runtime_local_reference_digests(config),
    }
    return dict(sorted(digests.items()))


def _inspect_component_interfaces(
    config: Mapping[str, Any] | DictConfig,
    profile: str | None = None,
    *,
    include_content_versions: bool,
) -> tuple[ComponentInspection, ...]:
    plain = _as_plain_mapping(config)
    resolved_profile = profile or infer_integration_profile(plain)
    pins = pinned_component_roles(resolved_profile)
    provenance_by_component = (
        {(item.role, item.instance, item.implementation): item for item in resolve_component_provenance(plain)}
        if include_content_versions
        else {}
    )
    components = []
    for role, instance, implementation, server_config in _server_instances(plain):
        declared_requires = _string_set(server_config.get("requires"), component=instance, field_name="requires")
        declared_provides = _string_set(server_config.get("provides"), component=instance, field_name="provides")
        capability_declaration = component_capability_declaration(
            instance=instance,
            implementation=implementation,
            group=_COMPONENT_SUBDIRS[role],
            server_config={"requires": declared_requires, "provides": declared_provides},
        )
        components.append(
            ComponentInspection(
                role=role,
                instance=instance,
                implementation=implementation,
                version=(
                    _component_content_version(
                        role,
                        implementation,
                        server_config,
                        provenance_by_component.get((role, instance, implementation)),
                    )
                    if include_content_versions
                    else None
                ),
                entrypoint=str(server_config["entrypoint"]) if server_config.get("entrypoint") else None,
                boundary=_boundary(server_config),
                requires=capability_declaration.requires,
                provides=capability_declaration.provides,
                pinned=role in pins,
            )
        )
    rollout_driver = _inspect_rollout_driver(
        plain,
        resolved_profile,
        include_content_version=include_content_versions,
    )
    if rollout_driver is not None:
        components.append(rollout_driver)
    return tuple(components)


def inspect_components(
    config: Mapping[str, Any] | DictConfig, profile: str | None = None
) -> tuple[ComponentInspection, ...]:
    """Inspect selected component interfaces and immutable content versions."""

    return _inspect_component_interfaces(config, profile, include_content_versions=True)


def inspect_sandbox_providers(
    config: Mapping[str, Any] | DictConfig,
    manifest: EnvironmentManifest | None = None,
    *,
    include_content_versions: bool = True,
) -> tuple[tuple[ComponentInspection, ...], tuple[str, ...]]:
    """Inspect registered sandbox providers and enforce a manifest selection when present."""

    selections = _selected_sandbox_providers(_as_plain_mapping(config))
    expected = (
        next(
            (
                capability.removeprefix("sandbox:")
                for field_name, capability in manifest_implied_capabilities(manifest)
                if field_name == "sandbox"
            ),
            None,
        )
        if manifest is not None
        else None
    )
    if expected is not None:
        if not selections:
            raise ConfigError(
                f"Manifest sandbox {manifest.sandbox!r} does not select a sandbox_provider in the resolved config."
            )
        actual = sorted({selection.implementation for selection in selections})
        if actual != [expected]:
            raise ConfigError(
                f"Manifest sandbox {manifest.sandbox!r} does not match the selected registered sandbox "
                f"provider(s): {', '.join(actual)}."
            )

    components = tuple(
        ComponentInspection(
            role="sandbox_provider",
            instance=selection.instance,
            implementation=selection.implementation,
            version=(
                _sandbox_provider_content_version(
                    selection,
                    require_source_binding=manifest is not None,
                )
                if include_content_versions
                else None
            ),
            entrypoint=str(selection.source_file) if selection.source_file is not None else None,
            boundary=None,
            provides=(f"sandbox:{selection.implementation}",),
        )
        for selection in selections
    )
    decisions = tuple(
        f"sandbox provider {selection.implementation!r} selected by {selection.consumer}: registered"
        for selection in selections
    )
    return components, decisions


def resolve_composition_mirror(config: Mapping[str, Any] | DictConfig) -> CompositionMirror:
    """Project score-affecting composition fields from the resolved config.

    Component mirrors use Gym's deployable granularity: resources and agent
    implementations plus the model instance selected by the agent. The runtime
    remains authoritative.
    """

    plain = _as_plain_mapping(config)
    resources_by_instance: dict[str, tuple[str, Mapping[str, Any]]] = {}
    selected_resource_instances: list[str] = []
    agents: list[str] = []
    model_refs: list[str] = []
    grading_modes: list[str] = []
    datasets: list[ManifestDataset] = []

    components = tuple(_server_instances(plain))
    for role, instance, implementation, server_config in components:
        if role == "resources_server":
            resources_by_instance[instance] = (implementation, server_config)
        elif role == "agent_server":
            if implementation not in agents:
                agents.append(implementation)
            resource_ref = server_config.get("resources_server")
            if isinstance(resource_ref, Mapping) and resource_ref.get("name"):
                selected = str(resource_ref["name"])
                if selected not in selected_resource_instances:
                    selected_resource_instances.append(selected)
            model_ref = server_config.get("model_server")
            if isinstance(model_ref, Mapping) and model_ref.get("name"):
                selected = str(model_ref["name"])
                if selected not in model_refs:
                    model_refs.append(selected)
            for raw_dataset in server_config.get("datasets") or []:
                if not isinstance(raw_dataset, Mapping) or not raw_dataset.get("jsonl_fpath"):
                    continue
                datasets.append(
                    ManifestDataset(
                        name=str(raw_dataset.get("name") or "<unnamed>"),
                        type=str(raw_dataset.get("type") or "example"),
                        jsonl_fpath=str(raw_dataset["jsonl_fpath"]),
                        prepare_script=(
                            str(raw_dataset["prepare_script"]) if raw_dataset.get("prepare_script") else None
                        ),
                        prompt_config=(
                            str(raw_dataset["prompt_config"]) if raw_dataset.get("prompt_config") else None
                        ),
                        num_repeats=int(raw_dataset.get("num_repeats", 1)),
                    )
                )

    selected_resources = [
        resources_by_instance[instance]
        for instance in selected_resource_instances
        if instance in resources_by_instance
    ]
    if not selected_resources and len(resources_by_instance) == 1:
        selected_resources = list(resources_by_instance.values())
    resources = list(dict.fromkeys(implementation for implementation, _config in selected_resources))
    for _implementation, server_config in selected_resources:
        grading_mode = server_config.get("grading_mode")
        if grading_mode is not None and str(grading_mode) not in grading_modes:
            grading_modes.append(str(grading_mode))

    def only_or_none(values: list[str]) -> str | None:
        return values[0] if len(values) == 1 else None

    return CompositionMirror(
        resources_server=only_or_none(resources),
        agent_server=only_or_none(agents),
        model_server=only_or_none(model_refs),
        datasets=tuple(datasets),
        rollout_driver=str(plain["rollout_collection_driver"]) if plain.get("rollout_collection_driver") else None,
        grading_mode=only_or_none(grading_modes),
    )


def manifest_composition_deltas(
    manifest: EnvironmentManifest,
    mirror: CompositionMirror,
) -> tuple[str, ...]:
    """Return human-readable manifest/config mirror differences."""

    deltas: list[str] = []
    for field_name in ("resources_server", "agent_server", "model_server", "rollout_driver", "grading_mode"):
        declared = getattr(manifest, field_name)
        resolved = getattr(mirror, field_name)
        if declared != resolved:
            deltas.append(f"{field_name}: manifest={declared!r}, resolved config={resolved!r}")

    declared_datasets = [dataset.model_dump(mode="json") for dataset in (manifest.datasets or [])]
    resolved_datasets = [dataset.model_dump(mode="json") for dataset in mirror.datasets]
    if declared_datasets != resolved_datasets:
        deltas.append(f"datasets: manifest={declared_datasets!r}, resolved config={resolved_datasets!r}")
    return tuple(deltas)


def validate_grading_mode_constraints(
    config: Mapping[str, Any] | DictConfig,
    manifest: EnvironmentManifest | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate grading modes against finite sets recoverable from resource-server source.

    Returns compatibility decisions and limitations separately. Missing source or dynamic validation logic is
    reported as a warning only when a grading mode was actually declared; it is never guessed.
    """

    plain = _as_plain_mapping(config)
    decisions: list[str] = []
    warnings: list[str] = []
    for role, _instance, implementation, server_config in _server_instances(plain):
        if role != "resources_server":
            continue

        declared: list[tuple[str, str]] = []
        configured_mode = server_config.get("grading_mode")
        if configured_mode is not None:
            declared.append(("resolved config", str(configured_mode)))
        if (
            manifest is not None
            and manifest.resources_server == implementation
            and manifest.grading_mode is not None
            and str(manifest.grading_mode) != str(configured_mode)
        ):
            declared.append(("manifest", manifest.grading_mode))
        if not declared:
            continue

        entrypoint = str(server_config.get("entrypoint") or "app.py")
        source_directory, _selected_config_path = resolve_component_source_directory(
            "resources_servers",
            implementation,
            _resolved_config_paths(plain),
        )
        source_path = (
            source_directory / entrypoint
            if source_directory is not None
            else _resolve_under_cwd_or_install(Path("resources_servers") / implementation / entrypoint)
        )
        allowed = grading_modes_from_source(source_path)
        if not allowed:
            warnings.append(
                f"Resources server '{implementation}' declares grading_mode but no finite constraint "
                f"could be recovered statically from '{source_path}'."
            )
            continue

        allowed_text = ", ".join(repr(mode) for mode in allowed)
        for origin, value in declared:
            if value not in allowed:
                raise ConfigError(
                    f"Resources server '{implementation}' has unsupported {origin} grading_mode={value!r}; "
                    f"statically discovered modes: {allowed_text}."
                )
            decisions.append(f"{implementation} {origin} grading_mode={value!r}: supported")
    return tuple(decisions), tuple(warnings)


def _capability_providers(
    components: Sequence[ComponentInspection],
) -> dict[str, list[ComponentInspection]]:
    providers: dict[str, list[ComponentInspection]] = {}
    for component in components:
        for capability in component.provides:
            providers.setdefault(capability, []).append(component)
    return providers


def _providers_for_capability(
    providers: Mapping[str, Sequence[ComponentInspection]], capability: str
) -> list[ComponentInspection]:
    matches = list(providers.get(capability, ()))
    if capability.startswith("sandbox:"):
        matches = [provider for provider in matches if provider.role == "sandbox_provider"]
    return matches


def _referenced_server_instances(value: object) -> set[str]:
    references: set[str] = set()
    if isinstance(value, Mapping):
        if value.get("type") in SERVER_ROLE_BY_GROUP and isinstance(value.get("name"), str):
            references.add(str(value["name"]))
        for nested in value.values():
            references.update(_referenced_server_instances(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            references.update(_referenced_server_instances(nested))
    return references


def _manifest_bound_server_instances(
    config: Mapping[str, Any],
    manifest: EnvironmentManifest,
) -> tuple[tuple[str, str, str, Mapping[str, Any]], ...]:
    """Select manifest roots and every configured server they reference."""

    components = [
        component
        for component in _server_instances(config)
        if not (component[0] == "model_server" and component[2] == "dummy_model")
    ]
    by_instance = {instance: component for component in components for instance in (component[1],)}
    selected: list[tuple[str, str, str, Mapping[str, Any]]] = []
    selected_instances: set[str] = set()

    def add(component: tuple[str, str, str, Mapping[str, Any]]) -> None:
        if component[1] not in selected_instances:
            selected_instances.add(component[1])
            selected.append(component)

    references = {
        "resources_server": manifest.resources_server,
        "agent_server": manifest.agent_server,
        "model_server": manifest.model_server,
    }
    for role, reference in references.items():
        if reference is None:
            continue
        candidates = [component for component in components if component[0] == role]
        matches = [
            component
            for component in candidates
            if component[2] == reference or (role == "model_server" and component[1] == reference)
        ]
        # Full validation supports synchronizing a stale manifest. When one
        # runtime component is unambiguous, validate its source before syncing.
        if not matches and len(candidates) == 1:
            matches = candidates
        for component in matches:
            add(component)

    index = 0
    while index < len(selected):
        server_config = selected[index][3]
        index += 1
        for referenced_instance in _referenced_server_instances(server_config):
            component = by_instance.get(referenced_instance)
            if component is not None:
                add(component)
    return tuple(selected)


def _validate_selected_component_sources(
    config: Mapping[str, Any] | DictConfig,
    selected: Sequence[tuple[str, str, str, Mapping[str, Any]]],
) -> tuple[str, ...]:
    """Require each selected runtime entrypoint to resolve locally."""

    plain = _as_plain_mapping(config)
    config_paths = _resolved_config_paths(plain)
    provenance_by_component = {
        (item.role, item.instance, item.implementation): item for item in resolve_component_provenance(plain)
    }
    decisions: list[str] = []
    for role, instance, implementation, server_config in selected:
        group = _COMPONENT_SUBDIRS[role]
        source_directory, _selected_config = resolve_component_source_directory(
            group,
            implementation,
            config_paths,
        )
        if source_directory is None:
            source_directory = _resolve_under_cwd_or_install(Path(group) / implementation)
        label = f"Selected {role.replace('_', ' ')} {implementation!r} (instance {instance!r})"
        provenance = provenance_by_component.get((role, instance, implementation))
        entrypoint = _resolve_component_runtime_entrypoint(
            source_directory,
            server_config,
            required=True,
            component_label=label,
            entrypoint_source_directory=(provenance.entrypoint_source_directory if provenance is not None else None),
        )
        assert entrypoint is not None
        decisions.append(f"{role}:{implementation} entrypoint resolves to {entrypoint}")
    return tuple(decisions)


def validate_manifest_component_sources(
    config: Mapping[str, Any] | DictConfig,
    manifest: EnvironmentManifest,
) -> tuple[str, ...]:
    """Require source for the manifest-bound component graph."""

    plain = _as_plain_mapping(config)
    return _validate_selected_component_sources(plain, _manifest_bound_server_instances(plain, manifest))


def validate_manifest_launch_sources(
    config: Mapping[str, Any] | DictConfig,
    *,
    allow_static_dummy_model: bool = False,
) -> tuple[str, ...]:
    """Require source for every server a manifest-bound RunHelper will start."""

    plain = _as_plain_mapping(config)
    selected = tuple(
        component
        for component in _server_instances(plain, strict_launch=True)
        if not (allow_static_dummy_model and component[0] == "model_server" and component[2] == "dummy_model")
    )
    return _validate_selected_component_sources(plain, selected)


def _wired_component_instances(
    config: Mapping[str, Any] | DictConfig | None,
) -> dict[tuple[str, str, str], set[str]]:
    if config is None:
        return {}
    wired: dict[tuple[str, str, str], set[str]] = {}
    for role, instance, implementation, server_config in _server_instances(_as_plain_mapping(config)):
        references = _referenced_server_instances(server_config)
        sandbox_provider = server_config.get("sandbox_provider")
        if isinstance(sandbox_provider, str):
            references.add(sandbox_provider)
        elif isinstance(sandbox_provider, Mapping):
            references.add(f"{instance}.sandbox_provider")
        wired[(role, instance, implementation)] = references
    return wired


def _component_capability_results(
    components: Sequence[ComponentInspection],
    providers: Mapping[str, Sequence[ComponentInspection]],
    wired_instances: Mapping[tuple[str, str, str], set[str]],
) -> tuple[list[str], list[str]]:
    decisions: list[str] = []
    missing: list[str] = []
    for component in components:
        for capability in component.requires:
            matches = [
                provider for provider in _providers_for_capability(providers, capability) if provider is not component
            ]
            wired = wired_instances.get((component.role, component.instance, component.implementation), set())
            if wired:
                matches = [provider for provider in matches if provider.instance in wired]
            if matches:
                names = ", ".join(sorted(provider.instance for provider in matches))
                decisions.append(f"{component.instance} requires {capability}: satisfied by {names}")
            else:
                detail = f" through wired instances {', '.join(sorted(wired))}" if wired else ""
                missing.append(f"{component.instance} requires '{capability}'{detail}")
    return decisions, missing


def _manifest_selected_components(
    components: Sequence[ComponentInspection],
    *,
    role: str,
    reference: str | None,
) -> tuple[list[ComponentInspection], list[ComponentInspection]]:
    role_components = [component for component in components if component.role == role]
    if reference is None:
        return (role_components if len(role_components) == 1 else []), role_components
    if role != "model_server":
        return [component for component in role_components if component.implementation == reference], role_components

    # Model mirrors normally name the agent's instance (for example
    # ``policy_model``), with implementation as a fallback.
    selected = [component for component in role_components if component.instance == reference]
    if not selected:
        selected = [component for component in role_components if component.implementation == reference]
    return selected, role_components


def _missing_manifest_target_detail(
    selected: Sequence[ComponentInspection],
    role_components: Sequence[ComponentInspection],
    *,
    role: str,
    reference: str | None,
) -> str:
    selected_label = f"manifest-selected {role} {reference!r}" if reference else f"selected {role}"
    if selected:
        capabilities = sorted({capability for component in selected for capability in component.provides})
        return f"{selected_label} provides: {', '.join(capabilities) or 'none declared'}"
    if reference is not None:
        return f"{selected_label} does not resolve to a {role} component"
    if role_components:
        names = ", ".join(sorted(component.instance for component in role_components))
        return f"manifest does not select one {role}; candidates: {names}"
    return f"no {role} component is selected"


def _manifest_capability_results(
    components: Sequence[ComponentInspection],
    manifest: EnvironmentManifest,
    providers: Mapping[str, Sequence[ComponentInspection]],
) -> tuple[list[str], list[str]]:
    decisions: list[str] = []
    missing: list[str] = []
    implied_targets: dict[str, tuple[str, str, str | None]] = {}
    for field_name, capability in manifest_implied_capabilities(manifest):
        if field_name == "modality":
            implied_targets[capability] = (field_name, "model_server", manifest.model_server)
        elif field_name == "sandbox":
            implied_targets[capability] = (
                field_name,
                "sandbox_provider",
                capability.removeprefix("sandbox:"),
            )
        else:
            implied_targets[capability] = (field_name, "resources_server", manifest.resources_server)
    for capability in manifest_required_capabilities(manifest):
        target = implied_targets.get(capability)
        if target is None:
            matches = _providers_for_capability(providers, capability)
            if matches:
                names = ", ".join(sorted(provider.instance for provider in matches))
                decisions.append(f"environment requires {capability}: satisfied by {names}")
            else:
                missing.append(f"environment requires '{capability}'")
            continue

        field_name, role, reference = target
        selected, role_components = _manifest_selected_components(
            components,
            role=role,
            reference=reference,
        )
        matches = [
            component for component in selected if component in _providers_for_capability(providers, capability)
        ]
        if matches:
            names = ", ".join(sorted(provider.instance for provider in matches))
            decisions.append(f"environment requires {capability}: satisfied by {names}")
            continue
        detail = _missing_manifest_target_detail(
            selected,
            role_components,
            role=role,
            reference=reference,
        )
        missing.append(f"environment requires '{capability}' for manifest {field_name}, but {detail}")

    for capability in manifest.provides:
        matches = _providers_for_capability(providers, capability)
        if matches:
            names = ", ".join(sorted(provider.instance for provider in matches))
            decisions.append(f"environment provides {capability}: backed by {names}")
        else:
            missing.append(f"environment declares provided capability '{capability}' with no component backing")
    return decisions, missing


def _manifest_interface_declaration_results(
    components: Sequence[ComponentInspection],
    manifest: EnvironmentManifest,
) -> tuple[list[str], list[str]]:
    """Require manifest-selected adapters to declare their runtime protocol."""

    decisions: list[str] = []
    missing: list[str] = []
    resources, _resource_candidates = _manifest_selected_components(
        components,
        role="resources_server",
        reference=manifest.resources_server,
    )
    agents, _agent_candidates = _manifest_selected_components(
        components,
        role="agent_server",
        reference=manifest.agent_server,
    )

    if manifest.resources_server:
        if len(resources) != 1:
            missing.append("manifest-selected resources server must resolve to exactly one declared adapter")
        elif "verification" not in resources[0].provides:
            missing.append(f"{resources[0].instance} must declare provides: [verification]")
        else:
            decisions.append(f"{resources[0].instance} declares the verification provider interface")

    if manifest.agent_server:
        if len(agents) != 1:
            missing.append("manifest-selected agent server must resolve to exactly one declared adapter")
        else:
            agent = agents[0]
            required_agent_protocols: set[str] = set()
            profile = getattr(manifest, "integration_profile", None)
            external_loop = getattr(profile, "value", profile) == "external-loop"
            if manifest.resources_server and not external_loop:
                required_agent_protocols.add("verification")
            required_agent_protocols.update(
                capability
                for field_name, capability in manifest_implied_capabilities(manifest)
                if field_name in {"modality", "session_model", "state"}
            )
            if len(resources) == 1 and not external_loop:
                resource_protocols = set(resources[0].provides)
                required_agent_protocols.update(
                    capability for capability in manifest.requires if capability in resource_protocols
                )
            undeclared = sorted(required_agent_protocols.difference(agent.requires))
            if undeclared:
                missing.append(f"{agent.instance} must declare required protocol(s): {', '.join(undeclared)}")
            else:
                decisions.append(f"{agent.instance} declares its verifier/model protocol requirements")
    return decisions, missing


def validate_capabilities(
    components: Sequence[ComponentInspection],
    manifest: EnvironmentManifest | None = None,
    *,
    config: Mapping[str, Any] | DictConfig | None = None,
) -> tuple[str, ...]:
    """Reject requirements not provided by another selected component.

    Manifest-implied model and runtime requirements are scoped to the
    manifest-selected component. A component cannot satisfy its own requirement.
    """

    providers = _capability_providers(components)
    decisions, missing = _component_capability_results(
        components,
        providers,
        _wired_component_instances(config),
    )
    if manifest is not None:
        manifest_decisions, manifest_missing = _manifest_capability_results(components, manifest, providers)
        decisions.extend(manifest_decisions)
        missing.extend(manifest_missing)
        interface_decisions, interface_missing = _manifest_interface_declaration_results(components, manifest)
        decisions.extend(interface_decisions)
        missing.extend(interface_missing)
    if missing:
        available = ", ".join(sorted(providers)) or "none declared"
        raise ConfigError(
            "Incompatible component capabilities: " + "; ".join(missing) + f". Available capabilities: {available}."
        )
    return tuple(decisions)


def validate_execution_contracts(
    config: Mapping[str, Any] | DictConfig,
    manifest: EnvironmentManifest | None,
    *,
    profile: str | None = None,
    check_launch_sources: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate launch-critical contracts without reading datasets or hashing components."""

    plain = _as_plain_mapping(config)
    source_decisions = validate_manifest_launch_sources(plain) if manifest is not None and check_launch_sources else ()
    components = _inspect_component_interfaces(plain, profile, include_content_versions=False)
    sandbox_decisions: tuple[str, ...] = ()
    if manifest is not None:
        sandbox_components, sandbox_decisions = inspect_sandbox_providers(
            plain,
            manifest,
            include_content_versions=False,
        )
        components = (*components, *sandbox_components)
    capability_decisions = validate_capabilities(components, manifest, config=plain)
    grading_decisions, grading_warnings = validate_grading_mode_constraints(plain, manifest)
    driver_decisions, driver_warnings = validate_rollout_driver_contract(
        plain,
        require_local_source=manifest is not None,
    )
    preparation_decisions = validate_dataset_preparation_sources(plain) if manifest is not None else ()
    runtime_file_decisions = validate_runtime_local_references(plain) if manifest is not None else ()
    return (
        *source_decisions,
        *sandbox_decisions,
        *capability_decisions,
        *grading_decisions,
        *driver_decisions,
        *preparation_decisions,
        *runtime_file_decisions,
    ), (*grading_warnings, *driver_warnings)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield JSONL rows without retaining the dataset in memory."""

    found = False
    try:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigError(f"Dataset '{path}' line {line_number} is not valid JSON: {exc.msg}.") from None
                if not isinstance(row, dict):
                    raise ConfigError(f"Dataset '{path}' line {line_number} must be a JSON object.")
                found = True
                yield line_number, row
    except OSError as exc:
        raise ConfigError(f"Could not read dataset '{path}': {exc}.") from exc
    if not found:
        raise ConfigError(f"Dataset '{path}' contains no JSONL rows.")


def _validate_materialized_row(row: Mapping[str, Any], path: Path, line_number: int) -> bool:
    params = row.get("responses_create_params")
    if not isinstance(params, Mapping) or not params.get("input"):
        return False
    try:
        _DatasetResponseCreateParams.model_validate(params)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first.get("loc", ())) or "responses_create_params"
        raise ConfigError(
            f"Dataset '{path}' line {line_number} has invalid Responses API parameters at '{location}': "
            f"{first['msg']}."
        ) from exc
    return True


def _inspect_jsonl_stream(
    path: Path,
    *,
    prompt_config: PromptConfig | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Validate and preview a dataset in one bounded-memory pass."""

    row_count = 0
    missing_count = 0
    missing_lines: list[int] = []
    conflicting_count = 0
    conflicting_rows: list[int] = []
    first_materialized: Mapping[str, Any] | None = None
    for row_index, (line_number, row) in enumerate(_iter_jsonl(path)):
        row_count += 1
        if prompt_config is not None:
            params = row.get("responses_create_params")
            if isinstance(params, Mapping) and params.get("input"):
                conflicting_count += 1
                if len(conflicting_rows) < 8:
                    conflicting_rows.append(row_index)
                continue
            try:
                materialized = apply_prompt_to_row(row, prompt_config)
            except (KeyError, ValueError) as error:
                raise ConfigError(
                    f"Benchmark dataset '{path}' line {line_number} cannot apply prompt: {error}"
                ) from error
        else:
            materialized = row

        if not _validate_materialized_row(materialized, path, line_number):
            missing_count += 1
            if len(missing_lines) < 8:
                missing_lines.append(line_number)
        if first_materialized is None:
            first_materialized = materialized

    if conflicting_count:
        preview = ", ".join(map(str, conflicting_rows))
        suffix = f", ... ({conflicting_count} total)" if conflicting_count > len(conflicting_rows) else ""
        raise ConfigError(
            "Some rows have responses_create_params.input but prompt_config is also specified. "
            f"These are mutually exclusive. Use one or the other. Violating rows: [{preview}{suffix}]"
        )
    if missing_count:
        preview = ", ".join(map(str, missing_lines))
        suffix = f", ... ({missing_count} total)" if missing_count > len(missing_lines) else ""
        raise ConfigError(
            f"Dataset '{path}' must contain responses_create_params.input on every materialized row; "
            f"missing on line(s) {preview}{suffix}."
        )

    sample = _materialized_responses_sample([first_materialized]) if first_materialized is not None else None
    return row_count, sample


def _require_version_bound_dataset_file(path: Path, *, dataset_name: str) -> None:
    absolute = Path(os.path.abspath(path))
    for search_root in component_search_roots():
        root = Path(os.path.abspath(search_root))
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            continue
        cursor = root
        if cursor.is_symlink():
            raise ConfigError(f"Dataset '{dataset_name}' is reached through symbolic-link root '{cursor}'.")
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ConfigError(f"Dataset '{dataset_name}' uses symbolic-link path '{cursor}'.")
        try:
            absolute.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as error:
            raise ConfigError(f"Dataset '{dataset_name}' resolves outside component search root '{root}'.") from error
        return
    raise ConfigError(
        f"Dataset '{dataset_name}' at '{absolute}' is outside every component search root and cannot be "
        "included in an immutable environment version. Move it into the repository or a selected search root."
    )


def validate_datasets(
    config: Mapping[str, Any] | DictConfig,
    *,
    strict_missing: bool = False,
    standard_prompt_config: str | Path | None = None,
) -> tuple[DatasetInspection, ...]:
    """Parse declared datasets and validate their authoring path.

    Legacy workloads frequently keep large/generated data out of the checkout,
    so missing files are reported but remain backward-compatible by default.
    Manifest-backed validation opts into ``strict_missing=True``.
    """

    plain = _as_plain_mapping(config)
    reports: list[DatasetInspection] = []
    for role, instance, _implementation, server_config in _server_instances(plain):
        if role != "agent_server":
            continue
        datasets = server_config.get("datasets") or []
        if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes, bytearray)):
            raise ConfigError(f"Agent server '{instance}' field 'datasets' must be a list.")
        for raw_dataset in datasets:
            if not isinstance(raw_dataset, Mapping):
                raise ConfigError(f"Agent server '{instance}' contains a dataset entry that is not a mapping.")
            name = str(raw_dataset.get("name") or "<unnamed>")
            dataset_type = str(raw_dataset.get("type") or "<unknown>")
            raw_path = raw_dataset.get("jsonl_fpath")
            if not raw_path:
                raise ConfigError(f"Dataset '{name}' on agent server '{instance}' is missing jsonl_fpath.")
            path = _resolve_under_cwd_or_install(str(raw_path))
            if not path.is_file():
                detail = (
                    "Run `gym eval prepare` first."
                    if dataset_type == "benchmark"
                    else "Download or create this dataset."
                )
                if strict_missing:
                    raise ConfigError(f"Dataset '{name}' does not exist at '{path}'. {detail}")
                reports.append(DatasetInspection(name, dataset_type, path, None, "missing", detail))
                continue
            if strict_missing:
                _require_version_bound_dataset_file(path, dataset_name=name)

            prompt_path = raw_dataset.get("prompt_config") or standard_prompt_config
            if dataset_type == "benchmark" and prompt_path:
                try:
                    if strict_missing:
                        _require_version_bound_dataset_file(
                            _resolve_under_cwd_or_install(str(prompt_path)),
                            dataset_name=f"{name} prompt config",
                        )
                    prompt_config = load_prompt_config(str(prompt_path))
                except (OSError, ValueError, KeyError) as exc:
                    raise ConfigError(
                        f"Benchmark dataset '{name}' is incompatible with prompt config '{prompt_path}': {exc}"
                    ) from exc
                row_count, materialized_sample = _inspect_jsonl_stream(path, prompt_config=prompt_config)
                detail = f"domain JSONL -> responses_create_params via {prompt_path}"
            else:
                row_count, materialized_sample = _inspect_jsonl_stream(path)
                detail = "materialized Responses API JSONL"
            reports.append(
                DatasetInspection(
                    name,
                    dataset_type,
                    path,
                    row_count,
                    "valid",
                    detail,
                    materialized_sample,
                )
            )
    return tuple(reports)


def _composition_hash(
    config: Mapping[str, Any],
    manifest: EnvironmentManifest | None = None,
    inspected_components: Sequence[ComponentInspection] = (),
) -> str:
    versions = {
        (component.role, component.instance, component.implementation): component.version
        for component in inspected_components
    }
    components = [
        {
            "role": role,
            "instance": instance,
            "implementation": implementation,
            "config": _score_affecting_value(server_config, omit_component_metadata=True),
            "content_version": versions.get((role, instance, implementation))
            or _component_content_version(role, implementation, server_config),
        }
        for role, instance, implementation, server_config in _server_instances(config)
    ]
    if any(component.role == "sandbox_provider" for component in inspected_components):
        for selection in _selected_sandbox_providers(config):
            components.append(
                {
                    "role": "sandbox_provider",
                    "instance": selection.instance,
                    "implementation": selection.implementation,
                    "config": {
                        "provider": _score_affecting_value(selection.config),
                        "default_metadata": _score_affecting_value(selection.default_metadata),
                    },
                    "content_version": versions.get(("sandbox_provider", selection.instance, selection.implementation))
                    or _sandbox_provider_content_version(selection),
                }
            )
    for component in inspected_components:
        if component.role != "rollout_driver":
            continue
        components.append(
            {
                "role": component.role,
                "instance": component.instance,
                "implementation": component.implementation,
                "config": {},
                "content_version": component.version,
            }
        )
    manifest_contract = None
    manifest_files: dict[str, str] = {}
    if manifest is not None:
        manifest_contract = manifest.model_dump(
            mode="json",
            include={
                "kind",
                "integration_profile",
                "reward",
                "determinism",
                "grading_mode",
                "session_model",
                "state",
                "sandbox",
                "canonical_split",
                "standard_prompt_config",
                "requires",
                "provides",
            },
            exclude_none=True,
        )
        prompt_digest = _referenced_file_digest(manifest.standard_prompt_config)
        if prompt_digest is not None:
            manifest_files[f"standard_prompt_config:{manifest.standard_prompt_config}"] = prompt_digest
    payload = json.dumps(
        {
            "components": sorted(
                components, key=lambda item: (item["role"], item["instance"], item["implementation"])
            ),
            "rollout_collection_driver": config.get("rollout_collection_driver"),
            "resolved_overrides": _reported_overrides(config),
            "referenced_files": _score_affecting_file_digests(config),
            "manifest_contract": manifest_contract,
            "manifest_files": manifest_files,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def compute_composition_hash(
    config: Mapping[str, Any] | DictConfig,
    manifest: EnvironmentManifest | None = None,
) -> str:
    """Hash the resolved runtime composition without parsing dataset rows."""

    plain = _as_plain_mapping(config)
    profile = infer_integration_profile(plain)
    components = inspect_components(plain, profile)
    sandbox_components, _decisions = inspect_sandbox_providers(plain, manifest)
    components = (*components, *sandbox_components)
    validate_rollout_driver_contract(plain, require_local_source=manifest is not None)
    return _composition_hash(plain, manifest, components)


_REPORTED_OVERRIDE_KEYS = (
    "agent_name",
    "input_jsonl_fpath",
    "limit",
    "num_repeats",
    "num_repeats_add_seed",
    "policy_model_name",
    "prompt_config",
    "responses_create_params",
    "rollout_collection_driver",
    "skills",
    "split",
)
_OPERATIONAL_DRIVER_KEYS = frozenset(
    {
        "append",
        "concat_shards",
        "disable_aggregation",
        "force",
        "judge_failed_only",
        "materialized_inputs_jsonl_fpath",
        "materialized_jsonl_fpath",
        "num_samples_in_parallel",
        "output_jsonl_fpath",
        "overwrite",
        "policy_api_key",
        "policy_base_url",
        "resume_from_cache",
        "rollouts_jsonl_fpath",
        "wandb_api_key",
        "wandb_name",
        "wandb_project",
    }
)


def _rollout_driver_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the score-affecting top-level namespace visible to a custom driver."""

    if not config.get("rollout_collection_driver"):
        return {}

    # Imported lazily to keep this static-inspection module below the runtime
    # parser while sharing its authoritative operational-key vocabulary.
    from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS

    component_instances = {instance for _role, instance, _implementation, _server in _server_instances(config)}
    excluded = {
        *NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
        *_REPORTED_OVERRIDE_KEYS,
        *_OPERATIONAL_DRIVER_KEYS,
        *component_instances,
    }
    return {
        str(key): _score_affecting_value(value)
        for key, value in sorted(config.items(), key=lambda pair: str(pair[0]))
        if key not in excluded and not is_credential_key(key) and value not in (None, "", {}, [])
    }


def _reported_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    reported = {
        key: _score_affecting_value(config[key])
        for key in _REPORTED_OVERRIDE_KEYS
        if key in config and config[key] not in (None, "", {}, [])
    }
    driver_config = _rollout_driver_config(config)
    if driver_config:
        reported["rollout_driver_config"] = driver_config
    reported.update(_resolved_cli_overrides(config))
    return reported


_OVERRIDE_PATH_PART = re.compile(r"([^.[\]]+)|\[([0-9]+)\]")
_MISSING_OVERRIDE_VALUE = object()


def _resolved_override_value(config: Mapping[str, Any], path: str) -> Any:
    current: Any = config
    for name, index in _OVERRIDE_PATH_PART.findall(path):
        if name:
            if not isinstance(current, Mapping) or name not in current:
                return _MISSING_OVERRIDE_VALUE
            current = current[name]
        else:
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes, bytearray)):
                return _MISSING_OVERRIDE_VALUE
            offset = int(index)
            if offset >= len(current):
                return _MISSING_OVERRIDE_VALUE
            current = current[offset]
    return current


def _resolved_cli_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    raw_paths = config.get("environment_cli_override_paths") or ()
    if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes, bytearray)):
        return {}

    from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS

    excluded_roots = {*NEMO_GYM_RESERVED_TOP_LEVEL_KEYS, *_OPERATIONAL_DRIVER_KEYS}
    resolved: dict[str, Any] = {}
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path:
            continue
        root = raw_path.split(".", 1)[0].split("[", 1)[0]
        leaf = raw_path.rsplit(".", 1)[-1].split("[", 1)[0]
        if root in excluded_roots or is_credential_key(leaf):
            continue
        value = _resolved_override_value(config, raw_path)
        resolved[raw_path] = "<deleted>" if value is _MISSING_OVERRIDE_VALUE else _score_affecting_value(value)
    return resolved


def has_score_affecting_cli_overrides(config: Mapping[str, Any] | DictConfig) -> bool:
    """Return whether a launch intentionally derives from its resolved base composition."""

    plain = _as_plain_mapping(config)
    return bool(plain.get("environment_component_swaps") or _resolved_cli_overrides(plain))


def _responsibility_mapping(
    config: Mapping[str, Any],
    components: Sequence[ComponentInspection],
    datasets: Sequence[DatasetInspection],
    profile: str,
) -> ResponsibilityMapping:
    def component_owners(role: ResponsibilityOwnerRole) -> tuple[ResponsibilityOwner, ...]:
        return tuple(
            ResponsibilityOwner(
                role=role,
                instance=component.instance,
                implementation=component.implementation,
            )
            for component in components
            if component.role == role
        )

    resources = component_owners("resources_server")
    tools_and_state = tuple(
        ResponsibilityOwner(
            role="resources_server",
            instance=component.instance,
            implementation=component.implementation,
        )
        for component in components
        if component.role == "resources_server"
        and any(capability == "tools" or capability.startswith("state:") for capability in component.provides)
    )
    declared_verifiers = tuple(
        ResponsibilityOwner(
            role="resources_server",
            instance=component.instance,
            implementation=component.implementation,
        )
        for component in components
        if component.role == "resources_server" and "verification" in component.provides
    )
    if profile == "external-loop":
        rollout_owners = component_owners("agent_server")
    else:
        driver = config.get("rollout_collection_driver")
        rollout_owners = (
            ResponsibilityOwner(
                role="rollout_driver",
                instance="custom" if driver else "stock",
                implementation=str(driver or "nemo_gym.rollout_collection"),
            ),
        )
    return ResponsibilityMapping(
        task_preparation=tuple(
            ResponsibilityOwner(role="dataset", instance=dataset.name, implementation=str(dataset.path))
            for dataset in datasets
        ),
        model_interaction=(*component_owners("agent_server"), *component_owners("model_server")),
        tools_and_state=tools_and_state,
        # Before capability migration, a resources server still owns verification
        # even when the legacy config has not declared that responsibility yet.
        verification=declared_verifiers or resources,
        rollout_coordination=rollout_owners,
    )


def inspect_workload(
    config: Mapping[str, Any] | DictConfig,
    *,
    strict_missing_datasets: bool = False,
    standard_prompt_config: str | Path | None = None,
    manifest: EnvironmentManifest | None = None,
) -> WorkloadInspection:
    plain = _as_plain_mapping(config)
    profile = infer_integration_profile(plain)
    source_decisions = validate_manifest_component_sources(plain, manifest) if manifest is not None else ()
    components = inspect_components(plain, profile)
    sandbox_components, sandbox_decisions = inspect_sandbox_providers(plain, manifest)
    components = (*components, *sandbox_components)
    capability_decisions = validate_capabilities(components, manifest, config=plain)
    grading_decisions, grading_warnings = validate_grading_mode_constraints(plain, manifest)
    driver_decisions, driver_warnings = validate_rollout_driver_contract(
        plain,
        require_local_source=manifest is not None,
    )
    preparation_decisions = validate_dataset_preparation_sources(plain) if manifest is not None else ()
    runtime_file_decisions = validate_runtime_local_references(plain) if manifest is not None else ()
    datasets = validate_datasets(
        plain,
        strict_missing=strict_missing_datasets,
        standard_prompt_config=standard_prompt_config,
    )
    raw_config_paths = plain.get("config_paths") or []
    config_paths = tuple(str(path) for path in raw_config_paths) if isinstance(raw_config_paths, Sequence) else ()
    warnings = (
        *(
            f"Dataset '{dataset.name}' was not validated because '{dataset.path}' is missing."
            for dataset in datasets
            if dataset.status == "missing"
        ),
        *grading_warnings,
        *driver_warnings,
    )
    fixed_constraints = tuple(
        f"{component.role}:{component.implementation} is pinned by {profile}"
        for component in components
        if component.pinned
    )
    return WorkloadInspection(
        profile=profile,
        components=components,
        datasets=datasets,
        config_paths=config_paths,
        composition_hash=_composition_hash(plain, manifest, components),
        responsibilities=_responsibility_mapping(plain, components, datasets, profile),
        overrides=_reported_overrides(plain),
        fixed_constraints=fixed_constraints,
        compatibility_decisions=(
            *source_decisions,
            *sandbox_decisions,
            *capability_decisions,
            *grading_decisions,
            *driver_decisions,
            *preparation_decisions,
            *runtime_file_decisions,
        ),
        warnings=warnings,
    )
