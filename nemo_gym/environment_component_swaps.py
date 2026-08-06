# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest-bound component replacement for resolved Gym workloads."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf, open_dict

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.config_types import ConfigError
from nemo_gym.environment_manifest import (
    EnvironmentManifest,
    load_manifest,
    parse_python_callable_reference,
)
from nemo_gym.environment_validation import pinned_component_roles


ENVIRONMENT_COMPONENT_SWAPS_KEY = "environment_component_swaps"
_ROLLOUT_DRIVER_ROLE = "rollout_driver"

_GROUP_BY_ROLE = {
    "resources_server": "resources_servers",
    "agent_server": "responses_api_agents",
    "model_server": "responses_api_models",
}
_ROLE_BY_GROUP = {group: role for role, group in _GROUP_BY_ROLE.items()}


@dataclass(frozen=True)
class ComponentDependency:
    """One server referenced by a selected replacement component."""

    role: str
    instance: str
    implementation: str
    server_config: DictConfig


@dataclass(frozen=True)
class ComponentSwap:
    """One replacement selected from an isolated component config."""

    role: str
    config_path: Path
    instance: str
    implementation: str
    server_config: DictConfig
    dependencies: tuple[ComponentDependency, ...] = ()
    recipe_references: tuple[tuple[str, str], ...] = ()
    external_references: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AppliedComponentSwap:
    """Provenance recorded after replacing one manifest component."""

    role: str
    config_path: Path
    instance: str
    declared: str
    selected: str
    implementation: str
    declared_grading_mode: str | None = None
    selected_grading_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config_path": str(self.config_path),
            "instance": self.instance,
            "declared": self.declared,
            "selected": self.selected,
            "implementation": self.implementation,
        }
        if self.role == "resources_server":
            payload["declared_grading_mode"] = self.declared_grading_mode
            payload["selected_grading_mode"] = self.selected_grading_mode
        return payload


def _plain_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, DictConfig):
        plain = OmegaConf.to_container(value, resolve=False, throw_on_missing=False)
        if isinstance(plain, Mapping):
            return plain
    elif isinstance(value, Mapping):
        return value
    raise ConfigError(f"{label} must be a mapping.")


def requested_component_swap_paths(config: Mapping[str, Any] | DictConfig) -> dict[str, Path]:
    """Return validated role-to-config selections from CLI metadata."""

    raw = config.get(ENVIRONMENT_COMPONENT_SWAPS_KEY)
    if raw in (None, {}):
        return {}
    swaps = _plain_mapping(raw, label=ENVIRONMENT_COMPONENT_SWAPS_KEY)
    unknown = sorted(set(map(str, swaps)) - {*_GROUP_BY_ROLE, _ROLLOUT_DRIVER_ROLE})
    if unknown:
        raise ConfigError(f"{ENVIRONMENT_COMPONENT_SWAPS_KEY} contains unsupported role(s): {', '.join(unknown)}.")

    paths: dict[str, Path] = {}
    for role in _GROUP_BY_ROLE:
        value = swaps.get(role)
        if value is None:
            continue
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ConfigError(f"{ENVIRONMENT_COMPONENT_SWAPS_KEY}.{role} must name one config file.")
        path = _resolve_under_cwd_or_install(str(value))
        if not path.is_file():
            raise ConfigError(f"Selected {role.replace('_', ' ')} config does not exist at '{path}'.")
        paths[role] = Path(os.path.abspath(path))
    return paths


def requested_rollout_driver_swap(config: Mapping[str, Any] | DictConfig) -> str | None:
    """Return an explicit unresolved driver selection from CLI swap metadata."""

    raw = config.get(ENVIRONMENT_COMPONENT_SWAPS_KEY)
    if raw in (None, {}):
        return None
    swaps = _plain_mapping(raw, label=ENVIRONMENT_COMPONENT_SWAPS_KEY)
    value = swaps.get(_ROLLOUT_DRIVER_ROLE)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"{ENVIRONMENT_COMPONENT_SWAPS_KEY}.{_ROLLOUT_DRIVER_ROLE} must be an explicit "
            "'module.path:function' selection."
        )
    try:
        parse_python_callable_reference(value, field_name="rollout driver swap")
    except ValueError as error:
        raise ConfigError(str(error)) from error
    return value


def partition_component_swap_paths(
    config_paths: list[str],
    swaps: Mapping[str, Path],
) -> tuple[list[str], dict[str, Path]]:
    """Remove replacement roots from the recipe paths before the normal merge."""

    if not swaps:
        return list(config_paths), {}
    configured = {_resolve_under_cwd_or_install(path).resolve() for path in config_paths}
    selected = {path.resolve() for path in swaps.values()}
    recipe_paths = [path for path in config_paths if _resolve_under_cwd_or_install(path).resolve() not in selected]
    missing = [role for role, path in swaps.items() if path.resolve() not in configured]
    if missing:
        raise ConfigError(
            "Component swap metadata must reference a selected config_paths entry; missing "
            + ", ".join(sorted(missing))
            + "."
        )
    return recipe_paths, dict(swaps)


def select_component_from_config(role: str, config_path: Path, config: DictConfig) -> ComponentSwap:
    """Select the first launched implementation of ``role`` from one isolated config."""

    components = _launched_components(config)
    candidates = [component for component in components if component[0] == role]
    implementations = {component[2] for component in candidates}
    if len(implementations) > 1:
        rendered = ", ".join(sorted(implementations))
        raise ConfigError(
            f"Selected {role.replace('_', ' ')} config '{config_path}' is ambiguous; "
            f"it launches distinct implementations: {rendered}."
        )
    if candidates:
        _role, instance, implementation, server_config = candidates[0]
        by_instance = {component[1]: component for component in components}
        canonical_references: set[tuple[str, str]] = set()
        if role == "agent_server":
            for field_name, group in (
                ("resources_server", "resources_servers"),
                ("model_server", "responses_api_models"),
            ):
                reference = server_config.get(field_name)
                if isinstance(reference, (Mapping, DictConfig)) and isinstance(reference.get("name"), str):
                    canonical_references.add((group, str(reference["name"])))
        dependencies: list[ComponentDependency] = []
        external_references: set[tuple[str, str]] = set()
        pending = [
            reference for reference in _server_references(server_config) if reference not in canonical_references
        ]
        seen: set[tuple[str, str]] = set()
        while pending:
            dependency_group, dependency_instance = pending.pop(0)
            dependency_key = (dependency_group, dependency_instance)
            if dependency_key in seen or dependency_instance == instance:
                continue
            seen.add(dependency_key)
            dependency = by_instance.get(dependency_instance)
            if dependency is None:
                external_references.add((dependency_group, dependency_instance))
                continue
            dependency_role, _, dependency_implementation, dependency_config = dependency
            if _GROUP_BY_ROLE[dependency_role] != dependency_group:
                raise ConfigError(
                    f"Selected {role.replace('_', ' ')} config '{config_path}' references instance "
                    f"'{dependency_instance}' as {dependency_group}, but it provides "
                    f"{_GROUP_BY_ROLE[dependency_role]}."
                )
            dependencies.append(
                ComponentDependency(
                    role=dependency_role,
                    instance=dependency_instance,
                    implementation=dependency_implementation,
                    server_config=OmegaConf.create(deepcopy(OmegaConf.to_container(dependency_config, resolve=False))),
                )
            )
            pending.extend(
                reference
                for reference in _server_references(dependency_config)
                if reference not in canonical_references
            )
        return ComponentSwap(
            role=role,
            config_path=config_path,
            instance=instance,
            implementation=implementation,
            server_config=OmegaConf.create(deepcopy(OmegaConf.to_container(server_config, resolve=False))),
            dependencies=tuple(dependencies),
            recipe_references=tuple(sorted(canonical_references)),
            external_references=tuple(sorted(external_references)),
        )
    raise ConfigError(
        f"Selected {role.replace('_', ' ')} config '{config_path}' does not define a "
        f"{_GROUP_BY_ROLE[role]} implementation."
    )


def _launched_components(config: Mapping[str, Any] | DictConfig) -> list[tuple[str, str, str, Any]]:
    components: list[tuple[str, str, str, Any]] = []
    for instance, value in config.items():
        if not isinstance(value, (DictConfig, Mapping)) or not value:
            continue
        group = next(iter(value))
        role = _ROLE_BY_GROUP.get(group)
        if role is None:
            continue
        implementations = value[group]
        if not isinstance(implementations, (DictConfig, Mapping)) or not implementations:
            continue
        implementation = next(iter(implementations))
        server_config = implementations[implementation]
        if isinstance(server_config, (DictConfig, Mapping)):
            components.append((role, str(instance), str(implementation), server_config))
    return components


def _server_references(value: object) -> set[tuple[str, str]]:
    references: set[tuple[str, str]] = set()
    if isinstance(value, (DictConfig, Mapping)):
        if value.get("type") in _ROLE_BY_GROUP and isinstance(value.get("name"), str):
            references.add((str(value["type"]), str(value["name"])))
        for nested in value.values():
            references.update(_server_references(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            references.update(_server_references(nested))
    return references


def _rewrite_server_references(
    value: object,
    instance_names: Mapping[tuple[str, str], str],
) -> None:
    if isinstance(value, (DictConfig, dict)):
        reference = (str(value.get("type")), str(value.get("name")))
        if reference in instance_names:
            value["name"] = instance_names[reference]
        for nested in value.values():
            _rewrite_server_references(nested, instance_names)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _rewrite_server_references(nested, instance_names)


def _manifest_target(
    components: list[tuple[str, str, str, Any]],
    manifest: EnvironmentManifest,
    role: str,
) -> tuple[str, str, Any]:
    declared = getattr(manifest, role)
    if declared is None:
        raise ConfigError(f"Manifest '{manifest.name}' does not declare a {role.replace('_', ' ')} to replace.")
    matches = [
        (instance, implementation, server_config)
        for candidate_role, instance, implementation, server_config in components
        if candidate_role == role and (implementation == declared or (role == "model_server" and instance == declared))
    ]
    if len(matches) != 1:
        raise ConfigError(
            f"Manifest-selected {role.replace('_', ' ')} {declared!r} must resolve to exactly one "
            f"server instance before it can be replaced; found {len(matches)}."
        )
    return matches[0]


def _server_ref(group: str, instance: str) -> dict[str, str]:
    return {"type": group, "name": instance}


def apply_component_swaps(
    config: DictConfig,
    replacements: Mapping[str, ComponentSwap],
    component_overrides: Mapping[str, Any] | DictConfig | None = None,
    rollout_driver: str | None = None,
) -> tuple[AppliedComponentSwap, ...]:
    """Replace manifest-selected implementations while preserving recipe wiring."""

    if not replacements and rollout_driver is None:
        return ()
    manifest_path = config.get("manifest_path")
    if not manifest_path:
        raise ConfigError("Component replacement requires a manifest-bound environment or benchmark.")
    manifest = load_manifest(_resolve_under_cwd_or_install(str(manifest_path)))
    if "agent_server" in replacements and "agent_server" in pinned_component_roles(manifest.integration_profile.value):
        raise ConfigError(
            f"Profile '{manifest.integration_profile.value}' pins the agent server; the requested swap is invalid."
        )

    components = _launched_components(config)
    targets = {role: _manifest_target(components, manifest, role) for role in replacements}
    resource_target = _manifest_target(components, manifest, "resources_server") if manifest.resources_server else None
    model_target = _manifest_target(components, manifest, "model_server") if manifest.model_server else None
    agent_target = _manifest_target(components, manifest, "agent_server") if manifest.agent_server else None

    applied: list[AppliedComponentSwap] = []
    with open_dict(config):
        for role in ("resources_server", "model_server", "agent_server"):
            replacement = replacements.get(role)
            if replacement is None:
                continue
            target_instance, _target_implementation, old_server_config = targets[role]
            selected_config = OmegaConf.create(
                deepcopy(OmegaConf.to_container(replacement.server_config, resolve=False))
            )

            if role == "agent_server":
                recipe_targets = {
                    "resources_servers": resource_target,
                    "responses_api_models": model_target,
                }
                recipe_reference_names: dict[tuple[str, str], str] = {}
                for reference_group, reference_instance in replacement.recipe_references:
                    recipe_target = recipe_targets.get(reference_group)
                    if recipe_target is None:
                        raise ConfigError(
                            f"The replacement agent requires {reference_group}/'{reference_instance}', "
                            "but the recipe does not select that server type."
                        )
                    recipe_reference_names[(reference_group, reference_instance)] = recipe_target[0]
                if "datasets" in old_server_config:
                    selected_config["datasets"] = deepcopy(old_server_config["datasets"])
                else:
                    selected_config.pop("datasets", None)
                if "resources_server" in selected_config or "resources_server" in old_server_config:
                    if resource_target is None:
                        raise ConfigError("The replacement agent requires a resources server, but none is selected.")
                    selected_config["resources_server"] = _server_ref("resources_servers", resource_target[0])
                if "model_server" in selected_config or "model_server" in old_server_config:
                    if model_target is None:
                        raise ConfigError("The replacement agent requires a model server, but none is selected.")
                    selected_config["model_server"] = _server_ref("responses_api_models", model_target[0])

            override_server_config = None
            if component_overrides is not None:
                override_instance = component_overrides.get(target_instance)
                if isinstance(override_instance, (DictConfig, Mapping)):
                    override_group = override_instance.get(_GROUP_BY_ROLE[role])
                    if isinstance(override_group, (DictConfig, Mapping)):
                        override_server_config = override_group.get(replacement.implementation)
            if isinstance(override_server_config, (DictConfig, Mapping)):
                selected_config = OmegaConf.merge(selected_config, override_server_config)

            dependency_names: dict[tuple[str, str], str] = {
                (_GROUP_BY_ROLE[role], replacement.instance): target_instance,
            }
            if role == "agent_server":
                dependency_names.update(recipe_reference_names)
            live_by_instance = {
                instance: (candidate_role, implementation, server_config)
                for candidate_role, instance, implementation, server_config in _launched_components(config)
            }
            for reference_group, reference_instance in replacement.external_references:
                current = live_by_instance.get(reference_instance)
                if current is None or _GROUP_BY_ROLE[current[0]] != reference_group:
                    raise ConfigError(
                        f"Selected {role.replace('_', ' ')} config '{replacement.config_path}' references "
                        f"missing dependency {reference_group}/'{reference_instance}'. Include its config "
                        "or select a self-contained component."
                    )
                dependency_names[(reference_group, reference_instance)] = reference_instance
            reserved_instances = set(map(str, config))
            for dependency in replacement.dependencies:
                current = live_by_instance.get(dependency.instance)
                same_component = (
                    current is not None
                    and current[0] == dependency.role
                    and current[1] == dependency.implementation
                    and OmegaConf.to_container(current[2], resolve=False)
                    == OmegaConf.to_container(dependency.server_config, resolve=False)
                )
                if same_component:
                    dependency_names[(_GROUP_BY_ROLE[dependency.role], dependency.instance)] = dependency.instance
                    continue
                candidate = dependency.instance
                if candidate in reserved_instances:
                    candidate = f"{target_instance}__{dependency.instance}"
                    suffix = 2
                    while candidate in reserved_instances:
                        candidate = f"{target_instance}__{dependency.instance}_{suffix}"
                        suffix += 1
                dependency_names[(_GROUP_BY_ROLE[dependency.role], dependency.instance)] = candidate
                reserved_instances.add(candidate)

            _rewrite_server_references(selected_config, dependency_names)
            for dependency in replacement.dependencies:
                dependency_instance = dependency_names[(_GROUP_BY_ROLE[dependency.role], dependency.instance)]
                if dependency_instance in live_by_instance:
                    continue
                dependency_config = OmegaConf.create(
                    deepcopy(OmegaConf.to_container(dependency.server_config, resolve=False))
                )
                _rewrite_server_references(dependency_config, dependency_names)
                config[dependency_instance] = {
                    _GROUP_BY_ROLE[dependency.role]: {
                        dependency.implementation: dependency_config,
                    }
                }

            config[target_instance] = {_GROUP_BY_ROLE[role]: {replacement.implementation: selected_config}}
            applied.append(
                AppliedComponentSwap(
                    role=role,
                    config_path=replacement.config_path,
                    instance=target_instance,
                    declared=str(getattr(manifest, role)),
                    selected=replacement.implementation if role != "model_server" else target_instance,
                    implementation=replacement.implementation,
                    declared_grading_mode=(
                        str(manifest.grading_mode) if role == "resources_server" and manifest.grading_mode else None
                    ),
                    selected_grading_mode=(
                        str(selected_config["grading_mode"])
                        if role == "resources_server" and selected_config.get("grading_mode") is not None
                        else None
                    ),
                )
            )

        if agent_target is not None:
            agent_instance = agent_target[0]
            agent_group = config[agent_instance]["responses_api_agents"]
            agent_config = agent_group[next(iter(agent_group))]
            if "resources_server" in replacements and "resources_server" in agent_config:
                assert resource_target is not None
                agent_config["resources_server"] = _server_ref("resources_servers", resource_target[0])
            if "model_server" in replacements and "model_server" in agent_config:
                assert model_target is not None
                agent_config["model_server"] = _server_ref("responses_api_models", model_target[0])

        resolved_metadata: dict[str, Any] = {swap.role: swap.to_dict() for swap in applied}
        if rollout_driver is not None:
            if manifest.rollout_driver is None:
                raise ConfigError(f"Manifest '{manifest.name}' does not declare a rollout driver to replace.")
            if _ROLLOUT_DRIVER_ROLE in pinned_component_roles(manifest.integration_profile.value):
                raise ConfigError(
                    f"Profile '{manifest.integration_profile.value}' pins the rollout driver; "
                    "the requested swap is invalid."
                )
            if config.get("rollout_collection_driver") != rollout_driver:
                raise ConfigError(
                    "The resolved rollout_collection_driver no longer matches its explicit swap selection."
                )
            resolved_metadata[_ROLLOUT_DRIVER_ROLE] = {
                "declared": manifest.rollout_driver,
                "selected": rollout_driver,
                "integration_profile": manifest.integration_profile.value,
            }
        config[ENVIRONMENT_COMPONENT_SWAPS_KEY] = resolved_metadata
    return tuple(applied)


def authorized_swap_roles(config: Mapping[str, Any] | DictConfig) -> dict[str, Mapping[str, Any]]:
    """Return parser-verified replacement metadata, rejecting raw path-only input."""

    raw = config.get(ENVIRONMENT_COMPONENT_SWAPS_KEY)
    if raw in (None, {}):
        return {}
    swaps = _plain_mapping(raw, label=ENVIRONMENT_COMPONENT_SWAPS_KEY)
    authorized: dict[str, Mapping[str, Any]] = {}
    for role, value in swaps.items():
        if role not in {*_GROUP_BY_ROLE, _ROLLOUT_DRIVER_ROLE} or not isinstance(value, Mapping):
            raise ConfigError("Component swap metadata was not resolved by the Gym config parser.")
        required = (
            {"declared", "selected", "integration_profile"}
            if role == _ROLLOUT_DRIVER_ROLE
            else {"config_path", "instance", "declared", "selected", "implementation"}
        )
        if not required.issubset(value):
            raise ConfigError("Component swap metadata was not resolved by the Gym config parser.")
        authorized[str(role)] = value
    return authorized


def authorized_manifest_delta_fields(
    config: Mapping[str, Any] | DictConfig,
    manifest: EnvironmentManifest,
    mirror: object,
) -> frozenset[str]:
    """Validate resolved swap provenance and return the mirror fields it authorizes."""

    swaps = authorized_swap_roles(config)
    if not swaps:
        return frozenset()
    components = {
        (role, instance): implementation
        for role, instance, implementation, _server_config in _launched_components(config)
    }
    config_paths = {_resolve_under_cwd_or_install(str(path)).resolve() for path in (config.get("config_paths") or [])}
    fields: set[str] = set()
    for role, metadata in swaps.items():
        declared = getattr(manifest, role)
        resolved = getattr(mirror, role)
        if str(metadata["declared"]) != str(declared) or str(metadata["selected"]) != str(resolved):
            raise ConfigError(f"Resolved {role.replace('_', ' ')} no longer matches its explicit swap metadata.")
        if role == _ROLLOUT_DRIVER_ROLE:
            if metadata["integration_profile"] != manifest.integration_profile.value:
                raise ConfigError("Resolved rollout driver swap metadata has the wrong integration profile.")
            if role in pinned_component_roles(manifest.integration_profile.value):
                raise ConfigError(
                    f"Profile '{manifest.integration_profile.value}' pins the rollout driver; "
                    "the requested swap is invalid."
                )
            fields.add(role)
            continue
        instance = str(metadata["instance"])
        if components.get((role, instance)) != str(metadata["implementation"]):
            raise ConfigError(
                f"Resolved {role.replace('_', ' ')} implementation no longer matches its explicit swap metadata."
            )
        path = _resolve_under_cwd_or_install(str(metadata["config_path"])).resolve()
        if path not in config_paths:
            raise ConfigError(f"Resolved {role.replace('_', ' ')} swap config '{path}' is absent from config_paths.")
        fields.add(role)
        if role == "resources_server":
            required_grading_fields = {"declared_grading_mode", "selected_grading_mode"}
            if not required_grading_fields.issubset(metadata):
                raise ConfigError("Resolved resources server swap metadata is incomplete.")
            declared_grading_mode = str(manifest.grading_mode) if manifest.grading_mode is not None else None
            resolved_grading_mode = (
                str(getattr(mirror, "grading_mode")) if getattr(mirror, "grading_mode") is not None else None
            )
            if (
                metadata["declared_grading_mode"] != declared_grading_mode
                or metadata["selected_grading_mode"] != resolved_grading_mode
            ):
                raise ConfigError("Resolved grading mode no longer matches its resources server swap metadata.")
            fields.add("grading_mode")
    return frozenset(fields)


__all__ = [
    "ENVIRONMENT_COMPONENT_SWAPS_KEY",
    "ComponentSwap",
    "apply_component_swaps",
    "authorized_manifest_delta_fields",
    "authorized_swap_roles",
    "partition_component_swap_paths",
    "requested_component_swap_paths",
    "requested_rollout_driver_swap",
    "select_component_from_config",
]
