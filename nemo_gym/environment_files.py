# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover repository-local runtime inputs declared by resolved Gym config."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS


_RUNTIME_INPUT_PATH_FIELDS = frozenset(
    {
        "bird_sql_dir",
        "container_sqsh_path",
        "dataset_path",
        "exclude_domains_file_path",
        "gdpval_container_path",
        "judge_prompt_path",
        "judge_prompt_template_fpath",
        "judge_prompt_template_protocol_fpath",
        "media_base_dir",
        "policy_verifier_templates_path",
        "prompt_fpath",
        "reference_deliverables_dir",
        "retrieval_system_prompt_fpath",
        "rubric_fpath",
        "spider2_lite_dir",
        "test_data_fpath",
        "test_file",
        "turn2_prompt_fpath",
    }
)

# These paths are produced by an explicit preparation step. If materialized,
# they are versioned like other inputs; their absence is handled by that step.
_PREPARED_RUNTIME_INPUT_PATH_FIELDS = frozenset(
    {
        "harbor_tasks_cache_dir",
        "harbor_tasks_dir",
        "harness_skills_dir",
        "local_dataset_path",
        "skills_dir",
    }
)
_NON_RUNTIME_SOURCE_DIRS = frozenset(
    {".git", ".gym", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "venv"}
)


def is_runtime_source_path(path: Path) -> bool:
    """Whether a path below an explicit runtime directory belongs to its immutable input tree."""

    return not any(
        part.casefold() in _NON_RUNTIME_SOURCE_DIRS for part in path.parts[:-1]
    ) and path.suffix.casefold() not in {
        ".pyc",
        ".pyo",
    }


@dataclass(frozen=True)
class RuntimeLocalReference:
    """A path-like config field and its safe local target, when available."""

    field: str
    reference: str
    path: Path | None


def _plain_config(config: Mapping[str, Any] | DictConfig) -> Mapping[str, Any]:
    if isinstance(config, DictConfig):
        plain = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        if not isinstance(plain, Mapping):
            raise ConfigError("Resolved Gym configuration must be a mapping.")
        return plain
    return config


def _looks_like_input_path(key: object) -> bool:
    normalized = str(key).strip().casefold().replace("-", "_")
    return normalized in _RUNTIME_INPUT_PATH_FIELDS or normalized in _PREPARED_RUNTIME_INPUT_PATH_FIELDS


def _is_prepared_input(field: str) -> bool:
    return field.rpartition(".")[2] in _PREPARED_RUNTIME_INPUT_PATH_FIELDS


def iter_runtime_path_values(config: Mapping[str, Any] | DictConfig) -> Iterable[tuple[str, str]]:
    """Yield path-like runtime inputs declared anywhere in resolved config."""

    def walk(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not prefix and str(key) in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS:
                    continue
                location = f"{prefix}.{key}" if prefix else str(key)
                if str(key).strip().casefold().replace("-", "_") == "runtime_input_paths":
                    if isinstance(item, Mapping):
                        for label, reference in item.items():
                            if isinstance(reference, str) and reference:
                                yield f"{location}.{label}", reference
                    elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                        for index, reference in enumerate(item):
                            if isinstance(reference, str) and reference:
                                yield f"{location}[{index}]", reference
                    continue
                if _looks_like_input_path(key) and isinstance(item, str) and item:
                    yield location, item
                yield from walk(item, location)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                yield from walk(item, f"{prefix}[{index}]")

    yield from walk(_plain_config(config))


def resolve_runtime_local_references(
    config: Mapping[str, Any] | DictConfig,
    *,
    repo_root: Path,
    require_existing: bool = False,
    allow_external: bool = False,
    base_directories: Mapping[str, Path] | None = None,
) -> tuple[RuntimeLocalReference, ...]:
    """Resolve path-like inputs without permitting out-of-repository or symlink targets."""

    root = Path(os.path.abspath(repo_root))
    references: list[RuntimeLocalReference] = []
    seen: set[tuple[str, str]] = set()
    for field, reference in iter_runtime_path_values(config):
        if (field, reference) in seen:
            continue
        seen.add((field, reference))
        raw_candidate = Path(reference)
        instance = field.partition(".")[0]
        component_base = base_directories.get(instance) if base_directories else None
        component_root: Path | None = None
        if component_base is not None:
            component_base = Path(os.path.abspath(component_base))
            registry_names = {
                "benchmarks",
                "environments",
                "example_environments",
                "resources_servers",
                "responses_api_agents",
                "responses_api_models",
            }
            component_root = next(
                (
                    parent.parent
                    for parent in (component_base, *component_base.parents)
                    if parent.name in registry_names
                ),
                None,
            )
        candidate_options: list[tuple[Path, Path]] = []
        if raw_candidate.is_absolute():
            candidate_options.append((raw_candidate, component_root or root))
        else:
            if component_base is not None and component_root is not None:
                candidate_options.extend(
                    (
                        (component_base / raw_candidate, component_root),
                        (component_root / raw_candidate, component_root),
                    )
                )
            candidate_options.append((root / raw_candidate, root))
        selected_path, allowed_root = next(
            (
                (Path(os.path.abspath(path)), Path(os.path.abspath(candidate_root)))
                for path, candidate_root in candidate_options
                if path.exists()
            ),
            (
                Path(os.path.abspath(candidate_options[0][0])),
                Path(os.path.abspath(candidate_options[0][1])),
            ),
        )
        candidate = selected_path
        if not candidate.exists():
            if require_existing and not _is_prepared_input(field):
                raise ConfigError(f"Runtime input {field}='{reference}' does not exist.")
            references.append(RuntimeLocalReference(field, reference, None))
            continue
        if candidate.is_symlink():
            raise ConfigError(f"Runtime input {field}='{reference}' is a symbolic-link path.")
        try:
            relative = candidate.relative_to(allowed_root)
        except ValueError as error:
            if allow_external:
                if candidate.is_symlink():
                    raise ConfigError(f"Runtime input {field}='{reference}' is a symbolic link.") from error
                references.append(RuntimeLocalReference(field, reference, candidate))
                continue
            raise ConfigError(
                f"Runtime input {field}='{reference}' is outside repository or allowed source root '{allowed_root}'."
            ) from error
        cursor = allowed_root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ConfigError(f"Runtime input {field}='{reference}' uses symbolic-link path '{cursor}'.")
        try:
            candidate.resolve(strict=True).relative_to(allowed_root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as error:
            raise ConfigError(
                f"Runtime input {field}='{reference}' resolves outside repository or allowed source root "
                f"'{allowed_root}'."
            ) from error
        references.append(RuntimeLocalReference(field, reference, candidate))
    return tuple(references)


__all__ = [
    "RuntimeLocalReference",
    "is_runtime_source_path",
    "iter_runtime_path_values",
    "resolve_runtime_local_references",
]
