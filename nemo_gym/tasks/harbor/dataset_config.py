# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``[gym]`` table of a dataset's ``dataset.toml``: dataset-level settings for its tasks.

``task.toml`` is the task author's file and, once fetched, read-only. What a dataset needs on top
of it goes here, as data: the facts Harbor has no field for (``license`` and ``type``, which say
which split the rows may enter), defaults that scale every task, and per-task overrides expressed
with Harbor's own ``[environment]`` fields. The loader applies the task-shaping settings before
anything else sees a task, so servers, agents and rows only ever see the effective ``task.toml``.
Harbor's own tables in the same file (``[dataset]`` and friends) are left to Harbor.

Deployment settings (which sandbox provider, registry credentials, image mirrors) do not belong
here: a dataset must run unchanged on every deployment.
"""

import math
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.config_types import DatasetLicense
from nemo_gym.tasks.harbor.models import HarborArtifact, HarborEnvironment, HarborTaskConfig, HarborVerifier


DATASET_FILE = "dataset.toml"
GYM_TABLE = "gym"
# Gym's existing dataset splits, plus `benchmark`: evaluates, never enters a training split.
DatasetSplit = Literal["train", "validation", "example", "benchmark"]
TRAINING_SPLITS = ("train", "validation")
# [environment] fields that scale with `resource_multiplier`; GPUs never do.
SCALED_RESOURCES = ("cpus", "memory_mb", "storage_mb")


class GymDatasetDefaults(BaseModel):
    """``[gym.defaults]``: generic policy applied to every task of the dataset."""

    model_config = ConfigDict(extra="forbid")

    # Scales [agent].timeout_sec and [verifier].timeout_sec. Task timeouts are authored against Harbor's
    # own runtime; a slower deployment raises this instead of editing tasks.
    timeout_multiplier: float = Field(default=1.0, gt=0)
    # Scales cpus, memory_mb and storage_mb of the agent's and the verifier's environments.
    resource_multiplier: float = Field(default=1.0, gt=0)


class GymTaskOverride(BaseModel):
    """``[gym.tasks."<task_id>"]``: per-task overrides, written with Harbor's own fields.

    ``environment`` and ``verifier`` take any key of the task's ``[environment]`` and ``[verifier]``
    tables; ``env`` merges over the task's own and every other field replaces the task's value.
    ``artifacts`` are added after the task's own ``artifacts`` entries, in Harbor's shape (a path, or a
    table with ``source``, ``destination`` and ``service``).
    """

    model_config = ConfigDict(extra="forbid")

    environment: dict[str, Any] = Field(default_factory=dict)
    verifier: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Any] = Field(default_factory=list)


class GymDatasetConfig(BaseModel):
    """The whole ``[gym]`` table."""

    model_config = ConfigDict(extra="forbid")

    # One of Gym's dataset license values; required for the `train` and `validation` splits, Gym's rule
    # for every dataset. A hub dataset without one evaluates as a benchmark and never enters training.
    license: DatasetLicense | None = None
    # Which split the rows enter. Unset means `validation` when a license is set, else `benchmark`.
    type: DatasetSplit | None = None
    defaults: GymDatasetDefaults = Field(default_factory=GymDatasetDefaults)
    tasks: dict[str, GymTaskOverride] = Field(default_factory=dict)
    # The dataset.toml this table was read from, so that errors in applying it name the file.
    _path: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _training_splits_need_a_license(self) -> "GymDatasetConfig":
        if self.type in TRAINING_SPLITS and self.license is None:
            raise ValueError(f"type = {self.type!r} requires a license (Gym's rule for train and validation sets)")
        return self

    @property
    def effective_type(self) -> str:
        """The split the rows enter: the declared one, else ``validation`` with a license, else ``benchmark``."""
        if self.type is not None:
            return self.type
        return "validation" if self.license is not None else "benchmark"

    @property
    def shapes_tasks(self) -> bool:
        """Whether any setting changes the effective ``task.toml`` (identity fields never do)."""
        return bool(self.tasks) or self.defaults != GymDatasetDefaults()

    @property
    def is_empty(self) -> bool:
        return not self.shapes_tasks and self.license is None and self.type is None

    @property
    def path(self) -> Path | None:
        return self._path


def read_dataset_config(folder: Path) -> GymDatasetConfig:
    """The ``[gym]`` table of ``<folder>/dataset.toml``; empty when the file or the table is absent."""
    path = Path(folder) / DATASET_FILE
    if not path.is_file():
        return GymDatasetConfig()
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    table = data.get(GYM_TABLE) or {}
    if not isinstance(table, dict):
        raise ValueError(f"{path}: [{GYM_TABLE}] must be a table")
    try:
        config = GymDatasetConfig.model_validate(table)
    except ValueError as exc:
        raise ValueError(f"{path} [{GYM_TABLE}]: {exc}") from exc
    config._path = path
    return config


def _scale_environment(environment: HarborEnvironment, multiplier: float) -> HarborEnvironment:
    if multiplier == 1:
        return environment
    update = {
        name: math.ceil(getattr(environment, name) * multiplier)
        for name in SCALED_RESOURCES
        if getattr(environment, name) is not None
    }
    return environment.model_copy(update=update) if update else environment


def _override_environment(
    environment: HarborEnvironment, override: dict[str, Any], *, task_id: str
) -> HarborEnvironment:
    if not override:
        return environment
    table = f'[gym.tasks."{task_id}".environment]'
    # Harbor ignores unknown task.toml keys, but an override is written by the dataset author for this
    # loader, so a misspelt field is an error named in Harbor's own vocabulary rather than a silent no-op.
    unknown = HarborEnvironment.unknown_keys(override)
    if unknown:
        raise ValueError(f"{table} has unknown keys: {', '.join(unknown)}")
    env = override.get("env")
    if env is not None and not (isinstance(env, dict) and all(isinstance(v, str) for v in env.values())):
        raise ValueError(f"{table}.env must be a table of strings")
    # Harbor's legacy spellings (`memory`, `storage`, `allow_internet`) are normalized to the current field
    # names first, so an override written that way replaces the task's value instead of competing with it
    # (`memory` next to the task's `memory_mb`) or leaving it in place (`allow_internet` next to `network_mode`).
    try:
        normalized = HarborEnvironment.read_deprecated_fields(dict(override))
    except ValueError as exc:
        raise ValueError(f"{table}: {exc}") from exc
    merged = environment.model_dump(exclude_unset=True)
    for key, value in normalized.items():
        if key == "env":
            merged["env"] = dict(merged.get("env") or {}) | dict(value)
        else:
            merged[key] = value
    try:
        return HarborEnvironment.model_validate(merged)
    except ValueError as exc:
        raise ValueError(f"{table}: {exc}") from exc


def _override_verifier(verifier: HarborVerifier, override: dict[str, Any], *, task_id: str) -> HarborVerifier:
    if not override:
        return verifier
    table = f'[gym.tasks."{task_id}".verifier]'
    unknown = HarborVerifier.unknown_keys(override)
    if unknown:
        raise ValueError(f"{table} has unknown keys: {', '.join(unknown)}")
    env = override.get("env")
    if env is not None and not (isinstance(env, dict) and all(isinstance(v, str) for v in env.values())):
        # Checked before the merge, as `_override_environment` does: `dict(1)` below would be a TypeError, which
        # neither the loader nor the base-image scan catches, so one malformed task would abort the whole dataset's
        # preparation instead of being skipped with its reason.
        raise ValueError(f"{table}.env must be a table of strings")
    merged = verifier.model_dump(exclude_unset=True)
    for key, value in override.items():
        if key == "env":
            merged["env"] = dict(merged.get("env") or {}) | dict(value)
        elif key == "environment" and isinstance(value, dict) and isinstance(merged.get("environment"), dict):
            # The verifier's own sandbox is overridden like the agent's: `env` merges, other fields replace.
            base = HarborEnvironment.model_validate(merged["environment"])
            merged["environment"] = _override_environment(base, value, task_id=task_id).model_dump(exclude_unset=True)
        else:
            merged[key] = value
    try:
        return HarborVerifier.model_validate(merged)
    except ValueError as exc:
        raise ValueError(f'[gym.tasks."{task_id}".verifier]: {exc}') from exc


def _added_artifacts(entries: list[Any], *, task_id: str) -> list[HarborArtifact]:
    try:
        return [HarborArtifact.model_validate(entry) for entry in entries]
    except ValueError as exc:
        raise ValueError(f'[gym.tasks."{task_id}".artifacts]: {exc}') from exc


def apply_dataset_config(config: HarborTaskConfig, task_id: str, dataset: GymDatasetConfig) -> HarborTaskConfig:
    """The effective ``task.toml`` for ``task_id``: overrides first, then the dataset defaults."""
    if not dataset.shapes_tasks:
        return config
    defaults = dataset.defaults
    override = dataset.tasks.get(task_id)
    try:
        environment = _override_environment(
            config.environment, override.environment if override else {}, task_id=task_id
        )
    except ValueError as exc:
        raise ValueError(f"{dataset.path or DATASET_FILE}: {exc}") from exc
    environment = _scale_environment(environment, defaults.resource_multiplier)
    verifier = _override_verifier(config.verifier, override.verifier if override else {}, task_id=task_id)
    artifacts = config.artifacts + (_added_artifacts(override.artifacts, task_id=task_id) if override else [])
    verifier_update: dict[str, Any] = {}
    if defaults.timeout_multiplier != 1:
        verifier_update["timeout_sec"] = verifier.timeout_sec * defaults.timeout_multiplier
    if verifier.environment is not None and defaults.resource_multiplier != 1:
        verifier_update["environment"] = _scale_environment(verifier.environment, defaults.resource_multiplier)
    if verifier_update:
        verifier = verifier.model_copy(update=verifier_update)
    agent = config.agent
    if defaults.timeout_multiplier != 1:
        agent = agent.model_copy(update={"timeout_sec": agent.timeout_sec * defaults.timeout_multiplier})
    return config.model_copy(
        update={"environment": environment, "agent": agent, "verifier": verifier, "artifacts": artifacts}
    )
