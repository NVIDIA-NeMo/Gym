# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``[gym]`` table of a dataset's ``dataset.toml``: dataset-level settings for its tasks.

``task.toml`` is the task author's file and, once fetched, read-only. What a dataset needs on top
of it goes here, as data: defaults that scale every task, and per-task overrides expressed with
Harbor's own ``[environment]`` fields. The loader applies them before anything else sees a task,
so servers, agents and rows only ever see the effective ``task.toml``. Harbor's own tables in the
same file (``[dataset]`` and friends) are left to Harbor.

Deployment settings (which sandbox provider, registry credentials, image mirrors) do not belong
here: a dataset must run unchanged on every deployment.
"""

import math
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.tasks.harbor.models import HarborEnvironment, HarborTaskConfig


DATASET_FILE = "dataset.toml"
GYM_TABLE = "gym"
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
    """``[gym.tasks."<task_id>"]``: per-task overrides, written with Harbor ``[environment]`` fields.

    ``env`` merges over the task's own; every other field replaces the task's value.
    """

    model_config = ConfigDict(extra="forbid")

    environment: dict[str, Any] = Field(default_factory=dict)


class GymDatasetConfig(BaseModel):
    """The whole ``[gym]`` table."""

    model_config = ConfigDict(extra="forbid")

    defaults: GymDatasetDefaults = Field(default_factory=GymDatasetDefaults)
    tasks: dict[str, GymTaskOverride] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.tasks and self.defaults == GymDatasetDefaults()


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
        return GymDatasetConfig.model_validate(table)
    except ValueError as exc:
        raise ValueError(f"{path} [{GYM_TABLE}]: {exc}") from exc


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
    # Harbor ignores unknown task.toml keys, but an override is written by the dataset author for this
    # loader, so a misspelt field is an error named in Harbor's own vocabulary rather than a silent no-op.
    unknown = HarborEnvironment.unknown_keys(override)
    if unknown:
        raise ValueError(f'[gym.tasks."{task_id}".environment] has unknown keys: {", ".join(unknown)}')
    merged = environment.model_dump(exclude_unset=True)
    for key, value in override.items():
        if key == "env":
            merged["env"] = dict(merged.get("env") or {}) | dict(value)
        else:
            merged[key] = value
    return HarborEnvironment.model_validate(merged)


def apply_dataset_config(config: HarborTaskConfig, task_id: str, dataset: GymDatasetConfig) -> HarborTaskConfig:
    """The effective ``task.toml`` for ``task_id``: overrides first, then the dataset defaults."""
    if dataset.is_empty:
        return config
    defaults = dataset.defaults
    override = dataset.tasks.get(task_id)
    environment = _override_environment(config.environment, override.environment if override else {}, task_id=task_id)
    environment = _scale_environment(environment, defaults.resource_multiplier)
    verifier = config.verifier
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
    return config.model_copy(update={"environment": environment, "agent": agent, "verifier": verifier})
