# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Read suite manifests and recorded artifacts without reconstructing repeat metrics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import orjson
import yaml
from pydantic import ValidationError

from nemo_gym.comparison.loading import _read_agent_entries
from nemo_gym.comparison.statistics.unpaired_welch_fixed_suite import MetricScale, normalize_score
from nemo_gym.comparison.suite.schema import (
    SuiteBenchmarkDefinition,
    SuiteManifest,
    SuiteMetricResult,
    SuiteRunSource,
)
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.path_utils import aggregate_metrics_path_for, materialized_path_for
from nemo_gym.registry import resolve_catalog_entry


@dataclass(frozen=True)
class LoadedSuiteRun:
    """One selected agent's stored summaries and optional planned task identities."""

    rollouts: Path
    aggregate_metrics: Path
    materialized_inputs: Path | None
    agent: str
    repeat_rows: tuple[Mapping[str, object], ...]
    groups: tuple[Mapping[str, object], ...]
    agent_metrics: Mapping[str, object]
    repeat_count: int
    planned_tasks_by_repeat: Mapping[int, frozenset[int]] | None


def load_suite_manifest(path: Path) -> SuiteManifest:
    """Read and validate the explicit suite input contract."""
    try:
        payload = yaml.safe_load(path.read_text())
        return SuiteManifest.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"Cannot load suite manifest '{path}': {error}") from error


def resolve_metric_definitions(benchmark: str, definition: SuiteBenchmarkDefinition) -> list[SuiteMetricResult]:
    """Resolve only mean/reward from Gym's theoretical reward definition."""
    resolved = []
    for metric in definition.metrics:
        if metric.low is not None and metric.high is not None and metric.higher_is_better is not None:
            low, high, direction = metric.low, metric.high, metric.higher_is_better
            source = "suite manifest override"
        else:
            if metric.name != "mean/reward":
                raise ConfigError(
                    f"Benchmark '{benchmark}' metric '{metric.name}' has no general Gym scale definition. "
                    "Specify theoretical low, high, and higher_is_better in the suite manifest."
                )
            try:
                entry = resolve_catalog_entry(benchmark, kind="benchmark")
                if entry.manifest_path is None:
                    raise ConfigError(f"Benchmark '{benchmark}' has no reward manifest.")
                manifest = load_manifest(entry.manifest_path)
            except ConfigError as error:
                raise ConfigError(
                    f"Cannot resolve bounds for '{benchmark}/{metric.name}': {error} "
                    "Specify theoretical low, high, and higher_is_better in the suite manifest."
                ) from error
            low, high = manifest.reward.range
            direction = manifest.reward.higher_is_better
            source = str(entry.manifest_path.resolve()) + ":reward"
        try:
            normalize_score(low, MetricScale(low=low, high=high, higher_is_better=direction))
        except ValueError as error:
            raise ConfigError(f"Invalid scale for '{benchmark}/{metric.name}': {error}") from error
        resolved.append(
            SuiteMetricResult(
                name=metric.name, low=low, high=high, higher_is_better=direction, definition_source=source
            )
        )
    return resolved


def _resolve_path(path: Path, directory: Path) -> Path:
    return (directory / path).resolve() if not path.is_absolute() else path.resolve()


def _object_rows(value: object, *, field: str, path: Path) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ConfigError(f"'{path}' needs '{field}' to be a list of objects.")
    return tuple(value)


def _index(value: object, *, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{context} must be a nonnegative integer, got {value!r}.")
    return value


def _read_planned_tasks(path: Path, agent: str) -> dict[int, frozenset[int]]:
    tasks: dict[int, set[int]] = {}
    try:
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = orjson.loads(line)
                if not isinstance(row, dict):
                    raise ConfigError(f"'{path}' line {line_number} is not an input object.")
                reference = row.get(AGENT_REF_KEY_NAME) or {}
                if not isinstance(reference, dict):
                    raise ConfigError(f"'{path}' line {line_number} has invalid agent_ref.")
                if reference.get("name") not in (None, agent):
                    continue
                context = f"'{path}' line {line_number}"
                task = _index(row.get(TASK_INDEX_KEY_NAME), context=f"{context} task index")
                repeat = _index(row.get(ROLLOUT_INDEX_KEY_NAME), context=f"{context} repeat index")
                target = tasks.setdefault(repeat, set())
                if task in target:
                    raise ConfigError(f"'{path}' duplicates task {task}, repeat {repeat} for agent '{agent}'.")
                target.add(task)
    except (OSError, orjson.JSONDecodeError) as error:
        raise ConfigError(f"Cannot read materialized inputs '{path}': {error}") from error
    if not tasks:
        raise ConfigError(f"Materialized inputs '{path}' contain no tasks for agent '{agent}'.")
    return {repeat: frozenset(indices) for repeat, indices in tasks.items()}


def load_suite_run(source: SuiteRunSource, *, directory: Path) -> LoadedSuiteRun:
    """Load one aggregate file, select its agent, and retain recorded repeat rows."""
    rollouts = _resolve_path(source.rollouts, directory)
    metrics = (
        _resolve_path(source.aggregate_metrics, directory)
        if source.aggregate_metrics is not None
        else aggregate_metrics_path_for(rollouts)
    )
    entries = _read_agent_entries(metrics)
    agent = source.agent
    if agent is None:
        if len(entries) != 1:
            raise ConfigError(
                f"'{metrics}' contains multiple agents ({', '.join(sorted(entries))}); "
                "set 'agent' on this source in the suite manifest."
            )
        agent = next(iter(entries))
    if agent not in entries:
        raise ConfigError(f"Agent '{agent}' is not in '{metrics}'. Available agents: {', '.join(sorted(entries))}.")
    entry = entries[agent]
    if "repeat_level_metrics" not in entry:
        raise ConfigError(
            f"'{metrics}' agent '{agent}' lacks repeat_level_metrics. "
            "Regenerate it with `gym eval aggregate`; suite comparison does not reconstruct repeat metrics."
        )
    rows = _object_rows(entry["repeat_level_metrics"], field="repeat_level_metrics", path=metrics)
    groups = _object_rows(entry.get("group_level_metrics", []), field="group_level_metrics", path=metrics)
    repeat_indices = [_index(row.get(ROLLOUT_INDEX_KEY_NAME), context=f"'{metrics}' repeat index") for row in rows]
    if len(set(repeat_indices)) != len(repeat_indices):
        raise ConfigError(f"'{metrics}' agent '{agent}' contains duplicate repeat indices.")
    rows = tuple(sorted(rows, key=lambda row: row[ROLLOUT_INDEX_KEY_NAME]))
    agent_metrics = entry.get("agent_metrics") or {}
    if not isinstance(agent_metrics, dict):
        raise ConfigError(f"'{metrics}' agent_metrics must be an object.")
    count = len(rows)
    if not rows:
        recorded_count = agent_metrics.get("num_repeats")
        expected = [group.get("expected_num_rollouts") for group in groups]
        if recorded_count == 1 or (expected and all(value == 1 for value in expected)):
            count = 1
        else:
            raise ConfigError(
                f"'{metrics}' agent '{agent}' has no stored repeat rows and no verified single-repeat count. "
                "Regenerate aggregate metrics with `gym eval aggregate`."
            )
    inputs = (
        _resolve_path(source.materialized_inputs, directory)
        if source.materialized_inputs is not None
        else materialized_path_for(rollouts)
    )
    planned = _read_planned_tasks(inputs, agent) if source.materialized_inputs is not None or inputs.exists() else None
    return LoadedSuiteRun(
        rollouts=rollouts,
        aggregate_metrics=metrics,
        materialized_inputs=inputs if planned is not None else None,
        agent=agent,
        repeat_rows=rows,
        groups=groups,
        agent_metrics=agent_metrics,
        repeat_count=count,
        planned_tasks_by_repeat=planned,
    )
