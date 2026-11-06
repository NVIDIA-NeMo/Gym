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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import orjson
import yaml
from pydantic import ValidationError

from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.keys import (
    AGENT_REF_KEY_NAME,
    ENVIRONMENT_SERVER_STAMP_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.eval.comparison.loading import _read_agent_entries
from nemo_gym.eval.comparison.statistics.unpaired_welch_fixed_suite import MetricScale, normalize_score
from nemo_gym.eval.comparison.suite.schema import (
    SuiteBenchmarkDefinition,
    SuiteManifest,
    SuiteMetricResult,
    SuiteRunSource,
)
from nemo_gym.rollouts.rollout_records import resolve_rollout_path
from nemo_gym.utils.path_utils import aggregate_metrics_path_for, materialized_path_for


@dataclass(frozen=True)
class LoadedSuiteRun:
    """One selected aggregate entry and its optional planned task identities."""

    rollouts: Path
    aggregate_metrics: Path
    materialized_inputs: Path | None
    agent: str
    environment_server: str | None
    repeat_rows: tuple[Mapping[str, object], ...]
    groups: tuple[Mapping[str, object], ...]
    agent_metrics: Mapping[str, object]
    repeat_count: int
    planned_tasks_by_repeat: Mapping[int, frozenset[int]] | None
    unscored_repeat_indices: tuple[int, ...] = ()


def load_suite_manifest(path: Path) -> SuiteManifest:
    """Read and validate the explicit suite input contract."""
    try:
        payload = yaml.safe_load(path.read_text())
        return SuiteManifest.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"Cannot load suite manifest '{path}': {error}") from error


def resolve_metric_definitions(
    benchmark: str,
    definition: SuiteBenchmarkDefinition,
    *,
    reward_metric_resolver: Callable[[str], SuiteMetricResult] | None = None,
) -> list[SuiteMetricResult]:
    """Validate explicit metric scales, using supplied catalog metadata for mean/reward."""
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
            if reward_metric_resolver is None:
                raise ConfigError(
                    f"Benchmark '{benchmark}' metric '{metric.name}' has no catalog reward resolver. "
                    "Specify theoretical low, high, and higher_is_better in the suite manifest, "
                    "or supply reward_metric_resolver when calling build_suite_comparison."
                )
            try:
                reward = reward_metric_resolver(benchmark)
            except ConfigError as error:
                raise ConfigError(
                    f"Cannot resolve bounds for '{benchmark}/{metric.name}': {error} "
                    "Specify theoretical low, high, and higher_is_better in the suite manifest."
                ) from error
            low, high = reward.low, reward.high
            direction = reward.higher_is_better
            source = reward.definition_source
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


def _object_rows(value: object, *, field: str, path: Path) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ConfigError(f"'{path}' needs '{field}' to be a list of objects.")
    return tuple(value)


def _index(value: object, *, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{context} must be a nonnegative integer, got {value!r}.")
    return value


def _select_entry(
    entries: Mapping[str, dict[str, object]], source: SuiteRunSource, *, path: Path
) -> tuple[str, dict[str, object]]:
    """Resolve optional selectors independently for one source's aggregate file."""
    matches = [
        (agent, entry)
        for agent, entry in entries.items()
        if (source.agent is None or agent == source.agent)
        and (
            source.environment_server is None
            or entry.get(ENVIRONMENT_SERVER_STAMP_KEY_NAME) == source.environment_server
        )
    ]
    if len(matches) == 1:
        return matches[0]
    available = "; ".join(
        f"agent={agent!r}, environment_server={entry.get(ENVIRONMENT_SERVER_STAMP_KEY_NAME)!r}"
        for agent, entry in sorted(entries.items())
    )
    selectors = " and ".join(
        f"{name}={value!r}"
        for name, value in (("agent", source.agent), ("environment_server", source.environment_server))
        if value is not None
    )
    if not matches:
        raise ConfigError(f"No aggregate entry in '{path}' matches {selectors}. Available entries: {available}.")
    if not selectors:
        raise ConfigError(
            f"'{path}' contains multiple aggregate entries; set 'agent' or 'environment_server' "
            f"on this source in the suite manifest. Available entries: {available}."
        )
    raise ConfigError(
        f"'{path}' matches multiple aggregate entries for {selectors}; set both 'agent' and "
        f"'environment_server' to select one entry. Available entries: {available}."
    )


def _read_planned_tasks(path: Path, agent: str, *, environment_server: str | None) -> dict[int, frozenset[int]]:
    """Select the plan using the same server identity as aggregation, with unstamped-row fallback."""
    tasks: dict[int, set[int]] = {}
    try:
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = orjson.loads(line)
                if not isinstance(row, dict):
                    raise ConfigError(f"'{path}' line {line_number} is not an input object.")
                server = row.get(ENVIRONMENT_SERVER_STAMP_KEY_NAME)
                if server is not None:
                    if not isinstance(server, str) or not server:
                        raise ConfigError(f"'{path}' line {line_number} has invalid _ng_environment_server.")
                    if environment_server is None:
                        raise ConfigError(
                            f"Cannot match stamped materialized inputs '{path}' line {line_number}: selected "
                            f"aggregate entry '{agent}' has no _ng_environment_server. "
                            "Regenerate aggregate metrics with `gym eval aggregate`."
                        )
                    if server != environment_server:
                        continue
                else:
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
                    raise ConfigError(
                        f"'{path}' duplicates task {task}, repeat {repeat} for aggregate entry '{agent}'."
                    )
                target.add(task)
    except (OSError, orjson.JSONDecodeError) as error:
        raise ConfigError(f"Cannot read materialized inputs '{path}': {error}") from error
    if not tasks:
        raise ConfigError(
            f"Materialized inputs '{path}' contain no tasks for aggregate entry '{agent}' "
            f"(environment_server={environment_server!r}). "
            "Check the source selectors and the rollout's materialized-input companion."
        )
    return {repeat: frozenset(indices) for repeat, indices in tasks.items()}


def _repeat_count(
    *,
    rows: tuple[Mapping[str, object], ...],
    groups: tuple[Mapping[str, object], ...],
    agent_metrics: Mapping[str, object],
    path: Path,
    agent: str,
) -> int:
    """Retain verified zero/single-repeat counts when no repeat estimates are stored."""
    if rows:
        return len(rows)
    masked = agent_metrics.get("coverage/masked_rollouts")
    if agent_metrics.get("coverage/measured_rollouts") == 0 and isinstance(masked, (int, float)) and masked > 0:
        return 0
    expected = [group.get("expected_num_rollouts") for group in groups]
    if agent_metrics.get("num_repeats") == 1 or (expected and all(value == 1 for value in expected)):
        return 1
    raise ConfigError(
        f"'{path}' agent '{agent}' has no stored repeat rows and no verified single-repeat count. "
        "Regenerate aggregate metrics with `gym eval aggregate`."
    )


def load_suite_run(source: SuiteRunSource, *, directory: Path) -> LoadedSuiteRun:
    """Select one aggregate entry by agent and/or server, retaining its recorded repeat rows."""
    rollouts = resolve_rollout_path(directory / source.rollouts, read_only=True)
    metrics = aggregate_metrics_path_for(rollouts)
    entries = _read_agent_entries(metrics)
    agent, entry = _select_entry(entries, source, path=metrics)
    environment_server = entry.get(ENVIRONMENT_SERVER_STAMP_KEY_NAME)
    if environment_server is not None and (not isinstance(environment_server, str) or not environment_server):
        raise ConfigError(f"'{metrics}' aggregate entry '{agent}' has invalid _ng_environment_server.")
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
    count = _repeat_count(rows=rows, groups=groups, agent_metrics=agent_metrics, path=metrics, agent=agent)
    inputs = materialized_path_for(rollouts)
    planned = _read_planned_tasks(inputs, agent, environment_server=environment_server) if inputs.exists() else None
    return LoadedSuiteRun(
        rollouts=rollouts,
        aggregate_metrics=metrics,
        materialized_inputs=inputs if planned is not None else None,
        agent=agent,
        environment_server=environment_server,
        repeat_rows=rows,
        groups=groups,
        agent_metrics=agent_metrics,
        repeat_count=count,
        planned_tasks_by_repeat=planned,
    )
