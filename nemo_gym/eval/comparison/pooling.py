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
"""Pool independent repeat observations without averaging run-level estimates."""

from math import isfinite, sqrt
from statistics import fmean, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeGuard

from scipy.stats import t

from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.global_config import REWARD_KEY_NAME, ROLLOUT_INFOS_KEY_NAME
from nemo_gym.config.keys import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.eval.comparison.loading import LoadedRun, is_token_accounting_only
from nemo_gym.metrics.metrics_config import COMPLETION_TOKEN_METRIC_NAMES, Stat, is_primary_metric


def _finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _validated_rows(run: LoadedRun, label: str) -> List[Dict[str, Any]]:
    """Validate the observation structure before collecting any metric samples."""
    rows = run.repeat_level_metrics
    if not isinstance(rows, list) or not rows:
        raise ConfigError(
            f"Cannot pool '{label}' for agent '{run.agent_name}': no repeat-level observations were recorded. "
            "Single-repeat aggregates do not store quality repeat-level estimates. "
            "Collect at least two scored repeats per source and regenerate aggregate metrics; "
            "pooling requires stored repeat_level_metrics in every source."
        )
    indices = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ConfigError(f"Cannot pool '{label}': repeat row {position} is not an object.")
        index = row.get(ROLLOUT_INDEX_KEY_NAME)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ConfigError(
                f"Cannot pool '{label}': repeat row {position} requires a nonnegative integer "
                f"{ROLLOUT_INDEX_KEY_NAME}."
            )
        if index in indices:
            raise ConfigError(f"Cannot pool '{label}': duplicate local repeat index {index}.")
        indices.add(index)
        for name in ("sample_count", "missing_count"):
            if name in row and (isinstance(row[name], bool) or not isinstance(row[name], int) or row[name] < 0):
                raise ConfigError(f"Cannot pool '{label}': repeat {index} has an invalid {name}.")
    return rows


def _quality_rows(run: LoadedRun, rows: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """Keep scored metric estimates while recognizing accounting-only producer rows."""
    scored_indices = set()
    for group in run.group_level_metrics:
        infos = group.get(ROLLOUT_INFOS_KEY_NAME) if isinstance(group, dict) else None
        if not isinstance(infos, list):
            continue
        for info in infos:
            index = info.get(ROLLOUT_INDEX_KEY_NAME) if isinstance(info, dict) else None
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
                scored_indices.add(index)
    token_rows = [row for row in rows if is_token_accounting_only(row)]
    if any(row[ROLLOUT_INDEX_KEY_NAME] in scored_indices for row in token_rows):
        raise ConfigError(
            f"Cannot pool '{label}': scored repeats have no stored quality repeat-level estimates. "
            "Single-repeat aggregates do not store those estimates. "
            "Collect at least two scored repeats per source and regenerate aggregate metrics; "
            "whole-run point estimates cannot replace missing repeat observations."
        )
    return [row for row in rows if not is_token_accounting_only(row)]


def _pooled_metric_samples(
    name: str, rows_by_source: Sequence[List[Dict[str, Any]]], source_labels: Sequence[str]
) -> Tuple[List[float], List[str]]:
    """Require complete quality samples and retain explicitly available token accounting."""
    values = []
    notes = []
    token_metric = name in COMPLETION_TOKEN_METRIC_NAMES
    for label, rows in zip(source_labels, rows_by_source, strict=True):
        missing = []
        for row in rows:
            if token_metric and name not in row:
                missing.append(row[ROLLOUT_INDEX_KEY_NAME])
                continue
            value = row.get(name)
            if not _finite_number(value):
                raise ConfigError(
                    f"Cannot pool metric '{name}': source '{label}', repeat "
                    f"{row[ROLLOUT_INDEX_KEY_NAME]} has a missing or nonfinite numeric value. "
                    "Every constituent must provide every pooled metric for all its scored repeats; "
                    "only absent token accounting may be skipped."
                )
            values.append(float(value))
        if missing:
            notes.append(
                f"Source '{label}': metric '{name}' has no token accounting for local repeats {sorted(missing)}. "
                "Its pooled estimate uses available observations only."
            )
    return values, notes


def _pooled_groups(runs: Sequence[LoadedRun]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Recompute per-task rewards from observations; never average task summaries."""
    by_task: Dict[int, List[Dict[str, Any]]] = {}
    task_sets = []
    for run in runs:
        seen_tasks = set()
        if not run.group_level_metrics:
            return [], "a pooled source recorded no per-task rollout summaries."
        for group in run.group_level_metrics:
            if not isinstance(group, dict):
                return [], "a pooled source contains malformed per-task summaries."
            task_index = group.get(TASK_INDEX_KEY_NAME)
            infos = group.get(ROLLOUT_INFOS_KEY_NAME)
            if (
                isinstance(task_index, bool)
                or not isinstance(task_index, int)
                or task_index < 0
                or task_index in seen_tasks
                or not isinstance(infos, list)
                or not infos
            ):
                return [], "pooling task flips requires unique task indices and stored rollout_infos in every source."
            seen_tasks.add(task_index)
            seen_repeats = set()
            for info in infos:
                if not isinstance(info, dict) or not _finite_number(info.get(REWARD_KEY_NAME)):
                    return [], "pooling task flips requires finite reward observations in every source."
                repeat_index = info.get(ROLLOUT_INDEX_KEY_NAME)
                if (
                    isinstance(repeat_index, bool)
                    or not isinstance(repeat_index, int)
                    or repeat_index < 0
                    or repeat_index in seen_repeats
                    or info.get(TASK_INDEX_KEY_NAME, task_index) != task_index
                ):
                    return [], "pooling task flips requires distinct local repeats with consistent task indices."
                seen_repeats.add(repeat_index)
                collected = by_task.setdefault(task_index, [])
                collected.append({**info, ROLLOUT_INDEX_KEY_NAME: len(collected)})
        task_sets.append(seen_tasks)
    if any(tasks != task_sets[0] for tasks in task_sets[1:]):
        return [], "pooled sources cover different task indices, so their task summaries cannot be combined safely."
    groups = []
    for task_index, infos in sorted(by_task.items()):
        rewards = [info[REWARD_KEY_NAME] for info in infos]
        groups.append(
            {
                TASK_INDEX_KEY_NAME: task_index,
                f"{Stat.MEAN.prefix}{REWARD_KEY_NAME}": fmean(rewards),
                f"{Stat.MIN.prefix}{REWARD_KEY_NAME}": min(rewards),
                f"{Stat.MAX.prefix}{REWARD_KEY_NAME}": max(rewards),
                ROLLOUT_INFOS_KEY_NAME: infos,
            }
        )
    return groups, None


def pool_loaded_runs(runs: Sequence[LoadedRun], *, source_labels: Sequence[str]) -> LoadedRun:
    """Concatenate complete repeat samples and recompute point estimates and 95% intervals.

    Each repeat has equal weight, even when sources have different repeat counts. Metrics
    available only as whole-run aggregates (including pass@k summaries) are omitted. A metric
    other than token accounting must be finite in every scored row of every source.
    Token metrics use the observations that record them, including accounting-only rows
    from unscored repeats, without inventing a zero quality or token estimate.
    """
    if not runs or len(runs) != len(source_labels):
        raise ValueError("Pooling requires one source label per nonempty run collection.")
    rows_by_source = [_validated_rows(run, label) for run, label in zip(runs, source_labels, strict=True)]
    token_names = set(COMPLETION_TOKEN_METRIC_NAMES)
    quality_rows = [
        _quality_rows(run, rows, label) for run, rows, label in zip(runs, rows_by_source, source_labels, strict=True)
    ]
    metric_names = sorted(
        {
            name
            for rows in rows_by_source
            for row in rows
            for name in row
            if name != AGENT_REF_KEY_NAME and is_primary_metric(name)
        }
    )
    if not metric_names:
        raise ConfigError("Cannot pool runs: their repeat rows contain no primary metrics.")
    quality_names = set(metric_names) - token_names
    if not quality_names and any(
        is_primary_metric(name) and name not in token_names
        for run in runs
        for name in (*run.agent_metrics, *run.key_metrics)
    ):
        raise ConfigError(
            "Cannot pool runs: no stored quality repeat-level estimates are available; "
            "token accounting cannot replace them. Collect at least two scored repeats per source "
            "and regenerate aggregate metrics."
        )
    if quality_names:
        for label, rows in zip(source_labels, quality_rows, strict=True):
            if not rows:
                raise ConfigError(
                    f"Cannot pool '{label}': no stored quality repeat-level estimates are available. "
                    "Every constituent must provide scored repeat observations."
                )

    metrics: Dict[str, float] = {}
    accounting_notes = []
    for name in metric_names:
        observations = rows_by_source if name in token_names else quality_rows
        values, missing_notes = _pooled_metric_samples(name, observations, source_labels)
        accounting_notes.extend(missing_notes)
        mean = fmean(values)
        metrics[name] = metrics[f"{Stat.MEAN.across_repeats_prefix}{name}"] = mean
        if len(values) > 1:
            se = stdev(values) / sqrt(len(values))
            margin = float(t.isf(0.025, len(values) - 1)) * se
            metrics[f"{Stat.SE.across_repeats_prefix}{name}"] = se
            metrics[f"{Stat.CI_LOW_95.across_repeats_prefix}{name}"] = mean - margin
            metrics[f"{Stat.CI_HIGH_95.across_repeats_prefix}{name}"] = mean + margin

    rows = [row for source_rows in rows_by_source for row in source_rows]
    rows = [{**row, ROLLOUT_INDEX_KEY_NAME: index} for index, row in enumerate(rows)]
    groups, flip_reason = _pooled_groups(runs)
    task_sets = [
        {
            group[TASK_INDEX_KEY_NAME]
            for group in run.group_level_metrics
            if isinstance(group, dict)
            and isinstance(group.get(TASK_INDEX_KEY_NAME), int)
            and not isinstance(group[TASK_INDEX_KEY_NAME], bool)
            and group[TASK_INDEX_KEY_NAME] >= 0
        }
        for run in runs
    ]
    task_indices = set().union(*task_sets)
    known_task_sets = [tasks for tasks in task_sets if tasks]
    omitted = sorted(
        {
            name
            for run in runs
            for name in (*run.agent_metrics, *run.key_metrics)
            if is_primary_metric(name) and name not in metric_names
        }
    )
    sample_rows_by_source = quality_rows if quality_names else rows_by_source
    sample_count = sum(map(len, sample_rows_by_source))
    notes = [
        f"Pooled {len(runs)} independent sources ({', '.join(str(len(rows)) for rows in sample_rows_by_source)} "
        f"repeats; {sample_count} total). Each repeat is an independent, equally weighted observation; "
        "local repeat indices in different sources are independent."
    ]
    notes.extend(accounting_notes)
    if any(run.agent_metrics.get("coverage/masked_rollouts", 0) for run in runs) or any(
        len(scored) != len(recorded) for scored, recorded in zip(quality_rows, rows_by_source, strict=True)
    ):
        notes.append(
            "Unscored repeats and masked task observations are excluded from quality estimates; "
            "available token accounting is retained. Quality confidence intervals describe the scored observations "
            "and may be affected by incomplete task coverage."
        )
    if quality_names:
        for run, label, source_rows in zip(runs, source_labels, rows_by_source, strict=True):
            skipped = sorted(row[ROLLOUT_INDEX_KEY_NAME] for row in source_rows if is_token_accounting_only(row))
            if skipped:
                notes.append(
                    f"Source '{label}' for agent '{run.agent_name}': local repeats {skipped} contain only "
                    "token accounting and are excluded from quality estimates."
                )
    if omitted:
        notes.append(
            "Whole-run metrics without complete repeat-level samples were omitted from the pooled side: "
            + ", ".join(f"`{name}`" for name in omitted)
            + "."
        )
    if any(tasks != known_task_sets[0] for tasks in known_task_sets[1:]):
        notes.append(
            "Pooled sources cover different task indices; repeat confidence intervals may mix "
            "task-difficulty differences with repeat variance."
        )
    return LoadedRun(
        agent_name=", ".join(dict.fromkeys(run.agent_name for run in runs)),
        agent_metrics=metrics,
        key_metrics={name: metrics[name] for run in runs for name in run.key_metrics if name in metrics},
        group_level_metrics=groups,
        repeat_level_metrics=rows,
        num_tasks=len(task_indices) if task_indices else max(run.num_tasks for run in runs),
        num_repeats=sample_count,
        has_repeat_cis=len(rows) > 1,
        pooled=True,
        notes=tuple(notes),
        flip_unavailable_reason=flip_reason,
    )
