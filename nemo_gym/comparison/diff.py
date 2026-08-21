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
"""Diffing two loaded runs: metric rows and per-task sample flips.

Pure computation -- no filesystem access, no statistics. Confidence intervals are read verbatim
from what the runs already recorded; nothing here estimates, tests, or judges.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from nemo_gym.comparison.loading import (
    CI_HIGH_PREFIX,
    CI_LOW_PREFIX,
    MEAN_ACROSS_REPEATS_PREFIX,
    SE_PREFIX,
    STD_ERR_ACROSS_RUNS_SUFFIX,
    LoadedRun,
)
from nemo_gym.comparison.schema import (
    AgentComparison,
    CandidateMetricValue,
    FlipSummary,
    MetricRow,
    MetricValue,
    TaskFlip,
)
from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME


# Dispersion companions of a `mean/<field>` metric. They are summary statistics of the same
# underlying field, not metrics in their own right, so they never get their own row.
_DISPERSION_PREFIXES = (
    "median/",
    "std/",
    "min/",
    "max/",
    "p25/",
    "p75/",
    "sem/",
    "ci_low_95/",
    "ci_high_95/",
)
# The cross-repeat family is consumed as CI columns, not as rows.
_ACROSS_REPEATS_MARKER = "_across_repeats/"
_STAT_SUFFIXES = ("/std_dev_across_runs", STD_ERR_ACROSS_RUNS_SUFFIX)

# The per-task field flips are computed from. Every verify response carries `reward` at minimum.
FLIP_FIELD = "reward"
_TASK_MEAN_KEY = f"mean/{FLIP_FIELD}"
_TASK_MIN_KEY = f"min/{FLIP_FIELD}"
_TASK_MAX_KEY = f"max/{FLIP_FIELD}"
# A task "passes" when the majority of its repeats scored a pass.
_PASS_THRESHOLD = 0.5


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric(value: Any) -> Optional[float]:
    return float(value) if _is_number(value) else None


def is_comparable_metric(name: str) -> bool:
    """Whether an `agent_metrics` key earns its own row in the all-metrics table."""
    if name.startswith(_DISPERSION_PREFIXES):
        return False
    if _ACROSS_REPEATS_MARKER in name:
        return False
    return not name.endswith(_STAT_SUFFIXES)


def _ordered_metric_names(baseline: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> List[str]:
    """Baseline order first (it is the reference), then anything only the candidates reported."""
    names = [name for name in baseline if _is_number(baseline[name])]
    seen = set(names)
    for metrics in candidates:
        for name in metrics:
            if name not in seen and _is_number(metrics[name]):
                names.append(name)
                seen.add(name)
    return names


def _metric_value(metrics: Dict[str, Any], name: str) -> Optional[MetricValue]:
    value = _numeric(metrics.get(name))
    if value is None:
        return None
    return MetricValue(
        value=value,
        ci_low=_numeric(metrics.get(f"{CI_LOW_PREFIX}{name}")),
        ci_high=_numeric(metrics.get(f"{CI_HIGH_PREFIX}{name}")),
        se_across_repeats=_numeric(metrics.get(f"{SE_PREFIX}{name}")),
        mean_across_repeats=_numeric(metrics.get(f"{MEAN_ACROSS_REPEATS_PREFIX}{name}")),
        std_err_across_runs=_numeric(metrics.get(f"{name}{STD_ERR_ACROSS_RUNS_SUFFIX}")),
    )


def _candidate_metric_value(
    metrics: Dict[str, Any], name: str, baseline_value: Optional[float]
) -> Optional[CandidateMetricValue]:
    base = _metric_value(metrics, name)
    if base is None:
        return None
    delta = None if baseline_value is None else base.value - baseline_value
    delta_pct = None if delta is None or not baseline_value else delta / abs(baseline_value) * 100.0
    return CandidateMetricValue(**base.model_dump(), delta=delta, delta_pct=delta_pct)


def build_metric_rows(baseline: LoadedRun, candidates: Sequence[LoadedRun]) -> List[MetricRow]:
    """One row per metric reported by any side, key metrics flagged."""
    candidate_metrics = [run.agent_metrics for run in candidates]
    key_metric_names = set(baseline.key_metrics) | {name for run in candidates for name in run.key_metrics}

    rows: List[MetricRow] = []
    for name in _ordered_metric_names(baseline.agent_metrics, candidate_metrics):
        if not is_comparable_metric(name):
            continue
        baseline_value = _metric_value(baseline.agent_metrics, name)
        candidate_values = [
            _candidate_metric_value(metrics, name, baseline_value.value if baseline_value else None)
            for metrics in candidate_metrics
        ]
        present_in = ["baseline"] if baseline_value else []
        present_in += [f"candidate[{i}]" for i, value in enumerate(candidate_values) if value is not None]
        rows.append(
            MetricRow(
                metric=name,
                is_key_metric=name in key_metric_names,
                present_in=present_in,
                baseline=baseline_value,
                candidates=candidate_values,
            )
        )
    return rows


def _groups_by_task(run: LoadedRun) -> Dict[int, Dict[str, Any]]:
    return {group[TASK_INDEX_KEY_NAME]: group for group in run.group_level_metrics if TASK_INDEX_KEY_NAME in group}


def _per_repeat_rewards(group: Dict[str, Any]) -> Optional[List[float]]:
    rollout_infos = group.get("rollout_infos")
    if not isinstance(rollout_infos, list) or not rollout_infos:
        return None
    ordered = sorted(rollout_infos, key=lambda info: info.get(ROLLOUT_INDEX_KEY_NAME, 0))
    rewards = [_numeric(info.get(FLIP_FIELD)) for info in ordered]
    return None if any(reward is None for reward in rewards) else [reward for reward in rewards if reward is not None]


def _looks_binary(
    common: Sequence[int], baseline_groups: Dict[int, Dict[str, Any]], candidate_groups: Dict[int, Dict[str, Any]]
) -> bool:
    """Whether every observed reward on both sides is exactly 0 or 1.

    Uses each task's recorded min/max where available so a task whose repeats disagree (mean 0.5)
    is still recognised as binary.
    """
    for task_index in common:
        for groups in (baseline_groups, candidate_groups):
            group = groups[task_index]
            observed = [group.get(_TASK_MIN_KEY), group.get(_TASK_MAX_KEY)]
            if all(value is None for value in observed):
                observed = [group.get(_TASK_MEAN_KEY)]
            for value in observed:
                number = _numeric(value)
                if number is None or number not in (0.0, 1.0):
                    return False
    return True


def _flip_direction(baseline_score: float, candidate_score: float) -> Optional[str]:
    baseline_passed = baseline_score > _PASS_THRESHOLD
    candidate_passed = candidate_score > _PASS_THRESHOLD
    if baseline_score == _PASS_THRESHOLD or candidate_score == _PASS_THRESHOLD:
        return None
    if baseline_passed and not candidate_passed:
        return "pass_to_fail"
    if candidate_passed and not baseline_passed:
        return "fail_to_pass"
    return None


def build_flip_summary(baseline: LoadedRun, candidate: LoadedRun, *, candidate_index: int = 0) -> FlipSummary:
    """Per-task movement between the two runs, joined on task index.

    `*_aggregate_metrics.json` carries no task identity, so tasks are matched by
    `_ng_task_index` alone -- which assumes both runs used the same dataset, split, limit and
    ordering.
    """
    baseline_groups = _groups_by_task(baseline)
    candidate_groups = _groups_by_task(candidate)
    common = sorted(set(baseline_groups) & set(candidate_groups))
    counts = {
        "common_task_count": len(common),
        "baseline_only_task_count": len(set(baseline_groups) - set(candidate_groups)),
        "candidate_only_task_count": len(set(candidate_groups) - set(baseline_groups)),
    }

    if not baseline_groups or not candidate_groups:
        return FlipSummary(
            candidate_index=candidate_index,
            mode="unavailable",
            reason="one or both runs recorded no per-task metrics.",
            **counts,
        )
    if not common:
        return FlipSummary(
            candidate_index=candidate_index,
            mode="unavailable",
            reason=(
                f"no overlapping task indices (baseline has {len(baseline_groups)}, "
                f"candidate has {len(candidate_groups)})."
            ),
            **counts,
        )

    scored: List[Tuple[int, float, float]] = []
    for task_index in common:
        baseline_score = _numeric(baseline_groups[task_index].get(_TASK_MEAN_KEY))
        candidate_score = _numeric(candidate_groups[task_index].get(_TASK_MEAN_KEY))
        if baseline_score is not None and candidate_score is not None:
            scored.append((task_index, baseline_score, candidate_score))

    if not scored:
        return FlipSummary(
            candidate_index=candidate_index,
            mode="unavailable",
            reason=f"no `{_TASK_MEAN_KEY}` recorded for the overlapping tasks.",
            **counts,
        )

    def flip(task_index: int, baseline_score: float, candidate_score: float, direction: str) -> TaskFlip:
        return TaskFlip(
            task_index=task_index,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            delta=candidate_score - baseline_score,
            direction=direction,  # type: ignore[arg-type]
            baseline_rewards=_per_repeat_rewards(baseline_groups[task_index]),
            candidate_rewards=_per_repeat_rewards(candidate_groups[task_index]),
        )

    if not _looks_binary([task_index for task_index, _, _ in scored], baseline_groups, candidate_groups):
        movers = [
            flip(task_index, baseline_score, candidate_score, "changed")
            for task_index, baseline_score, candidate_score in scored
            if candidate_score != baseline_score
        ]
        movers.sort(key=lambda item: (-abs(item.delta), item.task_index))
        return FlipSummary(
            candidate_index=candidate_index,
            mode="continuous",
            unchanged_count=len(scored) - len(movers),
            flips=movers,
            **counts,
        )

    flips: List[TaskFlip] = []
    tied = unchanged = 0
    for task_index, baseline_score, candidate_score in scored:
        direction = _flip_direction(baseline_score, candidate_score)
        if direction is None:
            if _PASS_THRESHOLD in (baseline_score, candidate_score):
                tied += 1
            else:
                unchanged += 1
            continue
        flips.append(flip(task_index, baseline_score, candidate_score, direction))

    # Regressions first, then by how far the task moved.
    flips.sort(key=lambda item: (item.direction != "pass_to_fail", -abs(item.delta), item.task_index))
    pass_to_fail = sum(1 for item in flips if item.direction == "pass_to_fail")
    fail_to_pass = len(flips) - pass_to_fail
    return FlipSummary(
        candidate_index=candidate_index,
        mode="binary",
        pass_to_fail_count=pass_to_fail,
        fail_to_pass_count=fail_to_pass,
        tied_count=tied,
        unchanged_count=unchanged,
        net=fail_to_pass - pass_to_fail,
        flips=flips,
        **counts,
    )


def compare_runs(baseline: LoadedRun, candidates: Sequence[LoadedRun]) -> AgentComparison:
    """Build one agent's comparison block: metric rows, flips, and anything worth flagging."""
    rows = build_metric_rows(baseline, candidates)
    if not any(row.baseline is not None and any(row.candidates) for row in rows):
        baseline_only = sorted(row.metric for row in rows if row.baseline is not None)[:5]
        candidate_only = sorted(row.metric for row in rows if row.baseline is None)[:5]
        raise ConfigError(
            "Baseline and candidate share no metric keys, so there is nothing to compare.\n"
            f"  baseline-only (first 5): {', '.join(baseline_only) or 'none'}\n"
            f"  candidate-only (first 5): {', '.join(candidate_only) or 'none'}\n"
            "Runs collected with different --num-repeats produce different pass@k metric names."
        )

    notes: List[str] = []
    repeat_counts = [run.num_repeats for run in candidates]
    if baseline.num_repeats is not None and any(
        count is not None and count != baseline.num_repeats for count in repeat_counts
    ):
        notes.append(
            f"Repeat counts differ (baseline {baseline.num_repeats}, "
            f"candidate {', '.join(str(count) for count in repeat_counts)}). Metric names that embed k "
            "(e.g. pass@1[avg-of-k]) will not line up across the runs."
        )
    if not baseline.has_repeat_cis and not any(run.has_repeat_cis for run in candidates):
        notes.append(
            "Neither run recorded cross-repeat confidence intervals, so every CI cell is empty. "
            "They are written for `mean/*` metrics when a run has 2 or more repeats."
        )
    one_sided = [row.metric for row in rows if len(row.present_in) < 1 + len(candidates)]
    if one_sided:
        notes.append(f"{len(one_sided)} metric(s) were reported by only one of the runs.")

    return AgentComparison(
        baseline_agent=baseline.agent_name,
        candidate_agents=[run.agent_name for run in candidates],
        baseline_task_count=baseline.num_tasks,
        candidate_task_counts=[run.num_tasks for run in candidates],
        baseline_repeat_count=baseline.num_repeats,
        candidate_repeat_counts=list(repeat_counts),
        metrics=rows,
        flips=[build_flip_summary(baseline, run, candidate_index=index) for index, run in enumerate(candidates)],
        notes=notes,
    )
