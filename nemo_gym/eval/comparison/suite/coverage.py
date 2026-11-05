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
"""Validate recorded and planned task coverage for suite comparisons."""

from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.eval.comparison.suite.loading import LoadedSuiteRun
from nemo_gym.eval.comparison.suite.schema import SuiteCoverage


def _task_sets(run: LoadedSuiteRun) -> tuple[dict[int, frozenset[int]], list[str], bool]:
    """Read identities already recorded by aggregation; never derive repeat scores."""
    tasks: dict[int, set[int]] = {}
    seen_groups = set()
    notes = []
    has_all_infos = bool(run.groups)
    for group in run.groups:
        task = group.get(TASK_INDEX_KEY_NAME)
        if not isinstance(task, int) or isinstance(task, bool) or task < 0:
            raise ConfigError(f"'{run.aggregate_metrics}' has an invalid group task index {task!r}.")
        if task in seen_groups:
            raise ConfigError(f"'{run.aggregate_metrics}' has duplicate task group {task}.")
        seen_groups.add(task)
        missing = group.get("missing_num_rollouts", 0)
        if isinstance(missing, (int, float)) and missing > 0:
            notes.append(f"Task {task} is missing planned rollouts.")
        infos = group.get("rollout_infos")
        if not isinstance(infos, list) or not infos:
            has_all_infos = False
            continue
        for info in infos:
            if not isinstance(info, dict):
                raise ConfigError(f"'{run.aggregate_metrics}' has a non-object rollout summary for task {task}.")
            if info.get(TASK_INDEX_KEY_NAME, task) != task:
                raise ConfigError(f"'{run.aggregate_metrics}' rollout task identity disagrees with group {task}.")
            repeat = info.get(ROLLOUT_INDEX_KEY_NAME)
            if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 0:
                raise ConfigError(f"'{run.aggregate_metrics}' task {task} has invalid rollout index {repeat!r}.")
            target = tasks.setdefault(repeat, set())
            if task in target:
                raise ConfigError(f"'{run.aggregate_metrics}' duplicates task {task}, repeat {repeat}.")
            target.add(task)
        expected = group.get("expected_num_rollouts")
        if isinstance(expected, int) and expected > len(infos):
            notes.append(f"Task {task} is missing planned rollouts.")
    return {repeat: frozenset(indices) for repeat, indices in tasks.items()}, notes, has_all_infos


def _task_indices(tasks: set[int]) -> str:
    """Keep task indices useful for locating missing inputs without overwhelming reports."""
    preview = ", ".join(str(task) for task in sorted(tasks)[:10])
    return f"{preview}, ... (+{len(tasks) - 10} more)" if len(tasks) > 10 else preview


def _coverage(run: LoadedSuiteRun) -> SuiteCoverage:
    recorded, notes, complete_identities = _task_sets(run)
    incomplete = bool(notes)
    if run.unscored_repeat_indices:
        incomplete = True
        indices = ", ".join(map(str, run.unscored_repeat_indices))
        notes.append(
            f"Repeats with no scored task observations are excluded from score calculations (indices: {indices})."
        )
    for metric in ("coverage/masked_rollouts", "coverage/fully_masked_tasks"):
        count = run.agent_metrics.get(metric, 0)
        if isinstance(count, (int, float)) and count > 0:
            incomplete = True
            notes.append(f"Aggregate metrics report {metric}={count}; scored coverage is incomplete.")
    reported_repeats = run.agent_metrics.get("num_repeats")
    if isinstance(reported_repeats, int) and reported_repeats != run.repeat_count:
        incomplete = True
        notes.append("Reported repeat count differs from the number of scored repeats available for comparison.")
    sample_counts = {row["sample_count"] for row in run.repeat_rows if isinstance(row.get("sample_count"), int)}
    if len(sample_counts) > 1:
        incomplete = True
        notes.append("Stored repeat metrics report unequal task sample counts.")
    if recorded and len(set(recorded.values())) > 1:
        incomplete = True
        notes.append("Recorded task sets differ between repeats.")
    if complete_identities and run.repeat_rows:
        recorded_indices = set(recorded)
        metric_indices = {row[ROLLOUT_INDEX_KEY_NAME] for row in run.repeat_rows}
        if recorded_indices != metric_indices:
            incomplete = True
            notes.append("Stored repeat metric indices differ from recorded rollout indices.")
    for row in run.repeat_rows:
        missing = row.get("missing_count", 0)
        count = row.get("sample_count")
        if isinstance(missing, (int, float)) and missing > 0:
            incomplete = True
            notes.append(f"Repeat {row[ROLLOUT_INDEX_KEY_NAME]} reports missing task observations.")
        expected_count = len(recorded.get(row[ROLLOUT_INDEX_KEY_NAME], ())) if complete_identities else len(run.groups)
        if isinstance(count, int) and expected_count and count != expected_count:
            incomplete = True
            notes.append(f"Repeat {row[ROLLOUT_INDEX_KEY_NAME]} sample_count disagrees with recorded task identities.")
    planned_count = None
    verified = False
    if run.planned_tasks_by_repeat is not None:
        planned = run.planned_tasks_by_repeat
        planned_tasks = set().union(*planned.values())
        planned_count = len(planned_tasks)
        group_tasks = {group[TASK_INDEX_KEY_NAME] for group in run.groups}
        if run.groups and group_tasks != planned_tasks:
            incomplete = True
            missing_tasks = planned_tasks - group_tasks
            unplanned_tasks = group_tasks - planned_tasks
            if missing_tasks:
                notes.append(
                    f"Results are missing {len(missing_tasks)} of {planned_count} planned tasks "
                    f"(task indices: {_task_indices(missing_tasks)})."
                )
            if unplanned_tasks:
                noun = "task" if len(unplanned_tasks) == 1 else "tasks"
                notes.append(
                    f"Results include {len(unplanned_tasks)} unplanned {noun} "
                    f"(task indices: {_task_indices(unplanned_tasks)})."
                )
        if len(set(planned.values())) > 1:
            incomplete = True
            notes.append("Planned task sets differ between repeats.")
        if complete_identities:
            if recorded != planned:
                incomplete = True
                missing_rollouts = sum(
                    len(tasks - recorded.get(repeat, frozenset())) for repeat, tasks in planned.items()
                )
                unplanned_rollouts = sum(
                    len(tasks - planned.get(repeat, frozenset())) for repeat, tasks in recorded.items()
                )
                if missing_rollouts:
                    notes.append(
                        f"Results are missing {missing_rollouts} of {sum(map(len, planned.values()))} planned rollouts."
                    )
                if unplanned_rollouts:
                    noun = "rollout" if unplanned_rollouts == 1 else "rollouts"
                    notes.append(f"Results include {unplanned_rollouts} unplanned {noun}.")
            else:
                verified = True
        else:
            notes.append("Planned inputs exist, but recorded task identities are insufficient to verify coverage.")
    else:
        notes.append("Planned coverage is unverified: materialized inputs are unavailable.")
    if not complete_identities:
        notes.append("Recorded repeat task coverage is unverified: group rollout identities are unavailable.")
    status = "incomplete" if incomplete else "verified" if verified else "unverified"
    return SuiteCoverage(
        status=status,
        recorded_task_count=len(run.groups),
        planned_task_count=planned_count,
        notes=list(dict.fromkeys(notes)),
    )


def _check_paired_coverage(
    baseline: LoadedSuiteRun,
    candidate: LoadedSuiteRun,
    baseline_coverage: SuiteCoverage,
    candidate_coverage: SuiteCoverage,
) -> None:
    # Group identities remain evidence even when per-repeat rollout identities are absent.
    baseline_set = {group[TASK_INDEX_KEY_NAME] for group in baseline.groups} if baseline.groups else None
    candidate_set = {group[TASK_INDEX_KEY_NAME] for group in candidate.groups} if candidate.groups else None
    reasons = []
    if baseline_set is not None and candidate_set is not None and baseline_set != candidate_set:
        reasons.append(
            "Candidate and baseline have results for different tasks "
            f"(candidate: {len(candidate_set)} tasks; baseline: {len(baseline_set)} tasks)."
        )
    if baseline.planned_tasks_by_repeat is not None and candidate.planned_tasks_by_repeat is not None:
        planned_baseline = set().union(*baseline.planned_tasks_by_repeat.values())
        planned_candidate = set().union(*candidate.planned_tasks_by_repeat.values())
        if planned_baseline != planned_candidate:
            reasons.append("Baseline and candidate planned task sets differ.")
    for coverage in (baseline_coverage, candidate_coverage):
        if reasons:
            coverage.status = "incomplete"
            coverage.notes.extend(reasons)


def check_collection_coverage(
    baseline: tuple[LoadedSuiteRun, ...],
    candidate: tuple[LoadedSuiteRun, ...],
    baseline_coverage: list[SuiteCoverage],
    candidate_coverage: list[SuiteCoverage],
) -> None:
    """Keep the fixed task set identical across every constituent source."""
    if len(baseline) == len(candidate) == 1:
        _check_paired_coverage(baseline[0], candidate[0], baseline_coverage[0], candidate_coverage[0])
        return
    runs = (*baseline, *candidate)
    recorded = {frozenset(group[TASK_INDEX_KEY_NAME] for group in run.groups) for run in runs if run.groups}
    planned = {
        frozenset().union(*run.planned_tasks_by_repeat.values())
        for run in runs
        if run.planned_tasks_by_repeat is not None
    }
    reasons = []
    if len(recorded) > 1:
        reasons.append("Pooled source runs have results for different tasks.")
    if len(planned) > 1:
        reasons.append("Pooled source runs have different planned task sets.")
    if not reasons and len(recorded | planned) > 1:
        reasons.append("Recorded and planned task sets differ across pooled source runs.")
    for coverage in (*baseline_coverage, *candidate_coverage):
        if reasons:
            coverage.status = "incomplete"
            coverage.notes.extend(reasons)
