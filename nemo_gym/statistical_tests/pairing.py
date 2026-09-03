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
"""Per-task paired deltas for an arbitrary metric, and which metrics to test by default.

Reads `group_level_metrics` off `nemo_gym.comparison.loading.LoadedRun` -- the same read-only
loading layer `gym eval compare` uses -- but is otherwise independent of `nemo_gym.comparison`:
it does not import or modify `diff.py`, `schema.py`, or `report.py`.
"""

from typing import Any, Dict, List, Optional, Tuple

from nemo_gym.comparison.loading import LoadedRun
from nemo_gym.global_config import MEAN_PREFIX, TASK_INDEX_KEY_NAME


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _groups_by_task(run: LoadedRun) -> Dict[int, Dict[str, Any]]:
    return {group[TASK_INDEX_KEY_NAME]: group for group in run.group_level_metrics if TASK_INDEX_KEY_NAME in group}


def paired_task_deltas(baseline: LoadedRun, candidate: LoadedRun, metric: str) -> Optional[List[float]]:
    """`candidate - baseline` for every common task that has `mean/<metric>` on both sides.

    Deliberately the *full* set (every common task, not capped or filtered to "changed" tasks the
    way `compare`'s display-oriented flip list is) -- a statistical test needs the whole sample,
    ties included, or it silently biases/shrinks toward a subset. Returns `None` when no common
    task has the metric on both sides, so the caller can skip it rather than test on nothing.
    """
    key = f"{MEAN_PREFIX}{metric}"
    baseline_groups = _groups_by_task(baseline)
    candidate_groups = _groups_by_task(candidate)
    common = sorted(set(baseline_groups) & set(candidate_groups))

    deltas: List[float] = []
    for task_index in common:
        baseline_value = baseline_groups[task_index].get(key)
        candidate_value = candidate_groups[task_index].get(key)
        if _is_number(baseline_value) and _is_number(candidate_value):
            deltas.append(float(candidate_value) - float(baseline_value))
    return deltas or None


def resolve_metrics(
    baseline: LoadedRun, candidate: LoadedRun, requested: Optional[List[str]]
) -> Tuple[List[str], List[str]]:
    """`(metrics_to_test, skipped)`.

    `requested` (from `--metric`) is returned as-is, deduplicated -- an explicitly named metric
    with no pairing data is the caller's problem to raise on, not silently drop. Left unset, every
    key-metric name (from either side) is a candidate, narrowed to the ones `paired_task_deltas`
    can actually produce data for; the rest come back as `skipped` so the caller can note them.
    """
    if requested:
        return list(dict.fromkeys(requested)), []

    key_metric_names = set()
    for run in (baseline, candidate):
        key_metric_names.update(run.key_metrics)

    resolved: List[str] = []
    skipped: List[str] = []
    for name in sorted(key_metric_names):
        if not name.startswith(MEAN_PREFIX):
            # Not a `mean/<field>` metric (e.g. a `pass@k` aggregate) -- there is no per-task
            # value to pair on, so it can never be tested, not just "no data this time".
            skipped.append(name)
            continue
        metric = name[len(MEAN_PREFIX) :]
        if paired_task_deltas(baseline, candidate, metric):
            resolved.append(metric)
        else:
            skipped.append(metric)
    return resolved, skipped
