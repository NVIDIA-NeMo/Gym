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
"""Report independent repeats sampled up to the first verified success."""

from collections import Counter, defaultdict
from collections.abc import Iterable

from nemo_gym.config.global_config import (
    AGENT_REF_KEY_NAME,
    ENVIRONMENT_SERVER_STAMP_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    label_runs,
)
from nemo_gym.rollouts.repeat_stopping import STOP_ON_SUCCESS_REASON, repeat_group, verified_reward
from nemo_gym.rollouts.rollout_records import RUN_ID_KEY, logical_rollout_id
from nemo_gym.rollouts.rollout_store import RolloutStore


def adaptive_repeat_entries(runs: Iterable[tuple[RolloutStore, list[dict]]], threshold: float) -> list[dict]:
    """Use the selected observations and the full plan to account for every task, including unfinished ones."""
    reports = defaultdict(list)
    agent_names = {}
    environment_names = {}
    totals = defaultdict(Counter)
    for store, successes in runs:
        outcomes = {
            logical_rollout_id(row): row
            for rows in (successes, store.selected("failure"), store.selected("omitted"))
            for row in rows
        }
        groups = defaultdict(list)
        for row in store.planned_inputs():
            task, route = repeat_group(row)
            groups[route, task].append(row)
            agent_names[route] = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
            observed = outcomes.get(logical_rollout_id(row), {})
            environment = row.get(ENVIRONMENT_SERVER_STAMP_KEY_NAME) or observed.get(ENVIRONMENT_SERVER_STAMP_KEY_NAME)
            if environment is not None:
                environment_names[route] = environment
        for (route, task), rows in sorted(groups.items()):
            counts = Counter(
                planned_rollouts=len(rows),
                performed_rollouts=0,
                skipped_after_success=0,
                failed_rollouts=0,
                unscored_rollouts=0,
                unobserved_rollouts=0,
            )
            first_success = None
            for ordinal, row in enumerate(sorted(rows, key=lambda r: r[ROLLOUT_INDEX_KEY_NAME]), start=1):
                outcome = outcomes.get(logical_rollout_id(row))
                if outcome is None:
                    counts["unobserved_rollouts"] += 1
                    continue
                if outcome.get("_ng_omitted"):
                    if outcome.get("_ng_failure_message") == STOP_ON_SUCCESS_REASON:
                        counts["skipped_after_success"] += 1
                    else:
                        counts["unobserved_rollouts"] += 1
                    continue
                counts["performed_rollouts"] += 1
                if outcome.get("_ng_failure_class") is not None:
                    counts["failed_rollouts"] += 1
                elif (reward := verified_reward(outcome)) is None:
                    counts["unscored_rollouts"] += 1
                elif reward >= threshold and first_success is None:
                    first_success = ordinal
            if first_success is not None:
                status = "solved"
            elif counts["performed_rollouts"] == len(rows) and not (
                counts["failed_rollouts"] or counts["unscored_rollouts"]
            ):
                status = "budget_exhausted"
            else:
                status = "unresolved"
            reports[route].append(
                {
                    RUN_ID_KEY: store.manifest.run_id,
                    TASK_INDEX_KEY_NAME: task,
                    "status": status,
                    "first_successful_repeat": first_success,
                    **counts,
                }
            )
            totals[route].update(counts)
            totals[route][f"{status}_tasks"] += 1

    labels = label_runs(agent_names)
    entries = []
    for route, groups in reports.items():
        counts = totals[route]
        for status in ("solved", "budget_exhausted", "unresolved"):
            counts.setdefault(f"{status}_tasks", 0)
        counts["planned_tasks"] = len(groups)
        metrics = {f"adaptive_repeat/{name}": value for name, value in counts.items()}
        key_metrics = {}
        if not counts["unresolved_tasks"]:
            key_metrics["adaptive_repeat/success_within_budget"] = counts["solved_tasks"] / len(groups)
        entries.append(
            {
                AGENT_REF_KEY_NAME: {"name": labels[route]},
                **(
                    {ENVIRONMENT_SERVER_STAMP_KEY_NAME: environment_names[route]} if route in environment_names else {}
                ),
                "sampling_policy": {"metric": "reward", "stop_on_success_threshold": threshold},
                "agent_metrics": metrics | key_metrics,
                "key_metrics": key_metrics,
                "group_level_metrics": groups,
                "repeat_level_metrics": [],
            }
        )
    return sorted(entries, key=lambda entry: entry[AGENT_REF_KEY_NAME]["name"])
