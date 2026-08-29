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
from __future__ import annotations

import math
import re
import statistics
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson
from pandas import DataFrame, Series, notna
from pandas.core.groupby.generic import DataFrameGroupBy
from pydantic import Field
from scipy import stats
from wandb import Histogram

from nemo_gym.config_types import AggregateMetrics, BaseNeMoGymCLIConfig
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)


class RewardProfileConfig(BaseNeMoGymCLIConfig):
    materialized_inputs_jsonl_fpath: str = Field(
        description="The file path of the materialized inputs as output by `gym eval run`."
    )
    rollouts_jsonl_fpath: str = Field(description="The file path of the rollouts as output by `gym eval run`.")
    allow_partial_rollouts: bool = Field(
        default=False,
        description="Allow reward profiling from partial rollout outputs by dropping input rows with no completed rollouts.",
    )


def _rollout_key(row: Dict[str, Any]) -> Tuple[int, int]:
    return row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]


class RewardProfiler:
    def _index_by_rollout_key(self, rows: List[Dict[str, Any]], name: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
        indexed: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for row in rows:
            key = _rollout_key(row)
            if key in indexed:
                raise ValueError(f"Duplicate {name} row for rollout key {key}")
            indexed[key] = row
        return indexed

    def align_rows_and_results(
        self,
        rows: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        allow_partial_rollouts: bool = False,
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        rows_by_key = self._index_by_rollout_key(rows, "materialized input")
        results_by_key = self._index_by_rollout_key(results, "result")

        missing_results = sorted(rows_by_key.keys() - results_by_key.keys())
        missing_rows = sorted(results_by_key.keys() - rows_by_key.keys())

        if missing_rows:
            raise ValueError(
                "Rollout results contain rows with no matching materialized input rows. "
                f"Found {len(missing_rows)} extra rollout rows. Extra rollout keys: {missing_rows[:10]}"
            )

        if missing_results and not allow_partial_rollouts:
            raise ValueError(
                "Materialized input rows and rollout results do not have matching rollout keys. "
                f"Missing rollout results for {len(missing_results)} materialized input rows. "
                f"Missing rollout keys: {missing_results[:10]}.\n\n"
                "Use ++allow_partial_rollouts=True to profile completed rollouts from a partial collection."
            )

        matched_keys = rows_by_key.keys() & results_by_key.keys()
        return [(rows_by_key[key], results_by_key[key]) for key in sorted(matched_keys)]

    def profile_completion_summary(
        self,
        rows: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rows_by_key = self._index_by_rollout_key(rows, "materialized input")
        results_by_key = self._index_by_rollout_key(results, "result")
        matched_keys = rows_by_key.keys() & results_by_key.keys()

        # Fan-out dispatches the same task to several agents, and those copies share a task index.
        # Count completion per (task, agent) so one agent's missing rollouts don't hide behind
        # another's completed ones. Without fan-out each task has one agent and this reduces to
        # the plain per-task count.
        def _group_of(key: Tuple[int, int]) -> Tuple[int, Optional[str]]:
            return key[0], (rows_by_key[key].get(AGENT_REF_KEY_NAME) or {}).get("name")

        expected_by_task = Counter(_group_of(key) for key in rows_by_key)
        completed_by_task = Counter(_group_of(key) for key in matched_keys)

        complete_input_rows = 0
        partial_input_rows = 0
        missing_input_rows = 0
        for task_idx, expected_count in expected_by_task.items():
            completed_count = completed_by_task[task_idx]
            if completed_count == expected_count:
                complete_input_rows += 1
            elif completed_count > 0:
                partial_input_rows += 1
            else:
                missing_input_rows += 1

        expected_rollout_rows = len(rows_by_key)
        completed_rollout_rows = len(matched_keys)
        completion_pct = (
            100.0 if expected_rollout_rows == 0 else 100.0 * completed_rollout_rows / expected_rollout_rows
        )

        return {
            "expected_rollout_rows": expected_rollout_rows,
            "completed_rollout_rows": completed_rollout_rows,
            "missing_rollout_rows": expected_rollout_rows - completed_rollout_rows,
            "extra_rollout_rows": len(results_by_key.keys() - rows_by_key.keys()),
            "reward_profile_completion_pct": completion_pct,
            "total_input_rows": len(expected_by_task),
            "complete_input_rows": complete_input_rows,
            "partial_input_rows": partial_input_rows,
            "missing_input_rows": missing_input_rows,
        }

    def rollout_info_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        task_idx, rollout_idx = _rollout_key(result)
        rollout_info: Dict[str, Any] = {
            "rollout_id": f"{task_idx}:{rollout_idx}",
            TASK_INDEX_KEY_NAME: task_idx,
            ROLLOUT_INDEX_KEY_NAME: rollout_idx,
        }

        if "reward" in result:
            rollout_info["reward"] = result.get("reward")

        usage = (result.get("response") or {}).get("usage") or {}
        for k, v in usage.items():
            if isinstance(v, bool):
                rollout_info[k] = int(v)
            elif isinstance(v, (int, float)):
                rollout_info[k] = v

        for k, v in result.items():
            if k in {TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, "reward", "response"}:
                continue
            if isinstance(v, bool):
                rollout_info[k] = int(v)
            elif isinstance(v, (int, float)):
                rollout_info[k] = v

        return rollout_info

    def histogram(self, data: Series) -> Optional[Histogram]:
        # W&B doesn't accept empty histograms
        data = data.dropna()
        if data.empty:
            return

        return Histogram(data)

    def describe_dataframe(self, df: DataFrame) -> DataFrame:
        stat_index = ["mean", "max", "min", "median", "std", "histogram"]
        d: List[Series] = [
            df.mean(),
            df.max(),
            df.min(),
            df.median(),
            df.std(),
            df.apply(self.histogram, axis=0),
        ]  # type: ignore

        # Std is nore interpretable using 0 rather than NaN for no std
        if d[4].isna().all():
            not_na_columns = df.columns[df.notna().all()]
            d[4][not_na_columns] = d[4][not_na_columns].fillna(0)

        # We use future_stack=True due to:
        # FutureWarning: The previous implementation of stack is deprecated and will be removed in a future version of pandas. See the What's New notes for pandas 2.1.0 for details. Specify future_stack=True to adopt the new implementation and silence this warning.
        # Critically here, we need to return a valid result for all rows even if one row is null
        # dropna must be unspecified with future_stack=True as the new implementation does not introduce rows of NA values. This argument will be removed in a future version of pandas.
        return DataFrame(d, index=stat_index).stack(future_stack=True)

    def calculate_metrics_single_df(self, grouped_df: DataFrameGroupBy) -> List[Dict[str, Any]]:
        grouped_metrics_df: DataFrame = grouped_df.apply(self.describe_dataframe, include_groups=False)
        grouped_metrics_df.columns = grouped_metrics_df.columns.map("/".join)
        grouped_metrics_df: DataFrame = grouped_metrics_df.reset_index()
        grouped_metrics = grouped_metrics_df.to_dict("records")

        # Filter for None in the result
        return [
            {k: v for k, v in group_metrics.items() if v is not None and notna(v)} for group_metrics in grouped_metrics
        ]

    def _confidence_interval(
        self, mean: float, sem: float, n: int, confidence: float = 0.95
    ) -> Optional[Tuple[float, float]]:
        """Return (ci_low, ci_high) t-interval at the given confidence level, or None when n <= 1."""
        if n <= 1:
            return None
        if sem == 0:
            warnings.warn(
                f"Standard error is 0 (all {n} values are identical) -- confidence interval "
                "collapses to a single point rather than being computed.",
                stacklevel=2,
            )
            return mean, mean
        ci_low, ci_high = stats.t.interval(confidence, df=n - 1, loc=mean, scale=sem)
        return float(ci_low), float(ci_high)

    def _compute_repeat_level_metrics(self, df: DataFrame) -> List[Dict[str, Any]]:
        """Per-agent, per-rollout-index summary stats across all tasks.

        Only produced for agents that have more than one rollout index — agents with a
        single rollout contribute nothing to a repeat-level comparison and are skipped.
        If no agent qualifies, returns an empty list.
        """
        numeric_cols = df.select_dtypes(include="number").columns.difference(
            [TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME]
        )

        rollout_counts_by_agent = df.groupby("agent_name")[ROLLOUT_INDEX_KEY_NAME].nunique()
        multi_rollout_agents = set(rollout_counts_by_agent[rollout_counts_by_agent > 1].index)
        if not multi_rollout_agents:
            return []

        df = df[df["agent_name"].isin(multi_rollout_agents)]
        total_tasks_by_agent = df.groupby("agent_name")[TASK_INDEX_KEY_NAME].nunique()

        repeat_metrics = []
        for (agent_name, rollout_idx), group in df.groupby(["agent_name", ROLLOUT_INDEX_KEY_NAME]):
            present_tasks = int(group[TASK_INDEX_KEY_NAME].nunique())
            total_tasks = int(total_tasks_by_agent[agent_name])
            entry: Dict[str, Any] = {
                AGENT_REF_KEY_NAME: {"name": agent_name},
                ROLLOUT_INDEX_KEY_NAME: int(rollout_idx),
                "sample_count": present_tasks,
                "missing_count": total_tasks - present_tasks,
            }
            for col in numeric_cols:
                col_data = group[col].dropna()
                n = len(col_data)
                if n == 0:
                    continue
                mean = float(col_data.mean())
                std = float(col_data.std(ddof=1)) if n > 1 else 0.0
                sem = std / n**0.5
                entry.update(
                    {
                        f"mean/{col}": mean,
                        f"median/{col}": float(col_data.median()),
                        f"std/{col}": std,
                        f"sem/{col}": sem,
                        f"min/{col}": float(col_data.min()),
                        f"max/{col}": float(col_data.max()),
                        f"p25/{col}": float(col_data.quantile(0.25)),
                        f"p75/{col}": float(col_data.quantile(0.75)),
                    }
                )
                if ci := self._confidence_interval(mean, sem, n):
                    entry[f"ci_low_95/{col}"], entry[f"ci_high_95/{col}"] = ci
            repeat_metrics.append(entry)

        incomplete_repeats = [entry for entry in repeat_metrics if entry["missing_count"] > 0]
        if incomplete_repeats:
            warnings.warn(
                f"{len(incomplete_repeats)} repeat(s) are missing rollouts for some tasks "
                f"(e.g. agent={incomplete_repeats[0][AGENT_REF_KEY_NAME]['name']!r}, "
                f"{ROLLOUT_INDEX_KEY_NAME}={incomplete_repeats[0][ROLLOUT_INDEX_KEY_NAME]}, "
                f"missing_count={incomplete_repeats[0]['missing_count']}). "
                "repeat_level_metrics statistics were computed from unequal sample sizes across "
                "repeats and may not be directly comparable, and agent_level_metrics may be biased "
                "toward whichever tasks happened to complete. Consider obtaining the missing rollouts "
                "before drawing conclusions from these metrics.",
                stacklevel=2,
            )

        return repeat_metrics

    def _aggregate_repeat_level_metrics(self, repeat_level_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate per-repeat estimates (e.g. mean/reward) across repeats, per agent.

        Treats each repeat's stat as one observation and reports the statistics across
        repeats.
        """
        if not repeat_level_metrics:
            return []

        df = DataFrame.from_records(repeat_level_metrics)
        df["agent_name"] = df[AGENT_REF_KEY_NAME].apply(lambda ref: ref["name"])
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c.startswith("mean/")]

        aggregated_metrics = []
        for agent_name, group in df.groupby("agent_name"):
            entry: Dict[str, Any] = {AGENT_REF_KEY_NAME: {"name": agent_name}}
            for col in numeric_cols:
                col_data = group[col].dropna()
                n = len(col_data)
                if n == 0:
                    continue
                mean = float(col_data.mean())
                std = float(col_data.std(ddof=1)) if n > 1 else 0.0
                se = std / n**0.5
                entry[f"mean_across_repeats/{col}"] = mean
                entry[f"median_across_repeats/{col}"] = float(col_data.median())
                entry[f"std_across_repeats/{col}"] = std
                entry[f"min_across_repeats/{col}"] = float(col_data.min())
                entry[f"max_across_repeats/{col}"] = float(col_data.max())
                entry[f"se_across_repeats/{col}"] = se
                if ci := self._confidence_interval(mean, se, n):
                    entry[f"ci_low_95_across_repeats/{col}"], entry[f"ci_high_95_across_repeats/{col}"] = ci
            aggregated_metrics.append(entry)
        return aggregated_metrics

    def _append_repeat_level_aggregates_to_agent_level_metrics(
        self, agent_level_metrics: List[Dict[str, Any]], repeat_level_metrics: List[Dict[str, Any]]
    ) -> None:
        repeat_level_aggregates_by_agent = {
            entry[AGENT_REF_KEY_NAME]["name"]: entry
            for entry in self._aggregate_repeat_level_metrics(repeat_level_metrics)
        }
        for agent_metrics in agent_level_metrics:
            aggregate = repeat_level_aggregates_by_agent.get(agent_metrics[AGENT_REF_KEY_NAME]["name"])
            if aggregate is not None:
                agent_metrics.update({k: v for k, v in aggregate.items() if k != AGENT_REF_KEY_NAME})

    def profile_from_data(
        self,
        rows: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        allow_partial_rollouts: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        aligned_rows_and_results = self.align_rows_and_results(
            rows, results, allow_partial_rollouts=allow_partial_rollouts
        )

        # Under fan-out the same task index appears once per agent. Profile those copies as
        # separate groups — one per (task, agent) — so each harness keeps its own reward/length
        # stats. Runs where every task has a single agent keep the plain per-task grouping and
        # byte-identical output.
        # Tolerate rows without an agent_ref exactly as before: only matched rows ever required
        # one, so the detection must not introduce a new failure on unmatched (partial) rows.
        def _agent_name(row: Dict[str, Any]) -> Optional[str]:
            return (row.get(AGENT_REF_KEY_NAME) or {}).get("name")

        per_agent = len({(row[TASK_INDEX_KEY_NAME], _agent_name(row)) for row in rows}) > len(
            {row[TASK_INDEX_KEY_NAME] for row in rows}
        )

        def _group_key(row: Dict[str, Any]) -> Any:
            if per_agent:
                return row[TASK_INDEX_KEY_NAME], _agent_name(row)
            return row[TASK_INDEX_KEY_NAME]

        filtered_results: List[Dict] = []
        task_idx_to_row: Dict[Any, Dict] = dict()
        task_idx_to_rollout_infos: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        expected_rollouts_by_task = Counter(_group_key(row) for row in rows)
        for row, result in aligned_rows_and_results:
            task_idx, rollout_idx = _rollout_key(row)
            task_idx_to_rollout_infos[_group_key(row)].append(self.rollout_info_from_result(result))

            # Add additional helpful information
            result = result | (result["response"].get("usage") or {})

            # agent_name is a temporary column used for aggregations below
            numeric_result = {
                "agent_name": row["agent_ref"]["name"],
                TASK_INDEX_KEY_NAME: task_idx,
                ROLLOUT_INDEX_KEY_NAME: rollout_idx,
            }
            for k, v in result.items():
                if isinstance(v, bool):
                    numeric_result[k] = int(v)
                elif isinstance(v, (int, float)):
                    numeric_result[k] = v

            filtered_results.append(numeric_result)
            task_idx_to_row.setdefault(_group_key(row), row)

        if not filtered_results:
            return [], [], []

        df = DataFrame.from_records(filtered_results)

        if per_agent:
            group_level_df = df.drop(columns=[ROLLOUT_INDEX_KEY_NAME]).groupby([TASK_INDEX_KEY_NAME, "agent_name"])
        else:
            group_level_df = df.drop(columns=[ROLLOUT_INDEX_KEY_NAME, "agent_name"]).groupby(TASK_INDEX_KEY_NAME)
        group_level_metrics = self.calculate_metrics_single_df(group_level_df)
        for group_metrics in group_level_metrics:
            if per_agent:
                group_metrics[AGENT_REF_KEY_NAME] = {"name": group_metrics.pop("agent_name")}
                group_key = (group_metrics[TASK_INDEX_KEY_NAME], group_metrics[AGENT_REF_KEY_NAME]["name"])
            else:
                group_key = group_metrics[TASK_INDEX_KEY_NAME]
            row = task_idx_to_row[group_key]

            row = row.copy()
            row.pop(TASK_INDEX_KEY_NAME)
            row.pop(ROLLOUT_INDEX_KEY_NAME)

            group_metrics["sample"] = row
            num_rollouts = len(task_idx_to_rollout_infos[group_key])
            expected_num_rollouts = expected_rollouts_by_task[group_key]
            group_metrics["num_rollouts"] = num_rollouts
            group_metrics["expected_num_rollouts"] = expected_num_rollouts
            group_metrics["missing_num_rollouts"] = expected_num_rollouts - num_rollouts
            group_metrics["reward_profile_completion_pct"] = (
                100.0 if expected_num_rollouts == 0 else 100.0 * num_rollouts / expected_num_rollouts
            )
            group_metrics["rollout_infos"] = sorted(
                task_idx_to_rollout_infos[group_key],
                key=lambda r: r[ROLLOUT_INDEX_KEY_NAME],
            )

        agent_level_df = df.drop(columns=[ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME]).groupby("agent_name")
        agent_level_metrics = self.calculate_metrics_single_df(agent_level_df)
        for agent_metrics in agent_level_metrics:
            agent_metrics[AGENT_REF_KEY_NAME] = {"name": agent_metrics.pop("agent_name")}

        repeat_level_metrics = self._compute_repeat_level_metrics(df)
        self._append_repeat_level_aggregates_to_agent_level_metrics(agent_level_metrics, repeat_level_metrics)

        return group_level_metrics, agent_level_metrics, repeat_level_metrics

    def prepare_for_serialization(self, metrics: List[Dict]) -> List[Dict]:
        """
        Non-destructively cleans metrics output by RewardProfiler for downstream serialization.
        """
        results = []
        for row in metrics:
            row = row.copy()
            for key in list(row):
                if key.startswith("histogram"):
                    row.pop(key)

            results.append(row)

        return results

    def write_to_disk(
        self,
        group_level_metrics: List[Dict[str, Any]],
        agent_level_metrics: List[Dict[str, Any]],
        repeat_level_metrics: List[Dict[str, Any]],
        base_output_fpath: Path,
    ) -> Tuple[Path, Path, Path]:
        reward_profiling_fpath = base_output_fpath.with_stem(base_output_fpath.stem + "_reward_profiling").with_suffix(
            ".jsonl"
        )
        with reward_profiling_fpath.open("wb") as f:
            for row in self.prepare_for_serialization(group_level_metrics):
                f.write(orjson.dumps(row) + b"\n")

        agent_level_metrics_fpath = base_output_fpath.with_stem(base_output_fpath.stem + "_agent_metrics").with_suffix(
            ".json"
        )
        agent_level_metrics_fpath.write_bytes(orjson.dumps(self.prepare_for_serialization(agent_level_metrics)))

        repeat_level_metrics_fpath = base_output_fpath.with_stem(
            base_output_fpath.stem + "_repeat_level_metrics"
        ).with_suffix(".json")
        repeat_level_metrics_fpath.write_bytes(orjson.dumps(self.prepare_for_serialization(repeat_level_metrics)))

        return reward_profiling_fpath, agent_level_metrics_fpath, repeat_level_metrics_fpath


def compute_pass_majority_metrics(
    tasks: List[List[Dict[str, Any]]],
    score_fn: Optional[Any] = None,
    answer_key: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[List[Dict[str, float]]], List[str], int]:
    """Compute pass@k, majority@k, no_answer, and variance statistics from grouped task results.

    Shared utility for any resource server's compute_metrics() override.

    Args:
        tasks: tasks[i] is a list of rollout dicts for task i.
        score_fn: Callable(result_dict) -> Dict[str, float|bool] returning named scores.
            Defaults to ``lambda r: {"accuracy": r["reward"]}``.
        answer_key: Field name for extracted answer (enables majority@k and no_answer).
            If None, majority@k and no_answer are skipped.

    Returns:
        Metrics, all_score_dicts, score_names, max_k
        Flat dict of metrics keyed as ``{agg_mode}/{score_name}``:
        - ``pass@{k}/{name}``: combinatorial pass@k (binary) or max-of-k (continuous)
        - ``pass@1[avg-of-{k}]/{name}``: mean score across first k rollouts, averaged across tasks
        - ``majority@{k}/{name}``: majority-vote accuracy (only if answer_key is set)
        - ``pass@{k}/no_answer``, ``majority@{k}/no_answer``: fraction with no extracted answer
        - ``pass@1[avg-of-{k}]/{name}/std_dev_across_runs``, ``…/std_err_across_runs``: variance stats

        All accuracy values are percentages (0-100).
    """

    if not tasks:
        return {}, [], [], 0

    if score_fn is None:
        score_fn = lambda r: {"accuracy": r["reward"]}  # noqa: E731

    max_k = max(len(rollouts) for rollouts in tasks)
    metrics: Dict[str, Any] = {}

    # Extract per-task score dicts and answers.
    # When answer_key is set, inject "no_answer" as a binary score so it gets
    # the same pass@k / majority@k / variance treatment as every other score.
    all_score_dicts: List[List[Dict[str, float]]] = []
    all_answers: List[List[Optional[str]]] = []
    for rollouts in tasks:
        task_scores = []
        task_answers = []
        for r in rollouts:
            raw = score_fn(r)
            scores = {k: (int(v) if isinstance(v, bool) else v) for k, v in raw.items()}
            if answer_key is not None:
                answer = r.get(answer_key)
                task_answers.append(answer)
                scores["no_answer"] = 1 if answer is None else 0
            task_scores.append(scores)
        all_score_dicts.append(task_scores)
        if answer_key is not None:
            all_answers.append(task_answers)

    # Collect score names
    score_names = sorted({name for task_scores in all_score_dicts for s in task_scores for name in s})

    for k in range(1, max_k + 1):
        for name in score_names:
            # --- pass@k ---
            pass_values = []
            for task_scores in all_score_dicts:
                vals = [s.get(name) for s in task_scores if name in s]
                if not vals or k > len(vals):
                    continue
                is_binary = all(v in (0, 1, 0.0, 1.0) for v in vals)
                if is_binary:
                    n_total = len(vals)
                    n_incorrect = sum(1 for v in vals if not v)
                    if n_incorrect < k:
                        pass_values.append(1.0)
                    else:
                        pass_values.append(1.0 - math.comb(n_incorrect, k) / math.comb(n_total, k))
                else:
                    pass_values.append(max(vals[:k]))

            if pass_values:
                metrics[f"pass@{k}/{name}"] = 100.0 * sum(pass_values) / len(pass_values)

            # --- pass@1[avg-of-k] ---
            avg_values = []
            for task_scores in all_score_dicts:
                vals = [s.get(name) for s in task_scores[:k] if name in s]
                if vals:
                    avg_values.append(sum(vals) / len(vals))

            if avg_values:
                metrics[f"pass@1[avg-of-{k}]/{name}"] = 100.0 * sum(avg_values) / len(avg_values)

            # --- majority@k ---
            if answer_key is not None:
                majority_values = []
                for task_scores, task_answers in zip(all_score_dicts, all_answers):
                    valid = [
                        (a, s.get(name))
                        for a, s in zip(task_answers[:k], task_scores[:k])
                        if a is not None and name in s
                    ]
                    if not valid:
                        majority_values.append(0)
                        continue
                    counter = Counter(valid)
                    max_count = counter.most_common(1)[0][1]
                    tied = [(a, s) for (a, s), c in counter.items() if c == max_count]
                    majority_values.append(sum(s for _, s in tied) / len(tied))

                if majority_values:
                    metrics[f"majority@{k}/{name}"] = 100.0 * sum(majority_values) / len(majority_values)

    # --- per_sample_aggregate and variance statistics ---
    # per_sample_aggregate[score_name][i] = pass@1 using only rollout i across all tasks
    per_sample_agg: Dict[str, List[float]] = {name: [] for name in score_names}

    for run_idx in range(max_k):
        for name in score_names:
            run_scores = [
                task_scores[run_idx].get(name)
                for task_scores in all_score_dicts
                if run_idx < len(task_scores) and name in task_scores[run_idx]
            ]
            if run_scores:
                per_sample_agg[name].append(100.0 * sum(run_scores) / len(run_scores))

    # Remove empty entries
    per_sample_agg = {k: v for k, v in per_sample_agg.items() if v}
    metrics["per_sample_aggregate"] = per_sample_agg

    # Variance statistics for pass@1[avg-of-k]
    if max_k > 1:
        for k in range(2, max_k + 1):
            for name in score_names:
                run_averages = per_sample_agg.get(name, [])[:k]
                if len(run_averages) >= 2:
                    mean_val = sum(run_averages) / len(run_averages)
                    variance = sum((x - mean_val) ** 2 for x in run_averages) / (len(run_averages) - 1)
                    std_dev = math.sqrt(variance)
                    std_err = std_dev / math.sqrt(len(run_averages))
                    metrics[f"pass@1[avg-of-{k}]/{name}/std_dev_across_runs"] = std_dev
                    metrics[f"pass@1[avg-of-{k}]/{name}/std_err_across_runs"] = std_err

    return metrics, all_score_dicts, score_names, max_k


def add_avg_sample_std_dev(
    metrics: Dict[str, Any],
    all_score_dicts: List[List[Dict[str, float]]],
    score_names: list,
    max_k: int,
) -> None:
    """Add avg_sample_std_dev statistics to an existing metrics dict.

    Computes the average of per-task standard deviations across k rollouts — a measure of
    within-task variance that complements the across-run variance (std_dev_across_runs).

    Modifies ``metrics`` in place.
    """
    if max_k <= 1:
        return

    for k in range(2, max_k + 1):
        for name in score_names:
            sample_std_devs = []
            for task_scores in all_score_dicts:
                vals = [s.get(name) for s in task_scores[:k] if name in s]
                if len(vals) >= 2:
                    task_mean = sum(vals) / len(vals)
                    task_var = sum((v - task_mean) ** 2 for v in vals) / (len(vals) - 1)
                    sample_std_devs.append(math.sqrt(task_var))
            if sample_std_devs:
                metrics[f"pass@1[avg-of-{k}]/{name}/avg_sample_std_dev"] = sum(sample_std_devs) / len(sample_std_devs)


def compute_subset_metrics(
    tasks: List[List[Dict[str, Any]]],
    subset_key: str,
    score_fn: Optional[Any] = None,
    answer_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Group tasks by a field and compute pass@k metrics per subset.

    Returns flat dict with subset-prefixed keys, e.g. ``"easy/pass@1/accuracy"``.
    Skips the ``per_sample_aggregate`` key from each subset's metrics.

    Args:
        tasks: tasks[i] is a list of rollout dicts for task i.
        subset_key: Field name in rollout dicts to group by (e.g. ``"difficulty"``).
        score_fn: Passed through to ``compute_pass_majority_metrics``.
        answer_key: Passed through to ``compute_pass_majority_metrics``.
    """
    subsets: Dict[str, List[List[Dict[str, Any]]]] = {}
    for task_rollouts in tasks:
        value = task_rollouts[0].get(subset_key) if task_rollouts else None
        if value:
            subsets.setdefault(value, []).append(task_rollouts)

    metrics: Dict[str, Any] = {}
    for subset_name, subset_tasks in subsets.items():
        subset_metrics, _, _, _ = compute_pass_majority_metrics(subset_tasks, score_fn=score_fn, answer_key=answer_key)
        for key, value in subset_metrics.items():
            if key == "per_sample_aggregate":
                continue
            metrics[f"{subset_name}/{key}"] = value

    return metrics


def highest_k_metrics(
    agent_metrics: Dict[str, Any],
    pattern: str,
    score_names: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Select the highest-k entries matching a metric pattern.

    Finds all keys matching ``pattern`` (with ``{k}`` as the k placeholder), determines the
    highest k value, and returns all entries at that k.

    Args:
        agent_metrics: Full agent metrics dict.
        pattern: Pattern with ``{k}`` placeholder, e.g. ``"pass@{k}"`` or ``"pass@1[avg-of-{k}]"``.
        score_names: If provided, only return entries whose score name (after the last ``/``)
            is in this list. Stat suffixes (std_dev, std_err, avg_sample) are always excluded.
        exclude_names: Score names to exclude (e.g. ``["no_answer"]``). Applied after score_names.

    Returns:
        Dict of matching metrics at the highest k, e.g. ``{"pass@32/accuracy": 95.0}``.

    Example::

        # Get highest-k pass@k for accuracy only
        highest_k_metrics(am, "pass@{k}", score_names=["accuracy"])
        # → {"pass@32/accuracy": 95.0}

        # Get highest-k pass@1[avg-of-k] for all scores except no_answer, without stats
        highest_k_metrics(am, "pass@1[avg-of-{k}]", exclude_names=["no_answer"])
        # → {"pass@1[avg-of-32]/accuracy": 94.5, "pass@1[avg-of-32]/symbolic_accuracy": 93.2}
    """
    stat_suffixes = {"std_dev_across_runs", "std_err_across_runs", "avg_sample_std_dev"}

    # Build regex from pattern: "pass@{k}" → r"^pass@(\d+)/(.+)$"
    escaped = re.escape(pattern).replace(r"\{k\}", r"(\d+)")
    regex = re.compile(f"^{escaped}/(.+)$")

    # Find all matching keys and their k values
    matches = []
    for key in agent_metrics:
        m = regex.match(key)
        if not m:
            continue
        k_val = int(m.group(1))
        score_name = m.group(2)

        # Skip stat suffixes
        if any(score_name.endswith(s) for s in stat_suffixes):
            continue

        if score_names is not None and score_name not in score_names:
            continue
        if exclude_names is not None and score_name in exclude_names:
            continue

        matches.append((k_val, key))

    if not matches:
        return {}

    max_k = max(k for k, _ in matches)
    return {key: agent_metrics[key] for k, key in matches if k == max_k}


class AggregateMetricsMixin:
    """Mixin providing compute_metrics/get_key_metrics hooks and the aggregate_metrics endpoint.

    Inherited by both SimpleResourcesServer and SimpleResponsesAPIAgent so that
    benchmark-specific metric logic can live on either server type.
    """

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Override to compute custom metrics from all verify responses.

        Receives verify responses grouped by task: tasks[i] is a list of rollout
        dicts for task i. Each dict has at minimum reward, plus any custom fields
        from the verify response (e.g. symbolic_correct, judgement-gen-base).

        Use for metrics that need the full dataset at once:
        - Confidence intervals (ArenaMetrics)
        - Cross-task statistics (std_dev_across_runs)
        - pass@k with proper combinatorial computation

        The returned dict is merged into agent_metrics.
        Default: empty dict (no additional metrics).
        """
        return {}

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Override to select headline metrics for this benchmark.

        Default: all mean/* entries from agent_metrics.
        """
        return {k: v for k, v in agent_metrics.items() if k.startswith("mean/")}


def _group_by_task(verify_responses: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group verify responses by task index, returning a list of per-task rollout lists."""
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for vr in verify_responses:
        groups[vr.get(TASK_INDEX_KEY_NAME, 0)].append(vr)
    return [groups[k] for k in sorted(groups)]


def _stat(values: List[float], stat: str) -> float:
    if stat == "mean":
        return statistics.fmean(values)
    if stat == "median":
        return statistics.median(values)
    if stat == "total":
        return sum(values)
    if stat == "std":
        return statistics.pstdev(values)
    if stat == "min":
        return min(values)
    if stat == "max":
        return max(values)
    if stat.startswith("p"):
        # Series.quantile() default (linear interpolation) matches numpy.percentile's default
        # and, unlike statistics.quantiles, handles a single-value series with no special-casing.
        return float(Series(values).quantile(int(stat[1:]) / 100))
    raise ValueError(f"Unknown stat {stat!r}")


# Per-rollout ng_perf token fields, each summarized with the same mean/median/total stats
# (simpler than RFC R3's literal per-field list, which asymmetrically omits median/total for
# some fields). Result keys are f"{stat}_{field}" except total_latency_ms, which uses
# total_latency_{stat}_ms -- see the dedicated block in compute_perf_summary.
_PERF_SUMMARY_TOKEN_FIELDS: Tuple[str, ...] = (
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
)
_PERF_SUMMARY_TOKEN_STATS: Tuple[str, ...] = ("mean", "median", "total")
_PERF_SUMMARY_NUM_TURNS_STATS: Tuple[str, ...] = ("mean", "median", "std", "min", "p90", "max")
_PERF_SUMMARY_LATENCY_STATS: Tuple[str, ...] = ("p50", "p90", "p99", "mean")


def compute_perf_summary(ng_perf_records: List[Dict[str, Any]], total_rollouts: int) -> Optional[Dict[str, Any]]:
    """Aggregate per-rollout ``ng_perf`` dicts into the ``perf_summary`` block (RFC R3).

    ``total_rollouts`` is the full rollout count for this batch, including rollouts that never
    produced ``ng_perf`` at all -- the denominator for ``overall_observability_coverage`` and
    ``token_observability_coverage``.

    Returns ``None`` when there are no rollouts at all, or when none of them carried ``ng_perf``
    (observability off for the whole run, or on but nothing was collected) -- perf_summary is
    absent rather than present with a 0.0 coverage either way; a coverage value, when present, is
    always > 0. Every other stat is included only when at least one rollout reported the
    underlying field (a provider that never reports cache usage yields no
    ``mean_cached_prompt_tokens``, for example).
    """
    if total_rollouts <= 0 or not ng_perf_records:
        return None

    def _values(key: str) -> List[float]:
        return [record[key] for record in ng_perf_records if isinstance(record.get(key), (int, float))]

    summary: Dict[str, Any] = {
        "overall_observability_coverage": len(ng_perf_records) / total_rollouts,
        "token_observability_coverage": sum(
            1
            for record in ng_perf_records
            if isinstance(record.get("token_observability_coverage"), (int, float))
            and record["token_observability_coverage"] > 0
        )
        / total_rollouts,
    }

    for field in _PERF_SUMMARY_TOKEN_FIELDS:
        values = _values(field)
        if not values:
            continue
        for stat in _PERF_SUMMARY_TOKEN_STATS:
            summary[f"{stat}_{field}"] = _stat(values, stat)

    num_turns_values = _values("num_turns")
    if num_turns_values:
        for stat in _PERF_SUMMARY_NUM_TURNS_STATS:
            summary[f"{stat}_num_turns"] = _stat(num_turns_values, stat)

    # Per-rollout ratio, then distribution of that ratio across rollouts.
    tokens_per_turn = [
        record["completion_tokens"] / record["num_turns"]
        for record in ng_perf_records
        if isinstance(record.get("completion_tokens"), (int, float))
        and isinstance(record.get("num_turns"), (int, float))
        and record["num_turns"] > 0
    ]
    if tokens_per_turn:
        summary["mean_tokens_per_turn"] = _stat(tokens_per_turn, "mean")
        summary["median_tokens_per_turn"] = _stat(tokens_per_turn, "median")

    latency_values = _values("total_latency_ms")
    if latency_values:
        for stat in _PERF_SUMMARY_LATENCY_STATS:
            summary[f"total_latency_{stat}_ms"] = _stat(latency_values, stat)

    return summary


def compute_aggregate_metrics(
    verify_responses: List[Dict[str, Any]],
    compute_metrics_fn=None,
    get_key_metrics_fn=None,
) -> AggregateMetrics:
    """Shared aggregation logic for /aggregate_metrics.

    RewardProfiler runs with defaults to produce baseline stats (mean/max/min/median/std)
    for both group-level (per-task) and agent-level metrics.

    Optionally accepts custom functions for benchmark-specific customization:
      - compute_metrics_fn: receives ALL verify responses grouped by task
        (List[List[Dict]]) for metrics that need the full dataset (e.g. confidence
        intervals, cross-task statistics, pass@k). Returned dict is merged into agent_metrics.
      - get_key_metrics_fn: select headline metrics from agent_metrics
    """
    if not verify_responses:
        return AggregateMetrics()

    rp = RewardProfiler()

    rows = []
    results = []
    for vr in verify_responses:
        rows.append(
            {
                TASK_INDEX_KEY_NAME: vr.get(TASK_INDEX_KEY_NAME, 0),
                ROLLOUT_INDEX_KEY_NAME: vr.get(ROLLOUT_INDEX_KEY_NAME, 0),
                "agent_ref": {"name": "agent"},
            }
        )
        results.append(vr if "response" in vr else {**vr, "response": {}})

    group_level_metrics, agent_level_metrics, repeat_level_metrics = rp.profile_from_data(rows, results)

    # Flatten agent_level_metrics (one entry since we use a single agent name)
    agent_metrics: Dict[str, Any] = {}
    for entry in agent_level_metrics:
        for k, v in entry.items():
            if k != "agent_ref":
                agent_metrics[k] = v

    # Same as agent_level_metrics above — callers nest this list under the real agent_ref.
    serialized_repeat_level_metrics = [
        {k: v for k, v in entry.items() if k != "agent_ref"} for entry in repeat_level_metrics
    ]

    serialized_group = rp.prepare_for_serialization(group_level_metrics)

    # Keep task index explicit in aggregate metrics for downstream per-task joins.
    sorted_task_indices = sorted({vr.get(TASK_INDEX_KEY_NAME, 0) for vr in verify_responses})
    for group, task_idx in zip(serialized_group, sorted_task_indices):
        group[TASK_INDEX_KEY_NAME] = task_idx

    serialized_agent = rp.prepare_for_serialization([agent_metrics])[0] if agent_metrics else {}

    # Custom metrics computed from all raw verify responses grouped by task
    if compute_metrics_fn:
        tasks = _group_by_task(verify_responses)
        custom = compute_metrics_fn(tasks)

        # Merge per_task_metrics into group_level_metrics (keyed by task_index)
        per_task_metrics = custom.pop("per_task_metrics", None)
        if per_task_metrics:
            per_task_by_idx = {m[TASK_INDEX_KEY_NAME]: m for m in per_task_metrics}
            for group in serialized_group:
                task_idx = group.get(TASK_INDEX_KEY_NAME)
                if task_idx is not None and task_idx in per_task_by_idx:
                    ptm = per_task_by_idx[task_idx]
                    for k, v in ptm.items():
                        if k != TASK_INDEX_KEY_NAME:
                            group[k] = v

        serialized_agent.update(custom)

    if get_key_metrics_fn:
        key_metrics = get_key_metrics_fn(serialized_agent)
    else:
        key_metrics = {k: v for k, v in serialized_agent.items() if k.startswith("mean/")}

    ng_perf_records = [vr["ng_perf"] for vr in verify_responses if isinstance(vr.get("ng_perf"), dict)]

    return AggregateMetrics(
        group_level_metrics=serialized_group,
        agent_metrics=serialized_agent,
        key_metrics=key_metrics,
        perf_summary=compute_perf_summary(ng_perf_records, total_rollouts=len(verify_responses)),
        repeat_level_metrics=serialized_repeat_level_metrics,
    )


# Backward-compatibility shim (CLI refactor): this CLI entry point moved to `nemo_gym.cli.eval`.
# Re-exported lazily to avoid a circular import; accessing it emits a DeprecationWarning.
from nemo_gym.cli._compat import moved_attr_getter  # noqa: E402


__getattr__ = moved_attr_getter(
    __name__,
    {
        "reward_profile": "nemo_gym.cli.eval",
    },
)
