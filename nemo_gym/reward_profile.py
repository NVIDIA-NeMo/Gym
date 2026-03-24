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

from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import orjson
from pandas import DataFrame, Series, notna
from pandas.core.groupby.generic import DataFrameGroupBy
from pydantic import Field
from wandb import Histogram

from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    get_global_config_dict,
)


if TYPE_CHECKING:
    from nemo_gym.base_resources_server import AggregateMetrics


class RewardProfileConfig(BaseNeMoGymCLIConfig):
    materialized_inputs_jsonl_fpath: str = Field(
        description="The file path of the materialized inputs as output by ng_collect_rollouts."
    )
    rollouts_jsonl_fpath: str = Field(description="The file path of the rollouts as output by ng_collect_rollouts.")


class RewardProfiler:
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
        ]

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

    def profile_from_data(
        self,
        rows: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        filtered_results: List[Dict] = []
        task_idx_to_row: Dict[int, Dict] = dict()
        for row, result in zip(rows, results):
            # Add additional helpful information
            result = result | (result["response"].get("usage") or {})

            # agent_name is a temporary column used for aggregations below
            numeric_result = {"agent_name": row["agent_ref"]["name"]}
            for k, v in result.items():
                if isinstance(v, bool):
                    numeric_result[k] = int(v)
                elif isinstance(v, (int, float)):
                    numeric_result[k] = v

            filtered_results.append(numeric_result)
            task_idx_to_row.setdefault(row[TASK_INDEX_KEY_NAME], row)

        df = DataFrame.from_records(filtered_results)

        group_level_df = df.drop(columns=[ROLLOUT_INDEX_KEY_NAME, "agent_name"]).groupby(TASK_INDEX_KEY_NAME)
        group_level_metrics = self.calculate_metrics_single_df(group_level_df)
        for group_metrics in group_level_metrics:
            row = task_idx_to_row[group_metrics[TASK_INDEX_KEY_NAME]]

            row = row.copy()
            row.pop(TASK_INDEX_KEY_NAME)
            row.pop(ROLLOUT_INDEX_KEY_NAME)

            group_metrics["sample"] = row

            group_metrics.pop(TASK_INDEX_KEY_NAME)

        agent_level_df = df.drop(columns=[ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME]).groupby("agent_name")
        agent_level_metrics = self.calculate_metrics_single_df(agent_level_df)
        for agent_metrics in agent_level_metrics:
            agent_metrics[AGENT_REF_KEY_NAME] = {"name": agent_metrics.pop("agent_name")}

        return group_level_metrics, agent_level_metrics

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
        base_output_fpath: Path,
    ) -> Tuple[Path, Path]:
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

        return reward_profiling_fpath, agent_level_metrics_fpath


def compute_pass_majority_metrics(
    tasks: List[List[Dict[str, Any]]],
    score_fn: Optional[Any] = None,
    answer_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute pass@k, majority@k, no_answer, and variance statistics from grouped task results.

    Shared utility for any resource server's compute_metrics() override.

    Args:
        tasks: tasks[i] is a list of rollout dicts for task i.
        score_fn: Callable(result_dict) -> Dict[str, float|bool] returning named scores.
            Defaults to ``lambda r: {"accuracy": r["reward"]}``.
        answer_key: Field name for extracted answer (enables majority@k and no_answer).
            If None, majority@k and no_answer are skipped.

    Returns:
        Flat dict of metrics keyed as ``{agg_mode}/{score_name}``:
        - ``pass@{k}/{name}``: combinatorial pass@k (binary) or max-of-k (continuous)
        - ``pass@1[avg-of-{k}]/{name}``: mean score across first k rollouts, averaged across tasks
        - ``majority@{k}/{name}``: majority-vote accuracy (only if answer_key is set)
        - ``pass@{k}/no_answer``, ``majority@{k}/no_answer``: fraction with no extracted answer
        - ``pass@1[avg-of-{k}]/{name}/std_dev_across_runs``, ``…/std_err_across_runs``: variance stats

        All accuracy values are percentages (0-100).
    """
    import math as _math

    if not tasks:
        return {}

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
                        pass_values.append(1.0 - _math.comb(n_incorrect, k) / _math.comb(n_total, k))
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
                    std_dev = _math.sqrt(variance)
                    std_err = std_dev / _math.sqrt(len(run_averages))
                    metrics[f"pass@1[avg-of-{k}]/{name}/std_dev_across_runs"] = std_dev
                    metrics[f"pass@1[avg-of-{k}]/{name}/std_err_across_runs"] = std_err

    return metrics


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
    # Import here to avoid circular dependency (AggregateMetrics is defined in base_resources_server)
    from nemo_gym.base_resources_server import AggregateMetrics

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

    group_level_metrics, agent_level_metrics = rp.profile_from_data(rows, results)

    # Flatten agent_level_metrics (one entry since we use a single agent name)
    agent_metrics: Dict[str, Any] = {}
    for entry in agent_level_metrics:
        for k, v in entry.items():
            if k != "agent_ref":
                agent_metrics[k] = v

    serialized_group = rp.prepare_for_serialization(group_level_metrics)

    # Re-add task index (RewardProfiler pops it during profiling, but groups are
    # returned in sorted task index order from the pandas groupby)
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

    return AggregateMetrics(
        group_level_metrics=serialized_group,
        agent_metrics=serialized_agent,
        key_metrics=key_metrics,
    )


def reward_profile():  # pragma: no cover
    config = RewardProfileConfig.model_validate(get_global_config_dict())

    with open(config.materialized_inputs_jsonl_fpath) as f:
        rows = list(map(orjson.loads, f))

    with open(config.rollouts_jsonl_fpath) as f:
        results = list(map(orjson.loads, f))

    # Results may be out of order.
    results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

    rp = RewardProfiler()
    group_level_metrics, agent_level_metrics = rp.profile_from_data(rows, results)
    reward_profiling_fpath, agent_level_metrics_fpath = rp.write_to_disk(
        group_level_metrics, agent_level_metrics, Path(config.rollouts_jsonl_fpath)
    )

    print(f"""Profiling outputs:
Reward profiling outputs: {reward_profiling_fpath}
Agent-level metrics: {agent_level_metrics_fpath}""")
