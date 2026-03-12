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

from collections import defaultdict
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


class AggregateMetricsMixin:
    """Mixin providing compute_metrics/get_key_metrics hooks and the aggregate_metrics endpoint.

    Inherited by both SimpleResourcesServer and SimpleResponsesAPIAgent so that
    benchmark-specific metric logic can live on either server type.
    """

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute custom metrics from all verify responses grouped by task.

        Default: if config has metrics_type, resolves a BaseMetrics subclass and runs
        its compute() pipeline (pass@k, majority@k, per-sample statistics, usage).
        The MetricsOutput is flattened to flat keys and merged into agent_metrics.

        Override for fully custom aggregation (e.g. arena win-rate).
        """
        metrics_type = getattr(getattr(self, "config", None), "metrics_type", None)
        if not metrics_type:
            return {}

        from nemo_gym.metrics import get_metrics

        metrics = get_metrics(metrics_type)
        output = metrics.compute(tasks)

        # Flatten MetricsOutput.aggregate into flat keys
        flat: Dict[str, Any] = {}
        for mode, scores in output.aggregate.items():
            for name, val in scores.items():
                flat[f"{mode}/{name}"] = val
        for key, usage_stats in output.usage.items():
            for stat_name, stat_val in usage_stats.items():
                flat[f"usage/{key}/{stat_name}"] = stat_val
        flat["per_sample_aggregate"] = output.per_sample_aggregate
        flat["per_task"] = output.per_task
        return flat

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for this benchmark.

        Default: all mean/* entries from RewardProfiler, plus the highest-k
        pass@1[avg-of-*] scores from BaseMetrics (excluding statistics).
        """
        key = {}
        # RewardProfiler mean stats
        for k, v in agent_metrics.items():
            if k.startswith("mean/"):
                key[k] = v

        # BaseMetrics: find highest-k pass@1[avg-of-*] entry
        avg_of_keys = [k for k in agent_metrics if k.startswith("pass@1[avg-of-")]
        if avg_of_keys:
            # Group by prefix to find the highest k
            highest_k = max(avg_of_keys, key=lambda k: int(k.split("pass@1[avg-of-")[1].split("]")[0]))
            prefix = highest_k.split("/")[0] + "/"
            for k, v in agent_metrics.items():
                if k.startswith(prefix) and "std_dev" not in k and "std_err" not in k:
                    key[k] = v

        # BaseMetrics: highest-k pass@k
        pass_keys = [k for k in agent_metrics if k.startswith("pass@") and "[" not in k]
        if pass_keys:
            highest_pass = max(pass_keys, key=lambda k: int(k.split("@")[1].split("/")[0]))
            prefix = highest_pass.split("/")[0] + "/"
            for k, v in agent_metrics.items():
                if k.startswith(prefix):
                    key[k] = v

        return key


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
    serialized_agent = rp.prepare_for_serialization([agent_metrics])[0] if agent_metrics else {}

    # Custom metrics computed from all raw verify responses grouped by task
    if compute_metrics_fn:
        tasks = _group_by_task(verify_responses)
        serialized_agent.update(compute_metrics_fn(tasks))

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
