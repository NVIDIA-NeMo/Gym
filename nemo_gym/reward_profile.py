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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from nemo_gym.metrics import MetricsOutput, get_metrics


class RewardProfileConfig(BaseNeMoGymCLIConfig):
    materialized_inputs_jsonl_fpath: str = Field(
        description="The file path of the materialized inputs as output by ng_collect_rollouts."
    )
    rollouts_jsonl_fpath: str = Field(description="The file path of the rollouts as output by ng_collect_rollouts.")
    metrics_type: Optional[str] = Field(
        default=None, description="Metrics class path (e.g. 'resources_servers.code_gen.metrics::CodeGenMetrics')."
    )


class MetricsProfiler:
    """Computes metrics from rollout results using the pluggable metrics system."""

    def profile_from_data(
        self,
        results: List[Dict[str, Any]],
        metrics_type: Optional[str] = None,
    ) -> MetricsOutput:
        """Group results by task index, resolve metrics class, and compute metrics.

        Args:
            results: List of result dicts, each containing _ng_task_index and _ng_rollout_index.
            metrics_type: Optional metrics class path. Defaults to RewardMetrics.

        Returns:
            MetricsOutput with all computed metrics.
        """
        metrics = get_metrics(metrics_type)

        # Group results by task index, sorted by rollout index
        task_groups: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
        for result in results:
            task_idx = result[TASK_INDEX_KEY_NAME]
            rollout_idx = result[ROLLOUT_INDEX_KEY_NAME]
            task_groups[task_idx].append((rollout_idx, result))

        # Sort tasks by index, then sort rollouts within each task
        task_results: List[List[dict]] = []
        for task_idx in sorted(task_groups.keys()):
            rollouts = task_groups[task_idx]
            rollouts.sort(key=lambda x: x[0])
            task_results.append([result for _, result in rollouts])

        return metrics.compute(task_results)

    def write_to_disk(
        self,
        metrics_output: MetricsOutput,
        base_output_fpath: Path,
    ) -> Path:
        """Write metrics output to a single JSON file.

        Args:
            metrics_output: The computed MetricsOutput.
            base_output_fpath: Base path for the output file.

        Returns:
            Path to the written metrics file.
        """
        metrics_fpath = base_output_fpath.with_stem(base_output_fpath.stem + "_metrics").with_suffix(".json")
        metrics_fpath.write_bytes(orjson.dumps(metrics_output.model_dump(), option=orjson.OPT_INDENT_2))
        return metrics_fpath


class RewardProfiler:
    """Legacy reward profiler that computes pandas describe() stats over all numeric fields."""

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

        # Std is more interpretable using 0 rather than NaN for no std
        if d[4].isna().all():
            not_na_columns = df.columns[df.notna().all()]
            d[4][not_na_columns] = d[4][not_na_columns].fillna(0)

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
            result = result | result["response"].get("usage", dict())

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
        """Non-destructively cleans metrics output by RewardProfiler for downstream serialization."""
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


def reward_profile():  # pragma: no cover
    config = RewardProfileConfig.model_validate(get_global_config_dict())

    with open(config.materialized_inputs_jsonl_fpath) as f:
        rows = list(map(orjson.loads, f))

    with open(config.rollouts_jsonl_fpath) as f:
        results = list(map(orjson.loads, f))

    # Results may be out of order.
    rows.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
    results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

    # New pluggable metrics
    mp = MetricsProfiler()
    metrics_output = mp.profile_from_data(results, metrics_type=config.metrics_type)
    metrics_fpath = mp.write_to_disk(metrics_output, Path(config.rollouts_jsonl_fpath))

    # Legacy reward profiling
    rp = RewardProfiler()
    group_level_metrics, agent_level_metrics = rp.profile_from_data(rows, results)
    reward_profiling_fpath, agent_metrics_fpath = rp.write_to_disk(
        group_level_metrics, agent_level_metrics, Path(config.rollouts_jsonl_fpath)
    )

    print(f"""Profiling outputs:
Metrics: {metrics_fpath}
Reward profiling: {reward_profiling_fpath}
Agent metrics: {agent_metrics_fpath}""")

    # Print aggregate summary
    if metrics_output.aggregate:
        print("\nAggregate metrics:")
        for mode, scores in metrics_output.aggregate.items():
            scores_str = ", ".join(f"{name}: {val:.2f}" for name, val in scores.items())
            print(f"  {mode}: {scores_str}")
