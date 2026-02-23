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
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from pandas import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy
from pydantic import BaseModel, Field
from wandb import Histogram

from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    get_global_config_dict,
)


class ProfileConfig(BaseNeMoGymCLIConfig):
    input_jsonl_fpath: str = Field(description="Original task dataset.")
    rollouts_jsonl_fpath: str = Field(description="Rollouts file from ng_collect_rollouts with num_repeats.")
    output_jsonl_fpath: str = Field(description="Output file for profiled dataset.")
    pass_threshold: Optional[float] = Field(
        default=None, description="Reward threshold for pass_rate. If None, pass_rate not computed."
    )


class RewardProfilingMetrics(BaseModel):
    avg_reward: float = Field(description="Average reward across all rollouts for this task.")
    std_reward: float = Field(description="Standard deviation of rewards.")
    min_reward: float = Field(description="Minimum reward observed.")
    max_reward: float = Field(description="Maximum reward observed.")
    total_samples: int = Field(description="Number of rollout samples for this task.")
    pass_rate: Optional[float] = Field(default=None, description="Fraction of rollouts meeting pass_threshold.")
    pass_rate_total: Optional[int] = Field(default=None, description="Total rollouts used for pass_rate calculation.")
    pass_rate_passed: Optional[int] = Field(default=None, description="Number of rollouts that passed.")
    pass_threshold: Optional[float] = Field(default=None, description="Threshold used for pass_rate calculation.")


class RewardProfiler:
    def describe_dataframe(self, df: DataFrame) -> DataFrame:
        stat_index = ["mean", "max", "min", "median", "std", "histogram"]
        d = [
            df.mean(),
            df.max(),
            df.min(),
            df.median(),
            df.std(),
            df.apply(Histogram, axis=0),
        ]
        return DataFrame(d, index=stat_index).stack()

    def calculate_metrics_single_df(self, grouped_df: DataFrameGroupBy) -> List[Dict[str, Any]]:
        grouped_metrics_df = grouped_df.apply(self.describe_dataframe, include_groups=False)
        grouped_metrics_df.columns = grouped_metrics_df.columns.map("/".join)
        grouped_metrics_df = grouped_metrics_df.reset_index()
        grouped_metrics = grouped_metrics_df.to_dict("records")
        return grouped_metrics

    def profile_from_data(
        self,
        rows: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        filtered_results: List[Dict] = []
        task_idx_to_row: Dict[int, Dict] = dict()
        for row, result in zip(rows, results):
            # Add additional helpful information
            result = result | result["response"].get("usage", None)

            numeric_results = {k: v for k, v in result.items() if isinstance(v, (int, float))}

            # agent_name is a temporary column used for aggregations below
            numeric_results["agent_name"] = row["agent_ref"]["name"]

            filtered_results.append(numeric_results)
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


def profile():
    config = ProfileConfig.model_validate(get_global_config_dict())

    grouped_rewards: dict[int, list[float]] = defaultdict(list)
    with open(config.rollouts_jsonl_fpath) as f:
        for line in f:
            rollout = json.loads(line)
            task_idx = rollout.get(TASK_INDEX_KEY_NAME)
            if task_idx is not None:
                grouped_rewards[task_idx].append(rollout.get("reward", 0.0))

    Path(config.output_jsonl_fpath).parent.mkdir(exist_ok=True, parents=True)
    with open(config.input_jsonl_fpath) as f_in, open(config.output_jsonl_fpath, "w") as f_out:
        for task_idx, line in enumerate(f_in):
            if task_idx not in grouped_rewards:
                continue

            task = json.loads(line)
            rewards = grouped_rewards[task_idx]
            avg = sum(rewards) / len(rewards)

            metrics = RewardProfilingMetrics(
                avg_reward=avg,
                std_reward=(sum((r - avg) ** 2 for r in rewards) / len(rewards)) ** 0.5,
                min_reward=min(rewards),
                max_reward=max(rewards),
                total_samples=len(rewards),
            )

            if config.pass_threshold is not None:
                passed = sum(1 for r in rewards if r >= config.pass_threshold)
                metrics.pass_rate = passed / len(rewards)
                metrics.pass_rate_total = len(rewards)
                metrics.pass_rate_passed = passed
                metrics.pass_threshold = config.pass_threshold

            profiled_task = {**task, **metrics.model_dump(exclude_none=True)}
            f_out.write(json.dumps(profiled_task) + "\n")
