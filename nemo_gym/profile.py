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
from pathlib import Path
from typing import Any, Dict, List, Tuple

import orjson
from pandas import DataFrame
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


class RewardProfileConfig(BaseNeMoGymCLIConfig):
    materialized_inputs_jsonl_fpath: str = Field(
        description="The file path of the materialized inputs as output by ng_collect_rollouts."
    )
    rollouts_jsonl_fpath: str = Field(description="The file path of the rollouts as output by ng_collect_rollouts.")


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

    def profile_and_write_to_disk(
        self,
        rows: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        base_output_fpath: Path,
    ) -> Tuple[Path, Path]:
        group_level_metrics, agent_level_metrics = self.profile_from_data(rows, results)

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


def profile():  # pragma: no cover
    config = RewardProfileConfig.model_validate(get_global_config_dict())

    with open(config.materialized_inputs_jsonl_fpath) as f:
        rows = list(map(orjson.loads, f))

    with open(config.rollouts_jsonl_fpath) as f:
        results = list(map(orjson.loads, f))

    rp = RewardProfiler()
    reward_profiling_fpath, agent_level_metrics_fpath = rp.profile_and_write_to_disk(
        rows, results, Path(config.rollouts_jsonl_fpath)
    )

    print(f"""Profiling outputs:
Reward profiling outputs: {reward_profiling_fpath}
Agent-level metrics: {agent_level_metrics_fpath}""")
