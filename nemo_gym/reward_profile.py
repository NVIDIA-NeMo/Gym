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
from pydantic import Field

from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import (
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


def reward_profile():  # pragma: no cover
    config = RewardProfileConfig.model_validate(get_global_config_dict())

    with open(config.rollouts_jsonl_fpath) as f:
        results = list(map(orjson.loads, f))

    # Results may be out of order.
    results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

    mp = MetricsProfiler()
    metrics_output = mp.profile_from_data(results, metrics_type=config.metrics_type)
    metrics_fpath = mp.write_to_disk(metrics_output, Path(config.rollouts_jsonl_fpath))

    print(f"Metrics output: {metrics_fpath}")

    # Print aggregate summary
    if metrics_output.aggregate:
        print("\nAggregate metrics:")
        for mode, scores in metrics_output.aggregate.items():
            scores_str = ", ".join(f"{name}: {val:.2f}" for name, val in scores.items())
            print(f"  {mode}: {scores_str}")
