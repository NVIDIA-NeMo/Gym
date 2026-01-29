# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

from pydantic import Field

from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import get_global_config_dict


class ProfileConfig(BaseNeMoGymCLIConfig):
    input_jsonl_fpath: str = Field(description="Original task dataset.")
    rollouts_jsonl_fpath: str = Field(description="Rollouts file from ng_collect_rollouts with num_repeats.")
    output_jsonl_fpath: str = Field(description="Output file for profiled dataset.")
    pass_threshold: Optional[float] = Field(
        default=None, description="Reward threshold for pass_rate. If None, pass_rate not computed."
    )


def profile():
    config = ProfileConfig.model_validate(get_global_config_dict())

    with open(config.input_jsonl_fpath) as f:
        tasks = [json.loads(line) for line in f]

    with open(config.rollouts_jsonl_fpath) as f:
        rollouts = [json.loads(line) for line in f]

    grouped = defaultdict(list)
    for rollout in rollouts:
        task_idx = rollout.get("_task_index")
        if task_idx is not None:
            grouped[task_idx].append(rollout)

    Path(config.output_jsonl_fpath).parent.mkdir(exist_ok=True, parents=True)
    with open(config.output_jsonl_fpath, "w") as f:
        for task_idx, task_rollouts in sorted(grouped.items()):
            if task_idx >= len(tasks):
                continue

            rewards = [r.get("reward", 0.0) for r in task_rollouts]
            profiled_task = {**tasks[task_idx]}

            avg = sum(rewards) / len(rewards)
            profiled_task["avg_reward"] = avg
            profiled_task["std_reward"] = (sum((r - avg) ** 2 for r in rewards) / len(rewards)) ** 0.5
            profiled_task["min_reward"] = min(rewards)
            profiled_task["max_reward"] = max(rewards)
            profiled_task["total_samples"] = len(rewards)

            if config.pass_threshold is not None:
                passed = sum(1 for r in rewards if r >= config.pass_threshold)
                profiled_task["pass_rate"] = passed / len(rewards)
                profiled_task["pass_rate_total"] = len(rewards)
                profiled_task["pass_rate_passed"] = passed
                profiled_task["pass_threshold"] = config.pass_threshold

            f.write(json.dumps(profiled_task) + "\n")