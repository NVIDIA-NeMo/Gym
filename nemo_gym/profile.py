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
from typing import Optional

from pydantic import BaseModel, Field

from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import TASK_INDEX_KEY_NAME, get_global_config_dict


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
