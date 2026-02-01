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
from pathlib import Path

from pytest import MonkeyPatch

import nemo_gym.profile
from nemo_gym.global_config import TASK_INDEX_KEY_NAME
from nemo_gym.profile import ProfileConfig, RewardProfilingMetrics, profile


class TestProfile:
    def test_config_sanity(self) -> None:
        ProfileConfig(
            input_jsonl_fpath="",
            rollouts_jsonl_fpath="",
            output_jsonl_fpath="",
        )

    def test_reward_profiling_metrics(self) -> None:
        metrics = RewardProfilingMetrics(
            avg_reward=0.75,
            std_reward=0.25,
            min_reward=0.5,
            max_reward=1.0,
            total_samples=4,
        )
        assert metrics.avg_reward == 0.75
        assert metrics.pass_rate is None

        metrics.pass_rate = 0.5
        metrics.pass_rate_total = 4
        metrics.pass_rate_passed = 2
        metrics.pass_threshold = 1.0
        dumped = metrics.model_dump(exclude_none=True)
        assert "pass_rate" in dumped
        assert dumped["pass_rate"] == 0.5

    def test_profile_computes_correct_metrics(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        tasks = [
            {"question": "What is 2+2?"},
            {"question": "What is 3+3?"},
        ]
        input_fpath = tmp_path / "tasks.jsonl"
        with open(input_fpath, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")

        rollouts = [
            {TASK_INDEX_KEY_NAME: 0, "reward": 1.0},
            {TASK_INDEX_KEY_NAME: 0, "reward": 0.5},
            {TASK_INDEX_KEY_NAME: 1, "reward": 0.0},
            {TASK_INDEX_KEY_NAME: 1, "reward": 1.0},
        ]
        rollouts_fpath = tmp_path / "rollouts.jsonl"
        with open(rollouts_fpath, "w") as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + "\n")

        output_fpath = tmp_path / "profiled.jsonl"

        monkeypatch.setattr(
            nemo_gym.profile,
            "get_global_config_dict",
            lambda: {
                "input_jsonl_fpath": str(input_fpath),
                "rollouts_jsonl_fpath": str(rollouts_fpath),
                "output_jsonl_fpath": str(output_fpath),
            },
        )

        profile()

        with open(output_fpath) as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2

        assert results[0]["question"] == "What is 2+2?"
        assert results[0]["avg_reward"] == 0.75
        assert results[0]["min_reward"] == 0.5
        assert results[0]["max_reward"] == 1.0
        assert results[0]["total_samples"] == 2

        assert results[1]["question"] == "What is 3+3?"
        assert results[1]["avg_reward"] == 0.5
        assert results[1]["min_reward"] == 0.0
        assert results[1]["max_reward"] == 1.0
        assert results[1]["total_samples"] == 2
