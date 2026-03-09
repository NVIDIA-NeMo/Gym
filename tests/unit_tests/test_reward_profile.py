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

import pytest

from nemo_gym.reward_profile import MetricsProfiler


class TestMetricsProfiler:
    def test_profile_from_data(self) -> None:
        results = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "response": {"usage": {"abc_usage": 1}}, "reward": 0},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "response": {"usage": {"abc_usage": 1}}, "reward": 1},
            {"_ng_task_index": 1, "_ng_rollout_index": 0, "response": {"usage": {"abc_usage": 1}}, "reward": 0},
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "response": {"usage": {"abc_usage": 1}}, "reward": 1},
            {"_ng_task_index": 2, "_ng_rollout_index": 0, "response": {"usage": {"abc_usage": 1}}, "reward": 0},
            {"_ng_task_index": 2, "_ng_rollout_index": 1, "response": {"usage": {"abc_usage": 1}}, "reward": 1},
        ]

        mp = MetricsProfiler()
        output = mp.profile_from_data(results)

        # Each task has rollouts [0, 1]
        # pass@1: max([:1]) = 0.0 for all tasks -> 0%
        assert output.aggregate["pass@1"]["reward"] == pytest.approx(0.0)

        # pass@2: max([:2]) = max(0,1) = 1.0 for all tasks -> 100%
        assert output.aggregate["pass@2"]["reward"] == pytest.approx(100.0)

        # avg-of-2: (0+1)/2 = 0.5 for all tasks -> 50%
        assert output.aggregate["pass@1[avg-of-2]"]["reward"] == pytest.approx(50.0)

        # Per sample: [pass@1 using rollout 0, pass@1 using rollout 1]
        assert output.per_sample_aggregate["reward"][0] == pytest.approx(0.0)
        assert output.per_sample_aggregate["reward"][1] == pytest.approx(100.0)

        # Per task
        assert len(output.per_task) == 3
        assert output.per_task[0]["num_rollouts"] == 2

        # Usage (now has mean + std_dev)
        assert output.usage["abc_usage"]["mean"] == pytest.approx(1.0)
        assert output.usage["abc_usage"]["std_dev"] == pytest.approx(0.0)

    def test_profile_from_data_unsorted(self) -> None:
        """Results don't need to be pre-sorted; MetricsProfiler groups them."""
        results = [
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "response": {}, "reward": 1},
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "response": {}, "reward": 0},
            {"_ng_task_index": 1, "_ng_rollout_index": 0, "response": {}, "reward": 0},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "response": {}, "reward": 1},
        ]

        mp = MetricsProfiler()
        output = mp.profile_from_data(results)

        # Each task: rollout 0 = 0, rollout 1 = 1. pass@1: max([:1]) = 0 -> 0%
        assert output.aggregate["pass@1"]["reward"] == pytest.approx(0.0)
        assert len(output.per_task) == 2

    def test_profile_with_metrics_type(self) -> None:
        results = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "response": {}, "reward": 1.0},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "response": {}, "reward": 0.0},
        ]

        mp = MetricsProfiler()
        output = mp.profile_from_data(results, metrics_type="resources_servers.code_gen.metrics::CodeGenMetrics")

        # CodeGenMetrics uses "accuracy" instead of "reward"
        assert "accuracy" in output.aggregate["pass@1"]
        assert "reward" not in output.aggregate["pass@1"]

    def test_write_to_disk(self, tmp_path: Path) -> None:
        results = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "response": {}, "reward": 1.0},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "response": {}, "reward": 0.0},
        ]

        mp = MetricsProfiler()
        output = mp.profile_from_data(results)
        metrics_fpath = mp.write_to_disk(output, tmp_path / "rollouts.jsonl")

        assert metrics_fpath.exists()
        assert metrics_fpath.name == "rollouts_metrics.json"

        data = json.loads(metrics_fpath.read_text())
        assert "aggregate" in data
        assert "per_sample_aggregate" in data
        assert "per_task" in data
        assert "usage" in data
