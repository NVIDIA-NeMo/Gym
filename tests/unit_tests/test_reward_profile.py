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

        # Each task has 1 correct out of 2
        # pass@1: 1 - C(1,1)/C(2,1) = 0.5 for all tasks -> 50%
        assert output.aggregate["pass@1"]["reward"] == pytest.approx(50.0)

        # pass@2: 1 - C(1,2)/C(2,2) = 1.0 for all tasks -> 100%
        assert output.aggregate["pass@2"]["reward"] == pytest.approx(100.0)

        # avg-of-2: (0+1)/2 = 0.5 for all tasks -> 50%
        assert output.aggregate["pass@1[avg-of-2]"]["reward"] == pytest.approx(50.0)

        # Per sample
        assert output.per_sample_aggregate["sample_0"]["reward"] == pytest.approx(0.0)
        assert output.per_sample_aggregate["sample_1"]["reward"] == pytest.approx(100.0)

        # Per task
        assert len(output.per_task) == 3
        assert output.per_task[0]["num_rollouts"] == 2

        # Usage
        assert output.usage["abc_usage"] == pytest.approx(1.0)

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

        assert output.aggregate["pass@1"]["reward"] == pytest.approx(50.0)
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
        assert "statistics" in data
        assert "per_task" in data
        assert "usage" in data
