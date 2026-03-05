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
import math
from typing import Dict, List, Optional, Union

import pytest

from nemo_gym.metrics import MetricsOutput, get_metrics
from nemo_gym.metrics.base import BaseMetrics
from nemo_gym.metrics.reward_metrics import RewardMetrics


class TestGetMetrics:
    def test_default_returns_reward_metrics(self) -> None:
        assert isinstance(get_metrics(None), RewardMetrics)

    def test_reward_string_returns_reward_metrics(self) -> None:
        assert isinstance(get_metrics("reward"), RewardMetrics)

    def test_class_path_resolution(self) -> None:
        metrics = get_metrics("resources_servers.code_gen.metrics::CodeGenMetrics")
        from resources_servers.code_gen.metrics import CodeGenMetrics

        assert isinstance(metrics, CodeGenMetrics)

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid metrics_type"):
            get_metrics("not.a.valid.path")

    def test_invalid_class_raises(self) -> None:
        with pytest.raises(AttributeError):
            get_metrics("nemo_gym.metrics.base::NonExistentClass")

    def test_non_metrics_class_raises(self) -> None:
        with pytest.raises(TypeError, match="does not resolve to a BaseMetrics subclass"):
            get_metrics("nemo_gym.metrics.base::MetricsOutput")


class TestRewardMetrics:
    def test_simple_binary_rewards(self) -> None:
        metrics = RewardMetrics()
        # 3 tasks, 2 rollouts each
        task_results = [
            [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 0.0, "response": {}}, {"reward": 1.0, "response": {}}],
            [{"reward": 1.0, "response": {}}, {"reward": 1.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        assert isinstance(output, MetricsOutput)
        assert "pass@1" in output.aggregate
        assert "pass@2" in output.aggregate
        assert "pass@1[avg-of-1]" in output.aggregate
        assert "pass@1[avg-of-2]" in output.aggregate

        # pass@1 with k=1: uses combinatorial formula
        # Task 0: 1 correct out of 2, pass@1 = 1 - C(1,1)/C(2,1) = 1 - 1/2 = 0.5
        # Task 1: 1 correct out of 2, pass@1 = 0.5
        # Task 2: 2 correct out of 2, pass@1 = 1.0
        # Average: (0.5 + 0.5 + 1.0) / 3 = 0.6667 -> 66.67%
        assert output.aggregate["pass@1"]["reward"] == pytest.approx(100.0 * 2.0 / 3.0, abs=0.01)

        # pass@2: at least one correct in 2 samples
        # Task 0: 1 correct, pass@2 = 1 - C(1,2)/C(2,2) = 1 - 0/1 = 1.0
        # Task 1: same, 1.0
        # Task 2: 1.0
        # Average: 100%
        assert output.aggregate["pass@2"]["reward"] == pytest.approx(100.0)

        # avg-of-2: mean of both rollouts per task, then averaged
        # Task 0: (1+0)/2 = 0.5, Task 1: (0+1)/2 = 0.5, Task 2: (1+1)/2 = 1.0
        # Average: (0.5 + 0.5 + 1.0) / 3 = 0.6667 -> 66.67%
        assert output.aggregate["pass@1[avg-of-2]"]["reward"] == pytest.approx(100.0 * 2.0 / 3.0, abs=0.01)

    def test_per_sample_aggregate(self) -> None:
        metrics = RewardMetrics()
        task_results = [
            [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 0.0, "response": {}}, {"reward": 1.0, "response": {}}],
            [{"reward": 1.0, "response": {}}, {"reward": 1.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        # sample_0: uses rollout 0 from each task -> [1, 0, 1] -> mean = 66.67%
        assert output.per_sample_aggregate["sample_0"]["reward"] == pytest.approx(100.0 * 2.0 / 3.0, abs=0.01)

        # sample_1: uses rollout 1 from each task -> [0, 1, 1] -> mean = 66.67%
        assert output.per_sample_aggregate["sample_1"]["reward"] == pytest.approx(100.0 * 2.0 / 3.0, abs=0.01)

    def test_statistics(self) -> None:
        metrics = RewardMetrics()
        task_results = [
            [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 0.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 1.0, "response": {}}, {"reward": 1.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        assert "reward" in output.statistics
        stats = output.statistics["reward"]
        assert "std_dev_across_runs" in stats
        assert "std_err_across_runs" in stats
        assert "avg_sample_std_dev" in stats

        # Per-sample aggregates:
        # sample_0: [1, 0, 1] -> mean = 66.67%
        # sample_1: [0, 0, 1] -> mean = 33.33%
        # std_dev_across_runs = std([66.67, 33.33]) with ddof=1
        sample_means = [100.0 * 2.0 / 3.0, 100.0 * 1.0 / 3.0]
        mean_of_means = sum(sample_means) / len(sample_means)
        expected_std = math.sqrt(sum((x - mean_of_means) ** 2 for x in sample_means) / (len(sample_means) - 1))
        assert stats["std_dev_across_runs"] == pytest.approx(expected_std, abs=0.01)

        # std_err = std_dev / sqrt(k)
        assert stats["std_err_across_runs"] == pytest.approx(expected_std / math.sqrt(2), abs=0.01)

    def test_per_task(self) -> None:
        metrics = RewardMetrics()
        task_results = [
            [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 0.0, "response": {}}, {"reward": 1.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        assert len(output.per_task) == 2
        assert output.per_task[0]["task_index"] == 0
        assert output.per_task[0]["num_rollouts"] == 2
        assert len(output.per_task[0]["rollouts"]) == 2
        assert output.per_task[0]["rollouts"][0]["reward"] == 1.0
        assert output.per_task[0]["rollouts"][1]["reward"] == 0.0

        # Check aggregations
        agg = output.per_task[0]["aggregations"]
        assert agg["mean/reward"] == 0.5
        assert agg["max/reward"] == 1.0
        assert agg["min/reward"] == 0.0

    def test_usage_extraction(self) -> None:
        metrics = RewardMetrics()
        task_results = [
            [
                {"reward": 1.0, "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}},
                {"reward": 0.0, "response": {"usage": {"prompt_tokens": 120, "completion_tokens": 60}}},
            ],
        ]
        output = metrics.compute(task_results)

        assert output.usage["completion_tokens"] == pytest.approx(55.0)
        assert output.usage["prompt_tokens"] == pytest.approx(110.0)


class TestAllCorrectAllIncorrect:
    def test_all_correct(self) -> None:
        metrics = RewardMetrics()
        task_results = [
            [{"reward": 1.0, "response": {}}] * 3,
            [{"reward": 1.0, "response": {}}] * 3,
        ]
        output = metrics.compute(task_results)

        assert output.aggregate["pass@1"]["reward"] == pytest.approx(100.0)
        assert output.aggregate["pass@3"]["reward"] == pytest.approx(100.0)
        assert output.aggregate["pass@1[avg-of-3]"]["reward"] == pytest.approx(100.0)

    def test_all_incorrect(self) -> None:
        metrics = RewardMetrics()
        task_results = [
            [{"reward": 0.0, "response": {}}] * 3,
            [{"reward": 0.0, "response": {}}] * 3,
        ]
        output = metrics.compute(task_results)

        assert output.aggregate["pass@1"]["reward"] == pytest.approx(0.0)
        assert output.aggregate["pass@3"]["reward"] == pytest.approx(0.0)
        assert output.aggregate["pass@1[avg-of-3]"]["reward"] == pytest.approx(0.0)


class TestSingleRollout:
    def test_single_rollout(self) -> None:
        metrics = RewardMetrics()
        task_results = [
            [{"reward": 1.0, "response": {}}],
            [{"reward": 0.0, "response": {}}],
            [{"reward": 1.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        # k=1 only
        assert "pass@1" in output.aggregate
        assert "pass@2" not in output.aggregate

        # pass@1 for binary: C(n-c,1)/C(n,1) = (n-c)/n
        # Task 0: c=1, pass@1 = 1.0
        # Task 1: c=0, pass@1 = 0.0
        # Task 2: c=1, pass@1 = 1.0
        assert output.aggregate["pass@1"]["reward"] == pytest.approx(100.0 * 2.0 / 3.0, abs=0.01)

        # No statistics with k=1
        assert output.statistics == {}

        # Per sample: only sample_0
        assert len(output.per_sample_aggregate) == 1
        assert "sample_0" in output.per_sample_aggregate


class TestEmptyInput:
    def test_empty_task_results(self) -> None:
        metrics = RewardMetrics()
        output = metrics.compute([])

        assert output.aggregate == {}
        assert output.per_sample_aggregate == {}
        assert output.statistics == {}
        assert output.per_task == []
        assert output.usage == {}


class TestMultipleScores:
    def test_custom_metrics_with_multiple_scores(self) -> None:
        class MultiScoreMetrics(BaseMetrics):
            def get_score_dict(self, result: dict) -> Dict[str, Union[float, bool]]:
                return {
                    "accuracy": result["reward"],
                    "quality": result.get("quality", 0.5),
                }

        metrics = MultiScoreMetrics()
        task_results = [
            [
                {"reward": 1.0, "quality": 0.8, "response": {}},
                {"reward": 0.0, "quality": 0.3, "response": {}},
            ],
            [
                {"reward": 1.0, "quality": 0.9, "response": {}},
                {"reward": 1.0, "quality": 0.7, "response": {}},
            ],
        ]
        output = metrics.compute(task_results)

        assert "accuracy" in output.aggregate["pass@1"]
        assert "quality" in output.aggregate["pass@1"]

        # accuracy is binary, quality is continuous
        # pass@1 for accuracy: Task 0: c=1/n=2 -> 0.5, Task 1: c=2/n=2 -> 1.0. avg = 0.75
        assert output.aggregate["pass@1"]["accuracy"] == pytest.approx(75.0)

        # pass@1 for quality (continuous): max of first 1 value
        # Task 0: max([0.8]) = 0.8, Task 1: max([0.9]) = 0.9. avg = 0.85
        assert output.aggregate["pass@1"]["quality"] == pytest.approx(85.0)

        # pass@2 for quality (continuous): max of first 2 values
        # Task 0: max([0.8, 0.3]) = 0.8, Task 1: max([0.9, 0.7]) = 0.9. avg = 0.85
        assert output.aggregate["pass@2"]["quality"] == pytest.approx(85.0)


class TestMajorityAtK:
    def test_majority_voting(self) -> None:
        class AnswerMetrics(BaseMetrics):
            def get_score_dict(self, result: dict) -> Dict[str, Union[float, bool]]:
                return {"accuracy": result["reward"]}

            def get_answer(self, result: dict) -> Optional[str]:
                return result.get("answer")

        metrics = AnswerMetrics()
        task_results = [
            [
                # 2 votes for "42" (1 correct, 1 correct), 1 vote for "43" (incorrect)
                {"reward": 1.0, "answer": "42", "response": {}},
                {"reward": 1.0, "answer": "42", "response": {}},
                {"reward": 0.0, "answer": "43", "response": {}},
            ],
            [
                # 2 votes for "wrong" (incorrect), 1 vote for "right" (correct)
                {"reward": 0.0, "answer": "wrong", "response": {}},
                {"reward": 0.0, "answer": "wrong", "response": {}},
                {"reward": 1.0, "answer": "right", "response": {}},
            ],
        ]
        output = metrics.compute(task_results)

        assert "majority@3" in output.aggregate
        # Task 0: majority answer is "42" (2 votes), first "42" score = 1.0
        # Task 1: majority answer is "wrong" (2 votes), first "wrong" score = 0.0
        # Average: (1.0 + 0.0) / 2 = 0.5 -> 50%
        assert output.aggregate["majority@3"]["accuracy"] == pytest.approx(50.0)

    def test_no_answers_no_majority(self) -> None:
        metrics = RewardMetrics()
        task_results = [
            [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        # RewardMetrics doesn't override get_answer, so no majority@k
        assert not any(key.startswith("majority@") for key in output.aggregate)


class TestContinuousScores:
    def test_continuous_pass_at_k(self) -> None:
        """For non-binary scores, pass@k uses max of first k scores."""

        class ContinuousMetrics(BaseMetrics):
            def get_score_dict(self, result: dict) -> Dict[str, float]:
                return {"score": result["score"]}

        metrics = ContinuousMetrics()
        task_results = [
            [
                {"score": 0.3, "response": {}},
                {"score": 0.7, "response": {}},
                {"score": 0.5, "response": {}},
            ],
        ]
        output = metrics.compute(task_results)

        # pass@1: max of first 1 = 0.3 -> 30%
        assert output.aggregate["pass@1"]["score"] == pytest.approx(30.0)

        # pass@2: max of first 2 = max(0.3, 0.7) = 0.7 -> 70%
        assert output.aggregate["pass@2"]["score"] == pytest.approx(70.0)

        # pass@3: max of first 3 = max(0.3, 0.7, 0.5) = 0.7 -> 70%
        assert output.aggregate["pass@3"]["score"] == pytest.approx(70.0)


class TestComputeOverride:
    def test_full_compute_override(self) -> None:
        """Test that compute() can be fully overridden for custom aggregation."""

        class CustomMetrics(BaseMetrics):
            def get_score_dict(self, result: dict) -> Dict[str, float]:
                return {"score": result["score"]}

            def compute(self, task_results: List[List[dict]]) -> MetricsOutput:
                # Custom: just compute mean across all results
                all_scores = [r["score"] for results in task_results for r in results]
                mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
                return MetricsOutput(
                    aggregate={"custom": {"mean_score": mean_score}},
                    per_sample_aggregate={},
                    statistics={},
                    per_task=[],
                    usage={},
                )

        metrics = CustomMetrics()
        task_results = [
            [{"score": 0.8, "response": {}}, {"score": 0.6, "response": {}}],
            [{"score": 0.4, "response": {}}, {"score": 0.2, "response": {}}],
        ]
        output = metrics.compute(task_results)

        assert output.aggregate["custom"]["mean_score"] == pytest.approx(0.5)
        assert output.per_sample_aggregate == {}


class TestPassAtKCombinatorial:
    def test_formula_correctness(self) -> None:
        # n=5, c=2, k=1: 1 - C(3,1)/C(5,1) = 1 - 3/5 = 0.4
        assert BaseMetrics._pass_at_k_combinatorial(5, 2, 1) == pytest.approx(0.4)

        # n=5, c=2, k=5: 1 - C(3,5)/C(5,5) = 1 - 0/1 = 1.0 (n-c < k)
        assert BaseMetrics._pass_at_k_combinatorial(5, 2, 5) == pytest.approx(1.0)

        # n=5, c=0, k=1: 1 - C(5,1)/C(5,1) = 0
        assert BaseMetrics._pass_at_k_combinatorial(5, 0, 1) == pytest.approx(0.0)

        # n=5, c=5, k=1: 1 - C(0,1)/C(5,1) = 1 - 0/5 = 1.0
        assert BaseMetrics._pass_at_k_combinatorial(5, 5, 1) == pytest.approx(1.0)

        # n=10, c=3, k=2: 1 - C(7,2)/C(10,2) = 1 - 21/45 = 1 - 7/15
        assert BaseMetrics._pass_at_k_combinatorial(10, 3, 2) == pytest.approx(1.0 - 21.0 / 45.0)


class TestStatisticsCorrectness:
    def test_variance_stats(self) -> None:
        """Verify std_dev_across_runs, std_err, avg_sample_std_dev match expected values."""
        metrics = RewardMetrics()

        # 4 tasks, 3 rollouts each
        task_results = [
            [{"reward": 1.0, "response": {}}, {"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 0.0, "response": {}}, {"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}, {"reward": 1.0, "response": {}}],
            [{"reward": 0.0, "response": {}}, {"reward": 0.0, "response": {}}, {"reward": 1.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        # Per-sample means:
        # sample_0: [1, 0, 1, 0] -> mean = 50%
        # sample_1: [1, 1, 0, 0] -> mean = 50%
        # sample_2: [0, 0, 1, 1] -> mean = 50%
        for key in ["sample_0", "sample_1", "sample_2"]:
            assert output.per_sample_aggregate[key]["reward"] == pytest.approx(50.0)

        # std_dev_across_runs: std of [50, 50, 50] with ddof=1 = 0
        assert output.statistics["reward"]["std_dev_across_runs"] == pytest.approx(0.0)
        assert output.statistics["reward"]["std_err_across_runs"] == pytest.approx(0.0)

        # avg_sample_std_dev: std per task, then averaged
        # Each task has some mix of 0s and 1s with n=3
        # Task 0: [1,1,0] -> mean=2/3, var=((1-2/3)^2 + (1-2/3)^2 + (0-2/3)^2)/2 = (1/9+1/9+4/9)/2 = 1/3
        # Task 1: [0,1,0] -> mean=1/3, var=((1/3)^2 + (2/3)^2 + (1/3)^2)/2 = (1/9+4/9+1/9)/2 = 1/3
        # Task 2: [1,0,1] -> mean=2/3, same variance = 1/3
        # Task 3: [0,0,1] -> mean=1/3, same variance = 1/3
        expected_avg_std = math.sqrt(1 / 3)
        assert output.statistics["reward"]["avg_sample_std_dev"] == pytest.approx(expected_avg_std, abs=0.001)


class TestMathMetrics:
    def test_math_metrics_with_judge(self) -> None:
        from resources_servers.math_with_judge.metrics import MathMetrics

        metrics = MathMetrics()
        task_results = [
            [
                {
                    "reward": 1.0,
                    "library_reward": 1.0,
                    "judge_evaluations": [{"verdict": "A=B"}],
                    "extracted_answer": "42",
                    "response": {},
                },
                {
                    "reward": 0.0,
                    "library_reward": 0.0,
                    "judge_evaluations": [{"verdict": "A!=B"}],
                    "extracted_answer": "43",
                    "response": {},
                },
            ],
        ]
        output = metrics.compute(task_results)

        # Should have symbolic_accuracy, judge_accuracy, and accuracy
        assert "symbolic_accuracy" in output.aggregate["pass@1"]
        assert "judge_accuracy" in output.aggregate["pass@1"]
        assert "accuracy" in output.aggregate["pass@1"]

        # majority@2 should be present (different answers, so it's a tie, picks first most common)
        assert "majority@2" in output.aggregate

    def test_math_metrics_without_judge(self) -> None:
        from resources_servers.math_with_judge.metrics import MathMetrics

        metrics = MathMetrics()
        task_results = [
            [
                {
                    "reward": 1.0,
                    "library_reward": 1.0,
                    "judge_evaluations": None,
                    "extracted_answer": "42",
                    "response": {},
                },
            ],
        ]
        output = metrics.compute(task_results)

        assert "symbolic_accuracy" in output.aggregate["pass@1"]
        assert "judge_accuracy" not in output.aggregate["pass@1"]
        assert "accuracy" in output.aggregate["pass@1"]


class TestCodeGenMetrics:
    def test_code_gen_metrics(self) -> None:
        from resources_servers.code_gen.metrics import CodeGenMetrics

        metrics = CodeGenMetrics()
        task_results = [
            [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 0.0, "response": {}}, {"reward": 0.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        assert "accuracy" in output.aggregate["pass@1"]
        # Task 0: c=1/n=2, pass@1 = 0.5
        # Task 1: c=0/n=2, pass@1 = 0.0
        # Average: 25%
        assert output.aggregate["pass@1"]["accuracy"] == pytest.approx(25.0)

        # pass@2: Task 0: 1 - C(1,2)/C(2,2) = 1.0, Task 1: 0.0. avg = 50%
        assert output.aggregate["pass@2"]["accuracy"] == pytest.approx(50.0)
