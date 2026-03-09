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

from nemo_gym.metrics import NOT_FOUND, MetricsOutput, get_metrics
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

        # pass@1: max of first 1 score per task
        # Task 0: max([1.0]) = 1.0, Task 1: max([0.0]) = 0.0, Task 2: max([1.0]) = 1.0
        # Average: (1.0 + 0.0 + 1.0) / 3 = 0.6667 -> 66.67%
        assert output.aggregate["pass@1"]["reward"] == pytest.approx(100.0 * 2.0 / 3.0, abs=0.01)

        # pass@2: max of first 2 scores per task
        # Task 0: max([1.0, 0.0]) = 1.0, Task 1: max([0.0, 1.0]) = 1.0, Task 2: max([1.0, 1.0]) = 1.0
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

        # per_sample_aggregate["reward"] is a list: [pass@1 using rollout 0, pass@1 using rollout 1]
        assert "reward" in output.per_sample_aggregate
        assert len(output.per_sample_aggregate["reward"]) == 2

        # rollout 0: [1, 0, 1] -> mean = 66.67%
        assert output.per_sample_aggregate["reward"][0] == pytest.approx(100.0 * 2.0 / 3.0, abs=0.01)

        # rollout 1: [0, 1, 1] -> mean = 66.67%
        assert output.per_sample_aggregate["reward"][1] == pytest.approx(100.0 * 2.0 / 3.0, abs=0.01)

    def test_statistics_fused_into_aggregate(self) -> None:
        metrics = RewardMetrics()
        task_results = [
            [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 0.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 1.0, "response": {}}, {"reward": 1.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        # Statistics are fused into pass@1[avg-of-2] only
        avg_of_2 = output.aggregate["pass@1[avg-of-2]"]
        assert "reward_std_dev_across_runs" in avg_of_2
        assert "reward_std_err_across_runs" in avg_of_2

        # pass@1 and pass@2 should NOT have statistics
        assert "reward_std_dev_across_runs" not in output.aggregate["pass@1"]
        assert "reward_std_dev_across_runs" not in output.aggregate["pass@2"]

        # Per-sample aggregates:
        # rollout 0: [1, 0, 1] -> mean = 66.67%
        # rollout 1: [0, 0, 1] -> mean = 33.33%
        # std_dev_across_runs = std([66.67, 33.33]) with ddof=1
        sample_means = [100.0 * 2.0 / 3.0, 100.0 * 1.0 / 3.0]
        mean_of_means = sum(sample_means) / len(sample_means)
        expected_std = math.sqrt(sum((x - mean_of_means) ** 2 for x in sample_means) / (len(sample_means) - 1))
        assert avg_of_2["reward_std_dev_across_runs"] == pytest.approx(expected_std, abs=0.01)

        # std_err = std_dev / sqrt(k)
        assert avg_of_2["reward_std_err_across_runs"] == pytest.approx(expected_std / math.sqrt(2), abs=0.01)

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

        assert output.usage["completion_tokens"]["mean"] == pytest.approx(55.0)
        assert output.usage["prompt_tokens"]["mean"] == pytest.approx(110.0)
        # std_dev with 2 values
        assert output.usage["completion_tokens"]["std_dev"] > 0
        assert output.usage["prompt_tokens"]["std_dev"] > 0

    def test_usage_nested_dicts(self) -> None:
        """Nested usage dicts (e.g. input_token_details) are flattened with dot-separated keys."""
        metrics = RewardMetrics()
        task_results = [
            [
                {
                    "reward": 1.0,
                    "response": {
                        "usage": {
                            "prompt_tokens": 100,
                            "prompt_tokens_details": {"cached_tokens": 20, "audio_tokens": 0},
                        }
                    },
                },
            ],
        ]
        output = metrics.compute(task_results)

        assert "prompt_tokens" in output.usage
        assert "prompt_tokens_details.cached_tokens" in output.usage
        assert "prompt_tokens_details.audio_tokens" in output.usage
        assert output.usage["prompt_tokens_details.cached_tokens"]["mean"] == pytest.approx(20.0)


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

        # pass@1: max of first 1 score
        # Task 0: 1.0, Task 1: 0.0, Task 2: 1.0
        assert output.aggregate["pass@1"]["reward"] == pytest.approx(100.0 * 2.0 / 3.0, abs=0.01)

        # No statistics with k=1 (nothing fused into aggregate)
        assert "reward_std_dev_across_runs" not in output.aggregate.get("pass@1", {})

        # Per sample: only 1 rollout, so each score has a list of length 1
        assert "reward" in output.per_sample_aggregate
        assert len(output.per_sample_aggregate["reward"]) == 1


class TestEmptyInput:
    def test_empty_task_results(self) -> None:
        metrics = RewardMetrics()
        output = metrics.compute([])

        assert output.aggregate == {}
        assert output.per_sample_aggregate == {}
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

        # pass@1 for accuracy: max of first 1
        # Task 0: max([1.0]) = 1.0, Task 1: max([1.0]) = 1.0. avg = 1.0
        assert output.aggregate["pass@1"]["accuracy"] == pytest.approx(100.0)

        # pass@1 for quality: max of first 1 value
        # Task 0: max([0.8]) = 0.8, Task 1: max([0.9]) = 0.9. avg = 0.85
        assert output.aggregate["pass@1"]["quality"] == pytest.approx(85.0)

        # pass@2 for quality: max of first 2 values
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


class TestPassAtK:
    def test_max_of_k(self) -> None:
        """pass@k uses max of first k scores for all score types."""
        metrics = RewardMetrics()
        # 2 tasks, 3 rollouts each
        task_results = [
            [{"reward": 0.0, "response": {}}, {"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 0.0, "response": {}}, {"reward": 0.0, "response": {}}, {"reward": 1.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        # pass@1: max([:1]) -> Task 0: 0.0, Task 1: 0.0. avg = 0%
        assert output.aggregate["pass@1"]["reward"] == pytest.approx(0.0)

        # pass@2: max([:2]) -> Task 0: max(0,1)=1.0, Task 1: max(0,0)=0.0. avg = 50%
        assert output.aggregate["pass@2"]["reward"] == pytest.approx(50.0)

        # pass@3: max([:3]) -> Task 0: 1.0, Task 1: 1.0. avg = 100%
        assert output.aggregate["pass@3"]["reward"] == pytest.approx(100.0)


class TestStatisticsCorrectness:
    def test_variance_stats_fused(self) -> None:
        """Verify std_dev_across_runs, std_err, avg_sample_std_dev are fused into pass@1[avg-of-k]."""
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
        # rollout 0: [1, 0, 1, 0] -> mean = 50%
        # rollout 1: [1, 1, 0, 0] -> mean = 50%
        # rollout 2: [0, 0, 1, 1] -> mean = 50%
        assert len(output.per_sample_aggregate["reward"]) == 3
        for val in output.per_sample_aggregate["reward"]:
            assert val == pytest.approx(50.0)

        # Statistics fused into pass@1[avg-of-3]
        avg_of_3 = output.aggregate["pass@1[avg-of-3]"]

        # std_dev_across_runs: std of [50, 50, 50] with ddof=1 = 0
        assert avg_of_3["reward_std_dev_across_runs"] == pytest.approx(0.0)
        assert avg_of_3["reward_std_err_across_runs"] == pytest.approx(0.0)

        # avg_sample_std_dev: std per task, then averaged
        # Each task has some mix of 0s and 1s with n=3
        # Task 0: [1,1,0] -> mean=2/3, var=((1-2/3)^2 + (1-2/3)^2 + (0-2/3)^2)/2 = (1/9+1/9+4/9)/2 = 1/3
        # Task 1: [0,1,0] -> mean=1/3, var=((1/3)^2 + (2/3)^2 + (1/3)^2)/2 = (1/9+4/9+1/9)/2 = 1/3
        # Task 2: [1,0,1] -> mean=2/3, same variance = 1/3
        # Task 3: [0,0,1] -> mean=1/3, same variance = 1/3
        expected_avg_std = math.sqrt(1 / 3)
        assert avg_of_3["reward_avg_sample_std_dev"] == pytest.approx(expected_avg_std, abs=0.001)

        # pass@1 and majority should NOT have statistics
        assert "reward_std_dev_across_runs" not in output.aggregate["pass@1"]


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


class TestNoAnswer:
    def test_not_found_excluded_from_voting_and_tracked(self) -> None:
        """NOT_FOUND answers are excluded from majority voting but counted in no_answer metric."""

        class AnswerMetrics(BaseMetrics):
            def get_score_dict(self, result: dict) -> Dict[str, Union[float, bool]]:
                return {"accuracy": result["reward"]}

            def get_answer(self, result: dict) -> Optional[str]:
                return result.get("answer", NOT_FOUND)

        metrics = AnswerMetrics()
        task_results = [
            # Task 0: has answers
            [
                {"reward": 1.0, "answer": "42", "response": {}},
                {"reward": 0.0, "answer": "43", "response": {}},
            ],
            # Task 1: all NOT_FOUND (no "answer" key)
            [
                {"reward": 0.0, "response": {}},
                {"reward": 0.0, "response": {}},
            ],
        ]
        output = metrics.compute(task_results)

        # majority@2 should exist (task 0 has answers)
        assert "majority@2" in output.aggregate

        # no_answer should be 50% (1 out of 2 tasks has all NOT_FOUND)
        assert output.aggregate["majority@2"]["no_answer"] == pytest.approx(50.0)

    def test_custom_no_answer_label(self) -> None:
        class CustomLabelMetrics(BaseMetrics):
            def get_score_dict(self, result: dict) -> Dict[str, Union[float, bool]]:
                return {"accuracy": result["reward"]}

            def get_answer(self, result: dict) -> Optional[str]:
                return result.get("answer", NOT_FOUND)

            @property
            def no_answer_label(self) -> str:
                return "invalid_judgements"

        metrics = CustomLabelMetrics()
        task_results = [
            [{"reward": 1.0, "answer": "42", "response": {}}, {"reward": 0.0, "response": {}}],
            [{"reward": 0.0, "response": {}}, {"reward": 0.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        # Should use custom label
        assert "invalid_judgements" in output.aggregate.get("majority@2", {})
        assert "no_answer" not in output.aggregate.get("majority@2", {})

    def test_none_answers_skip_majority_entirely(self) -> None:
        """When get_answer() is not overridden (returns None), no majority@k or no_answer."""
        metrics = RewardMetrics()
        task_results = [
            [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}],
        ]
        output = metrics.compute(task_results)

        assert not any(key.startswith("majority@") for key in output.aggregate)


class TestMetricsOutputSchema:
    """Golden-output test that documents the full MetricsOutput schema.

    This test serves as living documentation of the exact shape of MetricsOutput
    for a concrete example: 3 math tasks, 2 rollouts each, with majority voting.
    """

    def test_full_output_schema(self) -> None:
        class MathLikeMetrics(BaseMetrics):
            def get_score_dict(self, result: dict) -> Dict[str, Union[float, bool]]:
                return {"accuracy": result["reward"]}

            def get_answer(self, result: dict) -> Optional[str]:
                return result.get("answer", NOT_FOUND)

        metrics = MathLikeMetrics()
        task_results = [
            # Task 0: both correct, same answer
            [
                {
                    "reward": 1.0,
                    "answer": "42",
                    "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
                },
                {
                    "reward": 1.0,
                    "answer": "42",
                    "response": {"usage": {"prompt_tokens": 110, "completion_tokens": 60}},
                },
            ],
            # Task 1: first wrong, second correct
            [
                {"reward": 0.0, "answer": "7", "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 40}}},
                {"reward": 1.0, "answer": "5", "response": {"usage": {"prompt_tokens": 120, "completion_tokens": 80}}},
            ],
            # Task 2: both wrong, answer extraction failed on second
            [
                {"reward": 0.0, "answer": "99", "response": {"usage": {"prompt_tokens": 90, "completion_tokens": 30}}},
                {"reward": 0.0, "response": {"usage": {"prompt_tokens": 95, "completion_tokens": 35}}},
            ],
        ]
        output = metrics.compute(task_results)

        # === aggregate ===
        # Keys: pass@1, pass@2, pass@1[avg-of-1], pass@1[avg-of-2], majority@1, majority@2
        # Only pass@1[avg-of-2] gets fused statistics.
        # no_answer appears when any task has all-NOT_FOUND answers for that k.

        # pass@1: max([:1]) per task -> [1.0, 0.0, 0.0] -> avg = 33.33%
        assert output.aggregate["pass@1"]["accuracy"] == pytest.approx(100.0 / 3.0, abs=0.01)

        # pass@2: max([:2]) per task -> [1.0, 1.0, 0.0] -> avg = 66.67%
        assert output.aggregate["pass@2"]["accuracy"] == pytest.approx(200.0 / 3.0, abs=0.01)

        # pass@1[avg-of-2]: mean of 2 rollouts per task -> [(1+1)/2, (0+1)/2, (0+0)/2] = [1.0, 0.5, 0.0] -> 50%
        assert output.aggregate["pass@1[avg-of-2]"]["accuracy"] == pytest.approx(50.0)
        # Statistics fused into pass@1[avg-of-2] only
        assert "accuracy_std_dev_across_runs" in output.aggregate["pass@1[avg-of-2]"]
        assert "accuracy_std_err_across_runs" in output.aggregate["pass@1[avg-of-2]"]
        assert "accuracy_avg_sample_std_dev" in output.aggregate["pass@1[avg-of-2]"]
        # Other agg keys don't get statistics
        assert "accuracy_std_dev_across_runs" not in output.aggregate["pass@1"]
        assert "accuracy_std_dev_across_runs" not in output.aggregate["pass@2"]

        # majority@2: vote among 2 answers, use winner's score
        assert "majority@2" in output.aggregate
        # Task 2 rollout 1 has no answer (NOT_FOUND) -> only 1 valid answer "99" -> score 0.0
        assert "accuracy" in output.aggregate["majority@2"]

        # no_answer: Task 2 rollout 1 is NOT_FOUND, but task 2 rollout 0 has answer "99"
        # So no task has ALL answers as NOT_FOUND for k=2 -> no no_answer key
        # (no_answer only appears when ALL k answers for a task are NOT_FOUND)
        assert "no_answer" not in output.aggregate.get("majority@2", {})

        # === per_sample_aggregate ===
        # Dict[str, List[float]] — element i = pass@1 using only rollout i
        assert set(output.per_sample_aggregate.keys()) == {"accuracy"}
        assert len(output.per_sample_aggregate["accuracy"]) == 2
        # rollout 0: [1.0, 0.0, 0.0] -> mean = 33.33%
        assert output.per_sample_aggregate["accuracy"][0] == pytest.approx(100.0 / 3.0, abs=0.01)
        # rollout 1: [1.0, 1.0, 0.0] -> mean = 66.67%
        assert output.per_sample_aggregate["accuracy"][1] == pytest.approx(200.0 / 3.0, abs=0.01)

        # === per_task ===
        # List of dicts, one per task, with rollout scores and task-level aggregations
        assert len(output.per_task) == 3
        task0 = output.per_task[0]
        assert task0["task_index"] == 0
        assert task0["num_rollouts"] == 2
        assert len(task0["rollouts"]) == 2
        assert task0["rollouts"][0] == {"rollout_index": 0, "accuracy": 1, "answer": "42"}
        assert task0["rollouts"][1] == {"rollout_index": 1, "accuracy": 1, "answer": "42"}
        assert task0["aggregations"]["mean/accuracy"] == pytest.approx(1.0)
        assert task0["aggregations"]["max/accuracy"] == 1
        assert task0["aggregations"]["min/accuracy"] == 1
        assert "pass@1/accuracy" in task0["aggregations"]
        assert "pass@2/accuracy" in task0["aggregations"]

        # === usage ===
        # Dict[str, Dict[str, float]] — each key has {"mean": ..., "std_dev": ...}
        assert "prompt_tokens" in output.usage
        assert "completion_tokens" in output.usage
        assert set(output.usage["prompt_tokens"].keys()) == {"mean", "std_dev"}
        assert output.usage["prompt_tokens"]["mean"] == pytest.approx((100 + 110 + 100 + 120 + 90 + 95) / 6.0)
        assert output.usage["prompt_tokens"]["std_dev"] > 0


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
        # pass@1: max([:1]) -> Task 0: 1.0, Task 1: 0.0. avg = 50%
        assert output.aggregate["pass@1"]["accuracy"] == pytest.approx(50.0)

        # pass@2: max([:2]) -> Task 0: max(1,0)=1.0, Task 1: max(0,0)=0.0. avg = 50%
        assert output.aggregate["pass@2"]["accuracy"] == pytest.approx(50.0)
