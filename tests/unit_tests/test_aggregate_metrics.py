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
from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseResourcesServerConfig,
    SimpleResourcesServer,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.reward_profile import (
    RewardProfiler,
    add_avg_sample_std_dev,
    compute_aggregate_metrics,
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from nemo_gym.server_utils import ServerClient


class _TestResourcesServer(SimpleResourcesServer):
    async def verify(self, body):
        pass


def _make_server():
    config = BaseResourcesServerConfig(host="127.0.0.1", port=12345, entrypoint="app.py", name="test_server")
    return _TestResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_verify_responses(tasks, rollouts_per_task, reward_fn=None):
    if reward_fn is None:
        reward_fn = lambda t, r: float((t + r) % 2)

    responses = []
    for task_idx in range(tasks):
        for rollout_idx in range(rollouts_per_task):
            responses.append(
                {
                    TASK_INDEX_KEY_NAME: task_idx,
                    ROLLOUT_INDEX_KEY_NAME: rollout_idx,
                    "reward": reward_fn(task_idx, rollout_idx),
                }
            )
    return responses


class TestAggregateMetricsRoute:
    @pytest.mark.asyncio
    async def test_basic_route(self) -> None:
        server = _make_server()
        responses = _make_verify_responses(tasks=2, rollouts_per_task=4)
        body = AggregateMetricsRequest(verify_responses=responses)

        result = await server.aggregate_metrics(body)

        assert isinstance(result, AggregateMetrics)
        assert len(result.group_level_metrics) == 2
        # Agent metrics should have reward stats
        assert "mean/reward" in result.agent_metrics

    @pytest.mark.asyncio
    async def test_group_level_has_reward_stats(self) -> None:
        server = _make_server()
        responses = _make_verify_responses(tasks=2, rollouts_per_task=3)
        body = AggregateMetricsRequest(verify_responses=responses)

        result = await server.aggregate_metrics(body)

        assert len(result.group_level_metrics) == 2
        group0 = result.group_level_metrics[0]
        assert "mean/reward" in group0

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        server = _make_server()
        body = AggregateMetricsRequest(verify_responses=[])

        result = await server.aggregate_metrics(body)

        assert result.group_level_metrics == []
        assert result.agent_metrics == {}

    @pytest.mark.asyncio
    async def test_agent_metrics_has_overall_stats(self) -> None:
        server = _make_server()
        responses = _make_verify_responses(tasks=3, rollouts_per_task=5, reward_fn=lambda t, r: 1.0)
        body = AggregateMetricsRequest(verify_responses=responses)

        result = await server.aggregate_metrics(body)

        assert result.agent_metrics["mean/reward"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_key_metrics_default(self) -> None:
        server = _make_server()
        responses = _make_verify_responses(tasks=2, rollouts_per_task=3, reward_fn=lambda t, r: 1.0)
        body = AggregateMetricsRequest(verify_responses=responses)

        result = await server.aggregate_metrics(body)

        assert "mean/reward" in result.key_metrics
        assert result.key_metrics["mean/reward"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_histograms_stripped(self) -> None:
        """RewardProfiler produces histograms; they should be stripped from the response."""
        server = _make_server()
        responses = _make_verify_responses(tasks=2, rollouts_per_task=3)
        body = AggregateMetricsRequest(verify_responses=responses)

        result = await server.aggregate_metrics(body)

        for group in result.group_level_metrics:
            assert not any(k.startswith("histogram") for k in group), f"Histogram key found in group: {group.keys()}"
        assert not any(k.startswith("histogram") for k in result.agent_metrics)


class TestComputeMetricsHook:
    @pytest.mark.asyncio
    async def test_compute_metrics_receives_grouped_responses(self) -> None:
        """compute_metrics receives all verify responses grouped by task."""

        class _MathServer(SimpleResourcesServer):
            async def verify(self, body):
                pass

            def compute_metrics(self, tasks):
                # tasks[i] is a list of rollout dicts for task i
                assert len(tasks) == 3
                assert all(len(rollouts) == 4 for rollouts in tasks)

                # Compute pass@k: fraction of tasks where any rollout got reward=1
                pass_at_k = sum(1 for rollouts in tasks if any(r["reward"] >= 1.0 for r in rollouts)) / len(tasks)

                # Compute pass@1 avg-of-k: average of per-task mean rewards
                pass_at_1 = sum(sum(r["reward"] for r in rollouts) / len(rollouts) for rollouts in tasks) / len(tasks)

                return {"pass@k": pass_at_k, "pass@1_avg_of_k": pass_at_1}

            def get_key_metrics(self, agent_metrics):
                return {k: agent_metrics[k] for k in ("pass@k", "pass@1_avg_of_k") if k in agent_metrics}

        config = BaseResourcesServerConfig(host="127.0.0.1", port=12345, entrypoint="app.py", name="test_server")
        server = _MathServer(config=config, server_client=MagicMock(spec=ServerClient))

        # Task 0: all correct, Task 1: all wrong, Task 2: mixed
        def reward_fn(t, r):
            if t == 0:
                return 1.0
            if t == 1:
                return 0.0
            return float(r % 2)

        responses = _make_verify_responses(tasks=3, rollouts_per_task=4, reward_fn=reward_fn)
        body = AggregateMetricsRequest(verify_responses=responses)
        result = await server.aggregate_metrics(body)

        # 2 of 3 tasks have at least one correct rollout (task 0 and task 2)
        assert result.agent_metrics["pass@k"] == pytest.approx(2.0 / 3.0)
        assert "pass@k" in result.key_metrics
        assert "pass@1_avg_of_k" in result.key_metrics

    @pytest.mark.asyncio
    async def test_compute_metrics_sees_custom_verify_fields(self) -> None:
        """compute_metrics has access to custom fields from verify responses."""

        class _JudgeServer(SimpleResourcesServer):
            async def verify(self, body):
                pass

            def compute_metrics(self, tasks):
                # Verify we can see custom fields
                for rollouts in tasks:
                    for r in rollouts:
                        assert "judgement" in r
                return {"custom_metric": 42.0}

        config = BaseResourcesServerConfig(host="127.0.0.1", port=12345, entrypoint="app.py", name="test_server")
        server = _JudgeServer(config=config, server_client=MagicMock(spec=ServerClient))

        responses = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "judgement": "[[A>>B]]"},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "judgement": "[[B>A]]"},
        ]
        body = AggregateMetricsRequest(verify_responses=responses)
        result = await server.aggregate_metrics(body)

        assert result.agent_metrics["custom_metric"] == 42.0


class TestComputePassMajorityMetrics:
    def test_pass_at_k_binary(self) -> None:
        """Combinatorial pass@k for binary rewards."""
        tasks = [
            [{"reward": 1.0}, {"reward": 1.0}, {"reward": 1.0}, {"reward": 1.0}],
            [{"reward": 0.0}, {"reward": 0.0}, {"reward": 0.0}, {"reward": 0.0}],
            [{"reward": 1.0}, {"reward": 0.0}, {"reward": 1.0}, {"reward": 0.0}],
        ]
        m, _, _, _ = compute_pass_majority_metrics(tasks)

        assert m["pass@1/accuracy"] == pytest.approx(50.0)
        assert m["pass@4/accuracy"] == pytest.approx(200.0 / 3.0, abs=0.01)

    def test_pass_at_1_avg_of_k(self) -> None:
        """Mean of individual scores across k rollouts."""
        tasks = [
            [{"reward": 1.0}, {"reward": 0.0}],
            [{"reward": 0.0}, {"reward": 1.0}],
            [{"reward": 1.0}, {"reward": 1.0}],
        ]
        m, _, _, _ = compute_pass_majority_metrics(tasks)

        assert m["pass@1[avg-of-2]/accuracy"] == pytest.approx(200.0 / 3.0, abs=0.01)

    def test_majority_at_k(self) -> None:
        """Majority voting with extracted_answer."""
        tasks = [
            [
                {"reward": 1.0, "extracted_answer": "A"},
                {"reward": 1.0, "extracted_answer": "A"},
                {"reward": 0.0, "extracted_answer": "B"},
            ],
            [
                {"reward": 0.0, "extracted_answer": "C"},
                {"reward": 0.0, "extracted_answer": "C"},
                {"reward": 1.0, "extracted_answer": "D"},
            ],
        ]
        m, _, _, _ = compute_pass_majority_metrics(tasks, answer_key="extracted_answer")

        assert m["majority@3/accuracy"] == pytest.approx(50.0)

    def test_no_answer(self) -> None:
        """no_answer tracks tasks where all rollouts failed to extract an answer."""
        tasks = [
            [{"reward": 1.0, "extracted_answer": "A"}, {"reward": 0.0, "extracted_answer": "B"}],
            [{"reward": 0.0, "extracted_answer": None}, {"reward": 0.0, "extracted_answer": None}],
        ]
        m, _, _, _ = compute_pass_majority_metrics(tasks, answer_key="extracted_answer")

        # no_answer is a binary score: Task 0 has 0/2, Task 1 has 2/2
        # pass@1[avg-of-2]/no_answer: Task 0: avg(0,0)=0, Task 1: avg(1,1)=1. Mean = 50%
        assert m["pass@1[avg-of-2]/no_answer"] == pytest.approx(50.0)

    def test_std_dev_across_runs(self) -> None:
        """Variance statistics are flat keys matching AIME format."""
        tasks = [
            [{"reward": 1.0}, {"reward": 0.0}],
            [{"reward": 0.0}, {"reward": 0.0}],
            [{"reward": 1.0}, {"reward": 1.0}],
        ]
        m, _, _, _ = compute_pass_majority_metrics(tasks)

        assert m["pass@1[avg-of-2]/accuracy/std_dev_across_runs"] > 0
        assert m["pass@1[avg-of-2]/accuracy/std_err_across_runs"] > 0

    def test_empty_input(self) -> None:
        assert compute_pass_majority_metrics([])[0] == {}

    def test_no_answer_key_skips_majority(self) -> None:
        """Without answer_key, majority@k and no_answer are not computed."""
        tasks = [
            [{"reward": 1.0, "extracted_answer": "A"}, {"reward": 0.0, "extracted_answer": "B"}],
        ]
        m, _, _, _ = compute_pass_majority_metrics(tasks)

        assert not any(k.startswith("majority@") for k in m)
        assert not any("no_answer" in k for k in m)

    def test_multiple_score_methods(self) -> None:
        """Multiple score methods produce separate keys under each agg mode."""
        tasks = [
            [{"reward": 1.0, "library_reward": 1.0}, {"reward": 0.0, "library_reward": 0.0}],
            [{"reward": 0.0, "library_reward": 1.0}, {"reward": 1.0, "library_reward": 1.0}],
        ]

        def score_fn(r):
            return {"accuracy": r["reward"], "symbolic_accuracy": r["library_reward"]}

        m, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=score_fn)

        assert "pass@1/accuracy" in m
        assert "pass@1/symbolic_accuracy" in m
        assert "pass@1[avg-of-2]/accuracy" in m
        assert "pass@1[avg-of-2]/symbolic_accuracy" in m


class TestDefaultAgentAggregateMetrics:
    @pytest.mark.asyncio
    async def test_default_fallback(self) -> None:
        """Base agent uses the same RewardProfiler logic as the resources server."""

        class TestAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=None):
                pass

            async def run(self, body=None):
                pass

        config = BaseResponsesAPIAgentConfig(host="127.0.0.1", port=12345, entrypoint="app.py", name="test_agent")
        agent = TestAgent(config=config, server_client=MagicMock(spec=ServerClient))

        responses = _make_verify_responses(tasks=2, rollouts_per_task=3, reward_fn=lambda t, r: 1.0)
        body = AggregateMetricsRequest(verify_responses=responses)
        result = await agent.aggregate_metrics(body)

        assert isinstance(result, AggregateMetrics)
        assert result.agent_metrics["mean/reward"] == pytest.approx(1.0)
        assert len(result.group_level_metrics) == 2
        assert "mean/reward" in result.key_metrics


class TestTaskIndexInGroupMetrics:
    def test_task_index_preserved(self) -> None:
        responses = [
            {TASK_INDEX_KEY_NAME: 5, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {}},
            {TASK_INDEX_KEY_NAME: 5, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "response": {}},
            {TASK_INDEX_KEY_NAME: 10, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.5, "response": {}},
            {TASK_INDEX_KEY_NAME: 10, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.5, "response": {}},
        ]
        result = compute_aggregate_metrics(responses)

        assert len(result.group_level_metrics) == 2
        indices = [g[TASK_INDEX_KEY_NAME] for g in result.group_level_metrics]
        assert indices == [5, 10]

    def test_non_sequential_indices(self) -> None:
        responses = [
            {TASK_INDEX_KEY_NAME: 100, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {}},
            {TASK_INDEX_KEY_NAME: 200, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0, "response": {}},
            {TASK_INDEX_KEY_NAME: 300, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.5, "response": {}},
        ]
        result = compute_aggregate_metrics(responses)

        indices = [g[TASK_INDEX_KEY_NAME] for g in result.group_level_metrics]
        assert indices == [100, 200, 300]


class TestMajorityNoAnswerCounting:
    """majority@k should count tasks with no valid answers as incorrect (score 0)."""

    def test_no_answer_tasks_count_as_incorrect(self) -> None:
        tasks = [
            # Task 0: all answers present, majority correct
            [{"reward": 1.0, "extracted_answer": "A"}, {"reward": 1.0, "extracted_answer": "A"}],
            # Task 1: no answers at all
            [{"reward": 0.0, "extracted_answer": None}, {"reward": 0.0, "extracted_answer": None}],
        ]
        m, _, _, _ = compute_pass_majority_metrics(tasks, answer_key="extracted_answer")
        # Task 0 correct (100), Task 1 no-answer should be 0 → average = 50
        assert m["majority@2/accuracy"] == pytest.approx(50.0)

    def test_all_no_answer_is_zero(self) -> None:
        tasks = [
            [{"reward": 0.0, "extracted_answer": None}, {"reward": 0.0, "extracted_answer": None}],
            [{"reward": 0.0, "extracted_answer": None}, {"reward": 0.0, "extracted_answer": None}],
        ]
        m, _, _, _ = compute_pass_majority_metrics(tasks, answer_key="extracted_answer")
        assert m["majority@2/accuracy"] == pytest.approx(0.0)


class TestComputeAggregateMetricsPerTask:
    """Test that compute_aggregate_metrics merges per_task_metrics from compute_metrics_fn."""

    def test_per_task_metrics_merged(self) -> None:
        responses = [
            {TASK_INDEX_KEY_NAME: 0, "_ng_rollout_index": 0, "reward": 1.0, "response": {}},
            {TASK_INDEX_KEY_NAME: 1, "_ng_rollout_index": 0, "reward": 0.0, "response": {}},
        ]

        def metrics_fn(tasks):
            return {
                "custom_agg": 99,
                "per_task_metrics": [
                    {TASK_INDEX_KEY_NAME: 0, "difficulty": "easy"},
                    {TASK_INDEX_KEY_NAME: 1, "difficulty": "hard"},
                ],
            }

        result = compute_aggregate_metrics(responses, compute_metrics_fn=metrics_fn)
        assert result.agent_metrics["custom_agg"] == 99
        assert "per_task_metrics" not in result.agent_metrics
        groups_by_idx = {g[TASK_INDEX_KEY_NAME]: g for g in result.group_level_metrics}
        assert groups_by_idx[0]["difficulty"] == "easy"
        assert groups_by_idx[1]["difficulty"] == "hard"


class TestHighestKMetrics:
    def test_basic(self) -> None:
        am = {
            "pass@1/accuracy": 50.0,
            "pass@2/accuracy": 75.0,
            "pass@4/accuracy": 90.0,
            "pass@4/no_answer": 5.0,
            "pass@1[avg-of-4]/accuracy/std_dev_across_runs": 1.5,
        }
        result = highest_k_metrics(am, "pass@{k}", score_names=["accuracy"])
        assert result == {"pass@4/accuracy": 90.0}

    def test_exclude_names(self) -> None:
        am = {"majority@2/accuracy": 80.0, "majority@2/no_answer": 3.0}
        result = highest_k_metrics(am, "majority@{k}", exclude_names=["no_answer"])
        assert result == {"majority@2/accuracy": 80.0}

    def test_avg_of_k(self) -> None:
        am = {
            "pass@1[avg-of-2]/accuracy": 60.0,
            "pass@1[avg-of-4]/accuracy": 65.0,
            "pass@1[avg-of-4]/no_answer": 1.0,
            "pass@1[avg-of-4]/accuracy/std_dev_across_runs": 2.0,
        }
        result = highest_k_metrics(am, "pass@1[avg-of-{k}]")
        assert "pass@1[avg-of-4]/accuracy" in result
        assert "pass@1[avg-of-4]/no_answer" in result
        assert "pass@1[avg-of-4]/accuracy/std_dev_across_runs" not in result

    def test_empty(self) -> None:
        assert highest_k_metrics({}, "pass@{k}") == {}
        assert highest_k_metrics({"unrelated": 1.0}, "pass@{k}") == {}


class TestComputeSubsetMetrics:
    def test_groups_by_field(self) -> None:
        tasks = [
            [{"reward": 1.0, "difficulty": "easy"}, {"reward": 1.0, "difficulty": "easy"}],
            [{"reward": 0.0, "difficulty": "hard"}, {"reward": 0.0, "difficulty": "hard"}],
            [{"reward": 1.0, "difficulty": "easy"}, {"reward": 0.0, "difficulty": "easy"}],
        ]
        m = compute_subset_metrics(tasks, "difficulty")
        assert "easy/pass@1/accuracy" in m
        assert "hard/pass@1/accuracy" in m
        assert m["easy/pass@1/accuracy"] > m["hard/pass@1/accuracy"]
        assert "per_sample_aggregate" not in m

    def test_no_subset_field(self) -> None:
        tasks = [[{"reward": 1.0}, {"reward": 0.0}]]
        m = compute_subset_metrics(tasks, "nonexistent")
        assert m == {}


class TestRepeatLevelMetrics:
    def test_absent_for_single_repeat(self) -> None:
        responses = _make_verify_responses(tasks=4, rollouts_per_task=1)
        result = compute_aggregate_metrics(responses)
        assert result.repeat_level_metrics == []

    def test_present_for_multi_repeat(self) -> None:
        responses = _make_verify_responses(tasks=4, rollouts_per_task=3)
        result = compute_aggregate_metrics(responses)
        assert len(result.repeat_level_metrics) == 3

    def test_repeat_entry_structure(self) -> None:
        responses = _make_verify_responses(tasks=4, rollouts_per_task=2)
        result = compute_aggregate_metrics(responses)
        for i, entry in enumerate(result.repeat_level_metrics):
            assert entry[ROLLOUT_INDEX_KEY_NAME] == i
            assert entry["sample_count"] == 4
            assert entry["missing_count"] == 0
            for stat in ("mean", "median", "std", "sem", "ci_low_95", "ci_high_95", "min", "max", "p25", "p75"):
                assert f"{stat}/reward" in entry, f"{stat}/reward missing from repeat entry"

    def test_known_values(self) -> None:
        """Spot-check mean and std for a controlled 2-repeat case."""
        # repeat 0 rewards: task 0→0, 1→1, 2→0, 3→1  ⟹ mean=0.5, std≈0.577
        responses = [
            {TASK_INDEX_KEY_NAME: t, ROLLOUT_INDEX_KEY_NAME: r, "reward": float((t + r) % 2)}
            for t in range(4)
            for r in range(2)
        ]
        result = compute_aggregate_metrics(responses)
        assert len(result.repeat_level_metrics) == 2
        r0 = next(e for e in result.repeat_level_metrics if e[ROLLOUT_INDEX_KEY_NAME] == 0)
        assert r0["mean/reward"] == pytest.approx(0.5)
        assert r0["std/reward"] == pytest.approx((1 / 3) ** 0.5, rel=1e-3)
        assert r0["sample_count"] == 4

    def test_ci_narrower_with_more_samples(self) -> None:
        """Wider sample set → narrower CI (same reward variance)."""
        small = _make_verify_responses(tasks=3, rollouts_per_task=2)
        large = _make_verify_responses(tasks=20, rollouts_per_task=2)

        def ci_width(metrics):
            return metrics[0]["ci_high_95/reward"] - metrics[0]["ci_low_95/reward"]

        assert ci_width(compute_aggregate_metrics(small).repeat_level_metrics) > ci_width(
            compute_aggregate_metrics(large).repeat_level_metrics
        )

    def test_missing_count(self) -> None:
        """Tasks absent for a rollout_index are reflected in missing_count."""
        responses = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.5},
        ]
        result = compute_aggregate_metrics(responses)
        assert len(result.repeat_level_metrics) == 2
        by_idx = {e[ROLLOUT_INDEX_KEY_NAME]: e for e in result.repeat_level_metrics}
        assert by_idx[0]["sample_count"] == 2
        assert by_idx[0]["missing_count"] == 0
        assert by_idx[1]["sample_count"] == 1
        assert by_idx[1]["missing_count"] == 1

    def test_no_ci_with_one_sample(self) -> None:
        """With only 1 task per repeat, sem is 0.0 (degenerate) and CI is not emitted."""
        responses = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.5},
        ]
        result = compute_aggregate_metrics(responses)
        for entry in result.repeat_level_metrics:
            assert entry["sem/reward"] == pytest.approx(0.0)
            assert "ci_low_95/reward" not in entry
            assert "ci_high_95/reward" not in entry


class TestAggregateRepeatLevelMetrics:
    """Unit tests for RewardProfiler._aggregate_repeat_level_metrics, which collapses the
    per-repeat estimates in repeat_level_metrics (one row per rollout_index) down to a single
    mean/median/se per agent -- how consistent the repeat-level estimates are with each other.
    """

    def test_empty_input(self) -> None:
        assert RewardProfiler()._aggregate_repeat_level_metrics([]) == []

    def test_single_repeat_metric_is_degenerate(self) -> None:
        """A single repeat has nothing to vary across, so se is 0 and mean == median == the value."""
        repeat_level_metrics = [
            {AGENT_REF_KEY_NAME: {"name": "agent"}, ROLLOUT_INDEX_KEY_NAME: 0, "mean/reward": 0.75},
        ]
        result = RewardProfiler()._aggregate_repeat_level_metrics(repeat_level_metrics)

        assert len(result) == 1
        assert result[0][AGENT_REF_KEY_NAME] == {"name": "agent"}
        assert result[0]["mean/mean/reward"] == pytest.approx(0.75)
        assert result[0]["median/mean/reward"] == pytest.approx(0.75)
        assert result[0]["se/mean/reward"] == pytest.approx(0.0)

    def test_known_values_across_repeats(self) -> None:
        """3 repeats with per-repeat means 1, 2, 3 -> mean=2, median=2, se=std([1,2,3])/sqrt(3)."""
        repeat_level_metrics = [
            {AGENT_REF_KEY_NAME: {"name": "agent"}, ROLLOUT_INDEX_KEY_NAME: i, "mean/reward": float(i + 1)}
            for i in range(3)
        ]
        result = RewardProfiler()._aggregate_repeat_level_metrics(repeat_level_metrics)

        assert len(result) == 1
        entry = result[0]
        assert entry["mean/mean/reward"] == pytest.approx(2.0)
        assert entry["median/mean/reward"] == pytest.approx(2.0)
        assert entry["se/mean/reward"] == pytest.approx(1.0 / (3**0.5))

    def test_skewed_repeats_median_differs_from_mean(self) -> None:
        """An outlier repeat pulls the mean away from the median, demonstrating both are useful."""
        repeat_level_metrics = [
            {AGENT_REF_KEY_NAME: {"name": "agent"}, ROLLOUT_INDEX_KEY_NAME: 0, "mean/reward": 0.0},
            {AGENT_REF_KEY_NAME: {"name": "agent"}, ROLLOUT_INDEX_KEY_NAME: 1, "mean/reward": 0.0},
            {AGENT_REF_KEY_NAME: {"name": "agent"}, ROLLOUT_INDEX_KEY_NAME: 2, "mean/reward": 9.0},
        ]
        result = RewardProfiler()._aggregate_repeat_level_metrics(repeat_level_metrics)

        entry = result[0]
        assert entry["median/mean/reward"] == pytest.approx(0.0)
        assert entry["mean/mean/reward"] == pytest.approx(3.0)
        assert entry["mean/mean/reward"] != entry["median/mean/reward"]

    def test_per_agent_grouping(self) -> None:
        """Repeats from different agents are aggregated independently, not pooled together."""
        repeat_level_metrics = [
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, ROLLOUT_INDEX_KEY_NAME: 0, "mean/reward": 0.0},
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, ROLLOUT_INDEX_KEY_NAME: 1, "mean/reward": 2.0},
            {AGENT_REF_KEY_NAME: {"name": "agent_b"}, ROLLOUT_INDEX_KEY_NAME: 0, "mean/reward": 10.0},
            {AGENT_REF_KEY_NAME: {"name": "agent_b"}, ROLLOUT_INDEX_KEY_NAME: 1, "mean/reward": 10.0},
        ]
        result = RewardProfiler()._aggregate_repeat_level_metrics(repeat_level_metrics)

        by_agent = {entry[AGENT_REF_KEY_NAME]["name"]: entry for entry in result}
        assert set(by_agent) == {"agent_a", "agent_b"}
        assert by_agent["agent_a"]["mean/mean/reward"] == pytest.approx(1.0)
        assert by_agent["agent_a"]["se/mean/reward"] > 0
        assert by_agent["agent_b"]["mean/mean/reward"] == pytest.approx(10.0)
        assert by_agent["agent_b"]["se/mean/reward"] == pytest.approx(0.0)

    def test_merged_into_agent_level_metrics_via_compute_aggregate_metrics(self) -> None:
        """End-to-end: the aggregated repeat-level stats land in the public agent_metrics dict,
        without disturbing the original (non-repeat-aggregated) agent_metrics keys.
        """
        # 4 tasks x 3 repeats; each repeat has a different, internally-constant reward so the
        # per-repeat "mean/reward" values are exactly 0.0, 1.0, 2.0.
        responses = [
            {TASK_INDEX_KEY_NAME: t, ROLLOUT_INDEX_KEY_NAME: r, "reward": float(r)} for t in range(4) for r in range(3)
        ]
        result = compute_aggregate_metrics(responses)

        # Original per-rollout agent-level stat is untouched.
        assert result.agent_metrics["mean/reward"] == pytest.approx(1.0)
        # Cross-repeat aggregate of the per-repeat means [0, 1, 2].
        assert result.agent_metrics["mean/mean/reward"] == pytest.approx(1.0)
        assert result.agent_metrics["median/mean/reward"] == pytest.approx(1.0)
        assert result.agent_metrics["se/mean/reward"] == pytest.approx(1.0 / (3**0.5))


class TestAddAvgSampleStdDev:
    def test_adds_stats(self) -> None:
        tasks = [
            [{"reward": 1.0}, {"reward": 0.0}],
            [{"reward": 0.0}, {"reward": 0.0}],
            [{"reward": 1.0}, {"reward": 1.0}],
        ]
        metrics, all_score_dicts, score_names, max_k = compute_pass_majority_metrics(tasks)
        assert "pass@1[avg-of-2]/accuracy/avg_sample_std_dev" not in metrics
        add_avg_sample_std_dev(metrics, all_score_dicts, score_names, max_k)
        assert "pass@1[avg-of-2]/accuracy/avg_sample_std_dev" in metrics
        assert metrics["pass@1[avg-of-2]/accuracy/avg_sample_std_dev"] > 0

    def test_noop_for_k1(self) -> None:
        tasks = [[{"reward": 1.0}], [{"reward": 0.0}]]
        metrics, all_score_dicts, score_names, max_k = compute_pass_majority_metrics(tasks)
        before = dict(metrics)
        add_avg_sample_std_dev(metrics, all_score_dicts, score_names, max_k)
        assert metrics == before
