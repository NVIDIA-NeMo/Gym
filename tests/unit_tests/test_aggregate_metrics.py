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
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.reward_profile import compute_pass_majority_metrics
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
        m = compute_pass_majority_metrics(tasks)

        assert m["pass@1/accuracy"] == pytest.approx(50.0)
        assert m["pass@4/accuracy"] == pytest.approx(200.0 / 3.0, abs=0.01)

    def test_pass_at_1_avg_of_k(self) -> None:
        """Mean of individual scores across k rollouts."""
        tasks = [
            [{"reward": 1.0}, {"reward": 0.0}],
            [{"reward": 0.0}, {"reward": 1.0}],
            [{"reward": 1.0}, {"reward": 1.0}],
        ]
        m = compute_pass_majority_metrics(tasks)

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
        m = compute_pass_majority_metrics(tasks, answer_key="extracted_answer")

        assert m["majority@3/accuracy"] == pytest.approx(50.0)

    def test_no_answer(self) -> None:
        """no_answer tracks tasks where all rollouts failed to extract an answer."""
        tasks = [
            [{"reward": 1.0, "extracted_answer": "A"}, {"reward": 0.0, "extracted_answer": "B"}],
            [{"reward": 0.0, "extracted_answer": None}, {"reward": 0.0, "extracted_answer": None}],
        ]
        m = compute_pass_majority_metrics(tasks, answer_key="extracted_answer")

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
        m = compute_pass_majority_metrics(tasks)

        assert m["pass@1[avg-of-2]/accuracy/std_dev_across_runs"] > 0
        assert m["pass@1[avg-of-2]/accuracy/std_err_across_runs"] > 0

    def test_empty_input(self) -> None:
        assert compute_pass_majority_metrics([]) == {}

    def test_no_answer_key_skips_majority(self) -> None:
        """Without answer_key, majority@k and no_answer are not computed."""
        tasks = [
            [{"reward": 1.0, "extracted_answer": "A"}, {"reward": 0.0, "extracted_answer": "B"}],
        ]
        m = compute_pass_majority_metrics(tasks)

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

        m = compute_pass_majority_metrics(tasks, score_fn=score_fn)

        assert "pass@1/accuracy" in m
        assert "pass@1/symbolic_accuracy" in m
        assert "pass@1[avg-of-2]/accuracy" in m
        assert "pass@1[avg-of-2]/symbolic_accuracy" in m


_has_math_verify = True
try:
    import math_verify  # noqa: F401
except ImportError:
    _has_math_verify = False


class TestMathWithJudgeComputeMetrics:
    @pytest.mark.asyncio
    @pytest.mark.skipif(not _has_math_verify, reason="math_verify not installed")
    async def test_math_server_produces_symbolic_and_judge_accuracy(self) -> None:
        from resources_servers.math_with_judge.app import (
            LibraryJudgeMathResourcesServer,
            LibraryJudgeMathResourcesServerConfig,
        )

        config = LibraryJudgeMathResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="math_with_judge",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            judge_responses_create_params={"model": "m", "input": []},
        )
        server = LibraryJudgeMathResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        responses = [
            # Task 0, rollout 0: symbolic correct, judge correct
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 1.0,
                "library_reward": 1.0,
                "judge_evaluations": [{"verdict": "A=B"}],
                "extracted_answer": "42",
            },
            # Task 0, rollout 1: symbolic wrong, judge wrong
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 1,
                "reward": 0.0,
                "library_reward": 0.0,
                "judge_evaluations": [{"verdict": "A!=B"}],
                "extracted_answer": "43",
            },
            # Task 1, rollout 0: symbolic correct, no judge
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 1.0,
                "library_reward": 1.0,
                "judge_evaluations": None,
                "extracted_answer": "7",
            },
            # Task 1, rollout 1: symbolic wrong, no judge
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 1,
                "reward": 0.0,
                "library_reward": 0.0,
                "judge_evaluations": None,
                "extracted_answer": "8",
            },
        ]
        body = AggregateMetricsRequest(verify_responses=responses)
        result = await server.aggregate_metrics(body)
        am = result.agent_metrics

        # Should have symbolic_accuracy for all agg modes
        assert "pass@1/symbolic_accuracy" in am
        assert "pass@1[avg-of-2]/symbolic_accuracy" in am
        assert "majority@2/symbolic_accuracy" in am

        # judge_accuracy only for task 0 (task 1 has judge_evaluations=None)
        assert "pass@1/judge_accuracy" in am

        # Variance stats at k=2
        assert "pass@1[avg-of-2]/symbolic_accuracy/std_dev_across_runs" in am

        # Key metrics should include pass@, majority@, token counts
        assert "pass@2/symbolic_accuracy" in result.key_metrics
        assert "majority@2/symbolic_accuracy" in result.key_metrics


class TestDefaultAgentAggregateMetrics:
    @pytest.mark.asyncio
    async def test_default_fallback(self) -> None:
        """Base agent uses the same RewardProfiler logic as the resources server."""
        from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent

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
        from nemo_gym.reward_profile import compute_aggregate_metrics

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
        from nemo_gym.reward_profile import compute_aggregate_metrics

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
        m = compute_pass_majority_metrics(tasks, answer_key="extracted_answer")
        # Task 0 correct (100), Task 1 no-answer should be 0 → average = 50
        assert m["majority@2/accuracy"] == pytest.approx(50.0)

    def test_all_no_answer_is_zero(self) -> None:
        tasks = [
            [{"reward": 0.0, "extracted_answer": None}, {"reward": 0.0, "extracted_answer": None}],
            [{"reward": 0.0, "extracted_answer": None}, {"reward": 0.0, "extracted_answer": None}],
        ]
        m = compute_pass_majority_metrics(tasks, answer_key="extracted_answer")
        assert m["majority@2/accuracy"] == pytest.approx(0.0)


class TestComputeAggregateMetricsPerTask:
    """Test that compute_aggregate_metrics merges per_task_metrics from compute_metrics_fn."""

    def test_per_task_metrics_merged(self) -> None:
        from nemo_gym.global_config import TASK_INDEX_KEY_NAME
        from nemo_gym.reward_profile import compute_aggregate_metrics

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
