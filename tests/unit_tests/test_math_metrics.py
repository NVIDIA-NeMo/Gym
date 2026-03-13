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
"""Tests for math_with_judge compute_metrics, get_key_metrics, and supporting functions."""

import sys
from unittest.mock import MagicMock

import pytest


# math_verify is only installed in the math_with_judge server venv, not the main venv.
# Mock it so we can import the module-level functions without the full dependency.
if "math_verify" not in sys.modules:
    sys.modules["math_verify"] = MagicMock()
    sys.modules["math_verify.grader"] = MagicMock()
    sys.modules["math_verify.errors"] = MagicMock()
    sys.modules["math_verify.metric"] = MagicMock()
    sys.modules["math_verify.parser"] = MagicMock()

from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    _compute_majority_at_k,
    _compute_pass_and_avg,
    _compute_per_sample,
    _extract_scores_and_answers,
    _get_score_dict,
)


class TestGetScoreDict:
    def test_basic_reward(self) -> None:
        result = _get_score_dict({"reward": 1.0})
        assert result == {"accuracy": 1.0}

    def test_with_library_reward(self) -> None:
        result = _get_score_dict({"reward": 1.0, "library_reward": 0.5})
        assert result == {"accuracy": 1.0, "symbolic_accuracy": 0.5}

    def test_with_judge(self) -> None:
        result = _get_score_dict({"reward": 1.0, "library_reward": 0.0, "judge_evaluations": [{"v": "A=B"}]})
        assert "judge_accuracy" in result
        assert result["judge_accuracy"] == 1.0

    def test_judge_none_excluded(self) -> None:
        result = _get_score_dict({"reward": 1.0, "library_reward": 1.0, "judge_evaluations": None})
        assert "judge_accuracy" not in result


class TestExtractScoresAndAnswers:
    def test_extracts_answers(self) -> None:
        tasks = [
            [{"reward": 1.0, "extracted_answer": "42"}, {"reward": 0.0, "extracted_answer": "7"}],
            [{"reward": 1.0, "extracted_answer": "5"}],
        ]
        scores, answers = _extract_scores_and_answers(tasks)
        assert len(scores) == 2
        assert answers[0] == ["42", "7"]
        assert answers[1] == ["5"]

    def test_none_answer(self) -> None:
        tasks = [[{"reward": 0.0}]]
        _, answers = _extract_scores_and_answers(tasks)
        assert answers[0] == [None]


class TestComputePassAndAvg:
    def test_pass_at_1(self) -> None:
        scores = [
            [{"accuracy": 1.0}, {"accuracy": 0.0}],
            [{"accuracy": 0.0}, {"accuracy": 1.0}],
        ]
        pass_k, avg_k = _compute_pass_and_avg(scores, ["accuracy"], k=1)
        # pass@1: max([:1]) → [1.0, 0.0] → avg 50%
        assert pass_k["accuracy"] == pytest.approx(50.0)
        assert avg_k["accuracy"] == pytest.approx(50.0)

    def test_pass_at_2(self) -> None:
        scores = [
            [{"accuracy": 0.0}, {"accuracy": 1.0}],
            [{"accuracy": 0.0}, {"accuracy": 0.0}],
        ]
        pass_k, avg_k = _compute_pass_and_avg(scores, ["accuracy"], k=2)
        # pass@2: max([:2]) → [1.0, 0.0] → avg 50%
        assert pass_k["accuracy"] == pytest.approx(50.0)
        # avg-of-2: mean([:2]) → [0.5, 0.0] → avg 25%
        assert avg_k["accuracy"] == pytest.approx(25.0)

    def test_all_correct(self) -> None:
        scores = [[{"accuracy": 1.0}] * 3] * 4
        pass_k, avg_k = _compute_pass_and_avg(scores, ["accuracy"], k=3)
        assert pass_k["accuracy"] == pytest.approx(100.0)
        assert avg_k["accuracy"] == pytest.approx(100.0)


class TestComputeMajorityAtK:
    def test_majority_voting(self) -> None:
        scores = [
            [{"accuracy": 1.0}, {"accuracy": 1.0}, {"accuracy": 0.0}],
        ]
        answers = [["42", "42", "7"]]
        result = _compute_majority_at_k(scores, answers, ["accuracy"], k=3)
        # Majority is "42" (2 votes), score = 1.0 → 100%
        assert result["accuracy"] == pytest.approx(100.0)

    def test_none_answers_excluded(self) -> None:
        scores = [
            [{"accuracy": 1.0}, {"accuracy": 0.0}, {"accuracy": 0.0}],
        ]
        answers = [["42", None, "7"]]
        result = _compute_majority_at_k(scores, answers, ["accuracy"], k=3)
        # "42" and "7" each have 1 vote, first most_common wins → "42" → score 1.0
        assert result["accuracy"] == pytest.approx(100.0)

    def test_all_none(self) -> None:
        scores = [[{"accuracy": 0.0}, {"accuracy": 0.0}]]
        answers = [[None, None]]
        result = _compute_majority_at_k(scores, answers, ["accuracy"], k=2)
        assert result == {}


class TestComputePerSample:
    def test_per_sample_values(self) -> None:
        # 3 tasks, 2 rollouts each
        scores = [
            [{"accuracy": 1.0}, {"accuracy": 0.0}],
            [{"accuracy": 0.0}, {"accuracy": 1.0}],
            [{"accuracy": 1.0}, {"accuracy": 1.0}],
        ]
        result = _compute_per_sample(scores, ["accuracy"], k=2)
        # rollout 0: [1, 0, 1] → 66.67%
        assert result["accuracy"][0] == pytest.approx(200.0 / 3.0, abs=0.01)
        # rollout 1: [0, 1, 1] → 66.67%
        assert result["accuracy"][1] == pytest.approx(200.0 / 3.0, abs=0.01)


class TestComputeMetricsIntegration:
    """Test the full compute_metrics method on LibraryJudgeMathResourcesServer."""

    def _make_tasks(self):
        """3 tasks × 4 rollouts with varying correctness and some no_answer."""
        return [
            # Task 0: 3 correct, 1 no_answer
            [
                {"reward": 1.0, "library_reward": 1.0, "extracted_answer": "204"},
                {"reward": 1.0, "library_reward": 1.0, "extracted_answer": "204"},
                {"reward": 1.0, "library_reward": 1.0, "extracted_answer": "204"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": None},
            ],
            # Task 1: 1 correct, 1 wrong, 2 no_answer
            [
                {"reward": 1.0, "library_reward": 1.0, "extracted_answer": "113"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": "42"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": None},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": None},
            ],
            # Task 2: all wrong, 1 no_answer
            [
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": "99"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": "42"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": "7"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": None},
            ],
        ]

    def test_pass_at_k(self) -> None:
        tasks = self._make_tasks()
        result = LibraryJudgeMathResourcesServer.compute_metrics(None, tasks)

        # pass@1: max([:1]) per task → [1.0, 1.0, 0.0] → 66.67%
        assert result["pass@1/accuracy"] == pytest.approx(200.0 / 3.0, abs=0.01)
        # pass@4: max([:4]) per task → [1.0, 1.0, 0.0] → 66.67%
        assert result["pass@4/accuracy"] == pytest.approx(200.0 / 3.0, abs=0.01)

    def test_majority_at_k(self) -> None:
        tasks = self._make_tasks()
        result = LibraryJudgeMathResourcesServer.compute_metrics(None, tasks)

        # majority@4 exists (answers are present)
        assert "majority@4/accuracy" in result

    def test_per_sample_aggregate(self) -> None:
        tasks = self._make_tasks()
        result = LibraryJudgeMathResourcesServer.compute_metrics(None, tasks)

        psa = result["per_sample_aggregate"]
        assert "accuracy" in psa
        assert len(psa["accuracy"]) == 4  # 4 rollouts

    def test_no_answer_tracking(self) -> None:
        tasks = self._make_tasks()
        result = LibraryJudgeMathResourcesServer.compute_metrics(None, tasks)

        # no_answer is top-level, not under pass@1[avg-of-k]
        assert "no_answer" in result
        assert not any(k.startswith("pass@1[avg-of-") and "no_answer" in k for k in result)

        # Per-sample no_answer in per_sample_aggregate
        psa = result["per_sample_aggregate"]
        assert "no_answer" in psa
        assert len(psa["no_answer"]) == 4

        # rollout 0: task 0 has answer, task 1 has answer, task 2 has answer → 0% no_answer
        assert psa["no_answer"][0] == pytest.approx(0.0)
        # rollout 3: task 0 None, task 1 None, task 2 None → 100% no_answer
        assert psa["no_answer"][3] == pytest.approx(100.0)

    def test_no_answer_stats(self) -> None:
        tasks = self._make_tasks()
        result = LibraryJudgeMathResourcesServer.compute_metrics(None, tasks)

        assert "no_answer/std_dev_across_runs" in result
        assert "no_answer/std_err_across_runs" in result
        assert result["no_answer/std_dev_across_runs"] > 0

    def test_stat_key_separator(self) -> None:
        tasks = self._make_tasks()
        result = LibraryJudgeMathResourcesServer.compute_metrics(None, tasks)

        # Stats should use / separator, not underscore
        stat_keys = [k for k in result if "std_dev_across_runs" in k]
        for k in stat_keys:
            assert "/std_dev_across_runs" in k, f"Expected / separator in {k}"

    def test_multi_score(self) -> None:
        tasks = self._make_tasks()
        result = LibraryJudgeMathResourcesServer.compute_metrics(None, tasks)

        assert "pass@1/symbolic_accuracy" in result
        assert "pass@1/accuracy" in result


class TestGetKeyMetrics:
    def test_selects_headlines(self) -> None:
        agent_metrics = {
            "mean/reward": 0.5,
            "mean/library_reward": 0.5,
            "mean/input_tokens": 100.0,
            "pass@1/accuracy": 50.0,
            "pass@1[avg-of-1]/accuracy": 50.0,
            "pass@1[avg-of-4]/accuracy": 45.0,
            "pass@1[avg-of-4]/accuracy/std_dev_across_runs": 3.0,
            "pass@4/accuracy": 70.0,
            "majority@4/accuracy": 60.0,
            "no_answer": 10.0,
        }
        result = LibraryJudgeMathResourcesServer.get_key_metrics(None, agent_metrics)

        assert "mean/reward" in result
        assert "pass@1[avg-of-4]/accuracy" in result
        assert "pass@4/accuracy" in result
        assert "majority@4/accuracy" in result
        # Stats should NOT be in key metrics
        assert "pass@1[avg-of-4]/accuracy/std_dev_across_runs" not in result


class TestTaskIndexInGroupMetrics:
    def test_task_index_preserved(self) -> None:
        from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
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
        from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
        from nemo_gym.reward_profile import compute_aggregate_metrics

        responses = [
            {TASK_INDEX_KEY_NAME: 100, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {}},
            {TASK_INDEX_KEY_NAME: 200, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0, "response": {}},
            {TASK_INDEX_KEY_NAME: 300, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.5, "response": {}},
        ]
        result = compute_aggregate_metrics(responses)

        indices = [g[TASK_INDEX_KEY_NAME] for g in result.group_level_metrics]
        assert indices == [100, 200, 300]


class TestBenchmarkConfigLoading:
    def test_benchmark_config_loads_defaults(self) -> None:
        """Benchmark config keys are loaded and merged as lowest-priority defaults."""

        from omegaconf import OmegaConf

        from nemo_gym import PARENT_DIR
        from nemo_gym.global_config import CONFIG_PATHS_KEY_NAME

        benchmark_name = "aime24"
        benchmark_config_path = PARENT_DIR / "benchmarks" / benchmark_name / "config.yaml"
        assert benchmark_config_path.exists(), f"AIME24 benchmark config not found at {benchmark_config_path}"

        benchmark_config = OmegaConf.load(benchmark_config_path)

        # Verify expected keys exist in benchmark config
        assert benchmark_config.get("agent_name") == "math_with_judge_simple_agent"
        assert "aime24_validation" in benchmark_config.get("input_jsonl_fpath", "")
        assert benchmark_config.get("prompt_config") is not None
        assert CONFIG_PATHS_KEY_NAME in benchmark_config

    def test_benchmark_not_found_raises(self) -> None:
        from omegaconf import DictConfig

        from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

        parser = GlobalConfigDictParser()
        with pytest.raises(FileNotFoundError, match="Benchmark config not found"):
            parser.parse(
                GlobalConfigDictParserConfig(
                    initial_global_config_dict=DictConfig({"benchmark": "nonexistent_benchmark"}),
                    skip_load_from_cli=True,
                    skip_load_from_dotenv=True,
                )
            )
