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


import pytest

from nemo_gym.reward_profile import RewardProfiler


class TestRewardProfile:
    def _clean_metrics(self, metrics: list[dict]) -> None:
        for row in metrics:
            for key in list(row):
                if key.startswith("histogram"):
                    row[key] = None
                elif key in {
                    "_ng_task_index",
                    "expected_num_rollouts",
                    "missing_num_rollouts",
                    "num_rollouts",
                    "reward_profile_completion_pct",
                    "rollout_infos",
                }:
                    row.pop(key)

    def _row(self, task_idx: int, rollout_idx: int) -> dict:
        return {
            "_ng_task_index": task_idx,
            "_ng_rollout_index": rollout_idx,
            "responses_create_params": {"input": []},
            "agent_ref": {"name": "my_agent"},
            "task": task_idx,
        }

    def _result(self, task_idx: int, rollout_idx: int, reward: float = 1.0, total_tokens: int = 7) -> dict:
        return {
            "_ng_task_index": task_idx,
            "_ng_rollout_index": rollout_idx,
            "response": {"usage": {"total_tokens": total_tokens}},
            "reward": reward,
        }

    def test_profile_from_data(self) -> None:
        rows = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 1}'},
                    "temperature": 0.1,
                },
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 1}'},
                    "temperature": 0.1,
                },
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 1}'},
                    "temperature": 0.1,
                },
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
        ]
        results = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "reward": 0,
                "bool": True,
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "reward": 1,
                "bool": False,
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "reward": 0,
                "bool": True,
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "reward": 1,
                "bool": False,
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "reward": 0,
                "bool": True,
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "reward": 1,
                "bool": False,
            },
        ]

        actual_group_level_metrics, actual_agent_level_metrics = RewardProfiler().profile_from_data(rows, results)

        self._clean_metrics(actual_group_level_metrics)
        self._clean_metrics(actual_agent_level_metrics)

        expected_group_level_metrics = [
            {
                "mean/bool": 0.5,
                "mean/reward": 0.5,
                "mean/abc usage": 1.0,
                "max/bool": True,
                "max/reward": 1,
                "max/abc usage": 1,
                "min/bool": False,
                "min/reward": 0,
                "min/abc usage": 1,
                "median/bool": 0.5,
                "median/reward": 0.5,
                "median/abc usage": 1.0,
                "std/bool": 0.7071067811865476,
                "std/reward": 0.7071067811865476,
                "std/abc usage": 0.0,
                "histogram/bool": None,
                "histogram/reward": None,
                "histogram/abc usage": None,
                "sample": {
                    "responses_create_params": {
                        "input": [],
                        "metadata": {"extra_body": '{"seed": 0}'},
                        "temperature": 0.1,
                    },
                    "x": 0,
                    "agent_ref": {"name": "my_agent"},
                },
            },
            {
                "mean/bool": 0.5,
                "mean/reward": 0.5,
                "mean/abc usage": 1.0,
                "max/bool": True,
                "max/reward": 1,
                "max/abc usage": 1,
                "min/bool": False,
                "min/reward": 0,
                "min/abc usage": 1,
                "median/bool": 0.5,
                "median/reward": 0.5,
                "median/abc usage": 1.0,
                "std/bool": 0.7071067811865476,
                "std/reward": 0.7071067811865476,
                "std/abc usage": 0.0,
                "histogram/bool": None,
                "histogram/reward": None,
                "histogram/abc usage": None,
                "sample": {
                    "responses_create_params": {
                        "input": [],
                        "metadata": {"extra_body": '{"seed": 0}'},
                        "temperature": 0.1,
                    },
                    "x": 1,
                    "agent_ref": {"name": "my_agent"},
                },
            },
            {
                "mean/bool": 0.5,
                "mean/reward": 0.5,
                "mean/abc usage": 1.0,
                "max/bool": True,
                "max/reward": 1,
                "max/abc usage": 1,
                "min/bool": False,
                "min/reward": 0,
                "min/abc usage": 1,
                "median/bool": 0.5,
                "median/reward": 0.5,
                "median/abc usage": 1.0,
                "std/bool": 0.7071067811865476,
                "std/reward": 0.7071067811865476,
                "std/abc usage": 0.0,
                "histogram/bool": None,
                "histogram/reward": None,
                "histogram/abc usage": None,
                "sample": {
                    "responses_create_params": {
                        "input": [],
                        "metadata": {"extra_body": '{"seed": 0}'},
                        "temperature": 0.1,
                    },
                    "x": 2,
                    "agent_ref": {"name": "my_agent"},
                },
            },
        ]
        assert expected_group_level_metrics == actual_group_level_metrics

        expected_agent_level_metrics = [
            {
                "agent_ref": {"name": "my_agent"},
                "mean/bool": 0.5,
                "mean/reward": 0.5,
                "mean/abc usage": 1.0,
                "max/bool": True,
                "max/reward": 1,
                "max/abc usage": 1,
                "min/bool": False,
                "min/reward": 0,
                "min/abc usage": 1,
                "median/bool": 0.5,
                "median/reward": 0.5,
                "median/abc usage": 1.0,
                "std/bool": 0.5477225575051661,
                "std/reward": 0.5477225575051661,
                "std/abc usage": 0.0,
                "histogram/bool": None,
                "histogram/reward": None,
                "histogram/abc usage": None,
            }
        ]
        assert expected_agent_level_metrics == actual_agent_level_metrics

    def test_profile_from_data_series(self) -> None:
        rows = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "agent_ref": {"name": "my_agent"},
            },
        ]
        results = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
            },
        ]

        # We just check that this doesn't error
        RewardProfiler().profile_from_data(rows, results)

    def test_rollout_infos_are_sorted_and_pass_rate_is_recoverable(self) -> None:
        rows = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": []},
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": []},
                "agent_ref": {"name": "my_agent"},
            },
        ]
        results = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "response": {"usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}},
                "reward": 1.0,
                "verifier_score": 3.5,
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "response": {"usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}},
                "reward": 0.0,
                "verifier_score": 1.5,
            },
        ]

        group_level_metrics, _ = RewardProfiler().profile_from_data(rows, results)
        row = RewardProfiler().prepare_for_serialization(group_level_metrics)[0]

        assert row["_ng_task_index"] == 0
        assert row["num_rollouts"] == 2
        assert row["rollout_infos"] == [
            {
                "rollout_id": "0:0",
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "reward": 0.0,
                "input_tokens": 3,
                "output_tokens": 4,
                "total_tokens": 7,
                "verifier_score": 1.5,
            },
            {
                "rollout_id": "0:1",
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "reward": 1.0,
                "input_tokens": 5,
                "output_tokens": 7,
                "total_tokens": 12,
                "verifier_score": 3.5,
            },
        ]
        assert row["mean/input_tokens"] == 4.0
        assert row["mean/verifier_score"] == 2.5

    def test_profile_from_data_missing_rollouts_requires_partial_flag(self) -> None:
        rows = [self._row(0, 0), self._row(0, 1)]
        results = [self._result(0, 0)]

        with pytest.raises(ValueError, match=r"\+\+allow_partial_rollouts=True"):
            RewardProfiler().profile_from_data(rows, results)

    def test_profile_from_data_allow_partial_profiles_completed_rollouts(self) -> None:
        rows = [self._row(task_idx, rollout_idx) for task_idx in range(3) for rollout_idx in range(2)]
        results = [self._result(0, 0, reward=0.0, total_tokens=5), self._result(0, 1), self._result(1, 0)]

        profiler = RewardProfiler()
        group_level_metrics, _ = profiler.profile_from_data(rows, results, allow_partial_rollouts=True)
        profile_rows = profiler.prepare_for_serialization(group_level_metrics)
        summary = profiler.profile_completion_summary(rows, results)

        assert [row["_ng_task_index"] for row in profile_rows] == [0, 1]
        assert [
            (row["num_rollouts"], row["expected_num_rollouts"], row["missing_num_rollouts"]) for row in profile_rows
        ] == [(2, 2, 0), (1, 2, 1)]
        assert [row["reward_profile_completion_pct"] for row in profile_rows] == [100.0, 50.0]

        assert summary == {
            "expected_rollout_rows": 6,
            "completed_rollout_rows": 3,
            "missing_rollout_rows": 3,
            "extra_rollout_rows": 0,
            "reward_profile_completion_pct": 50.0,
            "total_input_rows": 3,
            "complete_input_rows": 1,
            "partial_input_rows": 1,
            "missing_input_rows": 1,
        }

    def test_profile_from_data_allow_partial_rejects_extra_rollout_rows(self) -> None:
        rows = [self._row(0, 0)]
        results = [self._result(0, 0), self._result(1, 0)]

        with pytest.raises(ValueError, match="no matching materialized input"):
            RewardProfiler().profile_from_data(rows, results, allow_partial_rollouts=True)

    def test_profile_from_data_mismatched_keys(self) -> None:
        rows = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "agent_ref": {"name": "my_agent"},
            },
        ]
        results = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "first_col": 1,
            },
            {"_ng_task_index": 1, "_ng_rollout_index": 0, "response": {"usage": {"abc usage": 1}}, "second_col": 2},
        ]

        actual_group_level_metrics, actual_agent_level_metrics = RewardProfiler().profile_from_data(rows, results)

        self._clean_metrics(actual_group_level_metrics)
        self._clean_metrics(actual_agent_level_metrics)

        expected_group_level_metrics = [
            {
                "mean/first_col": 1.0,
                "mean/abc usage": 1.0,
                "max/first_col": 1.0,
                "max/abc usage": 1.0,
                "min/first_col": 1.0,
                "min/abc usage": 1.0,
                "median/first_col": 1.0,
                "median/abc usage": 1.0,
                "std/first_col": 0.0,
                "std/abc usage": 0.0,
                "histogram/first_col": None,
                "histogram/abc usage": None,
                "sample": {
                    "responses_create_params": {
                        "input": [],
                        "metadata": {"extra_body": '{"seed": 0}'},
                        "temperature": 0.1,
                    },
                    "agent_ref": {"name": "my_agent"},
                },
            },
            {
                "mean/abc usage": 1.0,
                "mean/second_col": 2.0,
                "max/abc usage": 1.0,
                "max/second_col": 2.0,
                "min/abc usage": 1.0,
                "min/second_col": 2.0,
                "median/abc usage": 1.0,
                "median/second_col": 2.0,
                "std/abc usage": 0.0,
                "std/second_col": 0.0,
                "histogram/abc usage": None,
                "histogram/second_col": None,
                "sample": {
                    "responses_create_params": {
                        "input": [],
                        "metadata": {"extra_body": '{"seed": 0}'},
                        "temperature": 0.1,
                    },
                    "agent_ref": {"name": "my_agent"},
                },
            },
        ]
        assert expected_group_level_metrics == actual_group_level_metrics

        expected_agent_level_metrics = [
            {
                "mean/first_col": 1.0,
                "mean/abc usage": 1.0,
                "mean/second_col": 2.0,
                "max/first_col": 1.0,
                "max/abc usage": 1.0,
                "max/second_col": 2.0,
                "min/first_col": 1.0,
                "min/abc usage": 1.0,
                "min/second_col": 2.0,
                "median/first_col": 1.0,
                "median/abc usage": 1.0,
                "median/second_col": 2.0,
                "std/abc usage": 0.0,
                "histogram/first_col": None,
                "histogram/abc usage": None,
                "histogram/second_col": None,
                "agent_ref": {"name": "my_agent"},
            }
        ]
        assert expected_agent_level_metrics == actual_agent_level_metrics
