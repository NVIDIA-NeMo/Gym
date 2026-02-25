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


from nemo_gym.profile import RewardProfiler


class TestProfile:
    def _clean_metrics(self, metrics: list[dict]) -> None:
        for row in metrics:
            for key in list(row):
                if key.startswith("histogram"):
                    row[key] = None

    def test_profile_from_data(self) -> None:
        rows = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1},
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1},
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1},
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
                    "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
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
                    "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
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
                    "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
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
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
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

    def test_profile_from_data_mismatched_keys(self) -> None:
        rows = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
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
                    "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
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
                    "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
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
