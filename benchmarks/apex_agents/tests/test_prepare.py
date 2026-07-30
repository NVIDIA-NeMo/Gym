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
from benchmarks.apex_agents.prepare import convert_task


def test_convert_task_builds_gym_row():
    task = {
        "task_id": "task_abc",
        "world_id": "world_1",
        "domain": "investment_banking",
        "task_name": "Accretion/Dilution",
        "prompt": "Build the model.",
        "rubric": [{"verifier_id": "v1", "criteria": "Model is correct"}],
        "task_input_files": ["/filesystem/in.xlsx"],
    }
    worlds = {"world_1": {"world_id": "world_1", "world_name": "World 1"}}
    row = convert_task(task, worlds)
    assert row["responses_create_params"] == {"input": []}
    assert row["task_id"] == "task_abc"
    assert row["world_id"] == "world_1"
    assert row["world_name"] == "World 1"
    assert row["domain"] == "investment_banking"
    assert row["prompt"] == "Build the model."
    assert row["rubric"] == [{"verifier_id": "v1", "criteria": "Model is correct"}]
    assert row["has_task_input_files"] is True


def test_convert_task_defaults_for_missing_optionals():
    task = {"task_id": "t", "world_id": "missing", "prompt": "p"}
    row = convert_task(task, worlds={})
    assert row["world_name"] == ""
    assert row["domain"] == ""
    assert row["task_name"] == ""
    assert row["rubric"] == []
    assert row["has_task_input_files"] is False
