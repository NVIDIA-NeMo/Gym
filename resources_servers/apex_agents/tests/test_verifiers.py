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
from resources_servers.apex_agents.verifiers import build_verifiers


def test_build_verifiers_marks_first_primary():
    rubric = [{"verifier_id": "a", "criteria": "C1"}, {"verifier_id": "b", "criteria": "C2"}]
    out = build_verifiers(rubric, world_id="w", task_id="t")
    assert len(out) == 2
    assert out[0]["eval_config_id"] == "ec_output_llm"
    assert out[0]["verifier_values"] == {"criteria": "C1", "is_primary_objective": True}
    assert out[1]["verifier_values"]["is_primary_objective"] is False
    assert out[0]["world_id"] == "w" and out[0]["task_id"] == "t"
    assert out[1]["verifier_index"] == 1


def test_build_verifiers_generates_id_when_missing():
    out = build_verifiers([{"criteria": "only"}], world_id="w", task_id="t")
    assert out[0]["verifier_id"] == "v0"
    assert out[0]["verifier_dependencies"] is None


def test_build_verifiers_empty_rubric():
    assert build_verifiers([], "w", "t") == []
