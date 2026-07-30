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
"""Build Archipelago grading verifiers from an Apex task rubric.

Mirrors ``examples/hugging_face_task/main.py`` in the Archipelago repo: each
rubric criterion becomes one ``output_llm`` verifier, and the first criterion is
flagged as the task's primary objective (weighted more heavily by the scorer).
"""

from __future__ import annotations


def build_verifiers(rubric: list[dict], world_id: str, task_id: str) -> list[dict]:
    """Convert a rubric (list of ``{verifier_id?, criteria}``) into verifier configs."""
    return [
        {
            "verifier_id": c.get("verifier_id", f"v{i}"),
            "verifier_version": 1,
            "world_id": world_id,
            "task_id": task_id,
            "eval_config_id": "ec_output_llm",
            "verifier_values": {"criteria": c["criteria"], "is_primary_objective": i == 0},
            "verifier_index": i,
            "verifier_dependencies": None,
        }
        for i, c in enumerate(rubric)
    ]
