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
"""Row construction for the TimeWarp benchmark (network-free)."""

import json
from pathlib import Path

import pytest

from benchmarks.timewarp.prepare import TOOLS, UI_VERSIONS, build_rows, in_split
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config


REPO_ROOT = Path(__file__).parents[3]


def _task(task_id: int, **overrides) -> dict:
    task = {
        "task_id": task_id,
        "intent": f"Question {task_id}?",
        "intent_template_id": task_id,
        "sites": ["wiki", "webshop"],
        "start_url": "__WEBSHOP__",
        "eval": {
            "eval_types": ["string_match"],
            "reference_answers": {"must_include": ["zanzibar"], "fuzzy_match": "Zanzibar"},
            "revision": 1,
            "annotation": {"confidence": "high", "notes": "reviewer notes"},
        },
    }
    return task | overrides


def test_every_goal_runs_in_all_six_ui_versions():
    rows = build_rows([_task(2), _task(1)])
    assert [(row["ui_version"], row["task_id"]) for row in rows] == [
        (version, task_id) for version in UI_VERSIONS for task_id in (1, 2)
    ]


def test_row_carries_the_task_contract_and_drops_upstream_only_fields():
    plan = "1. Search for X. 2. Send a message to the user: 'Zanzibar'."
    (row,) = build_rows([_task(150, additional_instructions=plan)], ui_versions=[4])
    assert row["start_site"] == "webshop"
    assert row["sites"] == ["wiki", "webshop"]
    assert row["verifier_metadata"] == {
        "eval_types": ["string_match"],
        "reference_answers": {"must_include": ["zanzibar"], "fuzzy_match": "Zanzibar"},
        "revision": 1,
    }
    assert row["responses_create_params"] == {"tools": TOOLS}
    assert "Search for X" not in json.dumps(row)


def test_unknown_start_url_is_rejected():
    with pytest.raises(ValueError, match="unexpected start_url"):
        build_rows([_task(1, start_url="http://example.com")])


@pytest.mark.parametrize("task_id, split", [(1, "test"), (103, "test"), (104, "train"), (231, "train")])
def test_split_boundary(task_id, split):
    assert in_split({"task_id": task_id}, split)
    assert not in_split({"task_id": task_id}, "train" if split == "test" else "test")


def test_prompt_puts_the_goal_in_the_user_message_and_keeps_the_tools():
    prompt = load_prompt_config(str(REPO_ROOT / "benchmarks" / "timewarp" / "prompt.yaml"))
    (row,) = build_rows([_task(7)], ui_versions=[1])
    materialized = apply_prompt_to_row(row, prompt)
    system, user = materialized["responses_create_params"]["input"]
    assert user == {"role": "user", "content": "Question 7?"}
    assert "observe" in system["content"]
    assert "zanzibar" not in json.dumps(materialized["responses_create_params"]).lower()
    assert materialized["responses_create_params"]["tools"] == TOOLS
