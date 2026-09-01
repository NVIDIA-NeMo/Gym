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

import json

import numpy as np
from pytest import raises

from resources_servers.arena.scripts.compute_rollout_scores import load_rollout_tasks
from resources_servers.arena.scripts.fit_anchored_elo import build_observations


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_saved_rollout_benchmark_must_match_selected_version(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    _write_jsonl(path, [{"_ng_task_index": 0, "question_id": "q1", "category": "lmarena_v2"}])

    with raises(ValueError, match="--version is 'lmarena_v3'"):
        load_rollout_tasks(path, "lmarena_v3", {})


def test_unmatched_prompt_ids_require_explicit_partial_mode(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    _write_jsonl(
        path,
        [
            {"_ng_task_index": 0, "question_id": "q1", "category": "lmarena_v3"},
            {"_ng_task_index": 1, "question_id": "missing", "category": "lmarena_v3"},
        ],
    )
    prompts = {
        "q1": {
            "question_id": "q1",
            "category": "lmarena_v3",
            "metadata": {},
            "responses_create_params": {"input": []},
            "style_reference_token_count": 100,
            "is_lmarena_v2_prompt": False,
        }
    }

    with raises(ValueError, match="absent from the selected prompt file"):
        load_rollout_tasks(path, "lmarena_v3", prompts)
    assert len(load_rollout_tasks(path, "lmarena_v3", prompts, allow_unmatched_prompts=True)) == 1


def test_anchored_elo_requires_two_parsed_judge_games():
    dataset = {"q1": {"question_id": "q1", "baseline_model": "opponent"}}
    verdicts = {"q1": {"games": [{"verdict": "[[A>>B]]"}, {"verdict": None}]}}

    observations = build_observations(dataset, verdicts, {"opponent": 1400}, "judge", "lmarena_v3")

    assert observations[2] == 0
    assert observations[4] == 1


def test_anchored_elo_both_bad_depends_on_version():
    dataset = {"q1": {"question_id": "q1", "baseline_model": "opponent"}}
    verdicts = {"q1": {"games": [{"verdict": "[[BB]]"}, {"verdict": "[[BB]]"}]}}

    v2 = build_observations(dataset, verdicts, {"opponent": 1400}, "judge", "lmarena_v2")
    v3 = build_observations(dataset, verdicts, {"opponent": 1400}, "judge", "lmarena_v3")

    assert v2[2] == 0
    assert v2[4] == 1
    assert v3[2] == 1
    np.testing.assert_array_equal(v3[1], [0.5, 0.5])
