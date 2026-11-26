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
import sys

import numpy as np
from pytest import mark, raises

from resources_servers.arena.scripts import compute_rollout_scores
from resources_servers.arena.scripts.compute_rollout_scores import load_rollout_tasks
from resources_servers.arena.scripts.count_rollout_tokens import count_rollout_tokens
from resources_servers.arena.scripts.fit_anchored_elo import build_observations
from resources_servers.arena.scripts.remove_failed_rollouts import is_failed_rollout


@mark.parametrize("category", ["vision_arena_v1", "lmarena_v3", "lmarena_v4", "lmarena_v2"])
@mark.parametrize("reason", [None, "max_output_tokens", "other"])
def test_cleanup_preserves_deliberate_zero_outputs(category, reason):
    row = {
        "category": category,
        "policy_answer": None,
        "policy_reasoning": "Reasoning without a final answer",
        "response": {"incomplete_details": {"reason": reason}} if reason else {"status": "completed"},
        "games": None,
    }
    preserve = category != "lmarena_v2" and reason in (None, "max_output_tokens")
    assert is_failed_rollout(row) is not preserve


def test_cleanup_still_removes_missing_judgments():
    assert is_failed_rollout({"category": "vision_arena_v1", "policy_answer": "Answer", "games": None})


def test_v4_token_report_does_not_count_policy_truncation_as_failed_judging(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    _write_jsonl(
        path,
        [
            {
                "category": "lmarena_v4",
                "policy_answer": "Truncated",
                "response": {"incomplete_details": {"reason": "max_output_tokens"}},
                "games": None,
            }
        ],
    )
    assert count_rollout_tokens(path, "gpt-4o")["failed_judgments"] == 0


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


@mark.parametrize("flag", ["--version", "--benchmark"])
def test_v4_offline_scoring_preserves_short_answers_and_prints_diagnostics(tmp_path, monkeypatch, capsys, flag):
    path = tmp_path / "rollouts.jsonl"
    prompt_path = tmp_path / "prompts.jsonl"
    row = {
        "_ng_task_index": 0,
        "question_id": "q1",
        "category": "lmarena_v4",
        "policy_answer": "Yes.",
        "baseline_answer": "Baseline.",
        "reward": 1.0,
        "style_reference_token_count": 1,
        "games": [{"verdict": "[[A>>B]]"}, {"verdict": "[[B>>A]]"}],
    }
    prompt = {
        "question_id": "q1",
        "category": "lmarena_v4",
        "metadata": {},
        "responses_create_params": {"input": []},
        "style_reference_token_count": 200,
        "is_lmarena_v2_prompt": False,
    }
    _write_jsonl(path, [row])
    _write_jsonl(prompt_path, [prompt])
    tasks = load_rollout_tasks(path, "lmarena_v4", {"q1": prompt})
    assert tasks[0][0]["style_reference_token_count"] == 200
    assert compute_rollout_scores.load_metrics("lmarena_v4", 0.01).compute(tasks)["win_rate"] == 1.0
    assert compute_rollout_scores.load_metrics("lmarena_v3", 0.01).compute(tasks)["win_rate"] == 0.0
    assert compute_rollout_scores.PROMPT_PATHS["lmarena_v4"].as_posix() == (
        "benchmarks/lmarena_v4/data/lmarena_v4_validation.jsonl"
    )
    monkeypatch.setattr(
        sys, "argv", ["compute_rollout_scores", str(path), flag, "lmarena_v4", "--prompts", str(prompt_path)]
    )
    compute_rollout_scores.main()
    output = capsys.readouterr().out
    assert "benchmark: lmarena_v4" in output
    assert "verbosity_acceptance_rate: 100.00%" in output
    assert "max_token_reached_rate: 0.00%" in output
    assert "context_window_exceeded_rate: 0.00%" in output


@mark.parametrize("category", ["lmarena_v2", "lmarena_v3"])
def test_v4_offline_scoring_rejects_an_unexpected_category(tmp_path, category):
    path = tmp_path / "rollouts.jsonl"
    _write_jsonl(path, [{"_ng_task_index": 0, "question_id": "q1", "category": category}])
    with raises(ValueError, match="--version is 'lmarena_v4'"):
        load_rollout_tasks(path, "lmarena_v4", {})


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
    v4 = build_observations(dataset, verdicts, {"opponent": 1400}, "judge", "lmarena_v4")

    assert v2[2] == 0
    assert v2[4] == 1
    assert v3[2] == 1
    np.testing.assert_array_equal(v3[1], [0.5, 0.5])
    assert v4[2] == 1
    np.testing.assert_array_equal(v4[1], [0.5, 0.5])
