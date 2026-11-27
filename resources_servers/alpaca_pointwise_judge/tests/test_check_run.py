# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.indic_alpaca_eval.check_run import check_run
from resources_servers.alpaca_pointwise_judge.tests.test_app import judgment, metadata, response


def write_case(tmp_path, answers, budget=None):
    inputs, outputs = [], []
    for index, answer in enumerate(answers):
        task = {"id": str(index), "verifier_metadata": metadata(row_id=str(index)), "responses_create_params": {}}
        if budget is not None:
            task["responses_create_params"]["max_output_tokens"] = budget
        inputs.append(task)
        score = 4 if answer else 1
        outputs.append(
            {
                **task,
                "responses_create_params": {"max_output_tokens": budget or 120000},
                "response": response(answer).model_dump(),
                "candidate_answer": answer,
                "judgment": judgment(score),
                "reward": (score - 1) / 4,
                "judge_completion": {"model": "test-judge"},
            }
        )
    source, results = tmp_path / "input.jsonl", tmp_path / "rollouts.jsonl"
    for path, rows in [(source, inputs), (results, outputs)]:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return source, results


def test_smoke_retains_empty_candidate_answers(tmp_path):
    source, results = write_case(tmp_path, ["चार।", ""])
    assert check_run(source, results, smoke=True)["empty_final_answers"] == 1


def test_all_empty_smoke_still_fails(tmp_path):
    source, results = write_case(tmp_path, ["", ""])
    with pytest.raises(ValueError, match="at least one nonempty"):
        check_run(source, results, smoke=True)


def test_context_limited_budget_must_match_prepared_task(tmp_path):
    source, results = write_case(tmp_path, ["चार।"], budget=27000)
    assert check_run(source, results)["complete"]
    row = json.loads(results.read_text())
    row["responses_create_params"]["max_output_tokens"] = 30000
    results.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="budget changed"):
        check_run(source, results)


def test_explicit_run_budget_overrides_prepared_default(tmp_path):
    source, results = write_case(tmp_path, ["चार।"], budget=120000)
    row = json.loads(results.read_text())
    row["responses_create_params"]["max_output_tokens"] = 4096
    results.write_text(json.dumps(row) + "\n")
    assert check_run(source, results, max_output_tokens=4096)["complete"]
    with pytest.raises(ValueError, match="budget changed"):
        check_run(source, results, max_output_tokens=8192)


def test_empty_input_is_not_a_completed_run(tmp_path):
    source, results = write_case(tmp_path, [])
    with pytest.raises(ValueError, match="must not be empty"):
        check_run(source, results)
