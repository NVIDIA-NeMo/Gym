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
"""A trial is observed when a saved grading artifact records how it was graded; a bare reward never is."""

import json

from nemo_gym.orchestration.recovery import grading_artifact
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir
from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run, _truncate


NO_FACTS = {
    "judge_response": False,
    "judgments": False,
    "judge_score": False,
    "judge_evaluations": False,
    "judge_verdicts": None,
    "judge_error": False,
    "judge_unparsed": False,
    "games_unjudged": False,
    "rubric": None,
    "library_reward": None,
    "sentence_metrics": [],
    "verifier_exit_code": None,
    "verifier_error": False,
    "environment_reward": None,
    "invalid_model_response": False,
    "test_run": None,
    "substep_tests": None,
    "unit_tests": None,
    "node_f1": None,
    "guarded_objectives": None,
    "native_score": None,
    "saved_model_patch": None,
    "collected_patch": False,
}


def _ungraded(row: dict, **facts) -> dict:
    """The row as a verifier without a grading operation or grading record saves it, with these artifact facts."""
    row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    row["_ng_collected_grading_facts"] = NO_FACTS | facts
    return row


def _judge_call(row: dict, task: int) -> None:
    """The verifier's own captured judge call, beside the policy's."""
    observed = row["ng_trial_observations"]
    [policy] = observed["model_calls"]
    call = json.loads(json.dumps(policy))
    call["model_call_id"] = f"judge-call-{task}"
    call["response_metadata"]["response_id"] = f"judge-response-{task}"
    call["response_metadata"]["model_ref"] = {"type": "responses_api_models", "name": "judge"}
    observed["model_calls"].append(call)
    ref = {"model_call_id": call["model_call_id"], "response_id": None, "model_ref": None}
    observed["verifier_invocations"][0]["model_calls"] = [ref]


def _observed(collected):
    _, (passed, reason, report), task = _gate(collected)
    return passed, reason, report, task


def test_a_judged_run_with_truncated_answers_is_fully_observed():
    collected = _run(120)
    for task, row in enumerate(collected.results[:89]):
        _ungraded(row, judgments=True, judge_score=True)
        _judge_call(row, task)
    for row in collected.results[89:]:
        _ungraded(row)
        _truncate(row)
    passed, reason, report, task = _observed(collected)
    assert passed is True and report["exact"] is True, reason
    assert task["trial_counts"]["unobserved"] == 0
    assert task["failures"]["response_failures"]["by_cause"]["truncated"]["count"] == 31
    assert task["health_verdict"].startswith(
        "evaluation health: PASS; response failures: 25.83% (truncated 31 of 120)"
    )


def test_an_in_process_metric_run_is_fully_observed():
    collected = _run(20)
    for row in collected.results:
        _ungraded(row, sentence_metrics=["sentence_chrf", "sentence_spbleu"])
    passed, reason, report, task = _observed(collected)
    assert passed is True and report["exact"] is True, reason
    assert task["trial_counts"]["unobserved"] == 0


def test_a_test_execution_run_is_fully_observed():
    collected = _run(20)
    for row in collected.results:
        _ungraded(row, verifier_exit_code=0)
        row["evaluation_completed"] = True
    passed, reason, report, _ = _observed(collected)
    assert passed is True and report["exact"] is True, reason


def test_an_in_environment_reward_run_is_fully_observed():
    collected = _run(20)
    for row in collected.results:
        _ungraded(row, environment_reward=row["reward"])
        row["ng_trial_observations"]["verifier_invocations"] = []
    passed, reason, report, task = _observed(collected)
    assert passed is True and report["exact"] is True, reason
    assert task["trial_counts"]["unobserved"] == 0


def test_a_row_with_only_a_bare_reward_stays_unverified():
    collected = _run(20)
    _ungraded(collected.results[0])
    assert grading_artifact(collected.results[0]) is None
    passed, reason, report, task = _observed(collected)
    assert passed is None and report["status"] == "UNVERIFIED"
    assert reason.startswith("trial: UNOBSERVED 1 of 20 expected trials")
    assert task["trial_counts"]["unobserved"] == 1


def test_a_judge_row_without_its_judge_response_or_another_artifact_stays_unverified():
    collected = _run(20)
    row = _ungraded(collected.results[0], judge_response=False)
    _judge_call(row, 0)
    passed, _, report, task = _observed(collected)
    assert passed is None and report["status"] == "UNVERIFIED"
    assert task["trial_counts"]["unobserved"] == 1


def test_a_symbolic_match_must_agree_with_the_reward_unless_a_judge_decided():
    row = _ungraded(_run(1).results[0], library_reward=1.0)
    assert grading_artifact(row) == "symbolic_match"
    row["_ng_collected_grading_facts"]["library_reward"] = 0.0
    assert grading_artifact(row) is None
    row["_ng_collected_grading_facts"]["judge_response"] = True
    assert grading_artifact(row) == "judge_record"


def test_an_answer_the_verifier_marked_invalid_is_observed_and_reported_malformed():
    collected = _run(20)
    row = _ungraded(collected.results[0], invalid_model_response=True)
    row["reward"] = 0.0
    passed, reason, _, task = _observed(collected)
    assert passed is True, reason
    assert task["failures"]["response_failures"]["by_cause"]["malformed"]["count"] == 1


def test_the_collector_projects_scalar_artifact_facts_only(tmp_path):
    response = {"id": "r", "status": "completed", "output": [{"type": "message", "role": "assistant"}]}
    rows = [
        ident(0, reward=1.0, response=response, judgments=[{"judge_score": 1, "text": "x"}], judge_response=None),
        ident(1, reward=0.0, response=response, sentence_chrf=12.5, library_reward=0.0),
        ident(2, reward=1.0, response=response, result={"reward_info": {"reward": 1.0}}, verifier_exit_code=0),
    ]
    job = write_job_dir(tmp_path / "job", n=3, results=rows, failures=[])
    facts = [row["_ng_collected_grading_facts"] for row in collect_with_score(job, "[]").results]
    assert facts[0] == NO_FACTS | {"judgments": True, "judge_score": True}
    assert facts[1] == NO_FACTS | {"sentence_metrics": ["sentence_chrf"], "library_reward": 0.0}
    assert facts[2] == NO_FACTS | {"environment_reward": 1.0, "verifier_exit_code": 0}


def test_a_zero_valued_artifact_is_projected_and_observed(tmp_path):
    from nemo_gym.orchestration.recovery import grading_artifact

    response = {"id": "r", "status": "completed", "output": [{"type": "message", "role": "assistant"}]}
    rows = [
        # A wrong answer by symbolic match, an environment that scored zero, a passing test run: each artifact's only
        # fact is a zero.
        ident(0, reward=0.0, response=response, library_reward=0.0),
        ident(1, reward=0.0, response=response, result={"reward_info": {"reward": 0.0}}),
        ident(2, reward=1.0, response=response, evaluation_completed=True, verifier_exit_code=0, verifier_error=None),
    ]
    job = write_job_dir(tmp_path / "job", n=3, results=rows, failures=[])
    collected = collect_with_score(job, "[]").results

    assert [grading_artifact(row) for row in collected] == ["symbolic_match", "environment_reward", "test_execution"]
