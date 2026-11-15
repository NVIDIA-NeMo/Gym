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
"""An assistant message without text is an empty answer, like an answer that holds no assistant message.

A verifier that cannot judge an empty answer scores it zero without a grading operation (an arena judge does). The
saved answer then holds one assistant message whose text is empty or whitespace. The collector projects only whether
each assistant message has text, never the text, so such a scored-zero row is the model's own failed answer: an
empty answer that accounts for its trial, as an answer without an assistant message does.
"""

import json
from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import (
    CollectedEvidence,
    _empty_answer,
    _model_failure,
    decide_from_collected,
    failure_facts,
    reverification_cause,
    saved_row_evidence,
    trial_verdict,
)
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir


FIXTURE = Path(__file__).parent / "fixtures/cpu-grading-mcqa-healthy.json"


def _answer(text: str | None, *, status: str = "completed") -> dict:
    """A saved answer with one assistant message holding ``text`` (no content at all when None)."""
    message = {"type": "message", "role": "assistant", "status": "completed", "id": "msg_1"}
    if text is not None:
        message["content"] = [{"type": "output_text", "text": text, "annotations": []}]
    return {
        "object": "response",
        "id": "resp_answer",
        "status": status,
        "usage": {"output_tokens": 1},
        "output": [{"type": "reasoning", "summary": []}, message],
    }


def _collected_row(tmp_path, response: dict, reward: float = 0.0) -> dict:
    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=reward, response=response)], failures=[])
    [row] = collect_with_score(job, "[]").results
    return row


@pytest.mark.parametrize("text", ["", " \n\t"], ids=["empty", "whitespace"])
def test_the_collector_projects_only_whether_an_assistant_message_has_text(tmp_path, text):
    row = _collected_row(tmp_path, _answer(text))
    assert row["response"]["output"] == [
        {"type": "reasoning"},
        {"type": "message", "role": "assistant", "status": "completed", "has_text": False},
    ]
    assert _empty_answer(row) is True
    assert _model_failure(row) is True


def test_an_assistant_message_with_text_is_projected_without_its_text_and_is_no_empty_answer(tmp_path):
    row = _collected_row(tmp_path, _answer("a private answer"))
    assert row["response"]["output"][1]["has_text"] is True
    assert "a private answer" not in json.dumps(row)
    assert _empty_answer(row) is False
    assert _model_failure(row) is False


def test_a_message_without_saved_content_keeps_todays_reading(tmp_path):
    # Nothing says the message is empty: it is counted as an answer, as before.
    row = _collected_row(tmp_path, _answer(None))
    assert "has_text" not in row["response"]["output"][1]
    assert _empty_answer(row) is False


def _gate(*, has_text: bool, status: str = "completed", graded: bool = False):
    """The fixture's one trial, its answer one assistant message, scored zero unless a grading operation scored it."""
    collected = CollectedEvidence.model_validate_json(FIXTURE.read_text())
    row = collected.results[0]
    observed = row["ng_trial_observations"]
    message = {"type": "message", "role": "assistant", "status": "completed", "has_text": has_text}
    row["response"]["output"] = [message]
    row["response"]["status"] = status
    if status == "incomplete":
        observed["invocations"][0]["status"] = "incomplete"
    if not graded:
        row["reward"] = 0.0
        observed["verifier_invocations"][0]["grading_operations"] = None
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, reason, evaluation = trial_verdict(decision, collected)
    empty = failure_facts(decision, collected)["response_failures"]["by_cause"]["empty"]["count"]
    return passed, reason, evaluation, empty


def test_a_completed_answer_without_text_scored_zero_is_an_empty_answer_and_the_run_is_covered():
    passed, reason, evaluation, empty = _gate(has_text=False)
    assert passed is True, reason
    assert evaluation["exact"] is True
    assert empty == 1


def test_an_incomplete_answer_without_text_scored_zero_is_a_counted_empty_answer():
    passed, reason, evaluation, empty = _gate(has_text=False, status="incomplete")
    assert passed is True, reason
    assert evaluation["exact"] is True
    assert evaluation["model"]["kinds"] == {"empty_answer": 1}
    assert empty == 1


def test_an_answer_with_text_scored_zero_without_grading_stays_unobserved():
    passed, reason, evaluation, empty = _gate(has_text=True)
    assert passed is None, reason
    assert evaluation["status"] == "UNVERIFIED" and evaluation["exact"] is False
    assert empty == 0


def test_an_answer_without_text_that_a_grading_operation_scored_is_its_grade():
    passed, reason, evaluation, empty = _gate(has_text=False, graded=True)
    assert passed is True, reason
    assert evaluation["exact"] is True
    assert evaluation["model"]["kinds"] == {}
    assert empty == 0


def test_a_saved_empty_answer_scored_zero_is_not_reverified_as_ungraded():
    trajectory = {
        "task_id": "t0",
        "rollout_id": "r0",
        "verifier_invocations": [{"invocation_id": "v0", "status": "completed", "model_calls": []}],
    }
    empty = ident(0, reward=0.0, response=_answer(" "), ng_trajectory=trajectory)
    answered = ident(0, reward=0.0, response=_answer("an answer"), ng_trajectory=trajectory)
    assert reverification_cause(saved_row_evidence(empty)) is None
    assert reverification_cause(saved_row_evidence(answered)) == "ungraded"
