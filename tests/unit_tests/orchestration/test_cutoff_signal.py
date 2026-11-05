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
"""A truncated answer needs an actual cutoff signal; an incomplete status alone is not one.

An agent can leave its saved response incomplete for reasons other than the output cap, for example when the answer
holds only reasoning. A row whose reward came from a recorded grade is never a response failure. An incomplete answer
the benchmark's rule scored zero is a model failure: truncated with a cutoff signal (a ``length`` finish, an
incomplete reason of ``max_output_tokens`` or ``max_tokens``, or output tokens at the output cap), otherwise empty
(no assistant message) or malformed.
"""

import pytest

from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir
from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run


def _incomplete(row: dict, *, message: bool = False) -> None:
    """The agent saved the answer as incomplete; the model call itself reported no cutoff."""
    row["response"]["status"] = "incomplete"
    output = [{"type": "reasoning"}] + (
        [{"type": "message", "role": "assistant", "status": "completed"}] if message else []
    )
    row["response"]["output"] = output
    [call] = row["ng_trial_observations"]["model_calls"]
    call["response_metadata"] |= {"response_status": "completed", "finish_reason": None, "dialect": "responses"}
    call["output_items"] = list(output)


def _scored_zero_by_rule(row: dict) -> None:
    """The benchmark's rule scored the answer zero without a grading operation."""
    row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    row["reward"] = 0.0


def _response_failures(task: dict) -> dict:
    return {cause: fact["count"] for cause, fact in task["failures"]["response_failures"]["by_cause"].items()}


def test_a_graded_row_with_an_incomplete_status_is_no_response_failure():
    collected = _run(20)
    for row in collected.results:
        _incomplete(row, message=True)
    _, (passed, reason, report), task = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["model"]["count"] == 0
    assert _response_failures(task) == {"truncated": 0, "empty": 0, "malformed": 0}


CUTOFF_SIGNALS = pytest.mark.parametrize(
    "signal",
    [
        {"finish_reason": "length"},
        {"finish_reason": "max_output_tokens"},
        {"incomplete_details": {"reason": "max_output_tokens"}},
        {"max_output_tokens": 512, "usage": {"output_tokens": 512}},
    ],
    ids=["length-finish", "max-output-tokens-finish", "incomplete-reason", "output-at-cap"],
)


def _cut_off(row: dict, signal: dict) -> None:
    if "finish_reason" in signal:
        row["ng_trial_observations"]["model_calls"][0]["response_metadata"]["finish_reason"] = signal["finish_reason"]
    else:
        row["response"] |= signal


@CUTOFF_SIGNALS
def test_an_incomplete_answer_with_a_cutoff_signal_scored_zero_is_truncated(signal):
    collected = _run(20)
    row = collected.results[0]
    _incomplete(row)
    _scored_zero_by_rule(row)
    _cut_off(row, signal)
    _, (passed, reason, report), task = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["model"]["kinds"] == {"truncated_answer": 1}
    assert _response_failures(task) == {"truncated": 1, "empty": 0, "malformed": 0}


@CUTOFF_SIGNALS
def test_a_graded_cut_off_answer_scored_zero_is_truncated(signal):
    # A recorded grade of zero does not hide the cutoff: the answer was cut off at the output cap.
    collected = _run(20)
    row = collected.results[0]
    _incomplete(row)
    row["reward"] = 0.0
    _cut_off(row, signal)
    _, (passed, reason, report), task = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["model"]["kinds"] == {"truncated_answer": 1}
    assert _response_failures(task) == {"truncated": 1, "empty": 0, "malformed": 0}


def test_a_graded_cut_off_answer_that_kept_its_score_is_no_failure():
    collected = _run(20)
    row = collected.results[0]
    _incomplete(row, message=True)
    _cut_off(row, {"incomplete_details": {"reason": "max_output_tokens"}})
    _, (passed, reason, report), task = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["model"]["count"] == 0
    assert _response_failures(task) == {"truncated": 0, "empty": 0, "malformed": 0}


@pytest.mark.parametrize("message,cause", [(False, "empty"), (True, "malformed")])
def test_an_incomplete_answer_without_a_cutoff_signal_scored_zero_is_still_a_model_failure(message, cause):
    collected = _run(20)
    row = collected.results[0]
    _incomplete(row, message=message)
    _scored_zero_by_rule(row)
    # Output below its cap is no cutoff either.
    row["response"] |= {"max_output_tokens": 4096, "usage": {"output_tokens": 395}}
    _, (passed, reason, report), task = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["model"]["kinds"] == {f"{cause}_answer": 1}
    assert report["evaluation"]["count"] == 0
    assert _response_failures(task) == {"truncated": 0, "empty": 0, "malformed": 0} | {cause: 1}


def test_the_collector_projects_only_the_cutoff_facts(tmp_path):
    response = {
        "id": "resp-answer",
        "status": "incomplete",
        "output": [{"type": "reasoning", "summary": "kept on the run host"}],
        "incomplete_details": {"reason": "max_output_tokens"},
        "max_output_tokens": 2048,
        "usage": {"input_tokens": 10, "output_tokens": 2048, "output_tokens_details": {"reasoning_tokens": 2048}},
        "instructions": "kept on the run host",
    }
    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=0.0, response=response)], failures=[])
    [row] = collect_with_score(job, "[]").results
    assert row["response"]["output"] == [{"type": "reasoning"}]
    assert row["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
    assert row["response"]["max_output_tokens"] == 2048
    assert row["response"]["usage"] == {"output_tokens": 2048}
    assert "instructions" not in row["response"]
