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
"""A scored-zero answer the model left incomplete is a counted model failure, observed through its answer alone.

The agent saved the model's cut-off answer (only reasoning, status ``incomplete``) and marked its own invocation
incomplete; the verifier completed and the benchmark's rule scored the answer zero without a grading operation. The
trial gate counts the row as a model failure (``truncated_answer``), so it must not break the coverage join of the
other trials. Its answer must still be the model's own observed answer, and a verifier error is never read as one.
"""

from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import CollectedEvidence, decide_from_collected, trial_verdict


FIXTURE = Path(__file__).parent / "fixtures/cpu-grading-mcqa-healthy.json"


def _cut_off(*, call_status: str | None, invalid_answer: bool) -> CollectedEvidence:
    """The one trial's answer was cut off; ``call_status`` is what the model server reported for the call."""
    collected = CollectedEvidence.model_validate_json(FIXTURE.read_text())
    row = collected.results[0]
    observed = row["ng_trial_observations"]
    row["reward"] = 0.0
    row["response"]["status"] = "incomplete"
    row["response"]["output"] = [{"type": "reasoning"}]
    observed["invocations"][0]["status"] = "incomplete"
    observed["verifier_invocations"][0]["grading_operations"] = None
    [call] = observed["model_calls"]
    call["response_metadata"] |= {"response_status": call_status, "finish_reason": None, "dialect": "responses"}
    call["output_items"] = [{"type": "reasoning"}]
    if invalid_answer:
        row["_ng_collected_grading_facts"] = {"invalid_model_response": True}
    return collected


SHAPES = [
    pytest.param({"call_status": "incomplete", "invalid_answer": False}, id="model-server-incomplete"),
    pytest.param({"call_status": "completed", "invalid_answer": True}, id="verifier-marked-invalid"),
]


def _verdict(collected: CollectedEvidence):
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    return trial_verdict(decision, collected)


@pytest.mark.parametrize("shape", SHAPES)
def test_a_cut_off_answer_scored_zero_is_a_counted_model_failure_and_the_run_is_covered(shape):
    passed, reason, evaluation = _verdict(_cut_off(**shape))
    assert passed is True, reason
    assert evaluation["exact"] is True
    assert evaluation["evaluation"]["count"] == 0
    # No cutoff signal and no assistant message: an empty answer, still the model's fault.
    assert evaluation["model"]["kinds"] == {"empty_answer": 1}


def _verifier_failed(collected):
    collected.results[0]["ng_trial_observations"]["verifier_invocations"][0] |= {
        "status": "failed",
        "failure_kind": "verifier_error",
    }


def _another_answer_saved(collected):
    collected.results[0]["response"]["id"] = "resp_another"


def _agent_error(collected):
    collected.results[0]["ng_trial_observations"]["invocations"][0]["error_type"] = "RuntimeError"


def _answer_operation_failed(collected):
    [call] = collected.results[0]["ng_trial_observations"]["model_calls"]
    call["response_metadata"]["model_operations"][0]["outcome"] = "failed"


@pytest.mark.parametrize(
    "unobserved", [_verifier_failed, _another_answer_saved, _agent_error, _answer_operation_failed]
)
@pytest.mark.parametrize("shape", SHAPES)
def test_a_cut_off_answer_that_is_not_the_models_observed_answer_withholds_a_pass(shape, unobserved):
    collected = _cut_off(**shape)
    unobserved(collected)
    passed, reason, evaluation = _verdict(collected)
    assert passed is None, reason
    assert evaluation["status"] == "UNVERIFIED" and evaluation["exact"] is False


def test_an_agents_own_answer_envelope_cut_off_on_its_last_call_is_the_models_observed_answer():
    # A sandboxed agent saves its own answer envelope (its own response id), not the model's reply: the cut-off is
    # observed on the model's own last call, which finished on its output cap.
    collected = _cut_off(call_status=None, invalid_answer=False)
    row = collected.results[0]
    row["response"] |= {"id": "resp_agent_envelope", "incomplete_details": {"reason": "max_output_tokens"}}
    [call] = row["ng_trial_observations"]["model_calls"]
    call["response_metadata"] |= {"dialect": "chat", "response_status": None, "finish_reason": "length"}
    passed, reason, evaluation = _verdict(collected)
    assert passed is True, reason
    assert evaluation["model"]["kinds"] == {"truncated_answer": 1}
    # The same envelope whose last call did not finish on a cutoff is not the model's observed answer.
    call["response_metadata"]["finish_reason"] = "stop"
    passed, reason, _ = _verdict(collected)
    assert passed is None, reason


def test_an_empty_envelope_whose_last_call_was_cut_is_not_the_models_observed_answer():
    # The envelope itself carries no cutoff: an empty or malformed envelope can be the agent's own assembly problem,
    # so a last call that happened to be cut does not bind it.
    collected = _cut_off(call_status=None, invalid_answer=False)
    row = collected.results[0]
    row["response"] |= {"id": "resp_agent_envelope", "incomplete_details": None}
    row["response"].pop("usage", None)
    [call] = row["ng_trial_observations"]["model_calls"]
    call["response_metadata"] |= {"dialect": "chat", "response_status": None, "finish_reason": "length"}
    passed, reason, _ = _verdict(collected)
    assert passed is None, reason
