# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A gymnasium environment run whose model answer was cut off by its own output limit is a counted model failure.

The environment grades in /step and has no verifier, so the cut-off answer is observed through the agent's one
invocation alone. ``gymnasium_agent`` records it (``ng_agent_observations``) with the model responses it asked for and
marks it incomplete. A run that recorded no invocation is unobserved and withholds a pass.
"""

import pytest

from nemo_gym.trajectory.trajectory_record import AgentInvocation, ModelCallRef
from tests.unit_tests.orchestration.test_cut_off_answer_trial_coverage import _cut_off, _verdict


def _gymnasium_cut_off(*, agent_invocation: bool):
    collected = _cut_off(call_status="incomplete", invalid_answer=False)
    observed = collected.results[0]["ng_trial_observations"]
    observed["verifier_invocations"] = []
    if agent_invocation:
        calls = [call["response_metadata"] for call in observed["model_calls"]]
        observed["invocations"] = [
            AgentInvocation(
                invocation_id="gymnasium_1",
                status="incomplete",
                model_calls=[
                    ModelCallRef(model_ref=meta["model_ref"], response_id=meta["response_id"]) for meta in calls
                ],
            ).model_dump(mode="json")
        ]
    else:
        observed["invocations"] = []
    return collected


def test_a_gymnasium_run_that_recorded_no_agent_invocation_is_unobserved():
    passed, reason, evaluation = _verdict(_gymnasium_cut_off(agent_invocation=False))
    assert passed is None, reason
    assert evaluation["status"] == "UNVERIFIED" and evaluation["exact"] is False


def test_a_gymnasium_answer_cut_off_by_the_models_output_limit_is_a_counted_model_failure():
    passed, reason, evaluation = _verdict(_gymnasium_cut_off(agent_invocation=True))
    assert passed is True, reason
    assert evaluation["exact"] is True
    assert evaluation["evaluation"]["count"] == 0
    assert evaluation["model"]["kinds"] == {"empty_answer": 1}


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_an_invocation_that_did_not_end_completed_or_incomplete_does_not_observe_the_answer(status):
    collected = _gymnasium_cut_off(agent_invocation=True)
    collected.results[0]["ng_trial_observations"]["invocations"][0]["status"] = status
    passed, reason, _ = _verdict(collected)
    assert passed is None, reason
