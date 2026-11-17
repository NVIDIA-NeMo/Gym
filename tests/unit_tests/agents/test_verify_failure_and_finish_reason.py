# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The small helpers agents and chat model servers share when a call fails or generation is cut short."""

from types import SimpleNamespace

from nemo_gym.agents.utils.verify_call import masked_verify_failure
from nemo_gym.core.failure_records import FailureBoundary
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.models.model_call_capture import note_finish_reason, start_model_execution


def test_a_failed_verify_call_is_a_masked_zero_that_keeps_the_answer() -> None:
    failure = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind="verifier_error",
        failure_reason="Verification failed",
        operation="verify",
    )

    result = masked_verify_failure({"response": {"id": "r"}}, failure)

    assert result == {
        "response": {"id": "r"},
        "reward": 0.0,
        "mask_sample": True,
        "failure_kind": "verifier_error",
        "failure_reason": "Verification failed",
    }


def test_a_finish_reason_is_noted_with_the_calls_execution_facts() -> None:
    request = SimpleNamespace(state=SimpleNamespace())

    note_finish_reason(request, None)
    assert not hasattr(request.state, "nemo_gym_model_execution")

    note_finish_reason(request, "length")
    assert request.state.nemo_gym_model_execution["finish_reason"] == "length"

    execution = start_model_execution(request, upstream_attempted=True)
    note_finish_reason(request, "stop")
    assert request.state.nemo_gym_model_execution is execution
    assert execution["finish_reason"] == "stop"
