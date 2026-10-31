# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_failure_payload_preserves_only_valid_typed_slots():
    from nemo_gym.trajectory.trajectory_record import failure_observations

    assert (
        failure_observations({"response": {"id": "not-a-result"}, "reward": 1, "ng_trajectory": {"bad": True}}) == {}
    )
    assert (
        failure_observations(
            {"ng_verifier_invocations": [{"invocation_id": "v", "status": "failed", "failure_kind": "verifier_error"}]}
        )["ng_verifier_invocations"][0]["invocation_id"]
        == "v"
    )
    assert failure_observations({"ng_verifier_invocations": [{"invocation_id": "v", "invented": True}]}) == {}


def test_runtime_and_cost_observations_reject_coercion_and_nonfinite_values():
    import pytest
    from pydantic import ValidationError

    from nemo_gym.failure_transport import ModelOperationObservation
    from nemo_gym.trajectory.trajectory_record import AgentInvocation

    with pytest.raises(ValidationError):
        AgentInvocation(invocation_id="a", in_session=1)
    with pytest.raises(ValidationError):
        AgentInvocation(invocation_id="a", verification_skipped="false")
    with pytest.raises(ValidationError):
        ModelOperationObservation(operation_id="o", outcome="completed", retry_wait_s=float("inf"))
