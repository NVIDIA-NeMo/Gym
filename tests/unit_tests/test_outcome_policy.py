# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared outcome selection preserves explicit evidence and boundary-specific capabilities."""

import pytest
from pydantic import ValidationError

from nemo_gym.agent_outcome import (
    AgentOutcomePolicyConfig,
    OutcomeAction,
    OutcomePolicyConfig,
    choose_action,
    failure_outcome_policy,
)
from nemo_gym.core.failure_kinds import AGENT_INVALID_OUTPUT, MODEL_POLICY_RESPONSE_TIMEOUT, MODEL_TRANSPORT_ERROR
from nemo_gym.server_runtime.episode_types import FailureBoundary
from nemo_gym.typed_exceptions import NgFailureEvent


def test_status_kind_default_and_partial_overrides():
    policy = AgentOutcomePolicyConfig(by_failure_kind={MODEL_TRANSPORT_ERROR: OutcomeAction.FAIL})
    assert policy.by_failure_kind[AGENT_INVALID_OUTPUT] == OutcomeAction.VERIFY
    assert choose_action(failure_kind=MODEL_TRANSPORT_ERROR, http_status=503, config=policy) == OutcomeAction.RETRY
    assert choose_action(failure_kind=MODEL_TRANSPORT_ERROR, http_status=None, config=policy) == OutcomeAction.FAIL
    assert choose_action(failure_kind="unknown", http_status=400, config=policy) == OutcomeAction.VERIFY
    assert (
        choose_action(failure_kind="unknown", http_status=418, config=OutcomePolicyConfig(default_action="fail"))
        == OutcomeAction.FAIL
    )
    with pytest.raises(ValidationError, match="Unregistered failure"):
        OutcomePolicyConfig(default_action="retry", by_failure_kind={"typo": "fail"})


def test_event_adapter_does_not_search_an_inner_http_status():
    event = NgFailureEvent(
        boundary=FailureBoundary.AGENT, failure_kind=MODEL_POLICY_RESPONSE_TIMEOUT, failure_reason="timeout"
    )
    event.__cause__ = NgFailureEvent(
        boundary=FailureBoundary.MODEL, failure_kind="model_request_error", failure_reason="upstream", http_status=503
    )
    policy = OutcomePolicyConfig(
        default_action="fail",
        by_status_code={503: "retry"},
        by_failure_kind={MODEL_POLICY_RESPONSE_TIMEOUT: "count_as_zero"},
    )
    before = event.to_record()
    assert failure_outcome_policy(event, config=policy) == OutcomeAction.COUNT_AS_ZERO
    assert event.to_record() == before


@pytest.mark.parametrize("complete", [True, False, None])
def test_upload_evidence_roundtrips_in_reports_and_process_projection(complete):
    import pickle

    import cloudpickle

    from nemo_gym.failure_transport import FailureHeader
    from nemo_gym.typed_exceptions import failure_report_for, rebuild_failure

    event = NgFailureEvent(
        boundary="model",
        failure_kind="model_request_error",
        failure_reason="observed",
        request_upload_complete=complete,
    )
    report = failure_report_for(event)
    assert FailureHeader.decode(FailureHeader.encode(report)) == report
    assert rebuild_failure(report).request_upload_complete is complete
    for serializer in [pickle, cloudpickle]:
        assert serializer.loads(serializer.dumps(event)).to_record() == event.to_record()
