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
from nemo_gym.episode_types import FailureBoundary
from nemo_gym.failure_kinds import AGENT_INVALID_OUTPUT, MODEL_POLICY_RESPONSE_TIMEOUT, MODEL_TRANSPORT_ERROR
from nemo_gym.typed_exceptions import NgFailureEvent
from responses_api_agents.simple_agent.app import SimpleAgentConfig, SimpleAgentOutcomePolicyConfig


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


@pytest.mark.parametrize("field", ["default_action", "by_failure_kind", "by_status_code"])
def test_boundary_rejects_unsupported_actions_at_load(field):
    def value(action):
        return (
            action
            if field == "default_action"
            else {MODEL_POLICY_RESPONSE_TIMEOUT if field == "by_failure_kind" else 500: action}
        )

    with pytest.raises(ValidationError, match="does not support verify"):
        SimpleAgentOutcomePolicyConfig(**{field: value("verify")})
    with pytest.raises(ValidationError, match="does not support count_as_zero"):
        AgentOutcomePolicyConfig(**{field: value("count_as_zero")})


def test_alias_resolves_once_conflicts_are_rejected_and_config_roundtrips():
    base = dict(
        host="127.0.0.1",
        port=8000,
        entrypoint="",
        name="agent",
        model_server={"type": "responses_api_models", "name": "model"},
    )
    legacy = SimpleAgentConfig(**base, final_on_model_timeout=True)
    assert legacy.outcome_policy.by_failure_kind[MODEL_POLICY_RESPONSE_TIMEOUT] == OutcomeAction.COUNT_AS_ZERO
    assert "final_on_model_timeout" not in legacy.model_dump()
    assert (
        SimpleAgentConfig.model_validate(legacy.model_dump(exclude_none=True)).outcome_policy == legacy.outcome_policy
    )
    explicit = SimpleAgentConfig(**base, outcome_policy={"by_failure_kind": {MODEL_POLICY_RESPONSE_TIMEOUT: "fail"}})
    assert explicit.outcome_policy.default_action == OutcomeAction.RETRY
    assert explicit.outcome_policy.by_failure_kind[MODEL_POLICY_RESPONSE_TIMEOUT] == OutcomeAction.FAIL
    for alias, action in [(True, "retry"), (False, "count_as_zero")]:
        with pytest.raises(ValidationError, match="conflicts"):
            SimpleAgentConfig(
                **base,
                final_on_model_timeout=alias,
                outcome_policy={"by_failure_kind": {MODEL_POLICY_RESPONSE_TIMEOUT: action}},
            )


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
