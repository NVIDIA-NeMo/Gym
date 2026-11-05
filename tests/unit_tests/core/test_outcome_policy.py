# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import get_args

import pytest
from pydantic import ValidationError

from nemo_gym.core.failure_kinds import (
    CANCELLED,
    KILL_SHAPED,
    PROVIDER_CREDENTIALS_REJECTED,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_OOM_KILLED,
    SHUTDOWN,
    FailureClass,
    FailureClassificationConfig,
    FailureStage,
)
from nemo_gym.core.outcome_policy import (
    OutcomeAction,
    StageOutcomePolicyConfig,
    StagePolicyConfig,
    choose_stage_outcome,
)
from nemo_gym.trajectory.trajectory_record import SandboxObservation


@pytest.mark.parametrize(
    "stage,action,attribution",
    [
        ("agent", OutcomeAction.COUNT_AS_ZERO, FailureClass.MODEL),
        ("seed", OutcomeAction.RETRY, FailureClass.EVALUATION),
        ("verification", OutcomeAction.RETRY, FailureClass.EVALUATION),
        ("admission", OutcomeAction.FAIL, FailureClass.UNKNOWN),
        ("cleanup", OutcomeAction.FAIL, FailureClass.UNKNOWN),
        (None, OutcomeAction.FAIL, FailureClass.UNKNOWN),
    ],
)
def test_shared_oom_defaults_depend_on_origin(stage, action, attribution):
    decision = choose_stage_outcome(
        failure_kind=PROVIDER_OOM_KILLED, http_status=None, stage=stage, config=StageOutcomePolicyConfig()
    )
    assert decision.action == action
    assert decision.attribution == attribution


def test_partial_overrides_retain_stage_and_kind_defaults():
    config = StageOutcomePolicyConfig.model_validate_json(
        json.dumps({"by_stage": {"agent": {"by_failure_kind": {PROVIDER_IO_TIMEOUT: "fail"}}}})
    )
    agent = choose_stage_outcome(failure_kind=PROVIDER_OOM_KILLED, http_status=None, stage="agent", config=config)
    verifier = choose_stage_outcome(
        failure_kind=PROVIDER_OOM_KILLED, http_status=None, stage="verification", config=config
    )
    assert agent.action == OutcomeAction.COUNT_AS_ZERO
    assert agent.attribution == FailureClass.MODEL
    assert verifier.action == OutcomeAction.RETRY
    assert verifier.attribution == FailureClass.EVALUATION


def test_status_action_override_preserves_explicit_attribution():
    config = StageOutcomePolicyConfig.model_validate_json(
        json.dumps({"by_stage": {"agent": {"by_status_code": {"410": "fail"}}}})
    )
    decision = choose_stage_outcome(failure_kind=PROVIDER_OOM_KILLED, http_status=410, stage="agent", config=config)
    assert decision.action == OutcomeAction.FAIL
    assert decision.attribution == FailureClass.MODEL


def test_action_does_not_establish_attribution():
    config = StageOutcomePolicyConfig(
        by_stage={"agent": StagePolicyConfig(by_failure_kind={PROVIDER_IO_TIMEOUT: OutcomeAction.COUNT_AS_ZERO})}
    )
    decision = choose_stage_outcome(failure_kind=PROVIDER_IO_TIMEOUT, http_status=None, stage="agent", config=config)
    assert decision.action == OutcomeAction.COUNT_AS_ZERO
    assert decision.attribution == FailureClass.EVALUATION
    # A partial model-valued override also retains the independently configured OOM rule.
    assert config.classification.by_stage["agent"].by_failure_kind[PROVIDER_OOM_KILLED] == FailureClass.MODEL


def test_explicit_oom_override_can_change_action_and_attribution():
    config = StageOutcomePolicyConfig.model_validate(
        {
            "by_stage": {
                "agent": {
                    "by_failure_kind": {PROVIDER_OOM_KILLED: "fail"},
                }
            },
            "classification": {"by_stage": {"agent": {"by_failure_kind": {PROVIDER_OOM_KILLED: "evaluation"}}}},
        }
    )
    decision = choose_stage_outcome(failure_kind=PROVIDER_OOM_KILLED, http_status=None, stage="agent", config=config)
    assert decision.action == OutcomeAction.FAIL
    assert decision.attribution == FailureClass.EVALUATION
    assert StageOutcomePolicyConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    "rules",
    [
        {"agent_execution": {}},
        {"setup": {}},
        {"agent": {"by_failure_kind": {"provider_oom_typo": "fail"}}},
        {"agent": {"attribution_by_failure_kind": {"provider_oom_typo": "model"}}},
        {"agent": {"by_status_code": {"600": "fail"}}},
        {"agent": {"default_attribution": "model_fault"}},
    ],
)
def test_invalid_policy_facts_are_rejected_at_config_load(rules):
    with pytest.raises(ValidationError):
        StageOutcomePolicyConfig.model_validate({"by_stage": rules})


def test_stage_defaults_are_not_shared_mutable_state():
    modified = StageOutcomePolicyConfig()
    modified.by_stage["agent"].by_failure_kind[PROVIDER_OOM_KILLED] = OutcomeAction.FAIL
    assert (
        StageOutcomePolicyConfig().by_stage["agent"].by_failure_kind[PROVIDER_OOM_KILLED]
        == OutcomeAction.COUNT_AS_ZERO
    )


@pytest.mark.parametrize("stage", get_args(FailureStage))
def test_observation_uses_the_same_stage_vocabulary(stage):
    observation = SandboxObservation(role="agent", stage=stage)
    assert SandboxObservation.model_validate_json(observation.model_dump_json()).stage == stage


def test_observation_rejects_a_parallel_phase_vocabulary():
    with pytest.raises(ValidationError):
        SandboxObservation(role="agent", stage="agent_execution")
    with pytest.raises(ValidationError):
        SandboxObservation(role="agent", phase="agent_execution")
    assert SandboxObservation(role="agent").stage is None


def test_inconclusive_verification_retains_bounded_retry_default():
    decision = choose_stage_outcome(
        failure_kind=PROVIDER_IO_TIMEOUT, http_status=None, stage="verification", config=StageOutcomePolicyConfig()
    )
    assert decision.action == OutcomeAction.RETRY
    assert decision.attribution == FailureClass.EVALUATION


def test_seed_does_not_gain_unsafe_general_retries():
    decision = choose_stage_outcome(
        failure_kind=PROVIDER_IO_TIMEOUT, http_status=None, stage="seed", config=StageOutcomePolicyConfig()
    )
    assert decision.action == OutcomeAction.FAIL
    assert decision.attribution == FailureClass.EVALUATION


@pytest.mark.parametrize("kind", [CANCELLED, SHUTDOWN, KILL_SHAPED, PROVIDER_CREDENTIALS_REJECTED])
def test_verification_does_not_retry_stop_or_credentials_failures(kind):
    config = StageOutcomePolicyConfig(by_stage={"verification": {"by_failure_kind": {PROVIDER_OOM_KILLED: "retry"}}})
    decision = choose_stage_outcome(failure_kind=kind, http_status=None, stage="verification", config=config)
    assert decision.action == OutcomeAction.FAIL
    assert decision.attribution == FailureClass.EVALUATION


def test_action_precedence_status_then_kind_then_class_then_default():
    from nemo_gym.core.outcome_policy import OutcomePolicyConfig, choose_action

    config = OutcomePolicyConfig(
        default_action="fail",
        by_failure_class={"evaluation": "retry"},
        by_failure_kind={PROVIDER_OOM_KILLED: "count_as_zero"},
        by_status_code={"503": "verify"},
    )
    assert (
        choose_action(
            failure_kind=PROVIDER_OOM_KILLED, http_status=503, failure_class=FailureClass.EVALUATION, config=config
        )
        == "verify"
    )
    assert (
        choose_action(
            failure_kind=PROVIDER_OOM_KILLED, http_status=None, failure_class=FailureClass.EVALUATION, config=config
        )
        == "count_as_zero"
    )
    assert (
        choose_action(
            failure_kind=PROVIDER_IO_TIMEOUT, http_status=None, failure_class=FailureClass.EVALUATION, config=config
        )
        == "retry"
    )
    assert (
        choose_action(
            failure_kind=PROVIDER_IO_TIMEOUT, http_status=None, failure_class=FailureClass.UNKNOWN, config=config
        )
        == "fail"
    )


def test_stage_class_override_changes_class_rule_but_not_more_specific_action():
    config = StageOutcomePolicyConfig(
        classification=FailureClassificationConfig(
            by_stage={"agent": {"by_failure_kind": {PROVIDER_IO_TIMEOUT: "model"}}}
        ),
        by_stage={"agent": {"by_failure_class": {"model": "count_as_zero"}}},
    )
    assert (
        choose_stage_outcome(failure_kind=PROVIDER_IO_TIMEOUT, stage="agent", http_status=None, config=config).action
        == "count_as_zero"
    )
    config.by_stage["agent"].by_failure_kind[PROVIDER_IO_TIMEOUT] = OutcomeAction.FAIL
    decision = choose_stage_outcome(failure_kind=PROVIDER_IO_TIMEOUT, stage="agent", http_status=None, config=config)
    assert decision.action == "fail" and decision.attribution == FailureClass.MODEL


def test_policy_classification_preserves_recovered_event_and_cause():
    from nemo_gym.core.failure_records import FailureReport
    from nemo_gym.core.typed_exceptions import NgFailureEvent

    inner = NgFailureEvent(
        boundary="sandbox",
        failure_kind=PROVIDER_IO_TIMEOUT,
        stage="agent",
        operation="read",
        failure_reason="read timed out",
    )
    outer = NgFailureEvent(
        boundary="agent", failure_kind="agent_route_failed", stage="agent", failure_reason="route failed"
    )
    outer.__cause__ = inner
    before = outer.to_record()
    decision = choose_stage_outcome(
        failure_kind=inner.failure_kind, stage=inner.stage, http_status=None, config=StageOutcomePolicyConfig()
    )
    result = {"reward": 1.0, **before}
    assert decision.attribution == FailureClass.EVALUATION
    assert outer.to_record() == before
    report = FailureReport(failure_id=before["failure_id"], events=before["failure_events"])
    assert report.events[inner.event_id].stage == "agent"
    assert report.events[outer.event_id].cause_id == inner.event_id
    assert result["reward"] == 1.0
