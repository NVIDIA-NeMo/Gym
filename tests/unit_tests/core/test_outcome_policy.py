# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import get_args

import pytest
from pydantic import ValidationError

from nemo_gym.core.failure_kinds import PROVIDER_IO_TIMEOUT, PROVIDER_OOM_KILLED, FailureStage
from nemo_gym.core.outcome_policy import (
    OutcomeAction,
    OutcomeAttribution,
    StageOutcomePolicyConfig,
    StagePolicyConfig,
    choose_stage_outcome,
)
from nemo_gym.trajectory.trajectory_record import SandboxObservation


@pytest.mark.parametrize(
    "stage,action,attribution",
    [
        ("agent", OutcomeAction.COUNT_AS_ZERO, OutcomeAttribution.MODEL),
        ("seed", OutcomeAction.RETRY, OutcomeAttribution.EVALUATION),
        ("verification", OutcomeAction.RETRY, OutcomeAttribution.EVALUATION),
        ("admission", OutcomeAction.FAIL, OutcomeAttribution.UNKNOWN),
        ("cleanup", OutcomeAction.FAIL, OutcomeAttribution.UNKNOWN),
        (None, OutcomeAction.FAIL, OutcomeAttribution.UNKNOWN),
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
    assert agent.attribution == OutcomeAttribution.MODEL
    assert verifier.action == OutcomeAction.RETRY
    assert verifier.attribution == OutcomeAttribution.EVALUATION


def test_status_action_override_preserves_explicit_attribution():
    config = StageOutcomePolicyConfig.model_validate_json(
        json.dumps({"by_stage": {"agent": {"by_status_code": {"410": "fail"}}}})
    )
    decision = choose_stage_outcome(failure_kind=PROVIDER_OOM_KILLED, http_status=410, stage="agent", config=config)
    assert decision.action == OutcomeAction.FAIL
    assert decision.attribution == OutcomeAttribution.MODEL


def test_action_does_not_establish_attribution():
    config = StageOutcomePolicyConfig(
        by_stage={"agent": StagePolicyConfig(by_failure_kind={PROVIDER_IO_TIMEOUT: OutcomeAction.COUNT_AS_ZERO})}
    )
    decision = choose_stage_outcome(failure_kind=PROVIDER_IO_TIMEOUT, http_status=None, stage="agent", config=config)
    assert decision.action == OutcomeAction.COUNT_AS_ZERO
    assert decision.attribution == OutcomeAttribution.UNKNOWN
    # A partial model-valued override also retains the independently configured OOM rule.
    assert config.by_stage["agent"].attribution_by_failure_kind[PROVIDER_OOM_KILLED] == OutcomeAttribution.MODEL


def test_explicit_oom_override_can_change_action_and_attribution():
    config = StageOutcomePolicyConfig.model_validate(
        {
            "by_stage": {
                "agent": {
                    "by_failure_kind": {PROVIDER_OOM_KILLED: "fail"},
                    "attribution_by_failure_kind": {PROVIDER_OOM_KILLED: "evaluation"},
                }
            }
        }
    )
    decision = choose_stage_outcome(failure_kind=PROVIDER_OOM_KILLED, http_status=None, stage="agent", config=config)
    assert decision.action == OutcomeAction.FAIL
    assert decision.attribution == OutcomeAttribution.EVALUATION
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
    assert decision.attribution == OutcomeAttribution.EVALUATION


def test_seed_does_not_gain_unsafe_general_retries():
    decision = choose_stage_outcome(
        failure_kind=PROVIDER_IO_TIMEOUT, http_status=None, stage="seed", config=StageOutcomePolicyConfig()
    )
    assert decision.action == OutcomeAction.FAIL
    assert decision.attribution == OutcomeAttribution.UNKNOWN
