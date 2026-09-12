# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from harbor.models.trial.result import ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

from responses_api_agents.harbor_agent_general.app import HarborAgent


@pytest.mark.parametrize("error", ["AgentTimeoutError", "NonZeroAgentExitCodeError"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_agent_error_keeps_the_official_verifier_reward(error, reward):
    trial = TrialResult.model_construct(
        exception_info=ExceptionInfo.from_exception(type(error, (Exception,), {})("agent stopped")),
        verifier_result=VerifierResult(rewards={"reward": reward}),
    )
    assert HarborAgent.has_graded_agent_exception(trial)


@pytest.mark.parametrize(
    "error,verifier",
    [
        ("AgentTimeoutError", None),
        ("AgentTimeoutError", VerifierResult(rewards={})),
        ("VerifierTimeoutError", VerifierResult(rewards={"reward": 1.0})),
        ("SandboxError", None),
        (None, VerifierResult(rewards={"reward": 1.0})),
    ],
)
def test_ungraded_or_infrastructure_errors_are_not_accepted(error, verifier):
    trial = TrialResult.model_construct(
        exception_info=ExceptionInfo.from_exception(type(error, (Exception,), {})()) if error else None,
        verifier_result=verifier,
    )
    assert not HarborAgent.has_graded_agent_exception(trial)
