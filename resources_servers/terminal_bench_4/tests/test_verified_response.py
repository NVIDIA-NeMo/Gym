# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.terminal_bench_4.app import TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.models import AgentTermination, SandboxedVerifyRequest, SandboxedVerifyResponse


def _verified(tmp_path, result, termination, diagnostics=()):
    session = SimpleNamespace(
        result=result,
        termination=termination,
        diagnostics=list(diagnostics),
        verify_body=SandboxedVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            response=NeMoGymResponse(
                id="resp-1",
                created_at=0,
                model="model",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ),
            session_id="session-1",
            termination=termination,
        ),
        request=SimpleNamespace(task_name="task-1"),
        directory=tmp_path,
    )
    return TerminalBench4ResourcesServer._verified_response(None, session, {}).model_dump()


def test_a_missing_reward_carries_the_exception_type(tmp_path):
    verified = _verified(
        tmp_path,
        {"exception_info": {"exception_type": "SandboxTimeoutException"}},
        AgentTermination(reason="completed"),
    )

    assert verified["_ng_failure_class"] == "infrastructure_error"
    assert verified["_ng_failure_type"] == "SandboxTimeoutException"


def _diagnostic(subphase, exception_type):
    return {"phase": "verifying", "subphase": subphase, "exception_type": exception_type, "exception_message": "x"}


@pytest.mark.parametrize(
    "exception_type",
    ["VerifierTimeoutError", "RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError"],
)
@pytest.mark.parametrize(
    "agent_exit,reason",
    [
        (None, "completed"),
        ("AgentTimeoutError", "timeout"),
        ("NonZeroAgentExitCodeError", "nonzero_exit"),
        (None, "cancelled"),
    ],
)
def test_a_verifier_outcome_terminal_bench_never_retries_scores_zero(tmp_path, exception_type, agent_exit, reason):
    """Terminal-Bench counts these as 0 without a retry, so the task stays in the pass-rate denominator. An agent
    exit recorded first does not hide the verifier's outcome."""
    first = agent_exit or exception_type
    diagnostics = [_diagnostic("quiesce", agent_exit)] if agent_exit else []
    diagnostics.append(_diagnostic("verifier_execution", exception_type))
    verified = _verified(
        tmp_path,
        {"exception_info": {"exception_type": first}},
        AgentTermination(reason=reason),
        diagnostics,
    )

    assert (verified["reward"], verified["evaluation_completed"], verified["mask_sample"]) == (0.0, False, False)
    assert (verified["failure_kind"], verified["failure_reason"]) == ("verifier_error", exception_type)
    assert verified["infrastructure_error"] is None
    assert "_ng_failure_class" not in verified and "_ng_failure_type" not in verified


@pytest.mark.parametrize(
    "exception_type",
    ["VerifierTimeoutError", "RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError"],
)
def test_an_agent_infrastructure_failure_stays_one_whatever_its_verifier_did(tmp_path, exception_type):
    """Harbor decides a retry from the trial's first exception, here the agent's infrastructure failure."""
    verified = _verified(
        tmp_path,
        {"exception_info": {"exception_type": "AgentInfrastructureError"}},
        AgentTermination(reason="infrastructure_error", detail="model provider unavailable"),
        [_diagnostic(None, "AgentInfrastructureError"), _diagnostic("verifier_execution", exception_type)],
    )

    assert verified["_ng_failure_class"] == "infrastructure_error"
    assert verified["_ng_failure_type"] == "AgentInfrastructureError"
    assert verified["infrastructure_error"] == "AgentInfrastructureError"
    assert verified["failure_kind"] is None


@pytest.mark.parametrize("exception_type", ["TimeoutError", "RuntimeError"])
def test_a_verifier_that_failed_on_infrastructure_is_not_scored(tmp_path, exception_type):
    """What run_verifier raises when a provider call times out within its budget (TimeoutError) or the sandbox
    reports no command outcome (RuntimeError)."""
    verified = _verified(
        tmp_path,
        {"exception_info": {"exception_type": exception_type}},
        AgentTermination(reason="completed"),
        [_diagnostic("verifier_execution", exception_type)],
    )

    assert verified["_ng_failure_class"] == "infrastructure_error"
    assert verified["_ng_failure_type"] == exception_type
    assert verified["failure_kind"] is None


def test_a_verifier_that_did_not_start_stays_an_infrastructure_failure(tmp_path):
    verified = _verified(
        tmp_path,
        {"exception_info": {"exception_type": "VerifierStartTimeoutError"}},
        AgentTermination(reason="completed"),
        [_diagnostic("verifier_setup", "VerifierStartTimeoutError")],
    )

    assert verified["_ng_failure_class"] == "infrastructure_error"
    assert verified["_ng_failure_type"] == "VerifierStartTimeoutError"
    assert verified["failure_kind"] is None


def test_a_missing_reward_without_exception_info_names_the_missing_reward(tmp_path):
    verified = _verified(tmp_path, {}, AgentTermination(reason="completed"))

    assert verified["_ng_failure_type"] == "MissingOfficialReward"


def test_a_free_text_infrastructure_detail_is_not_a_type(tmp_path):
    verified = _verified(
        tmp_path,
        {"verifier_result": {"rewards": {"reward": 0.0}}},
        AgentTermination(reason="infrastructure_error", detail="sandbox 10.0.0.7 went away after 3 retries"),
    )

    assert verified["_ng_failure_class"] == "infrastructure_error"
    assert verified["failure_reason"] == "sandbox 10.0.0.7 went away after 3 retries"
    assert "_ng_failure_type" not in verified


def test_a_scored_trial_carries_no_failure(tmp_path):
    verified = _verified(
        tmp_path, {"verifier_result": {"rewards": {"reward": 1.0}}}, AgentTermination(reason="completed")
    )

    assert "_ng_failure_class" not in verified
    assert "_ng_failure_type" not in verified


def test_test_counts_survive_rollout_serialization_and_replay_with_zero_reward(tmp_path):
    counts = {"num_tests_passed": 95, "num_tests_total": 96, "num_tests_failed": 1, "num_tests_skipped": 0}
    verified = _verified(
        tmp_path,
        {"verifier_result": {"rewards": {"reward": 0.0}, **counts}},
        AgentTermination(reason="completed"),
    )
    replayed = SandboxedVerifyResponse.model_validate_json(
        SandboxedVerifyResponse.model_validate(verified).model_dump_json()
    ).model_dump()
    assert {key: replayed[key] for key in counts} == counts
    assert replayed["reward"] == 0.0
    assert replayed["evaluation_completed"] is True
    assert replayed["infrastructure_error"] is None


def test_historical_trial_without_test_report_has_null_counts(tmp_path):
    verified = _verified(
        tmp_path, {"verifier_result": {"rewards": {"reward": 1.0}}}, AgentTermination(reason="completed")
    )
    for key in ("num_tests_passed", "num_tests_total", "num_tests_failed", "num_tests_skipped"):
        assert verified[key] is None
    assert verified["reward"] == 1.0
    assert verified["evaluation_completed"] is True


def _counts(passed, total, skipped=0):
    return {
        "num_tests_passed": passed,
        "num_tests_total": total,
        "num_tests_failed": total - passed - skipped,
        "num_tests_skipped": skipped,
    }


@pytest.mark.parametrize(
    "verifier_result,partial_credit",
    [
        ({"rewards": {"reward": 0.0}, **_counts(3, 4)}, 0.75),
        ({"rewards": {"reward": 1.0}, **_counts(5, 5)}, 1.0),
        # pytest exits 0 with skipped tests; an official pass keeps full credit.
        ({"rewards": {"reward": 1.0}, **_counts(4, 5, skipped=1)}, 1.0),
        # Without usable counts the official reward is the finest signal.
        ({"rewards": {"reward": 1.0}}, 1.0),
        ({"rewards": {"reward": 0.0}}, 0.0),
        ({"rewards": {"reward": 0.0}, **_counts(0, 0)}, 0.0),
        # No official reward: the evaluation did not complete, and like the reward it scores 0.
        ({"num_tests_passed": 2, "num_tests_total": 2}, 0.0),
        (None, 0.0),
    ],
)
def test_partial_credit_is_the_passed_fraction_floored_at_the_official_reward(
    tmp_path, verifier_result, partial_credit
):
    result = {} if verifier_result is None else {"verifier_result": verifier_result}
    verified = _verified(tmp_path, result, AgentTermination(reason="completed"))
    assert verified["partial_credit"] == partial_credit
    assert verified["reward"] == (verifier_result or {}).get("rewards", {}).get("reward", 0.0)
