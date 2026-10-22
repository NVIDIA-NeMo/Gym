# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.terminal_bench_4.app import TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.models import AgentTermination, SandboxedVerifyRequest, SandboxedVerifyResponse


def _verified(tmp_path, result, termination):
    session = SimpleNamespace(
        result=result,
        termination=termination,
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
