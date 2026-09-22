# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.terminal_bench_4.app import TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.models import AgentTermination, SandboxedVerifyRequest


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
