# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import is_terminal_failure
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


def test_an_agent_harness_failure_keeps_its_detail_in_training(tmp_path, monkeypatch):
    monkeypatch.setattr("resources_servers.terminal_bench_4.app.is_eval_mode", lambda: False)
    verified = _verified(
        tmp_path,
        {"verifier_result": {"rewards": {"reward": 0.0}}},
        AgentTermination(reason="infrastructure_error", detail="mini-SWE produced no trajectory"),
    )

    assert verified["failure_reason"] == "mini-SWE produced no trajectory"


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_a_trial_the_verifier_scored_keeps_its_reward_in_evaluation(tmp_path, monkeypatch, reward):
    """mini-SWE died during the agent's run, but the verifier still graded the sandbox: the grade stands.

    The model can end mini-SWE itself (kill -9 -1, a memory bomb, a full disk). Discarding the grade would turn a
    scored trial into a retry, which Terminal-Bench does not do: a trial that ran its tests keeps their verdict.
    """
    monkeypatch.setattr("resources_servers.terminal_bench_4.app.is_eval_mode", lambda: True)
    verified = _verified(
        tmp_path,
        {"verifier_result": {"rewards": {"reward": reward}}},
        AgentTermination(reason="infrastructure_error", detail="mini-SWE produced no trajectory"),
    )

    assert verified["reward"] == reward
    assert verified["evaluation_completed"] is True
    assert verified["failure_reason"] is None
    assert "_ng_failure_class" not in verified


def test_a_scored_trial_with_platform_evidence_is_still_retried(tmp_path, monkeypatch):
    # The platform, not the model, ended the agent's sandbox: the grade is not a measurement.
    monkeypatch.setattr("resources_servers.terminal_bench_4.app.is_eval_mode", lambda: True)
    detail = "SandboxPlatformEvictionError: Sandbox was evicted by the platform: reason='Evicted'"
    verified = _verified(
        tmp_path,
        {"verifier_result": {"rewards": {"reward": 1.0}}},
        AgentTermination(reason="infrastructure_error", detail=detail),
    )

    assert verified["_ng_failure_class"] == "infrastructure_error"
    assert not is_terminal_failure(verified)


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


@pytest.mark.parametrize(
    ("agent_started", "detail", "expected"),
    [
        (False, "mini-SWE command exited 1", "AgentSetupError"),
        (True, "mini-SWE command exited 1", "AgentInfrastructureError"),
        # The provider found the running sandbox deleted (round-2 case R9: deleted through the OpenSandbox API).
        (
            True,
            "SandboxPlatformEvictionError: Sandbox not found: the control plane no longer knows it "
            "(Get sandbox failed: HTTP 404); sandbox_id='tb4-sbx'",
            "SandboxPlatformEvictionError",
        ),
    ],
    ids=["before-the-agent-ran", "during-the-agents-run", "sandbox-deleted-during-the-agents-run"],
)
async def test_a_harness_failure_is_named_by_whether_the_agent_ran(
    tmp_path, monkeypatch, agent_started, detail, expected
):
    """mini-SWE failing to set up comes before the model's first command (retried); failing during its run is the
    model's (final, 0), unless the platform names its own cause."""
    import resources_servers.terminal_bench_4.app as tb4_app

    async def _finalize(session, grade):
        session.result.setdefault("verifier_result", {})

    monkeypatch.setattr(tb4_app.lifecycle, "finalize_session", _finalize)
    termination = AgentTermination(reason="infrastructure_error", detail=detail)
    body = SimpleNamespace(
        termination=termination,
        agent_timings={},
        agent_started=agent_started,
        harness_metadata={},
        model_dump_json=lambda **_: "{}",
    )
    session = SimpleNamespace(
        verify_body=body,
        request=SimpleNamespace(capture_rollout_id=None),
        termination=None,
        result={},
        diagnostics=[],
        phase="ready",
        subphase=None,
        directory=tmp_path,
        seed_response=SimpleNamespace(termination=None),
        persist=lambda: None,
    )
    server = SimpleNamespace(_verified_response=lambda session, extra: None)

    await tb4_app.TerminalBench4ResourcesServer._finalize_session(server, session)

    exception_type = session.result["exception_info"]["exception_type"]
    assert exception_type == expected
    row = {"_ng_failure_class": "infrastructure_error", "_ng_failure_type": exception_type}
    assert is_terminal_failure(row) is (expected == "AgentInfrastructureError")


@pytest.mark.parametrize("eval_mode", [False, True], ids=["training", "evaluation"])
async def test_a_pod_oom_kill_during_the_agents_turn_is_the_models_zero(tmp_path, monkeypatch, eval_mode):
    """The sandbox pod was OOM-killed while the agent ran: the resource-limit detector grades the trial
    0, as Harbor scores it. No failure is recorded, so the row is final and not retried."""
    import resources_servers.terminal_bench_4.app as tb4_app

    monkeypatch.setattr(tb4_app, "is_eval_mode", lambda: eval_mode)
    monkeypatch.setattr(tb4_app.lifecycle, "is_eval_mode", lambda: eval_mode)
    monkeypatch.setattr(tb4_app.lifecycle, "cleanup", AsyncMock())
    detail = (
        "Failed to run command: [SANDBOX::ENDED] sandbox sandbox-1 exited with code 137 (OOMKilled) "
        "before the command finished"
    )
    termination = AgentTermination(reason="infrastructure_error", detail=detail)
    body = SandboxedVerifyRequest(
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
        agent_started=True,
    )
    session = SimpleNamespace(
        verify_body=body,
        request=SimpleNamespace(capture_rollout_id=None, task_name="task-1"),
        session_id="session-1",
        termination=None,
        result={},
        diagnostics=[],
        phase="ready",
        subphase=None,
        directory=tmp_path,
        seed_response=SimpleNamespace(termination=None),
        environment=None,
        verifier_environment=None,
        persist=lambda: None,
    )
    server = SimpleNamespace(
        _verified_response=lambda session, extra: TerminalBench4ResourcesServer._verified_response(
            None, session, extra
        )
    )

    await tb4_app.TerminalBench4ResourcesServer._finalize_session(server, session)

    verified = session.verified_response.model_dump()
    assert session.result["verifier_result"]["source"] == "sandbox_resource_limit"
    assert verified["sandbox_resource_limit"]["resource"] == "memory"
    assert (verified["reward"], verified["evaluation_completed"]) == (0.0, True)
    assert verified["infrastructure_error"] is None and "_ng_failure_class" not in verified
