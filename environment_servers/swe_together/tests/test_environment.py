# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError

from environment_servers.swe_together import app, evaluation
from environment_servers.swe_together.app import Episode, SWETConfig, SWETEnvironmentServer, memory_mib
from environment_servers.swe_together.schemas import CandidateTurn, SWETRequest, UserTurn, episode_response
from environment_servers.swe_together.simulator import UserAgent
from environment_servers.swe_together.task import TaskData
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseUsage
from nemo_gym.rollout_collection import _episode_record


def activation(index=0, **kwargs):
    return CandidateTurn(
        activation_id=index + 1,
        response=NeMoGymResponse(
            id="r",
            object="response",
            created_at=1,
            model="test",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            status="completed",
        ),
        **kwargs,
    )


def request():
    task = TaskData(task_id="task", image="image", image_digest="sha256:" + "1" * 64)
    return SWETRequest(
        episode_id={"rollout_id": "one"},
        task={"task_id": {"taskset": "swet", "task_id": "task"}, "task_input": {"task_data": task}},
    )


def test_episode_usage_covers_all_activations_and_preserves_unknown_counts():
    turns = [activation(index) for index in range(2)]
    for index, turn in enumerate(turns):
        turn.response.usage = NeMoGymResponseUsage(
            input_tokens=10 + index,
            output_tokens=2,
            total_tokens=12 + index,
            input_tokens_details={"cached_tokens": None if index else 1},
            output_tokens_details={"reasoning_tokens": 1},
        )
    combined = episode_response(turns)
    assert combined.usage.input_tokens == 21
    assert combined.usage.output_tokens == 4
    assert combined.usage.total_tokens == 25
    assert combined.usage.input_tokens_details.cached_tokens is None
    assert turns[-1].response.usage.input_tokens == 11
    turns[-1].response.usage = None
    assert episode_response(turns).usage is None


def server(tmp_path, **updates):
    config = SWETConfig(
        name="swet",
        host="localhost",
        port=1234,
        entrypoint="app.py",
        asset_root=str(tmp_path),
        artifact_root=str(tmp_path),
        runtime_manifest_path="unused",
        agent_server={"type": "responses_api_agents", "name": "policy"},
        judge_agent={"type": "responses_api_agents", "name": "judge"},
        simulator_model={"type": "responses_api_models", "name": "sim"},
        interaction_model={"type": "responses_api_models", "name": "metrics"},
        **updates,
    )
    return SWETEnvironmentServer.model_construct(config=config)


def state(tmp_path, decisions):
    req = request()
    value = Episode(req, req.task.task_input.task_data, tmp_path, tmp_path)
    value.snapshots = SimpleNamespace(capture=AsyncMock(return_value="diff"))
    llm = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    content="", tool_calls=[{"function": {"name": action, "arguments": json.dumps({"content": text})}}]
                )
                for action, text in decisions
            ]
        )
    )
    value.simulator = UserAgent(llm, original_user_messages=["original"])
    value.simulator_model = SimpleNamespace(calls=[])
    value.interaction_started = monotonic()
    value.interaction_deadline = monotonic() + 30
    return value, llm


@pytest.mark.parametrize(
    ("value", "expected"), [("4G", 4096), ("8GiB", 8192), ("512M", 512), ("1.5G", 1536), ("1024K", 1)]
)
def test_memory_declarations(value, expected):
    assert memory_mib(value) == expected


@pytest.mark.parametrize("value", ["large", "0G", "-1G", "4", "4P"])
def test_invalid_memory_is_not_silently_changed(value):
    with pytest.raises(ValueError):
        memory_mib(value)


@pytest.mark.asyncio
async def test_simulator_one_call_even_on_noop_text(tmp_path):
    value, llm = state(tmp_path, [])
    llm.call.side_effect = None
    llm.call.return_value = SimpleNamespace(content="no-op", tool_calls=[])
    result = await server(tmp_path)._next_user_turn(value, activation())
    assert result.synthetic and result.responses_create_params.input[0].content == "continue"
    assert llm.call.await_count == 1


@pytest.mark.asyncio
async def test_simulator_turn_artifact_keeps_upstream_status(tmp_path):
    value, llm = state(tmp_path, [])
    upstream = ClientResponseError(
        SimpleNamespace(real_url="https://model.example/v1/chat/completions"),
        (),
        status=408,
        message="gateway timeout",
    )
    upstream.response_content = b"provider request timed out"

    llm.call.side_effect = upstream
    with pytest.raises(RuntimeError, match="User simulator request failed"):
        await server(tmp_path)._next_user_turn(value, activation())

    artifact = json.loads((tmp_path / "turn-0-simulator.json").read_text())
    detail = artifact["error_detail"]
    assert detail["message"] == "User simulator request failed"
    assert detail["cause"].startswith("ClientResponseError: 408")
    assert detail["status_code"] == 408
    assert detail["body"] == "provider request timed out"


@pytest.mark.asyncio
async def test_accepted_feedback_and_noop_stop(tmp_path):
    value, llm = state(tmp_path, [("redirect", "Fix tests")] + [("no-op", "")] * 4)
    environment = server(tmp_path)
    results = [await environment._next_user_turn(value, activation(i)) for i in range(5)]
    assert results[-1].stop_reason == "consecutive_noops"
    assert value.messages == [{"trial_idx": 1, "turn": 1, "action": "redirect", "text": "Fix tests"}]
    assert value.simulator.get_stats()["ground_truth_consumed"] == 1 and llm.call.await_count == 5


@pytest.mark.asyncio
async def test_already_expired_budget_stops_without_snapshot_or_simulator(tmp_path):
    value, llm = state(tmp_path, [])
    value.interaction_started = monotonic() - 60
    value.interaction_deadline = monotonic() - 1
    result = await server(tmp_path)._next_user_turn(value, activation())
    assert result.kind == "stop" and result.stop_reason == "session_budget_exhausted"
    value.snapshots.capture.assert_not_awaited()
    llm.call.assert_not_awaited()
    assert value.messages == []


@pytest.mark.asyncio
async def test_provider_timeout_is_not_an_interaction_budget_stop(tmp_path):
    value, _ = state(tmp_path, [])
    value.snapshots.capture.side_effect = TimeoutError("command status GET exceeded 10s")
    with pytest.raises(TimeoutError, match="status GET"):
        await server(tmp_path)._next_user_turn(value, activation())


@pytest.mark.asyncio
async def test_late_simulator_message_is_discarded(tmp_path):
    value, llm = state(tmp_path, [])
    value.interaction_started = monotonic()
    value.interaction_deadline = monotonic() + 0.03

    async def delayed(**kwargs):
        await asyncio.sleep(0.1)
        return SimpleNamespace(content="late feedback", tool_calls=[])

    llm.call.side_effect = delayed
    result = await server(tmp_path)._next_user_turn(value, activation())
    assert result.stop_reason == "session_budget_exhausted" and value.messages == []
    assert value.simulator._messages == []
    assert json.loads((tmp_path / "turn-0-simulator.json").read_text())["discarded"]


@pytest.mark.asyncio
@pytest.mark.parametrize("has_structured_error", [True, False])
async def test_refusal_preserves_semantic_outcome_and_close_before_verdict(
    tmp_path, monkeypatch, has_structured_error
):
    monkeypatch.setattr(evaluation, "uuid4", lambda: SimpleNamespace(hex="test"))
    monkeypatch.setattr(evaluation, "stage_judge", AsyncMock(return_value="judge"))
    monkeypatch.setattr(evaluation, "raise_for_status", AsyncMock())
    monkeypatch.setattr(
        evaluation,
        "get_response_json",
        AsyncMock(
            side_effect=[
                {},
                {
                    "status": "failed",
                    "error": {"message": "Usage Policy refusal"} if has_structured_error else None,
                    "metadata": {"stop_reason": "refusal"} if has_structured_error else None,
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Reading the rubric."}],
                        },
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Usage Policy refusal"}],
                        },
                    ],
                },
                {
                    "agent_session_id": "judge-test",
                    "agent_observations": {"source": "claude_code"},
                    "cleanup_confirmed": True,
                },
            ]
        ),
    )
    read = AsyncMock(side_effect=FileNotFoundError("no verdict"))
    monkeypatch.setattr(evaluation, "read_text", read)
    client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(cookies={})))
    evidence = {}
    req = request()
    with pytest.raises(RuntimeError, match="Usage Policy refusal"):
        await evaluation.run_judge(
            client=client,
            agent=SimpleNamespace(name="judge"),
            sandbox=SimpleNamespace(serialize=AsyncMock(return_value={})),
            provider="sandbox",
            task_dir=tmp_path,
            patch="patch",
            workdir="/workspace",
            episode=req.episode_id,
            task_id=req.task.task_id,
            timeout=100,
            cookies={},
            artifact_dir=tmp_path,
            evidence=evidence,
        )
    read.assert_awaited_once()
    seed = client.post.await_args_list[0].kwargs["json"]
    assert seed.sandbox_access.workdir == "/tmp/judge_inputs"
    assert evidence["judge_close"]["agent_observations"]["source"] == "claude_code"
    assert evidence["judge_close"]["cleanup_confirmed"] is True
    assert evidence["judge_outcome"]["metadata"] == ({"stop_reason": "refusal"} if has_structured_error else None)
    assert evidence["judge_outcome"]["terminal_text"] == "Usage Policy refusal"
    assert (tmp_path / "judge-close.json").exists()


ABSENT = SimpleNamespace(return_code=1, error_type=None)


async def run_closed_judge(
    tmp_path, monkeypatch, *, status, error_type, verdict_text=None, probe=ABSENT, evidence=None
):
    monkeypatch.setattr(evaluation, "uuid4", lambda: SimpleNamespace(hex="test"))
    monkeypatch.setattr(evaluation, "stage_judge", AsyncMock(return_value="judge"))
    monkeypatch.setattr(evaluation, "raise_for_status", AsyncMock())
    root = {"kind": "agent_invocation", "invocation_id": "root", "status": status, "error_type": error_type}
    child = {
        "kind": "agent_invocation",
        "invocation_id": "sub",
        "parent_invocation_id": "root",
        "error_type": "refusal",
    }
    close = {
        "agent_session_id": "judge-test",
        "agent_observations": {"source": "claude_code", "records": [child, root]},
        "cleanup_confirmed": True,
    }
    transcript = {"status": status, "output": [{"type": "message", "content": []}]}
    monkeypatch.setattr(evaluation, "get_response_json", AsyncMock(side_effect=[{}, transcript, close]))
    reader = (
        AsyncMock(return_value=verdict_text) if verdict_text else AsyncMock(side_effect=OSError("download failed"))
    )
    monkeypatch.setattr(evaluation, "read_text", reader)
    (tmp_path / "canonical_goals.json").write_text(json.dumps({"completeness_goals": [{"id": "goal", "weight": 1}]}))
    req = request()
    return await evaluation.run_judge(
        client=SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(cookies={}))),
        agent=SimpleNamespace(name="judge"),
        sandbox=SimpleNamespace(
            serialize=AsyncMock(return_value={}),
            exec=AsyncMock(side_effect=probe) if isinstance(probe, Exception) else AsyncMock(return_value=probe),
        ),
        provider="sandbox",
        task_dir=tmp_path,
        patch="patch",
        workdir="/workspace",
        episode=req.episode_id,
        task_id=req.task.task_id,
        timeout=100,
        cookies={},
        artifact_dir=tmp_path,
        evidence={} if evidence is None else evidence,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_type", "cause"),
    [
        ("failed", "refusal", "judge_refused"),  # 66-0, 75-0, 75-1: model stop_reason refusal
        ("incomplete", "error_max_turns", "judge_turn_limit"),  # 91-0: 50 turns, no verdict.json
        ("incomplete", "timeout", "judge_timeout"),
        ("completed", None, "judge_no_verdict"),
    ],
)
async def test_missing_verdict_names_the_judge_stop_cause(tmp_path, monkeypatch, status, error_type, cause):
    with pytest.raises(evaluation.JudgeMeasurementError, match="no readable verdict") as failure:
        await run_closed_judge(tmp_path, monkeypatch, status=status, error_type=error_type)
    assert failure.value.cause == cause


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe",
    [
        SimpleNamespace(return_code=0, error_type=None),  # verdict.json exists, so the download failed
        SimpleNamespace(return_code=124, error_type="timeout"),  # the sandbox could not answer
        OSError("sandbox gone"),
    ],
)
async def test_unconfirmed_missing_verdict_is_a_transfer_failure(tmp_path, monkeypatch, probe):
    # A refused judge with an unreadable verdict is not called a refusal unless the file is confirmed absent.
    evidence = {}
    with pytest.raises(evaluation.JudgeMeasurementError, match="read=OSError: download failed") as failure:
        await run_closed_judge(
            tmp_path, monkeypatch, status="failed", error_type="refusal", probe=probe, evidence=evidence
        )
    assert failure.value.cause == "judge_verdict_transfer_failed"
    assert evidence["judge_verdict_probe"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["{not json", "[]", json.dumps({"goal_results": [{"id": "other", "met": True}]})])
async def test_invalid_verdict_is_masked_and_raw_text_kept(tmp_path, monkeypatch, raw):
    with pytest.raises(evaluation.JudgeMeasurementError) as failure:
        await run_closed_judge(tmp_path, monkeypatch, status="completed", error_type=None, verdict_text=raw)
    assert failure.value.cause == "judge_verdict_invalid"
    assert (tmp_path / "judge-verdict-raw.json").read_text() == raw


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("patch", "error", "expected"),
    [
        ("x" * 100, evaluation.JudgeMeasurementError("submission_timeout", "freeze timed out"), "submission_timeout"),
        ("x" * 100, RuntimeError("unexpected"), "judge_error"),
        ("x" * 100, TimeoutError("provider poll deadline"), "judge_error"),
        ("short", None, None),
    ],
)
async def test_verify_keeps_judge_failure_cause_apart_from_valid_zero(tmp_path, monkeypatch, patch, error, expected):
    value, _ = state(tmp_path, [])
    value.snapshots = SimpleNamespace(capture=AsyncMock(), final_patch=patch, baseline=[])
    environment = server(tmp_path)
    environment.server_client = SimpleNamespace()
    monkeypatch.setattr(SWETEnvironmentServer, "_new_sandbox", AsyncMock())
    monkeypatch.setattr(app, "replicate_submission", AsyncMock(side_effect=error))
    monkeypatch.setattr(app, "interaction_metrics", AsyncMock(return_value={}))
    monkeypatch.setattr(app, "AuxiliaryModel", lambda *args, **kwargs: SimpleNamespace(calls=[]))
    initial = app.NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}])
    result = await environment._verify(value, initial, [activation()])
    assert result.get("judge_failure_cause") == expected
    if expected is None:
        # The upstream short-patch rule is a measured zero, not a missing measurement.
        assert result["judge_score"] == 0.0 and not result.get("mask_sample")
    else:
        assert result["mask_sample"] is True and result["judge_score"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_path", ["/v1/agent_sessions", "/ng-rollout/one-judge/v1/responses"])
@pytest.mark.parametrize("close_failure", [None, "unconfirmed", "request_error"])
async def test_verify_preserves_primary_judge_error_and_transport_classification(
    tmp_path, monkeypatch, failed_path, close_failure
):
    from aiohttp import ClientConnectionError

    from environment_servers.swe_together.tests.test_judge_errors import (
        CLOSE,
        CONFIRMED,
        UNCONFIRMED,
        http_error,
        run_judge,
    )

    value, _ = state(tmp_path, [])
    value.snapshots = SimpleNamespace(capture=AsyncMock(), final_patch="x" * 100, baseline=[])
    environment = server(tmp_path)
    environment.server_client = SimpleNamespace()
    primary = http_error(500)
    value.task_metadata = {"judge_timeout": 100}
    failures = {failed_path: primary}
    if close_failure == "request_error":
        failures[CLOSE] = ClientConnectionError("close lost")
    run, calls, evidence = await run_judge(
        tmp_path, monkeypatch, failures=failures, close=CONFIRMED if close_failure is None else UNCONFIRMED
    )

    async def grade(**kwargs):
        try:
            await run
        except Exception as error:
            assert error is primary
            raise
        finally:
            kwargs["evidence"].update(evidence)

    monkeypatch.setattr(app, "run_judge", grade)
    monkeypatch.setattr(SWETEnvironmentServer, "_new_sandbox", AsyncMock())
    monkeypatch.setattr(app, "replicate_submission", AsyncMock(return_value={}))
    monkeypatch.setattr(app, "interaction_metrics", AsyncMock(return_value={}))
    monkeypatch.setattr(app, "AuxiliaryModel", lambda *args, **kwargs: SimpleNamespace(calls=[]))
    initial = app.NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}])
    result = await environment._verify(value, initial, [activation()])
    assert calls[-1] == CLOSE
    assert result["mask_sample"] is True and result["judge_score"] is None
    assert result["judge_failure_cause"] == "judge_transport_error"
    assert result["failure_reason"] == f"ClientResponseError: {primary}"
    assert ("judge_close_error" in result) is (close_failure is not None)


@pytest.mark.asyncio
async def test_verify_names_grading_timeout_only_when_its_own_deadline_expires(tmp_path, monkeypatch):
    value, _ = state(tmp_path, [])
    value.snapshots = SimpleNamespace(capture=AsyncMock(), final_patch="x" * 100, baseline=[])
    environment = server(tmp_path, grading_timeout_seconds=0.05)
    environment.server_client = SimpleNamespace()

    async def slow_transfer(*args, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(SWETEnvironmentServer, "_new_sandbox", AsyncMock())
    monkeypatch.setattr(app, "replicate_submission", slow_transfer)
    monkeypatch.setattr(app, "interaction_metrics", AsyncMock(return_value={}))
    monkeypatch.setattr(app, "AuxiliaryModel", lambda *args, **kwargs: SimpleNamespace(calls=[]))
    initial = app.NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}])
    result = await environment._verify(value, initial, [activation()])
    assert result["judge_failure_cause"] == "grading_timeout" and result["mask_sample"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["request", "response_body"])
async def test_saved_judge_verdict_survives_lost_response(tmp_path, monkeypatch, failure_phase):
    from aiohttp import ClientConnectionError

    monkeypatch.setattr(evaluation, "uuid4", lambda: SimpleNamespace(hex="test"))
    monkeypatch.setattr(evaluation, "stage_judge", AsyncMock(return_value="judge"))
    monkeypatch.setattr(evaluation, "raise_for_status", AsyncMock())
    lost_response = ClientConnectionError("judge response lost")
    close = {"agent_session_id": "judge-test", "cleanup_confirmed": True}
    response = SimpleNamespace(cookies={})
    post = AsyncMock(side_effect=[response, lost_response if failure_phase == "request" else response, response])
    payloads = [{}, close] if failure_phase == "request" else [{}, lost_response, close]
    monkeypatch.setattr(evaluation, "get_response_json", AsyncMock(side_effect=payloads))
    raw = json.dumps({"goal_results": [{"id": "goal", "met": False}]})

    async def read(sandbox, *, path):
        assert post.await_args_list[-1].args[1] == "/v1/agent_sessions/close"
        assert path == "/tmp/judge_inputs/verdict.json"
        return raw

    monkeypatch.setattr(evaluation, "read_text", read)
    monkeypatch.setattr(evaluation, "read_test_reward", AsyncMock(return_value={"test_reward_raw": None}))
    (tmp_path / "canonical_goals.json").write_text(json.dumps({"completeness_goals": [{"id": "goal", "weight": 1}]}))
    req = request()
    evidence = {}
    verdict = await evaluation.run_judge(
        client=SimpleNamespace(post=post),
        agent=SimpleNamespace(name="judge"),
        sandbox=SimpleNamespace(serialize=AsyncMock(return_value={})),
        provider="sandbox",
        task_dir=tmp_path,
        patch="patch",
        workdir="/workspace",
        episode=req.episode_id,
        task_id=req.task.task_id,
        timeout=100,
        cookies={},
        artifact_dir=tmp_path,
        evidence=evidence,
    )
    assert verdict["judge_score"] == 0
    assert verdict["judge_runtime"] is None
    assert verdict["judge_close"]["cleanup_confirmed"] is True
    assert evidence["judge_transport_error"] == "ClientConnectionError: judge response lost"
    assert (tmp_path / "judge-verdict-raw.json").read_text() == raw


@pytest.mark.asyncio
async def test_webserver_lifespan_closes_runtime_helper(tmp_path):
    environment = server(tmp_path)
    environment.server_client = SimpleNamespace(global_config_dict={})
    runtime = SimpleNamespace(close=AsyncMock())
    environment._runtime = runtime
    webserver = environment.setup_webserver()
    async with webserver.router.lifespan_context(webserver):
        runtime.close.assert_not_awaited()
    runtime.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_sandbox_config_metadata_reaches_only_the_configured_role(tmp_path, monkeypatch):
    policy = {"defaultAction": "deny", "egress": [{"action": "allow", "target": "10.0.0.1"}]}
    environment = server(
        tmp_path,
        dependency_proxy_url="http://10.0.0.3:3128",
        sandbox_config={
            "metadata": {"nemo.nvidia.com/oom-kill": "process", "role": "spoofed"},
            "provider_options": {"network_policy": policy},
        },
        judge_sandbox_config={"metadata": {}},
    )
    environment.server_client = SimpleNamespace(global_config_dict={})
    specs = {}

    class Sandbox:
        def __init__(self, provider):
            pass

        async def start(self, spec):
            specs[spec.metadata["role"]] = spec

        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=0, error_type=None)

    monkeypatch.setattr(app, "AsyncSandbox", Sandbox)
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: object())
    monkeypatch.setattr(app, "install_trust_bundle", AsyncMock())
    episode = Episode(request(), request().task.task_input.task_data, tmp_path, tmp_path)
    episode.task_metadata = {"environment": {"memory": "4G"}}
    episode.runtime = SimpleNamespace(
        metadata=lambda: {"placement": "shared"}, mount_options=lambda *_: {"volumes": []}
    )
    episode.dependency_access = SimpleNamespace(
        certificate=AsyncMock(return_value="cert"), sandbox_settings=lambda token, policy, cert: (policy, {}, {})
    )
    await environment._new_sandbox(episode)
    await environment._new_sandbox(episode, judge=True)
    candidate, judge = specs["candidate"].metadata, specs["judge"].metadata
    assert candidate["nemo.nvidia.com/oom-kill"] == "process"
    assert candidate["placement"] == "shared" and judge["placement"] == "shared"
    assert "nemo.nvidia.com/oom-kill" not in judge
    assert specs["candidate"].resources.memory_mib == 4096


@pytest.mark.asyncio
@pytest.mark.parametrize("activation_error", [False, True])
async def test_local_lifecycle_closes_candidate_before_verification(tmp_path, monkeypatch, activation_error):
    environment = server(tmp_path)
    order = []
    sandbox = SimpleNamespace(serialize=AsyncMock(return_value={}))
    candidate_trajectory = {
        "task_id": "task",
        "rollout_id": "one",
        "turns": [
            {
                "invocation_id": "root",
                "task_id": "task",
                "rollout_id": "one",
                "turn_no": 1,
                "timestamp": 1.0,
                "step_count": 0,
                "answer": "candidate decision",
            }
        ],
    }

    async def prepare(self, value):
        value.sandbox = sandbox
        value.task_metadata = {"agent_kwargs": {}}
        value.prompt = "first instruction"
        order.append("prepare")

    async def stop(self, value, role):
        order.append(role + " sandbox close")
        value.cleanup_receipts[role + "_sandbox"] = {"cleanup_confirmed": True}

    async def verify(self, value, initial, activations):
        assert order[-1] == "agent close"
        order.append("verify")
        return {
            "responses_create_params": initial,
            "response": activations[-1].response,
            "reward": 1,
            "judge_score": 1,
            "task_id": "task",
            "evaluation_completed": True,
        }

    async def post(name, path, *, json, cookies, headers=None, _control=False):
        if path == "/v1/agent_sessions":
            order.append("seed")
            data = {
                "agent_session_id": json.agent_session_id,
                "capabilities": {"mode": "resume_harness_session", "observations": ["trajectory", "timing"]},
            }
        elif path == "/v1/agent_sessions/close":
            assert _control
            order.append("agent close")
            data = {
                "agent_session_id": json.agent_session_id,
                "cleanup_confirmed": True,
                "trajectory": candidate_trajectory,
            }
        else:
            order.append("activate")
            if activation_error:
                raise RuntimeError("native startup failed before provider inference")
            data = activation().response.model_dump(mode="json")
            assert json.input[0].content == "first instruction"
            assert headers == {"x-ng-agent-activation": "1"}
            assert 0 < float(json.metadata["agent_timeout_sec"]) <= environment.config.activation_timeout_seconds
        return SimpleNamespace(cookies={"agent_session_id": SimpleNamespace(value="cookie")}, payload=data)

    monkeypatch.setattr(SWETEnvironmentServer, "_prepare", prepare)
    monkeypatch.setattr(SWETEnvironmentServer, "_stop_owned_sandbox", stop)
    monkeypatch.setattr(SWETEnvironmentServer, "_verify", verify)
    monkeypatch.setattr(SWETEnvironmentServer, "_activation_path", lambda self, req: "/v1/responses")
    monkeypatch.setattr(
        SWETEnvironmentServer, "_next_user_turn", AsyncMock(return_value=UserTurn(activation_id=0, kind="stop"))
    )
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())

    async def payload(response):
        return response.payload

    monkeypatch.setattr(app, "get_response_json", payload)
    environment.server_client = SimpleNamespace(post=post)
    req = request()
    result = await environment.run_request(req)
    collected = _episode_record(result.model_dump(mode="json"))
    collector_task_id = "{'taskset': 'swet', 'task_id': 'task'}"
    assert collected["ng_trajectory"]["task_id"] == collector_task_id
    assert collected["ng_trajectory"]["turns"][0]["task_id"] == collector_task_id
    assert collected["ng_trajectory"]["turns"][0]["answer"] == "candidate decision"
    raw_close = json.loads((tmp_path / "one" / "candidate-agent-close.json").read_text())
    assert raw_close["trajectory"]["task_id"] == "task"
    assert raw_close["trajectory"]["turns"][0]["task_id"] == "task"
    if activation_error:
        assert result.result is None and result.failure.terminal
        assert result.failure.agent_close.cleanup_confirmed is True
        final = json.loads((tmp_path / "one" / "failure.json").read_text())
        assert final["agent_close"]["cleanup_confirmed"] is True
        assert final["agent_close"]["trajectory"]["task_id"] == "task"
        assert final["trajectory"]["task_id"] == collector_task_id
        assert final["cleanup_receipts"]["candidate_sandbox"]["cleanup_confirmed"] is True
        assert collected["ng_trajectory"]["rollout_id"] == "one"
        assert (tmp_path / "one" / "candidate-agent-close.json").exists()
        return
    assert result.result and result.result.reward == 1
    assert collected["reward"] == 1
    assert result.result.ng_agent_close.trajectory.task_id == "task"
    assert result.result.ng_trajectory.rollout_id == "one"
    assert collected["ng_trajectory"]["rollout_id"] == "one"
    assert "ng_agent_observations" in collected
    assert order == [
        "prepare",
        "seed",
        "activate",
        "agent close",
        "verify",
        "judge sandbox close",
        "candidate sandbox close",
    ]
    assert result.result.cleanup_receipts["candidate_sandbox"]["cleanup_confirmed"]


@pytest.mark.parametrize(
    "native_outcome,confirmed",
    [("interrupted", True), ("not_started", True), ("failed", True), ("interrupted", False)],
)
async def test_interaction_deadline_closes_remote_work_before_grading(
    tmp_path, monkeypatch, native_outcome, confirmed
):
    environment = server(tmp_path, interaction_timeout_seconds=0.03, cleanup_timeout_seconds=1)
    remote = None
    close_count = 0
    verified = False
    late_work = False

    async def work():
        nonlocal late_work
        await asyncio.sleep(0.1)
        late_work = True

    async def prepare(self, value):
        value.sandbox = SimpleNamespace(serialize=AsyncMock(return_value={}))
        value.prompt = "instruction"

    async def stop(self, value, role):
        value.cleanup_receipts[role + "_sandbox"] = {"cleanup_confirmed": True}

    async def verify(self, value, initial, activations):
        nonlocal verified
        verified = True
        assert close_count == 1 and remote.cancelled()
        turn = activations[-1]
        assert turn.response_source == "session_close"
        assert turn.response.status == "incomplete"
        assert turn.response.output == [] and turn.response.usage is None
        assert turn.response.metadata["harness_session_id"] == "native-one"
        return {
            "responses_create_params": initial,
            "response": turn.response,
            "reward": 0,
            "judge_score": 0,
            "task_id": "task",
            "evaluation_completed": True,
        }

    async def post(name, path, *, json, cookies, **kwargs):
        nonlocal remote, close_count
        if path == "/v1/agent_sessions":
            data = {
                "agent_session_id": json.agent_session_id,
                "capabilities": {"mode": "resume_harness_session", "observations": ["trajectory", "timing"]},
            }
        elif path == "/v1/agent_sessions/close":
            assert kwargs["_control"] is True
            close_count += 1
            remote.cancel()
            await asyncio.gather(remote, return_exceptions=True)
            records = (
                [
                    {
                        "kind": "agent_invocation",
                        "invocation_id": "native-one:1",
                        "activation_id": 1,
                        "harness_session_id": "native-one",
                        "status": "failed" if native_outcome == "failed" else "incomplete",
                        "error_type": "server_error" if native_outcome == "failed" else "cancelled",
                    }
                ]
                if native_outcome != "not_started"
                else []
            )
            data = {
                "agent_session_id": json.agent_session_id,
                "cleanup_confirmed": confirmed,
                "agent_observations": {"source": "opencode", "records": records},
            }
        else:
            # The real operation ledger keeps execution alive after an HTTP
            # waiter disconnects. Only explicit session close cancels that work.
            remote = asyncio.create_task(work())
            await asyncio.shield(remote)
            pytest.fail("A reply after the interaction deadline must not enter grading")
        return SimpleNamespace(cookies={"agent_session_id": SimpleNamespace(value="cookie")}, payload=data)

    monkeypatch.setattr(SWETEnvironmentServer, "_prepare", prepare)
    monkeypatch.setattr(SWETEnvironmentServer, "_stop_owned_sandbox", stop)
    monkeypatch.setattr(SWETEnvironmentServer, "_verify", verify)
    monkeypatch.setattr(SWETEnvironmentServer, "_activation_path", lambda self, req: "/v1/responses")
    simulator = AsyncMock()
    monkeypatch.setattr(SWETEnvironmentServer, "_next_user_turn", simulator)
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "get_response_json", AsyncMock(side_effect=lambda response: response.payload))
    environment.server_client = SimpleNamespace(post=post)
    result = await environment.run_request(request())
    assert remote.cancelled() and not late_work
    simulator.assert_not_awaited()
    if native_outcome == "interrupted" and confirmed:
        assert verified and result.result.reward == 0
        assert result.result.ng_steps[-1].stop_reason == "session_budget_exhausted"
        assert close_count == 1
    else:
        assert not verified and result.failure.terminal and result.result is None


@pytest.mark.parametrize("transient", [True, False])
async def test_setup_failures_preserve_retry_classification_and_evidence(tmp_path, monkeypatch, transient):
    from aiohttp import ClientConnectionError

    environment = server(tmp_path)
    error = ClientConnectionError("transport unavailable") if transient else ValueError("asset digest mismatch")
    monkeypatch.setattr(SWETEnvironmentServer, "_prepare", AsyncMock(side_effect=error))
    result = await environment.run_request(request())
    assert result.result is None
    assert result.failure.terminal is (not transient)
    assert result.failure.dependency_error["message"] == str(error)
    saved = json.loads((tmp_path / "one" / "failure.json").read_text())
    assert saved["failure_reason"].endswith(str(error))


async def test_wrapped_simulator_failure_keeps_upstream_status_in_failure(tmp_path, monkeypatch):
    from aiohttp import ClientResponseError

    url = "http://127.0.0.1:1/ng-rollout/one/v1/chat/completions"
    request_info = SimpleNamespace(url=url, real_url=url, method="POST", headers={})
    upstream = ClientResponseError(request_info=request_info, history=(), status=408, message="Request Timeout")
    upstream.response_content = b'{"error":{"message":"litellm.Timeout: Timeout passed=360.0"}}'
    error = RuntimeError("User simulator request failed")
    error.__cause__ = upstream
    monkeypatch.setattr(SWETEnvironmentServer, "_prepare", AsyncMock(side_effect=error))
    result = await server(tmp_path).run_request(request())
    saved = json.loads((tmp_path / "one" / "failure.json").read_text())["dependency_error"]
    assert saved == result.failure.dependency_error
    assert saved["status_code"] == 408 and "Timeout passed=360.0" in saved["body"]
    assert saved["cause"].startswith("ClientResponseError: 408")
