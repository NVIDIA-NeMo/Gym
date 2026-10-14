# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from environment_servers.swe_together import app, evaluation
from environment_servers.swe_together.app import Episode, SWETConfig, SWETEnvironmentServer, memory_mib
from environment_servers.swe_together.schemas import SWETRequest, UserTurn
from environment_servers.swe_together.simulator import UserAgent
from environment_servers.swe_together.task import TaskData, runtime_policy
from nemo_gym.agent_activation_types import AgentActivationResponse, InteractionBudget
from nemo_gym.base_environment_server import CleanupContext
from nemo_gym.openai_utils import NeMoGymResponse


def activation(index=0, **kwargs):
    return AgentActivationResponse(
        activation_id=index,
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
    value.interaction_budget = InteractionBudget(started_at_unix_seconds=time(), deadline_unix_seconds=time() + 30)
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


def test_task_policy_translation_is_local():
    assert runtime_policy({"disallowed_tools": "WebFetch, WebSearch"}) == {
        "format": "opencode.permissions.v1",
        "settings": {"deny_tools": ["webfetch", "websearch"]},
    }
    with pytest.raises(ValueError):
        runtime_policy({"unknown_option": True})


@pytest.mark.asyncio
async def test_simulator_one_call_even_on_noop_text(tmp_path):
    value, llm = state(tmp_path, [])
    llm.call.side_effect = None
    llm.call.return_value = SimpleNamespace(content="no-op", tool_calls=[])
    result = await server(tmp_path)._next_user_turn(value, activation())
    assert result.synthetic and result.responses_create_params.input[0].content == "continue"
    assert llm.call.await_count == 1


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
    value.interaction_budget = InteractionBudget(started_at_unix_seconds=time() - 60, deadline_unix_seconds=time() - 1)
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
    value.interaction_budget = InteractionBudget(started_at_unix_seconds=time(), deadline_unix_seconds=time() + 0.03)

    async def delayed(**kwargs):
        await asyncio.sleep(0.1)
        return SimpleNamespace(content="late feedback", tool_calls=[])

    llm.call.side_effect = delayed
    result = await server(tmp_path)._next_user_turn(value, activation())
    assert result.stop_reason == "session_budget_exhausted" and value.messages == []
    assert value.simulator._messages == []
    assert json.loads((tmp_path / "turn-0-simulator.json").read_text())["discarded"]


@pytest.mark.asyncio
async def test_refusal_preserves_semantic_outcome_and_close_before_verdict(tmp_path, monkeypatch):
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
                    "error": {"message": "Usage Policy refusal"},
                    "metadata": {"stop_reason": "refusal"},
                },
                {"cleanup_confirmed": True},
            ]
        ),
    )
    read = AsyncMock()
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
    read.assert_not_awaited()
    assert evidence["judge_close"]["cleanup_confirmed"]
    assert evidence["judge_outcome"]["metadata"]["stop_reason"] == "refusal"
    assert (tmp_path / "judge-close.json").exists()


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
async def test_local_lifecycle_closes_candidate_before_verification(tmp_path, monkeypatch):
    environment = server(tmp_path)
    order = []
    sandbox = SimpleNamespace(serialize=AsyncMock(return_value={}))

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
        }

    async def post(name, path, *, json, cookies):
        if path == "/v1/agent_sessions":
            order.append("seed")
            data = {
                "agent_session_id": json.agent_session_id,
                "capabilities": {"mode": "native_conversation", "supports_interaction_budget": True},
            }
        elif path == "/v1/agent_sessions/close":
            order.append("agent close")
            data = {"agent_session_id": json.agent_session_id, "cleanup_confirmed": True}
        else:
            order.append("activate")
            data = activation().model_dump(mode="json")
            assert json.responses_create_params.input[0].content == "first instruction"
        return SimpleNamespace(cookies={}, payload=data)

    monkeypatch.setattr(SWETEnvironmentServer, "_prepare", prepare)
    monkeypatch.setattr(SWETEnvironmentServer, "_stop_owned_sandbox", stop)
    monkeypatch.setattr(SWETEnvironmentServer, "_verify", verify)
    monkeypatch.setattr(SWETEnvironmentServer, "_activation_path", lambda self, req: "/activate")
    monkeypatch.setattr(
        SWETEnvironmentServer, "_next_user_turn", AsyncMock(return_value=UserTurn(activation_id=0, kind="stop"))
    )
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())

    async def payload(response):
        return response.payload

    monkeypatch.setattr(app, "get_response_json", payload)
    environment.server_client = SimpleNamespace(post=post)
    req = request()
    result = await environment.run(req, CleanupContext(req.episode_id, 5))
    assert result.result and result.result.reward == 1
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
