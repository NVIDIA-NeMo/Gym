# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle and evidence tests; verdict fixtures test stopping, not semantic grading."""

import asyncio
import copy
import json
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from benchmarks.partner_if.tests.test_replay import PARAMS, SOURCE, call, probe, response
from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from responses_api_agents.ngc_recorded_context import native_agent as adapter
from responses_api_agents.ngc_recorded_context.replay import ReplayConfig


def seed(identity="step1", **overrides):
    return AgentSeedSessionRequest(
        agent_session_id=identity,
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task_id=TaskId(taskset="test", task_id="pif_029"),
        **overrides,
    )


def make_agent(tmp_path):
    config = adapter.NativeReplayAgentConfig.model_validate(
        {
            "name": "partner_if_ngc_replay_agent",
            "host": "127.0.0.1",
            "port": 9999,
            "entrypoint": "native_agent.py",
            "num_workers": 1,
            "model": "test",
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
            "replay": ReplayConfig(
                package_dir=SOURCE / "assets", artifacts_dir=tmp_path, coverage_profile="expanded-v4", max_new_turns=6
            ).model_dump(),
        }
    )
    return adapter.NativeReplayAgent(config=config, server_client=MagicMock(spec=ServerClient))


def request_for(identity="step1", capture_key="rollout-a1"):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"x-ng-agent-activation", b"1")],
            "session": {"agent_session_id": identity},
            "path_params": {"rollout_id": capture_key},
        }
    )


def activation(turn=1, previous=None):
    value = copy.deepcopy(PARAMS) | {"parent_turn_limit": turn}
    if previous:
        value.update(saved_prefix=previous.replay_artifacts_dir, saved_prefix_sha256=previous.saved_prefix_sha256)
    return adapter.ReplayActivation.model_validate(value)


def provider(monkeypatch, queue, seen):
    async def invoke(method, endpoint, **kwargs):
        seen.append((method, endpoint, copy.deepcopy(kwargs)))
        assert queue, "Unexpected duplicate or post-decision inference"
        raw = queue.pop(0)
        if isinstance(raw, Exception):
            raise raw
        raw["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}

        class Result:
            status = 200
            cookies = SimpleCookie("model_cookie=kept")

            async def json(self, **_):
                return raw

            def release(self):
                pass

        return Result()

    monkeypatch.setattr(adapter, "http_request", invoke)
    monkeypatch.setattr(
        adapter.NativeReplayAgent,
        "resolve_model_base_url",
        lambda self, model, rollout: f"http://gym/ng-rollout/{rollout}/v1",
    )


@pytest.mark.parametrize(
    "override,reason",
    [
        (
            {
                "tool_accesses": [
                    {"kind": "direct_http", "name": "external", "required": True, "base_url": "http://unused.invalid"}
                ]
            },
            "required external grants",
        ),
        ({"agent_session_id": "../escape"}, "safe single path"),
        (
            {
                "sandbox_access": {
                    "connection": {"kind": "direct", "provider_config_ref": "unused", "descriptor": {}},
                    "workdir": "/workspace",
                }
            },
            "does not consume a task sandbox",
        ),
        ({"continuation": {}}, "does not support the required multi-turn continuation mode"),
    ],
)
def test_seed_refuses_unsupported_contract_through_http(tmp_path, override, reason):
    agent = make_agent(tmp_path)
    if "agent_session_id" in override:
        value = seed().model_copy(update=override)
    else:
        value = seed(**override)
    assert reason in check_agent_seed_rejected(agent.setup_webserver(), value)
    assert not agent._session_records


@pytest.mark.asyncio
async def test_seed_activation_retry_close_and_unknown_session(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    seen = []
    provider(monkeypatch, [response(probe())], seen)
    seed_request = SimpleNamespace(session={})
    await agent.seed_agent_session(seed_request, seed())
    await agent.seed_agent_session(seed_request, seed())
    req, body = request_for(), activation()
    reply = await agent.responses(req, body)
    assert (await agent.responses(req, body)).model_dump() == reply.model_dump()
    assert len(seen) == 1
    assert seen[0][1] == "http://gym/ng-rollout/rollout-a1/v1/chat/completions"
    assert seen[0][2]["headers"]["x-session-id"] == "step1"
    assert seen[0][2]["_max_num_tries"] == 1
    assert seen[0][2]["_max_connection_retries"] == 0
    assert not seen[0][2]["_resend_after_disconnect"]
    assert "verifier_metadata" not in json.dumps(seen[0][2]["json"])
    assert reply.usage.total_tokens == 120
    assert reply.usage.input_tokens_details.cached_tokens is None
    trajectory = reply.model_extra["_ng_trajectory"]
    assert len(trajectory["turns"]) == len(trajectory["tool_calls"]) == 1
    assert trajectory["tool_calls"][0]["timing_source"] == "executor"
    assert trajectory["tool_calls"][0]["duration_ms"] >= 0
    assert trajectory["turns"][0]["question"] == seen[0][2]["json"]["messages"]
    saved_call = trajectory["invocations"][0]["conversation"][0]
    raw_call = trajectory["turns"][0]["answer"]["tool_calls"][0]
    assert saved_call["name"] == raw_call["function"]["name"]
    assert saved_call["arguments"] == raw_call["function"]["arguments"]
    assert saved_call["call_id"] == raw_call["id"]
    for other in (activation(2),):
        with pytest.raises(HTTPException) as error:
            await agent.responses(req, other)
        assert error.value.status_code == 409
    with pytest.raises(HTTPException) as error:
        await agent.responses(request_for(capture_key="another"), body)
    assert error.value.status_code == 409
    closed = await agent.close_agent_session(
        req, AgentCloseSessionRequest(agent_session_id="step1", episode_id=seed().episode_id)
    )
    again = await agent.close_agent_session(
        req, AgentCloseSessionRequest(agent_session_id="step1", episode_id=seed().episode_id)
    )
    assert closed == again and closed.cleanup_confirmed
    assert len(closed.agent_observations.records) == 2
    for identity in ("step1", "missing"):
        with pytest.raises(HTTPException) as error:
            await agent.responses(request_for(identity), body)
        assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_resume_reconstructs_old_calls_without_new_observations_or_inference(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    seen = []
    provider(
        monkeypatch,
        [response(call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"})), response(probe())],
        seen,
    )
    first = await agent._seed_agent_session_state(seed())
    initial = await agent._execute_step(request_for(), first, activation())
    second = await agent._seed_agent_session_state(seed("step2"))
    resumed = await agent._execute_step(request_for("step2"), second, activation(2, initial))
    assert len(seen) == 2 and resumed.replay_diagnostics["model_calls"] == 2
    assert len(resumed.model_extra["_ng_trajectory"]["turns"]) == 1
    assert len(second.records) == 2 and resumed.usage.total_tokens == 120
    assert second.records[0].invocation_id == "step2"
    first_metadata = json.loads((adapter.Path(initial.replay_artifacts_dir) / "identity.json").read_text())
    next_metadata = json.loads((adapter.Path(resumed.replay_artifacts_dir) / "identity.json").read_text())
    assert first_metadata["session_metadata"] == next_metadata["session_metadata"]
    assert (
        json.loads((adapter.Path(resumed.replay_artifacts_dir) / "native-step.json").read_text())[
            "reused_policy_calls"
        ]
        == 1
    )


@pytest.mark.asyncio
async def test_worker_calls_keep_capture_owners_cookies_and_actual_tool_failures(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    seen = []
    queue = [
        response(
            call(
                "Agent",
                {"description": "Inspect source", "prompt": "Read the skill", "subagent_type": "Explore"},
                "worker",
            )
        ),
        response(call("Bash", {"command": "ls /opt/repos/exampleproject/libraries/exampleproject_worker"}, "missing")),
        response(content="The directory does not exist."),
    ]
    provider(monkeypatch, queue, seen)
    state = await agent._seed_agent_session_state(seed())
    reply = await agent._execute_step(request_for(), state, activation())
    assert len(seen) == 3
    assert seen[1][2]["headers"]["x-session-id"].startswith("step1:delegate-")
    assert seen[2][2]["cookies"] == {"model_cookie": "kept"}
    traj = reply.model_extra["_ng_trajectory"]
    child = next(i for i in traj["invocations"] if i["parent_invocation_id"])
    assert child["parent_invocation_id"] == "step1" and child["spawned_by_tool_call_id"] == "worker"
    failed = next(t for t in traj["tool_calls"] if t["tool_call_id"] == "missing")
    assert failed["status"] == "failed" and failed["error_type"].startswith("tool_exit_")
    assert reply.usage.total_tokens == 360


@pytest.mark.asyncio
async def test_parent_and_worker_reusing_call_id_keep_results_in_their_conversation(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    seen = []
    provider(
        monkeypatch,
        [
            response(call("Agent", {"prompt": "Read the skill", "subagent_type": "Explore"}, "same")),
            response(call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"}, "same")),
            response(content="Worker found the registry instructions."),
            response(probe()),
        ],
        seen,
    )
    first = await agent._seed_agent_session_state(seed())
    reply = await agent._execute_step(request_for(), first, activation())
    trajectory = reply.model_extra["_ng_trajectory"]
    child = next(i for i in trajectory["invocations"] if i["parent_invocation_id"])
    assert child["spawned_by_tool_call_id"] == "same"
    outputs = {t["tool_name"]: t for t in trajectory["tool_calls"] if t["tool_call_id"] == "same"}
    assert outputs["Agent"]["invocation_id"] == "step1"
    assert "Worker found" in outputs["Agent"]["output"]
    assert outputs["Read"]["invocation_id"] == child["invocation_id"]
    assert "Worker found" not in outputs["Read"]["output"]
    assert reply.usage.total_tokens == 360
    second = await agent._seed_agent_session_state(seed("step2"))
    resumed = await agent._execute_step(request_for("step2"), second, activation(2, reply))
    assert len(seen) == 4
    fresh = resumed.model_extra["_ng_trajectory"]["tool_calls"]
    assert all(t["tool_call_id"] != "same" for t in fresh)


@pytest.mark.asyncio
async def test_provider_failure_is_not_resumable_and_close_preserves_observations(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    seen = []
    provider(monkeypatch, [TimeoutError("test provider timeout")], seen)
    state = await agent._seed_agent_session_state(seed())
    reply = await agent._execute_step(request_for(), state, activation())
    assert reply.saved_prefix_sha256 is None
    assert reply.usage is None and not reply.replay_diagnostics["evaluation_completed"]
    close = await agent._close_agent_session_state(state)
    assert close.cleanup_confirmed and close.agent_observations.records[0].status == "failed"
    assert any(g.code == "usage_unavailable" for g in close.agent_observations.gaps)


@pytest.mark.asyncio
async def test_close_cancels_inflight_call_and_keeps_failed_activation_from_retrying(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    began = asyncio.Event()

    async def waiting(*args, **kwargs):
        began.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(adapter, "http_request", waiting)
    monkeypatch.setattr(adapter.NativeReplayAgent, "resolve_model_base_url", lambda *args: "http://gym/v1")
    await agent.seed_agent_session(SimpleNamespace(session={}), seed())
    running = asyncio.create_task(agent.responses(request_for(), activation()))
    await asyncio.wait_for(began.wait(), 2)
    close = await asyncio.wait_for(
        agent.close_agent_session(
            request_for(), AgentCloseSessionRequest(agent_session_id="step1", episode_id=seed().episode_id)
        ),
        2,
    )
    assert running.cancelled() and close.cleanup_confirmed
    assert close.agent_observations.records[0].status == "incomplete"


def test_missing_usage_never_becomes_zero():
    assert adapter.summed_usage([]) is None
    assert adapter.summed_usage([{"prompt_tokens": 1}]) is None


def test_gym_transport_reasoning_does_not_become_visible_advice():
    raw = response(probe(), content="<think>Consider publishing before checking.</think>Checking the namespace.")
    original = copy.deepcopy(raw)
    decoded = adapter.decode_gym_reasoning(raw)
    assert raw == original
    message = decoded["choices"][0]["message"]
    assert message["content"] == "Checking the namespace."
    assert message["reasoning_content"] == "Consider publishing before checking."
    assert message["tool_calls"] == raw["choices"][0]["message"]["tool_calls"]
    assert (
        adapter.decode_gym_reasoning(response(content="Literal example: <think>x</think>"))["choices"][0]["message"][
            "content"
        ]
        == "Literal example: <think>x</think>"
    )


@pytest.mark.asyncio
async def test_decoded_reasoning_is_retained_in_trace_and_history_but_not_verifier_input(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    agent.config.replay.gym_reasoning_decode = True
    seen = []
    provider(monkeypatch, [response(probe(), content="<think>Publish now? Check the sibling first.</think>")], seen)
    state = await agent._seed_agent_session_state(seed())
    reply = await agent._execute_step(request_for(), state, activation())
    assert "Publish now?" not in json.dumps(reply.verification_input)
    assert all(item.type != "message" for item in reply.output)
    assert (
        reply.model_extra["_ng_trajectory"]["turns"][0]["reasoning_content"] == "Publish now? Check the sibling first."
    )
