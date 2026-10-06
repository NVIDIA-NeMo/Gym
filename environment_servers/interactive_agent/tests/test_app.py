# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from http.cookies import SimpleCookie
from unittest.mock import MagicMock

import orjson
import pytest
from pydantic import BaseModel

from environment_servers.interactive_agent.app import (
    InteractiveAgentEnvironmentServer,
    InteractiveAgentEnvironmentServerConfig,
)
from nemo_gym.base_resources_server import ResourcesVerifyRequest
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.interactive_agent_types import (
    AgentActivationResponse,
    InteractiveAgentRequest,
    InteractiveAgentResponse,
    InteractiveVerificationInput,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


class Reply:
    ok = True

    def __init__(self, body, cookie="session"):
        self.body = orjson.dumps(body)
        self.cookies = SimpleCookie({"session": cookie})

    async def read(self):
        return self.body


def agent_response():
    return NeMoGymResponse(
        id="reply",
        created_at=0,
        model="test",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class ScriptedComposition:
    """Independent wire-only endpoints; no environment imports of Resources or harness code."""

    def __init__(self):
        self.calls = []
        self.activations = []
        self.inputs = []
        self.failure_path = None
        self.cleanup_confirmed = True
        self.capabilities = {"mode": "native_conversation", "observations": ["ordered_events", "timing"]}
        self.runtime_policy = None
        self.closed = False
        self.invalid_activation_id = False
        self.invalid_step_id = False
        self.mask = False
        self.block_activation = False
        self.entered = asyncio.Event()

    async def post(self, *, server_name, url_path, json, cookies=None):
        body = json.model_dump(mode="json") if isinstance(json, BaseModel) else json
        path = url_path.removeprefix("/ng-rollout/rollout-a2")
        self.calls.append((server_name, path, body, cookies))
        if path == self.failure_path:
            raise TimeoutError(f"lost reply: {path}")
        if path == "/seed_session":
            return Reply(
                {
                    "resources_session_id": body["resources_session_id"],
                    "responses_create_params": {"input": "first instruction"},
                    "runtime_policy": self.runtime_policy,
                },
                "resources-cookie",
            )
        if path == "/v1/agent_sessions":
            assert body["runtime_policy"] == self.runtime_policy
            assert body["continuation"] == {
                "mode": "native_conversation",
                "observations": ["ordered_events", "timing"],
            }
            return Reply(
                {"agent_session_id": body["agent_session_id"], "capabilities": self.capabilities}, "agent-cookie"
            )
        if path == "/v1/agent_sessions/activate":
            self.entered.set()
            if self.block_activation:
                await asyncio.Event().wait()
            assert cookies == {"session": "agent-cookie"}
            self.inputs.append(body["responses_create_params"]["input"])
            response = AgentActivationResponse(
                activation_id=body["activation_id"] + int(self.invalid_activation_id), response=agent_response()
            )
            self.activations.append(response.model_dump(mode="json"))
            return Reply(self.activations[-1], "agent-cookie")
        if path == "/step":
            assert cookies == {"session": "resources-cookie"}
            index = body["activation"]["activation_id"]
            return Reply(
                {
                    "activation_id": index + int(self.invalid_step_id),
                    "continue_episode": index == 0,
                    "responses_create_params": {"input": "correction"} if index == 0 else None,
                    "stop_reason": None if index == 0 else "resources_done",
                },
                "resources-cookie",
            )
        if path == "/v1/agent_sessions/close":
            self.closed = self.cleanup_confirmed
            return Reply(
                {
                    "agent_session_id": body["agent_session_id"],
                    "cleanup_confirmed": self.cleanup_confirmed,
                    "activations": self.activations,
                },
                "agent-cookie",
            )
        if path == "/verify":
            assert self.closed
            verify = ResourcesVerifyRequest[InteractiveVerificationInput].model_validate(body)
            assert verify.verification_input.agent_close.cleanup_confirmed
            assert len(verify.verification_input.activations) == 2
            return Reply(
                {
                    "reward": 1,
                    "mask_sample": self.mask,
                    "response": agent_response().model_dump(mode="json"),
                    "responses_create_params": {"input": "first instruction"},
                    "benchmark_metric": 0.85,
                },
                "resources-cookie",
            )
        if path == "/close_session":
            return Reply({"resources_session_id": body["resources_session_id"]}, "resources-cookie")
        raise AssertionError(path)


def request():
    return InteractiveAgentRequest.model_validate(
        {
            "episode_id": {"rollout_id": "rollout", "attempt": 2},
            "task": {
                "task_id": {"taskset": "interactive-fixture", "task_id": "repair"},
                "task_input": {"task_data": {"hidden": "resources only"}},
            },
        }
    )


def environment():
    script = ScriptedComposition()
    client = MagicMock(spec=ServerClient)
    client.post = script.post
    env = InteractiveAgentEnvironmentServer(
        config=InteractiveAgentEnvironmentServerConfig(
            name="interactive",
            host="localhost",
            port=1,
            entrypoint="app.py",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            agent_server=AgentServerRef(type="responses_api_agents", name="candidate"),
            cleanup_timeout_seconds=1,
        ),
        server_client=client,
    )
    return env, script


@pytest.mark.parametrize("mask", [False, True])
async def test_two_turn_composition_preserves_identity_cookies_verdict_and_order(mask):
    env, script = environment()
    script.mask = mask
    result = await env.run_request(request())
    assert result.failure is None
    assert result.result.reward == 1
    assert result.result.mask_sample == mask
    assert result.result.model_extra["benchmark_metric"] == 0.85
    assert script.inputs == ["first instruction", "correction"]
    assert [path for _, path, _, _ in script.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/v1/agent_sessions/activate",
        "/step",
        "/v1/agent_sessions/activate",
        "/step",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    assert all("hidden" not in orjson.dumps(body).decode() for name, _, body, _ in script.calls if name == "candidate")
    assert len(result.result.ng_activations) == len(result.result.ng_agent_close.activations) == 2
    assert result.result.ng_steps[-1].stop_reason == "resources_done"
    assert InteractiveAgentResponse.model_validate_json(result.model_dump_json()).model_dump() == result.model_dump()


async def test_runtime_policy_reaches_adapter_without_environment_interpretation():
    env, script = environment()
    script.runtime_policy = {
        "format": "independent-fixture.v3",
        "settings": {"nested": {"names": ["MixedCase", "native.name"], "enabled": False}, "limit": 7},
    }
    result = await env.run_request(request())
    assert result.failure is None and result.result.reward == 1
    agent_seed = next(body for _, path, body, _ in script.calls if path == "/v1/agent_sessions")
    assert agent_seed["runtime_policy"] == script.runtime_policy
    assert [path for _, path, _, _ in script.calls][-1] == "/close_session"


@pytest.mark.parametrize(
    "path,stage",
    [
        ("/seed_session", "seed"),
        ("/v1/agent_sessions", "agent"),
        ("/v1/agent_sessions/activate", "agent"),
        ("/step", "step"),
        ("/verify", "verification"),
    ],
)
async def test_dependency_failure_closes_sessions_and_preserves_failure_stage(path, stage):
    env, script = environment()
    script.failure_path = path
    result = await env.run_request(request())
    assert result.result is None and result.failure.stage == stage
    assert not result.failure.terminal
    paths = [path for _, path, _, _ in script.calls]
    assert paths[-1] == "/close_session"
    if stage != "seed":
        assert "/v1/agent_sessions/close" in paths
    if stage != "verification":
        assert "/verify" not in paths


@pytest.mark.parametrize("fault", ["cleanup_confirmed", "capabilities", "invalid_activation_id", "invalid_step_id"])
async def test_invalid_contract_cannot_reach_verification(fault):
    env, script = environment()
    setattr(script, fault, None if fault == "capabilities" else fault.startswith("invalid"))
    result = await env.run_request(request())
    assert result.result is None and result.failure.terminal
    assert "/verify" not in [path for _, path, _, _ in script.calls]


@pytest.mark.parametrize("cancel", [False, True])
async def test_interruption_closes_agent_before_resources(cancel):
    env, script = environment()
    env.config.default_episode_timeout_seconds = 0.03
    script.block_activation = True
    task = asyncio.create_task(env.run_request(request()))
    await script.entered.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result.failure.failure_reason == "Episode timed out"
    assert [path for _, path, _, _ in script.calls][-2:] == ["/v1/agent_sessions/close", "/close_session"]
