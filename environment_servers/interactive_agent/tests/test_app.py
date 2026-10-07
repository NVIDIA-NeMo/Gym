# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from http.cookies import SimpleCookie
from unittest.mock import MagicMock

import orjson
import pytest
from aiohttp import ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from pydantic import BaseModel
from yarl import URL

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
from nemo_gym.rollout_collection import _episode_record
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import ServerClient


class Reply:
    ok = True

    def __init__(self, body, cookie="session"):
        self.body = orjson.dumps(body)
        self.cookies = SimpleCookie({"session": cookie})

    async def read(self):
        return self.body


class ErrorReply(Reply):
    ok = False

    def __init__(self, body):
        super().__init__(body)
        self.content = self
        self.request_info = RequestInfo(
            url=URL("http://candidate/activate?credential=request-value"),
            real_url=URL("http://candidate/activate?credential=request-value"),
            method="POST",
            headers=CIMultiDictProxy(CIMultiDict({"Authorization": "Bearer request-header-value"})),
        )

    def raise_for_status(self):
        raise ClientResponseError(
            self.request_info,
            (),
            status=502,
            message="Bad Gateway",
            headers=CIMultiDict({"Set-Cookie": "response-header-value"}),
        )


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
        self.activation_failure = None
        self.close_observations = None
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
            if body["activation_id"] == 1 and self.activation_failure is not None:
                return self.activation_failure
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
                    "agent_observations": self.close_observations,
                    "resources_cookies": {"session": "resources-cookie"},
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


async def test_failed_activation_retains_http_body_and_deferred_close_observations():
    env, script = environment()
    script.activation_failure = ErrorReply(
        {
            "detail": {
                "message": "Native runtime ended without successful terminal result",
                "finish": "tool-calls",
                "stderr": "permission denied: /outside",
                "nested": {"api_key": "body-secret-value"},
                "headers": {"X-Custom": "body-header-value"},
                "url": "https://name:password@provider/error?credential=query-value",
            }
        }
    )
    script.close_observations = {
        "source": "independent-native-fixture",
        "records": [
            {
                "kind": "agent_invocation",
                "invocation_id": "native-session",
                "status": "failed",
                "error_type": "permission denied: /outside",
                "conversation": [{"role": "assistant", "content": "Partial diagnostic before failure"}],
            }
        ],
    }
    result = await env.run_request(request())
    assert result.result is None and result.failure.stage == "agent" and not result.failure.terminal
    failure = result.failure
    assert failure.dependency_error.status_code == 502
    detail = orjson.loads(failure.dependency_error.body)["detail"]
    assert detail["finish"] == "tool-calls" and detail["stderr"] == "permission denied: /outside"
    assert not failure.dependency_error.body_truncated
    assert failure.agent_close.cleanup_confirmed
    assert failure.agent_close.agent_observations == AgentObservationBundle.model_validate(script.close_observations)
    assert (
        failure.agent_close.agent_observations.records[0].conversation[0].content
        == "Partial diagnostic before failure"
    )
    assert len(failure.activations) == len(failure.agent_close.activations) == 1
    assert failure.partial_response == failure.activations[0].response
    assert [path for _, path, _, _ in script.calls][-2:] == ["/v1/agent_sessions/close", "/close_session"]
    serialized = result.model_dump_json()
    for sensitive in [
        "request-header-value",
        "response-header-value",
        "request-value",
        "body-secret-value",
        "body-header-value",
        "name:password",
        "query-value",
        "resources-cookie",
    ]:
        assert sensitive not in serialized
    assert InteractiveAgentResponse.model_validate_json(serialized).failure == failure
    collected = _episode_record(orjson.loads(serialized))
    assert collected["_ng_failure"] == orjson.loads(serialized)["failure"]
    assert collected["_ng_failure"]["agent_close"]["agent_observations"]["records"][0]["status"] == "failed"
    assert collected["_ng_failure"]["dependency_error"]["status_code"] == 502


async def test_dependency_body_is_bounded_and_redacted_when_not_json():
    env, script = environment()
    reply = ErrorReply({})
    reply.body = b'Authorization: Bearer body-auth-value\napi_key="body-key-value"\n' + b"x" * 40000
    script.activation_failure = reply
    result = await env.run_request(request())
    detail = result.failure.dependency_error
    assert detail.status_code == 502 and detail.body_truncated and len(detail.body) == 8192
    assert "body-auth-value" not in detail.body and "body-key-value" not in detail.body
    assert result.failure.agent_close.cleanup_confirmed


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
