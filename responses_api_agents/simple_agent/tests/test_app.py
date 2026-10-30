# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, call

import httpx
import orjson
import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pytest import MonkeyPatch
from yarl import URL

from nemo_gym.agents import (
    AGENT_ACTIVATION_HEADER,
    AGENT_OUTCOME_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentOutcome,
    AgentSeedSessionRequest,
)
from nemo_gym.agents.base_responses_api_agent import AGENT_SESSION_COOKIE_KEY
from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
)
from nemo_gym.core import (
    EpisodeId,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    TaskId,
)
from nemo_gym.rollouts.rollout_collection import _attach_trajectory_record
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    DirectHTTPToolAccess,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    RolloutIdRequiredError,
    SimpleResourcesServer,
)
from nemo_gym.server_runtime._checkpoint.agent import AgentSessionParticipant, AgentSessionRecord
from nemo_gym.server_runtime._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ParticipantControlPlane,
    RestoreRequest,
)
from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant
from nemo_gym.server_runtime._checkpoint.errors import StaleAttemptError
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from nemo_gym.trajectory import TrajectoryRecord
from responses_api_agents.simple_agent.app import (
    ModelServerRef,
    ResourcesServerRef,
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentLoopState,
    SimpleAgentRunRequest,
    SimpleAgentVerifyRequest,
)


def _drop_nulls(value):
    """Remove dictionary entries with a value of ``None`` recursively.

    SDK releases can add optional response fields at any depth.
    Exact comparisons should ignore these unset fields.
    Expected non-null values remain part of the comparison.
    """
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


def _make_agent(
    observability_enabled: bool, agent_type: type[SimpleAgent] = SimpleAgent
) -> tuple[SimpleAgent, MagicMock]:
    config = SimpleAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="simple",
        model_server=ModelServerRef(type="responses_api_models", name="model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"observability_enabled": observability_enabled}
    return agent_type(config=config, server_client=server_client), server_client


def _mock_response(payload=None, *, status=200, content="") -> MagicMock:
    response = MagicMock(status=status, cookies={}, ok=status < 400)
    response.read = AsyncMock(return_value=json.dumps(payload or {}))
    response.content.read = AsyncMock(return_value=content.encode())
    return response


class TestApp:
    def test_resource_only_metadata_is_not_sent_to_policy(self) -> None:
        body = NeMoGymResponseCreateParamsNonStreaming(
            input="hello",
            metadata={"baseline_response": "reference", "keep": "value"},
        )

        stripped = SimpleAgent._strip_resource_only_metadata(body)

        assert stripped.metadata == {"keep": "value"}
        assert body.metadata["baseline_response"] == "reference"

    def test_sanity(self) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_responses(self, monkeypatch: MonkeyPatch) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        server.server_client.global_config_dict = {"observability_enabled": True}
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.return_value = json.dumps(mock_response_data)
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200
        server.server_client.post.assert_called_with(
            server_name="my server name",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
            ),
            cookies=None,
        )

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
            "conversation": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
        }
        assert _drop_nulls(expected_responses_dict) == _drop_nulls(actual_responses_dict)

        prefixed_response = client.post(
            "/ng-rollout/0-0/v1/responses", json={"input": [{"role": "user", "content": "hello"}]}
        )
        assert prefixed_response.status_code == 200
        assert prefixed_response.json()["_ng_trajectory"]["rollout_id"] == "0-0"

    @pytest.mark.parametrize("observability_enabled", [False, True])
    async def test_agent_session_uses_seeded_direct_http_tool_access(
        self, monkeypatch: MonkeyPatch, observability_enabled: bool
    ) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="simple",
            model_server=ModelServerRef(type="responses_api_models", name="model"),
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {"observability_enabled": observability_enabled}
        server = SimpleAgent(config=config, server_client=server_client)

        response_base = {
            "created_at": 1.0,
            "model": "model",
            "object": "response",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        server_client.post = AsyncMock(
            side_effect=[
                _mock_response(
                    response_base
                    | {
                        "id": "resp-tool",
                        "output": [
                            {
                                "id": "fc-1",
                                "call_id": "call-1",
                                "name": "get_weather",
                                "arguments": '{"city":"San Francisco"}',
                                "type": "function_call",
                                "status": "completed",
                            }
                        ],
                    }
                ),
                _mock_response(
                    response_base
                    | {
                        "id": "resp-final",
                        "output": [
                            {
                                "id": "msg-1",
                                "content": [{"annotations": [], "text": "Cold.", "type": "output_text"}],
                                "role": "assistant",
                                "status": "completed",
                                "type": "message",
                            }
                        ],
                    }
                ),
            ]
        )
        tool_response = _mock_response(content='{"city":"San Francisco","weather_description":"cold"}')
        tool_response.cookies = {"session_id": MagicMock(value="updated-resource-cookie")}
        direct_request = AsyncMock(return_value=tool_response)
        monkeypatch.setattr("responses_api_agents.simple_agent.app.http_request", direct_request)

        app = server.setup_webserver()
        client = TestClient(app)
        episode_id = EpisodeId(rollout_id="rollout", attempt=0)
        seed = client.post(
            "/v1/agent_sessions",
            json=AgentSeedSessionRequest(
                agent_session_id="agent-session",
                episode_id=episode_id,
                task_id=TaskId(taskset="example", task_id="0"),
                tool_accesses=[
                    DirectHTTPToolAccess(
                        name="weather.direct_http",
                        required=True,
                        base_url="http://resources:8000",
                        cookies={"session_id": "seeded-resource-cookie"},
                        headers={"X-Scoped-Access": "token"},
                    )
                ],
            ).model_dump(mode="json"),
        )
        assert seed.status_code == 200

        # The Environment Server calls the attempt-qualified route, which enables trajectory collection.
        result = client.post(
            "/ng-rollout/rollout/v1/responses" if observability_enabled else "/v1/responses",
            json={
                "input": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }
                ],
            },
        )
        assert result.status_code == 200
        assert result.json()["output"][-1]["content"][0]["text"] == "Cold."
        direct_request.assert_awaited_once_with(
            method="POST",
            url="http://resources:8000/get_weather",
            json={"city": "San Francisco"},
            cookies={"session_id": "seeded-resource-cookie"},
            headers={"X-Scoped-Access": "token"},
            _internal=True,
        )
        assert [item.kwargs["server_name"] for item in server_client.post.await_args_list] == ["model", "model"]

        close = client.post(
            "/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(
                agent_session_id=seed.json()["agent_session_id"],
                episode_id=episode_id,
            ).model_dump(mode="json"),
        )
        assert close.status_code == 200
        assert close.json()["resources_cookies"] == {"session_id": "updated-resource-cookie"}
        # A session returns its agent evidence at close, which becomes the episode's ng_agent_observations.
        observations = close.json()["agent_observations"]
        if observability_enabled:
            assert [record["kind"] for record in observations["records"]] == ["agent_invocation", "tool_call"]
            tool = observations["records"][1]
            assert tool["tool_name"] == "get_weather"
            assert tool["status"] == "completed"
            assert tool["invocation_id"] == observations["records"][0]["invocation_id"]
            assert observations["source"] == "simple_agent"
        else:
            assert observations is None

    async def test_agent_session_returns_observations_from_every_activation(self) -> None:
        server, server_client = _make_agent(True)
        response_base = {
            "created_at": 1.0,
            "model": "model",
            "object": "response",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        def message_response(response_id: str, text: str) -> MagicMock:
            return _mock_response(
                response_base
                | {
                    "id": response_id,
                    "output": [
                        {
                            "id": f"msg-{response_id}",
                            "content": [{"annotations": [], "text": text, "type": "output_text"}],
                            "role": "assistant",
                            "status": "completed",
                            "type": "message",
                        }
                    ],
                }
            )

        server_client.post = AsyncMock(
            side_effect=[message_response("resp-1", "First."), message_response("resp-2", "Second.")]
        )
        client = TestClient(server.setup_webserver())
        episode_id = EpisodeId(rollout_id="rollout", attempt=0)
        seed = client.post(
            "/v1/agent_sessions",
            json=AgentSeedSessionRequest(
                agent_session_id="agent-session",
                episode_id=episode_id,
                task_id=TaskId(taskset="example", task_id="0"),
            ).model_dump(mode="json"),
        )
        assert seed.status_code == 200

        for question in ("first?", "second?"):
            result = client.post("/ng-rollout/rollout/v1/responses", json={"input": question})
            assert result.status_code == 200

        close = client.post(
            "/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(
                agent_session_id=seed.json()["agent_session_id"], episode_id=episode_id
            ).model_dump(mode="json"),
        )

        assert close.status_code == 200
        records = close.json()["agent_observations"]["records"]
        assert [record["invocation_id"] for record in records] == ["root", "activation-2"]
        assert [record["model_calls"][0]["response_id"] for record in records] == ["resp-1", "resp-2"]

    async def test_agent_session_can_return_unresolved_tool_calls(self) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="simple",
            model_server=ModelServerRef(type="responses_api_models", name="model"),
            execute_tools=False,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {"observability_enabled": False}
        server_client.post = AsyncMock(
            return_value=_mock_response(
                {
                    "id": "resp-tool",
                    "created_at": 1.0,
                    "model": "model",
                    "object": "response",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [
                        {
                            "id": "fc-1",
                            "call_id": "call-1",
                            "name": "get_weather",
                            "arguments": '{"city":"San Francisco"}',
                            "type": "function_call",
                            "status": "completed",
                        }
                    ],
                }
            )
        )
        client = TestClient(SimpleAgent(config=config, server_client=server_client).setup_webserver())
        seed = client.post(
            "/v1/agent_sessions",
            json=AgentSeedSessionRequest(
                agent_session_id="agent-session",
                episode_id=EpisodeId(rollout_id="rollout", attempt=0),
                task_id=TaskId(taskset="example", task_id="0"),
            ).model_dump(mode="json"),
        )
        assert seed.status_code == 200

        result = client.post("/v1/responses", json={"input": "weather?"})

        assert result.status_code == 200
        assert [item["type"] for item in result.json()["output"]] == ["function_call"]
        assert result.json()["output"][0]["call_id"] == "call-1"
        server_client.post.assert_awaited_once()
        assert server_client.post.await_args.kwargs["server_name"] == "model"

    def test_execute_tools_defaults_to_true(self) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="simple",
            model_server=ModelServerRef(type="responses_api_models", name="model"),
        )

        assert config.execute_tools is True
        assert "agent-session" in SimpleAgentConfig.model_fields["execute_tools"].description

    async def test_execute_tools_false_is_rejected_outside_agent_session(self) -> None:
        server, server_client = _make_agent(False)
        server = type(server)(
            config=server.config.model_copy(update={"execute_tools": False}), server_client=server_client
        )
        request = MagicMock(session={}, path_params={}, cookies={})
        error = "execute_tools=false is supported only for agent-session requests"

        with pytest.raises(ValueError, match=error):
            await server.responses(
                request,
                Response(),
                NeMoGymResponseCreateParamsNonStreaming(input="question"),
            )
        with pytest.raises(ValueError, match=error):
            await server.run(request, SimpleAgentRunRequest(responses_create_params={"input": "question"}))
        server_client.post.assert_not_called()

    async def test_agent_session_rejects_required_mcp_access(self) -> None:
        server, _ = _make_agent(False)
        request = MagicMock(session={})
        body = AgentSeedSessionRequest(
            agent_session_id="agent-session",
            episode_id=EpisodeId(rollout_id="rollout", attempt=0),
            task_id=TaskId(taskset="example", task_id="0"),
            tool_accesses=[
                MCPToolAccess(
                    name="resources",
                    required=True,
                    connection=MCPStreamableHTTPConnection(url="http://resources:8000/mcp"),
                )
            ],
        )

        with pytest.raises(ValueError, match="does not support required MCP"):
            await server.seed_agent_session(request, body)

    @pytest.mark.parametrize(
        ("unsupported", "reason"),
        [
            (
                {
                    "tool_accesses": [
                        MCPToolAccess(
                            name="resources",
                            required=True,
                            connection=MCPStreamableHTTPConnection(url="http://resources:8000/mcp"),
                        )
                    ]
                },
                "Simple Agent does not support required MCP tool access: resources",
            ),
            (
                {
                    "sandbox_access": SandboxAccess(
                        connection=DirectSandboxConnection(provider_config_ref="runtime", descriptor={}),
                        workdir="/work",
                    )
                },
                "Simple Agent does not support sandbox access",
            ),
            (
                {
                    "tool_accesses": [
                        DirectHTTPToolAccess(name=name, required=True, base_url=f"http://{name}:8000")
                        for name in ("first", "second")
                    ]
                },
                "Simple Agent supports at most one direct HTTP tool access per session",
            ),
        ],
    )
    def test_unsupported_seed_is_refused_with_a_final_422(self, unsupported: dict, reason: str) -> None:
        server, _ = _make_agent(False)
        body = AgentSeedSessionRequest(
            agent_session_id="agent-session",
            episode_id=EpisodeId(rollout_id="rollout", attempt=0),
            task_id=TaskId(taskset="example", task_id="0"),
            **unsupported,
        )

        assert check_agent_seed_rejected(server.setup_webserver(), body) == reason

    async def test_several_workers_seed_sessions(self) -> None:
        """Session routing sends a session's later requests to the worker that seeded it."""
        server, _ = _make_agent(False)
        server = type(server)(
            config=server.config.model_copy(update={"num_workers": 2}), server_client=server.server_client
        )
        body = AgentSeedSessionRequest(
            agent_session_id="agent-session",
            episode_id=EpisodeId(rollout_id="rollout", attempt=0),
            task_id=TaskId(taskset="example", task_id="0"),
        )

        request = MagicMock(session={})
        await server.seed_agent_session(request, body)

        assert server._require_agent_session("agent-session").request == body
        assert request.session[AGENT_SESSION_COOKIE_KEY] == "agent-session"

    async def test_agent_session_without_a_grant_refuses_tool_calls(self, monkeypatch: MonkeyPatch) -> None:
        # The configured resources_server has no session for this episode, so a tool call must not fall back to it.
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="simple",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            model_server=ModelServerRef(type="responses_api_models", name="model"),
            # The model keeps calling the tool; a step limit ends the loop if a regression falls back.
            max_steps=2,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client.post = AsyncMock(
            return_value=_mock_response(
                {
                    "id": "resp-tool",
                    "created_at": 1.0,
                    "model": "model",
                    "object": "response",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [
                        {
                            "id": "fc-1",
                            "call_id": "call-1",
                            "name": "get_weather",
                            "arguments": '{"city":"San Francisco"}',
                            "type": "function_call",
                            "status": "completed",
                        }
                    ],
                }
            )
        )
        direct_request = AsyncMock()
        monkeypatch.setattr("responses_api_agents.simple_agent.app.http_request", direct_request)
        client = TestClient(SimpleAgent(config=config, server_client=server_client).setup_webserver())

        seed = client.post(
            "/v1/agent_sessions",
            json=AgentSeedSessionRequest(
                agent_session_id="agent-session",
                episode_id=EpisodeId(rollout_id="rollout", attempt=0),
                task_id=TaskId(taskset="example", task_id="0"),
            ).model_dump(mode="json"),
        )
        assert seed.status_code == 200

        with pytest.raises(RuntimeError, match="no direct HTTP tool access"):
            client.post("/v1/responses", json={"input": [{"role": "user", "content": "weather?"}]})
        assert [call.kwargs["server_name"] for call in server_client.post.await_args_list] == ["model"]
        direct_request.assert_not_awaited()

    async def test_agent_session_without_a_grant_closes_without_a_resources_cookie_jar(self) -> None:
        # The session never held the Resources cookie, so close must not report an empty jar.
        # An empty jar would replace the seed's cookie that the Environment Server sends to verify.
        server, server_client = _make_agent(False)
        server_client.post = AsyncMock(
            return_value=_mock_response(
                {
                    "id": "resp-final",
                    "created_at": 1.0,
                    "model": "model",
                    "object": "response",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [
                        {
                            "id": "msg-1",
                            "content": [{"annotations": [], "text": "42", "type": "output_text"}],
                            "role": "assistant",
                            "status": "completed",
                            "type": "message",
                        }
                    ],
                }
            )
        )
        client = TestClient(server.setup_webserver())
        episode_id = EpisodeId(rollout_id="rollout", attempt=0)
        seed = client.post(
            "/v1/agent_sessions",
            json=AgentSeedSessionRequest(
                agent_session_id="agent-session",
                episode_id=episode_id,
                task_id=TaskId(taskset="example", task_id="0"),
            ).model_dump(mode="json"),
        )
        assert seed.status_code == 200

        result = client.post("/v1/responses", json={"input": [{"role": "user", "content": "answer?"}]})
        assert result.status_code == 200

        close = client.post(
            "/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=episode_id).model_dump(
                mode="json"
            ),
        )
        assert close.status_code == 200
        assert close.json()["resources_cookies"] is None

    async def test_agent_session_seed_and_close_are_idempotent(self) -> None:
        server, _ = _make_agent(False)
        request = MagicMock(session={})
        body = AgentSeedSessionRequest(
            agent_session_id="agent-session",
            episode_id=EpisodeId(rollout_id="rollout", attempt=0),
            task_id=TaskId(taskset="example:test", task_id="0"),
        )

        first = await server.seed_agent_session(request, body)
        second = await server.seed_agent_session(request, body)
        assert first == second

        close_body = AgentCloseSessionRequest(
            agent_session_id=body.agent_session_id,
            episode_id=body.episode_id,
        )
        first_close = await server.close_agent_session(request, close_body)
        second_close = await server.close_agent_session(request, close_body)
        assert second_close == first_close

        # The base session bookkeeping rejects a seed after close with a conflict, which callers do not retry.
        with pytest.raises(HTTPException) as error:
            await server.seed_agent_session(request, body)
        assert error.value.status_code == 409

    @pytest.mark.parametrize("resolved", [False, None])
    async def test_run_emits_standard_turns_and_tool_observation(self, resolved: bool | None) -> None:
        server, server_client = _make_agent(True)
        response_base = {
            "created_at": 1.0,
            "model": "model",
            "object": "response",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        model_payloads = iter(
            (
                response_base
                | {
                    "id": "resp-tool",
                    "output": [
                        {
                            "id": "reasoning-1",
                            "summary": [{"text": "look up the answer", "type": "summary_text"}],
                            "status": "completed",
                            "type": "reasoning",
                        },
                        {
                            "id": "fc-1",
                            "call_id": "call-1",
                            "name": "lookup",
                            "arguments": '{"q":"x"}',
                            "type": "function_call",
                            "status": "completed",
                        },
                    ],
                },
                response_base
                | {
                    "id": "resp-final",
                    "created_at": 2.0,
                    "output": [
                        {
                            "id": "msg-1",
                            "content": [{"annotations": [], "text": "done", "type": "output_text"}],
                            "role": "assistant",
                            "status": "completed",
                            "type": "message",
                        }
                    ],
                },
            )
        )

        async def post(*, server_name, url_path, **kwargs):
            if url_path == "/seed_session":
                return _mock_response()
            if server_name == "simple":
                nested_request = MagicMock(cookies=kwargs["cookies"], path_params={"rollout_id": "4-1"})
                model_response = await server.responses(nested_request, Response(), kwargs["json"])
                return _mock_response(model_response.model_dump(mode="json"))
            if server_name == "model":
                return _mock_response(next(model_payloads))
            if url_path == "/lookup":
                return _mock_response(status=422, content="bad input")
            assert url_path == "/verify"
            result = kwargs["json"] | {"reward": 0.0}
            if resolved is not None:
                result["resolved"] = resolved
            return _mock_response(result)

        server_client.post = AsyncMock(side_effect=post)
        body = SimpleAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "question"}]},
                "instance_id": 0,
                "_ng_task_index": 4,
                "_ng_rollout_index": 1,
            }
        )
        request = MagicMock()
        request.cookies = {}
        result = await server.run(request, body)

        assert [
            (item.kwargs["server_name"], item.kwargs["url_path"]) for item in server_client.post.await_args_list
        ] == [
            ("resources", "/seed_session"),
            ("simple", "/ng-rollout/4-1/v1/responses"),
            ("model", "/ng-rollout/4-1/v1/responses"),
            ("resources", "/lookup"),
            ("model", "/ng-rollout/4-1/v1/responses"),
            ("resources", "/verify"),
        ]

        result_data = result.model_dump(mode="json")
        result_data["ng_model_call_capture"] = {
            "calls": [
                {
                    "model_call_id": f"model-call-{index}",
                    "response_id": response_id,
                    "request": {"input": f"model-visible-input-{index}"},
                    "response": {"status": "completed", "output": f"model-visible-output-{index}"},
                }
                for index, response_id in enumerate(("resp-tool", "resp-final"), start=1)
            ]
        }
        row = {TASK_INDEX_KEY_NAME: 4, ROLLOUT_INDEX_KEY_NAME: 1, "instance_id": 0}
        _attach_trajectory_record(row, result_data)
        serialized = orjson.loads(orjson.dumps(result_data))
        trajectory = TrajectoryRecord.model_validate(serialized["ng_trajectory"])

        assert trajectory.schema_version == "1.0"
        assert [call.response_metadata.response_id for call in trajectory.model_calls] == ["resp-tool", "resp-final"]
        assert all(call.response_metadata.response_status == "completed" for call in trajectory.model_calls)
        assert trajectory.model_calls[0].request == {"input": "model-visible-input-1"}
        assert trajectory.model_calls[0].response == {"status": "completed", "output": "model-visible-output-1"}
        assert trajectory.invocations[0].conversation[-1].type == "message"
        turns = trajectory.turns
        assert [(turn.task_id, turn.rollout_id, turn.turn_no, turn.step_count) for turn in turns] == [
            ("0", "4-1", 1, 1),
            ("0", "4-1", 2, 1),
        ]
        assert all(turn.timestamp > 0 for turn in turns)
        assert [turn.model_calls[0].response_id for turn in turns] == ["resp-tool", "resp-final"]
        assert _drop_nulls(turns[0].model_dump(mode="json")["question"]) == [
            {"role": "user", "content": "question", "type": "message"}
        ]
        assert [item["type"] for item in turns[1].model_dump(mode="json")["question"]] == [
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
        ]
        assert [item["type"] for item in turns[0].model_dump(mode="json")["answer"]] == ["function_call"]
        assert turns[0].reasoning_content[0]["summary"][0]["text"] == "look up the answer"
        assert turns[-1].resolved is resolved
        assert ("resolution_unavailable" in {gap.code for gap in trajectory.gaps}) is (resolved is None)
        [tool] = trajectory.tool_calls
        assert (tool.output, tool.status, tool.error_type) == ("bad input", "failed", "http_422")
        assert tool.started_at is not None and tool.completed_at is not None and tool.duration_ms is not None

    @pytest.mark.parametrize(("capture_enabled", "override_responses"), ((False, False), (True, False), (True, True)))
    async def test_run_preserves_self_dispatch(self, capture_enabled: bool, override_responses: bool) -> None:
        agent_type = SimpleAgent
        if override_responses:

            async def overridden_responses(*args, **kwargs):
                raise AssertionError("run must preserve self-dispatch for responses overrides")

            agent_type = type("OverriddenSimpleAgent", (SimpleAgent,), {"responses": overridden_responses})
        server, server_client = _make_agent(capture_enabled, agent_type)

        model_response = {
            "id": "response-1",
            "created_at": 1.0,
            "model": "model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        async def post(*, url_path, **kwargs):
            if url_path == "/seed_session":
                return _mock_response()
            if url_path.endswith("/v1/responses"):
                return _mock_response(model_response)
            assert url_path == "/verify"
            return _mock_response(kwargs["json"] | {"reward": 1.0})

        server_client.post = AsyncMock(side_effect=post)
        body = SimpleAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": "question"},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
        )
        request = MagicMock(cookies={})

        result = await server.run(request, body)

        assert [call.kwargs["url_path"] for call in server_client.post.await_args_list] == [
            "/seed_session",
            "/ng-rollout/0-0/v1/responses" if capture_enabled else "/v1/responses",
            "/verify",
        ]
        assert "ng_trajectory" not in result.model_dump(mode="json")

    async def test_run_sends_legacy_equivalent_verify_payload(self) -> None:
        server, server_client = _make_agent(True)
        response_payload = NeMoGymResponse.model_validate(
            {
                "id": "response-1",
                "created_at": 1.0,
                "model": "model",
                "object": "response",
                "output": [
                    {
                        "id": "message-1",
                        "content": [{"annotations": [], "text": "answer", "type": "output_text"}],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                        "prompt_token_ids": [10, 11],
                        "generation_token_ids": [12, 13],
                        "generation_log_probs": [-0.1, -0.2],
                        "routed_experts": [[[0, 1]], [[1, 0]]],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        ).model_dump(mode="json")
        response_with_trajectory = response_payload | {
            "_ng_trajectory": TrajectoryRecord(task_id="unscoped", rollout_id="unscoped").model_dump(mode="json")
        }

        async def post(*, url_path, **kwargs):
            if url_path == "/seed_session":
                return _mock_response()
            if url_path.endswith("/v1/responses"):
                return _mock_response(response_with_trajectory)
            assert url_path == "/verify"
            return _mock_response(kwargs["json"] | {"reward": 1.0, "resolved": True})

        server_client.post = AsyncMock(side_effect=post)
        run_payload = {
            "responses_create_params": {
                "input": [{"role": "user", "content": "question"}],
                "max_output_tokens": 128,
                "metadata": {"extra_body": '{"seed": 7}'},
                "temperature": 0.25,
            },
            "verifier_metadata": {"expected_answer": "answer", "weight": 2},
            "task_id": "task-7",
            "dataset_field": {"nested": ["value"]},
            TASK_INDEX_KEY_NAME: 7,
            ROLLOUT_INDEX_KEY_NAME: 3,
            ATTEMPT_INDEX_KEY_NAME: 2,
            ROLLOUT_ID_KEY_NAME: "explicit-rollout",
            AGENT_REF_KEY_NAME: {"type": "responses_api_agents", "name": "simple"},
            TASK_SOURCE_KEY_NAME: "resources",
        }
        body = SimpleAgentRunRequest.model_validate(run_payload)
        body_before = body.model_dump()

        result = await server.run(MagicMock(cookies={}), body)

        verify_payload = server_client.post.await_args_list[-1].kwargs["json"]
        legacy_payload = SimpleAgentVerifyRequest.model_validate(
            body_before | {"response": response_payload}
        ).model_dump()
        assert verify_payload == legacy_payload
        assert verify_payload["responses_create_params"] == body_before["responses_create_params"]
        assert verify_payload["verifier_metadata"] == run_payload["verifier_metadata"]
        assert verify_payload["task_id"] == "task-7"
        assert verify_payload["dataset_field"] == {"nested": ["value"]}
        assert verify_payload["response"] == response_payload
        assert verify_payload["response"]["output"][0]["generation_token_ids"] == [12, 13]
        assert {
            key: verify_payload[key]
            for key in (
                TASK_INDEX_KEY_NAME,
                ROLLOUT_INDEX_KEY_NAME,
                ATTEMPT_INDEX_KEY_NAME,
                AGENT_REF_KEY_NAME,
                TASK_SOURCE_KEY_NAME,
            )
        } == {
            key: body_before[key]
            for key in (
                TASK_INDEX_KEY_NAME,
                ROLLOUT_INDEX_KEY_NAME,
                ATTEMPT_INDEX_KEY_NAME,
                AGENT_REF_KEY_NAME,
                TASK_SOURCE_KEY_NAME,
            )
        }
        assert "_ng_trajectory" not in verify_payload["response"]
        assert "ng_trajectory" not in verify_payload
        assert result.ng_trajectory is not None
        assert result.ng_trajectory["rollout_id"] == "explicit-rollout-a2"
        assert body.model_dump() == body_before

    async def test_run_preserves_response_aliases_in_verify_payload(self) -> None:
        server, server_client = _make_agent(False)
        json_schema_format = {
            "type": "json_schema",
            "name": "answer",
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "strict": True,
        }
        # FastAPI serializes the model server response by alias, so the wire field is `schema`, not `schema_`.
        response_payload = NeMoGymResponse.model_validate(
            {
                "id": "response-1",
                "created_at": 1.0,
                "model": "model",
                "object": "response",
                "output": [
                    {
                        "id": "message-1",
                        "content": [{"annotations": [], "text": '{"answer": "yes"}', "type": "output_text"}],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
                "parallel_tool_calls": True,
                "text": {"format": json_schema_format},
                "tool_choice": "auto",
                "tools": [],
            }
        ).model_dump(mode="json", by_alias=True)

        async def post(*, url_path, **kwargs):
            if url_path == "/seed_session":
                return _mock_response()
            if url_path.endswith("/v1/responses"):
                return _mock_response(response_payload)
            assert url_path == "/verify"
            return _mock_response(kwargs["json"] | {"reward": 1.0})

        server_client.post = AsyncMock(side_effect=post)
        body = SimpleAgentRunRequest.model_validate(
            {"responses_create_params": {"input": "question", "text": {"format": json_schema_format}}}
        )

        await server.run(MagicMock(cookies={}), body)

        verify_payload = server_client.post.await_args_list[-1].kwargs["json"]
        assert verify_payload["response"]["text"]["format"]["schema"] == json_schema_format["schema"]
        assert "schema_" not in verify_payload["response"]["text"]["format"]
        verify_request = SimpleAgentVerifyRequest.model_validate(verify_payload)
        assert verify_request.response.text.format.schema_ == json_schema_format["schema"]

    async def test_responses_continues_on_malformed_tool_call_arguments(self, monkeypatch: MonkeyPatch) -> None:
        """Malformed JSON in a tool-call's arguments must not crash the rollout.

        The agent should surface the parse error back to the model as a
        function_call_output and let the loop continue (ultimately terminating
        on a normal assistant message).
        """
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="my resources server",
            ),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_bad_tool_call = {
            "id": "resp_bad_tool_call",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "my_tool",
                    # Not valid JSON.
                    "arguments": "{not json",
                    "type": "function_call",
                    "status": "completed",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        mock_response_chat_data = {
            "id": "resp_final",
            "created_at": 1753983921.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_final",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Sorry, I'll stop calling that tool.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.side_effect = [
            json.dumps(mock_response_bad_tool_call),
            json.dumps(mock_response_chat_data),
        ]
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res.status_code == 200

        # The resources server must not be called for a malformed tool call —
        # only the two model calls should hit server_client.post.
        post_call_kwargs = [c.kwargs for c in server.server_client.post.call_args_list]
        server_names_called = [kw["server_name"] for kw in post_call_kwargs]
        assert server_names_called == ["my server name", "my server name"]

        # The second model call's input must include the original function_call
        # plus a function_call_output describing the parse error.
        second_call_input = post_call_kwargs[1]["json"].input
        assert any(
            isinstance(item, NeMoGymResponseFunctionToolCall) and item.call_id == "call_1"
            for item in second_call_input
        )
        error_outputs = [
            item
            for item in second_call_input
            if isinstance(item, NeMoGymFunctionCallOutput) and item.call_id == "call_1"
        ]
        assert len(error_outputs) == 1
        error_payload = json.loads(error_outputs[0].output)
        assert "error" in error_payload
        assert "Invalid tool call arguments" in error_payload["error"]
        # The exception type must be visible to the model — repr(e) on a
        # JSONDecodeError starts with the class name.
        assert "JSONDecodeError" in error_payload["error"]

    @pytest.mark.parametrize("empty_output", [False, True])
    async def test_responses_stops_without_message_or_tool_calls(self, caplog, empty_output) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_reasoning_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "status": "completed",
                    "type": "reasoning",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        if empty_output:
            mock_response_reasoning_data["output"] = []
        mock_response_reasoning_data["status"] = "completed"
        mock_response_reasoning_data["metadata"] = {"existing_key": "preserved"}
        dotjson_mock = AsyncMock()
        dotjson_mock.read.side_effect = [json.dumps(mock_response_reasoning_data)]
        dotjson_mock.cookies = {}
        server.server_client.post.return_value = dotjson_mock

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})

        assert res.status_code == 200
        server.server_client.post.assert_awaited_once()
        assert res.json()["status"] == "incomplete"
        assert res.json()["incomplete_details"] is None
        assert res.json()["metadata"]["existing_key"] == "preserved"
        assert res.json()["metadata"]["ng_termination_reason"] == (
            "empty_output" if empty_output else "incomplete_reasoning"
        )
        assert res.json()["metadata"]["ng_termination_message"] in caplog.text
        assert [item["type"] for item in res.json()["output"]] == ([] if empty_output else ["reasoning"])
        if not empty_output:
            assert res.json()["output"][0]["summary"] == mock_response_reasoning_data["output"][0]["summary"]
        assert "Ending trajectory" in caplog.text
        assert "finish_reason='stop'" in caplog.text
        assert "finish_reason='length', handled separately" in caplog.text
        assert "badly trained model requiring training-level fixes" in caplog.text
        assert "bug in the inference engine" in caplog.text
        assert mock_response_reasoning_data["id"] in caplog.text

    async def test_reasoning_only_trajectory_is_incomplete(self) -> None:
        server, server_client = _make_agent(True)
        payload = {
            "id": "reasoning-response",
            "created_at": 1.0,
            "model": "model",
            "object": "response",
            "output": [{"id": "reasoning-1", "type": "reasoning", "summary": []}],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 8,
            },
        }
        server_client.post = AsyncMock(return_value=_mock_response(payload))

        response, trajectory, _, _ = await server._create_episode(
            NeMoGymResponseCreateParamsNonStreaming(input="question"),
            model_url_path="/v1/responses",
            rollout_id="reasoning-rollout",
            collect_trajectory=True,
        )

        server_client.post.assert_awaited_once()
        assert response.usage.total_tokens == 8
        assert [item.type for item in response.output] == ["reasoning"]
        assert trajectory.invocations[0].status == "incomplete"
        assert len(trajectory.turns) == 1
        assert trajectory.turns[0].answer == []
        assert trajectory.turns[0].reasoning_content[0]["id"] == "reasoning-1"
        assert trajectory.tool_calls == []

    @pytest.mark.parametrize("with_compaction", [False, True])
    @pytest.mark.parametrize("skip_verification", [False, True])
    async def test_termination_metadata_survives_run(self, with_compaction, skip_verification) -> None:
        from responses_api_agents.simple_agent_with_compaction.app import (
            SimpleAgentWithCompaction,
            SimpleAgentWithCompactionConfig,
            SimpleAgentWithCompactionRunRequest,
        )

        agent_cls = SimpleAgentWithCompaction if with_compaction else SimpleAgent
        config_cls = SimpleAgentWithCompactionConfig if with_compaction else SimpleAgentConfig
        request_cls = SimpleAgentWithCompactionRunRequest if with_compaction else SimpleAgentRunRequest
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = {"observability_enabled": False}
        server = agent_cls(
            config=config_cls(
                host="localhost",
                port=8080,
                entrypoint="",
                name="agent",
                model_server=ModelServerRef(type="responses_api_models", name="model"),
                resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
                skip_verification=skip_verification,
            ),
            server_client=client,
        )
        payload = {
            "id": "reasoning-response",
            "created_at": 1.0,
            "model": "model",
            "object": "response",
            "output": [{"id": "reasoning-1", "type": "reasoning", "summary": []}],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
        }
        request = MagicMock(cookies={})

        async def post(*, server_name, url_path, **kwargs):
            if url_path == "/seed_session":
                return _mock_response()
            if server_name == "agent":
                response = await server.responses(request, Response(), kwargs["json"])
                return _mock_response(response.model_dump(mode="json"))
            if server_name == "model":
                return _mock_response(payload)
            assert url_path == "/verify"
            assert kwargs["json"]["response"]["metadata"]["ng_termination_reason"] == "incomplete_reasoning"
            return _mock_response(kwargs["json"] | {"reward": 0.0})

        client.post = AsyncMock(side_effect=post)
        result = await server.run(request, request_cls(responses_create_params={"input": "question"}))
        saved = json.loads(result.model_dump_json())
        assert "ng_trajectory" not in saved
        assert saved["response"]["status"] == "incomplete"
        assert saved["response"]["metadata"]["ng_termination_reason"] == "incomplete_reasoning"
        assert "training-level fixes" in saved["response"]["metadata"]["ng_termination_message"]
        assert sum(call.kwargs["server_name"] == "model" for call in client.post.await_args_list) == 1

    async def test_usage_sanity(self, monkeypatch: MonkeyPatch) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            max_steps=3,
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "my_tool",
                    "arguments": "{invalid json",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        response_1 = mock_response_data | {
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 3,
            },
        }
        response_2 = mock_response_data | {"usage": None}
        response_3 = mock_response_data | {
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 200,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 300,
            },
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.side_effect = [json.dumps(response_1), json.dumps(response_2), json.dumps(response_3)]
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200

        actual_responses_dict = res_no_model.json()
        actual_usage_dict = actual_responses_dict["usage"]
        expected_usage_dict = {
            "input_tokens": 101,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 202,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 303,
        }
        assert expected_usage_dict == actual_usage_dict

    async def test_incomplete_details(self, monkeypatch: MonkeyPatch, caplog) -> None:
        await self._test_incomplete_details_helper(monkeypatch, {"reason": "max_output_tokens"})
        await self._test_incomplete_details_helper(monkeypatch, {"reason": "content_filter"})
        assert "Ending trajectory" not in caplog.text

    async def _test_incomplete_details_helper(self, monkeypatch: MonkeyPatch, incomplete_details) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_reasoning_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "status": "completed",
                    "type": "reasoning",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "incomplete_details": incomplete_details,
        }

        mock_response_chat_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.side_effect = [json.dumps(mock_response_reasoning_data), json.dumps(mock_response_chat_data)]
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200

        expected_calls = [
            call(
                server_name="my server name",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
                ),
                cookies=None,
            ),
            call().ok.__bool__(),
            call().read(),
            call().cookies.items(),
            call().cookies.items().__iter__(),
            call().cookies.items().__len__(),
        ]
        server.server_client.post.assert_has_calls(expected_calls)

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "error": None,
            "incomplete_details": incomplete_details,
            "instructions": None,
            "metadata": None,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": None,
                    "encrypted_content": None,
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "type": "reasoning",
                },
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
            "conversation": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
        }
        assert _drop_nulls(expected_responses_dict) == _drop_nulls(actual_responses_dict)

    async def test_run_skip_verification_uses_configured_reward(self) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="simple_agent",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my model server",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="my resources server",
            ),
            skip_verification=True,
            skip_verification_reward=0.25,
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        seed_response = AsyncMock()
        seed_response.ok = True
        seed_response.cookies = {"session": "seeded"}

        model_response_payload = {
            "id": "response_id",
            "created_at": 1,
            "model": "dummy_model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        model_response = AsyncMock()
        model_response.ok = True
        model_response.cookies = {"session": "model"}
        model_response.read.return_value = json.dumps(model_response_payload).encode()

        server.server_client.post.side_effect = [seed_response, model_response]

        response = client.post(
            "/run",
            json={"responses_create_params": {"input": [{"role": "user", "content": "hello"}]}},
        )

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["reward"] == 0.25
        assert response_json["verification_skipped"] is True
        assert response_json["response"]["id"] == "response_id"

        post_call_kwargs = [post_call.kwargs for post_call in server.server_client.post.call_args_list]
        assert [kwargs["url_path"] for kwargs in post_call_kwargs] == [
            "/seed_session",
            "/v1/responses",
        ]
        assert post_call_kwargs[0]["server_name"] == "my resources server"
        assert post_call_kwargs[1]["server_name"] == "simple_agent"
        assert post_call_kwargs[1]["cookies"] == {"session": "seeded"}


class _RecordingActivation:
    """Stands in for a checkpoint activation: supplies a continuation and records boundaries."""

    def __init__(self, continuation=None) -> None:
        self.continuation = continuation
        self.snapshots: list = []
        self.eager: list[dict] = []

    async def boundary(self, snapshot) -> None:
        # Keep the lazy snapshot and also evaluate it now, to check that evaluating it later agrees.
        self.snapshots.append(snapshot)
        self.eager.append(snapshot())

    @property
    def boundaries(self) -> list[dict]:
        return [snapshot() for snapshot in self.snapshots]

    @asynccontextmanager
    async def awaiting_model(self):
        yield


_TOOL_CALL_RESPONSE = {
    "id": "response-1",
    "created_at": 1.0,
    "model": "model",
    "object": "response",
    "output": [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "get_weather",
            "arguments": '{"city": "SF"}',
            "id": "fc-1",
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}
_FINAL_RESPONSE = _TOOL_CALL_RESPONSE | {
    "id": "response-2",
    "output": [
        {
            "type": "message",
            "id": "msg-1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "cold", "annotations": []}],
        }
    ],
}


class TestCheckpointContinuation:
    async def test_boundaries_capture_a_resumable_loop_and_resume_skips_regeneration(self) -> None:
        server, server_client = _make_agent(False)
        calls: list[str] = []
        # The first model call asks for the tool; every later one answers.
        model_calls = 0

        async def post(*, server_name, url_path, **kwargs):
            nonlocal model_calls
            calls.append(url_path)
            if url_path == "/get_weather":
                return _mock_response(content="cold")
            model_calls += 1
            return _mock_response(_TOOL_CALL_RESPONSE if model_calls == 1 else _FINAL_RESPONSE)

        server_client.post = AsyncMock(side_effect=post)
        body = NeMoGymResponseCreateParamsNonStreaming(input="weather?")
        original = _RecordingActivation()
        first, *_ = await server._create_episode(body, model_url_path="/v1/responses", activation=original)

        # Snapshots evaluated after the loop finished still describe their own boundary.
        assert original.boundaries == original.eager
        # The boundary after the first model response has its tool call pending.
        pending = next(state for state in original.boundaries if state["pending_tools"])
        assert pending["step"] == 1 and calls[:2] == ["/v1/responses", "/get_weather"]

        calls.clear()
        resumed_activation = _RecordingActivation(continuation=pending)
        resumed, *_ = await server._create_episode(body, model_url_path="/v1/responses", activation=resumed_activation)

        assert calls == ["/get_weather", "/v1/responses"]
        assert resumed.model_dump(mode="json")["output"] == first.model_dump(mode="json")["output"]
        assert resumed.usage.total_tokens == first.usage.total_tokens == 30

    async def test_a_boundary_leaves_out_turn_questions_and_a_resumed_loop_rebuilds_them(self) -> None:
        """Each turn's question repeats the input and the earlier outputs, so a boundary keeps only their count."""
        server, server_client = _make_agent(True)
        model_calls = 0

        async def post(*, server_name, url_path, **kwargs):
            nonlocal model_calls
            if url_path == "/get_weather":
                return _mock_response(content="cold")
            model_calls += 1
            return _mock_response(_TOOL_CALL_RESPONSE if model_calls < 3 else _FINAL_RESPONSE)

        server_client.post = AsyncMock(side_effect=post)
        body = NeMoGymResponseCreateParamsNonStreaming(input="weather?")
        original = _RecordingActivation()
        _, trajectory, *_ = await server._create_episode(
            body, model_url_path="/v1/responses", collect_trajectory=True, activation=original
        )
        turns = [turn.model_dump(mode="json") for turn in trajectory.turns]

        resumed_from = [state for state in original.boundaries if state["turns"]]
        for boundary in resumed_from:
            # The loop had made one model call per step when it recorded the boundary.
            model_calls = boundary["step"]
            _, resumed, *_ = await server._create_episode(
                body,
                model_url_path="/v1/responses",
                collect_trajectory=True,
                activation=_RecordingActivation(continuation=boundary),
            )
            restored = len(boundary["turns"])
            resumed_turns = [turn.model_dump(mode="json") for turn in resumed.turns]

            assert all("question" not in turn for turn in boundary["turns"])
            # The restored turns are the ones the loop recorded, questions included.
            assert resumed_turns[:restored] == turns[:restored]
            assert [turn["question"] for turn in resumed_turns] == [turn["question"] for turn in turns]
        # After each of the two tool-calling responses, and after each tool round.
        assert len(resumed_from) == 4

    async def test_restored_legacy_run_skips_resources_seeding(self) -> None:
        # Observability is off: checkpointing alone must key the legacy /run by its rollout.
        server, server_client = _make_agent(False)
        participant = AgentSessionParticipant(server)
        server._checkpoint_participant = participant
        await participant.install(
            [
                AgentSessionRecord(
                    session_key="run:0-0",
                    episode_id=EpisodeId(rollout_id="0-0"),
                    session={},
                    boundary={"step": 0},
                    episode={"next": "loop", "cookies": {}},
                )
            ],
            [EpisodeId(rollout_id="0-0")],
        )

        async def post(*, url_path, **kwargs):
            if url_path.endswith("/v1/responses"):
                return _mock_response(_FINAL_RESPONSE)
            assert url_path == "/verify"
            return _mock_response(kwargs["json"] | {"reward": 1.0})

        server_client.post = AsyncMock(side_effect=post)
        body = SimpleAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": "question"},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                ATTEMPT_INDEX_KEY_NAME: 1,
            }
        )

        await server.run(MagicMock(cookies={}), body)

        assert [call.kwargs["url_path"] for call in server_client.post.await_args_list] == [
            "/ng-rollout/0-0-a1/v1/responses",
            "/verify",
        ]

    @pytest.mark.parametrize("reported, blocks_while_verifying", [("replay", False), (None, True)])
    async def test_legacy_resources_verify_step_type_comes_from_the_seed_reply(
        self, reported: str | None, blocks_while_verifying: bool
    ) -> None:
        server, server_client = _make_agent(False)
        participant = AgentSessionParticipant(server)
        server._checkpoint_participant = participant
        blocked_during_verify: list[bool] = []

        async def post(*, url_path, **kwargs):
            if url_path == "/seed_session":
                response = _mock_response({})
                response.headers = {"x-ng-checkpoint-verify": reported} if reported else {}
                return response
            if url_path.endswith("/v1/responses"):
                return _mock_response(_FINAL_RESPONSE)
            if url_path == "/close_session":
                return _mock_response({})
            # A replayed verification does not hold a checkpoint; a waited-on one does.
            blocked_during_verify.append(not participant.readiness().ready)
            return _mock_response(kwargs["json"] | {"reward": 1.0})

        server_client.post = AsyncMock(side_effect=post)
        body = SimpleAgentRunRequest.model_validate(
            {"responses_create_params": {"input": "question"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        )

        await server.run(MagicMock(cookies={}), body)

        assert blocked_during_verify == [blocks_while_verifying]

    async def test_a_legacy_run_on_a_resources_server_that_cannot_capture_it_is_a_restart(self) -> None:
        server, server_client = _make_agent(False)
        participant = AgentSessionParticipant(server)
        server._checkpoint_participant = participant
        during_verify: list[tuple[bool, list[str], int]] = []

        async def post(*, url_path, **kwargs):
            if url_path == "/seed_session":
                response = _mock_response({})
                response.headers = {"x-ng-checkpoint-restart": "1"}
                return response
            if url_path.endswith("/v1/responses"):
                return _mock_response(_FINAL_RESPONSE)
            # Its verification would hold up a checkpoint, but nothing of a restart is in one.
            report = participant.readiness()
            during_verify.append((report.ready, report.restarts, len(await participant.export(None))))
            return _mock_response(kwargs["json"] | {"reward": 1.0})

        server_client.post = AsyncMock(side_effect=post)
        body = SimpleAgentRunRequest.model_validate(
            {"responses_create_params": {"input": "question"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        )

        await server.run(MagicMock(cookies={}), body)

        assert during_verify == [(True, ["0-0"], 0)]

    async def test_direct_legacy_responses_call_leaves_no_session(self) -> None:
        server, server_client = _make_agent(False)
        participant = AgentSessionParticipant(server)
        server._checkpoint_participant = participant
        server_client.post = AsyncMock(return_value=_mock_response(_FINAL_RESPONSE))
        request = MagicMock(path_params={"rollout_id": "direct-1"}, cookies={}, session={})

        await server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="q"))

        assert participant.readiness().counts["sessions"] == 0


def _message_response(response_id: str) -> MagicMock:
    return _mock_response(
        {
            "id": response_id,
            "created_at": 1.0,
            "model": "model",
            "object": "response",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": f"msg-{response_id}",
                    "content": [{"annotations": [], "text": "Done.", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
        }
    )


async def _session_after_two_activations() -> tuple[str, dict]:
    """Seed a session, run two activations, and export it as a checkpoint would."""
    server, server_client = _make_agent(True)
    server_client.post = AsyncMock(side_effect=[_message_response("resp-1"), _message_response("resp-2")])
    client = TestClient(server.setup_webserver())
    seed = client.post(
        "/v1/agent_sessions",
        json=AgentSeedSessionRequest(
            agent_session_id="agent-session",
            episode_id=EpisodeId(rollout_id="rollout", attempt=0),
            task_id=TaskId(taskset="example", task_id="0"),
        ).model_dump(mode="json"),
    )
    for question in ("first?", "second?"):
        assert client.post("/ng-rollout/rollout/v1/responses", json={"input": question}).status_code == 200
    session_id = seed.json()["agent_session_id"]
    return session_id, (await server.export_agent_sessions([session_id]))[session_id]


async def _restore_and_activate(session_id: str, session: dict, boundary: dict | None) -> list[str]:
    """Restore the session into a fresh server, run one activation, and return the invocation IDs at close."""
    server, server_client = _make_agent(True)
    participant = AgentSessionParticipant(server)
    server._checkpoint_participant = participant
    await participant.install(
        [
            AgentSessionRecord(
                session_key=session_id, episode_id=EpisodeId(rollout_id="rollout"), session=session, boundary=boundary
            )
        ],
        [EpisodeId(rollout_id="rollout")],
    )
    server_client.post = AsyncMock(return_value=_message_response("resp-3"))
    request = MagicMock(cookies={}, session={AGENT_SESSION_COOKIE_KEY: session_id}, headers={})
    request.path_params = {"rollout_id": "rollout-a1"}
    await server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="third?"))
    return [record.invocation_id for record in server._session_state(session_id).observations.records]


async def test_a_session_without_a_resources_cookie_jar_restores_without_one() -> None:
    """A session with no direct tool grant holds no resources cookie jar, and a restore must not invent one."""
    session_id, session = await _session_after_two_activations()
    assert session["resources_cookies"] is None

    server, _ = _make_agent(True)
    participant = AgentSessionParticipant(server)
    server._checkpoint_participant = participant
    await participant.install(
        [AgentSessionRecord(session_key=session_id, episode_id=EpisodeId(rollout_id="rollout"), session=session)],
        [EpisodeId(rollout_id="rollout")],
    )

    # Close then reports no jar, so the environment server keeps the cookie from its resources seed.
    assert server._session_state(session_id).resources_cookies is None


class TestRestoredSessionActivations:
    async def test_the_next_activation_of_a_restored_session_does_not_reuse_an_invocation_id(self) -> None:
        session_id, session = await _session_after_two_activations()

        invocation_ids = await _restore_and_activate(session_id, session, boundary=None)

        assert invocation_ids == ["root", "activation-2", "activation-3"]

    async def test_an_activation_continued_from_its_restored_boundary_keeps_its_invocation_id(self) -> None:
        session_id, session = await _session_after_two_activations()
        # The checkpoint caught a third activation before its first model call.
        session = session | {"activations": 3}
        boundary = SimpleAgentLoopState(
            step=0,
            pending_tools=False,
            new_outputs=[],
            usage=None,
            last_model_response=None,
            model_server_cookies={},
            resources_server_cookies={},
            turns=[],
            question_outputs=[],
            tool_records=[],
            model_calls=[],
            gaps=[],
        ).model_dump(mode="json")

        invocation_ids = await _restore_and_activate(session_id, session, boundary=boundary)

        assert invocation_ids == ["root", "activation-2", "activation-3"]


async def _until(condition) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.01)


def _checkpoint(checkpoint_id: str, **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


_SESSION_ID = "agent-session"
_STEP_ZERO = SimpleAgentLoopState(
    step=0,
    pending_tools=False,
    new_outputs=[],
    usage=None,
    last_model_response=None,
    model_server_cookies={},
    resources_server_cookies={},
    turns=[],
    question_outputs=[],
    tool_records=[],
    model_calls=[],
    gaps=[],
).model_dump(mode="json")


async def _restored_session(server: SimpleAgent, *, activations: int, boundary: dict | None) -> MagicMock:
    """Restore one agent session into ``server`` and return a request that activates it."""
    participant = AgentSessionParticipant(server)
    server._checkpoint_participant = participant
    seed = AgentSeedSessionRequest(
        agent_session_id=_SESSION_ID,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="example", task_id="0"),
    )
    session = {
        "request": seed.model_dump(mode="json"),
        "resources_cookies": {},
        "observations": None,
        "activations": activations,
        "completed_activation": activations - 1 if boundary is not None else activations,
        "completed_reply": None,
    }
    await participant.install(
        [
            AgentSessionRecord(
                session_key=_SESSION_ID, episode_id=EpisodeId(rollout_id="rollout"), session=session, boundary=boundary
            )
        ],
        [EpisodeId(rollout_id="rollout")],
    )
    await participant.open_admission()
    request = MagicMock(cookies={}, session={AGENT_SESSION_COOKIE_KEY: _SESSION_ID}, headers={})
    request.path_params = {"rollout_id": "rollout-a1"}
    return request


def _held_model(server_client: MagicMock, payload: dict) -> tuple[list[str], asyncio.Event]:
    """Answer model calls with ``payload`` once the returned event is set; record each call."""
    calls: list[str] = []
    reply = asyncio.Event()

    async def post(*, url_path, **kwargs):
        calls.append(url_path)
        await reply.wait()
        return _mock_response(payload)

    server_client.post = AsyncMock(side_effect=post)
    return calls, reply


class TestCheckpointAcrossModelCalls:
    async def test_a_restored_activation_keeps_its_boundary_through_a_second_checkpoint(self) -> None:
        server, server_client = _make_agent(False)
        # A checkpoint caught the session's first activation before its model call.
        request = await _restored_session(server, activations=1, boundary=_STEP_ZERO)
        participant = server.checkpoint_participant
        controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
        calls, reply = _held_model(server_client, _FINAL_RESPONSE)
        activation = asyncio.create_task(
            server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="q"))
        )
        await _until(lambda: calls)
        # A second checkpoint while the restored activation waits on its model call.
        prepared = await controller.prepare(CheckpointRequest(**_checkpoint("c2")))
        [record] = await participant.export(None)
        await controller.resume(CheckpointRequest(**_checkpoint("c2")))
        reply.set()
        await asyncio.wait_for(activation, 5)

        assert prepared["phase"] == "prepared"
        assert record.boundary == _STEP_ZERO
        assert server._session_state(_SESSION_ID).activations == 1

    async def test_a_model_reply_that_arrives_after_prepare_is_held_at_the_boundary_before_the_call(
        self, tmp_path
    ) -> None:
        server, server_client = _make_agent(False)
        request = await _restored_session(server, activations=0, boundary=None)
        participant = server.checkpoint_participant
        controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
        calls, reply = _held_model(server_client, _FINAL_RESPONSE)
        held = asyncio.Event()
        park = participant.park

        async def signalling_park(session) -> None:
            held.set()
            await park(session)

        participant.park = signalling_park
        activation = asyncio.create_task(
            server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="q"))
        )
        await _until(lambda: calls)
        prepared = await controller.prepare(CheckpointRequest(**_checkpoint("c1")))
        # The model server delivered the reply before its own prepare; the agent reads it only now.
        reply.set()
        await asyncio.wait_for(held.wait(), 5)
        ready_with_the_reply = participant.ready()
        committed = await controller.commit(CommitRequest(**_checkpoint("c1", checkpoint_dir=str(tmp_path))))
        [record] = await participant.export(None)
        await controller.resume(CheckpointRequest(**_checkpoint("c1")))
        result = await asyncio.wait_for(activation, 5)

        assert prepared["phase"] == "prepared" and ready_with_the_reply
        assert committed["phase"] == "committed"
        # The boundary before the call: a restore calls the model again from there.
        assert record.boundary == _STEP_ZERO
        # After resume the activation continues with the reply it held.
        assert len(calls) == 1
        assert result.output[0].content[0].text == "cold"

    async def test_a_reply_whose_headers_arrived_before_prepare_and_whose_body_is_held_does_not_hold_up_prepare(
        self,
    ) -> None:
        server, server_client = _make_agent(False)
        request = await _restored_session(server, activations=0, boundary=None)
        participant = server.checkpoint_participant
        controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
        # The model server prepared first: its gate sent the reply's headers and holds the body until it resumes.
        model_server_resumed = asyncio.Event()
        reading_body = asyncio.Event()

        async def read() -> bytes:
            reading_body.set()
            await model_server_resumed.wait()
            return json.dumps(_FINAL_RESPONSE).encode()

        reply = _mock_response()
        reply.read = AsyncMock(side_effect=read)
        server_client.post = AsyncMock(return_value=reply)
        activation = asyncio.create_task(
            server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="q"))
        )
        await asyncio.wait_for(reading_body.wait(), 5)
        # The agent server prepares after the headers arrived, while the body is still held.
        prepared = await controller.prepare(CheckpointRequest(**_checkpoint("c1") | {"deadline_ts": time.time() + 1}))
        [record] = await participant.export(None)
        await controller.resume(CheckpointRequest(**_checkpoint("c1")))
        model_server_resumed.set()
        result = await asyncio.wait_for(activation, 5)

        assert prepared["phase"] == "prepared"
        # The boundary before the call: a restore calls the model again from there.
        assert record.boundary == _STEP_ZERO
        assert server_client.post.await_count == 1
        assert result.output[0].content[0].text == "cold"

    async def test_an_activation_already_completed_is_answered_with_its_reply_without_running_again(self) -> None:
        server, server_client = _make_agent(True)
        server_client.global_config_dict = {
            "observability_enabled": True,
            "checkpoint": {"enabled": True, "control_auth_token": "t"},
        }
        server_client.post = AsyncMock(
            side_effect=lambda **kwargs: _message_response(f"resp-{server_client.post.await_count}")
        )
        app = server.setup_webserver()
        seed = AgentSeedSessionRequest(
            agent_session_id=_SESSION_ID,
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="example", task_id="0"),
        )
        path = "/ng-rollout/rollout/v1/responses"

        def activation(index: int) -> dict:
            return {"json": {"input": "q"}, "headers": {AGENT_ACTIVATION_HEADER: str(index)}}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
            assert (await http.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))).status_code == 200
            first = await http.post(path, **activation(1))
            state = server._session_state(_SESSION_ID)
            observations = state.observations.model_dump(mode="json")
            # The environment server was restored from a checkpoint taken before it received the reply.
            again = await http.post(path, **activation(1))
            calls_after_again = server_client.post.await_count
            unchanged = (state.observations.model_dump(mode="json"), state.activations) == (observations, 1)
            ahead = await http.post(path, **activation(3))
            second = await http.post(path, **activation(2))
            behind = await http.post(path, **activation(1))
            unnumbered = await http.post(path, json={"input": "q"})

        assert first.status_code == again.status_code == 200
        assert again.json() == first.json()
        assert calls_after_again == 1 and unchanged
        assert ahead.status_code == 409 and ahead.json()["error"]["code"] == "checkpoint_activation_out_of_order"
        assert second.status_code == 200 and second.json()["id"] == "resp-2"
        assert behind.status_code == 409 and behind.json()["error"]["code"] == "checkpoint_activation_out_of_order"
        # Without the header every invocation is a new activation, as before.
        assert unnumbered.status_code == 200 and unnumbered.json()["id"] == "resp-3"


@pytest.mark.parametrize("stage", ["verify", "return"])
async def test_a_legacy_run_restored_at_verify_or_return_still_reports_its_trajectory(stage: str) -> None:
    trajectory = TrajectoryRecord(task_id="0", rollout_id="0-0").model_dump(mode="json")

    async def post(*, url_path, **kwargs):
        if url_path == "/seed_session":
            return _mock_response({})
        if url_path.endswith("/v1/responses"):
            return _mock_response(_FINAL_RESPONSE | {"_ng_trajectory": trajectory})
        return _mock_response(kwargs["json"] | {"reward": 1.0})

    def run_request(attempt: int) -> SimpleAgentRunRequest:
        return SimpleAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": "question"},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                ATTEMPT_INDEX_KEY_NAME: attempt,
            }
        )

    server, server_client = _make_agent(True)
    participant = AgentSessionParticipant(server)
    server._checkpoint_participant = participant
    server_client.post = AsyncMock(side_effect=post)
    boundaries: list[dict] = []
    record_boundary = participant.legacy_episodes.boundary

    async def recording(key: str, state: dict) -> None:
        boundaries.append(state)
        await record_boundary(key, state)

    participant.legacy_episodes.boundary = recording
    await server.run(MagicMock(cookies={}), run_request(0))
    [episode] = [state for state in boundaries if state["next"] == stage]

    restored, restored_client = _make_agent(True)
    restored_participant = AgentSessionParticipant(restored)
    restored._checkpoint_participant = restored_participant
    restored_client.post = AsyncMock(side_effect=post)
    await restored_participant.install(
        [
            AgentSessionRecord(
                session_key="run:0-0", episode_id=EpisodeId(rollout_id="0-0"), session={}, episode=episode
            )
        ],
        [EpisodeId(rollout_id="0-0")],
    )
    result = await restored.run(MagicMock(cookies={}), run_request(1))

    assert result.model_dump()["ng_trajectory"]["rollout_id"] == "0-0-a1"


def _legacy_run_request(attempt: int) -> SimpleAgentRunRequest:
    return SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": "weather?"},
            TASK_INDEX_KEY_NAME: 0,
            ROLLOUT_INDEX_KEY_NAME: 0,
            ATTEMPT_INDEX_KEY_NAME: attempt,
        }
    )


def _route_legacy_run(server: SimpleAgent, server_client: MagicMock, calls: list[str], after_turn_loop=None) -> None:
    """Answer a legacy /run's calls, sending its self-dispatch to ``server.responses``.

    The turn loop asks the model, calls one tool, and asks the model again.
    ``after_turn_loop`` runs once the self-dispatched activation has replied, before /run reads the reply.
    """
    model_calls = 0

    async def post(*, server_name, url_path, **kwargs):
        nonlocal model_calls
        calls.append(f"{server_name} {url_path}")
        if server_name == "model":
            model_calls += 1
            return _mock_response(_TOOL_CALL_RESPONSE if model_calls == 1 else _FINAL_RESPONSE)
        if url_path.endswith("/get_weather"):
            return _mock_response(content="cold")
        if url_path == "/seed_session":
            return _mock_response({})
        if url_path == "/verify":
            return _mock_response(kwargs["json"] | {"reward": 1.0})
        request = MagicMock(
            path_params={"rollout_id": url_path.split("/")[2]},
            cookies=kwargs.get("cookies") or {},
            session={},
            headers=kwargs.get("headers") or {},
        )
        reply = await server.responses(request, Response(), kwargs["json"])
        payload = json.loads(reply.body) if isinstance(reply, JSONResponse) else reply.model_dump(mode="json")
        if after_turn_loop is not None:
            await after_turn_loop()
        return _mock_response(payload)

    server_client.post = AsyncMock(side_effect=post)


def _checkpoint_agent(
    agent_type: type[SimpleAgent] = SimpleAgent,
) -> tuple[SimpleAgent, MagicMock, ParticipantControlPlane]:
    server, server_client = _make_agent(False, agent_type)
    server._checkpoint_participant = AgentSessionParticipant(server)
    controller = ParticipantControlPlane(server.checkpoint_participant, instance_name="agent", lease_grace_seconds=60)
    return server, server_client, controller


async def test_a_restore_after_the_legacy_turn_loop_replied_gives_its_reply_without_running_it_again(
    tmp_path,
) -> None:
    """A checkpoint after the self-dispatched turn loop replied, before /run recorded its next boundary.

    The replacement /run continues at the loop step.
    It gets the loop's reply again, without calling the model or repeating the tool call.
    """
    server, server_client, controller = _checkpoint_agent()
    committed: list[dict] = []

    async def checkpoint() -> None:
        await controller.prepare(CheckpointRequest(**_checkpoint("c1")))
        committed.append(await controller.commit(CommitRequest(**_checkpoint("c1", checkpoint_dir=str(tmp_path)))))
        await controller.resume(CheckpointRequest(**_checkpoint("c1")))

    first_calls: list[str] = []
    _route_legacy_run(server, server_client, first_calls, after_turn_loop=checkpoint)
    first = await server.run(MagicMock(cookies={}), _legacy_run_request(0))

    restored, restored_client, restored_controller = _checkpoint_agent()
    scope = [EpisodeId(rollout_id="0-0")]
    await restored_controller.restore(
        RestoreRequest(**_checkpoint("r1", checkpoint_dir=str(tmp_path), episode_ids=scope))
    )
    await restored_controller.resume(CheckpointRequest(**_checkpoint("r1")))
    calls: list[str] = []
    _route_legacy_run(restored, restored_client, calls)
    result = await restored.run(MagicMock(cookies={}), _legacy_run_request(1))

    assert committed[0]["episode_ids"] == ["0-0"]
    assert (
        first_calls.count("model /ng-rollout/0-0/v1/responses") == 2
        and "resources /ng-rollout/0-0/get_weather" in first_calls
    )
    assert calls == ["simple /ng-rollout/0-0-a1/v1/responses", "resources /verify"]
    assert result.response.output == first.response.output and result.reward == 1.0


async def test_a_request_for_an_activation_still_running_waits_for_it_and_gets_its_reply() -> None:
    server, server_client = _make_agent(False)
    server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    calls, reply = _held_model(server_client, _FINAL_RESPONSE)
    app = server.setup_webserver()
    seed = AgentSeedSessionRequest(
        agent_session_id=_SESSION_ID,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="example", task_id="0"),
    )
    index_reads: list[int] = []
    activation_index = server._activation_index

    def counting_activation_index(request):
        index_reads.append(1)
        return activation_index(request)

    server._activation_index = counting_activation_index
    path = "/ng-rollout/rollout/v1/responses"
    activation = {"json": {"input": "q"}, "headers": {AGENT_ACTIVATION_HEADER: "1"}}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
        assert (await http.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))).status_code == 200
        first = asyncio.create_task(http.post(path, **activation))
        await _until(lambda: calls)
        # The environment server sends the same activation again, for example after its connection dropped.
        second = asyncio.create_task(http.post(path, **activation))
        await _until(lambda: len(index_reads) == 2)
        reply.set()
        first, second = await asyncio.wait_for(asyncio.gather(first, second), 5)

    assert first.status_code == 200, first.json()
    assert second.status_code == 200, second.json()
    assert second.json() == first.json()
    assert len(calls) == 1


@pytest.mark.parametrize("index", [1, 2])
async def test_a_request_on_the_replaced_attempt_s_path_is_refused_for_the_restored_session(index: int) -> None:
    """A harness of the crashed attempt calls the restored session with the replaced attempt's rollout prefix.

    Activation 1 already completed, so index 1 would replay its reply and index 2 would start a new activation.
    """
    server, server_client = _make_agent(False)
    request = await _restored_session(server, activations=1, boundary=None)
    server._session_state(_SESSION_ID).completed_reply = {"response": _FINAL_RESPONSE, "cookies": {}}
    server_client.post = AsyncMock(return_value=_mock_response(_FINAL_RESPONSE))
    request.path_params = {"rollout_id": "rollout"}
    request.headers = {AGENT_ACTIVATION_HEADER: str(index)}

    with pytest.raises(StaleAttemptError):
        await server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="q"))

    assert server_client.post.await_count == 0
    assert server._session_state(_SESSION_ID).completed_activation == 1


@pytest.mark.parametrize("index", [None, 1, 2])
async def test_an_activation_without_a_rollout_prefix_is_refused_for_the_restored_session(index: int | None) -> None:
    """A harness of the crashed attempt calls the restored session with its cookie and no rollout prefix.

    The restore already rewrote the session's own episode to the replacement attempt,
    so the session cannot tell whose call this is.
    Activation 1 already completed, so index 1 would replay its reply and claim the session,
    and index 2 or no index would start a new activation.
    """
    server, server_client = _make_agent(False)
    request = await _restored_session(server, activations=1, boundary=None)
    participant = server.checkpoint_participant
    server._session_state(_SESSION_ID).completed_reply = {"response": _FINAL_RESPONSE, "cookies": {}}
    server_client.post = AsyncMock(return_value=_mock_response(_FINAL_RESPONSE))
    request.headers = {AGENT_ACTIVATION_HEADER: str(index)} if index is not None else {}
    body = NeMoGymResponseCreateParamsNonStreaming(input="q")

    request.path_params = {}
    with pytest.raises(RolloutIdRequiredError):
        await server.responses(request, Response(), body)
    pending_after_unprefixed = [episode.capture_key for episode in await participant.unclaimed()]
    request.path_params = {"rollout_id": "rollout"}
    with pytest.raises(StaleAttemptError):
        await server.responses(request, Response(), body)
    model_calls_after_refusals = server_client.post.await_count
    completed_after_refusals = server._session_state(_SESSION_ID).completed_activation
    # The replacement attempt's own call still continues the session.
    request.path_params = {"rollout_id": "rollout-a1"}
    reply = await server.responses(request, Response(), body)

    assert pending_after_unprefixed == ["rollout-a1"]
    assert model_calls_after_refusals == 0 and completed_after_refusals == 1
    assert await participant.unclaimed() == []
    if index == 1:
        assert json.loads(reply.body)["id"] == _FINAL_RESPONSE["id"] and server_client.post.await_count == 0
    else:
        assert reply.id == _FINAL_RESPONSE["id"] and server_client.post.await_count == 1


@pytest.mark.parametrize("checkpointing", [True, False])
async def test_an_activation_without_a_rollout_prefix_is_refused_only_with_checkpointing(checkpointing: bool) -> None:
    """A session seeded in this process takes no unprefixed activation while checkpointing is on.

    Without checkpointing nothing records attempts, so the session's cookie alone still activates it.
    """
    server, server_client = _make_agent(False)
    if checkpointing:
        server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    server_client.post = AsyncMock(side_effect=lambda **kwargs: _mock_response(_FINAL_RESPONSE))
    app = server.setup_webserver()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
        await _seeded_session(http)
        unprefixed = await http.post("/v1/responses", json={"input": "q"}, headers={AGENT_ACTIVATION_HEADER: "1"})
        model_calls_after_unprefixed = server_client.post.await_count
        prefixed = await http.post(
            "/ng-rollout/rollout/v1/responses", json={"input": "q"}, headers={AGENT_ACTIVATION_HEADER: "1"}
        )

    assert prefixed.status_code == 200, prefixed.json()
    if checkpointing:
        assert unprefixed.status_code == 400 and unprefixed.json()["error"]["code"] == "checkpoint_rollout_id_required"
        # The refused call started nothing, so the prefixed call is still activation 1.
        assert model_calls_after_unprefixed == 0 and server_client.post.await_count == 1
    else:
        assert unprefixed.status_code == 200 and model_calls_after_unprefixed == 1


class _RestartOnlySimpleAgent(SimpleAgent):
    """Simple Agent as an agent without session hooks would be installed: restart-only."""

    checkpoint_mode = "restart_only"


@pytest.mark.parametrize("agent_type", [SimpleAgent, _RestartOnlySimpleAgent])
async def test_a_direct_responses_call_is_a_restart_and_one_without_a_rollout_is_refused(
    agent_type: type[SimpleAgent], tmp_path
) -> None:
    """A /v1/responses call with neither an agent session nor a legacy /run, on both kinds of agent."""
    server, server_client = _make_agent(False, agent_type)
    server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    calls, _ = _held_model(server_client, _FINAL_RESPONSE)
    app = server.setup_webserver()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agent", headers={"Authorization": "Bearer t"}
    ) as http:
        direct = asyncio.create_task(http.post("/ng-rollout/direct-1/v1/responses", json={"input": "q"}))
        await _until(lambda: calls)
        prepared = await http.post("/ng-control/v1/checkpoint/prepare", json=_checkpoint("c1"))
        unattributed = await asyncio.wait_for(http.post("/v1/responses", json={"input": "q"}), 5)
        refused = await http.post(
            "/ng-control/v1/checkpoint/commit",
            json=_checkpoint("c1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "direct-1"}]),
        )
        await http.post("/ng-control/v1/checkpoint/resume", json=_checkpoint("c1"))
        retired = await http.post(
            "/ng-control/v1/checkpoint/retire", json=_checkpoint("retire", episode_ids=[{"rollout_id": "direct-1"}])
        )
        # The retire replied after the call it retired had stopped.
        stopped = direct.done()

    assert prepared.json()["phase"] == "prepared"
    assert prepared.json()["report"]["restarts"] == ["direct-1"] and prepared.json()["report"]["blockers"] == []
    assert unattributed.status_code == 400 and unattributed.json()["error"]["code"] == "checkpoint_rollout_id_required"
    assert refused.json()["error"]["code"] == "checkpoint_restart_in_scope"
    assert retired.status_code == 200 and stopped
    assert len(calls) == 1


async def test_an_activation_without_an_index_keeps_no_reply() -> None:
    """Only a caller that sends the activation index can ask for a reply again, so no other reply is kept."""
    server, server_client = _make_agent(False)
    request = await _restored_session(server, activations=0, boundary=None)
    server_client.post = AsyncMock(return_value=_message_response("resp-1"))

    await server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="q"))

    state = server._session_state(_SESSION_ID)
    assert state.completed_activation == 1 and state.completed_reply is None


async def test_a_restored_session_answered_with_its_reply_survives_a_commit_that_leaves_its_episode_out(
    tmp_path,
) -> None:
    """The replayed reply claims the restored session for the live episode, so a commit no longer frees it."""
    server, server_client = _make_agent(False)
    request = await _restored_session(server, activations=1, boundary=None)
    participant = server.checkpoint_participant
    controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
    server._session_state(_SESSION_ID).completed_reply = {"response": _FINAL_RESPONSE, "cookies": {}}
    server_client.post = AsyncMock(return_value=_mock_response(_FINAL_RESPONSE))
    request.headers = {AGENT_ACTIVATION_HEADER: "1"}

    reply = await server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="q"))
    pending = await participant.unclaimed()
    # Another participant reports the episode as a restart, so the commit leaves it out.
    await controller.prepare(CheckpointRequest(**_checkpoint("c1")))
    await controller.commit(CommitRequest(**_checkpoint("c1", checkpoint_dir=str(tmp_path), episode_ids=[])))
    await controller.resume(CheckpointRequest(**_checkpoint("c1")))

    assert json.loads(reply.body)["id"] == _FINAL_RESPONSE["id"]
    assert pending == []
    assert server._session_state(_SESSION_ID) is not None and participant.has_session(_SESSION_ID)
    assert server_client.post.await_count == 0


class _SlowCloseAgent(SimpleAgent):
    """A Simple Agent whose close hook awaits, as one that stops a sandbox does."""

    closing: Any = None
    release: Any = None
    events: Any = None

    async def _close_agent_session_state(self, state):
        self.events.append("cleanup")
        self.closing.set()
        await self.release.wait()
        return await super()._close_agent_session_state(state)


def _slow_close_agent() -> tuple[_SlowCloseAgent, MagicMock]:
    server, server_client = _make_agent(False, _SlowCloseAgent)
    server.closing, server.release, server.events = asyncio.Event(), asyncio.Event(), []
    return server, server_client


def _close(server: SimpleAgent):
    request = MagicMock(session={AGENT_SESSION_COOKIE_KEY: _SESSION_ID})
    body = AgentCloseSessionRequest(
        agent_session_id=_SESSION_ID, episode_id=EpisodeId(rollout_id="rollout", attempt=1)
    )
    return server.close_agent_session(request, body)


async def test_a_close_that_arrives_after_prepare_waits_for_resume_and_the_commit_exports_its_session(
    tmp_path,
) -> None:
    server, _ = _slow_close_agent()
    await _restored_session(server, activations=1, boundary=None)
    participant = server.checkpoint_participant
    controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)

    prepared = await controller.prepare(CheckpointRequest(**_checkpoint("c1")))
    closing = asyncio.create_task(_close(server))
    await _until(lambda: server._session_records[_SESSION_ID].lock.locked())
    committed = await controller.commit(CommitRequest(**_checkpoint("c1", checkpoint_dir=str(tmp_path))))
    closed_before_resume = closing.done() or server.closing.is_set()
    await controller.resume(CheckpointRequest(**_checkpoint("c1")))
    server.release.set()
    closed = await asyncio.wait_for(closing, 5)

    assert prepared["phase"] == "prepared"
    assert committed["episode_ids"] == ["rollout-a1"]
    assert not closed_before_resume
    assert closed.agent_session_id == _SESSION_ID and not participant.has_session(_SESSION_ID)


def _model_that_unwinds_slowly(server_client: MagicMock, events: list[str]) -> tuple[asyncio.Event, asyncio.Event]:
    """A model call that returns a tool call once ``reply`` is set, and whose cancellation waits for ``unwound``."""
    reply, unwound = asyncio.Event(), asyncio.Event()

    async def post(*, server_name, url_path, **kwargs):
        events.append(f"{server_name} {url_path}")
        if server_name != "model":
            return _mock_response(content="cold")
        try:
            await reply.wait()
        except asyncio.CancelledError:
            await unwound.wait()
            raise
        return _mock_response(_TOOL_CALL_RESPONSE)

    server_client.post = AsyncMock(side_effect=post)
    return reply, unwound


async def test_a_close_stops_the_activation_before_its_cleanup_starts() -> None:
    server, server_client = _slow_close_agent()
    request = await _restored_session(server, activations=0, boundary=None)
    reply, unwound = _model_that_unwinds_slowly(server_client, server.events)
    activation = asyncio.create_task(
        server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="q"))
    )
    await _until(lambda: server.events)

    closing = asyncio.create_task(_close(server))
    await _until(lambda: server.closing.is_set() or activation.cancelling())
    # The model replies now: an activation still running would go on to call the tool.
    reply.set()
    unwound.set()
    await asyncio.wait([activation], timeout=5)
    server.release.set()
    await asyncio.wait_for(closing, 5)

    assert activation.cancelled()
    assert server.events == ["model /ng-rollout/rollout-a1/v1/responses", "cleanup"]


async def test_a_close_cancelled_while_it_stops_the_activation_runs_the_cleanup_once_when_retried() -> None:
    server, server_client = _slow_close_agent()
    server.release.set()
    request = await _restored_session(server, activations=0, boundary=None)
    _, unwound = _model_that_unwinds_slowly(server_client, server.events)
    activation = asyncio.create_task(
        server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="q"))
    )
    await _until(lambda: server.events)

    # The caller disconnects while the close waits for the activation to stop.
    first = asyncio.create_task(_close(server))
    await _until(lambda: activation.cancelling())
    first.cancel()
    await asyncio.wait([first])
    unwound.set()
    await asyncio.wait([activation], timeout=5)
    closed = await asyncio.wait_for(_close(server), 5)

    assert first.cancelled() and activation.cancelled()
    assert closed.agent_session_id == _SESSION_ID
    assert server.events.count("cleanup") == 1


async def test_a_retire_that_frees_a_session_while_its_close_waits_leaves_no_receipt_behind(
    monkeypatch: MonkeyPatch,
) -> None:
    server, server_client = _make_agent(False)
    request = await _restored_session(server, activations=0, boundary=None)
    participant = server.checkpoint_participant
    events: list[str] = []
    _, unwound = _model_that_unwinds_slowly(server_client, events)
    activation = asyncio.create_task(
        server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="q"))
    )
    await _until(lambda: events)
    attempt = EpisodeId(rollout_id="rollout", attempt=1)
    await participant.mark_retired([attempt])
    retiring = asyncio.create_task(participant.retire(attempt))
    await _until(lambda: activation.cancelling())
    closing = asyncio.create_task(_close(server))
    await _until(lambda: server._session_records[_SESSION_ID].lock.locked())
    unwound.set()
    await asyncio.wait([activation, retiring, closing], timeout=5)

    # Long after the receipt would have expired, an unrelated seed sweeps the expired receipts.
    later = time.monotonic() + 3600
    monkeypatch.setattr("nemo_gym.agents.base_responses_api_agent.monotonic", lambda: later)
    other = AgentSeedSessionRequest(
        agent_session_id="other-session",
        episode_id=EpisodeId(rollout_id="other"),
        task_id=TaskId(taskset="example", task_id="1"),
    )
    seeded = await server.seed_agent_session(MagicMock(session={}), other, Response())

    assert retiring.done() and retiring.exception() is None
    assert isinstance(closing.exception(), StaleAttemptError)
    assert _SESSION_ID not in server._session_records and _SESSION_ID not in server._closed_session_records
    assert seeded.agent_session_id == "other-session"


_RESTORED_ATTEMPT = EpisodeId(rollout_id="rollout", attempt=1)


async def test_an_environment_episode_s_cleanup_close_finishes_before_the_agent_stage_of_prepare(tmp_path) -> None:
    """Prepare runs environment servers first, and a finishing episode blocks them until its cleanup ends.

    So the cleanup's agent close always reaches an agent whose admission is still open.
    """
    server, _ = _slow_close_agent()
    await _restored_session(server, activations=1, boundary=None)
    agent = ParticipantControlPlane(server.checkpoint_participant, instance_name="agent", lease_grace_seconds=60)
    environment = EnvironmentParticipant(mode="exported")
    environment_plane = ParticipantControlPlane(environment, instance_name="env", lease_grace_seconds=60)
    environment.begin(_RESTORED_ATTEMPT, {"task": 0}, None)
    await environment.boundary(_RESTORED_ATTEMPT, {"next": "return"})

    async def final_cleanup() -> None:
        environment.finishing(_RESTORED_ATTEMPT)
        await _close(server)
        await environment.end(_RESTORED_ATTEMPT)

    cleanup = asyncio.create_task(final_cleanup())
    await _until(lambda: server.closing.is_set())
    environment_prepare = asyncio.create_task(environment_plane.prepare(CheckpointRequest(**_checkpoint("c1"))))
    await _until(lambda: environment.steps.closed)
    # The finishing episode blocks the environment server, so prepare cannot reach the agent stage.
    environment_waited = not environment.ready() and not environment_prepare.done()
    server.release.set()
    prepared_environment = await asyncio.wait_for(environment_prepare, 5)
    closed_before_agent_stage = cleanup.done()
    prepared_agent = await asyncio.wait_for(agent.prepare(CheckpointRequest(**_checkpoint("c1"))), 5)

    assert environment_waited and closed_before_agent_stage
    assert prepared_environment["phase"] == prepared_agent["phase"] == "prepared"


async def test_a_cleanup_close_of_a_parked_episode_after_the_agent_stage_waits_without_holding_up_prepare(
    tmp_path,
) -> None:
    """An environment episode cancelled while parked, for example by a client disconnect, closes its agent session.

    The environment still exports the episode at its boundary until it ends, so the agent must export the session.
    """
    server, _ = _slow_close_agent()
    server.release.set()
    await _restored_session(server, activations=1, boundary=None)
    agent = ParticipantControlPlane(server.checkpoint_participant, instance_name="agent", lease_grace_seconds=60)
    environment = EnvironmentParticipant(mode="exported")
    environment_plane = ParticipantControlPlane(environment, instance_name="env", lease_grace_seconds=60)
    environment.begin(_RESTORED_ATTEMPT, {"task": 0}, None)
    await environment.boundary(_RESTORED_ATTEMPT, {"next": "close_agent"})

    async def episode() -> None:
        try:
            await environment.boundary(_RESTORED_ATTEMPT, {"next": "close_agent"})
        finally:
            environment.finishing(_RESTORED_ATTEMPT)
            await _close(server)
            await environment.end(_RESTORED_ATTEMPT)

    await environment_plane.prepare(CheckpointRequest(**_checkpoint("c1")))
    running = asyncio.create_task(episode())
    await _until(lambda: environment.steps.blocker_count() == 0 and not running.done())
    prepared_agent = await asyncio.wait_for(agent.prepare(CheckpointRequest(**_checkpoint("c1"))), 5)
    running.cancel()
    await _until(lambda: server._session_records[_SESSION_ID].lock.locked())
    still_ready = server.checkpoint_participant.ready() and environment.ready()
    environment_records = environment.export_records(None)
    agent_committed = await agent.commit(CommitRequest(**_checkpoint("c1", checkpoint_dir=str(tmp_path))))
    await agent.resume(CheckpointRequest(**_checkpoint("c1")))
    await environment_plane.resume(CheckpointRequest(**_checkpoint("c1")))
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 5)

    assert prepared_agent["phase"] == "prepared" and still_ready
    assert [record.episode_id for record in environment_records] == [_RESTORED_ATTEMPT]
    assert agent_committed["episode_ids"] == ["rollout-a1"]
    assert server.events == ["cleanup"]


def _observed_checkpoint_agent() -> tuple[SimpleAgent, MagicMock, Any]:
    """A Simple Agent with checkpointing and observability on, whose model answers every call at once."""
    server, server_client = _make_agent(True)
    server_client.global_config_dict = {
        "observability_enabled": True,
        "checkpoint": {"enabled": True, "control_auth_token": "t"},
    }
    server_client.post = AsyncMock(side_effect=lambda **kwargs: _mock_response(_FINAL_RESPONSE))
    return server, server_client, server.setup_webserver()


async def _seeded_session(http: httpx.AsyncClient) -> None:
    seed = AgentSeedSessionRequest(
        agent_session_id=_SESSION_ID,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="example", task_id="0"),
    )
    assert (await http.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))).status_code == 200


async def test_an_indexed_session_reply_and_its_replay_carry_no_trajectory_and_neither_does_the_export() -> None:
    """The session's evidence comes back at close, so the reply a checkpoint keeps holds no second copy of it."""
    server, _, app = _observed_checkpoint_agent()
    path = "/ng-rollout/rollout/v1/responses"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
        await _seeded_session(http)
        original = await http.post(path, json={"input": "q"}, headers={AGENT_ACTIVATION_HEADER: "1"})
        replayed = await http.post(path, json={"input": "q"}, headers={AGENT_ACTIVATION_HEADER: "1"})
        # A caller that sends no index, such as the UserSim environment server, still reads the trajectory.
        unindexed = await http.post(path, json={"input": "q"})
    exported = await server.export_agent_sessions([_SESSION_ID])

    assert original.status_code == 200 and "_ng_trajectory" not in original.json()
    assert replayed.content == original.content
    assert "_ng_trajectory" not in json.dumps(exported)
    # The evidence is still there, for the close to return.
    assert [record.invocation_id for record in server._session_state(_SESSION_ID).observations.records] == [
        "root",
        "activation-2",
    ]
    assert unindexed.json()["_ng_trajectory"]["rollout_id"] == "rollout"


async def test_an_indexed_reply_is_kept_unserialized_until_a_replay_or_an_export_needs_it() -> None:
    server, _, app = _observed_checkpoint_agent()
    path = "/ng-rollout/rollout/v1/responses"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
        await _seeded_session(http)
        original = await http.post(path, json={"input": "q"}, headers={AGENT_ACTIVATION_HEADER: "1"})
        kept = server._session_state(_SESSION_ID).completed_reply["response"]
        replayed = await http.post(path, json={"input": "q"}, headers={AGENT_ACTIVATION_HEADER: "1"})
    exported = (await server.export_agent_sessions([_SESSION_ID]))[_SESSION_ID]["completed_reply"]

    # The normal path keeps the response object; nothing is dumped until a replay or an export.
    assert isinstance(kept, NeMoGymResponse)
    assert replayed.content == original.content
    assert exported["response"] == original.json() == kept.model_dump(mode="json")


async def test_a_checkpoint_of_a_legacy_run_parked_at_verify_holds_its_reply_once_and_a_restore_still_replies(
    tmp_path,
) -> None:
    """The verify boundary holds the turn loop's reply, so the session no longer exports its own copy."""
    verifying, release = asyncio.Event(), asyncio.Event()

    def route(server: SimpleAgent, server_client: MagicMock) -> None:
        calls: list[str] = []
        _route_legacy_run(server, server_client, calls)
        answer = server_client.post.side_effect

        async def post(*, server_name, url_path, **kwargs):
            if url_path == "/seed_session":
                reply = _mock_response({})
                # A replayable verification, which a checkpoint does not wait for.
                reply.headers = {"x-ng-checkpoint-verify": "replay"}
                return reply
            if url_path == "/verify":
                verifying.set()
                await release.wait()
            return await answer(server_name=server_name, url_path=url_path, **kwargs)

        server_client.post = AsyncMock(side_effect=post)

    def observed_agent() -> tuple[SimpleAgent, MagicMock, ParticipantControlPlane]:
        server, server_client = _make_agent(True)
        server._checkpoint_participant = AgentSessionParticipant(server)
        controller = ParticipantControlPlane(
            server.checkpoint_participant, instance_name="agent", lease_grace_seconds=60
        )
        route(server, server_client)
        return server, server_client, controller

    server, _, controller = observed_agent()
    running = asyncio.create_task(server.run(MagicMock(cookies={}), _legacy_run_request(0)))
    await asyncio.wait_for(verifying.wait(), 5)
    await controller.prepare(CheckpointRequest(**_checkpoint("c1")))
    await controller.commit(CommitRequest(**_checkpoint("c1", checkpoint_dir=str(tmp_path))))
    [path] = (tmp_path / "gym" / "agent").glob("**/records-*.jsonl")
    [record] = [json.loads(line) for line in path.read_text().splitlines()]
    await controller.resume(CheckpointRequest(**_checkpoint("c1")))
    release.set()
    first = await running

    restored, _, restored_controller = observed_agent()
    scope = [EpisodeId(rollout_id="0-0")]
    await restored_controller.restore(
        RestoreRequest(**_checkpoint("r1", checkpoint_dir=str(tmp_path), episode_ids=scope))
    )
    await restored_controller.resume(CheckpointRequest(**_checkpoint("r1")))
    result = await restored.run(MagicMock(cookies={}), _legacy_run_request(1))

    assert record["episode"]["next"] == "verify"
    assert record["session"] == {}
    assert json.dumps(record).count('"_ng_trajectory"') == 1
    assert result.response.output == first.response.output and result.reward == 1.0
    assert result.model_dump()["ng_trajectory"]["rollout_id"] == "0-0-a1"


class _WeatherResources(SimpleResourcesServer):
    """A resources server that exports a per-session count of the tool calls it served."""

    checkpoint_mode: ClassVar[str] = "exported"
    calls: dict[str, int] = {}

    def setup_webserver(self) -> Any:
        app = super().setup_webserver()
        app.post("/get_weather")(self.get_weather)
        return app

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        self.calls.setdefault(request.session[SESSION_ID_KEY], 0)
        return BaseSeedSessionResponse()

    async def get_weather(self, request: Request) -> dict:
        self.calls[request.session[SESSION_ID_KEY]] += 1
        return {"weather": "cold"}

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)

    async def export_session_states(self, session_ids: list[str]) -> dict:
        return {session_id: self.calls[session_id] for session_id in session_ids if session_id in self.calls}

    async def restore_session_states(self, states: dict) -> None:
        self.calls.update(states)

    async def retire_session_state(self, session_id: str) -> None:
        self.calls.pop(session_id, None)


def _weather_resources() -> tuple[_WeatherResources, httpx.AsyncClient]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = OmegaConf.create({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="weather")
    server = _WeatherResources(config=config, server_client=server_client, calls={})
    app = server.setup_webserver()
    return server, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://resources:8000")


_RESOURCES_AUTH = {"authorization": "Bearer t"}


async def test_a_restored_session_s_direct_http_tool_calls_reach_the_restored_resources_session(
    monkeypatch: MonkeyPatch, tmp_path
) -> None:
    """A continuation through Simple Agent whose tool call reaches the restored session under the replacement's prefix.

    The grant still names the attempt that seeded the session, which the restore replaced.
    """
    resources, client = _weather_resources()
    async with client:
        await client.post("/ng-rollout/rollout/seed_session", json={})
        await client.post("/ng-rollout/rollout/get_weather", json={})
        cookies = dict(client.cookies)
        await client.post("/ng-control/v1/checkpoint/prepare", json=_checkpoint("c1"), headers=_RESOURCES_AUTH)
        await client.post(
            "/ng-control/v1/checkpoint/commit",
            json=_checkpoint("c1", checkpoint_dir=str(tmp_path)),
            headers=_RESOURCES_AUTH,
        )

    restored_resources, fresh = _weather_resources()
    server, server_client = _make_agent(False)
    participant = AgentSessionParticipant(server)
    server._checkpoint_participant = participant
    grant = DirectHTTPToolAccess(
        name="weather.direct_http",
        required=True,
        base_url="http://resources:8000/ng-rollout/rollout",
        cookies=cookies,
    )
    seed = AgentSeedSessionRequest(
        agent_session_id=_SESSION_ID,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="example", task_id="0"),
        tool_accesses=[grant],
    )
    session = {
        "request": seed.model_dump(mode="json"),
        "resources_cookies": cookies,
        "observations": None,
        "activations": 0,
        "completed_activation": 0,
        "completed_reply": None,
    }
    await participant.install(
        [AgentSessionRecord(session_key=_SESSION_ID, episode_id=EpisodeId(rollout_id="rollout"), session=session)],
        [EpisodeId(rollout_id="rollout")],
    )
    await participant.open_admission()
    server_client.post = AsyncMock(side_effect=[_mock_response(_TOOL_CALL_RESPONSE), _mock_response(_FINAL_RESPONSE)])
    paths: list[str] = []

    async def resources_request(*, method: str, url: str, json: Any, cookies: Any, headers: Any, _internal: bool):
        path = url.removeprefix("http://resources:8000")
        paths.append(path)
        fresh.cookies.clear()
        fresh.cookies.update(cookies or {})
        reply = await fresh.request(method, path, json=json, headers=headers)
        response = MagicMock(status=reply.status_code, cookies={})
        response.content.read = AsyncMock(return_value=reply.content)
        return response

    monkeypatch.setattr("responses_api_agents.simple_agent.app.http_request", resources_request)
    async with fresh:
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=_checkpoint("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "rollout"}]),
            headers=_RESOURCES_AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=_checkpoint("r1"), headers=_RESOURCES_AUTH)
        request = MagicMock(cookies={}, session={AGENT_SESSION_COOKIE_KEY: _SESSION_ID}, headers={})
        request.path_params = {"rollout_id": "rollout-a1"}
        await server.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="weather?"))

    assert paths == ["/ng-rollout/rollout-a1/get_weather"]
    # The restored count of one, and the replacement's call.
    assert list(restored_resources.calls.values()) == [2]


@pytest.mark.parametrize("resources_verify_step_type, closes", [("replay", True), ("wait", False)])
async def test_a_legacy_run_closes_its_resources_session_after_a_verification_a_checkpoint_does_not_wait_for(
    resources_verify_step_type: str, closes: bool
) -> None:
    server, server_client = _make_agent(False)
    server._checkpoint_participant = AgentSessionParticipant(server)
    calls: list[tuple[str, dict]] = []

    async def post(*, url_path, **kwargs):
        calls.append((url_path, kwargs))
        if url_path == "/seed_session":
            response = _mock_response({})
            response.headers = {"x-ng-checkpoint-verify": resources_verify_step_type}
            response.cookies = {"session": MagicMock(value="resources-cookie")}
            return response
        if url_path.endswith("/v1/responses"):
            # The turn loop hands the resources cookie back to /run.
            response = _mock_response(_FINAL_RESPONSE)
            response.cookies = {"session": MagicMock(value="resources-cookie")}
            return response
        if url_path == "/close_session":
            return _mock_response({"resources_session_id": kwargs["json"]["resources_session_id"]})
        return _mock_response(kwargs["json"] | {"reward": 1.0})

    server_client.post = AsyncMock(side_effect=post)

    result = await server.run(MagicMock(cookies={}), _legacy_run_request(0))

    [verify] = [kwargs for path, kwargs in calls if path == "/verify"]
    closes_seen = [kwargs for path, kwargs in calls if path == "/close_session"]
    assert result.reward == 1.0
    # The resources server keeps a session whose caller says it closes it, so a restore can verify again.
    assert verify["headers"] == ({"x-ng-checkpoint-close-follows": "1"} if closes else {})
    if closes:
        [close] = closes_seen
        assert close["cookies"] == {"session": "resources-cookie"}
        assert close["json"]["episode_id"] == {"rollout_id": "0-0", "attempt": 0}
        assert [path for path, _ in calls][-1] == "/close_session"
    else:
        assert closes_seen == []


async def test_a_skip_verification_run_parks_at_its_return_boundary_until_resume(tmp_path) -> None:
    """With ``skip_verification``, prepare can finish while the self-dispatched turn loop's reply is on its way back.

    The /run then parks at a return boundary that holds the configured reward instead of finishing,
    so the commit still holds its episode until resume.
    A restore from that commit returns the reward without running the model or the tool loop again.
    """

    def skipping_agent() -> tuple[SimpleAgent, MagicMock, ParticipantControlPlane]:
        server, server_client, controller = _checkpoint_agent()
        server.config.skip_verification = True
        server.config.skip_verification_reward = 0.25
        return server, server_client, controller

    server, server_client, controller = skipping_agent()
    participant = server.checkpoint_participant
    prepared: list[dict] = []

    async def checkpoint() -> None:
        # The turn loop has replied, and its reply has not reached /run yet.
        prepared.append(await controller.prepare(CheckpointRequest(**_checkpoint("c1"))))

    first_calls: list[str] = []
    _route_legacy_run(server, server_client, first_calls, after_turn_loop=checkpoint)
    running = asyncio.create_task(server.run(MagicMock(cookies={}), _legacy_run_request(0)))

    def parked_at_return() -> bool:
        return (participant.legacy_episodes.exported().get("run:0-0") or {}).get("next") == "return"

    await _until(lambda: running.done() or parked_at_return())
    assert prepared[0]["phase"] == "prepared"
    # Parked, so its legacy session is still exported.
    assert not running.done() and participant.has_session("run:0-0")
    commit = await controller.commit(
        CommitRequest(**_checkpoint("c1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "0-0"}]))
    )
    [path] = (tmp_path / "gym" / "agent").glob("**/records-*.jsonl")
    [record] = [json.loads(line) for line in path.read_text().splitlines()]
    await controller.resume(CheckpointRequest(**_checkpoint("c1")))
    first = await running

    restored, restored_client, restored_controller = skipping_agent()
    scope = [EpisodeId(rollout_id="0-0")]
    await restored_controller.restore(
        RestoreRequest(**_checkpoint("r1", checkpoint_dir=str(tmp_path), episode_ids=scope))
    )
    await restored_controller.resume(CheckpointRequest(**_checkpoint("r1")))
    calls: list[str] = []
    _route_legacy_run(restored, restored_client, calls)
    result = await restored.run(MagicMock(cookies={}), _legacy_run_request(1))

    assert commit["episode_ids"] == ["0-0"]
    assert record["episode"]["next"] == "return" and record["episode"]["result"]["reward"] == 0.25
    assert first.reward == 0.25 and first.verification_skipped
    assert "resources /verify" not in first_calls
    # Restored at its return step: no seed, model call, tool call, or verification runs again.
    assert calls == []
    assert result.reward == 0.25 and result.response.output == first.response.output


def _model_http_error(status: int) -> MagicMock:
    response = _mock_response(status=status, content='{"error":{"message":"controlled model failure"}}')
    url = URL("http://model/v1/responses")
    response.raise_for_status.side_effect = ClientResponseError(
        RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url), (), status=status, message="model failure"
    )
    return response


@pytest.mark.parametrize("status", [408, 425, 429, 500, 503])
async def test_model_http_retries_keep_one_decision(status: int, monkeypatch: MonkeyPatch) -> None:
    server, client = _make_agent(True)
    client.post = AsyncMock(
        side_effect=[_model_http_error(status), _model_http_error(status), _message_response("ok")]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("responses_api_agents.simple_agent.app.asyncio.sleep", sleep)

    response, trajectory, _, _ = await server._create_episode(
        NeMoGymResponseCreateParamsNonStreaming(input="hello"),
        model_url_path="/ng-rollout/rollout/v1/responses",
        in_session=True,
        collect_trajectory=True,
    )

    assert response.output[-1].content[0].text == "Done."
    assert client.post.await_count == 3
    requests = [call.kwargs for call in client.post.await_args_list]
    assert requests[0] == requests[1] == requests[2]
    assert requests[0]["headers"] == {"x-session-id": "root"}
    assert len(trajectory.turns) == 1
    assert trajectory.invocations[0].status == "completed"
    assert [ref.response_id for ref in trajectory.invocations[0].model_calls] == ["ok"]
    assert sleep.await_args_list == [call(0.5), call(1.0)]


@pytest.mark.parametrize(
    ("status", "retries", "attempts"), [(400, 2, 1), (401, 2, 1), (422, 2, 1), (429, 0, 1), (500, 2, 3)]
)
async def test_terminal_model_http_error_returns_cause_and_zero_usage(
    status: int, retries: int, attempts: int, monkeypatch: MonkeyPatch
) -> None:
    server, client = _make_agent(True)
    server.config.max_model_retries = retries
    client.post = AsyncMock(side_effect=[_model_http_error(status) for _ in range(attempts)])
    monkeypatch.setattr("responses_api_agents.simple_agent.app.asyncio.sleep", AsyncMock())

    response, trajectory, _, _ = await server._create_episode(
        NeMoGymResponseCreateParamsNonStreaming(input="hello"),
        model_url_path="/ng-rollout/rollout/v1/responses",
        in_session=True,
        collect_trajectory=True,
    )

    assert client.post.await_count == attempts
    assert response.status == "failed"
    assert "controlled model failure" in response.error.message
    assert AgentOutcome.model_validate_json(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == AgentOutcome(
        failure_kind="model_request_error", status_code=status
    )
    assert response.output == [] and response.usage.total_tokens == 0
    assert trajectory.turns == [] and trajectory.tool_calls == []
    invocation = trajectory.invocations[0]
    assert invocation.status == "failed" and invocation.error_type == "model_request_error"
    assert invocation.model_calls == []  # The rejected attempts have no provider response IDs.


@pytest.mark.parametrize("has_usage", [False, True])
async def test_exhausted_model_retries_preserve_partial_work_without_repeating_tools(
    has_usage: bool, monkeypatch: MonkeyPatch
) -> None:
    server, client = _make_agent(True)
    tool_response = {
        "id": "tool-response",
        "created_at": 1,
        "model": "model",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "function_call",
                "id": "tool-1",
                "call_id": "tool-1",
                "name": "bash",
                "arguments": '{"command":"printf done"}',
            }
        ],
    }
    if has_usage:
        tool_response["usage"] = {
            "input_tokens": 20,
            "output_tokens": 8,
            "total_tokens": 28,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        }
    client.post = AsyncMock(side_effect=[_mock_response(tool_response), *[_model_http_error(500) for _ in range(3)]])
    tool = AsyncMock(return_value=_mock_response(content="done"))
    monkeypatch.setattr("responses_api_agents.simple_agent.app.http_request", tool)
    monkeypatch.setattr("responses_api_agents.simple_agent.app.asyncio.sleep", AsyncMock())

    response, trajectory, _, _ = await server._create_episode(
        NeMoGymResponseCreateParamsNonStreaming(input="hello"),
        model_url_path="/ng-rollout/rollout/v1/responses",
        in_session=True,
        tool_access=DirectHTTPToolAccess(name="resources", base_url="http://resources", required=True),
        collect_trajectory=True,
    )

    assert response.status == "failed"
    assert client.post.await_count == 4
    assert tool.await_count == 1
    requests = [entry.kwargs["json"].model_dump() for entry in client.post.await_args_list]
    assert requests[1] == requests[2] == requests[3]
    assert requests[1]["input"][-1]["output"] == "done"
    assert response.output[-1].output == "done"
    assert len(trajectory.turns) == len(trajectory.tool_calls) == 1
    assert trajectory.tool_calls[0].status == "completed"
    assert [ref.response_id for ref in trajectory.invocations[0].model_calls] == ["tool-response"]
    if has_usage:
        assert response.usage.total_tokens == 28
    else:
        assert response.usage is None
