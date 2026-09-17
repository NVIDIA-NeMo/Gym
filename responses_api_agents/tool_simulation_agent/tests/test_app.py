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
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import httpx
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pytest import fixture, raises

from nemo_gym._checkpoint import (
    AGENT_EXECUTION_GENERATION_HEADER,
    AGENT_RESOURCE_DEPENDENCY_INDEX_FEATURE,
    CHECKPOINT_CONTROL_TOKEN_ENV,
    CONTROL_URL_PREFIX,
    RESOURCE_REQUEST_ID_HEADER,
    AgentBoundaryKind,
    AgentBoundaryRecord,
    AgentExecutionState,
    PendingModelPayload,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import MODEL_CALL_ID_HEADER
from nemo_gym.server_utils import ServerClient
from responses_api_agents.tool_simulation_agent.app import ToolSimulationAgent, ToolSimulationAgentConfig


def _drop_nulls(value):
    """Remove dictionary entries with a value of ``None`` recursively.

    SDK releases can add optional response fields.
    Exact payload comparisons should ignore these unset fields.
    Expected non-null values remain part of the comparison.
    """
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


def _calls_without_nulls(calls):
    return [(c.args, _drop_nulls(c.kwargs)) for c in calls]


class TestApp:
    @fixture
    def agent_config(self) -> ToolSimulationAgentConfig:
        return ToolSimulationAgentConfig(
            host="localhost",
            port=10001,
            entrypoint="",
            name="tool_agent",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="tool_resources_server",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="model_server",
            ),
        )

    def _set_server_client_post_responses(
        self,
        server_client_post_mock: AsyncMock,
        first_response: dict[str, Any],
        *additional_responses: dict[str, Any],
    ) -> None:
        server_client_post_mock.reset_mock(
            return_value=True,
            side_effect=True,
        )
        responses = [first_response]
        additional_responses_present = len(additional_responses) > 0
        if additional_responses_present:
            responses.extend(additional_responses)

        post_responses = []
        for response in responses:
            post_response_mock = AsyncMock()
            post_response_mock.ok = True
            post_response_mock.json.return_value = response
            post_response_mock.read.return_value = json.dumps(response).encode()
            post_responses.append(post_response_mock)

        if additional_responses_present:
            server_client_post_mock.side_effect = post_responses
        else:
            server_client_post_mock.return_value = post_responses[0]

    async def test_advertises_resource_dependency_index(
        self,
        agent_config: ToolSimulationAgentConfig,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv(CHECKPOINT_CONTROL_TOKEN_ENV, "checkpoint-secret")
        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = AsyncMock()
        agent_server = ToolSimulationAgent(config=agent_config, server_client=server_client_mock)

        capabilities = TestClient(agent_server.setup_webserver()).get(f"{CONTROL_URL_PREFIX}/capabilities")

        assert capabilities.status_code == 200
        assert AGENT_RESOURCE_DEPENDENCY_INDEX_FEATURE in capabilities.json()["features"]

    async def test_responses(self, agent_config: ToolSimulationAgentConfig) -> None:
        server_client_post_mock = AsyncMock()
        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = server_client_post_mock
        agent_server = ToolSimulationAgent(
            config=agent_config,
            server_client=server_client_mock,
        )
        webserver = agent_server.setup_webserver()
        test_client = TestClient(webserver)

        no_time_response_object = {
            "id": "no_created_at",
        }
        self._set_server_client_post_responses(server_client_post_mock, no_time_response_object)
        with raises(RuntimeError, match="Received an invalid response from the model server: "):
            test_client.post(
                "/v1/responses",
                json={
                    "input": [],
                },
            )

        server_client_post_mock.assert_called_once_with(
            server_name="model_server",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
            ),
        )

        chat_response_object = {
            "id": "chat_response_id",
            "created_at": 1,
            "model": "response_model",
            "object": "response",
            "output": [
                {
                    "id": "output_message_1_id",
                    "content": [
                        {
                            "annotations": [],
                            "text": "What is the question?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        self._set_server_client_post_responses(server_client_post_mock, chat_response_object)
        chat_response = test_client.post(
            "/v1/responses",
            json={
                "input": [
                    {
                        "role": "user",
                        "content": "I'd like to ask a question.",
                    }
                ]
            },
        )
        assert chat_response.status_code == 200
        expected_chat_response_json = {
            "id": "chat_response_id",
            "created_at": 1,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "response_model",
            "object": "response",
            "output": [
                {
                    "id": "output_message_1_id",
                    "content": [
                        {
                            "annotations": [],
                            "text": "What is the question?",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "conversation": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "prompt_cache_key": None,
            "reasoning": None,
            "safety_identifier": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
        }
        assert _drop_nulls(expected_chat_response_json) == _drop_nulls(chat_response.json())
        server_client_post_mock.assert_called_once_with(
            server_name="model_server",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    NeMoGymEasyInputMessage(
                        role="user",
                        content="I'd like to ask a question.",
                    )
                ],
            ),
        )

    async def test_run(self, agent_config: ToolSimulationAgentConfig) -> None:
        server_client_post_mock = AsyncMock()
        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = server_client_post_mock
        agent_server = ToolSimulationAgent(
            config=agent_config,
            server_client=server_client_mock,
        )
        webserver = agent_server.setup_webserver()
        test_client = TestClient(webserver)

        no_model_response_object = {
            "id": "no_model",
        }
        self._set_server_client_post_responses(server_client_post_mock, no_model_response_object)
        with raises(ValidationError, match="ToolSimulationAgentVerifyRequest"):
            test_client.post(
                "/run",
                json={
                    "responses_create_params": {
                        "input": [],
                    }
                },
            )

        server_client_post_mock.assert_called_once_with(
            server_name="tool_agent",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
            ),
        )

        tools = [
            {
                "name": "acknowledge",
                "parameters": None,
                "strict": None,
                "type": "function",
                "description": None,
            }
        ]
        tool_call_response_object = {
            "id": "tool_call_response_id",
            "created_at": 2,
            "model": "run_model",
            "object": "response",
            "output": [
                {
                    "arguments": "",
                    "call_id": "function_tool_call_1_id",
                    "name": "acknowledge",
                    "type": "function_call",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": tools,
        }
        invalid_verify_response_object = {
            "reward": 0.5,
        }
        self._set_server_client_post_responses(
            server_client_post_mock, tool_call_response_object, invalid_verify_response_object
        )
        with raises(ValidationError, match="ToolSimulationAgentVerifyResponse"):
            test_client.post(
                "/run",
                json={
                    "responses_create_params": {
                        "input": [
                            {
                                "role": "user",
                                "content": "Please provide an acknowledgment.",
                            }
                        ],
                        "tools": tools,
                    }
                },
            )

        full_tool_call_response = {
            "id": "tool_call_response_id",
            "created_at": 2,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "run_model",
            "object": "response",
            "output": [
                {
                    "arguments": "",
                    "call_id": "function_tool_call_1_id",
                    "name": "acknowledge",
                    "type": "function_call",
                    "id": None,
                    "status": None,
                }
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": tools,
            "top_p": None,
            "background": None,
            "conversation": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "prompt_cache_key": None,
            "reasoning": None,
            "safety_identifier": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
        }
        expected_invalid_verify_response_calls = [
            call(
                server_name="tool_agent",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[
                        NeMoGymEasyInputMessage(
                            role="user",
                            content="Please provide an acknowledgment.",
                        )
                    ],
                    tools=tools,
                ),
            ),
            call(
                server_name="tool_resources_server",
                url_path="/verify",
                json={
                    "responses_create_params": {
                        "background": None,
                        "include": None,
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": "Please provide an acknowledgment.",
                            }
                        ],
                        "instructions": None,
                        "max_output_tokens": None,
                        "max_tool_calls": None,
                        "metadata": None,
                        "model": None,
                        "parallel_tool_calls": True,
                        "previous_response_id": None,
                        "prompt": None,
                        "reasoning": None,
                        "service_tier": None,
                        "store": None,
                        "temperature": None,
                        "text": None,
                        "tool_choice": "auto",
                        "tools": tools,
                        "top_logprobs": None,
                        "top_p": None,
                        "truncation": None,
                        "user": None,
                        "stream": None,
                    },
                    "response": full_tool_call_response,
                },
            ),
        ]
        assert _calls_without_nulls(server_client_post_mock.call_args_list) == _calls_without_nulls(
            expected_invalid_verify_response_calls
        )

        valid_verify_response_object = {
            "responses_create_params": {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Could you acknowledge this query?",
                    }
                ],
                "tools": tools,
            },
            "response": tool_call_response_object,
            "reward": 1,
        }
        self._set_server_client_post_responses(
            server_client_post_mock, tool_call_response_object, valid_verify_response_object
        )
        valid_verify_response = test_client.post(
            "/run",
            json={
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": "Please provide an acknowledgment.",
                        }
                    ],
                    "tools": tools,
                }
            },
        )
        assert valid_verify_response.status_code == 200
        expected_valid_verify_response_json = {
            "responses_create_params": {
                "background": None,
                "include": None,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Could you acknowledge this query?",
                    }
                ],
                "instructions": None,
                "max_output_tokens": None,
                "max_tool_calls": None,
                "metadata": None,
                "model": None,
                "parallel_tool_calls": True,
                "previous_response_id": None,
                "prompt": None,
                "reasoning": None,
                "service_tier": None,
                "store": None,
                "temperature": None,
                "text": None,
                "tool_choice": "auto",
                "tools": tools,
                "top_logprobs": None,
                "top_p": None,
                "truncation": None,
                "user": None,
                "stream": None,
            },
            "response": full_tool_call_response,
            "reward": 1,
            "mask_sample": False,
            "failure_kind": None,
            "failure_reason": None,
        }
        assert _drop_nulls(expected_valid_verify_response_json) == _drop_nulls(valid_verify_response.json())
        assert _calls_without_nulls(server_client_post_mock.call_args_list) == _calls_without_nulls(
            expected_invalid_verify_response_calls
        )

    async def test_run_skip_verification_uses_configured_reward(self, agent_config: ToolSimulationAgentConfig) -> None:
        server_client_post_mock = AsyncMock()
        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = server_client_post_mock
        agent_server = ToolSimulationAgent(
            config=agent_config.model_copy(
                update={
                    "skip_verification": True,
                    "skip_verification_reward": 0.5,
                }
            ),
            server_client=server_client_mock,
        )
        webserver = agent_server.setup_webserver()
        test_client = TestClient(webserver)

        response_object = {
            "id": "chat_response_id",
            "created_at": 1,
            "model": "response_model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        self._set_server_client_post_responses(server_client_post_mock, response_object)

        response = test_client.post(
            "/run",
            json={
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": "Please answer directly.",
                        }
                    ]
                }
            },
        )

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["reward"] == 0.5
        assert response_json["verification_skipped"] is True
        assert response_json["response"]["id"] == "chat_response_id"
        server_client_post_mock.assert_called_once_with(
            server_name="tool_agent",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    NeMoGymEasyInputMessage(
                        role="user",
                        content="Please answer directly.",
                    )
                ],
            ),
        )

    async def test_run_commits_completed_model_before_verify(
        self,
        agent_config: ToolSimulationAgentConfig,
    ) -> None:
        rollout_id = "tool-simulation-rollout"
        model_call_id = "model-call-1"
        model_response = {
            "id": "response-1",
            "created_at": 1,
            "model": "response_model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        verify_result = {
            "responses_create_params": {"input": []},
            "response": model_response,
            "reward": 1.0,
        }

        model_http_response = AsyncMock()
        model_http_response.ok = True
        model_http_response.headers = {MODEL_CALL_ID_HEADER: model_call_id}
        model_http_response.json.return_value = model_response
        model_http_response.read.return_value = json.dumps(model_response).encode()
        verify_http_response = AsyncMock()
        verify_http_response.ok = True
        verify_http_response.headers = {}
        verify_http_response.json.return_value = verify_result
        verify_http_response.read.return_value = json.dumps(verify_result).encode()

        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = AsyncMock(side_effect=[model_http_response, verify_http_response])
        agent_server = ToolSimulationAgent(config=agent_config, server_client=server_client_mock)
        participant = agent_server.checkpoint_participant()
        test_client = TestClient(agent_server.setup_webserver())

        response = test_client.post(
            "/run",
            json={
                "_ng_rollout_id": rollout_id,
                "_ng_attempt_index": 0,
                "responses_create_params": {"input": []},
            },
        )

        assert response.status_code == 200
        execution = participant.resolve(rollout_id, 0)
        assert execution is not None
        assert execution.state == AgentExecutionState.COMPLETED
        assert execution.boundary is not None
        assert execution.boundary.boundary_kind == AgentBoundaryKind.PENDING_MODEL
        assert execution.boundary.last_committed_model_capture_key == rollout_id
        assert execution.boundary.last_committed_model_call_id == model_call_id
        assert execution.boundary.resource_state_revisions == {}
        assert execution.boundary.pending_model is not None
        assert execution.boundary.pending_model.response == model_response
        assert participant.completion_receipt(rollout_id, 0).terminal_model_call_id == model_call_id

        model_call = server_client_mock.post.await_args_list[0]
        assert model_call.kwargs["headers"][AGENT_EXECUTION_GENERATION_HEADER] == "1"
        verify_call = server_client_mock.post.await_args_list[1]
        assert verify_call.kwargs["headers"][RESOURCE_REQUEST_ID_HEADER]

    async def test_run_restores_pending_model_without_regenerating(
        self,
        agent_config: ToolSimulationAgentConfig,
    ) -> None:
        rollout_id = "tool-simulation-rollout"
        model_call_id = "model-call-1"
        verify_request_id = "verify-request-1"
        model_response = {
            "id": "response-1",
            "created_at": 1,
            "model": "response_model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        verify_result = {
            "responses_create_params": {"input": []},
            "response": model_response,
            "reward": 1.0,
        }
        verify_http_response = AsyncMock()
        verify_http_response.ok = True
        verify_http_response.headers = {}
        verify_http_response.json.return_value = verify_result
        verify_http_response.read.return_value = json.dumps(verify_result).encode()

        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = AsyncMock(return_value=verify_http_response)
        agent_server = ToolSimulationAgent(config=agent_config, server_client=server_client_mock)
        participant = agent_server.checkpoint_participant()
        participant.install_restored(
            [
                AgentBoundaryRecord(
                    rollout_id=rollout_id,
                    attempt_index=0,
                    boundary_index=1,
                    turn_index=1,
                    boundary_kind=AgentBoundaryKind.PENDING_MODEL,
                    pending_model=PendingModelPayload(
                        model_call_id=model_call_id,
                        response=model_response,
                        pending_action_cursor=0,
                        resource_request_id=verify_request_id,
                    ),
                    output_items=[],
                    last_committed_model_capture_key=rollout_id,
                    last_committed_model_call_id=model_call_id,
                )
            ]
        )
        await participant.resume()
        test_client = TestClient(agent_server.setup_webserver())

        response = test_client.post(
            "/run",
            json={
                "_ng_rollout_id": rollout_id,
                "_ng_attempt_index": 1,
                "responses_create_params": {"input": []},
            },
        )

        assert response.status_code == 200
        server_client_mock.post.assert_awaited_once()
        verify_call = server_client_mock.post.await_args
        assert verify_call.kwargs["server_name"] == "tool_resources_server"
        assert verify_call.kwargs["url_path"] == "/verify"
        assert verify_call.kwargs["headers"] == {RESOURCE_REQUEST_ID_HEADER: verify_request_id}
        receipt = participant.completion_receipt(rollout_id, 1)
        assert receipt.manifest_capture_key == rollout_id
        assert receipt.terminal_model_call_id == model_call_id

    async def test_checkpoint_prepare_freezes_inflight_verify(
        self,
        agent_config: ToolSimulationAgentConfig,
    ) -> None:
        rollout_id = "tool-simulation-rollout"
        verify_started = asyncio.Event()
        release_verify = asyncio.Event()
        model_response = {
            "id": "response-1",
            "created_at": 1,
            "model": "response_model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        verify_result = {
            "responses_create_params": {"input": []},
            "response": model_response,
            "reward": 1.0,
        }

        model_http_response = AsyncMock()
        model_http_response.ok = True
        model_http_response.headers = {MODEL_CALL_ID_HEADER: "model-call-1"}
        model_http_response.json.return_value = model_response
        model_http_response.read.return_value = json.dumps(model_response).encode()
        verify_http_response = AsyncMock()
        verify_http_response.ok = True
        verify_http_response.headers = {}
        verify_http_response.json.return_value = verify_result
        verify_http_response.read.return_value = json.dumps(verify_result).encode()

        async def post(*, server_name: str, **kwargs):
            del kwargs
            if server_name == "tool_agent":
                return model_http_response
            assert server_name == "tool_resources_server"
            verify_started.set()
            await release_verify.wait()
            return verify_http_response

        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = AsyncMock(side_effect=post)
        agent_server = ToolSimulationAgent(config=agent_config, server_client=server_client_mock)
        participant = agent_server.checkpoint_participant()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=agent_server.setup_webserver()),
            base_url="http://agent.test",
        ) as client:
            run_task = asyncio.create_task(
                client.post(
                    "/run",
                    json={
                        "_ng_rollout_id": rollout_id,
                        "_ng_attempt_index": 0,
                        "responses_create_params": {"input": []},
                    },
                )
            )
            await asyncio.wait_for(verify_started.wait(), timeout=1)

            report = await participant.prepare(time.time() + 1)
            assert report["ready_to_commit"] is True
            assert report["parked_with_boundary"] == 1
            assert report["selected_boundaries"][0]["boundary_kind"] == AgentBoundaryKind.PENDING_MODEL.value

            release_verify.set()
            await asyncio.sleep(0)
            assert not run_task.done()

            await participant.resume()
            response = await asyncio.wait_for(run_task, timeout=1)

        assert response.status_code == 200
