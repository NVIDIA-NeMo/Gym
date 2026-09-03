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
import json
from unittest.mock import AsyncMock, MagicMock, call

import orjson
import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.rollout_collection import _attach_trajectory_record
from nemo_gym.rollout_observability import TrajectoryRecord
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import (
    ModelServerRef,
    ResourcesServerRef,
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
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

    async def test_responses_continues_on_reasoning_only(self, monkeypatch: MonkeyPatch) -> None:
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
            call(
                server_name="my server name",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[
                        NeMoGymEasyInputMessage(content="hello", role="user", type="message"),
                        NeMoGymResponseReasoningItem(
                            id="msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                            summary=[NeMoGymSummary(text="I'm thinking how to respond", type="summary_text")],
                            type="reasoning",
                            encrypted_content=None,
                            status="completed",
                        ),
                    ]
                ),
                cookies=dotjson_mock.cookies,
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
            "incomplete_details": None,
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
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "summary": [
                        {
                            "text": "Hello! How can I help you today?",
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

    async def test_incomplete_details(self, monkeypatch: MonkeyPatch) -> None:
        await self._test_incomplete_details_helper(monkeypatch, {"reason": "max_output_tokens"})
        await self._test_incomplete_details_helper(monkeypatch, {"reason": "content_filter"})

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
