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
import logging
from typing import Any, Union
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch, mark, raises

import nemo_gym.server_utils
from nemo_gym import PARENT_DIR
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionAssistantMessageForTrainingParam,
    NeMoGymChatCompletionAssistantMessageParam,
    NeMoGymChatCompletionContentPartTextParam,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChatCompletionMessageForTraining,
    NeMoGymChatCompletionMessageToolCall,
    NeMoGymChatCompletionMessageToolCallFunctionParam,
    NeMoGymChatCompletionMessageToolCallParam,
    NeMoGymChatCompletionSystemMessageParam,
    NeMoGymChatCompletionToolMessageParam,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymChoice,
    NeMoGymEasyInputMessage,
    NeMoGymFunction,
    NeMoGymFunctionCallOutput,
    NeMoGymFunctionToolParam,
    NeMoGymMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInput,
    NeMoGymResponseInputText,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.token_id_capture import (
    CaptureContext,
    InMemoryLineageStore,
    TokenCaptureStore,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from responses_api_models.vllm_model.app import (
    VLLMConverter,
    VLLMModel,
    VLLMModelConfig,
    _append_transport_io,
    _transport_images,
    _transport_log_context,
)


# Used for mocking created_at timestamp generation
FIXED_TIME = 1691418000
FIXED_UUID = "123"

_TEST_LINEAGE = InMemoryLineageStore()


def lineage_index():
    return _TEST_LINEAGE.index


def test_transport_io_writer_keeps_full_payload(monkeypatch: MonkeyPatch, tmp_path) -> None:
    log_path = tmp_path / "model-io-transport.jsonl"
    monkeypatch.setenv("NEMO_GYM_VLLM_TRANSPORT_LOG", str(log_path))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
                {"type": "text", "text": "inspect"},
            ],
        }
    ]

    _append_transport_io(
        {
            "schema_version": 1,
            "event": "transport_request",
            "request_payload": {"messages": messages},
            "embedded_images": _transport_images(messages),
        }
    )

    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["request_payload"]["messages"] == messages
    assert row["embedded_images"][0]["decoded_bytes"] == 3


def test_transport_log_context_reads_generic_headers_without_body_fields() -> None:
    request = MagicMock()
    request.headers = {
        "x-nemo-gym-log-run-id": "run-001",
        "x-nemo-gym-log-adapter": "gym",
        "x-nemo-gym-log-task-id": "task-001",
        "x-nemo-gym-log-domain": "chrome",
        "x-nemo-gym-log-task-attempt": "2",
        "x-nemo-gym-log-step": "3",
        "x-nemo-gym-log-parse-attempt": "1",
    }

    assert _transport_log_context(request) == {
        "run_id": "run-001",
        "adapter": "gym",
        "task_id": "task-001",
        "domain": "chrome",
        "task_attempt": 2,
        "step": 3,
        "parse_attempt": 1,
    }


class FakeUUID:
    """Used for mocking UUIDs"""

    hex = FIXED_UUID


COMMON_RESPONSE_PARAMS = dict(
    parallel_tool_calls=True,
    status="completed",
    tool_choice="auto",
)

PARAMETERIZE_DATA = [
    # ----- EasyInputMessageParam: content as a list, id: "ez_list" -----
    (
        [
            NeMoGymEasyInputMessage(
                role="user",
                content=[{"type": "input_text", "text": "hello"}],
                type="message",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                NeMoGymChatCompletionUserMessageParam(
                    content=[
                        NeMoGymChatCompletionContentPartTextParam(
                            text="hello",
                            type="text",
                        ),
                    ],
                    role="user",
                )
            ],
        ),
        NeMoGymChatCompletion(
            id="chtcmpl-123",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(role="assistant", content="hi :) how are you?"),
                )
            ],
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
        ),
        NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            created_at=FIXED_TIME,
            model="dummy_model",
            tools=[],
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            text="hi :) how are you?",
                            type="output_text",
                            annotations=[],
                        )
                    ],
                )
            ],
            object="response",
        ),
    ),
    # ----- EasyInputMessageParam: content as a string, id: "ez_str" -----
    (
        [
            NeMoGymEasyInputMessage(
                role="user",
                content="hello",
                type="message",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                NeMoGymChatCompletionUserMessageParam(
                    content="hello",
                    role="user",
                )
            ],
        ),
        NeMoGymChatCompletion(
            id="chtcmpl-123",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(role="assistant", content="hi :) how are you?"),
                )
            ],
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
        ),
        NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            created_at=FIXED_TIME,
            model="dummy_model",
            tools=[],
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            text="hi :) how are you?",
                            type="output_text",
                            annotations=[],
                        )
                    ],
                )
            ],
            object="response",
        ),
    ),
    # ----- EasyInputMessageParam: content as a string, id: "str_only" -----
    (
        "hello",
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                NeMoGymChatCompletionUserMessageParam(
                    content=[
                        NeMoGymChatCompletionContentPartTextParam(
                            type="text",
                            text="hello",
                        )
                    ],
                    role="user",
                )
            ],
        ),
        NeMoGymChatCompletion(
            id="chtcmpl-123",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(role="assistant", content="hi :) how are you?"),
                )
            ],
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
        ),
        NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            created_at=FIXED_TIME,
            model="dummy_model",
            tools=[],
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            text="hi :) how are you?",
                            type="output_text",
                            annotations=[],
                        )
                    ],
                )
            ],
            object="response",
        ),
    ),
    # ----- Message, id: "input_msg" -----
    (
        [
            NeMoGymMessage(
                content=[
                    {
                        "text": "hello",
                        "type": "input_text",
                    }
                ],
                role="user",
                status="completed",
                type="message",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                NeMoGymChatCompletionUserMessageParam(
                    content=[NeMoGymChatCompletionContentPartTextParam(type="text", text="hello")],
                    role="user",
                )
            ],
        ),
        NeMoGymChatCompletion(
            id="chtcmpl-123",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(role="assistant", content="hi :) how are you?"),
                )
            ],
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
        ),
        NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            created_at=FIXED_TIME,
            model="dummy_model",
            tools=[],
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            text="hi :) how are you?",
                            type="output_text",
                            annotations=[],
                        )
                    ],
                )
            ],
            object="response",
        ),
    ),
    # ----- ResponseFunctionToolCallParam, id: "tc" -----
    (
        [
            NeMoGymResponseFunctionToolCall(
                arguments='{"city":"San Francisco"}',
                call_id="call_123",
                name="get_weather",
                type="function_call",
                id="func_123",
                status="completed",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                NeMoGymChatCompletionAssistantMessageParam(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        NeMoGymChatCompletionMessageToolCallParam(
                            id="call_123",
                            type="function",
                            function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                                arguments='{"city":"San Francisco"}',
                                name="get_weather",
                            ),
                        )
                    ],
                )
            ],
        ),
        NeMoGymChatCompletion(
            id="chtcmpl-123",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="Getting the weather for San Francisco, CA..",
                        tool_calls=[
                            NeMoGymChatCompletionMessageToolCall(
                                id="call_123",
                                function=NeMoGymFunction(
                                    name="get_weather",
                                    arguments='{"city":"San Francisco"}',
                                ),
                                type="function",
                            )
                        ],
                    ),
                )
            ],
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
        ),
        NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            created_at=FIXED_TIME,
            model="dummy_model",
            tools=[],
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            text="Getting the weather for San Francisco, CA..",
                            type="output_text",
                            annotations=[],
                        )
                    ],
                ),
                NeMoGymResponseFunctionToolCall(
                    arguments='{"city":"San Francisco"}',
                    call_id="call_123",
                    name="get_weather",
                    type="function_call",
                    id="call_123",
                    status="completed",
                ),
            ],
            object="response",
        ),
    ),
    # ----- FunctionCallOutput, id: "fc_output" -----
    (
        [
            NeMoGymFunctionCallOutput(
                call_id="call_123",
                output='{"temperature": 65, "condition": "partly cloudy", "humidity": 72}',
                type="function_call_output",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                NeMoGymChatCompletionToolMessageParam(
                    content='{"temperature": 65, "condition": "partly cloudy", "humidity": 72}',
                    role="tool",
                    tool_call_id="call_123",
                )
            ],
        ),
        NeMoGymChatCompletion(
            id="chtcmpl-123",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="It is 65 degrees Fahrenheit with 72% humidity in SF",
                    ),
                )
            ],
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
        ),
        NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            created_at=FIXED_TIME,
            model="dummy_model",
            tools=[],
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            text="It is 65 degrees Fahrenheit with 72% humidity in SF",
                            type="output_text",
                            annotations=[],
                        )
                    ],
                )
            ],
            object="response",
        ),
    ),
    # ----- ResponseReasoningItemParam, id: "rzning" -----
    (
        [
            NeMoGymResponseReasoningItem(
                id="rs_123",
                summary=[
                    NeMoGymSummary(
                        text="I have identified the city as San Francisco based on user input.",
                        type="summary_text",
                    )
                ],
                type="reasoning",
                status="completed",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                NeMoGymChatCompletionAssistantMessageParam(
                    role="assistant",
                    content="<think>I have identified the city as San Francisco based on user input.</think>",
                )
            ],
        ),
        NeMoGymChatCompletion(
            id="chtcmpl-123",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="<think>I have identified the city as San Francisco based on user input.</think>",
                    ),
                )
            ],
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
        ),
        NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            created_at=FIXED_TIME,
            model="dummy_model",
            tools=[],
            output=[
                NeMoGymResponseReasoningItem(
                    id="rs_123",
                    type="reasoning",
                    summary=[
                        NeMoGymSummary(
                            text="I have identified the city as San Francisco based on user input.",
                            type="summary_text",
                        )
                    ],
                    status="completed",
                ),
            ],
            object="response",
        ),
    ),
    # ----- Multi-reasoning, id: "multi_rzning" -----
    (
        [
            NeMoGymResponseReasoningItem(
                id="rs_123",
                summary=[
                    NeMoGymSummary(
                        text="I'll first think about the user's question.",
                        type="summary_text",
                    ),
                    NeMoGymSummary(text="Then I will answer.", type="summary_text"),
                ],
                type="reasoning",
                status="completed",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                NeMoGymChatCompletionAssistantMessageParam(
                    role="assistant",
                    content="<think>I'll first think about the user's question.</think><think>Then I will answer.</think>",
                )
            ],
        ),
        NeMoGymChatCompletion(
            id="chtcmpl-123",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="<think>I'll first think about the user's question.</think><think>Then I will answer.</think>Hello!",
                    ),
                )
            ],
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
        ),
        NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            created_at=FIXED_TIME,
            model="dummy_model",
            tools=[],
            output=[
                NeMoGymResponseReasoningItem(
                    id="rs_123",
                    type="reasoning",
                    summary=[
                        NeMoGymSummary(
                            text="I'll first think about the user's question.",
                            type="summary_text",
                        ),
                        NeMoGymSummary(
                            text="Then I will answer.",
                            type="summary_text",
                        ),
                    ],
                    status="completed",
                ),
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            text="Hello!",
                            type="output_text",
                            annotations=[],
                        )
                    ],
                ),
            ],
            object="response",
        ),
    ),
    # ----- ResponseOutputMessageParam, id: "output_msg" -----
    (
        [
            NeMoGymResponseOutputMessage(
                id="msg_123",
                role="assistant",
                content=[
                    NeMoGymResponseOutputText(
                        text="Hello! How can I assist you today?",
                        type="output_text",
                        annotations=[],
                    )
                ],
                type="message",
                status="completed",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                NeMoGymChatCompletionAssistantMessageParam(
                    role="assistant",
                    content="Hello! How can I assist you today?",
                )
            ],
        ),
        NeMoGymChatCompletion(
            id="chtcmpl-123",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="By the way, I can give you the current weather if you provide a city and region.",
                    ),
                )
            ],
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
        ),
        NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            created_at=FIXED_TIME,
            model="dummy_model",
            tools=[],
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            text="By the way, I can give you the current weather if you provide a city and region.",
                            type="output_text",
                            annotations=[],
                        )
                    ],
                )
            ],
            object="response",
        ),
    ),
]


class TestApp:
    def _setup_server(self, monkeypatch: MonkeyPatch):
        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8081,
            base_url="http://api.openai.com/v1",
            api_key="dummy_key",  # pragma: allowlist secret
            model="dummy_model",
            entrypoint="",
            name="",
            return_token_id_information=False,
            uses_reasoning_parser=False,
        )

        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = dict()
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))

    async def test_sanity(self, monkeypatch: MonkeyPatch) -> None:
        self._setup_server(monkeypatch)

    def test_session_client_routing_is_stable_across_workers(self, monkeypatch: MonkeyPatch) -> None:
        workers = [self._setup_server(monkeypatch) for _ in range(2)]
        for worker in workers:
            worker._clients = [MagicMock(spec=NeMoGymAsyncOpenAI) for _ in range(4)]

        workers[0]._session_id_to_client = {"prior-a": workers[0]._clients[0]}
        workers[1]._session_id_to_client = {
            "prior-b": workers[1]._clients[0],
            "prior-c": workers[1]._clients[1],
            "prior-d": workers[1]._clients[2],
        }
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "target-session"}

        client_indices = []
        for worker in workers:
            selected_client = worker._resolve_client(request)
            client_indices.append(next(i for i, client in enumerate(worker._clients) if client is selected_client))

        assert client_indices[0] == client_indices[1]

    def test_responses_multistep(self, monkeypatch: MonkeyPatch):
        server = self._setup_server(monkeypatch)
        app = server.setup_webserver()
        client = TestClient(app)

        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="<think>Gathering order status and delivery info...</think>",
                        tool_calls=[
                            NeMoGymChatCompletionMessageToolCall(
                                id="call_123",
                                function=NeMoGymFunction(
                                    name="get_order_status",
                                    arguments='{"order_id": "123"}',
                                ),
                                type="function",
                            ),
                            NeMoGymChatCompletionMessageToolCall(
                                id="call_234",
                                function=NeMoGymFunction(
                                    name="get_delivery_date",
                                    arguments='{"order_id": "234"}',
                                ),
                                type="function",
                            ),
                        ],
                    ),
                )
            ],
        )

        input_messages = [
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="Check my order status", type="input_text")],
                status="completed",
            ),
            NeMoGymEasyInputMessage(
                type="message",
                role="assistant",
                content=[NeMoGymResponseInputText(text="Sure, one sec.", type="input_text")],
                status="completed",
            ),
        ]

        input_tools = [
            NeMoGymFunctionToolParam(
                name="get_order_status",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order",
                        },
                    },
                    "required": ["order_id"],
                },
                type="function",
                description="Get the current status for a given order",
                strict=True,
            ),
            NeMoGymFunctionToolParam(
                name="get_delivery_date",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order",
                        },
                    },
                    "required": ["order_id"],
                },
                type="function",
                description="Get the estimated delivery date for a given order",
                strict=True,
            ),
        ]

        expected_response = NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            object="response",
            tools=input_tools,
            created_at=FIXED_TIME,
            model="dummy_model",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text="<think>Gathering order status and delivery info...</think>",
                            annotations=[],
                        )
                    ],
                ),
                NeMoGymResponseFunctionToolCall(
                    type="function_call",
                    name="get_order_status",
                    arguments='{"order_id": "123"}',
                    call_id="call_123",
                    status="completed",
                    id="call_123",
                ),
                NeMoGymResponseFunctionToolCall(
                    type="function_call",
                    name="get_delivery_date",
                    arguments='{"order_id": "234"}',
                    call_id="call_234",
                    status="completed",
                    id="call_234",
                ),
            ],
        )

        mock_method = AsyncMock(return_value=mock_chat_completion)
        monkeypatch.setattr(
            VLLMModel,
            "chat_completions",
            mock_method,
        )

        monkeypatch.setattr("responses_api_models.vllm_model.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            tools=input_tools,
        )

        response = client.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        data = response.json()

        expected_dict = expected_response.model_dump()
        assert data == expected_dict

        # Verify input_messages made it to the model
        assert mock_method.await_args is not None
        called_args, _ = mock_method.await_args
        sent_tools = called_args[1].tools

        def _standardize(messages: list) -> list:
            return [
                (
                    i["role"],
                    i["content"][0]["text"] if isinstance(i["content"], list) else i["content"],
                )
                for i in messages
            ]

        assert _standardize([m.model_dump() for m in input_messages]) == _standardize(called_args[1].messages)

        actual_sent_tools = [t["function"] for t in sent_tools]
        expected_sent_tools = [
            {
                "name": "get_order_status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order",
                        }
                    },
                    "required": ["order_id"],
                },
                "description": "Get the current status for a given order",
            },
            {
                "name": "get_delivery_date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order",
                        }
                    },
                    "required": ["order_id"],
                },
                "description": "Get the estimated delivery date for a given order",
            },
        ]
        assert expected_sent_tools == actual_sent_tools

    def test_responses_multiturn(self, monkeypatch: MonkeyPatch):
        server = self._setup_server(monkeypatch)
        app = server.setup_webserver()
        client = TestClient(app)

        mock_chat_completion_data = NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="<think>Searching for a location before analyzing weather patterns...</think>What city and/or region do you need weather data for?",
                        tool_calls=[],
                    ),
                )
            ],
            usage=None,
        )

        input_messages = [
            NeMoGymMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="Hello", type="input_text")],
                status="completed",
            ),
            NeMoGymResponseReasoningItem(
                id="rs_123",
                type="reasoning",
                summary=[
                    NeMoGymSummary(
                        type="summary_text",
                        text="Considering ways to greet the user...",
                    )
                ],
                status="completed",
            ),
            NeMoGymResponseOutputMessage(
                id="msg_123",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    NeMoGymResponseOutputText(type="output_text", text="Hi, how can I help?", annotations=[]),
                ],
            ),
            NeMoGymMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(type="input_text", text="What's the weather?")],
                status="completed",
            ),
        ]

        expected_response = NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            object="response",
            tools=[],
            created_at=FIXED_TIME,
            model="dummy_model",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    status="completed",
                    role="assistant",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text="<think>Searching for a location before analyzing weather patterns...</think>What city and/or region do you need weather data for?",
                            annotations=[],
                        )
                    ],
                ),
            ],
        )

        mock_method = AsyncMock(return_value=mock_chat_completion_data)
        monkeypatch.setattr(
            VLLMModel,
            "chat_completions",
            mock_method,
        )
        monkeypatch.setattr("responses_api_models.vllm_model.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
        )

        response = client.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        data = response.json()

        expected_dict = expected_response.model_dump()
        assert data == expected_dict

        # Verify input_messages made it to the model
        assert mock_method.await_args is not None
        called_args, _ = mock_method.await_args
        sent_messages = called_args[1].messages

        expected_sent_messages = [
            {"content": [{"text": "Hello", "type": "text"}], "role": "user"},
            {
                "content": "<think>Considering ways to greet the user...</think>Hi, how can I help?",
                "role": "assistant",
            },
            {
                "content": [{"text": "What's the weather?", "type": "text"}],
                "role": "user",
            },
        ]

        assert expected_sent_messages == sent_messages

    def test_responses_multistep_multiturn(self, monkeypatch: MonkeyPatch):
        server = self._setup_server(monkeypatch)
        app = server.setup_webserver()
        client = TestClient(app)

        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="<think>Order #1234 is shipped and scheduled for delivery tomorrow. Tomorrow's date is 2025-08-14. The next day is 2025-08-15 and is not a holiday. I need to send a note to the courier to update the delivery date to 2025-08-15.</think>",
                        tool_calls=[
                            NeMoGymChatCompletionMessageToolCall(
                                id="call_123",
                                function=NeMoGymFunction(
                                    name="get_order_status",
                                    arguments='{"order_id": "1234"}',
                                ),
                                type="function",
                            ),
                            NeMoGymChatCompletionMessageToolCall(
                                id="call_123",
                                function=NeMoGymFunction(
                                    name="get_delivery_date",
                                    arguments='{"order_id": "1234"}',
                                ),
                                type="function",
                            ),
                            NeMoGymChatCompletionMessageToolCall(
                                id="call_123",
                                function=NeMoGymFunction(
                                    name="reschedule_delivery",
                                    arguments='{"order_id": "1234", "date": "2025-08-15"}',
                                ),
                                type="function",
                            ),
                        ],
                    ),
                )
            ],
            usage=None,
        )

        input_messages = [
            NeMoGymMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="Hi, can you check my order?", type="input_text")],
                status="completed",
            ),
            NeMoGymResponseReasoningItem(
                id="rs_123",
                type="reasoning",
                summary=[
                    NeMoGymSummary(
                        type="summary_text",
                        text="Checking order details...",
                    )
                ],
                status="completed",
            ),
            NeMoGymResponseOutputMessage(
                id="msg_123",
                type="message",
                role="assistant",
                status="completed",
                content=[NeMoGymResponseOutputText(text="Sure, one sec.", type="output_text", annotations=[])],
            ),
            NeMoGymResponseOutputMessage(
                id="msg_123",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    NeMoGymResponseOutputText(
                        text="Gathering order status and delivery info..",
                        type="output_text",
                        annotations=[],
                    )
                ],
            ),
            NeMoGymResponseFunctionToolCall(
                type="function_call",
                call_id="call_123",
                name="get_order_status",
                arguments='{"order_id": "1234"}',
                status="completed",
            ),
            NeMoGymFunctionCallOutput(
                call_id="call_123",
                output='{"order_status": "shipped"}',
                type="function_call_output",
            ),
            NeMoGymResponseFunctionToolCall(
                type="function_call",
                call_id="call_123",
                name="get_delivery_date",
                arguments='{"order_id": "1234"}',
                status="completed",
            ),
            NeMoGymFunctionCallOutput(
                call_id="call_123",
                output='{"delivery_date": "2025-08-14"}',
                type="function_call_output",
            ),
            NeMoGymResponseOutputMessage(
                id="msg_123",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    NeMoGymResponseOutputText(
                        text="Order #1234 is shipped and arrives tomorrow.",
                        type="output_text",
                        annotations=[],
                    )
                ],
            ),
            NeMoGymMessage(
                type="message",
                role="user",
                content=[
                    NeMoGymResponseInputText(
                        text="I need to change my delivery date to the day after.",
                        type="input_text",
                    )
                ],
                status="completed",
            ),
        ]

        input_tools = [
            NeMoGymFunctionToolParam(
                name="reschedule_delivery",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order",
                        },
                        "date": {
                            "type": "date",
                            "description": "New delivery date in YYYY-MM-DD format",
                        },
                        "note": {
                            "type": "string",
                            "description": "Leave a note for the driver",
                        },
                    },
                    "required": ["order_id", "date"],
                },
                type="function",
                description="Request to postpone delivery to a later date",
                strict=True,
            ),
        ]

        expected_response = NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            object="response",
            tools=input_tools,
            created_at=FIXED_TIME,
            model="dummy_model",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    status="completed",
                    role="assistant",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text="<think>Order #1234 is shipped and scheduled for delivery tomorrow. Tomorrow's date is 2025-08-14. The next day is 2025-08-15 and is not a holiday. I need to send a note to the courier to update the delivery date to 2025-08-15.</think>",
                            annotations=[],
                        )
                    ],
                ),
                NeMoGymResponseFunctionToolCall(
                    call_id="call_123",
                    type="function_call",
                    name="get_order_status",
                    arguments='{"order_id": "1234"}',
                    status="completed",
                    id="call_123",
                ),
                NeMoGymResponseFunctionToolCall(
                    call_id="call_123",
                    type="function_call",
                    name="get_delivery_date",
                    arguments='{"order_id": "1234"}',
                    status="completed",
                    id="call_123",
                ),
                NeMoGymResponseFunctionToolCall(
                    call_id="call_123",
                    type="function_call",
                    name="reschedule_delivery",
                    arguments='{"order_id": "1234", "date": "2025-08-15"}',
                    status="completed",
                    id="call_123",
                ),
            ],
        )

        mock_method = AsyncMock(return_value=mock_chat_completion)
        monkeypatch.setattr(
            VLLMModel,
            "chat_completions",
            mock_method,
        )
        monkeypatch.setattr("responses_api_models.vllm_model.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            tools=input_tools,
        )

        response = client.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        data = response.json()

        expected_dict = expected_response.model_dump()
        assert data == expected_dict

        # Verify input_messages made it to the model
        assert mock_method.await_args is not None
        called_args, _ = mock_method.await_args
        sent_messages = called_args[1].messages
        sent_tools = called_args[1].tools

        expected_sent_messages = [
            {
                "content": [{"text": "Hi, can you check my order?", "type": "text"}],
                "role": "user",
            },
            {
                "content": "<think>Checking order details...</think>Sure, one sec.Gathering order status and delivery info..",
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "arguments": '{"order_id": "1234"}',
                            "name": "get_order_status",
                        },
                        "type": "function",
                    }
                ],
            },
            {
                "content": '{"order_status": "shipped"}',
                "role": "tool",
                "tool_call_id": "call_123",
            },
            {
                "content": None,
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "arguments": '{"order_id": "1234"}',
                            "name": "get_delivery_date",
                        },
                        "type": "function",
                    }
                ],
            },
            {
                "content": '{"delivery_date": "2025-08-14"}',
                "role": "tool",
                "tool_call_id": "call_123",
            },
            {
                "content": "Order #1234 is shipped and arrives tomorrow.",
                "role": "assistant",
            },
            {
                "content": [
                    {
                        "text": "I need to change my delivery date to the day after.",
                        "type": "text",
                    }
                ],
                "role": "user",
            },
        ]

        assert expected_sent_messages == sent_messages

        actual_sent_tools = [t["function"] for t in sent_tools]
        expected_sent_tools = [
            {
                "name": "reschedule_delivery",
                "description": "Request to postpone delivery to a later date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order",
                        },
                        "date": {
                            "type": "date",
                            "description": "New delivery date in YYYY-MM-DD format",
                        },
                        "note": {
                            "type": "string",
                            "description": "Leave a note for the driver",
                        },
                    },
                    "required": ["order_id", "date"],
                },
            }
        ]
        assert expected_sent_tools == actual_sent_tools

    @mark.parametrize(
        "single_input, _, mock_chat_completion, expected_response",
        PARAMETERIZE_DATA,
        ids=[
            "ez_list",
            "ez_str",
            "str_only",
            "input_msg",
            "tc",
            "fc_out",
            "rzning",
            "multi_rzning",
            "output_msg",
        ],
    )
    def test_responses_e2e(
        self,
        monkeypatch: MonkeyPatch,
        single_input: Union[str, NeMoGymResponseInput],
        _,
        mock_chat_completion: NeMoGymChatCompletion,
        expected_response: NeMoGymResponse,
    ):
        """
        Test entire pipeline from api endpoint -> final output:
        Response Create Params -> Response
        """
        server = self._setup_server(monkeypatch)
        server._converter.uses_reasoning_parser = True
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("responses_api_models.vllm_model.app.time", lambda: FIXED_TIME)

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input=single_input)

        monkeypatch.setattr(
            VLLMModel,
            "chat_completions",
            AsyncMock(return_value=mock_chat_completion),
        )

        response = client.post(
            "/v1/responses",
            json=responses_create_params.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        assert expected_response.model_dump() == response.json()

    @mark.parametrize(
        "single_input, expected_chat_completion_create_params, _, __",
        PARAMETERIZE_DATA,
        ids=[
            "ez_list",
            "ez_str",
            "str_only",
            "input_msg",
            "tc",
            "fc_out",
            "rzning",
            "multi_rzning",
            "output_msg",
        ],
    )
    def test_responses_to_chat_completion_create_params(
        self,
        monkeypatch: MonkeyPatch,
        single_input: Union[str, NeMoGymResponseInput],
        expected_chat_completion_create_params: NeMoGymChatCompletionCreateParamsNonStreaming,
        _,
        __,
    ):
        """
        Tests conversion from api endpoint -> internal request schema
        Response Params -> Chat Completion Params
        """
        server = self._setup_server(monkeypatch)
        app = server.setup_webserver()
        client = TestClient(app)

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input=single_input)

        captured_params: dict[str, Any] = {}

        # Returning this dummy response allows us to call /responses vs.
        # server._converter.responses_to_chat_completion_create_params() directly
        async def _mock_and_capture(self, request, create_params):
            captured_params["value"] = create_params
            return NeMoGymChatCompletion(
                id="chtcmpl-123",
                choices=[
                    NeMoGymChoice(
                        index=0,
                        finish_reason="stop",
                        message=NeMoGymChatCompletionMessage(role="assistant", content="some response"),
                    )
                ],
                created=123,
                model="mock-model",
                object="chat.completion",
            )

        monkeypatch.setattr(VLLMModel, "chat_completions", _mock_and_capture)

        response = client.post(
            "/v1/responses",
            json=responses_create_params.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        assert captured_params["value"] == expected_chat_completion_create_params

    def test_client_session_routing(self):
        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8081,
            base_url=["http://api.openai.com/v1", "http://api.openai.com/v2"],
            api_key="dummy_key",  # pragma: allowlist secret
            model="dummy_model",
            entrypoint="",
            name="",
            return_token_id_information=False,
            uses_reasoning_parser=False,
        )
        server = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        app = server.setup_webserver()

        assert len(server._clients) == 2

        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="",
                        tool_calls=[],
                    ),
                )
            ],
        )

        input_messages = [
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="Check my order status", type="input_text")],
                status="completed",
            ),
        ]
        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
        )

        mock_chat_completion_1 = mock_chat_completion.model_copy(deep=True)
        mock_chat_completion_1.choices[0].message.content = "1"
        mock_method_1 = AsyncMock(return_value=mock_chat_completion_1.model_dump())
        client_1 = MagicMock(spec=NeMoGymAsyncOpenAI)
        client_1.create_chat_completion = mock_method_1

        mock_chat_completion_2 = mock_chat_completion.model_copy(deep=True)
        mock_chat_completion_2.choices[0].message.content = "2"
        mock_method_2 = AsyncMock(return_value=mock_chat_completion_2.model_dump())
        client_2 = MagicMock(spec=NeMoGymAsyncOpenAI)
        client_2.create_chat_completion = mock_method_2

        server._clients = [client_1, client_2]

        # Record the backend selected for each new session.
        client_1 = TestClient(app)
        response_1_1 = client_1.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_1_1.status_code == 200
        data = response_1_1.json()
        routed_output_1 = data["output"][0]["content"][0]["text"]

        client_2 = TestClient(app)
        response_2_1 = client_2.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_2_1.status_code == 200
        data = response_2_1.json()
        routed_output_2 = data["output"][0]["content"][0]["text"]

        client_3 = TestClient(app)
        response_3_1 = client_3.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_3_1.status_code == 200
        data = response_3_1.json()
        routed_output_3 = data["output"][0]["content"][0]["text"]

        # TestClient preserves the session cookie, so every second request must
        # return through the same backend selected for that session's first request.
        response_1_2 = client_1.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_1_2.status_code == 200
        data = response_1_2.json()
        assert data["output"][0]["content"][0]["text"] == routed_output_1

        # We do this out of order as 1 -> 3 -> 2 instead of 1 -> 2 -> 3 to test any ordering effects.
        response_3_2 = client_3.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_3_2.status_code == 200
        data = response_3_2.json()
        assert data["output"][0]["content"][0]["text"] == routed_output_3

        response_2_2 = client_2.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_2_2.status_code == 200
        data = response_2_2.json()
        assert data["output"][0]["content"][0]["text"] == routed_output_2

    def test_responses_reasoning_parser(self, monkeypatch: MonkeyPatch):
        server = self._setup_server(monkeypatch)
        server.config.uses_reasoning_parser = True
        server._converter.uses_reasoning_parser = True

        app = server.setup_webserver()
        client = TestClient(app)

        # START: First turn
        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content=" hello hello",
                        tool_calls=[
                            NeMoGymChatCompletionMessageToolCall(
                                id="call_123",
                                function=NeMoGymFunction(
                                    name="get_order_status",
                                    arguments='{"order_id": "123"}',
                                ),
                                type="function",
                            ),
                            NeMoGymChatCompletionMessageToolCall(
                                id="call_234",
                                function=NeMoGymFunction(
                                    name="get_delivery_date",
                                    arguments='{"order_id": "234"}',
                                ),
                                type="function",
                            ),
                        ],
                        reasoning_content="Gathering order status and delivery info...",
                    ),
                )
            ],
        )

        input_messages = [
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="Check my order status", type="input_text")],
                status="completed",
            ),
            NeMoGymResponseReasoningItem(
                id="rs_123",
                status="completed",
                type="reasoning",
                summary=[
                    NeMoGymSummary(
                        type="summary_text",
                        text="First reasoning item",
                    )
                ],
            ),
            NeMoGymEasyInputMessage(
                type="message",
                role="assistant",
                content=[NeMoGymResponseInputText(text="Sure, one sec.", type="input_text")],
                status="completed",
            ),
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="cool", type="input_text")],
                status="completed",
            ),
            NeMoGymEasyInputMessage(
                type="message",
                role="assistant",
                content=[NeMoGymResponseInputText(text="I'm still checking", type="input_text")],
                status="completed",
            ),
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="ok", type="input_text")],
                status="completed",
            ),
        ]

        input_tools = [
            NeMoGymFunctionToolParam(
                name="get_order_status",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order",
                        },
                    },
                    "required": ["order_id"],
                },
                type="function",
                description="Get the current status for a given order",
                strict=True,
            ),
            NeMoGymFunctionToolParam(
                name="get_delivery_date",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order",
                        },
                    },
                    "required": ["order_id"],
                },
                type="function",
                description="Get the estimated delivery date for a given order",
                strict=True,
            ),
        ]

        expected_response = NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            object="response",
            tools=input_tools,
            created_at=FIXED_TIME,
            model="dummy_model",
            output=[
                NeMoGymResponseReasoningItem(
                    id="rs_123",
                    status="completed",
                    type="reasoning",
                    summary=[
                        NeMoGymSummary(
                            type="summary_text",
                            text="Gathering order status and delivery info...",
                        )
                    ],
                ),
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=" hello hello",
                            annotations=[],
                            logprobs=None,
                        )
                    ],
                ),
                NeMoGymResponseFunctionToolCall(
                    type="function_call",
                    name="get_order_status",
                    arguments='{"order_id": "123"}',
                    call_id="call_123",
                    status="completed",
                    id="call_123",
                ),
                NeMoGymResponseFunctionToolCall(
                    type="function_call",
                    name="get_delivery_date",
                    arguments='{"order_id": "234"}',
                    call_id="call_234",
                    status="completed",
                    id="call_234",
                ),
            ],
        )

        mock_method = AsyncMock(return_value=mock_chat_completion.model_dump())
        monkeypatch.setattr(
            server._clients[0].__class__,
            "create_chat_completion",
            mock_method,
        )

        monkeypatch.setattr("responses_api_models.vllm_model.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            tools=input_tools,
        )

        response = client.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        data = response.json()

        expected_dict = expected_response.model_dump()
        assert data == expected_dict

        expected_messages = [
            {"content": [{"text": "Check my order status", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "Sure, one sec.",
                "reasoning_content": "First reasoning item",
                "reasoning": "First reasoning item",
            },
            {"content": [{"text": "cool", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "I'm still checking",
            },
            {"content": [{"text": "ok", "type": "text"}], "role": "user"},
        ]
        actual_messages = mock_method.call_args.kwargs["messages"]
        assert expected_messages == actual_messages

        # START: Second turn
        input_messages = [
            *input_messages,
            *data["output"],
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="user", type="input_text")],
                status="completed",
            ),
        ]
        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            tools=input_tools,
        )

        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        # Test the None path ehre
                        content=None,
                        tool_calls=[],
                        reasoning_content="None content test",
                    ),
                )
            ],
        )
        mock_method = AsyncMock(return_value=mock_chat_completion.model_dump())
        monkeypatch.setattr(
            server._clients[0].__class__,
            "create_chat_completion",
            mock_method,
        )

        response = client.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        data = response.json()

        expected_response = NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            object="response",
            tools=input_tools,
            created_at=FIXED_TIME,
            model="dummy_model",
            output=[
                NeMoGymResponseReasoningItem(
                    id="rs_123",
                    status="completed",
                    type="reasoning",
                    summary=[
                        NeMoGymSummary(
                            type="summary_text",
                            text="None content test",
                        )
                    ],
                ),
            ],
        )
        expected_dict = expected_response.model_dump()
        assert data == expected_dict

        expected_messages = [
            {"content": [{"text": "Check my order status", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "Sure, one sec.",
                "reasoning_content": "First reasoning item",
                "reasoning": "First reasoning item",
            },
            {"content": [{"text": "cool", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "I'm still checking",
            },
            {"content": [{"text": "ok", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": " hello hello",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"arguments": '{"order_id": "123"}', "name": "get_order_status"},
                        "type": "function",
                    },
                    {
                        "id": "call_234",
                        "function": {"arguments": '{"order_id": "234"}', "name": "get_delivery_date"},
                        "type": "function",
                    },
                ],
                "reasoning_content": "Gathering order status and delivery info...",
                "reasoning": "Gathering order status and delivery info...",
            },
            {"content": [{"text": "user", "type": "text"}], "role": "user"},
        ]
        actual_messages = mock_method.call_args.kwargs["messages"]
        assert expected_messages == actual_messages

        # START: Third turn
        input_messages = [
            *input_messages,
            *data["output"],
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="user", type="input_text")],
                status="completed",
            ),
        ]
        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            tools=input_tools,
        )

        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        # Test the None path ehre
                        content="None reasoning test",
                        tool_calls=[],
                        reasoning_content=None,
                    ),
                )
            ],
        )
        mock_method = AsyncMock(return_value=mock_chat_completion.model_dump())
        monkeypatch.setattr(
            server._clients[0].__class__,
            "create_chat_completion",
            mock_method,
        )

        response = client.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        data = response.json()

        expected_response = NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            object="response",
            tools=input_tools,
            created_at=FIXED_TIME,
            model="dummy_model",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text="None reasoning test",
                            annotations=[],
                            logprobs=None,
                        )
                    ],
                ),
            ],
        )
        expected_dict = expected_response.model_dump()
        assert data == expected_dict

        expected_messages = [
            {"content": [{"text": "Check my order status", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "Sure, one sec.",
                "reasoning_content": "First reasoning item",
                "reasoning": "First reasoning item",
            },
            {"content": [{"text": "cool", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "I'm still checking",
            },
            {"content": [{"text": "ok", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": " hello hello",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"arguments": '{"order_id": "123"}', "name": "get_order_status"},
                        "type": "function",
                    },
                    {
                        "id": "call_234",
                        "function": {"arguments": '{"order_id": "234"}', "name": "get_delivery_date"},
                        "type": "function",
                    },
                ],
                "reasoning_content": "Gathering order status and delivery info...",
                "reasoning": "Gathering order status and delivery info...",
            },
            {"content": [{"text": "user", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "None content test",
                "reasoning": "None content test",
            },
            {"content": [{"text": "user", "type": "text"}], "role": "user"},
        ]
        actual_messages = mock_method.call_args.kwargs["messages"]
        assert expected_messages == actual_messages

    def test_responses_reasoning_parser_reasoning(self, monkeypatch: MonkeyPatch):
        # See the TODO wrt reasoning_content in vllm_model/app.py
        server = self._setup_server(monkeypatch)
        server.config.uses_reasoning_parser = True
        server._converter.uses_reasoning_parser = True

        app = server.setup_webserver()
        client = TestClient(app)

        # START: First turn
        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content=" hello hello",
                        tool_calls=[
                            NeMoGymChatCompletionMessageToolCall(
                                id="call_123",
                                function=NeMoGymFunction(
                                    name="get_order_status",
                                    arguments='{"order_id": "123"}',
                                ),
                                type="function",
                            ),
                            NeMoGymChatCompletionMessageToolCall(
                                id="call_234",
                                function=NeMoGymFunction(
                                    name="get_delivery_date",
                                    arguments='{"order_id": "234"}',
                                ),
                                type="function",
                            ),
                        ],
                        reasoning="Gathering order status and delivery info...",
                    ),
                )
            ],
        )

        input_messages = [
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="Check my order status", type="input_text")],
                status="completed",
            ),
            NeMoGymResponseReasoningItem(
                id="rs_123",
                status="completed",
                type="reasoning",
                summary=[
                    NeMoGymSummary(
                        type="summary_text",
                        text="First reasoning item",
                    )
                ],
            ),
            NeMoGymEasyInputMessage(
                type="message",
                role="assistant",
                content=[NeMoGymResponseInputText(text="Sure, one sec.", type="input_text")],
                status="completed",
            ),
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="cool", type="input_text")],
                status="completed",
            ),
            NeMoGymEasyInputMessage(
                type="message",
                role="assistant",
                content=[NeMoGymResponseInputText(text="I'm still checking", type="input_text")],
                status="completed",
            ),
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="ok", type="input_text")],
                status="completed",
            ),
        ]

        input_tools = [
            NeMoGymFunctionToolParam(
                name="get_order_status",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order",
                        },
                    },
                    "required": ["order_id"],
                },
                type="function",
                description="Get the current status for a given order",
                strict=True,
            ),
            NeMoGymFunctionToolParam(
                name="get_delivery_date",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order",
                        },
                    },
                    "required": ["order_id"],
                },
                type="function",
                description="Get the estimated delivery date for a given order",
                strict=True,
            ),
        ]

        expected_response = NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            object="response",
            tools=input_tools,
            created_at=FIXED_TIME,
            model="dummy_model",
            output=[
                NeMoGymResponseReasoningItem(
                    id="rs_123",
                    status="completed",
                    type="reasoning",
                    summary=[
                        NeMoGymSummary(
                            type="summary_text",
                            text="Gathering order status and delivery info...",
                        )
                    ],
                ),
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=" hello hello",
                            annotations=[],
                            logprobs=None,
                        )
                    ],
                ),
                NeMoGymResponseFunctionToolCall(
                    type="function_call",
                    name="get_order_status",
                    arguments='{"order_id": "123"}',
                    call_id="call_123",
                    status="completed",
                    id="call_123",
                ),
                NeMoGymResponseFunctionToolCall(
                    type="function_call",
                    name="get_delivery_date",
                    arguments='{"order_id": "234"}',
                    call_id="call_234",
                    status="completed",
                    id="call_234",
                ),
            ],
        )

        mock_method = AsyncMock(return_value=mock_chat_completion.model_dump())
        monkeypatch.setattr(
            server._clients[0].__class__,
            "create_chat_completion",
            mock_method,
        )

        monkeypatch.setattr("responses_api_models.vllm_model.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            tools=input_tools,
        )

        response = client.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        data = response.json()

        expected_dict = expected_response.model_dump()
        assert data == expected_dict

        expected_messages = [
            {"content": [{"text": "Check my order status", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "Sure, one sec.",
                "reasoning_content": "First reasoning item",
                "reasoning": "First reasoning item",
            },
            {"content": [{"text": "cool", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "I'm still checking",
            },
            {"content": [{"text": "ok", "type": "text"}], "role": "user"},
        ]
        actual_messages = mock_method.call_args.kwargs["messages"]
        assert expected_messages == actual_messages

        # START: Second turn
        input_messages = [
            *input_messages,
            *data["output"],
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="user", type="input_text")],
                status="completed",
            ),
        ]
        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            tools=input_tools,
        )

        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        # Test the None path ehre
                        content=None,
                        tool_calls=[],
                        reasoning="None content test",
                    ),
                )
            ],
        )
        mock_method = AsyncMock(return_value=mock_chat_completion.model_dump())
        monkeypatch.setattr(
            server._clients[0].__class__,
            "create_chat_completion",
            mock_method,
        )

        response = client.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        data = response.json()

        expected_response = NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            object="response",
            tools=input_tools,
            created_at=FIXED_TIME,
            model="dummy_model",
            output=[
                NeMoGymResponseReasoningItem(
                    id="rs_123",
                    status="completed",
                    type="reasoning",
                    summary=[
                        NeMoGymSummary(
                            type="summary_text",
                            text="None content test",
                        )
                    ],
                ),
            ],
        )
        expected_dict = expected_response.model_dump()
        assert data == expected_dict

        expected_messages = [
            {"content": [{"text": "Check my order status", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "Sure, one sec.",
                "reasoning_content": "First reasoning item",
                "reasoning": "First reasoning item",
            },
            {"content": [{"text": "cool", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "I'm still checking",
            },
            {"content": [{"text": "ok", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": " hello hello",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"arguments": '{"order_id": "123"}', "name": "get_order_status"},
                        "type": "function",
                    },
                    {
                        "id": "call_234",
                        "function": {"arguments": '{"order_id": "234"}', "name": "get_delivery_date"},
                        "type": "function",
                    },
                ],
                "reasoning_content": "Gathering order status and delivery info...",
                "reasoning": "Gathering order status and delivery info...",
            },
            {"content": [{"text": "user", "type": "text"}], "role": "user"},
        ]
        actual_messages = mock_method.call_args.kwargs["messages"]
        assert expected_messages == actual_messages

        # START: Third turn
        input_messages = [
            *input_messages,
            *data["output"],
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="user", type="input_text")],
                status="completed",
            ),
        ]
        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            tools=input_tools,
        )

        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        # Test the None path ehre
                        content="None reasoning test",
                        tool_calls=[],
                        reasoning=None,
                    ),
                )
            ],
        )
        mock_method = AsyncMock(return_value=mock_chat_completion.model_dump())
        monkeypatch.setattr(
            server._clients[0].__class__,
            "create_chat_completion",
            mock_method,
        )

        response = client.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        data = response.json()

        expected_response = NeMoGymResponse(
            **COMMON_RESPONSE_PARAMS,
            id="resp_123",
            object="response",
            tools=input_tools,
            created_at=FIXED_TIME,
            model="dummy_model",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text="None reasoning test",
                            annotations=[],
                            logprobs=None,
                        )
                    ],
                ),
            ],
        )
        expected_dict = expected_response.model_dump()
        assert data == expected_dict

        expected_messages = [
            {"content": [{"text": "Check my order status", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "Sure, one sec.",
                "reasoning_content": "First reasoning item",
                "reasoning": "First reasoning item",
            },
            {"content": [{"text": "cool", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "I'm still checking",
            },
            {"content": [{"text": "ok", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": " hello hello",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"arguments": '{"order_id": "123"}', "name": "get_order_status"},
                        "type": "function",
                    },
                    {
                        "id": "call_234",
                        "function": {"arguments": '{"order_id": "234"}', "name": "get_delivery_date"},
                        "type": "function",
                    },
                ],
                "reasoning_content": "Gathering order status and delivery info...",
                "reasoning": "Gathering order status and delivery info...",
            },
            {"content": [{"text": "user", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "None content test",
                "reasoning": "None content test",
            },
            {"content": [{"text": "user", "type": "text"}], "role": "user"},
        ]
        actual_messages = mock_method.call_args.kwargs["messages"]
        assert expected_messages == actual_messages

    def test_responses_sequential_reasoning_allowed_False(self, monkeypatch: MonkeyPatch):
        server = self._setup_server(monkeypatch)
        server.config.uses_reasoning_parser = True
        server.config.sequential_reasoning_allowed = False

        app = server.setup_webserver()
        client = TestClient(app)

        input_messages = [
            NeMoGymEasyInputMessage(
                type="message",
                role="user",
                content=[NeMoGymResponseInputText(text="Check my order status", type="input_text")],
                status="completed",
            ),
            NeMoGymResponseReasoningItem(
                id="rs_123",
                status="completed",
                type="reasoning",
                summary=[
                    NeMoGymSummary(
                        type="summary_text",
                        text="First reasoning item",
                    )
                ],
            ),
        ]

        expected_response = NeMoGymResponse(
            **(COMMON_RESPONSE_PARAMS | {"status": "incomplete"}),
            id="resp_123",
            object="response",
            tools=[],
            created_at=FIXED_TIME,
            model="dummy_model",
            output=[
                {
                    "content": [
                        {
                            "annotations": [],
                            "logprobs": None,
                            "text": "",
                            "type": "output_text",
                        },
                    ],
                    "id": "msg_123",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ],
            incomplete_details={"reason": "content_filter"},
        )

        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            tools=[],
        )

        monkeypatch.setattr("responses_api_models.vllm_model.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        response = client.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        data = response.json()

        expected_dict = expected_response.model_dump()
        assert data == expected_dict


class TestVLLMConverter:
    def setup_method(self, _):
        self.converter = VLLMConverter(return_token_id_information=False)

    def test_responses_input_types_EasyInputMessageParam(self) -> None:
        """
        Tests the conversion of ResponseCreateParams to ChatCompletionCreateParams
        """

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                # ----- Baseline -----
                NeMoGymEasyInputMessage(
                    content="my content",
                    role="user",
                    type="message",
                ),
                # ----- Ablate `content` -----
                NeMoGymEasyInputMessage(
                    content=[
                        NeMoGymResponseInputText(
                            type="input_text",
                            text="my content 1",
                        ),
                        NeMoGymResponseInputText(
                            type="input_text",
                            text="my content 2",
                        ),
                    ],
                    role="user",
                    type="message",
                ),
                # ----- Ablate `role` -----
                NeMoGymEasyInputMessage(
                    content=[NeMoGymResponseInputText(text="assistant content", type="input_text")],
                    role="assistant",
                    type="message",
                ),
                NeMoGymEasyInputMessage(
                    content=[NeMoGymResponseInputText(text="system content", type="input_text")],
                    role="system",
                    type="message",
                ),
                NeMoGymEasyInputMessage(
                    content=[NeMoGymResponseInputText(text="developer content", type="input_text")],
                    role="developer",
                    type="message",
                ),
                # ----- Ablate `type` -----
                NeMoGymEasyInputMessage(
                    content=[NeMoGymResponseInputText(text="user content", type="input_text")],
                    role="user",
                    # type (omitted)
                ),
            ],
        )

        expected_chat_completion_create_params = NeMoGymChatCompletionCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS,
            messages=[
                {
                    "role": "user",
                    "content": "my content",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "my content 1"},
                        {"type": "text", "text": "my content 2"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "assistant content",
                },
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "system content"}],
                },
                {
                    "role": "developer",
                    "content": [{"type": "text", "text": "developer content"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "user content"}],
                    # No type
                },
            ],
        )
        actual_chat_completion_create_params = self.converter.responses_to_chat_completion_create_params(
            responses_create_params
        )
        assert expected_chat_completion_create_params.messages == actual_chat_completion_create_params.messages

    @mark.parametrize(
        "_, __, mock_chat_completion, expected_response",
        PARAMETERIZE_DATA,
        ids=[
            "ez_list",
            "ez_str",
            "str_only",
            "input_msg",
            "tc",
            "fc_out",
            "rzning",
            "multi_rzning",
            "output_msg",
        ],
    )
    def test_chat_completion_to_responses_postprocessing(
        self,
        monkeypatch: MonkeyPatch,
        _,
        __,
        mock_chat_completion: NeMoGymChatCompletion,
        expected_response: NeMoGymResponse,
    ):
        """
        Test internal postprocessing logic
        ChatCompletion output -> Response output
        """

        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        monkeypatch.setattr("responses_api_models.vllm_model.app.time", lambda: FIXED_TIME)

        choice = mock_chat_completion.choices[0]

        processed_output = self.converter.postprocess_chat_response(choice)

        assert processed_output == expected_response.output

    def test_extract_reasoning_from_content(self):
        # Single reasoning block
        content_single = "This is some main content.<think>Here is the reasoning.</think>More content."
        reasoning_single, main_content_single = self.converter._extract_reasoning_from_content(content_single)
        assert reasoning_single == ["Here is the reasoning."]
        assert main_content_single == "This is some main content.More content."

        # Multiple reasoning blocks
        content_multiple = "First part.<think>Thought 1.</think>Second part.<think>Thought 2.</think>Final part."
        reasoning_multiple, main_content_multiple = self.converter._extract_reasoning_from_content(content_multiple)
        assert reasoning_multiple == ["Thought 1.", "Thought 2."]
        assert main_content_multiple == "First part.Second part.Final part."

        # No reasoning
        content_none = "Just plain content here."
        reasoning_none, main_content_none = self.converter._extract_reasoning_from_content(content_none)
        assert reasoning_none == []
        assert main_content_none == "Just plain content here."

    def test_postprocess_chat_response_multiple_reasoning_items(self, monkeypatch: MonkeyPatch):
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("responses_api_models.vllm_model.app.time", lambda: FIXED_TIME)

        raw_model_response = (
            "<think>I need to check the user's order ID.</think>"
            "<think>The order ID is 12345.</think>"
            "Your order has been shipped."
        )

        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content=raw_model_response,
                    ),
                )
            ],
        )

        expected_output = [
            NeMoGymResponseReasoningItem(
                id=f"rs_{FIXED_UUID}",
                type="reasoning",
                summary=[
                    NeMoGymSummary(text="I need to check the user's order ID.", type="summary_text"),
                    NeMoGymSummary(text="The order ID is 12345.", type="summary_text"),
                ],
                status="completed",
            ),
            NeMoGymResponseOutputMessage(
                id=f"msg_{FIXED_UUID}",
                role="assistant",
                content=[
                    NeMoGymResponseOutputText(
                        type="output_text",
                        text="Your order has been shipped.",
                        annotations=[],
                    )
                ],
                status="completed",
                type="message",
            ),
        ]

        choice = mock_chat_completion.choices[0]
        actual_output = self.converter.postprocess_chat_response(
            choice,
        )

        assert actual_output == expected_output

    @mark.parametrize(
        "single_input, expected_chat_completion_create_params, _, __",
        PARAMETERIZE_DATA,
        ids=[
            "ez_list",
            "ez_str",
            "str_only",
            "input_msg",
            "tc",
            "fc_out",
            "rzning",
            "multi_rzning",
            "output_msg",
        ],
    )
    def test_responses_to_chat_completion_create_params_converter(
        self,
        single_input: Union[str, NeMoGymResponseInput],
        expected_chat_completion_create_params: NeMoGymChatCompletionCreateParamsNonStreaming,
        _,
        __,
    ):
        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input=single_input)

        actual_chat_completion_create_params = self.converter.responses_to_chat_completion_create_params(
            responses_create_params
        )

        assert actual_chat_completion_create_params.messages == expected_chat_completion_create_params.messages

    def test_round_trip_chat_completions(self) -> None:
        message = NeMoGymChatCompletionMessage(
            content="<think>I'm thinking</think>I'm chatting!",
            role="assistant",
            tool_calls=[
                NeMoGymChatCompletionMessageToolCall(
                    id="tool call 1",
                    function=NeMoGymFunction(name="get_weather", arguments='{"city_name": "new york"}'),
                    type="function",
                ),
                NeMoGymChatCompletionMessageToolCall(
                    id="tool call 2",
                    function=NeMoGymFunction(name="get_weather", arguments='{"city_name": "boston"}'),
                    type="function",
                ),
            ],
        )
        actual_response_output_items = self.converter.postprocess_chat_response(
            choice=NeMoGymChoice(
                finish_reason="tool_calls",
                index=0,
                message=message,
            )
        )
        assert len(actual_response_output_items) == 4

        chat_completion_create_params = self.converter.responses_to_chat_completion_create_params(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    NeMoGymEasyInputMessage(
                        content="system",
                        role="system",
                    ),
                    NeMoGymEasyInputMessage(
                        content="hello!",
                        role="user",
                    ),
                    *actual_response_output_items,
                ],
            )
        )
        actual_messages = chat_completion_create_params.messages

        expected_messages = [
            NeMoGymChatCompletionSystemMessageParam(
                content="system",
                role="system",
            ),
            NeMoGymChatCompletionUserMessageParam(
                content="hello!",
                role="user",
            ),
            NeMoGymChatCompletionAssistantMessageParam(
                role="assistant",
                content="<think>I'm thinking</think>I'm chatting!",
                tool_calls=[
                    NeMoGymChatCompletionMessageToolCallParam(
                        id="tool call 1",
                        function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                            name="get_weather", arguments='{"city_name": "new york"}'
                        ),
                        type="function",
                    ),
                    NeMoGymChatCompletionMessageToolCallParam(
                        id="tool call 2",
                        function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                            name="get_weather", arguments='{"city_name": "boston"}'
                        ),
                        type="function",
                    ),
                ],
            ),
        ]
        assert expected_messages == actual_messages

        test_data_fpath = f"{PARENT_DIR}/responses_api_models/vllm_model/tests/round_trip_test_data.json"
        with open(test_data_fpath) as f:
            test_data = json.load(f)

        chat_completion_create_params = self.converter.responses_to_chat_completion_create_params(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    NeMoGymEasyInputMessage(
                        content="system",
                        role="system",
                    ),
                    NeMoGymEasyInputMessage(
                        content="hello!",
                        role="user",
                    ),
                    *test_data["input"]["output"],
                ],
            )
        )

        expected_output = test_data["expected_output"]
        assert expected_output == chat_completion_create_params.model_dump()

    def test_round_trip_chat_completions_return_token_id_information(self) -> None:
        converter = VLLMConverter(return_token_id_information=True)

        message = NeMoGymChatCompletionMessageForTraining(
            content="<think>I'm thinking</think>I'm chatting!",
            role="assistant",
            tool_calls=[
                NeMoGymChatCompletionMessageToolCall(
                    id="tool call 1",
                    function=NeMoGymFunction(name="get_weather", arguments='{"city_name": "new york"}'),
                    type="function",
                ),
                NeMoGymChatCompletionMessageToolCall(
                    id="tool call 2",
                    function=NeMoGymFunction(name="get_weather", arguments='{"city_name": "boston"}'),
                    type="function",
                ),
            ],
            prompt_token_ids=[1, 2, 3],
            generation_token_ids=[4, 5, 6],
            generation_log_probs=[7.0, 8.0, 9.0],
        )
        actual_response_output_items = converter.postprocess_chat_response(
            choice=NeMoGymChoice(
                finish_reason="tool_calls",
                index=0,
                message=message,
            )
        )
        assert len(actual_response_output_items) == 4

        chat_completion_create_params = converter.responses_to_chat_completion_create_params(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    NeMoGymEasyInputMessage(
                        content="system",
                        role="system",
                    ),
                    NeMoGymEasyInputMessage(
                        content="hello!",
                        role="user",
                    ),
                    *actual_response_output_items,
                ],
            )
        )
        actual_messages = chat_completion_create_params.messages

        expected_messages = [
            NeMoGymChatCompletionSystemMessageParam(
                content="system",
                role="system",
            ),
            NeMoGymChatCompletionUserMessageParam(
                content="hello!",
                role="user",
            ),
            NeMoGymChatCompletionAssistantMessageForTrainingParam(
                role="assistant",
                content="<think>I'm thinking</think>I'm chatting!",
                tool_calls=[
                    NeMoGymChatCompletionMessageToolCallParam(
                        id="tool call 1",
                        function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                            name="get_weather", arguments='{"city_name": "new york"}'
                        ),
                        type="function",
                    ),
                    NeMoGymChatCompletionMessageToolCallParam(
                        id="tool call 2",
                        function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                            name="get_weather", arguments='{"city_name": "boston"}'
                        ),
                        type="function",
                    ),
                ],
                prompt_token_ids=[1, 2, 3],
                generation_token_ids=[4, 5, 6],
                generation_log_probs=[7.0, 8.0, 9.0],
            ),
        ]
        assert expected_messages == actual_messages

        test_data_fpath = f"{PARENT_DIR}/responses_api_models/vllm_model/tests/round_trip_test_data.json"
        with open(test_data_fpath) as f:
            test_data = json.load(f)

        chat_completion_create_params = converter.responses_to_chat_completion_create_params(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    NeMoGymEasyInputMessage(
                        content="system",
                        role="system",
                    ),
                    NeMoGymEasyInputMessage(
                        content="hello!",
                        role="user",
                    ),
                    *test_data["input"]["output"],
                ],
            )
        )

        expected_output = test_data["expected_output_return_token_id_information"]
        assert expected_output == chat_completion_create_params.model_dump()

    def test_whitespace_round_trip_chat_completions(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        message = NeMoGymChatCompletionMessage(
            content="<think> \n \n I'm thinking \n \n </think> \n \n I'm chatting! \n \n ",
            role="assistant",
            tool_calls=[
                NeMoGymChatCompletionMessageToolCall(
                    id="tool call 1",
                    function=NeMoGymFunction(name="get_weather", arguments='{"city_name": "new york"}'),
                    type="function",
                ),
                NeMoGymChatCompletionMessageToolCall(
                    id="tool call 2",
                    function=NeMoGymFunction(name="get_weather", arguments='{"city_name": "boston"}'),
                    type="function",
                ),
            ],
        )
        actual_response_output_items = self.converter.postprocess_chat_response(
            choice=NeMoGymChoice(
                finish_reason="tool_calls",
                index=0,
                message=message,
            )
        )
        expected_response_output_items = [
            NeMoGymResponseReasoningItem(
                id="rs_123",
                summary=[NeMoGymSummary(text=" \n \n I'm thinking \n \n ", type="summary_text")],
                type="reasoning",
                encrypted_content=None,
            ),
            NeMoGymResponseOutputMessage(
                id="msg_123",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[], text=" \n \n I'm chatting! \n \n ", type="output_text", logprobs=None
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
            NeMoGymResponseFunctionToolCall(
                arguments='{"city_name": "new york"}',
                call_id="tool call 1",
                name="get_weather",
                type="function_call",
                id="tool call 1",
                status="completed",
            ),
            NeMoGymResponseFunctionToolCall(
                arguments='{"city_name": "boston"}',
                call_id="tool call 2",
                name="get_weather",
                type="function_call",
                id="tool call 2",
                status="completed",
            ),
        ]
        assert expected_response_output_items == actual_response_output_items

        chat_completion_create_params = self.converter.responses_to_chat_completion_create_params(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    NeMoGymEasyInputMessage(
                        content=" \n \n system \n \n ",
                        role="system",
                    ),
                    NeMoGymEasyInputMessage(
                        content=" \n \n hello! \n \n ",
                        role="user",
                    ),
                    *actual_response_output_items,
                ],
            )
        )
        actual_messages = chat_completion_create_params.messages

        expected_messages = [
            NeMoGymChatCompletionSystemMessageParam(
                content=" \n \n system \n \n ",
                role="system",
            ),
            NeMoGymChatCompletionUserMessageParam(
                content=" \n \n hello! \n \n ",
                role="user",
            ),
            NeMoGymChatCompletionAssistantMessageParam(
                role="assistant",
                content="<think> \n \n I'm thinking \n \n </think> \n \n I'm chatting! \n \n ",
                tool_calls=[
                    NeMoGymChatCompletionMessageToolCallParam(
                        id="tool call 1",
                        function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                            name="get_weather", arguments='{"city_name": "new york"}'
                        ),
                        type="function",
                    ),
                    NeMoGymChatCompletionMessageToolCallParam(
                        id="tool call 2",
                        function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                            name="get_weather", arguments='{"city_name": "boston"}'
                        ),
                        type="function",
                    ),
                ],
            ),
        ]
        assert expected_messages == actual_messages

    def test_metadata_chat_template_kwargs_override(self, monkeypatch: MonkeyPatch):
        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8081,
            base_url="http://api.openai.com/v1",
            api_key="dummy_key",  # pragma: allowlist secret
            model="dummy_model",
            entrypoint="",
            name="",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            chat_template_kwargs={"enable_thinking": True, "some_other_param": "value1"},
        )
        server = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        app = server.setup_webserver()

        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="response",
                        tool_calls=[],
                    ),
                )
            ],
        )

        captured_kwargs = {}

        async def mock_create_chat_completion(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_chat_completion.model_dump()

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        server._clients = [mock_client]

        request_body_with_override = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    type="message",
                    role="user",
                    content="hello",
                )
            ],
            metadata={"chat_template_kwargs": json.dumps({"enable_thinking": False})},
        )

        client = TestClient(app)
        response = client.post(
            "/v1/responses",
            json=request_body_with_override.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        assert "chat_template_kwargs" in captured_kwargs
        assert captured_kwargs["chat_template_kwargs"]["enable_thinking"] is False
        assert captured_kwargs["chat_template_kwargs"]["some_other_param"] == "value1"

        captured_kwargs.clear()
        request_body_no_override = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    type="message",
                    role="user",
                    content="hello",
                )
            ],
        )

        response = client.post(
            "/v1/responses",
            json=request_body_no_override.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        assert "chat_template_kwargs" in captured_kwargs
        assert captured_kwargs["chat_template_kwargs"]["enable_thinking"] is True
        assert captured_kwargs["chat_template_kwargs"]["some_other_param"] == "value1"

        captured_kwargs.clear()
        request_body_null_metadata = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    type="message",
                    role="user",
                    content="hello",
                )
            ],
            metadata=None,
        )

        response = client.post(
            "/v1/responses",
            json=request_body_null_metadata.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        assert "chat_template_kwargs" in captured_kwargs
        assert captured_kwargs["chat_template_kwargs"]["enable_thinking"] is True
        assert captured_kwargs["chat_template_kwargs"]["some_other_param"] == "value1"

        captured_kwargs.clear()
        request_body_multi_override = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    type="message",
                    role="user",
                    content="hello",
                )
            ],
            metadata={
                "chat_template_kwargs": json.dumps(
                    {"enable_thinking": False, "some_other_param": "value2", "new_param": "new"}
                )
            },
        )

        response = client.post(
            "/v1/responses",
            json=request_body_multi_override.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        assert captured_kwargs["chat_template_kwargs"]["enable_thinking"] is False
        assert captured_kwargs["chat_template_kwargs"]["some_other_param"] == "value2"
        assert captured_kwargs["chat_template_kwargs"]["new_param"] == "new"

    def test_metadata_extra_body_override(self, monkeypatch: MonkeyPatch):
        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8081,
            base_url="http://api.openai.com/v1",
            api_key="dummy_key",  # pragma: allowlist secret
            model="dummy_model",
            entrypoint="",
            name="",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            extra_body={"guided_json": '{"type": "object"}', "min_tokens": 10},
        )
        server = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        app = server.setup_webserver()

        mock_chat_completion = NeMoGymChatCompletion(
            id="chtcmpl",
            object="chat.completion",
            created=FIXED_TIME,
            model="dummy_model",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="response",
                        tool_calls=[],
                    ),
                )
            ],
        )

        captured_kwargs = {}

        async def mock_create_chat_completion(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_chat_completion.model_dump()

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        server._clients = [mock_client]

        request_body_with_override = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    type="message",
                    role="user",
                    content="hello",
                )
            ],
            metadata={"extra_body": json.dumps({"min_tokens": 20, "new_param": "value"})},
        )

        client = TestClient(app)
        response = client.post(
            "/v1/responses",
            json=request_body_with_override.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200

        assert captured_kwargs["guided_json"] == '{"type": "object"}'
        assert captured_kwargs["min_tokens"] == 20
        assert captured_kwargs["new_param"] == "value"


# ──────────────────────────────────────────────────────────────────────────────
# Assistant reasoning history preprocessing
# ──────────────────────────────────────────────────────────────────────────────


def _make_reasoning_history_model(*, preserve_content: bool) -> VLLMModel:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vllm_model",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy-model",
        return_token_id_information=False,
        uses_reasoning_parser=True,
        uses_interleaved_reasoning=True,
        preserve_reasoning_in_assistant_content=preserve_content,
    )
    return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient))


class TestAssistantReasoningHistoryPreprocess:
    @staticmethod
    def _body(content: Any) -> dict[str, Any]:
        return {
            "model": "caller-model",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "assistant", "content": content},
                {"role": "user", "content": "next"},
            ],
        }

    def test_default_still_splits_think_history(self) -> None:
        model = _make_reasoning_history_model(preserve_content=False)
        result = model._preprocess_chat_completion_create_params(
            MagicMock(), self._body("<think>reason</think>\n## Action:\nact")
        )

        assistant = result["messages"][1]
        assert assistant["content"] == "\n## Action:\nact"
        assert assistant["reasoning_content"] == "reason"
        assert assistant["reasoning"] == "reason"

    def test_preserve_mode_keeps_string_history_byte_for_byte(self) -> None:
        model = _make_reasoning_history_model(preserve_content=True)
        original = "<think>reason</think>\n## Action:\nact"
        result = model._preprocess_chat_completion_create_params(MagicMock(), self._body(original))

        assistant = result["messages"][1]
        assert assistant == {"role": "assistant", "content": original}

    def test_preserve_mode_keeps_list_history_byte_for_byte(self) -> None:
        model = _make_reasoning_history_model(preserve_content=True)
        original = [{"type": "text", "text": "<think>reason</think>\n## Action:\nact"}]
        result = model._preprocess_chat_completion_create_params(MagicMock(), self._body(original))

        assistant = result["messages"][1]
        assert assistant == {"role": "assistant", "content": original}


# ──────────────────────────────────────────────────────────────────────────────
# Audio sidechannel splice (metadata.audio_data → user-message content block)
#
# Lets audio benchmarks like librispeech_pc carry audio data-URIs through the
# Responses API even though `ResponseInputContentParam` has no audio variant.
# These tests exercise `_preprocess_chat_completion_create_params` directly
# with synthetic body_dicts so they don't need a running vLLM endpoint.
# Note that the spliced *content block* is still typed ``audio_url`` because
# that's vLLM's wire format — only the *metadata key* on the JSONL row is
# named ``audio_data``.
# ──────────────────────────────────────────────────────────────────────────────


_AUDIO_DATA = "data:audio/wav;base64,QUFB"  # placeholder bytes


def _make_minimal_audio_model() -> VLLMModel:
    """A VLLMModel instance with the minimum config needed to hit the splice path."""
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vllm_model",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy-model",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
    )
    return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient))


class TestAudioDataSplice:
    def test_no_metadata_passthrough(self) -> None:
        """No audio_data in metadata → user message content stays a plain string."""
        model = _make_minimal_audio_model()
        body = {
            "model": "dummy-model",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)
        assert result["messages"][1]["content"] == "hello"

    def test_splice_into_string_user_content(self) -> None:
        model = _make_minimal_audio_model()
        body = {
            "model": "dummy-model",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Transcribe please."},
            ],
            "metadata": {"audio_data": _AUDIO_DATA},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        # metadata cleared because the only key was consumed
        assert "metadata" not in result

        user_content = result["messages"][1]["content"]
        assert isinstance(user_content, list)
        # Audio block must come BEFORE the text part (some audio models care).
        assert user_content[0] == {"type": "audio_url", "audio_url": {"url": _AUDIO_DATA}}
        assert user_content[1] == {"type": "text", "text": "Transcribe please."}

    def test_splice_into_list_user_content(self) -> None:
        model = _make_minimal_audio_model()
        body = {
            "model": "dummy-model",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Transcribe please."}],
                }
            ],
            "metadata": {"audio_data": _AUDIO_DATA, "other": "keep"},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        # audio_data stripped, but the "other" key is preserved
        assert result["metadata"] == {"other": "keep"}
        assert result["messages"][0]["content"][0]["type"] == "audio_url"
        assert result["messages"][0]["content"][1]["type"] == "text"

    def test_splice_targets_most_recent_user(self) -> None:
        """Multi-turn: audio attaches to the LATEST user message."""
        model = _make_minimal_audio_model()
        body = {
            "model": "dummy-model",
            "messages": [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second turn"},
            ],
            "metadata": {"audio_data": _AUDIO_DATA},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        assert result["messages"][0]["content"] == "first turn"
        assert isinstance(result["messages"][2]["content"], list)
        assert result["messages"][2]["content"][0]["type"] == "audio_url"

    def test_no_user_message_creates_one(self) -> None:
        model = _make_minimal_audio_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "system", "content": "sys"}],
            "metadata": {"audio_data": _AUDIO_DATA},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        assert len(result["messages"]) == 2
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"][0]["type"] == "audio_url"

    def test_empty_audio_data_is_noop(self) -> None:
        """Falsy audio_data string skips the splice (and leaves metadata untouched)."""
        model = _make_minimal_audio_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"audio_data": ""},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)
        assert result["messages"][0]["content"] == "hi"


# ──────────────────────────────────────────────────────────────────────────────
# Audio sidechannel: file-path mode (metadata.audio_path)
#
# Path-mode is the size-friendly alternative to the data-URI form: the JSONL
# only carries a path, and the URL is materialized at request time per
# ``audio_path_mode``. Tests below exercise the splice with a real on-disk
# WAV-shaped file so the data-URI branch has bytes to encode.
# ──────────────────────────────────────────────────────────────────────────────

import base64 as _b64  # noqa: E402  (kept local to the audio_path tests)


_AUDIO_BYTES = b"RIFF" + b"\x00" * 8 + b"WAVEfmt "  # not a valid WAV — splice doesn't decode


def _make_audio_path_model(audio_root: str | None = None) -> VLLMModel:
    """Like _make_minimal_audio_model but lets tests pin audio_root."""
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vllm_model",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy-model",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
        audio_root=audio_root,
    )
    return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient))


class TestAudioPathSplice:
    def test_absolute_path_encodes_to_data_uri(self, tmp_path) -> None:
        wav = tmp_path / "clip.wav"
        wav.write_bytes(_AUDIO_BYTES)

        model = _make_audio_path_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Transcribe."}],
            "metadata": {"audio_path": str(wav)},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        # audio_path consumed; metadata empty → dropped.
        assert "metadata" not in result

        audio_block = result["messages"][0]["content"][0]
        assert audio_block["type"] == "audio_url"
        url = audio_block["audio_url"]["url"]
        prefix = "data:audio/wav;base64,"
        assert url.startswith(prefix)
        assert _b64.b64decode(url[len(prefix) :]) == _AUDIO_BYTES

    def test_relative_path_resolves_against_audio_root(self, tmp_path) -> None:
        (tmp_path / "sub").mkdir()
        wav = tmp_path / "sub" / "clip.flac"
        wav.write_bytes(_AUDIO_BYTES)

        model = _make_audio_path_model(audio_root=str(tmp_path))
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Transcribe."}],
            "metadata": {"audio_path": "sub/clip.flac"},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        url = result["messages"][0]["content"][0]["audio_url"]["url"]
        # MIME picked from extension, not hard-coded to wav.
        assert url.startswith("data:audio/flac;base64,")

    def test_relative_path_without_audio_root_raises(self) -> None:
        model = _make_audio_path_model(audio_root=None)
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Transcribe."}],
            "metadata": {"audio_path": "sub/clip.wav"},
        }
        try:
            model._preprocess_chat_completion_create_params(MagicMock(), body)
        except ValueError as e:
            assert "audio_root" in str(e)
        else:
            raise AssertionError("expected ValueError for relative path without audio_root")

    def test_missing_file_raises(self, tmp_path) -> None:
        model = _make_audio_path_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Transcribe."}],
            "metadata": {"audio_path": str(tmp_path / "does_not_exist.wav")},
        }
        try:
            model._preprocess_chat_completion_create_params(MagicMock(), body)
        except FileNotFoundError as e:
            assert "does_not_exist.wav" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError for missing audio file")

    def test_unknown_extension_raises(self, tmp_path) -> None:
        weird = tmp_path / "clip.xyz"
        weird.write_bytes(_AUDIO_BYTES)

        model = _make_audio_path_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Transcribe."}],
            "metadata": {"audio_path": str(weird)},
        }
        try:
            model._preprocess_chat_completion_create_params(MagicMock(), body)
        except ValueError as e:
            assert ".xyz" in str(e)
        else:
            raise AssertionError("expected ValueError for unknown audio extension")

    def test_audio_paths_list_emits_one_block_per_clip_in_order(self, tmp_path) -> None:
        # Mirrors Skills' ``audios`` multi-clip schema: each entry becomes a
        # separate audio block, all placed before the text in order.
        a = tmp_path / "a.wav"
        b = tmp_path / "b.flac"
        a.write_bytes(b"AAAA")
        b.write_bytes(b"BBBB")

        model = _make_audio_path_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Compare these."}],
            "metadata": {"audio_paths": [str(a), str(b)]},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        content = result["messages"][0]["content"]
        assert len(content) == 3  # two audio + one text
        assert content[0]["type"] == "audio_url"
        assert content[0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
        assert _b64.b64decode(content[0]["audio_url"]["url"].split(",", 1)[1]) == b"AAAA"
        assert content[1]["type"] == "audio_url"
        assert content[1]["audio_url"]["url"].startswith("data:audio/flac;base64,")
        assert _b64.b64decode(content[1]["audio_url"]["url"].split(",", 1)[1]) == b"BBBB"
        assert content[2] == {"type": "text", "text": "Compare these."}

    def test_audio_paths_must_be_a_list(self, tmp_path) -> None:
        model = _make_audio_path_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "x"}],
            "metadata": {"audio_paths": str(tmp_path / "a.wav")},  # string, not list
        }
        try:
            model._preprocess_chat_completion_create_params(MagicMock(), body)
        except ValueError as e:
            assert "must be a list" in str(e)
        else:
            raise AssertionError("expected ValueError when audio_paths is not a list")

    def test_audio_keys_mutually_exclusive(self, tmp_path) -> None:
        wav = tmp_path / "clip.wav"
        wav.write_bytes(_AUDIO_BYTES)

        model = _make_audio_path_model()
        # Any pairing of the three keys should raise — we spot-check two
        # combinations rather than the full three pairs.
        for metadata in (
            {"audio_data": _AUDIO_DATA, "audio_path": str(wav)},
            {"audio_path": str(wav), "audio_paths": [str(wav)]},
        ):
            body = {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Transcribe."}],
                "metadata": metadata,
            }
            try:
                model._preprocess_chat_completion_create_params(MagicMock(), body)
            except ValueError as e:
                assert "mutually exclusive" in str(e)
            else:
                raise AssertionError(f"expected ValueError for metadata={metadata}")


# ──────────────────────────────────────────────────────────────────────────────
# /v1/completions backend tests
#
# Exercises VLLMModel when ``use_completions_api`` is True: the chat
# completions handler talks to vLLM's /v1/completions instead of
# /v1/chat/completions, and a synthesized chat-completion shape is returned to
# the caller. Covers rendering rules, hard-rejects, and inline/fallback
# token-ID extraction.
# ──────────────────────────────────────────────────────────────────────────────


def _make_completions_backend_model(
    *,
    return_token_id_information: bool = False,
    extra_body: dict | None = None,
) -> VLLMModel:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vllm_completions",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="base-model",
        return_token_id_information=return_token_id_information,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
        use_completions_api=True,
        extra_body=extra_body,
    )
    return VLLMModel(
        config=config,
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )


class TestCompletionsBackendRawRender:
    def test_single_user_message_string_content(self) -> None:
        model = _make_completions_backend_model()
        prompt = model._render_messages_to_prompt([{"role": "user", "content": "Hello world"}])
        assert prompt == "Hello world"

    def test_single_user_message_text_block_list(self) -> None:
        model = _make_completions_backend_model()
        prompt = model._render_messages_to_prompt(
            [{"role": "user", "content": [{"type": "text", "text": "Hello world"}]}]
        )
        assert prompt == "Hello world"

    def test_input_text_block_type_accepted(self) -> None:
        # VLLMConverter._format_message rewrites input_text → text, but be
        # defensive: accept the input_text variant directly too.
        model = _make_completions_backend_model()
        prompt = model._render_messages_to_prompt(
            [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]
        )
        assert prompt == "Hi"

    def test_system_then_user_concatenated_with_double_newline(self) -> None:
        model = _make_completions_backend_model()
        prompt = model._render_messages_to_prompt(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        )
        assert prompt == "You are helpful.\n\nWhat is 2+2?"

    def test_assistant_message_rejected(self) -> None:
        model = _make_completions_backend_model()
        try:
            model._render_messages_to_prompt(
                [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": "y"},
                ]
            )
        except ValueError as e:
            assert "system, user" in str(e)
        else:
            raise AssertionError("expected ValueError for assistant message")

    def test_three_messages_rejected(self) -> None:
        model = _make_completions_backend_model()
        try:
            model._render_messages_to_prompt(
                [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u1"},
                    {"role": "user", "content": "u2"},
                ]
            )
        except ValueError as e:
            assert "at most" in str(e)
        else:
            raise AssertionError("expected ValueError for >2 messages")

    def test_lone_system_message_rejected(self) -> None:
        model = _make_completions_backend_model()
        try:
            model._render_messages_to_prompt([{"role": "system", "content": "s"}])
        except ValueError as e:
            assert "user message" in str(e)
        else:
            raise AssertionError("expected ValueError for lone system message")

    def test_image_block_rejected(self) -> None:
        model = _make_completions_backend_model()
        try:
            model._render_messages_to_prompt(
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
            )
        except ValueError as e:
            assert "text content blocks" in str(e)
        else:
            raise AssertionError("expected ValueError for image block")


class TestCompletionsBackendHardRejects:
    def test_tools_rejected(self) -> None:
        import asyncio

        model = _make_completions_backend_model()
        body = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object"}},
                }
            ],
        )
        try:
            asyncio.run(model._chat_completions_via_completions_api(MagicMock(), body))
        except ValueError as e:
            assert "tools are not supported" in str(e)
        else:
            raise AssertionError("expected ValueError when tools are set")

    def test_audio_metadata_rejected(self) -> None:
        import asyncio

        model = _make_completions_backend_model()
        body = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[{"role": "user", "content": "hi"}],
            metadata={"audio_data": "data:audio/wav;base64,QUFB"},
        )
        try:
            asyncio.run(model._chat_completions_via_completions_api(MagicMock(), body))
        except ValueError as e:
            assert "audio metadata" in str(e)
        else:
            raise AssertionError("expected ValueError when audio metadata is set")


class TestCompletionsBackendBodyTranslation:
    def test_basic_fields_passthrough(self) -> None:
        model = _make_completions_backend_model()
        out = model._build_completion_body_from_chat_body(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
                "temperature": 0.7,
                "top_p": 0.9,
                "seed": 42,
                "stop": ["</s>"],
            },
            prompt="hi",
        )
        assert out["model"] == "base-model"
        assert out["prompt"] == "hi"
        assert out["max_tokens"] == 64
        assert out["temperature"] == 0.7
        assert out["top_p"] == 0.9
        assert out["seed"] == 42
        assert out["stop"] == ["</s>"]
        # No token-id machinery requested.
        assert "logprobs" not in out
        assert "return_token_ids" not in out
        assert "return_tokens_as_token_ids" not in out
        assert "prompt_logprobs" not in out

    def test_max_completion_tokens_aliased(self) -> None:
        model = _make_completions_backend_model()
        out = model._build_completion_body_from_chat_body({"max_completion_tokens": 32}, prompt="x")
        assert out["max_tokens"] == 32

    def test_top_logprobs_maps_to_logprobs_int(self) -> None:
        model = _make_completions_backend_model()
        out = model._build_completion_body_from_chat_body({"top_logprobs": 5}, prompt="x")
        assert out["logprobs"] == 5

    def test_response_format_passthrough(self) -> None:
        model = _make_completions_backend_model()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
            },
        }
        out = model._build_completion_body_from_chat_body(
            {"response_format": response_format},
            prompt="x",
        )
        assert out["response_format"] == response_format

    def test_extra_body_does_not_override_request_fields(self) -> None:
        model = _make_completions_backend_model(extra_body={"top_p": 0.1, "min_p": 0.05})
        out = model._build_completion_body_from_chat_body(
            {"top_p": 0.95},
            prompt="x",
        )
        # Request-level top_p wins; min_p only present via extra_body.
        assert out["top_p"] == 0.95
        assert out["min_p"] == 0.05

    def test_token_id_information_sets_logprobs_and_extra_flags(self) -> None:
        model = _make_completions_backend_model(return_token_id_information=True)
        out = model._build_completion_body_from_chat_body({}, prompt="x")
        assert out["logprobs"] == 0
        assert out["return_token_ids"] is True
        assert out["return_tokens_as_token_ids"] is True
        assert "prompt_logprobs" not in out


class TestCompletionsBackendResponseTranslation:
    def test_text_lifted_into_assistant_message(self) -> None:
        model = _make_completions_backend_model()
        chat_completion = model._completion_dict_to_chat_completion(
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 123,
                "model": "base-model",
                "choices": [{"index": 0, "text": "the answer is 4", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            }
        )
        assert chat_completion.choices[0].message.role == "assistant"
        assert chat_completion.choices[0].message.content == "the answer is 4"
        assert chat_completion.choices[0].finish_reason == "stop"
        assert chat_completion.usage.prompt_tokens == 5

    def test_matched_stop_not_appended(self) -> None:
        model = _make_completions_backend_model()
        chat_completion = model._completion_dict_to_chat_completion(
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 123,
                "model": "base-model",
                "choices": [
                    {
                        "index": 0,
                        "text": "answer",
                        "finish_reason": "stop",
                        "matched_stop": "</s>",
                    }
                ],
            }
        )
        assert chat_completion.choices[0].message.content == "answer"

    def test_inline_token_ids_used(self) -> None:
        model = _make_completions_backend_model(return_token_id_information=True)
        chat_completion = model._completion_dict_to_chat_completion(
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 123,
                "model": "base-model",
                "choices": [
                    {
                        "index": 0,
                        "text": "ok",
                        "finish_reason": "stop",
                        "logprobs": {
                            "tokens": ["token_id:42", "token_id:43"],
                            "token_logprobs": [-0.1, -0.2],
                        },
                        "prompt_token_ids": [6, 7, 8],
                        "token_ids": [42, 43],
                    }
                ],
            }
        )
        msg = chat_completion.choices[0].message
        assert isinstance(msg, NeMoGymChatCompletionMessageForTraining)
        assert msg.generation_token_ids == [42, 43]
        assert msg.generation_log_probs == [-0.1, -0.2]
        assert msg.prompt_token_ids == [6, 7, 8]

    def test_default_id_uses_chat_completion_prefix(self) -> None:
        model = _make_completions_backend_model()
        chat_completion = model._completion_dict_to_chat_completion(
            {
                "object": "text_completion",
                "created": 123,
                "model": "base-model",
                "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
            }
        )
        assert chat_completion.id == "chatcmpl-completions"

    def test_missing_logprobs_raises(self) -> None:
        model = _make_completions_backend_model(return_token_id_information=True)
        with raises(RuntimeError, match="Cannot extract token IDs or logprobs"):
            model._completion_dict_to_chat_completion(
                {
                    "id": "cmpl-1",
                    "object": "text_completion",
                    "created": 123,
                    "model": "base-model",
                    "choices": [
                        {
                            "index": 0,
                            "text": "ok",
                            "finish_reason": "stop",
                            "logprobs": None,
                            "prompt_token_ids": [6, 7],
                            "token_ids": [42],
                        }
                    ],
                }
            )


class TestCompletionsBackendEndToEnd:
    @staticmethod
    def _install_fake_client(model: VLLMModel, completion_response: dict) -> tuple[MagicMock, dict]:
        """Replace model._clients with a single mock client whose create_completion
        returns ``completion_response``. Returns the mock and a captured-kwargs dict.
        """
        captured: dict = {}

        async def fake_create_completion(**kwargs):
            captured["kwargs"] = kwargs
            return completion_response

        fake_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        fake_client.create_completion = fake_create_completion
        fake_client.create_chat_completion = AsyncMock(
            side_effect=AssertionError("create_chat_completion must not be called")
        )
        model._clients = [fake_client]
        # Reset the per-session client routing so the new mock is picked up.
        model._session_id_to_client = {}
        return fake_client, captured

    def test_chat_completions_call_routes_to_create_completion(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", MagicMock(return_value=dict()))

        model = _make_completions_backend_model()
        _, captured = self._install_fake_client(
            model,
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1,
                "model": "base-model",
                "choices": [{"index": 0, "text": "world", "finish_reason": "stop"}],
            },
        )

        app = model.setup_webserver()
        test_client = TestClient(app)
        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "ignored-by-server",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "world"
        # Forwarded prompt is the verbatim message content.
        assert captured["kwargs"]["prompt"] == "hello"
        assert captured["kwargs"]["max_tokens"] == 8
        # Server-config model wins over whatever the caller put in "model".
        assert captured["kwargs"]["model"] == "base-model"

    def test_responses_with_string_input_routes_to_completions(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", MagicMock(return_value=dict()))
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("responses_api_models.vllm_model.app.time", lambda: FIXED_TIME)

        model = _make_completions_backend_model()
        _, captured = self._install_fake_client(
            model,
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": FIXED_TIME,
                "model": "base-model",
                "choices": [{"index": 0, "text": " 4", "finish_reason": "stop"}],
            },
        )

        app = model.setup_webserver()
        test_client = TestClient(app)
        resp = test_client.post(
            "/v1/responses",
            json={"input": "What is 2+2?"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        output_texts = [
            block["text"]
            for item in data["output"]
            if item.get("type") == "message"
            for block in item.get("content", [])
            if block.get("type") == "output_text"
        ]
        assert output_texts == [" 4"]
        assert captured["kwargs"]["prompt"] == "What is 2+2?"

    def test_inline_token_ids_skip_tokenize(self) -> None:
        import asyncio

        model = _make_completions_backend_model(return_token_id_information=True)
        fake_client, captured = self._install_fake_client(
            model,
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1,
                "model": "base-model",
                "choices": [
                    {
                        "index": 0,
                        "text": "ok",
                        "finish_reason": "stop",
                        "logprobs": {
                            "tokens": ["token_id:42"],
                            "token_logprobs": [-0.1],
                        },
                        "prompt_token_ids": [6, 7],
                        "token_ids": [42],
                    }
                ],
            },
        )
        fake_client.create_tokenize = AsyncMock(
            side_effect=AssertionError("create_tokenize must not be called when inline IDs are present")
        )
        model._resolve_client = lambda _request: fake_client  # type: ignore[assignment]

        body = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[{"role": "user", "content": "hello"}],
        )
        result = asyncio.run(model._chat_completions_via_completions_api(MagicMock(), body))

        assert captured["kwargs"]["return_token_ids"] is True
        fake_client.create_tokenize.assert_not_awaited()
        message = result.choices[0].message
        assert isinstance(message, NeMoGymChatCompletionMessageForTraining)
        assert message.prompt_token_ids == [6, 7]
        assert message.generation_token_ids == [42]

    def test_missing_inline_token_ids_fall_back(self) -> None:
        import asyncio

        model = _make_completions_backend_model(
            return_token_id_information=True,
            extra_body={"add_special_tokens": False},
        )
        fake_client, _ = self._install_fake_client(
            model,
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1,
                "model": "base-model",
                "choices": [
                    {
                        "index": 0,
                        "text": "ok",
                        "finish_reason": "stop",
                        "logprobs": {
                            "tokens": ["token_id:42", "token_id:43"],
                            "token_logprobs": [-0.1, -0.2],
                        },
                    }
                ],
            },
        )
        fake_client.create_tokenize = AsyncMock(return_value={"tokens": [6, 7, 8]})
        model._resolve_client = lambda _request: fake_client  # type: ignore[assignment]

        body = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[{"role": "user", "content": "hello"}],
        )
        result = asyncio.run(model._chat_completions_via_completions_api(MagicMock(), body))

        fake_client.create_tokenize.assert_awaited_once_with(
            model="base-model",
            prompt="hello",
            add_special_tokens=False,
        )
        message = result.choices[0].message
        assert isinstance(message, NeMoGymChatCompletionMessageForTraining)
        assert message.prompt_token_ids == [6, 7, 8]
        assert message.generation_token_ids == [42, 43]
        assert message.generation_log_probs == [-0.1, -0.2]


# ──────────────────────────────────────────────────────────────────────────────
# /v1/completions backend with render_chat_template=True
#
# Exercises the client-side HF AutoTokenizer.apply_chat_template render path.
# Uses a stand-in tokenizer object (no transformers / HF download in tests) to
# stay hermetic — VLLMModel only calls ``apply_chat_template(messages,
# tokenize=False, add_generation_prompt=True, tools=..., **chat_template_kwargs)``,
# so the test double captures those args and returns a known string.
# ──────────────────────────────────────────────────────────────────────────────


class _FakeTokenizer:
    """Minimal stand-in for AutoTokenizer used by the chat-template render path.

    Records the apply_chat_template call args and returns a deterministic
    string composed from the messages so tests can assert the render path
    forwarded the right inputs.
    """

    def __init__(self, *, has_chat_template: bool = True):
        self.chat_template = "<jinja>" if has_chat_template else None
        self.last_call: dict | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_call = {"messages": messages, **kwargs}
        rendered_parts = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content)
            rendered_parts.append(f"<|{m['role']}|>{content}")
        if kwargs.get("add_generation_prompt"):
            rendered_parts.append("<|assistant|>")
        return "".join(rendered_parts)


def _make_chat_template_model(
    *,
    has_chat_template: bool = True,
    chat_template_kwargs: dict | None = None,
) -> tuple[VLLMModel, _FakeTokenizer]:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vllm_completions_ct",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="base-model",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
        use_completions_api=True,
        # Note: render_chat_template is left False so VLLMModel doesn't try to
        # actually load HF AutoTokenizer at construction. We flip it on after
        # injecting a _FakeTokenizer instance.
        chat_template_kwargs=chat_template_kwargs,
    )
    model = VLLMModel(
        config=config,
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    fake_tokenizer = _FakeTokenizer(has_chat_template=has_chat_template)
    model._chat_template_tokenizer = fake_tokenizer
    # Now flip the config flag — model_post_init has already run; we just
    # need _chat_completions_via_completions_api to hit the chat_template branch.
    object.__setattr__(model.config, "render_chat_template", True)
    return model, fake_tokenizer


class TestCompletionsBackendChatTemplateRender:
    def test_multi_turn_messages_pass_through_to_apply_chat_template(self) -> None:
        model, tokenizer = _make_chat_template_model()
        messages = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "what's 2+2?"},
        ]
        prompt = model._render_messages_via_chat_template({"messages": messages})

        assert tokenizer.last_call["messages"] == messages
        assert tokenizer.last_call["tokenize"] is False
        assert tokenizer.last_call["add_generation_prompt"] is True
        assert tokenizer.last_call["tools"] is None
        # Sanity-check the rendered string surfaces the multi-turn structure.
        assert prompt == ("<|system|>Be terse.<|user|>hi<|assistant|>hello<|user|>what's 2+2?<|assistant|>")

    def test_tools_pass_through_to_apply_chat_template(self) -> None:
        model, tokenizer = _make_chat_template_model()
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        model._render_messages_via_chat_template(
            {
                "messages": [{"role": "user", "content": "x"}],
                "tools": tools,
            }
        )
        assert tokenizer.last_call["tools"] == tools

    def test_image_block_rejected(self) -> None:
        model, _ = _make_chat_template_model()
        with raises(ValueError, match="only accepts text content blocks"):
            model._render_messages_via_chat_template(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                        }
                    ]
                }
            )

    def test_chat_template_kwargs_merged_with_metadata_override(self) -> None:
        model, tokenizer = _make_chat_template_model(
            chat_template_kwargs={"enable_thinking": True, "tool_use_mode": "tool_calls"},
        )
        body_dict = {
            "messages": [{"role": "user", "content": "x"}],
            "metadata": {
                # Per-request override — JSON string, mirroring the chat-completions path.
                "chat_template_kwargs": json.dumps({"enable_thinking": False, "extra_knob": 7}),
            },
        }
        model._render_messages_via_chat_template(body_dict)

        # Per-request override beats global config; non-overridden global keys persist;
        # new metadata-only keys are added.
        kwargs_seen = {
            k: v
            for k, v in tokenizer.last_call.items()
            if k not in {"messages", "tokenize", "add_generation_prompt", "tools"}
        }
        assert kwargs_seen == {
            "enable_thinking": False,  # metadata override
            "tool_use_mode": "tool_calls",  # global config preserved
            "extra_knob": 7,  # metadata-only addition
        }

    def test_post_init_loads_tokenizer_when_render_chat_template_is_set(self, monkeypatch: MonkeyPatch) -> None:
        """Construction with render_chat_template=True triggers tokenizer load."""
        captured = {}

        class _StubAutoTokenizer:
            @staticmethod
            def from_pretrained(name_or_path, **kwargs):
                captured["name_or_path"] = name_or_path
                captured["kwargs"] = kwargs
                tok = _FakeTokenizer(has_chat_template=True)
                return tok

        # Inject a fake transformers module so `from transformers import AutoTokenizer` works.
        import sys as _sys
        import types as _types

        fake_module = _types.SimpleNamespace(AutoTokenizer=_StubAutoTokenizer)
        monkeypatch.setitem(_sys.modules, "transformers", fake_module)

        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ct",
            base_url="http://localhost:9999/v1",
            api_key="dummy",  # pragma: allowlist secret
            model="base-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            uses_interleaved_reasoning=False,
            use_completions_api=True,
            render_chat_template=True,
            tokenizer="some-instruct-model",
        )
        VLLMModel(
            config=config,
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )

        assert captured["name_or_path"] == "some-instruct-model"
        assert captured["kwargs"].get("trust_remote_code") is True

    def test_post_init_falls_back_to_model_when_tokenizer_unset(self, monkeypatch: MonkeyPatch) -> None:
        captured = {}

        class _StubAutoTokenizer:
            @staticmethod
            def from_pretrained(name_or_path, **kwargs):
                captured["name_or_path"] = name_or_path
                return _FakeTokenizer(has_chat_template=True)

        import sys as _sys
        import types as _types

        monkeypatch.setitem(_sys.modules, "transformers", _types.SimpleNamespace(AutoTokenizer=_StubAutoTokenizer))

        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ct",
            base_url="http://localhost:9999/v1",
            api_key="dummy",  # pragma: allowlist secret
            model="base-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            uses_interleaved_reasoning=False,
            use_completions_api=True,
            render_chat_template=True,
            # tokenizer left unset
        )
        VLLMModel(
            config=config,
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )
        assert captured["name_or_path"] == "base-model"

    def test_post_init_raises_when_loaded_tokenizer_has_no_chat_template(self, monkeypatch: MonkeyPatch) -> None:
        class _StubAutoTokenizer:
            @staticmethod
            def from_pretrained(name_or_path, **kwargs):
                return _FakeTokenizer(has_chat_template=False)

        import sys as _sys
        import types as _types

        monkeypatch.setitem(_sys.modules, "transformers", _types.SimpleNamespace(AutoTokenizer=_StubAutoTokenizer))

        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ct",
            base_url="http://localhost:9999/v1",
            api_key="dummy",  # pragma: allowlist secret
            model="base-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            uses_interleaved_reasoning=False,
            use_completions_api=True,
            render_chat_template=True,
        )
        try:
            VLLMModel(
                config=config,
                server_client=MagicMock(spec=ServerClient, global_config_dict={}),
            )
        except RuntimeError as e:
            assert "no chat_template" in str(e)
        else:
            raise AssertionError("expected RuntimeError when tokenizer has no chat_template")

    def test_tools_allowed_in_chat_template_mode(self) -> None:
        """Tools must NOT be rejected when render_chat_template=True."""
        import asyncio

        model, _tokenizer = _make_chat_template_model()
        body = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object"}},
                }
            ],
        )
        # Mock the underlying client.create_completion so the request reaches
        # rendering but doesn't actually hit the network.
        captured: dict = {}

        async def fake_create_completion(**kwargs):
            captured["kwargs"] = kwargs
            return {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1,
                "model": "base-model",
                "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
            }

        fake_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        fake_client.create_completion = fake_create_completion
        model._clients = [fake_client]
        model._session_id_to_client = {}

        # Patch _resolve_client to skip the session machinery.
        model._resolve_client = lambda r: fake_client  # type: ignore[assignment]
        request = MagicMock()

        result = asyncio.run(model._chat_completions_via_completions_api(request, body))
        assert result.choices[0].message.content == "ok"
        # Prompt forwarded contains the rendered template (with the tool slot
        # delegated to the template implementation; here our fake just emits
        # the role+content marker).
        assert "<|user|>hi" in captured["kwargs"]["prompt"]


def _make_top_logprobs_model(
    return_token_id_information: bool,
    *,
    extra_body: dict[str, Any] | None = None,
    request_prompt_and_generation_token_ids: bool = False,
) -> VLLMModel:
    """A VLLMModel with the minimum config needed to exercise top_logprobs handling."""
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vllm_model",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy_model",
        return_token_id_information=return_token_id_information,
        request_prompt_and_generation_token_ids=request_prompt_and_generation_token_ids,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
        extra_body=extra_body,
    )
    return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


class TestTopLogprobsHandling:
    """With logprobs=True, vLLM treats top_logprobs=null as "no logprobs" but a missing
    field as its default (0, the chosen token's logprob). A null commonly appears in
    replayed rollout JSONL and empties the captured token ids, zeroing the training loss
    mask. These tests cover how the model server defends against that.
    """

    def test_capture_path_pins_top_logprobs_to_zero(self) -> None:
        """With return_token_id_information=True the server pins top_logprobs=0 regardless
        of the inbound value, so no dataset value can defeat token-id capture."""
        model = _make_top_logprobs_model(return_token_id_information=True)

        # Inbound explicit null — the regression case.
        result = model._preprocess_chat_completion_create_params(
            MagicMock(),
            {"model": "dummy_model", "messages": [{"role": "user", "content": "hi"}], "top_logprobs": None},
        )
        assert result["top_logprobs"] == 0
        assert result["logprobs"] is True
        assert result["return_tokens_as_token_ids"] is True
        assert "return_token_ids" not in result

        # Inbound non-zero value must also be overridden, not inherited.
        result = model._preprocess_chat_completion_create_params(
            MagicMock(),
            {"model": "dummy_model", "messages": [{"role": "user", "content": "hi"}], "top_logprobs": 5},
        )
        assert result["top_logprobs"] == 0

    def test_capture_path_can_request_prompt_and_generation_token_ids(self) -> None:
        model = _make_top_logprobs_model(
            return_token_id_information=True,
            request_prompt_and_generation_token_ids=True,
        )

        result = model._preprocess_chat_completion_create_params(
            MagicMock(),
            {"model": "dummy_model", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert result["return_token_ids"] is True

    def test_capture_path_rejects_multiple_choices(self) -> None:
        model = _make_top_logprobs_model(return_token_id_information=True)

        with raises(ValueError, match="requires n=1"):
            model._preprocess_chat_completion_create_params(
                MagicMock(),
                {
                    "model": "dummy_model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "n": 2,
                },
            )

    def test_noncapture_path_strips_null_top_logprobs(self) -> None:
        """On the non-capture path a null top_logprobs is dropped (letting vLLM apply its
        default) rather than forwarded as null (which vLLM reads as "no logprobs")."""
        model = _make_top_logprobs_model(return_token_id_information=False)

        result = model._preprocess_chat_completion_create_params(
            MagicMock(),
            {
                "model": "dummy_model",
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True,
                "top_logprobs": None,
            },
        )
        assert "top_logprobs" not in result

        # A genuine integer value must pass through untouched.
        result = model._preprocess_chat_completion_create_params(
            MagicMock(),
            {
                "model": "dummy_model",
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True,
                "top_logprobs": 3,
            },
        )
        assert result["top_logprobs"] == 3

        # An omitted field stays omitted.
        result = model._preprocess_chat_completion_create_params(
            MagicMock(),
            {"model": "dummy_model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert "top_logprobs" not in result

    def _capture_chat_completion_dict(
        self,
        logprobs: Union[dict, None],
        message_extra: Union[dict, None] = None,
        choice_extra: Union[dict, None] = None,
        response_extra: Union[dict, None] = None,
    ) -> dict:
        message = {"role": "assistant", "content": "hi", "tool_calls": None}
        if message_extra:
            message.update(message_extra)
        choice = {
            "index": 0,
            "finish_reason": "stop",
            "message": message,
            "logprobs": logprobs,
        }
        if choice_extra:
            choice.update(choice_extra)
        response = {
            "id": "chtcmpl",
            "object": "chat.completion",
            "created": FIXED_TIME,
            "model": "dummy_model",
            "choices": [choice],
        }
        if response_extra:
            response.update(response_extra)
        return response

    def test_capture_path_succeeds_with_inbound_null_top_logprobs(self) -> None:
        """End-to-end regression: a request with top_logprobs=null no longer empties capture;
        token ids and logprobs come back populated (and coerced to int)."""
        model = _make_top_logprobs_model(return_token_id_information=True)
        app = model.setup_webserver()

        captured_kwargs: dict[str, Any] = {}

        async def mock_create_chat_completion(**kwargs):
            captured_kwargs.update(kwargs)
            return self._capture_chat_completion_dict(
                logprobs={
                    "content": [
                        {"token": "token_id:123", "logprob": -0.1, "bytes": None, "top_logprobs": []},
                        {"token": "token_id:456", "logprob": -0.2, "bytes": None, "top_logprobs": []},
                    ]
                }
            )

        async def mock_create_tokenize(**kwargs):
            return {"tokens": [10, 20, 30]}

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        mock_client.create_tokenize = AsyncMock(side_effect=mock_create_tokenize)
        model._clients = [mock_client]

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "top_logprobs": None},
        )
        assert response.status_code == 200

        # The request forwarded to vLLM had top_logprobs pinned to 0, not the inbound null.
        assert captured_kwargs["top_logprobs"] == 0

        message = response.json()["choices"][0]["message"]
        assert message["generation_token_ids"] == [123, 456]
        assert message["generation_log_probs"] == [-0.1, -0.2]
        assert message["prompt_token_ids"] == [10, 20, 30]

    def test_capture_path_prefers_vllm_response_token_ids(self) -> None:
        model = _make_top_logprobs_model(
            return_token_id_information=True,
            request_prompt_and_generation_token_ids=True,
        )
        app = model.setup_webserver()

        async def mock_create_chat_completion(**kwargs):
            assert kwargs["return_token_ids"] is True
            return self._capture_chat_completion_dict(
                logprobs={
                    "content": [
                        {"token": "token_id:123", "logprob": -0.1, "bytes": None, "top_logprobs": []},
                        {"token": "token_id:456", "logprob": -0.2, "bytes": None, "top_logprobs": []},
                    ]
                },
                choice_extra={"token_ids": [123, 456]},
                response_extra={"prompt_token_ids": [10, 20, 30]},
            )

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        mock_client.create_tokenize = AsyncMock(
            side_effect=AssertionError("create_tokenize must not be called when inline IDs are present")
        )
        model._clients = [mock_client]

        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 200
        mock_client.create_tokenize.assert_not_awaited()
        data = response.json()
        message = data["choices"][0]["message"]
        assert message["prompt_token_ids"] == [10, 20, 30]
        assert message["generation_token_ids"] == [123, 456]
        assert message["generation_log_probs"] == [-0.1, -0.2]
        assert "prompt_token_ids" not in data
        assert "token_ids" not in data["choices"][0]

    def test_capture_path_accepts_framework_message_bundle(self) -> None:
        model = _make_top_logprobs_model(return_token_id_information=True)
        app = model.setup_webserver()
        token_bundle = {
            "prompt_token_ids": [10, 20],
            "generation_token_ids": [123],
            "generation_log_probs": [-0.1],
        }

        async def mock_create_chat_completion(**kwargs):
            return self._capture_chat_completion_dict(
                logprobs=None,
                message_extra=token_bundle,
                choice_extra={"token_ids": [123]},
                response_extra={"prompt_token_ids": [10, 20]},
            )

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        mock_client.create_tokenize = AsyncMock(
            side_effect=AssertionError("create_tokenize must not be called for a framework token bundle")
        )
        model._clients = [mock_client]

        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        for field, value in token_bundle.items():
            assert message[field] == value
        mock_client.create_tokenize.assert_not_awaited()

    def test_capture_path_rejects_disagreeing_duplicate_ids(self) -> None:
        model = _make_top_logprobs_model(return_token_id_information=True)
        app = model.setup_webserver()

        async def mock_create_chat_completion(**kwargs):
            return self._capture_chat_completion_dict(
                logprobs=None,
                message_extra={
                    "prompt_token_ids": [10, 20],
                    "generation_token_ids": [123],
                    "generation_log_probs": [-0.1],
                },
                choice_extra={"token_ids": [999]},
                response_extra={"prompt_token_ids": [10, 20]},
            )

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        model._clients = [mock_client]

        with raises(RuntimeError, match="disagrees with vLLM response token IDs"):
            TestClient(app).post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    def test_capture_path_forwards_prompt_affecting_fields_to_tokenize(self) -> None:
        model = _make_top_logprobs_model(
            return_token_id_information=True,
            extra_body={
                "mm_processor_kwargs": {"fps": 2},
                "required_prefix_token_ids": [1, 2, 3],
            },
        )
        app = model.setup_webserver()

        async def mock_create_chat_completion(**kwargs):
            return self._capture_chat_completion_dict(
                logprobs={
                    "content": [
                        {"token": "token_id:123", "logprob": -0.1, "bytes": None, "top_logprobs": []},
                    ]
                }
            )

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        mock_client.create_tokenize = AsyncMock(return_value={"tokens": [10, 20]})
        model._clients = [mock_client]

        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 200
        mock_client.create_tokenize.assert_awaited_once_with(
            model="dummy_model",
            messages=[{"role": "user", "content": "hi"}],
            mm_processor_kwargs={"fps": 2},
            required_prefix_token_ids=[1, 2, 3],
        )

    def test_capture_path_rejects_partial_message_bundle(self) -> None:
        model = _make_top_logprobs_model(return_token_id_information=True)
        app = model.setup_webserver()

        async def mock_create_chat_completion(**kwargs):
            return self._capture_chat_completion_dict(
                logprobs=None,
                message_extra={"prompt_token_ids": [10, 20]},
            )

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        model._clients = [mock_client]

        with raises(RuntimeError, match="partial token metadata"):
            TestClient(app).post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    def test_capture_path_rejects_mismatched_generation_lengths(self) -> None:
        model = _make_top_logprobs_model(return_token_id_information=True)
        app = model.setup_webserver()

        async def mock_create_chat_completion(**kwargs):
            return self._capture_chat_completion_dict(
                logprobs=None,
                message_extra={
                    "prompt_token_ids": [10, 20],
                    "generation_token_ids": [123, 456],
                    "generation_log_probs": [-0.1],
                },
            )

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        model._clients = [mock_client]

        with raises(RuntimeError, match="mismatched generation token IDs"):
            TestClient(app).post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    def test_capture_path_preserves_routed_experts(self) -> None:
        model = _make_top_logprobs_model(return_token_id_information=True)
        app = model.setup_webserver()
        routed_experts = [
            [[0, 1]],
            [[2, 3]],
            [[4, 5]],
        ]

        async def mock_create_chat_completion(**kwargs):
            return self._capture_chat_completion_dict(
                logprobs={
                    "content": [
                        {"token": "token_id:123", "logprob": -0.1, "bytes": None, "top_logprobs": []},
                    ]
                },
                message_extra={"routed_experts": routed_experts},
            )

        async def mock_create_tokenize(**kwargs):
            return {"tokens": [10, 20]}

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        mock_client.create_tokenize = AsyncMock(side_effect=mock_create_tokenize)
        model._clients = [mock_client]

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["routed_experts"] == routed_experts

    def test_capture_path_preserves_routed_experts_string_envelope(self) -> None:
        """Training frameworks may ship routes as one opaque string (e.g. NeMo-RL's
        "nrlre1:<dtype>:<SxLxK>:<base64>"); it must pass through unmodified."""
        model = _make_top_logprobs_model(return_token_id_information=True)
        app = model.setup_webserver()
        routed_experts = "nrlre1:int16:3x1x2:AAABAAIAAwAEAAUA"

        async def mock_create_chat_completion(**kwargs):
            return self._capture_chat_completion_dict(
                logprobs={
                    "content": [
                        {"token": "token_id:123", "logprob": -0.1, "bytes": None, "top_logprobs": []},
                    ]
                },
                message_extra={"routed_experts": routed_experts},
            )

        async def mock_create_tokenize(**kwargs):
            return {"tokens": [10, 20]}

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        mock_client.create_tokenize = AsyncMock(side_effect=mock_create_tokenize)
        model._clients = [mock_client]

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["routed_experts"] == routed_experts

    def test_capture_path_raises_when_logprobs_missing(self) -> None:
        """If capture is on but vLLM returns no logprobs, fail loudly with an actionable
        error instead of an opaque TypeError or silently empty training data."""
        model = _make_top_logprobs_model(return_token_id_information=True)
        app = model.setup_webserver()

        async def mock_create_chat_completion(**kwargs):
            return self._capture_chat_completion_dict(logprobs=None)

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        mock_client.create_tokenize = AsyncMock()
        model._clients = [mock_client]

        client = TestClient(app)
        with raises(RuntimeError, match="Cannot extract token ids"):
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        # The tokenize endpoint must not be reached once the contract check fails.
        mock_client.create_tokenize.assert_not_called()


class TestSamplingOverrides:
    """Forcing the sampling params on every request.

    An external harness picks its own temperature and top_p. On-policy RL requires
    generation to match the distribution the policy is optimized under, so the
    server overrides whatever the client sent rather than trusting it.
    """

    @staticmethod
    def _server(overrides: dict[str, object] | None, **kwargs: object) -> VLLMModel:
        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8081,
            base_url="http://api.openai.com/v1",
            api_key="dummy_key",  # pragma: allowlist secret
            model="dummy_model",
            entrypoint="",
            name="",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            sampling_overrides=overrides,
            **kwargs,
        )
        return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))

    def test_overrides_replace_what_the_client_sent(self) -> None:
        server = self._server({"temperature": 1.0, "top_p": 1.0})
        out = server._preprocess_chat_completion_create_params(
            MagicMock(), {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.2, "top_p": 0.5}
        )
        assert out["temperature"] == 1.0
        assert out["top_p"] == 1.0

    def test_overrides_apply_even_when_the_client_sent_nothing(self) -> None:
        server = self._server({"temperature": 1.0})
        out = server._preprocess_chat_completion_create_params(
            MagicMock(), {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert out["temperature"] == 1.0

    def test_unset_leaves_the_request_alone(self) -> None:
        server = self._server(None)
        out = server._preprocess_chat_completion_create_params(
            MagicMock(), {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.2}
        )
        assert out["temperature"] == 0.2

    def test_overrides_reach_the_completions_api_path(self) -> None:
        """``use_completions_api`` skips chat preprocessing entirely.

        ``chat_completions`` branches into ``_chat_completions_via_completions_api`` before
        ``_preprocess_chat_completion_create_params`` runs, so a pin applied only there would be
        silently inert on the path base-model training uses.
        """
        server = self._server({"temperature": 1.0, "top_p": 1.0, "top_k": -1}, use_completions_api=True)
        out = server._build_completion_body_from_chat_body(
            {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.2, "top_p": 0.5, "top_k": 50},
            "hi",
        )
        assert out["temperature"] == 1.0
        assert out["top_p"] == 1.0
        # No first-class /v1/completions field; the body is forwarded as raw JSON so it still lands.
        assert out["top_k"] == -1

    def test_overrides_win_over_extra_body_on_the_completions_path(self) -> None:
        """``extra_body`` merges under the request body; the pin is applied after both."""
        server = self._server(
            {"temperature": 1.0},
            use_completions_api=True,
            extra_body={"temperature": 0.7, "min_p": 0.05},
        )
        out = server._build_completion_body_from_chat_body({"messages": [], "temperature": 0.2}, "hi")
        assert out["temperature"] == 1.0
        assert out["min_p"] == 0.05

    def test_overrides_reach_the_responses_native_path(self) -> None:
        server = self._server({"temperature": 1.0}, is_responses_native=True)
        body = {"model": "dummy_model", "temperature": 0.2}
        assert server._apply_sampling_overrides(body)["temperature"] == 1.0


class TestPrefixSupply:
    """Supply a verified parent's exact tokens to the engine.

    Re-rendering can tokenize an assistant turn differently from generation.
    A reasoning template can also omit earlier thinking.
    Either case breaks the token chain.
    """

    @staticmethod
    def _server(monkeypatch: MonkeyPatch, *, enabled: bool) -> VLLMModel:
        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8081,
            base_url="http://api.openai.com/v1",
            api_key="dummy_key",  # pragma: allowlist secret
            model="dummy_model",
            entrypoint="",
            name="",
            return_token_id_information=enabled,
            uses_reasoning_parser=False,
            supply_prefix_token_ids=enabled,
        )
        get_global_config_dict_mock = MagicMock(return_value={})
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)
        return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))

    @staticmethod
    def _armed(rollout_id: str, tmp_path) -> CaptureContext:
        return CaptureContext(
            rollout_id=rollout_id,
            model_call_id="call-x",
            token_sink=TokenCaptureStore(tmp_path),
            lineage_store=_TEST_LINEAGE,
        )

    def test_requires_token_information_for_generation_proof(self) -> None:
        with raises(ValueError, match="return_token_id_information=true"):
            VLLMModelConfig(
                host="0.0.0.0",
                port=8081,
                base_url="http://api.openai.com/v1",
                api_key="dummy_key",  # pragma: allowlist secret
                model="dummy_model",
                entrypoint="",
                name="",
                return_token_id_information=False,
                uses_reasoning_parser=False,
                supply_prefix_token_ids=True,
            )

    def test_rejects_completions_api_prefix_supply(self) -> None:
        with raises(ValueError, match="not supported with use_completions_api=true"):
            VLLMModelConfig(
                host="0.0.0.0",
                port=8081,
                base_url="http://api.openai.com/v1",
                api_key="dummy_key",  # pragma: allowlist secret
                model="dummy_model",
                entrypoint="",
                name="",
                return_token_id_information=True,
                uses_reasoning_parser=False,
                supply_prefix_token_ids=True,
                use_completions_api=True,
            )

    def test_supplies_the_parents_cumulative_tokens(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        server = self._server(monkeypatch, enabled=True)
        first_turn = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        lineage_index().for_rollout("sup-0").record("parent", first_turn, [1, 2, 3, 4], "d")

        token = set_token_sink(self._armed("sup-0", tmp_path))
        try:
            asyncio.run(resolve_parent(first_turn + [{"role": "user", "content": "next"}]))
            out = server._apply_prefix_supply({"messages": first_turn + [{"role": "user", "content": "next"}]})
        finally:
            reset_token_sink(token)

        assert out["required_prefix_token_ids"] == [1, 2, 3, 4]

    def test_off_by_default(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        server = self._server(monkeypatch, enabled=False)
        turn = [{"role": "assistant", "content": "hello"}]
        lineage_index().for_rollout("sup-1").record("parent", turn, [1, 2], "d")

        token = set_token_sink(self._armed("sup-1", tmp_path))
        try:
            asyncio.run(resolve_parent(turn))
            out = server._apply_prefix_supply({"messages": turn})
        finally:
            reset_token_sink(token)

        assert "required_prefix_token_ids" not in out

    def test_uncorrelated_call_is_left_alone(self, monkeypatch: MonkeyPatch) -> None:
        server = self._server(monkeypatch, enabled=True)
        asyncio.run(resolve_parent([{"role": "assistant", "content": "hello"}]))
        out = server._apply_prefix_supply({"messages": [{"role": "assistant", "content": "hello"}]})
        assert "required_prefix_token_ids" not in out

    def test_fingerprint_miss_falls_back_rather_than_supplying_something_wrong(
        self, monkeypatch: MonkeyPatch, tmp_path
    ) -> None:
        """Leave a rewritten history untouched.

        The backend trusts the supplied prefix.
        A wrong prefix would generate from a conversation that the harness never requested.
        """
        server = self._server(monkeypatch, enabled=True)
        lineage_index().for_rollout("sup-2").record("parent", [{"role": "assistant", "content": "hello"}], [1, 2], "d")

        token = set_token_sink(self._armed("sup-2", tmp_path))
        try:
            asyncio.run(resolve_parent([{"role": "assistant", "content": "a summary"}]))
            out = server._apply_prefix_supply({"messages": [{"role": "assistant", "content": "a summary"}]})
        finally:
            reset_token_sink(token)

        assert "required_prefix_token_ids" not in out

    def test_ambiguous_parent_is_not_supplied(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        server = self._server(monkeypatch, enabled=True)
        turn = [{"role": "assistant", "content": "same"}]
        lineage = lineage_index().for_rollout("sup-3")
        lineage.record("a", turn, [1, 2], "da")
        lineage.record("b", turn, [3, 4], "db")

        token = set_token_sink(self._armed("sup-3", tmp_path))
        try:
            asyncio.run(resolve_parent(turn))
            out = server._apply_prefix_supply({"messages": turn})
        finally:
            reset_token_sink(token)

        assert "required_prefix_token_ids" not in out

    def test_a_fork_gets_the_parents_prefix_not_the_previous_calls(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        """Two sub-agents branching from one parent must both get cum(parent).

        A running cursor would include the first branch's generation in the second branch.
        The backend would then generate from a conversation that never happened.
        """
        server = self._server(monkeypatch, enabled=True)
        shared = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "plan"}]
        lineage = lineage_index().for_rollout("sup-4")
        lineage.record("parent", shared, [1, 2, 3], "dp")
        lineage.record(
            "branch-a",
            shared + [{"role": "user", "content": "a"}, {"role": "assistant", "content": "A"}],
            [1, 2, 3, 9, 9],
            "da",
        )

        token = set_token_sink(self._armed("sup-4", tmp_path))
        try:
            asyncio.run(resolve_parent(shared + [{"role": "user", "content": "b"}]))
            out = server._apply_prefix_supply({"messages": shared + [{"role": "user", "content": "b"}]})
        finally:
            reset_token_sink(token)

        assert out["required_prefix_token_ids"] == [1, 2, 3]

    def test_reasoning_stripped_history_still_supplies_the_real_tokens(
        self, monkeypatch: MonkeyPatch, tmp_path
    ) -> None:
        """Restore reasoning tokens omitted from the rendered history.

        A reasoning template can drop earlier thinking from a later prompt.
        The rendered prompt then cannot extend the previous generation.
        The verified parent's recorded tokens restore the chain.
        """
        server = self._server(monkeypatch, enabled=True)
        # The recorded turn included reasoning.
        # The history returned by the harness does not.
        recorded_turn = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}]
        real_tokens_including_reasoning = [1, 2, 3, 4, 5, 6, 7]
        lineage_index().for_rollout("sup-5").record("parent", recorded_turn, real_tokens_including_reasoning, "d")

        token = set_token_sink(self._armed("sup-5", tmp_path))
        try:
            asyncio.run(resolve_parent(recorded_turn + [{"role": "user", "content": "next"}]))
            out = server._apply_prefix_supply({"messages": recorded_turn + [{"role": "user", "content": "next"}]})
        finally:
            reset_token_sink(token)

        assert out["required_prefix_token_ids"] == real_tokens_including_reasoning


class TestPrefixSupplyAccounting:
    """Supply must be auditable after the fact.

    Contiguous chains do not prove that supply fired.
    prefix_requested records intent only.
    prefix_supplied records generation-time proof.
    A lock protects the diagnostic counters.
    """

    def test_supplied_call_is_only_requested_until_generation_proves_it(
        self, monkeypatch: MonkeyPatch, tmp_path
    ) -> None:
        server = TestPrefixSupply._server(monkeypatch, enabled=True)
        turn = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        lineage_index().for_rollout("acct-0").record("parent", turn, [1, 2, 3], "d")

        ctx = CaptureContext(
            rollout_id="acct-0",
            model_call_id="c",
            token_sink=TokenCaptureStore(tmp_path),
            lineage_store=_TEST_LINEAGE,
        )
        token = set_token_sink(ctx)
        try:
            asyncio.run(resolve_parent(turn + [{"role": "user", "content": "next"}]))
            out = server._apply_prefix_supply({"messages": turn + [{"role": "user", "content": "next"}]})
        finally:
            reset_token_sink(token)

        assert out["required_prefix_token_ids"] == [1, 2, 3]
        assert ctx.prefix_requested is True
        assert ctx.prefix_supplied is False
        assert server._prefix_supply_counts == [0, 1]

    def test_fallback_counts_as_eligible_but_not_supplied(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        server = TestPrefixSupply._server(monkeypatch, enabled=True)
        lineage_index().for_rollout("acct-1").record("parent", [{"role": "assistant", "content": "a"}], [1, 2], "d")

        ctx = CaptureContext(
            rollout_id="acct-1",
            model_call_id="c",
            token_sink=TokenCaptureStore(tmp_path),
            lineage_store=_TEST_LINEAGE,
        )
        token = set_token_sink(ctx)
        try:
            # A rewritten history has no unique verified parent.
            # Prefix supply must decline.
            asyncio.run(resolve_parent([{"role": "assistant", "content": "rewritten"}]))
            out = server._apply_prefix_supply({"messages": [{"role": "assistant", "content": "rewritten"}]})
        finally:
            reset_token_sink(token)

        assert "required_prefix_token_ids" not in out
        assert ctx.prefix_supplied is False
        assert server._prefix_supply_counts == [0, 1]  # Configured but not proven.


class TestPrefixSupplyReachesTokenize:
    """Require generation-time proof for supplied prefixes.

    The generation request carries required_prefix_token_ids.
    A supplied request skips the separate tokenize call.
    The generation response must return the actual prompt_token_ids.
    Those token IDs prove whether the backend applied the requested prefix.
    """

    @staticmethod
    def _model() -> VLLMModel:
        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="vllm_model",
            base_url="http://localhost:9999/v1",
            api_key="dummy_key",  # pragma: allowlist secret
            model="dummy_model",
            return_token_id_information=True,
            uses_reasoning_parser=False,
            uses_interleaved_reasoning=False,
            supply_prefix_token_ids=True,
        )
        return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))

    def test_supplied_prefix_uses_generation_prompt_proof(self, tmp_path) -> None:
        model = self._model()
        app = model.setup_webserver()

        turn = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        lineage_index().for_rollout("tok-0").record("parent", turn, [11, 12, 13], "d")

        chat_kwargs: dict[str, Any] = {}
        tokenize_kwargs: dict[str, Any] = {}

        async def mock_create_chat_completion(**kwargs):
            chat_kwargs.update(kwargs)
            return {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "dummy_model",
                "prompt_token_ids": [11, 12, 13, 77],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                        "logprobs": {
                            "content": [{"token": "token_id:77", "logprob": -0.5, "bytes": None, "top_logprobs": []}]
                        },
                    }
                ],
            }

        async def mock_create_tokenize(**kwargs):
            tokenize_kwargs.update(kwargs)
            return {"tokens": [11, 12, 13, 77]}

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        mock_client.create_tokenize = AsyncMock(side_effect=mock_create_tokenize)
        model._clients = [mock_client]

        sink = set_token_sink(
            CaptureContext(
                rollout_id="tok-0",
                model_call_id="call-y",
                token_sink=TokenCaptureStore(tmp_path),
                lineage_store=_TEST_LINEAGE,
            )
        )
        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": turn + [{"role": "user", "content": "next"}]},
            )
        finally:
            reset_token_sink(sink)

        assert response.status_code == 200
        assert chat_kwargs["required_prefix_token_ids"] == [11, 12, 13]
        assert tokenize_kwargs == {}
        assert mock_client.create_tokenize.await_count == 0

    def _supply_and_return_prompt(
        self, tmp_path, prompt_tokens: list[int] | None, rollout: str
    ) -> tuple[CaptureContext, RuntimeError | None]:
        """Run one supplied call against an engine that returns ``prompt_tokens``."""
        model = self._model()
        app = model.setup_webserver()

        turn = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        lineage_index().for_rollout(rollout).record("parent", turn, [11, 12, 13], "d")

        async def mock_create_chat_completion(**kwargs):
            return {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "dummy_model",
                "prompt_token_ids": prompt_tokens,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                        "logprobs": {
                            "content": [{"token": "token_id:77", "logprob": -0.5, "bytes": None, "top_logprobs": []}]
                        },
                    }
                ],
            }

        async def mock_create_tokenize(**kwargs):
            return {"tokens": prompt_tokens}

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        mock_client.create_tokenize = AsyncMock(side_effect=mock_create_tokenize)
        model._clients = [mock_client]

        context = CaptureContext(
            rollout_id=rollout,
            model_call_id="call-v",
            token_sink=TokenCaptureStore(tmp_path),
            lineage_store=_TEST_LINEAGE,
        )
        sink = set_token_sink(context)
        error = None
        try:
            response = TestClient(app).post(
                "/v1/chat/completions",
                json={"messages": turn + [{"role": "user", "content": "next"}]},
            )
        except RuntimeError as caught:
            error = caught
            response = None
        finally:
            reset_token_sink(sink)
        if response is not None:
            assert response.status_code == 200
        return context, error

    def test_a_backend_that_ignores_the_prefix_is_recorded_as_not_supplied(self, tmp_path, caplog) -> None:
        """Treat a requested prefix as intent until the generation response proves application.

        A backend can ignore an unsupported request field and answer normally.
        Only generation-time prompt_token_ids can detect that behavior.
        """
        with caplog.at_level(logging.ERROR):
            context, error = self._supply_and_return_prompt(tmp_path, [900, 901, 77], "ver-ignored")
        assert error is not None
        assert context.prefix_supplied is False
        assert "do not start with required_prefix_token_ids" in str(error)

    def test_a_backend_that_applies_the_prefix_is_recorded_as_supplied(self, tmp_path) -> None:
        """Record supply when generation-time prompt_token_ids start with the requested prefix."""
        context, error = self._supply_and_return_prompt(tmp_path, [11, 12, 13, 77], "ver-applied")
        assert error is None
        assert context.prefix_supplied is True

    def test_a_backend_without_generation_prompt_proof_fails_closed(self, tmp_path) -> None:
        context, error = self._supply_and_return_prompt(tmp_path, None, "ver-unproved")
        assert error is not None
        assert "did not include prompt_token_ids" in str(error)
        assert context.prefix_supplied is False

    def test_tokenize_body_omits_the_prefix_when_supply_did_not_fire(self, tmp_path) -> None:
        """Omit the prefix when no verified parent was resolved."""
        model = self._model()
        app = model.setup_webserver()

        tokenize_kwargs: dict[str, Any] = {}

        async def mock_create_chat_completion(**kwargs):
            return {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "dummy_model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                        "logprobs": {
                            "content": [{"token": "token_id:77", "logprob": -0.5, "bytes": None, "top_logprobs": []}]
                        },
                    }
                ],
            }

        async def mock_create_tokenize(**kwargs):
            tokenize_kwargs.update(kwargs)
            return {"tokens": [5]}

        mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
        mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
        mock_client.create_tokenize = AsyncMock(side_effect=mock_create_tokenize)
        model._clients = [mock_client]

        sink = set_token_sink(
            CaptureContext(
                rollout_id="tok-unknown",
                model_call_id="call-z",
                token_sink=TokenCaptureStore(tmp_path),
                lineage_store=_TEST_LINEAGE,
            )
        )
        try:
            client = TestClient(app)
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        finally:
            reset_token_sink(sink)

        assert response.status_code == 200
        assert "required_prefix_token_ids" not in tokenize_kwargs


class TestPrefixSupplyUsesTheRequestAsReceived:
    """Supply consumes a parent resolved before the request was dispatched.

    Responses-to-Chat conversion and preprocessing can reshape the request body.
    The lineage index records the request as received.
    Resolving from the reshaped body could miss a verified parent.
    """

    def test_supply_does_not_fire_without_a_resolved_parent(self, monkeypatch, tmp_path):
        server = TestPrefixSupply._server(monkeypatch, enabled=True)
        turn = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        lineage_index().for_rollout("sup-x").record("parent", turn, [1, 2, 3], "d")

        token = set_token_sink(TestPrefixSupply._armed("sup-x", tmp_path))
        try:
            # No parent resolution proved that this request continues another call.
            out = server._apply_prefix_supply({"messages": turn})
        finally:
            reset_token_sink(token)

        assert "required_prefix_token_ids" not in out

    def test_a_body_reshaped_after_resolution_still_supplies(self, monkeypatch, tmp_path):
        """Preserve a verified parent decision across later request reshaping."""
        server = TestPrefixSupply._server(monkeypatch, enabled=True)
        original = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        lineage_index().for_rollout("sup-y").record("parent", original, [7, 8, 9], "d")

        token = set_token_sink(TestPrefixSupply._armed("sup-y", tmp_path))
        try:
            asyncio.run(resolve_parent(original))
            # Conversion reshaped the messages before prefix supply ran.
            out = server._apply_prefix_supply(
                {"messages": [{"role": "user", "content": "reshaped beyond recognition"}]}
            )
        finally:
            reset_token_sink(token)

        assert out["required_prefix_token_ids"] == [7, 8, 9]
