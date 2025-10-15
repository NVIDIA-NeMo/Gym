# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from typing import Any, Union
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch, mark

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
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import (
    VLLMConverter,
    VLLMModel,
    VLLMModelConfig,
)


# Used for mocking created_at timestamp generation
FIXED_TIME = 1691418000
FIXED_UUID = "123"


class FakeUUID:
    """Used for mocking UUIDs"""

    hex = FIXED_UUID


COMMON_RESPONSE_PARAMS = dict(
    parallel_tool_calls=True,
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
                    tool_calls=[],
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
                    tool_calls=[],
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
                    tool_calls=[],
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

        return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_sanity(self, monkeypatch: MonkeyPatch) -> None:
        self._setup_server(monkeypatch)

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
        monkeypatch.setattr("responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID())

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
                "strict": True,
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
                "strict": True,
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
                NeMoGymResponseReasoningItem(
                    id="rs_123",
                    status="completed",
                    type="reasoning",
                    summary=[
                        NeMoGymSummary(
                            type="summary_text",
                            text="Searching for a location before analyzing weather patterns...",
                        )
                    ],
                ),
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    status="completed",
                    role="assistant",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text="What city and/or region do you need weather data for?",
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
        monkeypatch.setattr("responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID())

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
                "tool_calls": [],
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
                NeMoGymResponseReasoningItem(
                    id="rs_123",
                    status="completed",
                    type="reasoning",
                    summary=[
                        NeMoGymSummary(
                            type="summary_text",
                            text="Order #1234 is shipped and scheduled for delivery tomorrow. Tomorrow's date is 2025-08-14. The next day is 2025-08-15 and is not a holiday. I need to send a note to the courier to update the delivery date to 2025-08-15.",
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
        monkeypatch.setattr("responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID())

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
                "tool_calls": [],
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
                "strict": True,
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
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID())
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

    def test_client_session_routing(self, monkeypatch: MonkeyPatch):
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
        server = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient))
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

        # Test first query by client 1 goes to underlying client 1
        client_1 = TestClient(app)
        response_1_1 = client_1.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_1_1.status_code == 200
        data = response_1_1.json()
        assert data["output"][0]["content"][0]["text"] == "1"

        # Test first query by client 2 goes to underlying client 2 (round robin)
        client_2 = TestClient(app)
        response_2_1 = client_2.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_2_1.status_code == 200
        data = response_2_1.json()
        assert data["output"][0]["content"][0]["text"] == "2"

        # Test first query by client 3 goes to underlying client 1 = 3 % 2 (round robin)
        client_3 = TestClient(app)
        response_3_1 = client_3.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_3_1.status_code == 200
        data = response_3_1.json()
        assert data["output"][0]["content"][0]["text"] == "1"

        # Test second query by client 1 goes to the same underlying client 1 (not round robin since we've called it before)
        # Here, we assume that TestClient will extract and propogate the response cookies
        response_1_2 = client_1.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_1_2.status_code == 200
        data = response_1_2.json()
        assert data["output"][0]["content"][0]["text"] == "1"

        # Test second query by client 3 goes to the same underlying client 1 (not round robin since we've called it before)
        # We do this out of order as 1 -> 3 -> 2 instead of 1 -> 2 -> 3 to test any ordering effects.
        response_3_2 = client_3.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_3_2.status_code == 200
        data = response_3_2.json()
        assert data["output"][0]["content"][0]["text"] == "1"

        # Test second query by client 2 goes to the same underlying client 2
        response_2_2 = client_2.post(
            "/v1/responses",
            json=request_body.model_dump(exclude_unset=True, mode="json"),
        )
        assert response_2_2.status_code == 200
        data = response_2_2.json()
        assert data["output"][0]["content"][0]["text"] == "2"

    def test_responses_reasoning_parser(self, monkeypatch: MonkeyPatch):
        server = self._setup_server(monkeypatch)
        server.config.uses_reasoning_parser = True

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
        monkeypatch.setattr("responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID())

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
                "tool_calls": [],
                "reasoning_content": "First reasoning item",
            },
            {"content": [{"text": "cool", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "I'm still checking",
                "tool_calls": [],
            },
            {"content": [{"text": "ok", "type": "text"}], "role": "user"},
        ]
        actual_messages = mock_method.call_args.kwargs["messages"]
        assert expected_messages == actual_messages

        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages + data["output"],
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
                "tool_calls": [],
                "reasoning_content": "First reasoning item",
            },
            {"content": [{"text": "cool", "type": "text"}], "role": "user"},
            {
                "role": "assistant",
                "content": "I'm still checking",
                "tool_calls": [],
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
            },
        ]
        actual_messages = mock_method.call_args.kwargs["messages"]
        assert expected_messages == actual_messages


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
                    "tool_calls": [],
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

        monkeypatch.setattr("responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID())

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
        monkeypatch.setattr("responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID())
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
        monkeypatch.setattr("responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID())

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
