from pytest import MonkeyPatch, mark
from unittest.mock import AsyncMock, MagicMock
from typing import Union, Any
from fastapi.testclient import TestClient
from responses_api_models.vllm_model.app import (
    VLLMModel,
    VLLMModelConfig,
    VLLMConverter,
)

from nemo_gym.server_utils import ServerClient
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseInputParam,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseReasoningItemParam,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTextParam,
    NeMoGymResponseInputTextParam,
    NeMoGymResponseFunctionToolCallParam,
    NeMoGymResponseOutputMessageParam,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymChatCompletionMessageToolCallParam,
    NeMoGymChatCompletionAssistantMessageParam,
    NeMoGymChatCompletionMessageToolCall,
    NeMoGymChatCompletionToolMessageParam,
    NeMoGymChatCompletionContentPartTextParam,
    NeMoGymChoice,
    NeMoGymFunction,
    NeMoGymMessage,
    NeMoGymSummary,
    NeMoGymFunctionCall,
    NeMoGymEasyInputMessageParam,
    NeMoGymFunctionCallOutput,
    NeMoGymFunctionToolParam,
)

# Used for mocking created_at timestamp generation
FIXED_TIME = 1691418000
FIXED_UUID = "123"


class FakeUUID:
    """Used for mocking UUIDs"""

    hex = FIXED_UUID


COMMON_RESPONSE_PARAMS = dict(
    background=None,
    instructions=None,
    max_output_tokens=None,
    max_tool_calls=None,
    metadata=None,
    parallel_tool_calls=True,
    previous_response_id=None,
    prompt=None,
    reasoning=None,
    service_tier=None,
    temperature=None,
    text={},
    tool_choice="auto",
    top_p=None,
    top_logprobs=None,
    truncation=None,
    user="",
)

PARAMETERIZE_DATA = [
    # ----- EasyInputMessageParam: content as a list, id: "ez_list" -----
    (
        [
            NeMoGymEasyInputMessageParam(
                role="user",
                content=[{"type": "input_text", "text": "hello"}],
                type="message",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS,
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
                    message=NeMoGymChatCompletionMessage(
                        role="assistant", content="hi :) how are you?"
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
            NeMoGymEasyInputMessageParam(
                role="user",
                content="hello",
                type="message",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS,
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
                    message=NeMoGymChatCompletionMessage(
                        role="assistant", content="hi :) how are you?"
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
            **COMMON_RESPONSE_PARAMS,
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
                    message=NeMoGymChatCompletionMessage(
                        role="assistant", content="hi :) how are you?"
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
            **COMMON_RESPONSE_PARAMS,
            messages=[
                NeMoGymChatCompletionUserMessageParam(
                    content=[
                        NeMoGymChatCompletionContentPartTextParam(
                            type="text", text="hello"
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
                    message=NeMoGymChatCompletionMessage(
                        role="assistant", content="hi :) how are you?"
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
            NeMoGymResponseFunctionToolCallParam(
                arguments='{"city":"San Francisco"}',
                call_id="call_123",
                name="get_weather",
                type="function_call",
                id="func_123",
                status="completed",
            )
        ],
        NeMoGymChatCompletionCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS,
            messages=[
                NeMoGymChatCompletionAssistantMessageParam(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        NeMoGymChatCompletionMessageToolCallParam(
                            id="call_123",
                            type="function",
                            function=NeMoGymFunction(
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
                        function_call=NeMoGymFunctionCall(
                            arguments='{"city":"San Francisco"}',
                            name="get_weather",
                        ),
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
            **COMMON_RESPONSE_PARAMS,
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
            NeMoGymResponseReasoningItemParam(
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
            **COMMON_RESPONSE_PARAMS,
            messages=[
                NeMoGymChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=[
                        NeMoGymChatCompletionContentPartTextParam(
                            type="text",
                            text="<think>I have identified the city as San Francisco based on user input.</think>",
                        )
                    ],
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
            NeMoGymResponseReasoningItemParam(
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
            **COMMON_RESPONSE_PARAMS,
            messages=[
                NeMoGymChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=[
                        NeMoGymChatCompletionContentPartTextParam(
                            type="text",
                            text="<think>I'll first think about the user's question.</think><think>Then I will answer.</think>",
                        )
                    ],
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
            NeMoGymResponseOutputMessageParam(
                id="msg_123",
                role="assistant",
                content=[
                    NeMoGymResponseOutputTextParam(
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
            **COMMON_RESPONSE_PARAMS,
            messages=[
                NeMoGymChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=[
                        NeMoGymChatCompletionContentPartTextParam(
                            type="text",
                            text="Hello! How can I assist you today?",
                        )
                    ],
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
    def _setup_server(self):
        config = VLLMModelConfig(
            host="0.0.0.0",
            port=8081,
            base_url="http://api.openai.com/v1",
            api_key="dummy_key",
            model="dummy_model",
            entrypoint="",
        )
        return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_sanity(self) -> None:
        self._setup_server()

    def test_responses_multistep(self, monkeypatch: MonkeyPatch):
        server = self._setup_server()
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
            NeMoGymEasyInputMessageParam(
                type="message",
                role="user",
                content=[
                    NeMoGymResponseInputTextParam(
                        text="Check my order status", type="input_text"
                    )
                ],
                status="completed",
            ),
            NeMoGymEasyInputMessageParam(
                type="message",
                role="assistant",
                content=[
                    NeMoGymResponseInputTextParam(
                        text="Sure, one sec.", type="input_text"
                    )
                ],
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

        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.time", lambda: FIXED_TIME
        )
        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID()
        )

        request_body = NeMoGymResponseCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS,
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
        sent_tools = called_args[0].tools

        assert [(i.role, i.content[0]["text"]) for i in input_messages] == [
            (i["role"], i["content"][0]["text"]) for i in called_args[0].messages
        ]

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
        server = self._setup_server()
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
                content=[
                    NeMoGymResponseInputTextParam(text="Hello", type="input_text")
                ],
                status="completed",
            ),
            NeMoGymResponseReasoningItemParam(
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
            NeMoGymResponseOutputMessageParam(
                id="msg_123",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    NeMoGymResponseOutputTextParam(
                        type="output_text", text="Hi, how can I help?", annotations=[]
                    ),
                ],
            ),
            NeMoGymMessage(
                type="message",
                role="user",
                content=[
                    NeMoGymResponseInputTextParam(
                        type="input_text", text="What's the weather?"
                    )
                ],
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
        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.time", lambda: FIXED_TIME
        )
        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID()
        )

        request_body = NeMoGymResponseCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS,
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
        sent_messages = called_args[0].messages

        expected_sent_messages = [
            {"content": [{"text": "Hello", "type": "text"}], "role": "user"},
            {
                "content": [
                    {
                        "type": "text",
                        "text": "<think>Considering ways to greet the user...</think>Hi, how can I help?",
                    }
                ],
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
        server = self._setup_server()
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
                content=[
                    NeMoGymResponseInputTextParam(
                        text="Hi, can you check my order?", type="input_text"
                    )
                ],
                status="completed",
            ),
            NeMoGymResponseReasoningItemParam(
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
            NeMoGymResponseOutputMessageParam(
                id="msg_123",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    NeMoGymResponseOutputTextParam(
                        text="Sure, one sec.", type="output_text", annotations=[]
                    )
                ],
            ),
            NeMoGymResponseOutputMessageParam(
                id="msg_123",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    NeMoGymResponseOutputTextParam(
                        text="Gathering order status and delivery info..",
                        type="output_text",
                        annotations=[],
                    )
                ],
            ),
            NeMoGymResponseFunctionToolCallParam(
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
            NeMoGymResponseFunctionToolCallParam(
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
            NeMoGymResponseOutputMessageParam(
                id="msg_123",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    NeMoGymResponseOutputTextParam(
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
                    NeMoGymResponseInputTextParam(
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
        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.time", lambda: FIXED_TIME
        )
        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID()
        )

        request_body = NeMoGymResponseCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS,
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
        sent_messages = called_args[0].messages
        sent_tools = called_args[0].tools

        expected_sent_messages = [
            {
                "content": [{"text": "Hi, can you check my order?", "type": "text"}],
                "role": "user",
            },
            {
                "content": [
                    {
                        "type": "text",
                        "text": "<think>Checking order details...</think>Sure, one sec.",
                    }
                ],
                "role": "assistant",
                "tool_calls": [],
            },
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Gathering order status and delivery info..",
                    }
                ],
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
                "content": [
                    {
                        "type": "text",
                        "text": "Order #1234 is shipped and arrives tomorrow.",
                    }
                ],
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
        single_input: Union[str, NeMoGymResponseInputParam],
        _,
        mock_chat_completion: NeMoGymChatCompletion,
        expected_response: NeMoGymResponse,
    ):
        """
        Test entire pipeline from api endpoint -> final output:
        Response Create Params -> Response
        """
        server = self._setup_server()
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID()
        )
        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.time", lambda: FIXED_TIME
        )

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS, input=single_input
        )

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
        single_input: Union[str, NeMoGymResponseInputParam],
        expected_chat_completion_create_params: NeMoGymChatCompletionCreateParamsNonStreaming,
        _,
        __,
    ):
        """
        Tests conversion from api endpoint -> internal request schema
        Response Params -> Chat Completion Params
        """
        server = self._setup_server()
        app = server.setup_webserver()
        client = TestClient(app)

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS, input=single_input
        )

        captured_params: dict[str, Any] = {}

        # Returning this dummy response allows us to call /responses vs.
        # server._converter.responses_to_chat_completion_create_params() directly
        async def _mock_and_capture(self, create_params):
            captured_params["value"] = create_params
            return NeMoGymChatCompletion(
                id="chtcmpl-123",
                choices=[
                    NeMoGymChoice(
                        index=0,
                        finish_reason="stop",
                        message=NeMoGymChatCompletionMessage(
                            role="assistant", content="some response"
                        ),
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


class TestVLLMConverter:
    def setup_method(self, _):
        self.converter = VLLMConverter()

    def test_responses_input_types_EasyInputMessageParam(self) -> None:
        """
        Tests the conversion of ResponseCreateParams to ChatCompletionCreateParams
        """

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS,
            input=[
                # ----- Baseline -----
                NeMoGymEasyInputMessageParam(
                    content="my content",
                    role="user",
                    type="message",
                ),
                # ----- Ablate `content` -----
                NeMoGymEasyInputMessageParam(
                    content=[
                        NeMoGymResponseInputTextParam(
                            type="input_text",
                            text="my content 1",
                        ),
                        NeMoGymResponseInputTextParam(
                            type="input_text",
                            text="my content 2",
                        ),
                    ],
                    role="user",
                    type="message",
                ),
                # ----- Ablate `role` -----
                NeMoGymEasyInputMessageParam(
                    content=[
                        NeMoGymResponseInputTextParam(
                            text="assistant content", type="input_text"
                        )
                    ],
                    role="assistant",
                    type="message",
                ),
                NeMoGymEasyInputMessageParam(
                    content=[
                        NeMoGymResponseInputTextParam(
                            text="system content", type="input_text"
                        )
                    ],
                    role="system",
                    type="message",
                ),
                NeMoGymEasyInputMessageParam(
                    content=[
                        NeMoGymResponseInputTextParam(
                            text="developer content", type="input_text"
                        )
                    ],
                    role="developer",
                    type="message",
                ),
                # ----- Ablate `type` -----
                NeMoGymEasyInputMessageParam(
                    content=[
                        NeMoGymResponseInputTextParam(
                            text="user content", type="input_text"
                        )
                    ],
                    role="user",
                    # type (omitted)
                ),
            ],
        )

        expected_chat_completion_create_params = (
            NeMoGymChatCompletionCreateParamsNonStreaming(
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
                        "content": [{"type": "text", "text": "assistant content"}],
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
        )
        actual_chat_completion_create_params = (
            self.converter.responses_to_chat_completion_create_params(
                responses_create_params
            )
        )
        assert (
            expected_chat_completion_create_params.messages
            == actual_chat_completion_create_params.messages
        )

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

        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID()
        )

        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.time", lambda: FIXED_TIME
        )

        choice = mock_chat_completion.choices[0]

        processed_output = self.converter.postprocess_chat_response(choice)

        assert processed_output == expected_response.output

    def test_extract_reasoning_from_content(self):
        # Single reasoning block
        content_single = "This is some main content.<think>Here is the reasoning.</think>More content."
        reasoning_single, main_content_single = (
            self.converter._extract_reasoning_from_content(content_single)
        )
        assert reasoning_single == ["Here is the reasoning."]
        assert main_content_single == "This is some main content.More content."

        # Multiple reasoning blocks
        content_multiple = "First part.<think>Thought 1.</think>Second part.<think>Thought 2.</think>Final part."
        reasoning_multiple, main_content_multiple = (
            self.converter._extract_reasoning_from_content(content_multiple)
        )
        assert reasoning_multiple == ["Thought 1.", "Thought 2."]
        assert main_content_multiple == "First part.Second part.Final part."

        # No reasoning
        content_none = "Just plain content here."
        reasoning_none, main_content_none = (
            self.converter._extract_reasoning_from_content(content_none)
        )
        assert reasoning_none == []
        assert main_content_none == "Just plain content here."

    def test_postprocess_chat_response_multiple_reasoning_items(
        self, monkeypatch: MonkeyPatch
    ):
        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID()
        )
        monkeypatch.setattr(
            "responses_api_models.vllm_model.app.time", lambda: FIXED_TIME
        )

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
                    NeMoGymSummary(
                        text="I need to check the user's order ID.", type="summary_text"
                    ),
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
        single_input: Union[str, NeMoGymResponseInputParam],
        expected_chat_completion_create_params: NeMoGymChatCompletionCreateParamsNonStreaming,
        _,
        __,
    ):
        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            **COMMON_RESPONSE_PARAMS, input=single_input
        )

        actual_chat_completion_create_params = (
            self.converter.responses_to_chat_completion_create_params(
                responses_create_params
            )
        )

        assert (
            actual_chat_completion_create_params.messages
            == expected_chat_completion_create_params.messages
        )
