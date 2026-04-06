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
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from resources_servers.terminal_multi_harness.common.response_utils import (
    extract_action,
    extract_action_from_backend_response,
    extract_action_from_responses_api_response,
)


class TestResponseUtils:
    def _create_response(self, output_list: list[NeMoGymResponseOutputItem]) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test_response",
            created_at=101.0,
            model="test_model",
            object="response",
            output=output_list,
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

    def test_extract_action(self) -> None:
        assert extract_action(self._create_response([])) is None

        reasoning_item = NeMoGymResponseReasoningItem(
            id="reasoning_item",
            summary=[
                NeMoGymSummary(
                    type="summary_text",
                    text="this is reasoning text",
                )
            ],
        )
        assert extract_action(self._create_response([reasoning_item])) is None

        single_text_message = NeMoGymResponseOutputMessage(
            id="single_text",
            content=[
                NeMoGymResponseOutputText(
                    annotations=[],
                    text="this is the first output text",
                )
            ],
        )
        extracted_message = extract_action(self._create_response([single_text_message]))
        assert extracted_message is not None
        assert extracted_message.type == "message"
        assert extracted_message.content == "this is the first output text"

        first_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="call_1",
            name="exec_command",
            arguments='{"cmd": "pwd"}',
        )
        extracted_tool_call = extract_action(self._create_response([first_tool_call]))
        assert extracted_tool_call is not None
        assert extracted_tool_call.type == "function_call"
        assert extracted_tool_call.name == "exec_command"

        second_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="call_2",
            name="exec_command",
            arguments='{"cmd": "git status --short"}',
        )
        extracted_batch = extract_action(self._create_response([first_tool_call, second_tool_call]))
        assert extracted_batch is not None
        assert extracted_batch.type == "function_call_batch"
        assert [call.name for call in extracted_batch.calls] == ["exec_command", "exec_command"]

        extracted_prefer_tool = extract_action(
            self._create_response(
                [
                    single_text_message,
                    first_tool_call,
                ]
            )
        )
        assert extracted_prefer_tool is not None
        assert extracted_prefer_tool.type == "function_call"

    def test_extract_action_from_responses_api_response(self) -> None:
        extracted_message = extract_action_from_responses_api_response(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "hello from responses",
                            }
                        ],
                    }
                ]
            }
        )
        assert extracted_message is not None
        assert extracted_message.type == "message"
        assert extracted_message.content == "hello from responses"

        extracted_batch = extract_action_from_responses_api_response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": '{"cmd": "pwd"}',
                    },
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": '{"cmd": "git status --short"}',
                    },
                ]
            }
        )
        assert extracted_batch is not None
        assert extracted_batch.type == "function_call_batch"
        assert [call.arguments for call in extracted_batch.calls] == [
            '{"cmd": "pwd"}',
            '{"cmd": "git status --short"}',
        ]

    def test_extract_action_from_backend_response(self) -> None:
        extracted_tool = extract_action_from_backend_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I will inspect two files next.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": '{"cmd": "cat /tmp/a"}',
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": '{"cmd": "cat /tmp/b"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        assert extracted_tool is not None
        assert extracted_tool.type == "function_call_batch"
        assert [call.arguments for call in extracted_tool.calls] == [
            '{"cmd": "cat /tmp/a"}',
            '{"cmd": "cat /tmp/b"}',
        ]

        extracted_message = extract_action_from_backend_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        assert extracted_message is not None
        assert extracted_message.type == "message"
        assert extracted_message.content == "done"

    def test_extract_action_from_backend_response_matches_responses_payload(self) -> None:
        backend_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd": "cat /app/legacy_sys/netware/segment_A.log"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd": "cat /app/legacy_sys/rotated/segment_B.log"}',
                                },
                            },
                        ],
                        "reasoning": "Let me examine the content of each log file to understand their formats.\n",
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        responses_api_response = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd": "cat /app/legacy_sys/netware/segment_A.log"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "exec_command",
                    "arguments": '{"cmd": "cat /app/legacy_sys/rotated/segment_B.log"}',
                    "status": "completed",
                },
            ]
        }

        backend_action = extract_action_from_backend_response(backend_response)
        responses_action = extract_action_from_responses_api_response(responses_api_response)
        assert backend_action is not None
        assert responses_action is not None
        assert backend_action.model_dump() == responses_action.model_dump()
