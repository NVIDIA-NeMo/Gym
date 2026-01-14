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
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from resources_servers.single_step_tool_use_with_judge.common.response_utils import (
    extract_canonical_text,
    extract_output_text_from_content,
    extract_output_text_from_response,
    extract_tool_call_or_output_message_from_response,
)


class TestResponseUtils:
    def _create_response(self, output_list: list[NeMoGymResponseOutputItem]) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test_response_utils",
            created_at=121.0,
            model="test_response_model",
            object="response",
            output=output_list,
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

    def test_extract_tool_call_or_output_message_from_response(self) -> None:
        assert extract_tool_call_or_output_message_from_response(self._create_response([])) is None

        reasoning_item = NeMoGymResponseReasoningItem(
            id="tool_call_message_reasoning_item",
            summary=[
                NeMoGymSummary(
                    type="summary_text",
                    text="tool call or message reasoning summary",
                )
            ],
        )
        assert extract_tool_call_or_output_message_from_response(self._create_response([reasoning_item])) is None

        first_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="the first text",
        )
        single_text_message = NeMoGymResponseOutputMessage(
            id="text_single",
            content=[first_output_text],
        )
        assert (
            extract_tool_call_or_output_message_from_response(self._create_response([single_text_message]))
            is single_text_message
        )
        assert (
            extract_tool_call_or_output_message_from_response(
                self._create_response(
                    [
                        reasoning_item,
                        single_text_message,
                    ]
                )
            )
            is single_text_message
        )

        second_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="the second text",
        )
        multiple_texts_message = NeMoGymResponseOutputMessage(
            id="texts_multiple",
            content=[
                second_output_text,
                first_output_text,
            ],
        )
        assert (
            extract_tool_call_or_output_message_from_response(self._create_response([multiple_texts_message]))
            is multiple_texts_message
        )
        assert (
            extract_tool_call_or_output_message_from_response(
                self._create_response(
                    [
                        reasoning_item,
                        multiple_texts_message,
                        single_text_message,
                    ]
                )
            )
            is multiple_texts_message
        )

        refusal_output_message = NeMoGymResponseOutputMessage(
            id="refusal_output_message",
            content=[
                NeMoGymResponseOutputRefusal(refusal="refusal description"),
            ],
        )
        assert (
            extract_tool_call_or_output_message_from_response(
                self._create_response(
                    [
                        reasoning_item,
                        refusal_output_message,
                        single_text_message,
                    ]
                )
            )
            is single_text_message
        )

        tool_call = NeMoGymResponseFunctionToolCall(
            call_id="tool_call_id",
            name="get_status",
            arguments='{"name": "test"}',
        )
        assert extract_tool_call_or_output_message_from_response(self._create_response([tool_call])) is tool_call
        assert (
            extract_tool_call_or_output_message_from_response(
                self._create_response(
                    [
                        single_text_message,
                        tool_call,
                    ]
                )
            )
            is tool_call
        )
        assert (
            extract_tool_call_or_output_message_from_response(
                self._create_response(
                    [
                        tool_call,
                        multiple_texts_message,
                    ]
                )
            )
            is tool_call
        )
        assert (
            extract_tool_call_or_output_message_from_response(
                self._create_response(
                    [
                        reasoning_item,
                        refusal_output_message,
                        single_text_message,
                        multiple_texts_message,
                        tool_call,
                    ]
                )
            )
            is tool_call
        )

    def test_extract_output_text_from_response(self) -> None:
        assert extract_output_text_from_response(self._create_response([])) is None

        reasoning_item = NeMoGymResponseReasoningItem(
            id="text_reasoning_item",
            summary=[
                NeMoGymSummary(
                    type="summary_text",
                    text="text reasoning summary",
                )
            ],
        )
        assert extract_output_text_from_response(self._create_response([reasoning_item])) is None

        refusal_output_message = NeMoGymResponseOutputMessage(
            id="refusal_message",
            content=[
                NeMoGymResponseOutputRefusal(refusal="explanation of refusal"),
            ],
        )
        assert (
            extract_output_text_from_response(
                self._create_response(
                    [
                        reasoning_item,
                        refusal_output_message,
                    ]
                )
            )
            is None
        )

        first_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="output text one",
        )
        single_text_message = NeMoGymResponseOutputMessage(
            id="single_text_message",
            content=[first_output_text],
        )
        assert extract_output_text_from_response(self._create_response([single_text_message])) is first_output_text
        assert (
            extract_output_text_from_response(
                self._create_response(
                    [
                        reasoning_item,
                        single_text_message,
                    ]
                )
            )
            is first_output_text
        )

        second_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="output text two",
        )
        multiple_texts_message = NeMoGymResponseOutputMessage(
            id="multiple_texts_message",
            content=[
                second_output_text,
                first_output_text,
            ],
        )
        assert extract_output_text_from_response(self._create_response([multiple_texts_message])) is second_output_text
        assert (
            extract_output_text_from_response(
                self._create_response(
                    [
                        single_text_message,
                        multiple_texts_message,
                    ]
                )
            )
            is first_output_text
        )
        assert (
            extract_output_text_from_response(
                self._create_response(
                    [
                        reasoning_item,
                        refusal_output_message,
                        multiple_texts_message,
                        single_text_message,
                    ]
                )
            )
            is second_output_text
        )

    def test_extract_output_text_from_content(self) -> None:
        assert extract_output_text_from_content([]) is None

        first_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="first output text",
        )
        assert extract_output_text_from_content([first_output_text]) is first_output_text

        second_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="second output text",
        )
        assert (
            extract_output_text_from_content(
                [
                    second_output_text,
                    first_output_text,
                ]
            )
            is second_output_text
        )

        output_refusal = NeMoGymResponseOutputRefusal(refusal="refusal reason")
        assert extract_output_text_from_content([output_refusal]) is None
        assert (
            extract_output_text_from_content(
                [
                    output_refusal,
                    second_output_text,
                ]
            )
            is second_output_text
        )
        assert (
            extract_output_text_from_content(
                [
                    first_output_text,
                    output_refusal,
                    second_output_text,
                ]
            )
            is first_output_text
        )

    def test_extract_canonical_text(self) -> None:
        assert extract_canonical_text("") == ""
        assert extract_canonical_text("this is some text") == "this is some text"
        assert extract_canonical_text("  here are words ") == "here are words"
        assert extract_canonical_text('```json\n{"key1": "value1"}') == '\n{"key1": "value1"}'
        assert extract_canonical_text('{"key2": "value2"}\n```') == '{"key2": "value2"}\n'
        assert extract_canonical_text('```json\n{"key3": "value3"}\n```') == '\n{"key3": "value3"}\n'
        assert extract_canonical_text(' ```json\n\n{"key4": "value4"}\n```\n') == '\n\n{"key4": "value4"}\n'
