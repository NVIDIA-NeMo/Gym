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
from resources_servers.single_step_tool_use_with_argument_comparison.common.response_utils import (
    extract_tool_call_or_text,
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

    def test_extract_tool_call_or_text(self) -> None:
        assert extract_tool_call_or_text(self._create_response([])) is None

        reasoning_item = NeMoGymResponseReasoningItem(
            id="reasoning_item",
            summary=[
                NeMoGymSummary(
                    type="summary_text",
                    text="this is reasoning text",
                )
            ],
        )
        assert extract_tool_call_or_text(self._create_response([reasoning_item])) is None

        first_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="this is the first output text",
        )
        single_text_message = NeMoGymResponseOutputMessage(
            id="single_text",
            content=[first_output_text],
        )
        assert extract_tool_call_or_text(self._create_response([single_text_message])) is first_output_text

        second_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="this is the second output text",
        )
        multiple_texts_message = NeMoGymResponseOutputMessage(
            id="multiple_texts",
            content=[
                second_output_text,
                first_output_text,
            ],
        )
        assert (
            extract_tool_call_or_text(
                self._create_response(
                    [
                        reasoning_item,
                        multiple_texts_message,
                        single_text_message,
                    ]
                )
            )
            is second_output_text
        )

        refusal_message = NeMoGymResponseOutputMessage(
            id="refusal",
            content=[
                NeMoGymResponseOutputRefusal(refusal="this is a refusal"),
            ],
        )
        assert (
            extract_tool_call_or_text(
                self._create_response(
                    [
                        reasoning_item,
                        refusal_message,
                        single_text_message,
                    ]
                )
            )
            is first_output_text
        )

        tool_call = NeMoGymResponseFunctionToolCall(
            call_id="tool_call",
            name="respond",
            arguments="",
        )
        assert extract_tool_call_or_text(self._create_response([tool_call])) is tool_call
        assert (
            extract_tool_call_or_text(
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
            extract_tool_call_or_text(
                self._create_response(
                    [
                        tool_call,
                        single_text_message,
                    ]
                )
            )
            is tool_call
        )
        assert (
            extract_tool_call_or_text(
                self._create_response(
                    [
                        reasoning_item,
                        single_text_message,
                        refusal_message,
                        tool_call,
                        multiple_texts_message,
                    ]
                )
            )
            is tool_call
        )
