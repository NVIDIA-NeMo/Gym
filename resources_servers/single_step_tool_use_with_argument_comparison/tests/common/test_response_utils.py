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
from resources_servers.single_step_tool_use_with_argument_comparison.common.response_utils import extract_action
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    FunctionCallAction,
    FunctionCallBatchAction,
    MessageAction,
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

    def test_extract_action_finds_nothing_to_judge(self) -> None:
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

        refusal_message = NeMoGymResponseOutputMessage(
            id="refusal",
            content=[
                NeMoGymResponseOutputRefusal(refusal="this is a refusal"),
            ],
        )
        assert extract_action(self._create_response([reasoning_item, refusal_message])) is None

    def test_extract_action_takes_the_first_assistant_text(self) -> None:
        first_output_text = NeMoGymResponseOutputText(
            annotations=[],
            text="this is the first output text",
        )
        single_text_message = NeMoGymResponseOutputMessage(
            id="single_text",
            content=[first_output_text],
        )
        assert extract_action(self._create_response([single_text_message])) == MessageAction(
            type="message", content="this is the first output text"
        )

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
        assert extract_action(self._create_response([multiple_texts_message, single_text_message])) == MessageAction(
            type="message", content="this is the second output text"
        )

        # A refusal carries no output text, so the next message supplies the assistant text.
        refusal_message = NeMoGymResponseOutputMessage(
            id="refusal",
            content=[
                NeMoGymResponseOutputRefusal(refusal="this is a refusal"),
            ],
        )
        assert extract_action(self._create_response([refusal_message, single_text_message])) == MessageAction(
            type="message", content="this is the first output text"
        )

    def test_extract_action_prefers_tool_calls_over_text(self) -> None:
        tool_call = NeMoGymResponseFunctionToolCall(
            call_id="tool_call",
            name="respond",
            arguments="",
        )
        expected_action = FunctionCallAction(type="function_call", name="respond", arguments="")
        text_message = NeMoGymResponseOutputMessage(
            id="single_text",
            content=[NeMoGymResponseOutputText(annotations=[], text="this is the output text")],
        )

        assert extract_action(self._create_response([tool_call])) == expected_action
        assert extract_action(self._create_response([text_message, tool_call])) == expected_action
        assert extract_action(self._create_response([tool_call, text_message])) == expected_action

    def test_extract_action_batches_parallel_tool_calls_in_order(self) -> None:
        first_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="first_tool_call",
            name="respond",
            arguments="",
        )
        second_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="second_tool_call",
            name="lookup",
            arguments='{"query": "alpha"}',
        )
        text_message = NeMoGymResponseOutputMessage(
            id="single_text",
            content=[NeMoGymResponseOutputText(annotations=[], text="this is the output text")],
        )

        assert extract_action(
            self._create_response([text_message, first_tool_call, second_tool_call])
        ) == FunctionCallBatchAction(
            type="function_call_batch",
            calls=[
                FunctionCallAction(type="function_call", name="respond", arguments=""),
                FunctionCallAction(type="function_call", name="lookup", arguments='{"query": "alpha"}'),
            ],
        )
