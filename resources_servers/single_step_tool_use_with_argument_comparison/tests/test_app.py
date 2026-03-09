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
from unittest.mock import MagicMock

from openai.types.responses import FunctionTool
from pytest import approx, fixture

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.single_step_tool_use_with_argument_comparison.app import (
    SingleStepToolUseArgumentComparisonResourcesServer,
    SingleStepToolUseArgumentComparisonResourcesServerConfig,
    SingleStepToolUseArgumentComparisonVerifyRequest,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedAction,
    ExpectedFunctionCall,
    ExpectedMessage,
    StepRewardCategory,
    ToolCallComparatorConfig,
)


class TestApp:
    @fixture
    def resources_server(self) -> SingleStepToolUseArgumentComparisonResourcesServer:
        tool_call_comparator_config = ToolCallComparatorConfig(word_count_similarity_threshold=0.1)
        resources_server_config = SingleStepToolUseArgumentComparisonResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="tool_argument_comparison_server",
            tool_call_comparator_config=tool_call_comparator_config,
        )
        return SingleStepToolUseArgumentComparisonResourcesServer(
            config=resources_server_config,
            server_client=MagicMock(spec=ServerClient),
        )

    async def _verify_and_compare_response(
        self,
        resources_server: SingleStepToolUseArgumentComparisonResourcesServer,
        responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
        tool: FunctionTool,
        expected_action: ExpectedAction,
        response_id: str,
        output_item: NeMoGymResponseOutputItem,
        expected_reward: float,
        expected_reward_category: StepRewardCategory,
    ) -> None:
        response = NeMoGymResponse(
            id=response_id,
            created_at=1001,
            model="test_model",
            object="response",
            output=[output_item],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[tool],
        )
        verify_request = SingleStepToolUseArgumentComparisonVerifyRequest(
            responses_create_params=responses_create_params,
            response=response,
            expected_action=expected_action,
        )
        verify_response = await resources_server.verify(verify_request)
        assert verify_response.responses_create_params == responses_create_params
        assert verify_response.response == response
        assert verify_response.expected_action == expected_action
        assert verify_response.reward == approx(expected_reward)
        assert verify_response.category == expected_reward_category

    async def test_verify(self, resources_server: SingleStepToolUseArgumentComparisonResourcesServer) -> None:
        tool = FunctionTool(
            type="function",
            name="set_metric_count",
            parameters={
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                    },
                    "metric_count": {
                        "type": "integer",
                    },
                },
                "required": [
                    "metric_name",
                    "metric_count",
                ],
            },
        )
        tool_param = tool.model_dump()
        tool_call_responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    role="user",
                    content="Set the views metric count to 75.",
                )
            ],
            tools=[tool_param],
        )

        expected_arguments = {
            "metric_name": "views",
            "metric_count": 75,
        }
        expected_arguments_string = json.dumps(expected_arguments)
        expected_tool_call = ExpectedFunctionCall(
            type="function_call",
            name="set_metric_count",
            arguments=expected_arguments_string,
        )

        reasoning_item = NeMoGymResponseReasoningItem(
            id="reasoning_item",
            summary=[
                NeMoGymSummary(
                    type="summary_text",
                    text="this is reasoning",
                )
            ],
        )
        await self._verify_and_compare_response(
            resources_server,
            tool_call_responses_create_params,
            tool,
            expected_tool_call,
            "no_output",
            reasoning_item,
            0.0,
            StepRewardCategory.NO_ACTION_FOUND,
        )

        different_arguments = {
            "metric_name": "views",
            "metric_count": "75",
        }
        different_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="different_value",
            name="set_metric_count",
            arguments=json.dumps(different_arguments),
        )
        await self._verify_and_compare_response(
            resources_server,
            tool_call_responses_create_params,
            tool,
            expected_tool_call,
            "different_arguments",
            different_tool_call,
            0.0,
            StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT,
        )

        matching_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="matching_arguments",
            name="set_metric_count",
            arguments=expected_arguments_string,
        )
        await self._verify_and_compare_response(
            resources_server,
            tool_call_responses_create_params,
            tool,
            expected_tool_call,
            "matching_tool_call",
            matching_tool_call,
            1.0,
            StepRewardCategory.EXPECTED_TOOL_CALL,
        )

        chat_message = NeMoGymResponseOutputMessage(
            id="chat_message",
            content=[
                NeMoGymResponseOutputText(
                    annotations=[],
                    text="How can I help you?",
                )
            ],
        )
        await self._verify_and_compare_response(
            resources_server,
            tool_call_responses_create_params,
            tool,
            expected_tool_call,
            "chat_message_instead_of_tool",
            chat_message,
            0.0,
            StepRewardCategory.NO_EXPECTED_TOOL_CALL,
        )

        chat_message_responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    role="user",
                    content="This is a greeting.",
                )
            ],
            tools=[tool_param],
        )
        expected_message = ExpectedMessage(
            type="message",
            content="This is a message.",
        )

        await self._verify_and_compare_response(
            resources_server,
            chat_message_responses_create_params,
            tool,
            expected_message,
            "different_chat_message",
            chat_message,
            1.0,
            StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND,
        )
        await self._verify_and_compare_response(
            resources_server,
            chat_message_responses_create_params,
            tool,
            expected_message,
            "tool_call_instead_of_chat_message",
            matching_tool_call,
            0.0,
            StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE,
        )
