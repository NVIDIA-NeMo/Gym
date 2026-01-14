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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.single_step_tool_use_with_judge.app import (
    SingleStepToolUseJudgeResourcesServer,
    SingleStepToolUseJudgeResourcesServerConfig,
    SingleStepToolUseJudgeVerifyRequest,
)
from resources_servers.single_step_tool_use_with_judge.common.verification_utils import (
    Evaluation,
    ModelStepVerifierConfig,
    StepRewardCategory,
)


class TestApp:
    @fixture
    def resources_server_config(self) -> SingleStepToolUseJudgeResourcesServerConfig:
        verifier_config = ModelStepVerifierConfig(
            evaluation_model_server=ModelServerRef(
                type="responses_api_models",
                name="agent_evaluation_model",
            ),
            evaluation_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
            ),
            evaluation_generation_attempts=10,
        )
        return SingleStepToolUseJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=21001,
            entrypoint="",
            name="tool_judge_server",
            model_step_verifier_config=verifier_config,
        )

    def _create_response(self, output_item: NeMoGymResponseOutputItem) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test_app_response_id",
            created_at=405,
            model="agent_evaluation_model",
            object="response",
            output=[output_item],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

    def _set_server_client_post_response(self, server_client_post_mock: AsyncMock, response: dict[str, Any]) -> None:
        server_client_post_mock.reset_mock(return_value=True)
        post_response_mock = AsyncMock()
        post_response_mock.json.return_value = response
        server_client_post_mock.return_value = post_response_mock

    async def test_verify(self, resources_server_config: SingleStepToolUseJudgeResourcesServerConfig) -> None:
        server_client_post_mock = AsyncMock()
        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = server_client_post_mock
        resources_server = SingleStepToolUseJudgeResourcesServer(
            config=resources_server_config,
            server_client=server_client_mock,
        )

        tools = [
            {
                "type": "function",
                "name": "check_state",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "object_name",
                    ],
                },
                "strict": True,
                "description": "Check the state of the object with the specified name.",
            },
        ]
        chat_message_responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    role="user",
                    content="Who are you?",
                )
            ],
            tools=tools,
        )

        refusal_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="refusal_message_id",
                content=[
                    NeMoGymResponseOutputRefusal(refusal="I won't tell you."),
                ],
            )
        )
        refusal_verify_request = SingleStepToolUseJudgeVerifyRequest(
            responses_create_params=chat_message_responses_create_params,
            response=refusal_response,
            domain_policy="Your task is to refuse the user's request.",
        )
        refusal_verify_response = await resources_server.verify(refusal_verify_request)
        assert refusal_verify_response.responses_create_params == chat_message_responses_create_params
        assert refusal_verify_response.response == refusal_response
        assert refusal_verify_response.reward == approx(0.0)
        assert refusal_verify_response.category == StepRewardCategory.NO_ACTION_FOUND
        assert refusal_verify_response.explanation is None
        assert refusal_verify_response.evaluation_responses_create_params is None
        assert refusal_verify_response.evaluation_response_list is None
        server_client_post_mock.assert_not_called()

        failure_evaluation = Evaluation(
            success=False,
            explanation="The agent's chat message is a failure.",
        )
        failure_evaluation_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="failure_evaluation_message_id",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=failure_evaluation.model_dump_json(),
                    )
                ],
            )
        )
        self._set_server_client_post_response(
            server_client_post_mock, failure_evaluation_response.model_dump(mode="json")
        )
        chat_message_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="chat_message_id",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text="I am an agent who is here to help you.",
                    )
                ],
            )
        )
        chat_message_verify_request = SingleStepToolUseJudgeVerifyRequest(
            responses_create_params=chat_message_responses_create_params,
            response=chat_message_response,
            domain_policy="Your task is to respond to the user's request.",
        )
        chat_message_verify_response = await resources_server.verify(chat_message_verify_request)
        assert chat_message_verify_response.responses_create_params == chat_message_responses_create_params
        assert chat_message_verify_response.response == chat_message_response
        assert chat_message_verify_response.reward == approx(0.0)
        assert chat_message_verify_response.category == StepRewardCategory.CHAT_MESSAGE_FAILURE
        assert chat_message_verify_response.explanation == "The agent's chat message is a failure."
        assert chat_message_verify_response.evaluation_responses_create_params is not None
        assert chat_message_verify_response.evaluation_response_list == [failure_evaluation_response]
        server_client_post_mock.assert_called_once()

        tool_call_responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    role="user",
                    content="Could you check the state of the view object?",
                )
            ],
            tools=tools,
        )
        tool_call_response = self._create_response(
            NeMoGymResponseFunctionToolCall(
                call_id="tool_call_id",
                name="check_state",
                arguments=json.dumps(
                    {"object_name": "view"},
                ),
            )
        )
        success_evaluation = Evaluation(
            success=True,
            explanation="The agent's tool call is a success.",
        )
        success_evaluation_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="success_evaluation_message_id",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=success_evaluation.model_dump_json(),
                    )
                ],
            )
        )
        self._set_server_client_post_response(
            server_client_post_mock, success_evaluation_response.model_dump(mode="json")
        )
        tool_call_verify_request = SingleStepToolUseJudgeVerifyRequest(
            responses_create_params=tool_call_responses_create_params,
            response=tool_call_response,
            domain_policy="Your task is to call tools to address the user's request.",
        )
        tool_call_verify_response = await resources_server.verify(tool_call_verify_request)
        assert tool_call_verify_response.responses_create_params == tool_call_responses_create_params
        assert tool_call_verify_response.response == tool_call_response
        assert tool_call_verify_response.reward == approx(1.0)
        assert tool_call_verify_response.category == StepRewardCategory.TOOL_CALL_SUCCESS
        assert tool_call_verify_response.explanation == "The agent's tool call is a success."
        assert tool_call_verify_response.evaluation_responses_create_params is not None
        assert tool_call_verify_response.evaluation_response_list == [success_evaluation_response]
        server_client_post_mock.assert_called_once()
