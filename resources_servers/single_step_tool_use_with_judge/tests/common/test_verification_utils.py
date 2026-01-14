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
from typing import Any, Union
from unittest.mock import AsyncMock, MagicMock, call

from pytest import approx, fixture, raises

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.single_step_tool_use_with_judge.common.tool_utils import format_tool_definitions
from resources_servers.single_step_tool_use_with_judge.common.verification_utils import (
    Evaluation,
    ModelStepVerificationResult,
    ModelStepVerifier,
    ModelStepVerifierConfig,
    StepRewardCategory,
)


class TestModelStepVerifier:
    @fixture
    def verifier_config(self) -> ModelStepVerifierConfig:
        return ModelStepVerifierConfig(
            evaluation_model_server=ModelServerRef(
                type="responses_api_models",
                name="evaluation model",
            ),
            evaluation_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
            ),
            evaluation_generation_attempts=2,
        )

    def _create_response(
        self, first_output_item: NeMoGymResponseOutputItem, *additional_output_items: NeMoGymResponseOutputItem
    ) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test_response_id",
            created_at=301,
            model="evaluation_model",
            object="response",
            output=[first_output_item] + list(additional_output_items),
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

    def _set_server_client_post_responses(
        self,
        server_client_post_mock: AsyncMock,
        first_response: dict[str, Any],
        *additional_responses: dict[str, Any],
    ) -> None:
        server_client_post_mock.reset_mock(
            return_value=True,
            side_effect=True,
        )
        responses = [first_response]
        additional_responses_present = len(additional_responses) > 0
        if additional_responses_present:
            responses.extend(additional_responses)

        post_responses = []
        for response in responses:
            post_response_mock = AsyncMock()
            post_response_mock.json.return_value = response
            post_responses.append(post_response_mock)

        if additional_responses_present:
            server_client_post_mock.side_effect = post_responses
        else:
            server_client_post_mock.return_value = post_responses[0]

    def _compare_verification_results(
        self,
        actual_verification_result: ModelStepVerificationResult,
        expected_verification_result: ModelStepVerificationResult,
    ) -> None:
        assert actual_verification_result.reward == approx(expected_verification_result.reward)
        assert actual_verification_result.category == expected_verification_result.category
        assert actual_verification_result.explanation == expected_verification_result.explanation
        assert (
            actual_verification_result.evaluation_responses_create_params
            == expected_verification_result.evaluation_responses_create_params
        )
        assert (
            actual_verification_result.evaluation_response_list
            == expected_verification_result.evaluation_response_list
        )

    async def _invoke_method_and_compare_verification_result(
        self,
        model_step_verifier_config: ModelStepVerifierConfig,
        method_name: str,
        step: Union[NeMoGymResponseFunctionToolCall, NeMoGymResponseOutputMessage],
        evaluation: Evaluation,
        expected_formatted_step: str,
    ) -> None:
        server_client_post_mock = AsyncMock()
        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = server_client_post_mock
        agent_tools = [
            {
                "type": "function",
                "name": "reset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "state",
                    ],
                },
                "strict": True,
                "description": "Reset an object to the specified state.",
            },
            {
                "type": "function",
                "name": "initialize",
                "parameters": None,
                "strict": False,
                "description": "Initialize the system.",
            },
        ]
        verifier = ModelStepVerifier(
            server_client=server_client_mock,
            config=model_step_verifier_config,
            domain_policy="You are tasked with helping the user as an agent.",
            agent_tools=agent_tools,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
            ),
        )
        evaluation_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="evaluation_output_message_id",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=evaluation.model_dump_json(),
                    )
                ],
            )
        )
        self._set_server_client_post_responses(server_client_post_mock, evaluation_response.model_dump(mode="json"))

        method = getattr(verifier, method_name)
        verification_result = await method(step)
        evaluation_success = evaluation.success
        match step.type:
            case "function_call":
                if evaluation_success:
                    expected_category = StepRewardCategory.TOOL_CALL_SUCCESS
                else:
                    expected_category = StepRewardCategory.TOOL_CALL_FAILURE

            case "message":
                if evaluation_success:
                    expected_category = StepRewardCategory.CHAT_MESSAGE_SUCCESS
                else:
                    expected_category = StepRewardCategory.CHAT_MESSAGE_FAILURE

            case _:
                raise NotImplementedError

        system_message = ModelStepVerifier.SYSTEM_MESSAGE_TEMPLATE.format(
            user_tool_caller="",
            domain_policy="You are tasked with helping the user as an agent.",
            tool_definitions=format_tool_definitions(agent_tools),
            evaluation_schema=json.dumps(Evaluation.model_json_schema()),
        )
        evaluation_system_message = NeMoGymEasyInputMessage(
            role="system",
            content=system_message,
        )
        conversation_message = ModelStepVerifier.MESSAGE_CONVERSATION_TEMPLATE.format(
            previous_steps="",
            current_step=expected_formatted_step,
        )
        evaluation_conversation_message = NeMoGymEasyInputMessage(
            role="user",
            content=conversation_message,
        )
        evaluation_request = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                evaluation_system_message,
                evaluation_conversation_message,
            ]
        )
        expected_verification_result = ModelStepVerificationResult(
            reward=float(evaluation_success),
            category=expected_category,
            explanation=evaluation.explanation,
            evaluation_responses_create_params=evaluation_request,
            evaluation_response_list=[evaluation_response],
        )

        self._compare_verification_results(verification_result, expected_verification_result)
        server_client_post_mock.assert_called_once_with(
            server_name="evaluation model",
            url_path="/v1/responses",
            json=evaluation_request,
        )

    async def test_verify_step(self, verifier_config: ModelStepVerifierConfig) -> None:
        tool_call_arguments_string = json.dumps({"state": "start"})
        tool_call = NeMoGymResponseFunctionToolCall(
            call_id="verify_step_tool_call_id",
            name="reset",
            arguments=tool_call_arguments_string,
        )
        expected_formatted_tool_call = ModelStepVerifier.TOOL_CALL_MESSAGE_TEMPLATE.format(
            tool_caller_item="",
            execution_id="verify_step_tool_call_id",
            tool_name="reset",
            arguments=tool_call_arguments_string,
        )
        failure_evaluation = Evaluation(
            success=False,
            explanation="This tool call is a failure",
        )
        await self._invoke_method_and_compare_verification_result(
            verifier_config, "verify_step", tool_call, failure_evaluation, expected_formatted_tool_call
        )

        chat_message = NeMoGymResponseOutputMessage(
            id="verify_step_chat_message_id",
            content=[
                NeMoGymResponseOutputText(
                    annotations=[],
                    text="I am a customer service representative.",
                )
            ],
        )
        expected_formatted_chat_message = ModelStepVerifier.TEXT_MESSAGE_TEMPLATE.format(
            sender="Representative",
            content="I am a customer service representative.",
        )
        success_evaluation = Evaluation(
            success=True,
            explanation="This chat message is appropriate.",
        )
        await self._invoke_method_and_compare_verification_result(
            verifier_config, "verify_step", chat_message, success_evaluation, expected_formatted_chat_message
        )

    async def test_verify_tool_call(self, verifier_config: ModelStepVerifierConfig) -> None:
        server_client_post_mock = AsyncMock()
        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = server_client_post_mock
        agent_tools = [
            {
                "type": "function",
                "name": "absolute_value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer",
                        },
                    },
                    "required": ["number"],
                },
                "strict": False,
                "description": "Return the absolute value of a number.",
            },
            {
                "type": "function",
                "name": "add_values",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "integer",
                        },
                        "y": {
                            "type": "integer",
                        },
                    },
                    "required": [
                        "x",
                        "y",
                    ],
                },
                "strict": True,
                "description": "Add two numbers.",
            },
        ]
        verifier = ModelStepVerifier(
            server_client=server_client_mock,
            config=verifier_config,
            domain_policy="",
            agent_tools=agent_tools,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
            ),
        )

        tool_unknown_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="tool_unknown_id",
            name="add",
            arguments="",
        )
        tool_unknown_result = await verifier.verify_tool_call(tool_unknown_tool_call)
        expected_tool_unknown_result = ModelStepVerificationResult(
            reward=0.0,
            category=StepRewardCategory.UNKNOWN_TOOL,
            explanation="The tool add is unknown",
        )
        self._compare_verification_results(tool_unknown_result, expected_tool_unknown_result)
        server_client_post_mock.assert_not_called()

        invalid_encoded_arguments_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="invalid_encoded_arguments_id",
            name="absolute_value",
            arguments="three",
        )
        invalid_encoded_arguments_result = await verifier.verify_tool_call(invalid_encoded_arguments_tool_call)
        expected_invalid_encoded_arguments_result = ModelStepVerificationResult(
            reward=0.0,
            category=StepRewardCategory.ARGUMENTS_DECODE_ERROR,
            explanation=(
                "The arguments string three in the tool call with the ID "
                "invalid_encoded_arguments_id could not be converted to a JSON object: Expecting "
                "value"
            ),
        )
        self._compare_verification_results(invalid_encoded_arguments_result, expected_invalid_encoded_arguments_result)
        server_client_post_mock.assert_not_called()

        invalid_arguments_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="invalid_arguments_id",
            name="add_values",
            arguments=json.dumps(
                {"x": 3},
            ),
        )
        invalid_arguments_result = await verifier.verify_tool_call(invalid_arguments_tool_call)
        expected_invalid_arguments_result = ModelStepVerificationResult(
            reward=0.0,
            category=StepRewardCategory.ARGUMENTS_NOT_VALID_UNDER_SCHEMA,
            explanation=(
                "The arguments object {'x': 3} in the tool call with the ID invalid_arguments_id "
                "is not valid under the add_values tool parameters JSON Schema: 'y' is a "
                "required property"
            ),
        )
        self._compare_verification_results(invalid_arguments_result, expected_invalid_arguments_result)
        server_client_post_mock.assert_not_called()

        failure_arguments_string = json.dumps({"state": "fail"})
        failure_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="failure_id",
            name="reset",
            arguments=failure_arguments_string,
        )
        expected_formatted_failure_tool_call = ModelStepVerifier.TOOL_CALL_MESSAGE_TEMPLATE.format(
            tool_caller_item="",
            execution_id="failure_id",
            tool_name="reset",
            arguments=failure_arguments_string,
        )
        failure_evaluation = Evaluation(
            success=False,
            explanation="This tool call failed.",
        )
        await self._invoke_method_and_compare_verification_result(
            verifier_config,
            "verify_tool_call",
            failure_tool_call,
            failure_evaluation,
            expected_formatted_failure_tool_call,
        )

        success_arguments_string = json.dumps({"state": "succeed"})
        success_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="success_id",
            name="reset",
            arguments=success_arguments_string,
        )
        expected_formatted_success_tool_call = ModelStepVerifier.TOOL_CALL_MESSAGE_TEMPLATE.format(
            tool_caller_item="",
            execution_id="success_id",
            tool_name="reset",
            arguments=success_arguments_string,
        )
        success_evaluation = Evaluation(
            success=True,
            explanation="This tool call succeeded.",
        )
        await self._invoke_method_and_compare_verification_result(
            verifier_config,
            "verify_tool_call",
            success_tool_call,
            success_evaluation,
            expected_formatted_success_tool_call,
        )

        no_parameters_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="no_parameters_id",
            name="initialize",
            arguments="seven",
        )
        expected_formatted_no_parameters_tool_call = ModelStepVerifier.TOOL_CALL_MESSAGE_TEMPLATE.format(
            tool_caller_item="",
            execution_id="no_parameters_id",
            tool_name="initialize",
            arguments="seven",
        )
        await self._invoke_method_and_compare_verification_result(
            verifier_config,
            "verify_tool_call",
            no_parameters_tool_call,
            failure_evaluation,
            expected_formatted_no_parameters_tool_call,
        )

    async def test_verify_chat_message(self, verifier_config: ModelStepVerifierConfig) -> None:
        chat_message = NeMoGymResponseOutputMessage(
            id="verify_chat_message_id",
            content=[
                NeMoGymResponseOutputText(
                    annotations=[],
                    text="I am an agent who can help you.",
                )
            ],
        )
        expected_formatted_chat_message = ModelStepVerifier.TEXT_MESSAGE_TEMPLATE.format(
            sender="Representative",
            content="I am an agent who can help you.",
        )

        failure_evaluation = Evaluation(
            success=False,
            explanation="A failed chat message",
        )
        await self._invoke_method_and_compare_verification_result(
            verifier_config, "verify_chat_message", chat_message, failure_evaluation, expected_formatted_chat_message
        )

        success_evaluation = Evaluation(
            success=True,
            explanation="A successful chat message",
        )
        await self._invoke_method_and_compare_verification_result(
            verifier_config, "verify_chat_message", chat_message, success_evaluation, expected_formatted_chat_message
        )

    async def test_generate_verification_result(self, verifier_config: ModelStepVerifierConfig) -> None:
        server_client_post_mock = AsyncMock()
        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = server_client_post_mock
        agent_tools = [
            {
                "type": "function",
                "name": "set_value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                        },
                        "value": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "field",
                        "value",
                    ],
                },
                "strict": False,
                "description": "Set the value of a field",
            },
            {
                "type": "function",
                "name": "get_value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                        },
                    },
                    "required": ["field"],
                },
                "strict": True,
                "description": "Get the value of a field.",
            },
        ]
        previous_messages = []
        verifier = ModelStepVerifier(
            server_client=server_client_mock,
            config=verifier_config,
            domain_policy="Your job is to help the user.",
            agent_tools=agent_tools,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=previous_messages,
            ),
        )

        invalid_response_body = {
            "id": "invalid_response_id",
            "object": "response",
        }
        self._set_server_client_post_responses(server_client_post_mock, invalid_response_body)
        first_agent_message = NeMoGymResponseOutputMessage(
            id="first_agent_message_id",
            content=[
                NeMoGymResponseOutputText(
                    annotations=[],
                    text="How can I help you?",
                )
            ],
        )
        with raises(RuntimeError, match="Received an invalid response from the evaluation model server: "):
            await verifier._generate_verification_result(first_agent_message)

        system_message = ModelStepVerifier.SYSTEM_MESSAGE_TEMPLATE.format(
            user_tool_caller="",
            domain_policy="Your job is to help the user.",
            tool_definitions=format_tool_definitions(agent_tools),
            evaluation_schema=json.dumps(Evaluation.model_json_schema()),
        )
        evaluation_system_message = NeMoGymEasyInputMessage(
            role="system",
            content=system_message,
        )
        first_agent_message_step = ModelStepVerifier.TEXT_MESSAGE_TEMPLATE.format(
            sender="Representative",
            content="How can I help you?",
        )
        first_agent_conversation_message = f"""<previous_steps>

</previous_steps>

<current_step>
{first_agent_message_step}
</current_step>"""
        evaluation_first_agent_message = NeMoGymEasyInputMessage(
            role="user",
            content=first_agent_conversation_message,
        )
        server_client_post_mock.assert_called_once_with(
            server_name="evaluation model",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    evaluation_system_message,
                    evaluation_first_agent_message,
                ]
            ),
        )

        previous_messages.append(
            NeMoGymEasyInputMessage(
                role="system",
                content="This is the system message in the conversation.",
            )
        )
        previous_messages.append(
            NeMoGymEasyInputMessage(
                role="user",
                content="Could you set the status field to in_progress?",
            )
        )
        verifier.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input=previous_messages)
        second_agent_tool_call_argument_string = json.dumps(
            {
                "field": "status",
                "value": "in_progress",
            }
        )
        second_agent_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="set_status_id",
            name="set_value",
            arguments=second_agent_tool_call_argument_string,
        )
        no_output_text_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="no_output_text_id",
                content=[
                    NeMoGymResponseOutputRefusal(refusal="This is an evaluation refusal"),
                ],
            )
        )
        invalid_evaluation_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="invalid_evaluation_id",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=json.dumps(
                            {"success": True},
                        ),
                    )
                ],
            )
        )
        self._set_server_client_post_responses(
            server_client_post_mock,
            no_output_text_response.model_dump(mode="json"),
            invalid_evaluation_response.model_dump(mode="json"),
        )
        no_valid_evaluation_result = await verifier._generate_verification_result(second_agent_tool_call)
        first_user_message_step = ModelStepVerifier.TEXT_MESSAGE_TEMPLATE.format(
            sender="Customer",
            content="Could you set the status field to in_progress?",
        )
        second_agent_tool_call_step = ModelStepVerifier.TOOL_CALL_MESSAGE_TEMPLATE.format(
            tool_caller_item="",
            execution_id="set_status_id",
            tool_name="set_value",
            arguments=second_agent_tool_call_argument_string,
        )
        second_agent_conversation_message = f"""<previous_steps>
{first_user_message_step}
</previous_steps>

<current_step>
{second_agent_tool_call_step}
</current_step>"""
        evaluation_second_agent_message = NeMoGymEasyInputMessage(
            role="user",
            content=second_agent_conversation_message,
        )
        second_agent_evaluation_request = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                evaluation_system_message,
                evaluation_second_agent_message,
            ]
        )
        expected_no_valid_evaluation_result = ModelStepVerificationResult(
            reward=0.0,
            category=StepRewardCategory.EVALUATION_GENERATION_ERROR,
            explanation="A valid evaluation response was not generated by the verification model in 2 attempts",
            evaluation_responses_create_params=second_agent_evaluation_request,
            evaluation_response_list=[
                no_output_text_response,
                invalid_evaluation_response,
            ],
        )

        self._compare_verification_results(no_valid_evaluation_result, expected_no_valid_evaluation_result)
        second_agent_evaluation_call = call(
            server_name="evaluation model",
            url_path="/v1/responses",
            json=second_agent_evaluation_request,
        )
        assert server_client_post_mock.call_args_list == [
            second_agent_evaluation_call,
            second_agent_evaluation_call,
        ]

        previous_messages.append(second_agent_tool_call)
        previous_messages.append(
            NeMoGymFunctionCallOutput(
                call_id="set_status_id",
                output="value set",
            )
        )
        verifier.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input=previous_messages)
        third_agent_message = NeMoGymResponseOutputMessage(
            id="third_agent_message_id",
            content=[
                NeMoGymResponseOutputText(
                    annotations=[],
                    text="The status field was successfully set.",
                )
            ],
        )
        chat_message_failure_evaluation = Evaluation(
            success=False,
            explanation="This is a failed chat message.",
        )
        chat_message_failure_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="chat_message_failure_id",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=chat_message_failure_evaluation.model_dump_json(),
                    )
                ],
            )
        )
        self._set_server_client_post_responses(
            server_client_post_mock, chat_message_failure_response.model_dump(mode="json")
        )
        chat_message_failure_result = await verifier._generate_verification_result(third_agent_message)
        first_tool_execution_step = ModelStepVerifier.TOOL_EXECUTION_MESSAGE_TEMPLATE.format(
            execution_id="set_status_id",
            execution_result="value set",
        )
        third_agent_message_step = ModelStepVerifier.TEXT_MESSAGE_TEMPLATE.format(
            sender="Representative",
            content="The status field was successfully set.",
        )
        third_agent_conversation_message = f"""<previous_steps>
{first_user_message_step}

{second_agent_tool_call_step}

{first_tool_execution_step}
</previous_steps>

<current_step>
{third_agent_message_step}
</current_step>"""
        evaluation_third_agent_message = NeMoGymEasyInputMessage(
            role="user",
            content=third_agent_conversation_message,
        )
        third_agent_evaluation_request = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                evaluation_system_message,
                evaluation_third_agent_message,
            ]
        )
        expected_chat_message_failure_result = ModelStepVerificationResult(
            reward=0.0,
            category=StepRewardCategory.CHAT_MESSAGE_FAILURE,
            explanation="This is a failed chat message.",
            evaluation_responses_create_params=third_agent_evaluation_request,
            evaluation_response_list=[chat_message_failure_response],
        )

        self._compare_verification_results(chat_message_failure_result, expected_chat_message_failure_result)
        server_client_post_mock.assert_called_once_with(
            server_name="evaluation model",
            url_path="/v1/responses",
            json=third_agent_evaluation_request,
        )

        chat_message_success_evaluation = Evaluation(
            success=True,
            explanation="This is a successful chat message.",
        )
        chat_message_success_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="chat_message_success_id",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=f"```json\n{chat_message_success_evaluation.model_dump_json()}\n```",
                    )
                ],
            )
        )
        self._set_server_client_post_responses(
            server_client_post_mock, chat_message_success_response.model_dump(mode="json")
        )
        chat_message_success_result = await verifier._generate_verification_result(third_agent_message)
        expected_chat_message_success_result = ModelStepVerificationResult(
            reward=1.0,
            category=StepRewardCategory.CHAT_MESSAGE_SUCCESS,
            explanation="This is a successful chat message.",
            evaluation_responses_create_params=third_agent_evaluation_request,
            evaluation_response_list=[chat_message_success_response],
        )

        self._compare_verification_results(chat_message_success_result, expected_chat_message_success_result)
        server_client_post_mock.assert_called_once_with(
            server_name="evaluation model",
            url_path="/v1/responses",
            json=third_agent_evaluation_request,
        )

        previous_messages.append(third_agent_message)
        previous_messages.append(
            NeMoGymEasyInputMessage(role="user", content="Please get the value of the status field.")
        )
        verifier.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input=previous_messages)
        fourth_agent_tool_call_argument_string = json.dumps({"field": "status"})
        fourth_agent_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="get_status_id",
            name="get_value",
            arguments=fourth_agent_tool_call_argument_string,
        )
        tool_call_failure_evaluation = Evaluation(
            success=False,
            explanation="This is a tool call failure",
        )
        tool_call_failure_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="tool_call_failure_id",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=f"  ```json\n{tool_call_failure_evaluation.model_dump_json()}\n``` ",
                    )
                ],
            )
        )
        self._set_server_client_post_responses(
            server_client_post_mock, tool_call_failure_response.model_dump(mode="json")
        )
        tool_call_failure_result = await verifier._generate_verification_result(fourth_agent_tool_call)
        second_user_message_step = ModelStepVerifier.TEXT_MESSAGE_TEMPLATE.format(
            sender="Customer",
            content="Please get the value of the status field.",
        )
        fourth_agent_tool_call_step = ModelStepVerifier.TOOL_CALL_MESSAGE_TEMPLATE.format(
            tool_caller_item="",
            execution_id="get_status_id",
            tool_name="get_value",
            arguments=fourth_agent_tool_call_argument_string,
        )
        fourth_agent_conversation_message = f"""<previous_steps>
{first_user_message_step}

{second_agent_tool_call_step}

{first_tool_execution_step}

{third_agent_message_step}

{second_user_message_step}
</previous_steps>

<current_step>
{fourth_agent_tool_call_step}
</current_step>"""
        evaluation_fourth_agent_message = NeMoGymEasyInputMessage(
            role="user",
            content=fourth_agent_conversation_message,
        )
        fourth_agent_evaluation_request = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                evaluation_system_message,
                evaluation_fourth_agent_message,
            ]
        )
        expected_tool_call_failure_result = ModelStepVerificationResult(
            reward=0.0,
            category=StepRewardCategory.TOOL_CALL_FAILURE,
            explanation="This is a tool call failure",
            evaluation_responses_create_params=fourth_agent_evaluation_request,
            evaluation_response_list=[tool_call_failure_response],
        )

        self._compare_verification_results(tool_call_failure_result, expected_tool_call_failure_result)
        server_client_post_mock.assert_called_once_with(
            server_name="evaluation model",
            url_path="/v1/responses",
            json=fourth_agent_evaluation_request,
        )

        tool_call_success_evaluation = Evaluation(
            success=True,
            explanation="The tool call is successful.",
        )
        tool_call_success_response = self._create_response(
            NeMoGymResponseOutputMessage(
                id="tool_call_success_id",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=f" {tool_call_success_evaluation.model_dump_json()} ",
                    )
                ],
            )
        )
        self._set_server_client_post_responses(
            server_client_post_mock, tool_call_success_response.model_dump(mode="json")
        )
        tool_call_success_result = await verifier._generate_verification_result(fourth_agent_tool_call)
        expected_tool_call_success_result = ModelStepVerificationResult(
            reward=1.0,
            category=StepRewardCategory.TOOL_CALL_SUCCESS,
            explanation="The tool call is successful.",
            evaluation_responses_create_params=fourth_agent_evaluation_request,
            evaluation_response_list=[tool_call_success_response],
        )

        self._compare_verification_results(tool_call_success_result, expected_tool_call_success_result)
        server_client_post_mock.assert_called_once_with(
            server_name="evaluation model",
            url_path="/v1/responses",
            json=fourth_agent_evaluation_request,
        )

    def test_format_step(self) -> None:
        assert (
            ModelStepVerifier._format_step(
                NeMoGymResponseReasoningItem(
                    id="reasoning_item",
                    summary=[
                        NeMoGymSummary(
                            type="summary_text",
                            text="reasoning summary",
                        )
                    ],
                )
            )
            is None
        )
        assert (
            ModelStepVerifier._format_step(
                NeMoGymEasyInputMessage(
                    role="system",
                    content="This is a system message.",
                )
            )
            is None
        )

        assert (
            ModelStepVerifier._format_step(
                NeMoGymEasyInputMessage(
                    role="user",
                    content="This is a user message.",
                )
            )
            == """<message>
Sender: Customer
Content: This is a user message.
</message>"""
        )
        assert (
            ModelStepVerifier._format_step(
                NeMoGymResponseOutputMessage(
                    id="chat_message",
                    content=[
                        NeMoGymResponseOutputText(
                            annotations=[],
                            text="This is an assistant message",
                        )
                    ],
                )
            )
            == """<message>
Sender: Representative
Content: This is an assistant message
</message>"""
        )

        assert (
            ModelStepVerifier._format_step(
                NeMoGymResponseOutputMessage(
                    id="chat_message",
                    content=[
                        NeMoGymResponseOutputRefusal(refusal="This is a refusal explanation"),
                    ],
                )
            )
            is None
        )

        assert (
            ModelStepVerifier._format_step(
                NeMoGymResponseFunctionToolCall(
                    call_id="tool_call_id",
                    name="show_status",
                    arguments='{"channel": "communication"}',
                )
            )
            == """<execute_tool>
Execution ID: tool_call_id
Tool name: show_status
Arguments for execution: {"channel": "communication"}
</execute_tool>"""
        )

        assert (
            ModelStepVerifier._format_step(
                NeMoGymFunctionCallOutput(
                    call_id="function_call_output_id",
                    output="IN PROGRESS",
                )
            )
            == """<tool_result>
Execution ID: function_call_output_id
Execution result: IN PROGRESS
</tool_result>"""
        )
