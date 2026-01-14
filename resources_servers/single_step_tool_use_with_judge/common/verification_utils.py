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
from enum import StrEnum
from json import JSONDecodeError
from typing import ClassVar, Optional, Union

import jsonschema.validators
from jsonschema.exceptions import ValidationError as JSONValidationError
from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, Field
from pydantic import ValidationError as ModelValidationError

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputItem,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient, raise_for_status
from resources_servers.single_step_tool_use_with_judge.common.response_utils import (
    extract_canonical_text,
    extract_output_text_from_content,
    extract_output_text_from_response,
)
from resources_servers.single_step_tool_use_with_judge.common.tool_utils import format_tool_definitions


class StepRewardCategory(StrEnum):
    NO_ACTION_FOUND = "No tool call or chat message was found in the response"
    UNKNOWN_TOOL = "An unknown tool was executed in a tool call"
    ARGUMENTS_DECODE_ERROR = "An error occurred when decoding the arguments string in a tool call as a JSON object"
    ARGUMENTS_NOT_VALID_UNDER_SCHEMA = (
        "The arguments in a tool call are not valid under the tool parameters JSON schema"
    )
    EVALUATION_GENERATION_ERROR = "An error occurred when generating a response from the verification model"
    TOOL_CALL_SUCCESS = "A tool call was evaluated as a success by the verification model"
    TOOL_CALL_FAILURE = "A tool call was evaluated as a failure by the verification model"
    CHAT_MESSAGE_SUCCESS = "A chat message was evaluated as a success by the verification model"
    CHAT_MESSAGE_FAILURE = "A chat message was evaluated as a failure by the verification model"


class ModelStepVerifierConfig(BaseModel):
    evaluation_model_server: ModelServerRef
    evaluation_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    evaluation_generation_attempts: int


class ModelStepVerificationResult(BaseModel):
    reward: float
    category: StepRewardCategory
    explanation: Optional[str] = None
    evaluation_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    evaluation_response_list: Optional[list[NeMoGymResponse]] = None


class Evaluation(BaseModel):
    success: bool = Field(description="True for an evaluation of success, or false for an evaluation of failure.")
    explanation: str = Field(description="An explanation for the evaluation.")


class ModelStepVerifier(BaseModel):
    SYSTEM_MESSAGE_PREFIX: ClassVar[str] = """# Instructions
- Your task is to evaluate one step in an interaction between a customer and a customer service representative.
- The previous steps in the interaction are shown in order between the <previous_steps> and </previous_steps> tags.
- The step that you are to evaluate is the current step, which is shown between the <current_step> and </current_step> tags.
- Each step in the interaction is either a message from the customer to the representative, a message from the representative to the customer, the execution of a tool, or the result of executing a tool.
- A message from the customer to the representative or from the representative to the customer is shown between <message> and </message> tags.
- An execution of a tool can be requested by {user_tool_caller}the representative, and is shown between <execute_tool> and </execute_tool> tags.
- When {user_tool_caller}the representative requests the execution of a tool, the next step in the interaction should be the result of executing the tool, which is shown between <tool_result> and </tool_result> tags.
- The way that the representative handles customer requests is described below between the <policy> and </policy> tags.

<policy>
{domain_policy}
</policy>

"""

    SYSTEM_MESSAGE_TEMPLATE: ClassVar[str] = (
        SYSTEM_MESSAGE_PREFIX
        + """- The definitions of the tools that are available to the representative for execution are provided below between the <tools> and </tools> tags, with each tool definition appearing between <tool> and </tool> tags.

<tools>
{tool_definitions}
</tools>

- The current step that you are to evaluate is an action by the representative.
- The representative can either send a message to the customer, or execute a tool.
- The current step should be considered a success if the action taken by the representative is helpful in addressing the request from the customer.
- The current step should be considered a failure if the action taken by the representative does not help to handle the customer's request, or is not consistent with the previous steps in the interaction or the policy for handling customer requests.
- If the current step is the execution of a tool, then the current step should be considered a success if the tool to be executed helps to address the customer's request, the arguments for the tool execution conform to the tool definition, and the arguments are consistent with the previous steps in the conversation.  Otherwise, the current step should be considered a failure.

Please output the evaluation of the current step and an explanation for the evaluation using the following JSON schema:
{evaluation_schema}"""
    )

    TEXT_MESSAGE_TEMPLATE: ClassVar[str] = """<message>
Sender: {sender}
Content: {content}
</message>"""

    TOOL_CALL_MESSAGE_TEMPLATE: ClassVar[str] = """<execute_tool>
{tool_caller_item}Execution ID: {execution_id}
Tool name: {tool_name}
Arguments for execution: {arguments}
</execute_tool>"""

    TOOL_EXECUTION_MESSAGE_TEMPLATE: ClassVar[str] = """<tool_result>
Execution ID: {execution_id}
Execution result: {execution_result}
</tool_result>"""

    MESSAGE_CONVERSATION_TEMPLATE: ClassVar[str] = """<previous_steps>
{previous_steps}
</previous_steps>

<current_step>
{current_step}
</current_step>"""

    server_client: ServerClient
    config: ModelStepVerifierConfig
    domain_policy: str
    agent_tools: list[FunctionToolParam]
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    async def verify_step(
        self, step: Union[NeMoGymResponseFunctionToolCall, NeMoGymResponseOutputMessage]
    ) -> ModelStepVerificationResult:
        match step.type:
            case "function_call":
                return await self.verify_tool_call(step)

            case "message":
                return await self.verify_chat_message(step)

            case _:
                raise NotImplementedError

    async def verify_tool_call(self, tool_call: NeMoGymResponseFunctionToolCall) -> ModelStepVerificationResult:
        tool_name = tool_call.name
        executed_tools = [tool for tool in self.agent_tools if tool["name"] == tool_name]
        if len(executed_tools) < 1:
            return ModelStepVerificationResult(
                reward=0.0,
                category=StepRewardCategory.UNKNOWN_TOOL,
                explanation=f"The tool {tool_name} is unknown",
            )

        executed_tool = executed_tools[0]
        parameters = executed_tool.get("parameters")
        if parameters is not None:
            validator_class = jsonschema.validators.validator_for(parameters)
            validator_class.check_schema(parameters)
            validator = validator_class(parameters)
            arguments = tool_call.arguments

            try:
                deserialized_arguments = json.loads(arguments)
                validator.validate(deserialized_arguments)
            except JSONDecodeError as decode_error:
                decode_error_message = (
                    f"The arguments string {arguments} in the tool call with the ID {tool_call.call_id} could not be "
                    f"converted to a JSON object: {decode_error.msg}"
                )
                return ModelStepVerificationResult(
                    reward=0.0,
                    category=StepRewardCategory.ARGUMENTS_DECODE_ERROR,
                    explanation=decode_error_message,
                )
            except JSONValidationError as validation_error:
                validation_error_message = (
                    f"The arguments object {deserialized_arguments} in the tool call with the ID {tool_call.call_id} "
                    f"is not valid under the {tool_name} tool parameters JSON Schema: {validation_error.message}"
                )
                return ModelStepVerificationResult(
                    reward=0.0,
                    category=StepRewardCategory.ARGUMENTS_NOT_VALID_UNDER_SCHEMA,
                    explanation=validation_error_message,
                )

        return await self._generate_verification_result(tool_call)

    async def verify_chat_message(self, chat_message: NeMoGymResponseOutputMessage) -> ModelStepVerificationResult:
        return await self._generate_verification_result(chat_message)

    async def _generate_verification_result(
        self, current_step: Union[NeMoGymResponseFunctionToolCall, NeMoGymResponseOutputMessage]
    ) -> ModelStepVerificationResult:
        tool_definitions_string = format_tool_definitions(self.agent_tools)
        evaluation_schema_string = json.dumps(Evaluation.model_json_schema())
        system_message = self.SYSTEM_MESSAGE_TEMPLATE.format(
            user_tool_caller="",
            domain_policy=self.domain_policy,
            tool_definitions=tool_definitions_string,
            evaluation_schema=evaluation_schema_string,
        )

        formatted_steps = []
        for input_item in self.responses_create_params.input:
            formatted_step = self._format_step(input_item)
            if formatted_step is not None:
                formatted_steps.append(formatted_step)

        previous_steps_string = "\n\n".join(formatted_steps)
        current_step_string = self._format_step(current_step)
        if current_step_string is None:
            current_step_string = ""

        conversation_string = self.MESSAGE_CONVERSATION_TEMPLATE.format(
            previous_steps=previous_steps_string,
            current_step=current_step_string,
        )
        evaluation_inputs = [
            NeMoGymEasyInputMessage(
                role="system",
                content=system_message,
            ),
            NeMoGymEasyInputMessage(
                role="user",
                content=conversation_string,
            ),
        ]

        server_client = self.server_client
        config = self.config
        evaluation_responses_create_params = config.evaluation_responses_create_params.model_copy(
            update={
                "input": evaluation_inputs,
            }
        )
        generation_attempts = config.evaluation_generation_attempts

        evaluation_response_list = []
        for _ in range(generation_attempts):
            evaluation_server_response = await server_client.post(
                server_name=config.evaluation_model_server.name,
                url_path="/v1/responses",
                json=evaluation_responses_create_params,
            )
            await raise_for_status(evaluation_server_response)
            evaluation_response_body = await evaluation_server_response.json()

            try:
                evaluation_response = NeMoGymResponse.model_validate(evaluation_response_body)
            except ModelValidationError as validation_exception:
                raise RuntimeError(
                    "Received an invalid response from the evaluation model server: "
                    f"{json.dumps(evaluation_response_body)}"
                ) from validation_exception

            evaluation_response_list.append(evaluation_response)
            evaluation_output_text = extract_output_text_from_response(evaluation_response)
            if evaluation_output_text is None:
                continue

            evaluation_text = extract_canonical_text(evaluation_output_text.text)
            try:
                evaluation = Evaluation.model_validate_json(evaluation_text)
            except ModelValidationError:
                continue

            if current_step.type == "function_call":
                if evaluation.success:
                    reward = 1.0
                    category = StepRewardCategory.TOOL_CALL_SUCCESS
                else:
                    reward = 0.0
                    category = StepRewardCategory.TOOL_CALL_FAILURE
            else:
                if evaluation.success:
                    reward = 1.0
                    category = StepRewardCategory.CHAT_MESSAGE_SUCCESS
                else:
                    reward = 0.0
                    category = StepRewardCategory.CHAT_MESSAGE_FAILURE

            return ModelStepVerificationResult(
                reward=reward,
                category=category,
                explanation=evaluation.explanation,
                evaluation_responses_create_params=evaluation_responses_create_params,
                evaluation_response_list=evaluation_response_list,
            )

        return ModelStepVerificationResult(
            reward=0.0,
            category=StepRewardCategory.EVALUATION_GENERATION_ERROR,
            explanation=(
                f"A valid evaluation response was not generated by the verification model in {generation_attempts} "
                f"attempts"
            ),
            evaluation_responses_create_params=evaluation_responses_create_params,
            evaluation_response_list=evaluation_response_list,
        )

    @classmethod
    def _format_step(cls, step: NeMoGymResponseInputItem) -> Optional[str]:
        match step.type:
            case "message":
                role = step.role
                match role:
                    case "user":
                        message_source_label = "Customer"
                        message_text = step.content

                    case "assistant":
                        agent_output_text = extract_output_text_from_content(step.content)
                        if agent_output_text is None:
                            return None

                        message_source_label = "Representative"
                        message_text = agent_output_text.text

                    case _:
                        return None

                return cls.TEXT_MESSAGE_TEMPLATE.format(
                    sender=message_source_label,
                    content=message_text,
                )

            case "function_call":
                return cls.TOOL_CALL_MESSAGE_TEMPLATE.format(
                    tool_caller_item="",
                    execution_id=step.call_id,
                    tool_name=step.name,
                    arguments=step.arguments,
                )

            case "function_call_output":
                return cls.TOOL_EXECUTION_MESSAGE_TEMPLATE.format(
                    execution_id=step.call_id,
                    execution_result=step.output,
                )

            case _:
                return None
