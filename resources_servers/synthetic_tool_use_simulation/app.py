# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import json
import re
import textwrap
from enum import StrEnum
from json import JSONDecodeError
from typing import Any, ClassVar, Dict, List, Literal, Optional

import jsonschema.validators
from aiohttp import ClientConnectionError, ClientResponseError
from fastapi import FastAPI, HTTPException, Request
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status


TRAJECTORY_COMPLETE_INDICATOR = "###STOP###"
AGENT_TRANSFER_INDICATOR = "###TRANSFER###"


class JudgeProviderError(Exception):
    pass


class JudgeProviderExhaustedError(JudgeProviderError):
    pass


class JudgeProviderRequestError(JudgeProviderError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class MessageType(StrEnum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_EXECUTION = "tool_execution"


class Source(StrEnum):
    USER = "user"
    AGENT = "agent"
    ENVIRONMENT = "environment"


class TrajectoryEvaluationType(StrEnum):
    AGENT_CONVERSATION = "agent_conversation"


class VerificationType(StrEnum):
    MESSAGE = "message"
    COMPLETE_TRAJECTORY_COMBINED_EVALUATION = "complete_trajectory_combined_evaluation"


class TrajectoryInvalidReason(StrEnum):
    MESSAGE_GENERATION_ERROR = "message_generation_error"
    INVALID_USER_MESSAGE = "invalid_user_message"
    INVALID_AGENT_TOOL_CALL = "invalid_agent_tool_call"
    INVALID_ENVIRONMENT_TOOL_EXECUTION = "invalid_environment_tool_execution"
    EXCESSIVE_LENGTH = "excessive_length"
    VERIFICATION_GENERATION_ERROR = "verification_generation_error"
    NO_REWARD_USER_MESSAGE = "no_reward_user_message"
    NO_REWARD_ENVIRONMENT_MESSAGE = "no_reward_environment_message"


class VerificationFailureLabel(StrEnum):
    USER_FAILURE = "user_failure"
    AGENT_FAILURE = "agent_failure"
    TOOL_FAILURE = "tool_failure"
    VERIFICATION_GENERATION_ERROR = "verification_generation_error"
    TRAJECTORY_FAILURE = "trajectory_failure"


def _default_responses_create_params(temperature: float = 1.0) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(input=[], temperature=temperature, parallel_tool_calls=False)


class SyntheticToolUseSimulationConfig(BaseResourcesServerConfig):
    simulator_model_server: Optional[ModelServerRef] = None
    user_model_server: Optional[ModelServerRef] = None
    tool_simulator_model_server: Optional[ModelServerRef] = None
    judge_model_server: Optional[ModelServerRef] = None

    user_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: _default_responses_create_params(temperature=1.0)
    )
    tool_simulator_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: _default_responses_create_params(temperature=1.0)
    )
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: _default_responses_create_params(temperature=1.0)
    )
    generation_attempts: int = Field(default=3, ge=1)
    judge_provider_attempts: int = Field(default=3, ge=1)
    judge_provider_retry_initial_backoff_seconds: float = Field(default=0.5, ge=0.0)
    judge_provider_retry_max_backoff_seconds: float = Field(default=8.0, ge=0.0)
    enable_llm_judge: bool = True
    enable_termination: bool = True
    verification_type: VerificationType = VerificationType.MESSAGE
    enforce_transfer_ground_truth: bool = False


class CustomerScenario(BaseModel):
    model_config = ConfigDict(extra="allow")

    customer_persona: str = ""
    reason_for_contact: str = ""
    customer_details: str = ""
    unknown_info: Optional[str] = None
    task_instructions: str = ""
    representative_domain: Optional[str] = None
    outside_policy_scope: Optional[bool] = None

    def create_string(self, include_domain: bool = False) -> str:
        lines = []
        lines.append("Customer persona:")
        lines.append(textwrap.indent(self.customer_persona, "\t"))
        if include_domain:
            lines.append(f"Domain of representative: {self.representative_domain}")
        lines.append("Reason for contacting representative:")
        lines.append(textwrap.indent(self.reason_for_contact, "\t"))
        lines.append("Customer details:")
        lines.append(textwrap.indent(self.customer_details, "\t"))
        if self.unknown_info is not None:
            lines.append("Unknown info:")
            lines.append(textwrap.indent(self.unknown_info, "\t"))
        lines.append("Task instructions:")
        lines.append(textwrap.indent(self.task_instructions, "\t"))
        return "\n".join(lines)


class ToolSignature(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    returns: Optional[Dict[str, Any]] = None
    strict: bool = True
    doc: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

    @property
    def normalized_doc(self) -> str:
        return self.doc if self.doc is not None else (self.description or "")

    @property
    def normalized_params(self) -> Optional[Dict[str, Any]]:
        if self.params is not None:
            return self.params
        return self.parameters

    @property
    def model_visible_params(self) -> Dict[str, Any]:
        return self.normalized_params or {"type": "object", "properties": {}}


class Evaluation(BaseModel):
    success: bool = Field(description="True for an evaluation of success, or false for an evaluation of failure.")
    explanation: str = Field(description="An explanation for the evaluation.")


class UserAgentEnvironmentEvaluation(BaseModel):
    customer_success: bool = Field(
        description=(
            "True for an evaluation of success for the actions of the customer, or false for an evaluation of failure."
        )
    )
    customer_explanation: str = Field(description="An explanation for the evaluation of the customer.")
    representative_success: bool = Field(
        description=(
            "True for an evaluation of success for the actions of the customer "
            "service representative, or false for an evaluation of failure."
        )
    )
    representative_explanation: str = Field(
        description=("An explanation for the evaluation of the customer service representative.")
    )
    tool_results_success: bool = Field(
        description=("True for an evaluation of success for the tool results, or false for an evaluation of failure.")
    )
    tool_results_explanation: str = Field(description="An explanation for the evaluation of the tool results.")

    model_config = ConfigDict(title="Evaluation")


class VerificationResult(BaseModel):
    reward: Optional[int] = None
    explanation: Optional[str] = None
    judge_response: Optional[str] = None
    generation_error: Optional[str] = None
    responses: Optional[List[Dict[str, Any]]] = None


class AgentVerificationResult(BaseModel):
    conversation_verification_result: Optional[VerificationResult] = None
    overall_reward: Optional[int] = None
    trajectory_invalid_reasons: Optional[List[str]] = None


class SyntheticMessage(BaseModel):
    type: Literal[MessageType.TEXT, MessageType.TOOL_CALL, MessageType.TOOL_EXECUTION] = MessageType.TEXT
    source: Literal[Source.USER, Source.AGENT, Source.ENVIRONMENT]
    responses: Optional[List[Dict[str, Any]]] = None
    content: Any = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    arguments: Optional[str] = None
    deserialized_arguments: Optional[Any] = None
    deserialized_execution_result: Optional[Any] = None
    schema_valid: Optional[bool] = None
    verification_result: Optional[VerificationResult] = None


class SyntheticSessionState(BaseModel):
    domain_name: str
    policy: str
    tool_signatures: List[ToolSignature]
    customer_scenario: CustomerScenario
    messages: List[SyntheticMessage] = Field(default_factory=list)
    prefill_message_count: int = 0
    source_artifacts: Dict[str, Any] = Field(default_factory=dict)
    invalid_reasons: List[str] = Field(default_factory=list)
    failure_labels: List[str] = Field(default_factory=list)
    terminal_state: Optional[str] = None
    generation_invalid_reason: Optional[str] = None
    terminal_error: Optional[str] = None
    agent_verification_result: Optional[AgentVerificationResult] = None
    user_verification_result: Optional[VerificationResult] = None
    environment_verification_result: Optional[VerificationResult] = None
    verification_reward: Optional[float] = None
    judge_generation_error: Optional[str] = None
    judge_diagnostics: Dict[str, Any] = Field(default_factory=dict)


class SyntheticToolUseSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    domain_name: str = ""
    policy: str
    tools: List[ToolSignature] = Field(default_factory=list)
    customer_scenario: CustomerScenario = Field(default_factory=CustomerScenario)
    responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    initial_user_message: Optional[str] = None
    source_artifacts: Dict[str, Any] = Field(default_factory=dict)


class SessionToolsResponse(BaseModel):
    tools: List[Dict[str, Any]]


class PendingToolCall(BaseModel):
    tool_name: str
    tool_call_id: str
    arguments: str


class SessionResumeResponse(BaseModel):
    next_actor: Literal["user", "agent", "environment", "terminal"]
    pending_tool_calls: List[PendingToolCall] = Field(default_factory=list)
    is_initial: bool = False
    terminal_state: Optional[str] = None


class RecordAgentMessageRequest(BaseModel):
    content: str
    response: Optional[Dict[str, Any]] = None


class RecordAgentMessageResponse(BaseModel):
    should_continue: bool
    terminal_state: Optional[str] = None


class RecordGenerationErrorRequest(BaseModel):
    source: Source
    error: str


class RecordGenerationErrorResponse(BaseModel):
    should_continue: bool = False
    terminal_state: Optional[str] = None


class AgentToolCallAtStepLimit(BaseModel):
    tool_name: str
    tool_call_id: str
    arguments: str


class RecordAgentStepLimitRequest(BaseModel):
    max_agent_steps: int = Field(ge=1)
    tool_calls: List[AgentToolCallAtStepLimit] = Field(default_factory=list)
    response: Optional[Dict[str, Any]] = None


class RecordAgentStepLimitResponse(BaseModel):
    should_continue: bool = False
    terminal_state: Optional[str] = None


class NextUserMessageResponse(BaseModel):
    message: str
    should_continue: bool = True
    terminal_state: Optional[str] = None


class SyntheticToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class SyntheticToolCallResponse(BaseModel):
    output: Any
    schema_valid: bool
    error: Optional[str] = None
    should_continue: bool = True
    terminal_state: Optional[str] = None


class AgentToolCallRequest(BaseModel):
    tool_name: str
    tool_call_id: str
    arguments: str
    response: Optional[Dict[str, Any]] = None


class SyntheticToolUseVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SyntheticToolUseVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    instance_config: Dict[str, Any] = Field(default_factory=lambda: {"mask_sample": False})
    terminal_state: Optional[str] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    trajectory_invalid_reasons: List[str] = Field(default_factory=list)
    failure_labels: List[str] = Field(default_factory=list)
    num_user_messages: int = 0
    num_agent_messages: int = 0
    num_tool_calls: int = 0
    num_tool_results: int = 0
    num_tool_schema_failures: int = 0
    num_user_failures: int = 0
    num_agent_failures: int = 0
    num_tool_failures: int = 0
    outside_policy_scope: bool = False
    transferred: bool = False
    transfer_ground_truth_enforced: bool = False
    transfer_ground_truth_mismatch: bool = False
    judge_skipped_for_transfer_mismatch: bool = False
    judge_generation_error: Optional[str] = None
    judge_diagnostics: Dict[str, Any] = Field(default_factory=dict)


class SyntheticToolUseSimulationServer(SimpleResourcesServer):
    USER_RESPONSE_PREFIX_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^\s*Customer:\s*", flags=re.IGNORECASE)

    USER_SIMULATOR_SYSTEM_MESSAGE_TEMPLATE: ClassVar[str] = """# Instructions
- You are playing the role of a customer contacting a customer service representative.
- The scenario that describes the customer that you are acting as and the tasks that are to be completed is included below between the <scenario> and </scenario> tags.
- Strictly follow the instructions in the scenario description.
- To start a conversation, please make a request to the customer service representative in order to complete the tasks described in the scenario.
- Generate one message at a time, maintaining natural conversation flow.
- Each message should be a message from a customer to the representative.
- The goal is to continue the conversation until the tasks are complete.
- If the tasks have been completed during the previous messages in the interaction with the customer service representative, then instead of generating a message, output just '{complete_indicator}' to indicate that the conversation should end.
- If you are transferred to another agent by the customer service representative in a previous message in the conversation, then instead of generating a message, output just '{transfer_indicator}' to indicate the transfer.

<scenario>
{customer_scenario}
</scenario>

Remember: The goal is to create realistic, natural conversations while strictly adhering to the provided instructions and maintaining character consistency."""

    ENVIRONMENT_SIMULATOR_SYSTEM_MESSAGE_TEMPLATE: ClassVar[str] = """# Instructions
- Your task is to generate the result of executing a tool.
- The tool is executed by a customer service representative in the context of a conversation with a customer to retrieve information or accomplish a task.
- The conversation between the customer and the representative is shown between <conversation> and </conversation> tags, with each message sent by the customer appearing between <customer> and </customer> tags, and each message sent by the customer service representative appearing between <representative> and </representative> tags.
- The way that the representative handles customer requests is described below between the <policy> and </policy> tags.
- The scenario that describes the customer and the tasks that they are trying to complete is shown below between the <scenario> and </scenario> tags.
- The definitions of the available tools are provided below between the <tools> and </tools> tags, with each tool definition appearing between <tool> and </tool> tags.

<policy>
{domain_policy}
</policy>

<scenario>
{customer_scenario}
</scenario>

<tools>
{tool_definitions}
</tools>

Please make sure that the tool execution result that you generate is consistent with the definition of the tool being executed, the information about the customer in the scenario, and the previous conversation between the customer and the representative.  Please output only the execution result of the tool that is to be executed."""

    ENVIRONMENT_CONVERSATION_MESSAGE_TEMPLATE: ClassVar[str] = """<{sender}>
{message}
</{sender}>"""

    ENVIRONMENT_CONVERSATION_TEMPLATE: ClassVar[str] = """<conversation>
{conversation}
</conversation>

"""

    ENVIRONMENT_USER_MODEL_MESSAGE_TEMPLATE: ClassVar[str] = """{conversation}Tool to execute: {tool_name}
Arguments for tool execution: {arguments}"""

    MESSAGE_SYSTEM_MESSAGE_PREFIX: ClassVar[str] = """# Instructions
- Your task is to evaluate one step in an interaction between a customer and a customer service representative.
- The previous steps in the interaction are shown in order between the <previous_steps> and </previous_steps> tags.
- The step that you are to evaluate is the current step, which is shown between the <current_step> and </current_step> tags.
- Each step in the interaction is either a message from the customer to the representative, a message from the representative to the customer, the execution of a tool, or the result of executing a tool.
- A message from the customer to the representative or from the representative to the customer is shown between <message> and </message> tags.
- An execution of a tool can be requested by the representative, and is shown between <execute_tool> and </execute_tool> tags.
- When the representative requests the execution of a tool, the next step in the interaction should be the result of executing the tool, which is shown between <tool_result> and </tool_result> tags.
- The way that the representative handles customer requests is described below between the <policy> and </policy> tags.

<policy>
{policy}
</policy>

"""

    USER_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE: ClassVar[str] = (
        MESSAGE_SYSTEM_MESSAGE_PREFIX
        + """- The scenario that describes the customer and the tasks they are trying to complete is shown below between the <scenario> and </scenario> tags.

<scenario>
{customer_scenario}
</scenario>

- The current step that you are to evaluate is a message from the customer to the representative.
- The current step should be considered a success if the message from the customer to the representative follows naturally from the previous steps in the interaction and is consistent with the information about the customer in the scenario.
- If the message from the customer to the representative in the current step does not advance the interaction, is not consistent with the scenario, or is not consistent with the previous steps in the interaction, then the current step should be considered a failure.
- If the current step is the first step in the interaction, then it should be considered a success if it is consistent with the scenario, and it should be considered a failure if it is not consistent with the scenario.  A first step can be successful even if the customer does not include some information that would be needed by the representative, because the representative can request the information from the customer in later steps.
- The customer will send the message '{complete_indicator}' to indicate that they would like to end the interaction with the representative.  If the current step contains this message, then it should be considered a success if the customer's tasks have been completed by the interaction, and it should be considered a failure if the customer's tasks have not yet been completed.
- The customer will send the message '{transfer_indicator}' to indicate that they have been transferred to another agent by the representative.  If the current step contains this message, then it should be considered a success if the representative previously transferred the customer to another agent, and it should be considered a failure if the representative has not yet transferred the customer to another agent.

Please output the evaluation of the current step and an explanation for the evaluation using the following JSON schema:
{evaluation_schema}"""
    )

    AGENT_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE: ClassVar[str] = (
        MESSAGE_SYSTEM_MESSAGE_PREFIX
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

    ENVIRONMENT_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE: ClassVar[str] = (
        MESSAGE_SYSTEM_MESSAGE_PREFIX
        + """- The scenario that describes the customer and the tasks they are trying to complete is shown below between the <scenario> and </scenario> tags.
- The definitions of the tools that are available for execution are provided below between the <tools> and </tools> tags, with each tool definition appearing between <tool> and </tool> tags.

<scenario>
{customer_scenario}
</scenario>

<tools>
{tool_definitions}
</tools>

- The current step that you are to evaluate is the result of executing a tool.
- The previous step in the interaction should specify the tool to be executed and the arguments for execution.
- The current step should be considered a success if the tool execution result conforms to the definition of the tool being executed, and is consistent with the information about the customer in the scenario and the previous steps in the interaction.
- The current step should be considered a failure if the tool execution result is not consistent with the definition of the tool being executed, the information about the customer in the scenario, or the previous steps in the interaction.

Please output the evaluation of the current step and an explanation for the evaluation using the following JSON schema:
{evaluation_schema}"""
    )

    AGENT_CONVERSATION_EVALUATION_SYSTEM_MESSAGE_TEMPLATE: ClassVar[str] = """# Instructions
- Your task is to evaluate the actions taken by a customer service representative in an interaction between a customer and the representative.
- The steps in the interaction are shown in order between the <steps> and </steps> tags.
- Each step in the interaction is either a message from the customer to the representative, a message from the representative to the customer, the execution of a tool, or the result of executing a tool.
- A message from the customer to the representative or from the representative to the customer is shown between <message> and </message> tags.
- An execution of a tool can be requested by the representative, and is shown between <execute_tool> and </execute_tool> tags.
- When the representative requests the execution of a tool, the next step in the interaction should be the result of executing the tool, which is shown between <tool_result> and </tool_result> tags.
- The way that the representative handles customer requests is described below between the <policy> and </policy> tags.
- The definitions of the tools that are available to the representative for execution are provided below between the <tools> and </tools> tags, with each tool definition appearing between <tool> and </tool> tags.
- Whenever the representative takes an action, the representative can either send a message to the customer, or execute a tool.
- The actions of the representative should be considered a success if they are helpful in addressing the request from the customer.
- The actions of the representative should be considered a failure if they do not help to handle the customer's request, or are not consistent with the steps in the interaction that precede them or the policy for handling customer requests.

<policy>
{policy}
</policy>

<tools>
{tool_definitions}
</tools>

- The representative can transfer the customer to another agent.  This should be done if the customer's request cannot be handled by the representative according to the policy, but should not be done if the representative can handle the request under the policy.
- In this interaction, the representative should {transfer_instruction}transfer the customer to another agent.
- The actions of the representative should be considered a failure if the representative transfers the customer to another agent when they should not transfer the customer, or if the representative does not transfer the customer to another agent when they should transfer the customer.

Please output the evaluation of the actions taken by the customer service representative and an explanation for the evaluation using the following JSON schema:
{evaluation_schema}"""

    USER_AGENT_ENVIRONMENT_CONVERSATION_EVALUATION_SYSTEM_MESSAGE_TEMPLATE: ClassVar[str] = """# Instructions
- Your task is to evaluate an interaction between a customer and a customer service representative.
- Please evaluate separately the actions of the customer, the actions of the representative, and the tool results in the interaction.
- The steps in the interaction are shown in order between the <steps> and </steps> tags.
- Each step in the interaction is either a message from the customer to the representative, a message from the representative to the customer, the execution of a tool, or the result of executing a tool.
- A message from the customer to the representative or from the representative to the customer is shown between <message> and </message> tags.
- An execution of a tool can be requested by the representative, and is shown between <execute_tool> and </execute_tool> tags.
- When the representative requests the execution of a tool, the next step in the interaction should be the result of executing the tool, which is shown between <tool_result> and </tool_result> tags.
- The way that the representative handles customer requests is described below between the <policy> and </policy> tags.
- The definitions of the tools that are available to the representative for execution are provided below between the <tools> and </tools> tags, with each tool definition appearing between <tool> and </tool> tags.
- The scenario that describes the customer and the tasks they are trying to complete is shown below between the <scenario> and </scenario> tags.

<policy>
{policy}
</policy>

<tools>
{tool_definitions}
</tools>

<scenario>
{customer_scenario}
</scenario>

- The goal of the customer is to complete the tasks described in the scenario by communicating with the representative, while maintaining consistency with the information about the customer in the scenario.
- The actions of the customer should be considered a success if they follow naturally from the interaction with the representative and are consistent with the scenario.
- The actions of the customer should be considered a failure if they do not advance the interaction or are not consistent with the scenario.
- The customer should send the message '{complete_indicator}' to end the interaction with the representative when their tasks have been completed.
- The customer should send the message '{transfer_indicator}' when they have been transferred to another agent by the representative.
- The actions of the customer should be considered a failure if the customer ends the interaction when their tasks have not yet been completed, or sends the message '{transfer_indicator}' when they have not yet been transferred to another agent by the representative.
- Whenever the representative takes an action, the representative can either send a message to the customer, or execute a tool.
- The actions of the representative should be considered a success if they are helpful in addressing the request from the customer.
- The actions of the representative should be considered a failure if they do not help to handle the customer's request, or are not consistent with the steps in the interaction that precede them or the policy for handling customer requests.
- The representative can transfer the customer to another agent.  This should be done if the customer's request cannot be handled by the representative according to the policy, but should not be done if the representative can handle the request under the policy.
- In this interaction, the representative should {transfer_instruction}transfer the customer to another agent.
- The actions of the representative should be considered a failure if the representative transfers the customer to another agent when they should not transfer the customer, or if the representative does not transfer the customer to another agent when they should transfer the customer.
- When the representative executes a tool, the tool execution step should specify the tool to be executed and the arguments for execution, and the next step should be the result of executing the tool.
- The tool results in the interaction should be considered a success if they conform to the definitions of the tools being executed, and are consistent with the arguments for execution, the information about the customer in the scenario, the interaction between the customer and the representative, and each other.
- The tool results in the interaction should be considered a failure if they are not consistent with the definitions of the tools being executed, the arguments for execution, the information about the customer in the scenario, the interaction between the customer and the representative, or each other.

Please output the evaluation of the actions taken by the customer, the evaluation of the actions taken by the customer service representative, the evaluation of the tool results in the interaction, and explanations for the evaluations using the following JSON schema:
{evaluation_schema}"""

    MESSAGE_CONVERSATION_TEMPLATE: ClassVar[str] = """<previous_steps>
{previous_steps}
</previous_steps>

<current_step>
{current_step}
</current_step>"""

    COMPLETE_CONVERSATION_TEMPLATE: ClassVar[str] = """<steps>
{steps}
</steps>"""

    TEXT_MESSAGE_TEMPLATE: ClassVar[str] = """<message>
Sender: {sender}
Content: {content}
</message>"""

    TOOL_CALL_MESSAGE_TEMPLATE: ClassVar[str] = """<execute_tool>
Execution ID: {execution_id}
Tool name: {tool_name}
Arguments for execution: {arguments}
</execute_tool>"""

    TOOL_EXECUTION_MESSAGE_TEMPLATE: ClassVar[str] = """<tool_result>
Execution ID: {execution_id}
Execution result: {execution_result}
</tool_result>"""

    TOOL_DEFINITION_TEMPLATE: ClassVar[str] = """<tool>
Name: {name}
Documentation: {documentation}
Parameters in JSON Schema format: {parameters}
Return type in JSON Schema format: {return_type}
</tool>"""

    config: SyntheticToolUseSimulationConfig
    session_id_to_state: Dict[str, SyntheticSessionState] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/session_tools")(self.session_tools)
        app.post("/session_resume")(self.session_resume)
        app.post("/record_agent_message")(self.record_agent_message)
        app.post("/record_generation_error")(self.record_generation_error)
        app.post("/record_agent_step_limit")(self.record_agent_step_limit)
        app.post("/next_user_message")(self.next_user_message)
        app.post("/execute_agent_tool_call")(self.execute_agent_tool_call)
        app.post("/{tool_name}")(self.route_tool_call)
        return app

    async def seed_session(
        self, request: Request, body: SyntheticToolUseSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        state = SyntheticSessionState(
            domain_name=body.domain_name,
            policy=body.policy,
            tool_signatures=body.tools,
            customer_scenario=body.customer_scenario,
            source_artifacts=body.source_artifacts,
        )
        self._hydrate_prefilled_history(state, body.responses_create_params)

        prefilled_users = [
            message for message in state.messages if message.type == MessageType.TEXT and message.source == Source.USER
        ]
        if body.initial_user_message is not None:
            if prefilled_users:
                if prefilled_users[0].content != body.initial_user_message:
                    raise HTTPException(
                        status_code=400,
                        detail="initial_user_message must match the first user message in responses_create_params.input.",
                    )
            elif state.messages:
                raise HTTPException(
                    status_code=400,
                    detail="initial_user_message cannot be merged into an existing prefilled history without a user message.",
                )
            else:
                self._append_prefilled_message(
                    state,
                    SyntheticMessage(
                        type=MessageType.TEXT,
                        source=Source.USER,
                        content=body.initial_user_message,
                    ),
                )

        state.prefill_message_count = len(state.messages)
        self.session_id_to_state[session_id] = state
        return BaseSeedSessionResponse()

    def _hydrate_prefilled_history(
        self,
        state: SyntheticSessionState,
        params: Optional[NeMoGymResponseCreateParamsNonStreaming],
    ) -> None:
        if params is None:
            return
        if isinstance(params.input, str):
            items: List[Any] = [NeMoGymEasyInputMessage(role="user", content=params.input)]
        else:
            items = list(params.input or [])

        tool_calls: Dict[str, SyntheticMessage] = {}
        executed_call_ids: set[str] = set()
        for item in items:
            item_type = self._input_item_type(item)
            role = self._input_item_role(item)
            if item_type == "reasoning" or role in {"system", "developer"}:
                continue
            if item_type == "message":
                if role not in {"user", "assistant"}:
                    raise HTTPException(status_code=400, detail=f"Unsupported prefilled message role: {role}.")
                self._append_prefilled_message(
                    state,
                    SyntheticMessage(
                        type=MessageType.TEXT,
                        source=Source.USER if role == "user" else Source.AGENT,
                        content=self._input_item_text(item),
                    ),
                )
                continue
            if item_type == "function_call":
                tool_call_id = self._input_item_field(item, "call_id")
                if not isinstance(tool_call_id, str) or tool_call_id in tool_calls:
                    raise HTTPException(
                        status_code=400, detail="Prefilled function calls must have unique call_id values."
                    )
                tool_call = SyntheticMessage(
                    type=MessageType.TOOL_CALL,
                    source=Source.AGENT,
                    tool_call_id=tool_call_id,
                    tool_name=self._required_input_item_string(item, "name"),
                    arguments=self._required_input_item_string(item, "arguments"),
                )
                self._append_prefilled_message(state, tool_call)
                tool_calls[tool_call_id] = tool_call
                continue
            if item_type == "function_call_output":
                tool_call_id = self._required_input_item_string(item, "call_id")
                if tool_call_id not in tool_calls:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prefilled tool output references unknown call_id {tool_call_id}.",
                    )
                if tool_call_id in executed_call_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prefilled tool call {tool_call_id} has more than one output.",
                    )
                tool_call = tool_calls[tool_call_id]
                self._append_prefilled_message(
                    state,
                    SyntheticMessage(
                        type=MessageType.TOOL_EXECUTION,
                        source=Source.ENVIRONMENT,
                        tool_call_id=tool_call_id,
                        tool_name=tool_call.tool_name,
                        arguments=tool_call.arguments,
                        content=self._required_input_item_string(item, "output"),
                    ),
                )
                executed_call_ids.add(tool_call_id)
                continue
            raise HTTPException(status_code=400, detail=f"Unsupported prefilled input item type: {item_type}.")

    def _append_prefilled_message(self, state: SyntheticSessionState, message: SyntheticMessage) -> None:
        if state.terminal_state is not None:
            raise HTTPException(status_code=400, detail="Prefilled history continues after a terminal message.")

        invalid = self._validate_message(state, message)
        state.messages.append(message)
        if message.type == MessageType.TOOL_EXECUTION:
            message.schema_valid = invalid is None
        if invalid is not None:
            invalid_reason, error = invalid
            self._record_generation_failure(
                state,
                invalid_reason,
                self._failure_label_for_source(message.source),
                error,
            )
        elif message.source == Source.USER and self._is_trajectory_complete_message(message):
            state.terminal_state = "complete"

    def _input_item_type(self, item: Any) -> Optional[str]:
        item_type = self._input_item_field(item, "type")
        if item_type is not None:
            return str(item_type)
        if self._input_item_role(item) is not None:
            return "message"
        if self._input_item_field(item, "output") is not None:
            return "function_call_output"
        if self._input_item_field(item, "name") is not None:
            return "function_call"
        return None

    def _input_item_role(self, item: Any) -> Optional[str]:
        role = self._input_item_field(item, "role")
        return str(role) if role is not None else None

    def _input_item_field(self, item: Any, field: str) -> Any:
        return item.get(field) if isinstance(item, dict) else getattr(item, field, None)

    def _required_input_item_string(self, item: Any, field: str) -> str:
        value = self._input_item_field(item, field)
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"Prefilled input item field {field} must be a string.")
        return value

    def _input_item_text(self, item: Any) -> str:
        content = self._input_item_field(item, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for content_item in content:
                item_type = self._input_item_field(content_item, "type")
                text = self._input_item_field(content_item, "text")
                if item_type not in {"input_text", "output_text"} or not isinstance(text, str):
                    break
                text_parts.append(text)
            else:
                return "".join(text_parts)
        raise HTTPException(status_code=400, detail="Prefilled user and assistant messages must contain only text.")

    async def session_resume(self, request: Request) -> SessionResumeResponse:
        state = self._get_state(request)
        pending_tool_calls = self._pending_tool_calls(state)
        if state.terminal_state is not None:
            return SessionResumeResponse(
                next_actor="terminal",
                terminal_state=state.terminal_state,
            )
        if pending_tool_calls:
            return SessionResumeResponse(
                next_actor="environment",
                pending_tool_calls=pending_tool_calls,
            )
        if not state.messages:
            return SessionResumeResponse(next_actor="user", is_initial=True)

        last_message = state.messages[-1]
        if last_message.source == Source.AGENT and last_message.type == MessageType.TEXT:
            return SessionResumeResponse(next_actor="user")
        if last_message.source == Source.USER or last_message.type == MessageType.TOOL_EXECUTION:
            return SessionResumeResponse(next_actor="agent")
        raise HTTPException(status_code=400, detail="Prefilled history does not end at a resumable state.")

    def _pending_tool_calls(self, state: SyntheticSessionState) -> List[PendingToolCall]:
        executed_call_ids = {
            message.tool_call_id
            for message in state.messages
            if message.type == MessageType.TOOL_EXECUTION and message.tool_call_id is not None
        }
        return [
            PendingToolCall(
                tool_name=message.tool_name or "",
                tool_call_id=message.tool_call_id or "",
                arguments=message.arguments or "",
            )
            for message in state.messages
            if message.type == MessageType.TOOL_CALL and message.tool_call_id not in executed_call_ids
        ]

    async def session_tools(self, request: Request) -> SessionToolsResponse:
        state = self._get_state(request)
        return SessionToolsResponse(tools=[self._tool_to_responses_api(tool) for tool in state.tool_signatures])

    async def record_agent_message(
        self, request: Request, body: RecordAgentMessageRequest
    ) -> RecordAgentMessageResponse:
        state = self._get_state(request)
        if state.terminal_state:
            return RecordAgentMessageResponse(should_continue=False, terminal_state=state.terminal_state)

        message = SyntheticMessage(
            type=MessageType.TEXT,
            source=Source.AGENT,
            content=body.content,
            responses=[body.response] if body.response is not None else None,
        )
        state.messages.append(message)
        return RecordAgentMessageResponse(should_continue=True)

    async def record_generation_error(
        self, request: Request, body: RecordGenerationErrorRequest
    ) -> RecordGenerationErrorResponse:
        state = self._get_state(request)
        if not state.terminal_state:
            self._record_generation_failure(
                state,
                TrajectoryInvalidReason.MESSAGE_GENERATION_ERROR,
                self._failure_label_for_source(body.source),
                body.error,
            )
        return RecordGenerationErrorResponse(terminal_state=state.terminal_state)

    async def record_agent_step_limit(
        self, request: Request, body: RecordAgentStepLimitRequest
    ) -> RecordAgentStepLimitResponse:
        state = self._get_state(request)
        if state.terminal_state:
            return RecordAgentStepLimitResponse(terminal_state=state.terminal_state)

        for tool_call in body.tool_calls:
            message = SyntheticMessage(
                type=MessageType.TOOL_CALL,
                source=Source.AGENT,
                responses=[body.response] if body.response is not None else None,
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
            )
            state.messages.append(message)
            invalid = self._validate_message(state, message)
            if invalid is not None:
                invalid_reason, error = invalid
                self._record_generation_failure(
                    state,
                    invalid_reason,
                    VerificationFailureLabel.AGENT_FAILURE,
                    error,
                )
                return RecordAgentStepLimitResponse(terminal_state=state.terminal_state)

        self._record_generation_failure(
            state,
            TrajectoryInvalidReason.EXCESSIVE_LENGTH,
            VerificationFailureLabel.TRAJECTORY_FAILURE,
            f"The maximum number of agent steps of {body.max_agent_steps} has been reached",
        )
        return RecordAgentStepLimitResponse(terminal_state=state.terminal_state)

    async def next_user_message(self, request: Request) -> NextUserMessageResponse:
        state = self._get_state(request)
        if state.terminal_state:
            return NextUserMessageResponse(message="", should_continue=False, terminal_state=state.terminal_state)

        user_message = ""
        message = None
        invalid = None
        try:
            for _ in range(max(1, self.config.generation_attempts)):
                user_message, response_object = await self._generate_user_message(state)
                message = SyntheticMessage(
                    type=MessageType.TEXT,
                    source=Source.USER,
                    content=user_message,
                    responses=[response_object] if response_object is not None else None,
                )
                invalid = self._validate_message(state, message)
                if invalid is None:
                    break
        except Exception as exc:
            self._record_generation_failure(
                state,
                TrajectoryInvalidReason.MESSAGE_GENERATION_ERROR,
                VerificationFailureLabel.USER_FAILURE,
                f"{exc}",
            )
            return NextUserMessageResponse(
                message="",
                should_continue=False,
                terminal_state=state.terminal_state,
            )

        assert message is not None
        state.messages.append(message)
        if invalid is not None:
            invalid_reason, error = invalid
            self._record_generation_failure(
                state,
                invalid_reason,
                VerificationFailureLabel.USER_FAILURE,
                error,
            )
        elif self._is_trajectory_complete_message(message):
            state.terminal_state = "complete"

        return NextUserMessageResponse(
            message=user_message,
            should_continue=state.terminal_state is None,
            terminal_state=state.terminal_state,
        )

    async def route_tool_call(
        self, tool_name: str, body: SyntheticToolCallRequest, request: Request
    ) -> SyntheticToolCallResponse:
        state = self._get_state(request)
        raw_arguments = body.model_dump(exclude_none=True)
        tool_call_id = str(raw_arguments.pop("_tool_call_id", f"tool_call_{self._next_tool_call_index(state)}"))
        return await self._handle_agent_tool_call(
            state=state,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments_string=json.dumps(raw_arguments),
            response_object=None,
        )

    async def execute_agent_tool_call(self, request: Request, body: AgentToolCallRequest) -> SyntheticToolCallResponse:
        state = self._get_state(request)
        return await self._handle_agent_tool_call(
            state=state,
            tool_name=body.tool_name,
            tool_call_id=body.tool_call_id,
            arguments_string=body.arguments,
            response_object=body.response,
        )

    async def _handle_agent_tool_call(
        self,
        *,
        state: SyntheticSessionState,
        tool_name: str,
        tool_call_id: str,
        arguments_string: str,
        response_object: Optional[Dict[str, Any]],
    ) -> SyntheticToolCallResponse:
        if state.terminal_state:
            return SyntheticToolCallResponse(
                output={"error": f"Trajectory is already terminal: {state.terminal_state}"},
                schema_valid=False,
                error=f"Trajectory is already terminal: {state.terminal_state}",
                should_continue=False,
                terminal_state=state.terminal_state,
            )
        executed_call_ids = {
            message.tool_call_id
            for message in state.messages
            if message.type == MessageType.TOOL_EXECUTION and message.tool_call_id is not None
        }
        tool_call_message = next(
            (
                message
                for message in state.messages
                if message.type == MessageType.TOOL_CALL
                and message.tool_call_id == tool_call_id
                and message.tool_call_id not in executed_call_ids
            ),
            None,
        )
        resumed_prefilled_call = tool_call_message is not None
        if tool_call_message is not None:
            if tool_call_message.tool_name != tool_name or tool_call_message.arguments != arguments_string:
                raise HTTPException(
                    status_code=400,
                    detail=f"Pending tool call {tool_call_id} does not match the requested execution.",
                )
        else:
            tool_call_message = SyntheticMessage(
                type=MessageType.TOOL_CALL,
                source=Source.AGENT,
                responses=[response_object] if response_object is not None else None,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments_string,
            )
            state.messages.append(tool_call_message)

            invalid = self._validate_message(state, tool_call_message)
            if invalid is not None:
                invalid_reason, arguments_error = invalid
                self._record_generation_failure(
                    state,
                    invalid_reason,
                    VerificationFailureLabel.AGENT_FAILURE,
                    arguments_error or "Invalid tool call arguments.",
                )
                return SyntheticToolCallResponse(
                    output={"error": arguments_error},
                    schema_valid=False,
                    error=arguments_error,
                    should_continue=False,
                    terminal_state=state.terminal_state,
                )

        tool = self._find_tool(state, tool_name)
        assert tool is not None
        output: Any = ""
        schema_valid = False
        last_error: Optional[str] = None
        execution_message: Optional[SyntheticMessage] = None
        try:
            for _ in range(max(1, self.config.generation_attempts)):
                if resumed_prefilled_call:
                    output, response_object = await self._generate_tool_result(
                        state,
                        pending_tool_call=tool_call_message,
                    )
                else:
                    output, response_object = await self._generate_tool_result(state)
                execution_message = SyntheticMessage(
                    type=MessageType.TOOL_EXECUTION,
                    source=Source.ENVIRONMENT,
                    responses=[response_object] if response_object is not None else None,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments_string,
                    content=output,
                )
                invalid = self._validate_message(state, execution_message)
                if invalid is None:
                    schema_valid = True
                    break
                _, last_error = invalid
        except Exception as exc:
            self._record_generation_failure(
                state,
                TrajectoryInvalidReason.MESSAGE_GENERATION_ERROR,
                VerificationFailureLabel.TOOL_FAILURE,
                f"{exc}",
            )
            return SyntheticToolCallResponse(
                output={"error": f"{exc}"},
                schema_valid=False,
                error=f"{exc}",
                should_continue=False,
                terminal_state=state.terminal_state,
            )

        assert execution_message is not None
        execution_message.schema_valid = schema_valid
        state.messages.append(execution_message)

        if not schema_valid and last_error:
            self._record_generation_failure(
                state,
                TrajectoryInvalidReason.INVALID_ENVIRONMENT_TOOL_EXECUTION,
                VerificationFailureLabel.TOOL_FAILURE,
                last_error,
            )

        return SyntheticToolCallResponse(
            output=output,
            schema_valid=schema_valid,
            error=None if schema_valid else last_error,
            should_continue=state.terminal_state is None,
            terminal_state=state.terminal_state,
        )

    async def verify(self, request: Request, body: SyntheticToolUseVerifyRequest) -> SyntheticToolUseVerifyResponse:
        state = self._get_state(request)
        diagnostics = self._trajectory_diagnostics(state)
        judge_generation_error = None
        judge_diagnostics: Dict[str, Any] = {}
        invalid_reasons = self._dedupe_strings(state.invalid_reasons)
        failure_labels = self._dedupe_strings(state.failure_labels)
        transfer_ground_truth_mismatch = diagnostics["transfer_ground_truth_mismatch"]
        judge_skipped_for_transfer_mismatch = False

        if state.terminal_state != "complete":
            if not invalid_reasons:
                self._record_generation_failure(
                    state,
                    TrajectoryInvalidReason.MESSAGE_GENERATION_ERROR,
                    VerificationFailureLabel.TRAJECTORY_FAILURE,
                    "Trajectory reached verification without a complete terminal state.",
                )
                invalid_reasons = self._dedupe_strings(state.invalid_reasons)
                failure_labels = self._dedupe_strings(state.failure_labels)
                diagnostics = self._trajectory_diagnostics(state)
            reward = 0.0
        elif invalid_reasons:
            reward = 0.0
        elif transfer_ground_truth_mismatch:
            reward = 0.0
            failure_labels = self._dedupe_strings([*failure_labels, VerificationFailureLabel.AGENT_FAILURE])
            judge_skipped_for_transfer_mismatch = True
            state.agent_verification_result = AgentVerificationResult(
                conversation_verification_result=VerificationResult(
                    reward=0,
                    explanation=(
                        "Deterministic transfer-ground-truth mismatch: "
                        f"expected transferred={diagnostics['outside_policy_scope']}, "
                        f"observed transferred={diagnostics['transferred']}."
                    ),
                ),
                overall_reward=0,
            )
        elif self.config.enable_llm_judge and self.config.judge_model_server is not None:
            try:
                reward, invalid_reasons, failure_labels, judge_diagnostics = await self._verify_messages(state)
            except JudgeProviderExhaustedError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except JudgeProviderRequestError as exc:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
            except Exception as exc:
                judge_generation_error = f"{type(exc).__name__}: {exc}"
                invalid_reasons = [TrajectoryInvalidReason.VERIFICATION_GENERATION_ERROR]
                failure_labels = [VerificationFailureLabel.VERIFICATION_GENERATION_ERROR]
                reward = 0.0
        else:
            reward = 1.0

        diagnostics["judge_skipped_for_transfer_mismatch"] = judge_skipped_for_transfer_mismatch
        if self.config.enforce_transfer_ground_truth:
            judge_diagnostics = {
                **judge_diagnostics,
                "transfer_ground_truth_enforcement": {
                    "expected_transfer": diagnostics["outside_policy_scope"],
                    "observed_transfer": diagnostics["transferred"],
                    "mismatch": transfer_ground_truth_mismatch,
                    "judge_skipped": judge_skipped_for_transfer_mismatch,
                },
            }

        self._synchronize_verification_state(
            state=state,
            reward=reward,
            invalid_reasons=invalid_reasons,
            failure_labels=failure_labels,
            judge_generation_error=judge_generation_error,
            judge_diagnostics=judge_diagnostics,
        )
        response_payload = body.model_dump()
        response_payload.update(diagnostics)
        response_payload.update(self._failure_count_metrics(state.failure_labels))
        response_payload.update(
            {
                "reward": state.verification_reward,
                "terminal_state": state.terminal_state,
                "trajectory_invalid_reasons": state.invalid_reasons,
                "failure_labels": state.failure_labels,
                "judge_generation_error": state.judge_generation_error,
                "judge_diagnostics": state.judge_diagnostics,
                "instance_config": {"mask_sample": self._should_mask_sample(state)},
                "result": self._trajectory_result(state),
            }
        )
        return SyntheticToolUseVerifyResponse.model_validate(response_payload)

    def _get_state(self, request: Request) -> SyntheticSessionState:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_state:
            raise HTTPException(status_code=400, detail="Session not initialized. Call /seed_session first.")
        return self.session_id_to_state[session_id]

    def _find_tool(self, state: SyntheticSessionState, tool_name: str) -> Optional[ToolSignature]:
        for tool in state.tool_signatures:
            if tool.name == tool_name:
                return tool
        return None

    def _next_tool_call_index(self, state: SyntheticSessionState) -> int:
        return 1 + sum(1 for message in state.messages if message.type == MessageType.TOOL_CALL)

    def _record_generation_failure(
        self,
        state: SyntheticSessionState,
        invalid_reason: TrajectoryInvalidReason,
        failure_label: VerificationFailureLabel,
        error: str,
    ) -> None:
        state.generation_invalid_reason = str(invalid_reason)
        if str(invalid_reason) not in state.invalid_reasons:
            state.invalid_reasons.append(str(invalid_reason))
        if str(failure_label) not in state.failure_labels:
            state.failure_labels.append(str(failure_label))
        state.terminal_state = "incomplete"
        state.terminal_error = error
        state.source_artifacts.setdefault("terminal_errors", []).append(error)

    def _synchronize_verification_state(
        self,
        *,
        state: SyntheticSessionState,
        reward: float,
        invalid_reasons: List[str],
        failure_labels: List[str],
        judge_generation_error: Optional[str],
        judge_diagnostics: Dict[str, Any],
    ) -> None:
        state.invalid_reasons = self._dedupe_strings(invalid_reasons)
        state.failure_labels = self._dedupe_strings(failure_labels)
        state.verification_reward = float(reward)
        state.judge_generation_error = judge_generation_error
        state.judge_diagnostics = judge_diagnostics

        if judge_generation_error is not None and state.agent_verification_result is None:
            state.agent_verification_result = AgentVerificationResult(
                conversation_verification_result=VerificationResult(generation_error=judge_generation_error),
                trajectory_invalid_reasons=list(state.invalid_reasons),
            )

    def _failure_label_for_source(self, source: Source) -> VerificationFailureLabel:
        if source == Source.USER:
            return VerificationFailureLabel.USER_FAILURE
        if source == Source.AGENT:
            return VerificationFailureLabel.AGENT_FAILURE
        if source == Source.ENVIRONMENT:
            return VerificationFailureLabel.TOOL_FAILURE
        return VerificationFailureLabel.TRAJECTORY_FAILURE

    def _tool_to_responses_api(self, tool: ToolSignature) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.normalized_doc,
            "parameters": tool.model_visible_params,
            "strict": True,
        }

    async def _generate_user_message(self, state: SyntheticSessionState) -> tuple[str, Optional[Dict[str, Any]]]:
        model_server = self.config.user_model_server or self.config.simulator_model_server
        if model_server is None:
            if not any(message.source == Source.USER for message in state.messages):
                return state.customer_scenario.reason_for_contact or "I need help with my account.", None
            return "Thanks. Please continue helping me with this request.", None

        response = await self._call_model(
            model_server=model_server,
            params=self.config.user_responses_create_params,
            messages=self._user_simulator_messages(state),
        )
        response_text = self._first_response_text(response)
        canonical_response = self.USER_RESPONSE_PREFIX_PATTERN.sub("", response_text)
        return canonical_response, response.model_dump(mode="json", exclude_unset=True)

    def _user_simulator_messages(self, state: SyntheticSessionState) -> List[NeMoGymEasyInputMessage]:
        system_message = self.USER_SIMULATOR_SYSTEM_MESSAGE_TEMPLATE.format(
            complete_indicator=TRAJECTORY_COMPLETE_INDICATOR,
            transfer_indicator=AGENT_TRANSFER_INDICATOR,
            customer_scenario=state.customer_scenario.create_string(include_domain=True),
        )
        messages = [NeMoGymEasyInputMessage(role="system", content=system_message)]
        for message in state.messages:
            if message.type != MessageType.TEXT:
                continue
            if message.source == Source.USER:
                role = "assistant"
                sender = "Customer"
            elif message.source == Source.AGENT:
                role = "user"
                sender = "Representative"
            else:
                raise NotImplementedError
            messages.append(NeMoGymEasyInputMessage(role=role, content=f"{sender}: {message.content}"))
        return messages

    async def _generate_tool_result(
        self,
        state: SyntheticSessionState,
        pending_tool_call: Optional[SyntheticMessage] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        model_server = self.config.tool_simulator_model_server or self.config.simulator_model_server
        tool_call_message = pending_tool_call or state.messages[-1]
        if model_server is None:
            return json.dumps({"output": f"Simulated result for {tool_call_message.tool_name}."}), None

        messages = (
            self._environment_simulator_messages_for_pending_call(state, tool_call_message)
            if pending_tool_call is not None
            else self._environment_simulator_messages(state)
        )
        response = await self._call_model(
            model_server=model_server,
            params=self.config.tool_simulator_responses_create_params,
            messages=messages,
        )
        return self._first_response_text(response), response.model_dump(mode="json", exclude_unset=True)

    def _environment_simulator_system_message(self, state: SyntheticSessionState) -> str:
        return self.ENVIRONMENT_SIMULATOR_SYSTEM_MESSAGE_TEMPLATE.format(
            domain_policy=state.policy,
            customer_scenario=state.customer_scenario.create_string(),
            tool_definitions=self._format_tool_definitions(state.tool_signatures),
        )

    def _environment_simulator_messages(self, state: SyntheticSessionState) -> List[NeMoGymEasyInputMessage]:
        messages = [
            NeMoGymEasyInputMessage(
                role="system",
                content=self._environment_simulator_system_message(state),
            )
        ]
        conversation_messages: List[str] = []
        for message in state.messages:
            if message.type == MessageType.TEXT:
                if message.source == Source.USER:
                    sender = "customer"
                elif message.source == Source.AGENT:
                    sender = "representative"
                else:
                    raise NotImplementedError
                conversation_messages.append(
                    self.ENVIRONMENT_CONVERSATION_MESSAGE_TEMPLATE.format(
                        sender=sender,
                        message=message.content,
                    )
                )
            elif message.type == MessageType.TOOL_CALL:
                conversation_string = ""
                if conversation_messages:
                    conversation_string = self.ENVIRONMENT_CONVERSATION_TEMPLATE.format(
                        conversation="\n\n".join(conversation_messages)
                    )
                messages.append(
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=self.ENVIRONMENT_USER_MODEL_MESSAGE_TEMPLATE.format(
                            conversation=conversation_string,
                            tool_name=message.tool_name,
                            arguments=message.arguments,
                        ),
                    )
                )
            elif message.type == MessageType.TOOL_EXECUTION:
                messages.append(NeMoGymEasyInputMessage(role="assistant", content=message.content))
                conversation_messages.clear()
            else:
                raise NotImplementedError
        return messages

    def _environment_simulator_messages_for_pending_call(
        self,
        state: SyntheticSessionState,
        pending_tool_call: SyntheticMessage,
    ) -> List[NeMoGymEasyInputMessage]:
        messages = [
            NeMoGymEasyInputMessage(
                role="system",
                content=self._environment_simulator_system_message(state),
            )
        ]
        conversation_messages: List[str] = []
        tool_calls = {
            message.tool_call_id: message
            for message in state.messages
            if message.type == MessageType.TOOL_CALL and message.tool_call_id is not None
        }
        for message in state.messages:
            if message.type == MessageType.TEXT:
                sender = "customer" if message.source == Source.USER else "representative"
                conversation_messages.append(
                    self.ENVIRONMENT_CONVERSATION_MESSAGE_TEMPLATE.format(
                        sender=sender,
                        message=message.content,
                    )
                )
            elif message.type == MessageType.TOOL_EXECUTION:
                completed_call = tool_calls.get(message.tool_call_id)
                if completed_call is None:
                    continue
                conversation_string = ""
                if conversation_messages:
                    conversation_string = self.ENVIRONMENT_CONVERSATION_TEMPLATE.format(
                        conversation="\n\n".join(conversation_messages)
                    )
                messages.append(
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=self.ENVIRONMENT_USER_MODEL_MESSAGE_TEMPLATE.format(
                            conversation=conversation_string,
                            tool_name=completed_call.tool_name,
                            arguments=completed_call.arguments,
                        ),
                    )
                )
                messages.append(NeMoGymEasyInputMessage(role="assistant", content=message.content))
                conversation_messages.clear()

        conversation_string = ""
        if conversation_messages:
            conversation_string = self.ENVIRONMENT_CONVERSATION_TEMPLATE.format(
                conversation="\n\n".join(conversation_messages)
            )
        messages.append(
            NeMoGymEasyInputMessage(
                role="user",
                content=self.ENVIRONMENT_USER_MODEL_MESSAGE_TEMPLATE.format(
                    conversation=conversation_string,
                    tool_name=pending_tool_call.tool_name,
                    arguments=pending_tool_call.arguments,
                ),
            )
        )
        return messages

    async def _verify_messages(
        self, state: SyntheticSessionState
    ) -> tuple[float, List[str], List[str], Dict[str, Any]]:
        if self.config.verification_type == VerificationType.COMPLETE_TRAJECTORY_COMBINED_EVALUATION:
            return await self._verify_complete_trajectory_with_combined_evaluation(state)

        if self.config.enable_termination:
            invalid_reasons, overall_agent_reward, failure_labels = await self._verify_messages_with_termination(state)
        else:
            invalid_reasons, overall_agent_reward, failure_labels = await self._verify_all_messages(state)

        conversation_verification_result = None
        if not self.config.enable_termination or (len(invalid_reasons) < 1 and overall_agent_reward > 0):
            conversation_message = self.COMPLETE_CONVERSATION_TEMPLATE.format(
                steps=self._format_message_list(state.messages)
            )
            conversation_verification_result = await self._generate_judge_evaluation(
                TrajectoryEvaluationType.AGENT_CONVERSATION, conversation_message, state
            )
            conversation_reward = conversation_verification_result.reward
            if conversation_reward is None:
                invalid_reasons.append(TrajectoryInvalidReason.VERIFICATION_GENERATION_ERROR)
                failure_labels.append(VerificationFailureLabel.VERIFICATION_GENERATION_ERROR)
            else:
                overall_agent_reward = min(overall_agent_reward, conversation_reward)
                if conversation_reward < 1:
                    failure_labels.append(VerificationFailureLabel.AGENT_FAILURE)

        reward = 0.0 if invalid_reasons else float(overall_agent_reward)
        normalized_invalid_reasons = self._dedupe_strings(invalid_reasons)
        if normalized_invalid_reasons:
            state.agent_verification_result = AgentVerificationResult(
                conversation_verification_result=conversation_verification_result,
                trajectory_invalid_reasons=normalized_invalid_reasons,
            )
        else:
            state.agent_verification_result = AgentVerificationResult(
                conversation_verification_result=conversation_verification_result,
                overall_reward=int(overall_agent_reward),
            )
        diagnostics = {
            "verification_type": "message",
            "enable_termination": self.config.enable_termination,
            "message_verification_results": [
                {
                    "index": index,
                    "type": str(message.type),
                    "source": str(message.source),
                    "tool_name": message.tool_name,
                    "reward": None if message.verification_result is None else message.verification_result.reward,
                    "explanation": None
                    if message.verification_result is None
                    else message.verification_result.explanation,
                    "generation_error": None
                    if message.verification_result is None
                    else message.verification_result.generation_error,
                }
                for index, message in enumerate(state.messages)
            ],
            "conversation_verification_result": None
            if conversation_verification_result is None
            else conversation_verification_result.model_dump(exclude_none=True),
        }
        return reward, normalized_invalid_reasons, self._dedupe_strings(failure_labels), diagnostics

    async def _verify_complete_trajectory_with_combined_evaluation(
        self, state: SyntheticSessionState
    ) -> tuple[float, List[str], List[str], Dict[str, Any]]:
        conversation_message = self.COMPLETE_CONVERSATION_TEMPLATE.format(
            steps=self._format_message_list(state.messages)
        )
        (
            user_verification_result,
            agent_verification_result,
            environment_verification_result,
        ) = await self._generate_user_agent_environment_evaluation(state, conversation_message)

        invalid_reasons: List[str] = []
        failure_labels: List[str] = []
        if user_verification_result is None or environment_verification_result is None:
            invalid_reasons.append(TrajectoryInvalidReason.VERIFICATION_GENERATION_ERROR)
            failure_labels.append(VerificationFailureLabel.VERIFICATION_GENERATION_ERROR)
            reward = 0.0
        else:
            if user_verification_result.reward is not None and user_verification_result.reward < 1:
                invalid_reasons.append(TrajectoryInvalidReason.NO_REWARD_USER_MESSAGE)
                failure_labels.append(VerificationFailureLabel.USER_FAILURE)
            if environment_verification_result.reward is not None and environment_verification_result.reward < 1:
                invalid_reasons.append(TrajectoryInvalidReason.NO_REWARD_ENVIRONMENT_MESSAGE)
                failure_labels.append(VerificationFailureLabel.TOOL_FAILURE)

            agent_reward = agent_verification_result.reward
            if agent_reward is None:
                invalid_reasons.append(TrajectoryInvalidReason.VERIFICATION_GENERATION_ERROR)
                failure_labels.append(VerificationFailureLabel.VERIFICATION_GENERATION_ERROR)
                reward = 0.0
            else:
                if agent_reward < 1:
                    failure_labels.append(VerificationFailureLabel.AGENT_FAILURE)
                reward = 0.0 if invalid_reasons else float(agent_reward)

        normalized_invalid_reasons = self._dedupe_strings(invalid_reasons)
        state.user_verification_result = user_verification_result
        state.environment_verification_result = environment_verification_result
        if normalized_invalid_reasons:
            state.agent_verification_result = AgentVerificationResult(
                conversation_verification_result=agent_verification_result,
                trajectory_invalid_reasons=normalized_invalid_reasons,
            )
        else:
            state.agent_verification_result = AgentVerificationResult(
                conversation_verification_result=agent_verification_result,
                overall_reward=agent_verification_result.reward,
            )
        diagnostics = {
            "verification_type": str(VerificationType.COMPLETE_TRAJECTORY_COMBINED_EVALUATION),
            "enable_termination": self.config.enable_termination,
            "user_verification_result": None
            if user_verification_result is None
            else user_verification_result.model_dump(exclude_none=True),
            "conversation_verification_result": agent_verification_result.model_dump(exclude_none=True),
            "environment_verification_result": None
            if environment_verification_result is None
            else environment_verification_result.model_dump(exclude_none=True),
        }
        return reward, normalized_invalid_reasons, self._dedupe_strings(failure_labels), diagnostics

    async def _verify_messages_with_termination(
        self, state: SyntheticSessionState
    ) -> tuple[List[str], int, List[str]]:
        messages = state.messages
        for message_index, message in enumerate(messages):
            message_source = message.source
            if message_source == Source.AGENT:
                continue

            previous_messages = messages[:message_index]
            message_reward = await self._verify_message(state, message, previous_messages)
            if message_reward is None:
                return (
                    [TrajectoryInvalidReason.VERIFICATION_GENERATION_ERROR],
                    1,
                    [VerificationFailureLabel.VERIFICATION_GENERATION_ERROR],
                )
            if message_reward < 1:
                if message_source == Source.USER:
                    return [TrajectoryInvalidReason.NO_REWARD_USER_MESSAGE], 1, [VerificationFailureLabel.USER_FAILURE]
                if message_source == Source.ENVIRONMENT:
                    return (
                        [TrajectoryInvalidReason.NO_REWARD_ENVIRONMENT_MESSAGE],
                        1,
                        [VerificationFailureLabel.TOOL_FAILURE],
                    )
                raise NotImplementedError

        for message_index, message in enumerate(messages):
            if message.source != Source.AGENT:
                continue

            previous_messages = messages[:message_index]
            message_reward = await self._verify_message(state, message, previous_messages)
            if message_reward is None:
                return (
                    [TrajectoryInvalidReason.VERIFICATION_GENERATION_ERROR],
                    1,
                    [VerificationFailureLabel.VERIFICATION_GENERATION_ERROR],
                )
            if message_reward < 1:
                return [], 0, [VerificationFailureLabel.AGENT_FAILURE]

        return [], 1, []

    async def _verify_all_messages(self, state: SyntheticSessionState) -> tuple[List[str], int, List[str]]:
        messages = state.messages
        invalid_reasons: List[str] = []
        failure_labels: List[str] = []
        overall_agent_reward = 1
        for message_index, message in enumerate(messages):
            previous_messages = messages[:message_index]
            message_reward = await self._verify_message(state, message, previous_messages)
            if message_reward is None:
                invalid_reasons.append(TrajectoryInvalidReason.VERIFICATION_GENERATION_ERROR)
                failure_labels.append(VerificationFailureLabel.VERIFICATION_GENERATION_ERROR)
            elif message_reward < 1:
                if message.source == Source.USER:
                    invalid_reasons.append(TrajectoryInvalidReason.NO_REWARD_USER_MESSAGE)
                    failure_labels.append(VerificationFailureLabel.USER_FAILURE)
                elif message.source == Source.AGENT:
                    overall_agent_reward = min(overall_agent_reward, message_reward)
                    failure_labels.append(VerificationFailureLabel.AGENT_FAILURE)
                elif message.source == Source.ENVIRONMENT:
                    invalid_reasons.append(TrajectoryInvalidReason.NO_REWARD_ENVIRONMENT_MESSAGE)
                    failure_labels.append(VerificationFailureLabel.TOOL_FAILURE)
                else:
                    raise NotImplementedError

        return invalid_reasons, overall_agent_reward, failure_labels

    async def _verify_message(
        self,
        state: SyntheticSessionState,
        message: SyntheticMessage,
        previous_messages: List[SyntheticMessage],
    ) -> Optional[int]:
        previous_message_string = self._format_message_list(previous_messages)
        current_message_string = self._format_message(message)
        model_message = self.MESSAGE_CONVERSATION_TEMPLATE.format(
            previous_steps=previous_message_string,
            current_step=current_message_string,
        )
        verification_result = await self._generate_judge_evaluation(message.source, model_message, state)
        message.verification_result = verification_result
        return verification_result.reward

    async def _generate_judge_evaluation(
        self,
        evaluation_type: Source | TrajectoryEvaluationType,
        user_message: str,
        state: Optional[SyntheticSessionState] = None,
    ) -> VerificationResult:
        if self.config.judge_model_server is None:
            raise RuntimeError("judge_model_server is required for LLM trajectory verification.")
        if state is None:
            state = next(iter(self.session_id_to_state.values()))
        system_message = self._system_messages(state)[evaluation_type]
        response_objects: List[Dict[str, Any]] = []
        generation_error: Optional[str] = None

        for _ in range(max(1, self.config.generation_attempts)):
            try:
                response = await self._call_judge_model(
                    model_server=self.config.judge_model_server,
                    params=self.config.judge_responses_create_params,
                    messages=[
                        NeMoGymEasyInputMessage(role="system", content=system_message),
                        NeMoGymEasyInputMessage(role="user", content=user_message),
                    ],
                )
                response_objects.append(response.model_dump())
                response_text = self._last_text(response)
                canonical_text = self._strip_json_fence(response_text)
                judge_evaluation = Evaluation.model_validate_json(canonical_text)
                reward = 1 if judge_evaluation.success else 0
                return VerificationResult(
                    reward=reward,
                    explanation=judge_evaluation.explanation,
                    judge_response=canonical_text,
                    responses=response_objects,
                )
            except (ValidationError, JSONDecodeError, RuntimeError, ValueError) as exc:
                generation_error = repr(exc)

        return VerificationResult(generation_error=generation_error, responses=response_objects)

    async def _generate_user_agent_environment_evaluation(
        self, state: SyntheticSessionState, user_message: str
    ) -> tuple[Optional[VerificationResult], VerificationResult, Optional[VerificationResult]]:
        if self.config.judge_model_server is None:
            raise RuntimeError("judge_model_server is required for LLM trajectory verification.")

        system_message = self._combined_evaluation_system_message(state)
        response_objects: List[Dict[str, Any]] = []
        generation_error: Optional[str] = None

        for _ in range(max(1, self.config.generation_attempts)):
            try:
                response = await self._call_judge_model(
                    model_server=self.config.judge_model_server,
                    params=self.config.judge_responses_create_params,
                    messages=[
                        NeMoGymEasyInputMessage(role="system", content=system_message),
                        NeMoGymEasyInputMessage(role="user", content=user_message),
                    ],
                )
                response_objects.append(response.model_dump())
                response_text = self._last_text(response)
                canonical_text = self._strip_json_fence(response_text)
                evaluation = UserAgentEnvironmentEvaluation.model_validate_json(canonical_text)
                user_verification_result = VerificationResult(
                    reward=int(evaluation.customer_success),
                    explanation=evaluation.customer_explanation,
                )
                agent_verification_result = VerificationResult(
                    reward=int(evaluation.representative_success),
                    explanation=evaluation.representative_explanation,
                    judge_response=canonical_text,
                    responses=response_objects,
                )
                environment_verification_result = VerificationResult(
                    reward=int(evaluation.tool_results_success),
                    explanation=evaluation.tool_results_explanation,
                )
                return user_verification_result, agent_verification_result, environment_verification_result
            except (ValidationError, JSONDecodeError, RuntimeError, ValueError) as exc:
                generation_error = repr(exc)

        agent_verification_result = VerificationResult(
            generation_error=generation_error,
            responses=response_objects,
        )
        return None, agent_verification_result, None

    def _system_messages(self, state: SyntheticSessionState) -> Dict[Source | TrajectoryEvaluationType, str]:
        domain_policy = state.policy
        customer_scenario_string = state.customer_scenario.create_string()
        evaluation_schema_string = json.dumps(Evaluation.model_json_schema())
        tool_definition_string = self._format_tool_definitions(state.tool_signatures)

        system_messages: Dict[Source | TrajectoryEvaluationType, str] = {}
        system_messages[Source.USER] = self.USER_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE.format(
            policy=domain_policy,
            customer_scenario=customer_scenario_string,
            complete_indicator=TRAJECTORY_COMPLETE_INDICATOR,
            transfer_indicator=AGENT_TRANSFER_INDICATOR,
            evaluation_schema=evaluation_schema_string,
        )
        system_messages[Source.AGENT] = self.AGENT_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE.format(
            policy=domain_policy,
            tool_definitions=tool_definition_string,
            evaluation_schema=evaluation_schema_string,
        )
        system_messages[Source.ENVIRONMENT] = self.ENVIRONMENT_MESSAGE_EVALUATION_SYSTEM_MESSAGE_TEMPLATE.format(
            policy=domain_policy,
            customer_scenario=customer_scenario_string,
            tool_definitions=tool_definition_string,
            evaluation_schema=evaluation_schema_string,
        )

        transfer_instruction = "" if state.customer_scenario.outside_policy_scope else "not "
        system_messages[TrajectoryEvaluationType.AGENT_CONVERSATION] = (
            self.AGENT_CONVERSATION_EVALUATION_SYSTEM_MESSAGE_TEMPLATE.format(
                policy=domain_policy,
                tool_definitions=tool_definition_string,
                transfer_instruction=transfer_instruction,
                evaluation_schema=evaluation_schema_string,
            )
        )
        return system_messages

    def _combined_evaluation_system_message(self, state: SyntheticSessionState) -> str:
        transfer_instruction = "" if state.customer_scenario.outside_policy_scope else "not "
        return self.USER_AGENT_ENVIRONMENT_CONVERSATION_EVALUATION_SYSTEM_MESSAGE_TEMPLATE.format(
            policy=state.policy,
            tool_definitions=self._format_tool_definitions(state.tool_signatures),
            customer_scenario=state.customer_scenario.create_string(),
            complete_indicator=TRAJECTORY_COMPLETE_INDICATOR,
            transfer_indicator=AGENT_TRANSFER_INDICATOR,
            transfer_instruction=transfer_instruction,
            evaluation_schema=json.dumps(UserAgentEnvironmentEvaluation.model_json_schema()),
        )

    async def _call_judge_model(
        self,
        *,
        model_server: ModelServerRef,
        params: NeMoGymResponseCreateParamsNonStreaming,
        messages: List[NeMoGymEasyInputMessage],
    ) -> NeMoGymResponse:
        attempts = self.config.judge_provider_attempts
        for attempt in range(1, attempts + 1):
            try:
                return await self._call_model(
                    model_server=model_server,
                    params=params,
                    messages=messages,
                )
            except Exception as exc:
                provider_error = self._find_judge_provider_error(exc)
                if provider_error is None:
                    raise
                if not self._is_retryable_judge_provider_error(provider_error):
                    status_code = provider_error.status if isinstance(provider_error, ClientResponseError) else 500
                    raise JudgeProviderRequestError(
                        status_code,
                        f"Judge provider request failed: {type(provider_error).__name__}: {provider_error}",
                    ) from exc
                if attempt >= attempts:
                    raise JudgeProviderExhaustedError(
                        "Judge provider request failed after "
                        f"{attempts} attempts: {type(provider_error).__name__}: {provider_error}"
                    ) from exc

                delay = min(
                    self.config.judge_provider_retry_initial_backoff_seconds * (2 ** (attempt - 1)),
                    self.config.judge_provider_retry_max_backoff_seconds,
                )
                if delay > 0:
                    await asyncio.sleep(delay)

        raise AssertionError("Judge provider retry loop exited unexpectedly.")

    def _find_judge_provider_error(self, exc: BaseException) -> Optional[BaseException]:
        pending: List[BaseException] = [exc]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, (ClientResponseError, ClientConnectionError, asyncio.TimeoutError)):
                return current
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return None

    def _is_retryable_judge_provider_error(self, exc: BaseException) -> bool:
        if isinstance(exc, ClientResponseError):
            return exc.status in {408, 409, 425, 429} or exc.status >= 500
        return isinstance(exc, (ClientConnectionError, asyncio.TimeoutError))

    async def _call_model(
        self,
        *,
        model_server: ModelServerRef,
        params: NeMoGymResponseCreateParamsNonStreaming,
        messages: List[NeMoGymEasyInputMessage],
    ) -> NeMoGymResponse:
        request_body = params.model_copy(deep=True)
        request_body.input = messages
        response = await self.server_client.post(
            server_name=model_server.name,
            url_path="/v1/responses",
            json=request_body,
        )
        await raise_for_status(response)
        try:
            return NeMoGymResponse.model_validate(await get_response_json(response))
        except ValidationError as exc:
            raise RuntimeError("Model server returned an invalid Responses API payload.") from exc

    def _first_response_text(self, response: NeMoGymResponse) -> str:
        for output in response.output:
            if getattr(output, "type", None) != "message":
                continue
            content = getattr(output, "content", [])
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    return text
        raise RuntimeError(f"A response does not contain a message output: {response.output}")

    def _last_text(self, response: NeMoGymResponse) -> str:
        return self._first_response_text(response)

    def _format_message_list(self, messages: List[SyntheticMessage]) -> str:
        return "\n\n".join(self._format_message(message) for message in messages)

    def _format_message(self, message: SyntheticMessage) -> str:
        if message.type == MessageType.TEXT:
            if message.source == Source.USER:
                sender = "Customer"
            elif message.source == Source.AGENT:
                sender = "Representative"
            else:
                raise NotImplementedError
            return self.TEXT_MESSAGE_TEMPLATE.format(sender=sender, content=message.content)

        if message.type == MessageType.TOOL_CALL:
            return self.TOOL_CALL_MESSAGE_TEMPLATE.format(
                execution_id=message.tool_call_id,
                tool_name=message.tool_name,
                arguments=message.arguments,
            )

        if message.type == MessageType.TOOL_EXECUTION:
            return self.TOOL_EXECUTION_MESSAGE_TEMPLATE.format(
                execution_id=message.tool_call_id,
                execution_result=message.content,
            )

        raise NotImplementedError

    def _format_tool_definitions(self, tool_signatures: List[ToolSignature]) -> str:
        tool_definition_list = [
            self.TOOL_DEFINITION_TEMPLATE.format(
                name=tool_signature.name,
                documentation=tool_signature.normalized_doc,
                parameters=json.dumps(tool_signature.normalized_params),
                return_type=json.dumps(tool_signature.returns),
            )
            for tool_signature in tool_signatures
        ]
        return "\n\n".join(tool_definition_list)

    def _strip_json_fence(self, text: str) -> str:
        return text.strip().removeprefix("```json").removesuffix("```")

    def _validate_message(
        self, state: SyntheticSessionState, message: SyntheticMessage
    ) -> Optional[tuple[TrajectoryInvalidReason, str]]:
        if message.type == MessageType.TEXT:
            return self._validate_text_message(state, message)
        if message.type == MessageType.TOOL_CALL:
            return self._validate_tool_call_message(state, message)
        if message.type == MessageType.TOOL_EXECUTION:
            return self._validate_tool_execution_message(state, message)
        return None

    def _validate_text_message(
        self, state: SyntheticSessionState, message: SyntheticMessage
    ) -> Optional[tuple[TrajectoryInvalidReason, str]]:
        if message.source == Source.USER and len(state.messages) < 1 and self._is_trajectory_complete_message(message):
            return TrajectoryInvalidReason.INVALID_USER_MESSAGE, (
                "The first message in a trajectory from the user contains an "
                f"indicator that the trajectory is complete: {message.content}"
            )
        return None

    def _validate_tool_call_message(
        self, state: SyntheticSessionState, message: SyntheticMessage
    ) -> Optional[tuple[TrajectoryInvalidReason, str]]:
        tool_name = message.tool_name or ""
        tool = self._find_tool(state, tool_name)
        if tool is None:
            return TrajectoryInvalidReason.INVALID_AGENT_TOOL_CALL, (
                f"The tool {tool_name} in the tool call with the ID {message.tool_call_id} is unknown"
            )

        parameters_schema = tool.normalized_params
        if parameters_schema is None:
            return None

        try:
            deserialized_arguments = json.loads(message.arguments or "")
            self._validator_for(parameters_schema).validate(deserialized_arguments)
        except JSONDecodeError as decode_error:
            return TrajectoryInvalidReason.INVALID_AGENT_TOOL_CALL, (
                "The arguments string in the tool call with the ID "
                f"{message.tool_call_id} could not be converted to a JSON object: "
                f"{decode_error.msg}"
            )
        except JsonSchemaValidationError as validation_error:
            return TrajectoryInvalidReason.INVALID_AGENT_TOOL_CALL, (
                "The arguments object in the tool call with the ID "
                f"{message.tool_call_id} is not valid under the {tool_name} tool "
                f"parameters JSON Schema: {validation_error.message}"
            )

        message.deserialized_arguments = deserialized_arguments
        return None

    def _validate_tool_execution_message(
        self, state: SyntheticSessionState, message: SyntheticMessage
    ) -> Optional[tuple[TrajectoryInvalidReason, str]]:
        tool_name = message.tool_name or ""
        tool = self._find_tool(state, tool_name)
        if tool is None:
            return TrajectoryInvalidReason.INVALID_ENVIRONMENT_TOOL_EXECUTION, (
                f"The tool {tool_name} in the tool execution with the ID {message.tool_call_id} is unknown"
            )
        if tool.returns is None:
            return None
        try:
            execution_result = str(message.content)
            if execution_result.strip().startswith("```json"):
                execution_result = self._strip_json_fence(execution_result)
            return_value = json.loads(execution_result)
            self._validator_for(tool.returns).validate(return_value)
        except JSONDecodeError as decode_error:
            return TrajectoryInvalidReason.INVALID_ENVIRONMENT_TOOL_EXECUTION, (
                "The execution result for the tool call with the ID "
                f"{message.tool_call_id} could not be converted to a JSON object: "
                f"{decode_error.msg}"
            )
        except JsonSchemaValidationError as validation_error:
            return TrajectoryInvalidReason.INVALID_ENVIRONMENT_TOOL_EXECUTION, (
                "The execution result for the tool call with the ID "
                f"{message.tool_call_id} is not valid under the {tool_name} tool "
                f"return type JSON Schema: {validation_error.message}"
            )

        message.deserialized_execution_result = return_value
        return None

    def _validator_for(self, schema: Dict[str, Any]):
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
        return validator_class(schema)

    def _is_trajectory_complete_message(self, message: SyntheticMessage) -> bool:
        if message.type != MessageType.TEXT:
            return False
        message_text = str(message.content)
        return TRAJECTORY_COMPLETE_INDICATOR in message_text or AGENT_TRANSFER_INDICATOR in message_text

    def _trajectory_diagnostics(self, state: SyntheticSessionState) -> Dict[str, Any]:
        num_tool_schema_failures = sum(
            1
            for message in state.messages
            if message.type == MessageType.TOOL_EXECUTION and message.schema_valid is False
        )
        transferred = any(
            isinstance(message.content, str) and AGENT_TRANSFER_INDICATOR in message.content
            for message in state.messages
            if message.source == Source.USER
        )
        outside_policy_scope = bool(state.customer_scenario.outside_policy_scope)
        transfer_ground_truth_enforced = self.config.enforce_transfer_ground_truth
        return {
            "num_user_messages": sum(1 for message in state.messages if message.source == Source.USER),
            "num_agent_messages": sum(1 for message in state.messages if message.source == Source.AGENT),
            "num_tool_calls": sum(1 for message in state.messages if message.type == MessageType.TOOL_CALL),
            "num_tool_results": sum(1 for message in state.messages if message.type == MessageType.TOOL_EXECUTION),
            "num_tool_schema_failures": num_tool_schema_failures,
            "outside_policy_scope": outside_policy_scope,
            "transferred": transferred,
            "transfer_ground_truth_enforced": transfer_ground_truth_enforced,
            "transfer_ground_truth_mismatch": (transfer_ground_truth_enforced and transferred != outside_policy_scope),
            "judge_skipped_for_transfer_mismatch": False,
        }

    def _trajectory_result(self, state: SyntheticSessionState) -> Dict[str, Any]:
        return {
            "trajectory": {
                "messages": [self._message_dict(message) for message in state.messages],
                "prefill_message_count": state.prefill_message_count,
                "continuation_start_index": state.prefill_message_count,
                "terminal_state": state.terminal_state,
                "generation_invalid_reason": state.generation_invalid_reason,
                "terminal_error": state.terminal_error,
                "agent_verification_result": None
                if state.agent_verification_result is None
                else state.agent_verification_result.model_dump(mode="json"),
                "user_verification_result": None
                if state.user_verification_result is None
                else state.user_verification_result.model_dump(mode="json"),
                "environment_verification_result": None
                if state.environment_verification_result is None
                else state.environment_verification_result.model_dump(mode="json"),
            }
        }

    def _message_dict(self, message: SyntheticMessage) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": str(message.type),
            "source": str(message.source),
        }
        if message.responses is not None:
            payload["responses"] = message.responses
        if message.type == MessageType.TEXT:
            payload["text"] = message.content
        elif message.type == MessageType.TOOL_CALL:
            payload.update(
                {
                    "tool_call_id": message.tool_call_id,
                    "tool_name": message.tool_name,
                    "arguments": message.arguments,
                    "deserialized_arguments": message.deserialized_arguments,
                }
            )
        elif message.type == MessageType.TOOL_EXECUTION:
            payload.update(
                {
                    "tool_call_id": message.tool_call_id,
                    "tool_name": message.tool_name,
                    "arguments": message.arguments,
                    "execution_result": message.content,
                    "deserialized_execution_result": message.deserialized_execution_result,
                    "schema_valid": message.schema_valid,
                }
            )
        if message.verification_result is not None:
            payload["verification_result"] = message.verification_result.model_dump(mode="json", exclude_none=True)
        return payload

    def _failure_count_metrics(self, failure_labels: List[str]) -> Dict[str, int]:
        return {
            "num_user_failures": failure_labels.count(VerificationFailureLabel.USER_FAILURE),
            "num_agent_failures": failure_labels.count(VerificationFailureLabel.AGENT_FAILURE),
            "num_tool_failures": failure_labels.count(VerificationFailureLabel.TOOL_FAILURE),
        }

    def _should_mask_sample(self, state: SyntheticSessionState) -> bool:
        failure_labels = set(state.failure_labels)
        if failure_labels & {
            VerificationFailureLabel.USER_FAILURE,
            VerificationFailureLabel.TOOL_FAILURE,
            VerificationFailureLabel.VERIFICATION_GENERATION_ERROR,
        }:
            return True

        return (
            VerificationFailureLabel.TRAJECTORY_FAILURE in failure_labels
            and state.generation_invalid_reason == TrajectoryInvalidReason.MESSAGE_GENERATION_ERROR
        )

    def _dedupe_strings(self, values: List[Any]) -> List[str]:
        deduped = []
        for value in values:
            text = str(value)
            if text not in deduped:
                deduped.append(text)
        return deduped

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        flat = [rollout for task in tasks for rollout in task]
        if not flat:
            return {}
        return {
            "synthetic_tool_use/transfer_rate": sum(float(row.get("transferred", False)) for row in flat) / len(flat),
            "synthetic_tool_use/transfer_ground_truth_mismatch_rate": sum(
                float(row.get("transfer_ground_truth_mismatch", False)) for row in flat
            )
            / len(flat),
            "synthetic_tool_use/transfer_mismatch_judge_skip_rate": sum(
                float(row.get("judge_skipped_for_transfer_mismatch", False)) for row in flat
            )
            / len(flat),
            "synthetic_tool_use/tool_schema_failure_rate": sum(
                1.0 for row in flat if row.get("num_tool_schema_failures", 0) > 0
            )
            / len(flat),
            "synthetic_tool_use/user_failure_rate": sum(1.0 for row in flat if row.get("num_user_failures", 0) > 0)
            / len(flat),
            "synthetic_tool_use/tool_failure_rate": sum(1.0 for row in flat if row.get("num_tool_failures", 0) > 0)
            / len(flat),
        }

    def get_key_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: metrics[key]
            for key in (
                "mean/reward",
                "reward/mean",
                "synthetic_tool_use/transfer_rate",
                "synthetic_tool_use/transfer_ground_truth_mismatch_rate",
                "synthetic_tool_use/transfer_mismatch_judge_skip_rate",
                "synthetic_tool_use/tool_schema_failure_rate",
                "synthetic_tool_use/user_failure_rate",
                "synthetic_tool_use/tool_failure_rate",
            )
            if key in metrics
        }


if __name__ == "__main__":
    SyntheticToolUseSimulationServer.run_webserver()
