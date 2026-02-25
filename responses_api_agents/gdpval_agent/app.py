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
from typing import List

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInput,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status

FINISH_TOOL_NAME = "finish"

class GDPValAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 100
    max_tokens: int = 10000
    context_summarization_cutoff: float = 0.7
    step_warning_threshold: int | None = 80


class GDPValAgentRunRequest(BaseRunRequest):
    session_id: str
    model_config = ConfigDict(extra="allow")


class GDPValAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class GDPValAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class GDPValAgent(SimpleResponsesAPIAgent):
    config: GDPValAgentConfig

    def __init__(self):
        super().__init__()
        self.model_server_cookies = None
        self.resources_server_cookies = None

    async def single_response(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=self.model_server_cookies,
        )
        # We raise for status here since we expect model calls to always work.
        await raise_for_status(model_response)
        model_response_json = await get_response_json(model_response)
        self.model_server_cookies = model_response.cookies

        try:
            model_response = NeMoGymResponse.model_validate(model_response_json)
        except ValidationError as e:
            raise RuntimeError(
                f"Received an invalid response from model server: {json.dumps(model_response_json)}"
            ) from e

        return model_response

    
    async def run_tool(self, call: NeMoGymResponseFunctionToolCall) -> NeMoGymFunctionCallOutput:
        api_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=f"/{call.name}",
            json=json.loads(call.arguments),
            cookies=self.resources_server_cookies,
        )
        # We don't raise for status here since it's a valid return for the API to error e.g. 
        # if the model outputs an invalid call or something.
        self.resources_server_cookies = api_response.cookies

        tool_response = NeMoGymFunctionCallOutput(
            type="function_call_output",
            call_id=call.call_id,
            output=(await api_response.content.read()).decode(),
        )

        return tool_response


    async def step(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> tuple[List[NeMoGymResponseOutputMessage], List[NeMoGymFunctionCallOutput]]:
        """Execute one agent step: generate assistant message and run any requested tool calls.

        Args:
            body: Current conversation messages

        Returns the model outputs and tool outputs.

        """
        model_outputs = []
        tool_outputs = []
        finished = False

        model_response = await self.single_response(body)
        output = model_response.output
        model_outputs.extend(output)

        if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
            return model_outputs, tool_outputs, finished

        function_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
    
        for call in function_calls:
            tool_response = await self.run_tool(call)
            tool_outputs.append(tool_response)

            if tool_response.name == FINISH_TOOL_NAME and tool_response.status == "completed":
                finished = True

        return model_outputs, tool_outputs, finished


    def _get_steps_remaining_msg(self, remaining_steps: int):
        """Create a user message warning the agent about remaining turns before max_steps is reached."""
        if remaining_steps == 1:
            return NeMoGymEasyInputMessage(
                role="system",
                content="This is the last turn. Please finish the task by calling the finish tool."
            )

        return NeMoGymEasyInputMessage(
            role="system",
            content=(
                f"You have {remaining_steps} turns remaining to complete the task. "
                "Please continue. Remember you will need a separate turn to finish the task."
            )
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
        system_prompt: str = None,
        init_user_prompt: str = None,
    ) -> NeMoGymResponse:

        # 1. Init message history as body input
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            user_message = [NeMoGymEasyInputMessage(role="user", content=body.input)]
            body.input = user_message
        if init_user_prompt is not None and isinstance(body.input, NeMoGymResponseInput):
            user_message = [NeMoGymEasyInputMessage(role="user", content=init_user_prompt)]
            body.input.extend(user_message)

        if system_prompt is not None:
            system_message = [NeMoGymEasyInputMessage(role="system", content=system_prompt)]
            body.input = system_message + body.input

        # 2. Reset cookies
        self.model_server_cookies = None  # update the cookies on every model response
        self.resources_server_cookies = request.cookies  # update the cookies on every resources server response

        max_steps = self.config.max_steps
        warning_threshold = self.config.step_warning_threshold
        summary_cutoff = self.config.context_summarization_cutoff
        outputs = []

        # 3. Iterate until max number of turns is reached or finish tool is called
        for step_num in range(max_steps):

            # 3.1. Add warning message if max steps is near
            if warning_threshold is not None and \
                max_steps - step_num <= warning_threshold and step_num != 0:
                num_steps_remaining_msg = self._get_steps_remaining_msg(max_steps - step_num)
                body = body.model_copy(update={"input": body.input + [num_steps_remaining_msg]})
                outputs.append(num_steps_remaining_msg)

            # TODO: Add text only tool call message functionality
            # 3.2. Execute one turn of the agent
            model_outputs, tool_outputs, finished = self.step(body)
            body = body.model_copy(update={"input": body.input + model_outputs + tool_outputs})
            outputs.extend(model_outputs)
            outputs.extend(tool_outputs)

            # TODO: Add file saving and local access functionality

            if finished:
                break

            # TODO: Add context summarization functionality
            # 3.3. Summarize context if needed
            if summary_cutoff is not None:
                continue

        # 4. Propogate any extra cookies necessary for downstream verification
        for k, v in (*self.resources_server_cookies.items(), *self.model_server_cookies.items()):
            response.set_cookie(k, v)

        final_response = NeMoGymResponse(output=outputs)
        return final_response


    async def run(self, request: Request, body: GDPValAgentRunRequest) -> GDPValAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies
        session_id = seed_session_response.json()["session_id"]

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies = response.cookies

        verify_request = GDPValAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(response)}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return GDPValAgentVerifyResponse.model_validate(await get_response_json(verify_response))


if __name__ == "__main__":
    GDPValAgent.run_webserver()
