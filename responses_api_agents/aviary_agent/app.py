# Copyright (c) 2025, NVIDIA CORPORATION, PLACEHOLDER.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
from collections.abc import Sequence
from typing import List, cast

from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.integrations.aviary import (
    AviaryAgentVerifyRequest,
    AviaryAgentVerifyResponse,
    AviaryEnvStateEasyInputMessage,
    AviaryNeMoGymResponse,
    AviarySeedSessionResponse,
    AviaryStepResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInput,
    NeMoGymResponseOutputMessage,
)


logger = logging.getLogger(__name__)


class AviaryAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: int | None = None

    # Doesn't cause an issue if not set, but if it is, then
    # we can avoid sending requests that are guaranteed to
    # exceed the limit. If not set, vLLM will reject the request
    # for us (but also clutter logs with exceptions).
    # TODO: see if we can retrieve this from /models endpoint
    max_total_sequence_length: int | None = None

    collapse_old_env_states: bool = False
    old_env_state_message: str = "[Previous environment state - hidden]"


class AviaryAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )


class AviaryAgent(SimpleResponsesAPIAgent):
    config: AviaryAgentConfig

    def update_agent_state(
        self,
        agent_state: NeMoGymResponseCreateParamsNonStreaming,
        model_output: list[NeMoGymResponseOutputMessage],
        obs: list[NeMoGymEasyInputMessage | NeMoGymFunctionCallOutput],
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        """Update the agent state.

        Separate method so subclasses can override.
        """

        prev_messages = agent_state.input
        if self.config.collapse_old_env_states:
            hidden_message = NeMoGymEasyInputMessage(role="user", content=self.config.old_env_state_message)
            prev_messages = [
                hidden_message if isinstance(m, AviaryEnvStateEasyInputMessage) else m for m in prev_messages
            ]

        return agent_state.model_copy(update={"input": agent_state.input + model_output + obs})

    async def responses(self, req: AviaryAgentRunRequest) -> AviaryNeMoGymResponse:
        req = req.model_copy(deep=True)
        body = req.responses_create_params

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        reset_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json={"task_idx": req.task_idx},
        )
        seed_session_response = AviarySeedSessionResponse.model_validate(await reset_response.json())

        agent_state = body.model_copy(
            update={"input": body.input + seed_session_response.obs, "tools": seed_session_response.tools}
        )

        env_id = seed_session_response.env_id
        model_response: NeMoGymResponse | None = None
        agent_state_history: list[NeMoGymResponseInput] = []

        total_len: int | None = None

        steps = 0
        while True:
            if self.config.max_steps is not None and steps >= self.config.max_steps:
                print("Done, max steps reached", flush=True)
                break
            steps += 1

            # Sample action from model
            try:
                # TODO: don't send this request if token count already exceeds configured limit
                # Instead, break the loop at the bottom.
                raw_model_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/v1/responses",
                    json=agent_state,
                )
                model_response_json = await raw_model_response.json()
            except json.JSONDecodeError as e:
                # JSONDecodeError will be thrown if there's an underlying openai error.
                # for now, we break. Default reward of 0 will be returned when /verify is called.
                logger.warning(
                    f"Error calling /v1/responses: {e!r}. Response: {raw_model_response.text!r}. Calculated length: {total_len}."
                )
                break

            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                # Maybe this should be handled as above? i.e. if we got an incomplete message back due to
                # max token limits
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            # Parse model response
            model_output = model_response.output
            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                o for o in model_output if o.type == "function_call"
            ]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in model_output if o.type == "message" and o.role == "assistant"
            ]
            done = False

            if not all_fn_calls and all_output_messages:
                # Got non-tool-call outputs, so ask the model to try again.
                obs: Sequence[NeMoGymEasyInputMessage | NeMoGymFunctionCallOutput] = [
                    NeMoGymEasyInputMessage(
                        role="user",
                        content="You either responded with no tool calls or an invalid tool call "
                        "(invalid tool name and/or arguments). Please call at least one tool to "
                        "proceed",
                    )
                ]
            else:
                # Apply action to environment
                raw_env_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/step",
                    json={"action": [c.model_dump(mode="json") for c in all_fn_calls], "env_id": env_id},
                )
                env_response = AviaryStepResponse.model_validate(await raw_env_response.json())
                obs = env_response.obs
                done = env_response.done

            agent_state = self.update_agent_state(agent_state, model_output, obs)
            agent_state_history.append(cast(NeMoGymResponseInput, agent_state.input))

            if self.config.max_total_sequence_length is not None:
                # NOTE: this assumes vLLM backend.
                tokenize_response = await self.server_client.post(
                    server_name=self.config.model_server.name, url_path="/tokenize", json=agent_state
                )
                tokenize_response_json = await tokenize_response.json()
                if tokenize_response_json["count"] >= self.config.max_total_sequence_length:
                    print("Done, max sequence length reached", flush=True)
                    break

            if done:
                print("Done, last tool call:", [c.name for c in all_fn_calls], flush=True)
                break

        await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/close",
            json={"env_id": env_id},
        )

        assert model_response is not None

        output = AviaryNeMoGymResponse.model_validate(
            model_response.model_dump()
            | {"output": agent_state_history, "env_id": env_id, "group_id": str(req.task_idx)}
        )
        return output

    async def run(self, body: AviaryAgentRunRequest) -> AviaryAgentVerifyResponse:
        try:
            response = await self.responses(body)

            verify_request = AviaryAgentVerifyRequest.model_validate(body.model_dump() | {"response": response})
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name, url_path="/verify", json=verify_request.model_dump()
            )

            return AviaryAgentVerifyResponse.model_validate(await verify_response.json())
        except Exception as e:
            logger.exception("Error in run")
            raise e


if __name__ == "__main__":
    AviaryAgent.run_webserver()
