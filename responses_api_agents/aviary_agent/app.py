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

import aiohttp
from pydantic import ConfigDict, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInput,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
)
from resources_servers.aviary.schemas import (
    AviaryAgentVerifyRequest,
    AviaryAgentVerifyResponse,
    AviaryEnvStateEasyInputMessage,
    AviaryNeMoGymResponse,
    AviarySeedSessionResponse,
    AviaryStepResponse,
)


logger = logging.getLogger(__name__)


class AviaryAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: int | None = Field(
        default=None,
        description="The maximum number of steps to take in the environment. "
        "If not set, the agent will run indefinitely.",
    )
    return_transitions: bool = Field(
        default=True,
        description="If True, return a list of transitions, instead of the "
        "whole trajectory as a single NeMoGymResponseOutputItem.",
    )

    # Doesn't cause an issue if not set, but if it is, then
    # we can avoid sending requests that are guaranteed to
    # exceed the limit. If not set, vLLM will reject the request
    # for us (but also clutter logs with exceptions).
    # TODO: see if we can retrieve this from /models endpoint
    max_total_sequence_length: int | None = Field(
        default=None,
        description="If set, the rollout will stop when the agent state exceeds this length. "
        "If not set, will rely on a vLLM exception to tell us when we've exceeded the model's "
        "token limit. Setting this simply avoids that exception.",
    )

    done_if_no_tool_calls: bool = Field(
        default=True, description="If True, end the rollout if the model does not call any tools."
    )

    collapse_old_env_states: bool = Field(
        default=False,
        description="If True, collapse previous Aviary EnvStateMessages into a hidden message "
        "in the agent state. Can be used to compress the context, if supported by the environment.",
    )
    old_env_state_message: str = Field(
        default="[Previous environment state - hidden]",
        description="The message to use when collapsing previous Aviary EnvStateMessages into a hidden message.",
    )


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
        successful_transition: bool,
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        """Update the agent state.

        Separate method so subclasses can override.
        """

        prev_messages = agent_state.input
        if successful_transition and self.config.collapse_old_env_states:
            # only collapse if we had a successful transition - otherwise we'd be hiding previous
            # env state without supplying a new one in obs.
            hidden_message = NeMoGymEasyInputMessage(role="user", content=self.config.old_env_state_message)
            prev_messages = [
                hidden_message if isinstance(m, AviaryEnvStateEasyInputMessage) else m for m in prev_messages
            ]

        return agent_state.model_copy(update={"input": prev_messages + model_output + obs})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=5))
    async def _seed_session(self, task_idx: int) -> AviarySeedSessionResponse:
        reset_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json={"task_idx": task_idx},
        )
        reset_response.raise_for_status()
        seed_session_response = AviarySeedSessionResponse.model_validate(await reset_response.json())
        if not seed_session_response.obs:
            raise ValueError("No observations in seed session response")
        return seed_session_response

    async def responses(self, req: AviaryAgentRunRequest) -> AviaryNeMoGymResponse:
        req = req.model_copy(deep=True)
        body = req.responses_create_params

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        seed_session_response = await self._seed_session(req.task_idx)

        agent_state = body.model_copy(
            update={"input": body.input + seed_session_response.obs, "tools": seed_session_response.tools}
        )

        env_id = seed_session_response.env_id
        model_response: NeMoGymResponse | None = None
        agent_state_history: list[NeMoGymResponseInput] = []
        all_messages: list[NeMoGymResponseOutputItem] = []
        model_server_cookies = None

        step = 0
        try:
            while True:
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    break
                step += 1
                successful_transition = True

                # Sample action from model
                try:
                    raw_model_response = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path="/v1/responses",
                        json=agent_state,
                        cookies=model_server_cookies,
                    )
                    raw_model_response.raise_for_status()
                    model_server_cookies = raw_model_response.cookies
                    model_response_json = await raw_model_response.json()
                except (json.JSONDecodeError, aiohttp.ClientResponseError) as e:
                    # JSONDecodeError will be thrown if there's an underlying openai error.
                    # For now, we break. Default reward of 0 will be returned when /verify is called.
                    logger.warning(f"Error calling /v1/responses: {e!r}. Response: {raw_model_response.text!r}.")
                    break

                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as e:
                    logger.warning(f"Error validating model response: {e!r}. Response: {model_response_json!r}.")
                    break

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
                    if self.config.done_if_no_tool_calls:
                        done = True
                        obs = []
                    else:
                        # Got non-tool-call outputs, so ask the model to try again.
                        obs: Sequence[NeMoGymEasyInputMessage | NeMoGymFunctionCallOutput] = [
                            NeMoGymEasyInputMessage(
                                role="user",
                                content="You did not respond with a valid tool call. "
                                "This may mean you did not call tools, or you tried to "
                                "and got the formatting, tool name, or arguments "
                                "wrong. To proceed, please call at least one tool.",
                            )
                        ]
                        successful_transition = False
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

                agent_state = self.update_agent_state(agent_state, model_output, obs, successful_transition)
                if self.config.return_transitions:
                    agent_state_history.append(cast(NeMoGymResponseInput, agent_state.input))
                else:
                    all_messages.extend(model_output)
                    if successful_transition:
                        all_messages.extend(obs)

                if done:
                    break

        finally:
            await self.server_client.post(
                server_name=self.config.resources_server.name, url_path="/close", json={"env_id": env_id}
            )

        assert model_response is not None, (
            "Rollout crashed or terminated before first transition completed, cannot proceed."
        )

        output_overrides = {
            "env_id": env_id,
            "group_id": str(req.task_idx),
            "contains_transitions": self.config.return_transitions,
            "output": agent_state_history if self.config.return_transitions else all_messages,
        }
        output = AviaryNeMoGymResponse.model_validate(model_response.model_dump() | output_overrides)
        return output

    async def run(self, body: AviaryAgentRunRequest) -> AviaryAgentVerifyResponse:
        try:
            response = await self.responses(body)

            verify_request = AviaryAgentVerifyRequest.model_validate(body.model_dump() | {"response": response})
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
            )

            return AviaryAgentVerifyResponse.model_validate(await verify_response.json())
        except Exception as e:
            logger.exception("Error in run")
            raise e


if __name__ == "__main__":
    AviaryAgent.run_webserver()
