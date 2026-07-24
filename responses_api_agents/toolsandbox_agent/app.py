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
"""Agent harness for the ToolSandbox resources server.

ToolSandbox is a conversation, not a pure tool loop. When the model replies in natural language
(no tool calls) that reply must go to the **user simulator** and the episode
continues — so this harness forwards the *whole* model output (assistant text
AND tool calls) to ``/step`` and lets the resources server decide ``done``,
rather than ending the rollout on a no-tool-call turn.
"""

import json
import logging
from typing import List, cast

import aiohttp
from pydantic import ConfigDict, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInput,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
)
from resources_servers.toolsandbox.schemas import (
    ToolSandboxNeMoGymResponse,
    ToolSandboxSeedSessionResponse,
    ToolSandboxStepResponse,
    ToolSandboxVerifyRequest,
    ToolSandboxVerifyResponse,
)


logger = logging.getLogger(__name__)


class ToolSandboxAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: int = Field(
        default=30,
        description="Maximum agent turns before the episode is force-ended.",
    )
    return_transitions: bool = Field(
        default=False,
        description="If True, return per-transition agent-state snapshots "
        "(a list of lists) instead of a single flat message list. Keep False "
        "for callers that parse response.output as a flat list.",
    )


class ToolSandboxAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )


def _assistant_text(messages: List[NeMoGymResponseOutputMessage]) -> str:
    """Flatten assistant output_text parts into a single string."""
    parts: List[str] = []
    for msg in messages:
        for content in msg.content:
            if getattr(content, "type", None) == "output_text":
                parts.append(content.text)
    return "".join(parts)


class ToolSandboxAgent(SimpleResponsesAPIAgent):
    config: ToolSandboxAgentConfig

    def update_agent_state(
        self,
        agent_state: NeMoGymResponseCreateParamsNonStreaming,
        model_output: list,
        obs: list,
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        return agent_state.model_copy(update={"input": agent_state.input + model_output + obs})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=5))
    async def _seed_session(self, task_idx: int) -> ToolSandboxSeedSessionResponse:
        reset_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json={"task_idx": task_idx},
        )
        reset_response.raise_for_status()
        seed = ToolSandboxSeedSessionResponse.model_validate(await reset_response.json())
        if not seed.obs:
            raise ValueError("No observations in seed session response")
        return seed

    async def responses(self, req: ToolSandboxAgentRunRequest) -> ToolSandboxNeMoGymResponse:
        req = req.model_copy(deep=True)
        body = req.responses_create_params
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        seed = await self._seed_session(req.task_idx)
        agent_state = body.model_copy(update={"input": body.input + seed.obs, "tools": seed.tools})
        env_id = seed.env_id

        model_response: NeMoGymResponse | None = None
        agent_state_history: list[NeMoGymResponseInput] = []
        all_messages: list[NeMoGymResponseOutputItem] = []
        model_server_cookies = None

        step = 0
        try:
            while step < self.config.max_steps:
                step += 1

                # Sample the next agent turn from the policy model.
                try:
                    raw = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path="/v1/responses",
                        json=agent_state,
                        cookies=model_server_cookies,
                    )
                    raw.raise_for_status()
                    model_server_cookies = raw.cookies
                    model_response_json = await raw.json()
                except (json.JSONDecodeError, aiohttp.ClientResponseError) as e:
                    logger.warning(f"Error calling /v1/responses: {e!r}.")
                    break

                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as e:
                    logger.warning(f"Error validating model response: {e!r}.")
                    break

                model_output = model_response.output
                fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                    o for o in model_output if o.type == "function_call"
                ]
                output_messages: List[NeMoGymResponseOutputMessage] = [
                    o for o in model_output if o.type == "message" and o.role == "assistant"
                ]

                # Nothing actionable from the model -> end the episode.
                if not fn_calls and not output_messages:
                    break

                # Tool calls take precedence over text (matches ToolSandbox's
                # either/or agent semantics). Otherwise the text goes to the
                # user simulator.
                step_payload = {
                    "env_id": env_id,
                    "function_calls": [c.model_dump(mode="json") for c in fn_calls],
                    "text": None if fn_calls else _assistant_text(output_messages),
                }
                raw_step = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/step",
                    json=step_payload,
                )
                raw_step.raise_for_status()
                step_response = ToolSandboxStepResponse.model_validate(await raw_step.json())
                obs = step_response.obs
                done = step_response.done

                agent_state = self.update_agent_state(agent_state, model_output, obs)
                if self.config.return_transitions:
                    agent_state_history.append(cast(NeMoGymResponseInput, agent_state.input))
                else:
                    all_messages.extend(model_output)
                    all_messages.extend(obs)

                if done:
                    break
        finally:
            await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close",
                json={"env_id": env_id},
            )

        assert model_response is not None, "Rollout terminated before the first agent turn completed."

        output_overrides = {
            "env_id": env_id,
            "group_id": str(req.task_idx),
            "contains_transitions": self.config.return_transitions,
            "output": agent_state_history if self.config.return_transitions else all_messages,
        }
        return ToolSandboxNeMoGymResponse.model_validate(model_response.model_dump() | output_overrides)

    async def run(self, body: ToolSandboxAgentRunRequest) -> ToolSandboxVerifyResponse:
        try:
            response = await self.responses(body)
            verify_request = ToolSandboxVerifyRequest.model_validate(body.model_dump() | {"response": response})
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
            )
            return ToolSandboxVerifyResponse.model_validate(await verify_response.json())
        except Exception:
            logger.exception("Error in run")
            raise


if __name__ == "__main__":
    ToolSandboxAgent.run_webserver()
