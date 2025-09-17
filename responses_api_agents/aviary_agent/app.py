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
from typing import List

from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.integrations.aviary import (
    AviaryAgentVerifyRequest,
    AviaryAgentVerifyResponse,
    AviaryNeMoGymResponse,
    AviarySeedSessionResponse,
    AviaryStepResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)


class AviaryAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class AviaryAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    max_steps: int | None = None


class AviaryAgent(SimpleResponsesAPIAgent):
    config: AviaryAgentConfig

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
        seed_session_response = AviarySeedSessionResponse.model_validate(reset_response.json())

        agent_state = body.model_copy(
            update={"input": body.input + seed_session_response.obs, "tools": seed_session_response.tools}
        )

        env_id = seed_session_response.env_id
        model_response: NeMoGymResponse | None = None

        steps = 0
        while True:
            if req.max_steps is not None and steps >= req.max_steps:
                break
            steps += 1

            # Sample action from model
            raw_model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=agent_state,
            )
            model_response_json = raw_model_response.json()
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
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
                obs = [NeMoGymEasyInputMessage(role="user", content="Please call a tool to proceed.")]
            else:
                # Apply action to environment
                raw_env_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/step",
                    json={"action": [c.model_dump(mode="json") for c in all_fn_calls], "env_id": env_id},
                )
                env_response = AviaryStepResponse.model_validate(raw_env_response.json())
                obs = env_response.obs
                done = env_response.done

            agent_state = agent_state.model_copy(update={"input": agent_state.input + model_output + obs})
            if done:
                break

        await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/close",
            json={"env_id": env_id},
        )

        assert model_response is not None

        # This includes the observation messages from reset, which SimpleAgentStateful does not include.
        # TODO: understand if that's a problem
        output = AviaryNeMoGymResponse.model_validate(
            model_response.model_dump() | {"output": agent_state.input, "env_id": env_id}
        )
        return output

    async def run(self, body: AviaryAgentRunRequest) -> AviaryAgentVerifyResponse:
        response = await self.responses(body)

        verify_request = AviaryAgentVerifyRequest.model_validate(body.model_dump() | {"response": response})
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/verify", json=verify_request.model_dump()
        )

        return AviaryAgentVerifyResponse.model_validate(verify_response.json())


if __name__ == "__main__":
    AviaryAgent.run_webserver()
