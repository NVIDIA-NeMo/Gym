# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_gym.integrations.atropos import (
    AtroposAgentVerifyRequest,
    AtroposAgentVerifyResponse,
    AtroposNeMoGymResponse,
    AtroposSeedSessionResponse,
    AtroposStepResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
)


class AtroposAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class AtroposAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    task_idx: int
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    max_steps: int | None = None


class AtroposAgent(SimpleResponsesAPIAgent):
    config: AtroposAgentConfig

    async def responses(self, req: AtroposAgentRunRequest) -> AtroposNeMoGymResponse:
        req = req.model_copy(deep=True)
        body = req.responses_create_params

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        reset_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json={"task_idx": req.task_idx},
        )
        seed_session_response = AtroposSeedSessionResponse.model_validate(reset_response.json())
        agent_state = body.model_copy(update={"input": body.input + seed_session_response.obs})
        env_id = seed_session_response.env_id
        model_response: NeMoGymResponse | None = None
        steps = 0
        done = False

        while not done:
            if req.max_steps is not None and steps >= req.max_steps:
                break
            steps += 1

            raw_model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=agent_state,
            )
            try:
                model_response = NeMoGymResponse.model_validate(raw_model_response.json())
            except ValidationError as e:
                raise RuntimeError(f"Invalid model server response: {raw_model_response.json()}") from e

            model_output = model_response.output
            all_output_messages = [o for o in model_output if o.type == "message" and o.role == "assistant"]

            if all_output_messages:
                content_items = all_output_messages[0].content
                text_parts = [
                    item.text if hasattr(item, "text") else item.get("text", "")
                    for item in content_items
                ]
                response_text = "".join(text_parts)
            else:
                response_text = str(model_output[0]) if model_output else ""

            raw_env_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/step",
                json={"action": response_text, "env_id": env_id},
            )
            env_response = AtroposStepResponse.model_validate(raw_env_response.json())
            done = env_response.done
            agent_state = agent_state.model_copy(update={"input": agent_state.input + model_output + env_response.obs})

        assert model_response is not None
        return AtroposNeMoGymResponse.model_validate(
            model_response.model_dump() | {"output": agent_state.input, "env_id": env_id}
        )

    async def run(self, body: AtroposAgentRunRequest) -> AtroposAgentVerifyResponse:
        response = await self.responses(body)
        verify_request = AtroposAgentVerifyRequest.model_validate(body.model_dump() | {"response": response})
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/verify", json=verify_request.model_dump()
        )
        await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/close",
            json={"env_id": response.env_id},
        )
        return AtroposAgentVerifyResponse.model_validate(verify_response.json())


if __name__ == "__main__":
    AtroposAgent.run_webserver()
