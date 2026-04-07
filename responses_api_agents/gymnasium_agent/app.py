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

"""Agent for GymnasiumServer resources servers.

Drives the reset -> (model -> step)* loop.
"""

from fastapi import Body, Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.gymnasium import EnvResetResponse, EnvStepResponse, extract_text


class GymnasiumAgentConfig(BaseResponsesAPIAgentConfig):
    env_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 10


class GymnasiumAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class GymnasiumRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    terminated: bool = False
    truncated: bool = False
    info: dict = {}


class GymnasiumAgent(SimpleResponsesAPIAgent):
    config: GymnasiumAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_resp = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_resp)
        result = NeMoGymResponse.model_validate(await get_response_json(model_resp))
        for k, v in model_resp.cookies.items():
            response.set_cookie(k, v)
        return result

    async def run(self, request: Request, body: GymnasiumAgentRunRequest) -> GymnasiumRunResponse:
        cookies = request.cookies

        # --- reset ---
        reset_resp = await self.server_client.post(
            server_name=self.config.env_server.name,
            url_path="/reset",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(reset_resp)
        reset_data = EnvResetResponse.model_validate(await get_response_json(reset_resp))
        cookies = reset_resp.cookies

        current_input = body.responses_create_params.model_copy(deep=True)
        if isinstance(current_input.input, str):
            current_input = current_input.model_copy(
                update={"input": [NeMoGymEasyInputMessage(role="user", content=current_input.input)]}
            )
        if reset_data.observation:
            current_input = current_input.model_copy(
                update={
                    "input": list(current_input.input)
                    + [NeMoGymEasyInputMessage(role="user", content=reset_data.observation)]
                }
            )

        all_outputs = []
        usage = None
        model_server_cookies = None
        step_data = EnvStepResponse(terminated=False, truncated=True, reward=0.0)
        last_model_response = None

        # --- loop ---
        for _ in range(self.config.max_steps):
            model_resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=current_input,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_resp)
            model_response = NeMoGymResponse.model_validate(await get_response_json(model_resp))
            model_server_cookies = model_resp.cookies
            last_model_response = model_response

            all_outputs.extend(model_response.output)

            if model_response.usage:
                if usage is None:
                    usage = model_response.usage.model_copy(deep=True)
                else:
                    usage.input_tokens += model_response.usage.input_tokens
                    usage.output_tokens += model_response.usage.output_tokens
                    usage.total_tokens += model_response.usage.total_tokens
                    usage.input_tokens_details.cached_tokens = 0
                    usage.output_tokens_details.reasoning_tokens = 0

            step_resp = await self.server_client.post(
                server_name=self.config.env_server.name,
                url_path="/step",
                json=body.model_dump() | {"response": model_response.model_dump()},
                cookies=cookies,
            )
            await raise_for_status(step_resp)
            step_data = EnvStepResponse.model_validate(await get_response_json(step_resp))
            cookies = step_resp.cookies

            if step_data.terminated or step_data.truncated:
                break

            if step_data.observation:
                current_input = current_input.model_copy(
                    update={
                        "input": list(current_input.input)
                        + [
                            NeMoGymEasyInputMessage(role="assistant", content=extract_text(model_response)),
                            NeMoGymEasyInputMessage(role="user", content=step_data.observation),
                        ]
                    }
                )

        else:  # for/else: loop completed without break, meaning max_steps exhausted
            step_data = step_data.model_copy(update={"truncated": True})

        last_model_response.output = all_outputs
        last_model_response.usage = usage

        return GymnasiumRunResponse(
            responses_create_params=body.responses_create_params,
            response=last_model_response,
            reward=step_data.reward,
            terminated=step_data.terminated,
            truncated=step_data.truncated,
            info=step_data.info,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.env_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    GymnasiumAgent.run_webserver()
