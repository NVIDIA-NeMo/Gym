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

"""Agent for GymnasiumServer resources servers (resources_servers.base_gymnasium) which implements the Gymnasium API."""

from fastapi import Body, Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.base_gymnasium import EnvResetResponse, EnvStepResponse


class GymnasiumAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = Field(10, ge=1)


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
        env_cookies = request.cookies

        reset_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/reset",
            json=body.model_dump(),
            cookies=env_cookies,
        )
        await raise_for_status(reset_resp)
        reset_data = EnvResetResponse.model_validate(await get_response_json(reset_resp))
        env_cookies = reset_resp.cookies

        base_body = body.responses_create_params.model_copy(deep=True)
        if isinstance(base_body.input, str):
            base_body.input = [NeMoGymEasyInputMessage(role="user", content=base_body.input)]
        if reset_data.observation:
            base_body.input = list(base_body.input) + [
                NeMoGymEasyInputMessage(role="user", content=reset_data.observation)
            ]

        new_outputs = []
        total_reward = 0.0
        usage = None
        model_server_cookies = None
        step_data = EnvStepResponse(terminated=False, truncated=True, reward=0.0)
        last_model_response = None
        finished = False

        for _ in range(self.config.max_steps):
            new_body = base_body.model_copy(update={"input": base_body.input + new_outputs})

            model_resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_resp)
            model_response = NeMoGymResponse.model_validate(await get_response_json(model_resp))
            model_server_cookies = model_resp.cookies
            last_model_response = model_response

            new_outputs.extend(model_response.output)

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
                server_name=self.config.resources_server.name,
                url_path="/step",
                json=body.model_dump() | {"response": model_response.model_dump()},
                cookies=env_cookies,
            )
            await raise_for_status(step_resp)
            step_data = EnvStepResponse.model_validate(await get_response_json(step_resp))
            total_reward += step_data.reward
            env_cookies = step_resp.cookies

            if step_data.terminated or step_data.truncated:
                finished = True
                break

            for tool_output in (step_data.info or {}).get("tool_outputs", []):
                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=tool_output["call_id"],
                        output=tool_output["output"],
                    )
                )

            if step_data.observation:
                new_outputs.append(NeMoGymEasyInputMessage(role="user", content=step_data.observation))

        if not finished:
            step_data = step_data.model_copy(update={"truncated": True})

        last_model_response.output = new_outputs
        last_model_response.usage = usage

        return GymnasiumRunResponse(
            responses_create_params=body.responses_create_params,
            response=last_model_response,
            reward=total_reward,
            terminated=step_data.terminated,
            truncated=step_data.truncated,
            info=step_data.info,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    GymnasiumAgent.run_webserver()
