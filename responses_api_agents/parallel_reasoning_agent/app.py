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

from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, ValidationError

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
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.parallel_reasoning_agent.utils import ParallelReasoningUtils


class ParallelReasoningConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None
    num_planner: int 
    num_executor: int


class ParallelReasoningRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ParallelReasoningVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class BaseParallelReasoningVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    

class ParallelReasoningVerifyResponse(BaseModel):
    responses: List[BaseParallelReasoningVerifyResponse]



PLANNER_BEGIN_TAG = "<plan>"
PLANNER_END_TAG = "</plan>"

class ParallelReasoning(SimpleResponsesAPIAgent):
    config: ParallelReasoningConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> List[NeMoGymResponse]:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = request.cookies  # update the cookies on every resources server response
        
        # CONFIG
        num_planner = self.config.num_planner
        num_executor = self.config.num_executor

        # PLANNER STAGE
        planner_responses = []
        for _ in range(num_planner):
            planner_prompt = ParallelReasoningUtils.construct_planner_prompt(body.input[0].content)
            new_body = body.model_copy(update={"input": [NeMoGymEasyInputMessage(role="user", content=planner_prompt)]})
            
            # PLANNER RESPONSE
            planner_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            model_response_json = planner_response.json()
            print(f"DEBUG: model response: {model_response_json}")
            planner_cookies = planner_response.cookies
            
            
            try:
                planner_response: NeMoGymResponse = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e
                
            planner_responses.append(planner_response)
            
        # EXECUTOR STAGE
        executor_responses = []
        for planner_response in planner_responses:
            planner_output = planner_response.output[0].content[0].text
            for _ in range(num_executor):
                plan = ParallelReasoningUtils.parse_plan(planner_output)[0]
                executor_prompt = ParallelReasoningUtils.construct_executor_prompt(body.input[0].content, plan)
                executor_body = body.model_copy(update={"input": [NeMoGymEasyInputMessage(role="user", content=executor_prompt)]})
                executor_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/v1/responses",
                    json=executor_body,
                    cookies=planner_cookies,
                )
                executor_cookies = executor_response.cookies
                try: 
                    executor_response: NeMoGymResponse = NeMoGymResponse.model_validate(executor_response.json())
                except ValidationError as e:
                    raise RuntimeError(
                        f"Received an invalid response from model server: {json.dumps(executor_response.json())}"
                    ) from e
                
                executor_responses.append(executor_response)
            
            # Propogate any extra cookies necessary for downstream verification
            for k, v in (*resources_server_cookies.items(), *planner_cookies.items(), *executor_cookies.items()):
                response.set_cookie(k, v)
                
                
        # TODO: support this interface
        return planner_responses + executor_responses

    async def run(self, request: Request, body: ParallelReasoningRunRequest) -> ParallelReasoningVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        cookies = seed_session_response.cookies

        responses = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        responses = responses.json()
        
        planner_responses = responses[:self.config.num_planner]
        executor_responses = responses[self.config.num_planner:]
        
        executor_verify_responses = []
        for response in executor_responses:
            verify_request = ParallelReasoningVerifyRequest.model_validate(body.model_dump() | {"response": response})
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
            )
            with open("parallel_reasoning_verify_responses.json", "w") as f:
                f.write(json.dumps(verify_response.json(), indent=4))
            try:
                executor_verify_responses.append(BaseParallelReasoningVerifyResponse.model_validate(verify_response.json()))
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from resources server: {json.dumps(verify_response.json())}"
                ) from e
            
                    
        # Aggregate executor rewards for each planner response group.
        # We assume that executor_verify_responses are ordered such that
        # the first N correspond to planner 0, the next N to planner 1, etc.,
        # where N = config.num_executor
        # TODO: improve this using metadata from the responses
        num_planner = self.config.num_planner
        num_executor_per_planner = self.config.num_executor

        planner_rewards = []
        planner_verify_responses = []
        for planner_idx, planner_response in enumerate(planner_responses):
            start = planner_idx * num_executor_per_planner
            end = start + num_executor_per_planner
            group = executor_verify_responses[start:end]
            group_rewards = [
                resp.reward for resp in group if hasattr(resp, "reward")
            ]
            if group_rewards:
                planner_rewards.append(sum(group_rewards) / len(group_rewards))
            else:
                planner_rewards.append(0.0)
            planner_verify_responses.append(BaseParallelReasoningVerifyResponse.model_validate(body.model_dump() | {"response": planner_response, "reward": planner_rewards[planner_idx]}))
            
        parallel_reasoning_verify_responses = ParallelReasoningVerifyResponse(responses=planner_verify_responses + executor_verify_responses)

        return parallel_reasoning_verify_responses


if __name__ == "__main__":
    ParallelReasoning.run_webserver()
