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

"""Gym agent for one-domain conversational policy and tool generation."""

from __future__ import annotations

from typing import Any

from fastapi import Request, Response
from pydantic import Field

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.conversational_tool_use_policy_tool_generation.generation import PolicyToolGenerator
from responses_api_agents.conversational_tool_use_policy_tool_generation.models import (
    CallPhase,
    ModelRole,
    PolicyToolGenerationRunRequest,
    PolicyToolGenerationVerifyResponse,
)


class PolicyToolGenerationAgentConfig(BaseResponsesAPIAgentConfig):
    policy_model_server: ModelServerRef
    judge_model_server: ModelServerRef
    max_retries: int = Field(default=20, ge=0, le=20)


def message_only_payload(prompt: str) -> dict[str, Any]:
    """Build an internal chat request containing only messages."""

    return {"messages": [{"role": "user", "content": prompt}]}


class PolicyToolGenerationAgent(SimpleResponsesAPIAgent):
    config: PolicyToolGenerationAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.policy_model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        for key, value in model_response.cookies.items():
            response.set_cookie(key, value)
        return NeMoGymResponse.model_validate(await get_response_json(model_response))

    async def run(
        self,
        request: Request,
        body: PolicyToolGenerationRunRequest,
    ) -> PolicyToolGenerationVerifyResponse:
        async def caller(
            role: ModelRole,
            prompt: str,
            phase: CallPhase,
            attempt: int,
            ordinal: int,
        ) -> NeMoGymChatCompletion:
            del phase, attempt, ordinal
            model_server = self.config.policy_model_server if role == "policy" else self.config.judge_model_server
            model_response = await self.server_client.post(
                server_name=model_server.name,
                url_path=self.url_path_for_run("/v1/chat/completions", body),
                json=message_only_payload(prompt),
                cookies=request.cookies,
            )
            await raise_for_status(model_response)
            return NeMoGymChatCompletion.model_validate(await get_response_json(model_response))

        result, generation_trace, final_completion = await PolicyToolGenerator(
            max_retries=self.config.max_retries
        ).generate(body, caller)
        response_params = body.responses_create_params.model_copy(update={"model": final_completion.model})
        final_response = ResponsesConverter(return_token_id_information=False).chat_completion_to_response(
            response_params, final_completion
        )
        response_payload = body.model_dump()
        response_payload.update(
            response=final_response,
            reward=1.0,
            result=result,
            generation_trace=generation_trace,
        )
        return PolicyToolGenerationVerifyResponse.model_validate(response_payload)


if __name__ == "__main__":
    PolicyToolGenerationAgent.run_webserver()
