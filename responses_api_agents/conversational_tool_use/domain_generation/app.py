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

"""Gym agent for conversational tool-use domain sampling."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.conversational_tool_use.domain_generation.assets import load_followup_instruction


PROTOCOL_VERSION = "domain-generation/v1"
VARIABLE_FOLLOWUP_PROTOCOL_VERSION = "domain-generation/v2"


class DomainGenerationAgentConfig(BaseResponsesAPIAgentConfig):
    model_config = ConfigDict(extra="forbid")

    model_server: ModelServerRef
    followup_count: int = Field(default=1, ge=0)


class DomainGenerationRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class DomainGenerationPhaseTrace(BaseModel):
    phase: Literal["initial", "followup"]
    request: NeMoGymChatCompletionCreateParamsNonStreaming
    response: NeMoGymChatCompletion
    parsed_value: Any
    parse_error: str | None = None


class DomainGenerationTrace(BaseModel):
    protocol_version: Literal["domain-generation/v1", "domain-generation/v2"] = PROTOCOL_VERSION
    request_index: int | None = None
    followup_count: int = Field(default=1, ge=0)
    phases: list[DomainGenerationPhaseTrace] = Field(min_length=1)


class DomainGenerationResult(BaseModel):
    candidates: list[Any]


class DomainGenerationRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    result: DomainGenerationResult
    generation_trace: DomainGenerationTrace


def parse_domain_response(response: NeMoGymChatCompletion) -> tuple[Any, str | None]:
    """Parse one sampler response without schema validation."""
    try:
        text = response.choices[0].message.content
        content = text.strip().removeprefix("```json").removesuffix("```")
        return json.loads(content), None
    except Exception as exc:
        print(f"Error while parsing response: {exc}")
        return [], str(exc)


def followup_prompt(initial_prompt: str, known_domain_names: list[Any]) -> str:
    return (
        initial_prompt + f"\n\nPreviously brainstormed domains: {known_domain_names}.\n" + load_followup_instruction()
    )


def _initial_prompt(body: DomainGenerationRunRequest) -> str:
    response_input = body.responses_create_params.input
    if isinstance(response_input, str):
        return response_input
    if len(response_input) != 1:
        raise ValueError("domain generation requires exactly one input message")

    message = response_input[0]
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    if role != "user" or not isinstance(content, str):
        raise ValueError("domain generation requires one user message with string content")
    return content


class DomainGenerationAgent(SimpleResponsesAPIAgent):
    config: DomainGenerationAgentConfig

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=body,
        )
        await raise_for_status(model_response)
        return NeMoGymResponse.model_validate(await get_response_json(model_response))

    async def _chat_completion(
        self,
        body: DomainGenerationRunRequest,
        prompt: str,
    ) -> tuple[NeMoGymChatCompletionCreateParamsNonStreaming, NeMoGymChatCompletion]:
        chat_request = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": prompt}])
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_run("/v1/chat/completions", body),
            json=chat_request,
        )
        await raise_for_status(model_response)
        completion = NeMoGymChatCompletion.model_validate(await get_response_json(model_response))
        return chat_request, completion

    async def run(self, body: DomainGenerationRunRequest = Body()) -> DomainGenerationRunResponse:
        initial_prompt = _initial_prompt(body)
        candidates: list[Any] = []
        phases: list[DomainGenerationPhaseTrace] = []
        prompt = initial_prompt
        final_response: NeMoGymChatCompletion | None = None

        for phase_index in range(self.config.followup_count + 1):
            request, response = await self._chat_completion(body, prompt)
            parsed_value, parse_error = parse_domain_response(response)
            candidates.extend(parsed_value)
            phases.append(
                DomainGenerationPhaseTrace(
                    phase="initial" if phase_index == 0 else "followup",
                    request=request,
                    response=response,
                    parsed_value=parsed_value,
                    parse_error=parse_error,
                )
            )
            final_response = response

            if phase_index < self.config.followup_count:
                known_domain_names = [
                    candidate["name"]
                    for candidate in candidates
                    if isinstance(candidate, dict) and "name" in candidate
                ]
                prompt = followup_prompt(initial_prompt, known_domain_names)

        assert final_response is not None

        trace = DomainGenerationTrace(
            protocol_version=(
                PROTOCOL_VERSION if self.config.followup_count == 1 else VARIABLE_FOLLOWUP_PROTOCOL_VERSION
            ),
            request_index=body.model_dump().get(TASK_INDEX_KEY_NAME),
            followup_count=self.config.followup_count,
            phases=phases,
        )

        response_params = body.responses_create_params.model_copy(update={"model": final_response.model})
        response = ResponsesConverter(return_token_id_information=False).chat_completion_to_response(
            response_params,
            final_response,
        )
        response_payload = body.model_dump()
        response_payload.update(
            response=response,
            reward=1.0,
            result=DomainGenerationResult(candidates=candidates),
            generation_trace=trace,
        )
        return DomainGenerationRunResponse.model_validate(response_payload)


if __name__ == "__main__":
    DomainGenerationAgent.run_webserver()
