# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym agent for conversational tool-use domain sampling."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import Request
from pydantic import BaseModel, ConfigDict

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


PROTOCOL_VERSION = "domain-generation/v1"
FOLLOWUP_INSTRUCTION = "Do not repeat these domains. Try looking for other domains or find specific sub-domains."


class DomainGenerationAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef


class DomainGenerationRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class DomainGenerationPhaseTrace(BaseModel):
    phase: Literal["initial", "followup"]
    request: NeMoGymChatCompletionCreateParamsNonStreaming
    response: NeMoGymChatCompletion
    parsed_value: Any
    parse_error: str | None = None


class DomainGenerationTrace(BaseModel):
    protocol_version: Literal["domain-generation/v1"] = PROTOCOL_VERSION
    request_index: int | None = None
    phases: tuple[DomainGenerationPhaseTrace, DomainGenerationPhaseTrace]


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
    return initial_prompt + f"\n\nPreviously brainstormed domains: {known_domain_names}.\n" + FOLLOWUP_INSTRUCTION


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

        initial_request, initial_response = await self._chat_completion(body, initial_prompt)
        initial_results, initial_error = parse_domain_response(initial_response)

        candidates: list[Any] = []
        candidates.extend(initial_results)
        known_domain_names = [candidate["name"] for candidate in candidates]

        rendered_followup = followup_prompt(initial_prompt, known_domain_names)
        followup_request, followup_response = await self._chat_completion(body, rendered_followup)
        followup_results, followup_error = parse_domain_response(followup_response)
        candidates.extend(followup_results)

        trace = DomainGenerationTrace(
            request_index=body.model_dump().get(TASK_INDEX_KEY_NAME),
            phases=(
                DomainGenerationPhaseTrace(
                    phase="initial",
                    request=initial_request,
                    response=initial_response,
                    parsed_value=initial_results,
                    parse_error=initial_error,
                ),
                DomainGenerationPhaseTrace(
                    phase="followup",
                    request=followup_request,
                    response=followup_response,
                    parsed_value=followup_results,
                    parse_error=followup_error,
                ),
            ),
        )

        response_params = body.responses_create_params.model_copy(update={"model": followup_response.model})
        response = ResponsesConverter(return_token_id_information=False).chat_completion_to_response(
            response_params,
            followup_response,
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
