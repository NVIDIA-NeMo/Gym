# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Perplexity summarizer agent.

A custom agent with tool-call-limited loop: counts actual tool calls (not loop
steps) and sets tool_choice="none" when the limit is reached, forcing the model
to produce a final text response.
"""

import json
from typing import List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

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
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.perplexity_summarizer.prompts import TOOL_CALL_DISABLE_SUFFIX


class PerplexitySummarizerAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_tool_calls: int = 0  # -1 = unlimited
    bad_words: Optional[List[str]] = None  # Token strings to suppress via vLLM bad_words when tool_choice="none"

    # Inference hparams — applied on top of responses_create_params from the JSONL data.
    # None means "use whatever the JSONL data has".
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None


class PerplexitySummarizerAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class PerplexitySummarizerAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class PerplexitySummarizerAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class PerplexitySummarizerAgent(SimpleResponsesAPIAgent):
    config: PerplexitySummarizerAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        # Apply config defaults only when not already set (CLI takes precedence)
        if body.temperature is None and self.config.temperature is not None:
            body.temperature = self.config.temperature
        if body.top_p is None and self.config.top_p is not None:
            body.top_p = self.config.top_p
        if body.max_output_tokens is None and self.config.max_output_tokens is not None:
            body.max_output_tokens = self.config.max_output_tokens

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        usage = None
        tool_calls_count = 0
        model_server_cookies = None
        resources_server_cookies = request.cookies

        while True:
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            # When tool call limit reached, force text-only response
            if self.config.max_tool_calls != -1 and tool_calls_count >= self.config.max_tool_calls:
                new_body.tool_choice = "none"

                # Inject bad_words to suppress tool-call tokens (configurable via YAML)
                if self.config.bad_words:
                    new_body.metadata = {"extra_body": json.dumps({"bad_words": self.config.bad_words})}

                # Append hint to last tool result so the model generates a
                # summary instead of attempting more tool calls.
                for item in reversed(new_body.input):
                    if hasattr(item, "type") and item.type == "function_call_output":
                        if isinstance(item.output, str) and TOOL_CALL_DISABLE_SUFFIX not in item.output:
                            item.output += TOOL_CALL_DISABLE_SUFFIX
                        break

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            output = model_response.output
            new_outputs.extend(output)

            # Usage accumulation follows the simple_agent framework pattern:
            # first iteration assigns usage, then immediately adds again (intentional double-count
            # on first call to match NeMo-Gym training infrastructure expectations).
            if not usage:
                usage = model_response.usage

            if usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens

                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                tool_calls_count += 1
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=resources_server_cookies,
                )
                resources_server_cookies = api_response.cookies

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=(await api_response.content.read()).decode(),
                )
                new_outputs.append(tool_response)

        # Propagate cookies for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        model_response.output = new_outputs
        model_response.usage = usage
        return model_response

    async def run(
        self, request: Request, body: PerplexitySummarizerAgentRunRequest
    ) -> PerplexitySummarizerAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies = response.cookies

        verify_request = PerplexitySummarizerAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(response)}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return PerplexitySummarizerAgentVerifyResponse.model_validate(await get_response_json(verify_response))


if __name__ == "__main__":
    PerplexitySummarizerAgent.run_webserver()
