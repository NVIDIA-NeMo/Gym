# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom Wordle agent that extends SimpleAgent with game-over detection.

Stops the agentic loop immediately when the game ends (win or loss),
instead of waiting for the model to output plain text or hitting max_steps.
"""

import json
from typing import List

from fastapi import Request, Response
from pydantic import ConfigDict

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


class WordleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None


class WordleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class WordleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class WordleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class WordleAgent(SimpleResponsesAPIAgent):
    """Wordle-specific agent with game-over detection."""

    config: WordleAgentConfig

    def _is_game_over(self, tool_output: str) -> bool:
        """Check if a tool response indicates the game is over."""
        try:
            result = json.loads(tool_output)
            return result.get("game_over", False)
        except (json.JSONDecodeError, TypeError):
            return False

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        step = 0
        model_server_cookies = None
        resources_server_cookies = request.cookies
        game_over = False

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            model_response = NeMoGymResponse.model_validate(model_response_json)

            output = model_response.output
            new_outputs.extend(output)

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=resources_server_cookies,
                )
                resources_server_cookies = api_response.cookies

                tool_output = (await api_response.content.read()).decode()
                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=tool_output,
                )
                new_outputs.append(tool_response)

                # Check if game ended
                if self._is_game_over(tool_output):
                    game_over = True

            # Stop immediately if game is over
            if game_over:
                break

            if self.config.max_steps and step >= self.config.max_steps:
                break

        # Propagate cookies for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        model_response.output = new_outputs
        return model_response

    async def run(self, request: Request, body: WordleAgentRunRequest) -> WordleAgentVerifyResponse:
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

        verify_request = WordleAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(response)}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return WordleAgentVerifyResponse.model_validate(await get_response_json(verify_response))


if __name__ == "__main__":
    WordleAgent.run_webserver()
