# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate two independent policy responses for each opposing prompt pair."""

from __future__ import annotations

import asyncio
import json

from fastapi import Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


class EvenHandednessAgentConfig(BaseResponsesAPIAgentConfig):
    """References to the policy model and paired verifier."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class EvenHandednessRunRequest(BaseRunRequest):
    """One pair of opposing prompts."""

    model_config = ConfigDict(extra="allow")

    prompt_a: str
    prompt_b: str


class EvenHandednessRunResponse(BaseVerifyResponse):
    """Flexible verifier response returned by the resources server."""

    model_config = ConfigDict(extra="allow")


class EvenHandednessAgent(SimpleResponsesAPIAgent):
    """Run each political stance in an independent model context."""

    config: EvenHandednessAgentConfig
    _RESPONSE_B_METADATA_KEY = "even_handedness_response_b"

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        return NeMoGymResponse.model_validate(await get_response_json(model_response))

    @staticmethod
    def _params_for_prompt(body: EvenHandednessRunRequest, prompt: str) -> NeMoGymResponseCreateParamsNonStreaming:
        params = body.responses_create_params.model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        return params

    async def run(self, request: Request, body: EvenHandednessRunRequest) -> EvenHandednessRunResponse:
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(seed_response)
        resource_cookies = seed_response.cookies

        async def generate(prompt: str) -> dict:
            model_response = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=self._params_for_prompt(body, prompt),
                cookies=request.cookies,
            )
            await raise_for_status(model_response)
            return await get_response_json(model_response)

        response_a, response_b = await asyncio.gather(generate(body.prompt_a), generate(body.prompt_b))
        response_a_metadata = dict(response_a.get("metadata") or {})
        response_a_metadata[self._RESPONSE_B_METADATA_KEY] = json.dumps(response_b)
        response_a["metadata"] = response_a_metadata
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response_a, "response_b": response_b},
            cookies=resource_cookies,
        )
        await raise_for_status(verify_response)
        return EvenHandednessRunResponse.model_validate(await get_response_json(verify_response))


if __name__ == "__main__":
    EvenHandednessAgent.run_webserver()
