# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym collector adapter for a resource-owned Hermes runner."""

from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status


class HermesSandboxedConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef


class HermesRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HermesVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class HermesSandboxedAgent(SimpleResponsesAPIAgent):
    config: HermesSandboxedConfig

    async def responses(self, body):
        raise NotImplementedError("This adapter requires /run")

    async def run(self, request: Request, body: HermesRunRequest) -> HermesVerifyResponse:
        payload = body.model_dump(mode="json")
        rollout_id = self.rollout_id_from_run(body)
        payload["rollout_id"] = rollout_id or body.capture_rollout_id or payload.get("rollout_id") or uuid4().hex
        payload["client_session_id"] = request.session[SESSION_ID_KEY]
        if rollout_id:
            payload["_ng_rollout_id"] = rollout_id
        payload["capture_model_calls"] = bool(rollout_id)
        payload["capture_token_ids"] = self._token_id_capture_enabled()
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/run",
            json=payload,
            cookies=request.cookies,
        )
        await raise_for_status(response)
        return HermesVerifyResponse.model_validate(await get_response_json(response))


if __name__ == "__main__":
    HermesSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = HermesSandboxedAgent.run_webserver()
