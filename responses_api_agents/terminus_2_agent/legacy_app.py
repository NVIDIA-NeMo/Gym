# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated host episode entry point; new configurations use agent sessions."""

import json

from fastapi import HTTPException, Request

from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.core import NeMoGymResponse
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.terminus_2_agent.app import (
    Terminus2Agent,
    Terminus2AgentRunRequest,
    Terminus2AgentVerifyResponse,
)


class Terminus2LegacyAgent(Terminus2Agent):
    """Preserve the host resources/agent/verifier relay for legacy configurations."""

    async def run(self, request: Request, body: Terminus2AgentRunRequest) -> Terminus2AgentVerifyResponse:
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "Use the agent session responses and close routes")
        if self.config.resources_server is None:
            raise HTTPException(422, "Terminus 2 /run requires resources_server; use agent sessions instead")
        cookies = request.cookies
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        agent_response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(agent_response)
        cookies = agent_response.cookies
        response_json = await get_response_json(agent_response)

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response_json},
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        verify_json = await get_response_json(verify_response)

        gym_response = NeMoGymResponse.model_validate(response_json)
        turns = sum(
            1
            for item in gym_response.output
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
        )
        terminus_metadata_json = (gym_response.metadata or {}).get("terminus_2", "{}")
        try:
            terminus_metadata = json.loads(terminus_metadata_json)
        except (json.JSONDecodeError, TypeError):
            terminus_metadata = {}
        flags = terminus_metadata.get("agent_error_flags", {})
        timed_out = bool(terminus_metadata.get("timed_out"))
        naturally = bool(terminus_metadata.get("finished_naturally"))

        return Terminus2AgentVerifyResponse.model_validate(
            verify_json
            | {
                "turns_used": turns,
                "finished_naturally": naturally and not timed_out,
                "agent_timeout_error": int(timed_out),
                "context_length_exceeded_error": int(flags.get("context_length_exceeded", False)),
            }
        )



if __name__ == "__main__":
    Terminus2LegacyAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Terminus2LegacyAgent.run_webserver()  # noqa: F401
