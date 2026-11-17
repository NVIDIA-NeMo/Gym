# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated local episode entry point; new configurations use agent sessions."""

from fastapi import HTTPException, Request

from nemo_gym.agents import verification_failure_fields
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.core import NeMoGymResponse
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_agent.app import (
    _INTERNAL_OBSERVATIONS_KEY,
    _INTERNAL_TRAJECTORY_KEY,
    OpenCodeAgent,
    OpenCodeAgentRunRequest,
    OpenCodeAgentVerifyResponse,
)
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory


class OpenCodeLegacyAgent(OpenCodeAgent):
    """Preserve the local resources/agent/verifier relay for legacy configurations."""

    async def run(self, request: Request, body: OpenCodeAgentRunRequest) -> OpenCodeAgentVerifyResponse:
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "OpenCode sandbox sessions must use EnvironmentServer /run")
        if self.config.resources_server is None:
            raise HTTPException(422, "Local OpenCode /run requires resources_server")
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            rollout_id = self.rollout_id_from_run(body)
            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            raw_trajectory = agent_resp_json.pop(_INTERNAL_TRAJECTORY_KEY, None)
            trajectory = (
                scope_opencode_trajectory(TrajectoryRecord.model_validate(raw_trajectory), body, rollout_id)
                if isinstance(raw_trajectory, dict) and rollout_id is not None
                else None
            )
            raw_observations = (
                agent_resp_json.pop(_INTERNAL_OBSERVATIONS_KEY, None) if rollout_id is not None else None
            )
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
            )

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump()
                | {"response": agent_resp_json}
                | verification_failure_fields(NeMoGymResponse.model_validate(agent_resp_json)),
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return OpenCodeAgentVerifyResponse.model_validate(
                verify_json
                | {"turns_used": turns, "finished_naturally": naturally}
                | ({"ng_agent_observations": observations} if observations is not None else {})
                | ({"ng_trajectory": trajectory.model_dump(mode="json")} if trajectory is not None else {})
            )


if __name__ == "__main__":
    OpenCodeLegacyAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeLegacyAgent.run_webserver()  # noqa: F401
