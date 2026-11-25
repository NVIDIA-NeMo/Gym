# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab's Gym agent session contract for the OpenCode harness."""

from fastapi import Body, HTTPException

from benchmarks.robolab.episode_binding import RoboLabEpisodeBinding
from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


class RoboLabAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef


class RoboLabAgent(SimpleResponsesAPIAgent):
    config: RoboLabAgentConfig
    agent_context_model = RoboLabEpisodeBinding

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        if body.sandbox_access is not None:
            raise HTTPException(422, "RoboLabAgent does not support sandbox access")
        self.parse_agent_context(body)
        return AgentSessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id, cleanup_confirmed=True)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise HTTPException(422, "RoboLab OpenCode activation requires the pinned Bridge runtime adapter")


if __name__ == "__main__":
    RoboLabAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = RoboLabAgent.run_webserver()
