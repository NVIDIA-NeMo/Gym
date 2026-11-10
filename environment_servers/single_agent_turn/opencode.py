# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode reward shaping over the shared seed/activate/close/verify lifecycle."""

from typing import Any

from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import (
    SessionHandles,
    SingleAgentTurnEnvironmentServerConfig,
    SingleAgentTurnRequest,
    SingleAgentTurnResult,
)
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.config import get_first_server_config_dict
from nemo_gym.core import NeMoGymResponse
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_multimodal_agent.rewards import (
    EpisodeEvidence,
    OpenCodeToolRewardConfig,
    final_answer,
)


class OpenCodeEnvironmentServer(SingleAgentTurnLegacyEnvironmentServer):
    """Use main's session lifecycle and flat/native HTTP bridge with tool rewards."""

    config: SingleAgentTurnEnvironmentServerConfig

    def _agent_options(self) -> dict[str, Any]:
        config = get_first_server_config_dict(self.server_client.global_config_dict, self.config.agent_server.name)
        return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config

    def _scoring_response(self, agent_response: NeMoGymResponse) -> NeMoGymResponse:
        # Only the verifier sees the scoring copy. Training keeps the complete response below.
        scoring = agent_response.model_dump(mode="json")
        scoring["output"] = [
            {
                "id": "final",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": final_answer(
                            scoring["output"],
                            strip_reasoning=self._agent_options().get("strip_reasoning_for_grading", True),
                        ),
                        "annotations": [],
                    }
                ],
            }
        ]
        return NeMoGymResponse.model_validate(scoring)

    async def _verify(
        self,
        request: SingleAgentTurnRequest,
        handles: SessionHandles,
        agent_response: NeMoGymResponse,
        observations: AgentObservationBundle | None,
        trajectory: TrajectoryRecord | None = None,
    ) -> SingleAgentTurnResult:
        if agent_response.status == "failed":
            # Main has already selected VERIFY from the typed outcome policy. Never judge partial answers.
            return SingleAgentTurnResult.model_validate(
                super()._verify_body(request, agent_response)
                | {
                    "reward": 0.0,
                    "resolved": False,
                    "evaluation_completed": True,
                    "scorer": "agent-failure",
                    "finished_naturally": False,
                }
            )
        result = await super()._verify(
            request, handles, self._scoring_response(agent_response), observations, trajectory
        )
        evidence = EpisodeEvidence.model_validate((agent_response.model_extra or {}).get("opencode_evidence"))
        tools = evidence.tool_reward
        raw_config = self._agent_options().get("tool_rewards", {})
        base_reward = float(result.reward)
        # Main may verify an incomplete model response; its tool execution evidence is not final.
        shape_rewards = raw_config is not None and agent_response.status != "incomplete"
        if not shape_rewards:
            tools = None
        if shape_rewards:
            if tools is None:
                raise ValueError("Missing native tool evidence for reward shaping")
            tools = tools.for_answer(base_reward, OpenCodeToolRewardConfig.model_validate(raw_config))
        return result.model_copy(
            update={
                "response": agent_response,
                "reward": base_reward + (tools.auxiliary_reward if shape_rewards else 0.0),
                "resolved": base_reward == 1.0,
                "opencode_tool_reward": tools,
                "turns_used": evidence.turns_used,
                "finished_naturally": evidence.finished_naturally,
            }
        )


if __name__ == "__main__":
    OpenCodeEnvironmentServer.run_webserver()

elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeEnvironmentServer.run_webserver()  # noqa: F401
