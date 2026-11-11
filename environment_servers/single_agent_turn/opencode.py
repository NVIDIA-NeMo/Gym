# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode reward shaping over the shared seed/activate/close/verify lifecycle."""

from typing import Any

from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import (
    SessionHandles,
    SingleAgentTurnEnvironmentServerConfig,
    SingleAgentTurnRequest,
    SingleAgentTurnResponse,
    SingleAgentTurnResult,
)
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.config import get_first_server_config_dict
from nemo_gym.core import NeMoGymResponse
from nemo_gym.server_runtime import CleanupContext
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord, grading_result_binding
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_multimodal_agent.rewards import (
    EpisodeEvidence,
    OpenCodeEvalScoringConfig,
    OpenCodeScoringConfig,
    OpenCodeToolRewardConfig,
    final_answer,
)


class OpenCodeEnvironmentServer(SingleAgentTurnLegacyEnvironmentServer):
    """Use main's session lifecycle and flat/native HTTP bridge with tool rewards."""

    config: SingleAgentTurnEnvironmentServerConfig

    async def run(self, request: SingleAgentTurnRequest, cleanup: CleanupContext) -> SingleAgentTurnResponse:
        # Old datasets retain this marker when migrated to main's static routes.
        # It is an assertion about the bound verifier, never a runtime routing override.
        marker = request.task.task_input.task_data.get("image_tools_base_agent_ref")
        if marker is not None:
            expected = {
                "math_with_judge_simple_agent": "math_with_judge",
                "string_match_simple_agent": "string_match",
                "mcqa_simple_agent": "mcqa",
            }.get(marker.get("name") if isinstance(marker, dict) else None)
            resource = self.server_client.global_config_dict.get(self.config.resources_server.name, {})
            components = resource.get("resources_servers", {})
            if expected is None or expected not in components:
                raise self._failure(
                    stage="admission",
                    failure_reason="Verifier marker does not match this environment's resources server; select the matching task_source/profile",
                    terminal=True,
                )
        return await super().run(request, cleanup)

    def _agent_options(self) -> dict[str, Any]:
        config = get_first_server_config_dict(self.server_client.global_config_dict, self.config.agent_server.name)
        config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
        config_type = {
            "app.py": OpenCodeScoringConfig,
            "combined.py": OpenCodeScoringConfig,
            "eval.py": OpenCodeEvalScoringConfig,
        }[config.get("entrypoint", "app.py")]
        return config | config_type.model_validate(config).model_dump()

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
        evidence = (
            EpisodeEvidence.model_validate((agent_response.model_extra or {}).get("opencode_evidence"))
            if (agent_response.model_extra or {}).get("opencode_evidence")
            else None
        )
        if evidence and evidence.verification_error:
            raise self._failure(
                stage="verification",
                failure_reason=evidence.verification_error,
                terminal=True,
                partial_response=agent_response,
                agent_observations=observations,
                trajectory=trajectory,
            )
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
        if evidence is None:
            raise ValueError("Missing native OpenCode evidence")
        tools = evidence.tool_reward
        raw_config = self._agent_options().get("tool_rewards", {})
        base_reward = float(result.reward)
        # Accepted finalized traces retain shaping even without a natural final answer.
        shape_rewards = raw_config is not None
        if not shape_rewards:
            tools = None
        if shape_rewards:
            if tools is None:
                raise ValueError("Missing native tool evidence for reward shaping")
            tools = tools.for_answer(base_reward, OpenCodeToolRewardConfig.model_validate(raw_config))
        auxiliary_reward = tools.auxiliary_reward if shape_rewards else 0.0
        source = result.model_dump(mode="json")
        source = {key: source[key] for key in ("response", "reward", "reward_components") if key in source}
        transform = {
            "kind": "response_projection_reward_offset_v1",
            "input_result": source,
            "input_binding_sha256": grading_result_binding(source),
            "reward_offset": auxiliary_reward,
            "reward_component": "opencode_tool_reward",
            "parameters": {
                "operation": "opencode_final_answer_tool_reward_v1",
                "strip_reasoning_for_grading": self._agent_options()["strip_reasoning_for_grading"],
                "tool_rewards": (
                    OpenCodeToolRewardConfig.model_validate(raw_config).model_dump(mode="json")
                    if shape_rewards
                    else None
                ),
            },
        }
        transformed = result.model_copy(
            update={
                "response": agent_response,
                "reward": base_reward + auxiliary_reward,
                "reward_components": source.get("reward_components", {"verifier": base_reward})
                | {"opencode_tool_reward": auxiliary_reward},
                "ng_grading_transform": transform,
                "resolved": base_reward == 1.0,
                "opencode_tool_reward": tools,
                "turns_used": evidence.turns_used,
                "finished_naturally": evidence.finished_naturally,
            }
        )
        transform["output_binding_sha256"] = grading_result_binding(transformed)
        return transformed


if __name__ == "__main__":
    OpenCodeEnvironmentServer.run_webserver()

elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeEnvironmentServer.run_webserver()  # noqa: F401
