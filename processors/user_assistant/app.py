# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-assistant specialization of the reusable multi-agent processor."""

from typing import Any, cast

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.processors.base import BaseProcessorConfig
from nemo_gym.processors.multi_agent import (
    EpisodeEvent,
    MultiAgentEpisodeSpec,
    MultiAgentProcessor,
    ParticipantTurn,
)


class UserAssistantRunRequest(BaseRunRequest):
    """Preserve the original request shape for user-assistant rollouts."""

    model_config = ConfigDict(extra="allow")

    user_responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class UserAssistantSeedSessionResponse(BaseModel):
    """Optional per-episode user parameters resolved by the resources server."""

    model_config = ConfigDict(extra="allow")

    user_responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None


class UserAssistantVerifyRequest(BaseVerifyRequest):
    """Verification payload with separately attributed assistant and user turns."""

    model_config = ConfigDict(extra="allow")

    assistant_trajectory: list[ParticipantTurn]
    user_trajectory: list[ParticipantTurn]
    episode_trajectory: list[EpisodeEvent]
    termination_reason: str
    turns_completed: int


class UserAssistantVerifyResponse(BaseVerifyResponse):
    """Original user-assistant response shape returned to rollout callers."""

    model_config = ConfigDict(extra="allow")

    assistant_trajectory: list[ParticipantTurn]
    user_trajectory: list[ParticipantTurn]
    episode_trajectory: list[EpisodeEvent]
    termination_reason: str
    turns_completed: int


class UserAssistantProcessorConfig(BaseProcessorConfig):
    """Configuration for the two-participant user-assistant specialization."""

    assistant_agent: AgentServerRef
    user_agent: AgentServerRef
    resources_server: ResourcesServerRef
    max_turns: int = Field(8, ge=1)
    status_url_path: str = "/episode_status"


class UserAssistantProcessor(MultiAgentProcessor):
    """Preserve the user-assistant interface over the generic episode engine."""

    config: UserAssistantProcessorConfig

    def _episode_spec(self) -> MultiAgentEpisodeSpec:
        return MultiAgentEpisodeSpec(
            participants={
                "assistant": self.config.assistant_agent,
                "user": self.config.user_agent,
            },
            turn_order=["assistant", "user"],
            focal_participant="assistant",
            resources_server=self.config.resources_server,
            max_turns=self.config.max_turns,
            status_url_path=self.config.status_url_path,
        )

    def _params_by_participant(
        self,
        body: UserAssistantRunRequest,
    ) -> dict[str, NeMoGymResponseCreateParamsNonStreaming]:
        return {
            "assistant": body.responses_create_params,
            "user": body.user_responses_create_params,
        }

    def _resolve_seeded_body(
        self,
        body: BaseRunRequest,
        seed_result: dict[str, Any],
    ) -> UserAssistantRunRequest:
        user_assistant_body = UserAssistantRunRequest.model_validate(body)
        resolved = UserAssistantSeedSessionResponse.model_validate(seed_result)
        if resolved.user_responses_create_params is None:
            return user_assistant_body
        return user_assistant_body.model_copy(
            update={"user_responses_create_params": resolved.user_responses_create_params}
        )

    def _build_verify_request(
        self,
        *,
        body: BaseRunRequest,
        focal_response: NeMoGymResponse,
        trajectories: dict[str, list[ParticipantTurn]],
        events: list[EpisodeEvent],
        termination_reason: str,
        turns_completed: int,
    ) -> BaseVerifyRequest:
        return UserAssistantVerifyRequest.model_validate(
            body.model_dump(mode="json")
            | {
                "response": focal_response.model_dump(mode="json"),
                "assistant_trajectory": [turn.model_dump(mode="json") for turn in trajectories["assistant"]],
                "user_trajectory": [turn.model_dump(mode="json") for turn in trajectories["user"]],
                "episode_trajectory": [event.model_dump(mode="json") for event in events],
                "termination_reason": termination_reason,
                "turns_completed": turns_completed,
            }
        )

    def _build_verify_response(self, result: dict[str, Any]) -> BaseVerifyResponse:
        return UserAssistantVerifyResponse.model_validate(result)

    async def run(
        self,
        request: Request,
        body: UserAssistantRunRequest,
    ) -> UserAssistantVerifyResponse:
        return cast(UserAssistantVerifyResponse, await super().run(request, body))


if __name__ == "__main__":
    UserAssistantProcessor.run_webserver()
