# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public request, response, and configuration types for user-assistant processors."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.processors.base import BaseProcessorConfig


Participant = Literal["assistant", "user"]
EpisodeEventKind = Literal["response_item", "state", "termination"]


class ParticipantTurn(BaseModel):
    """One attributed agent invocation, including its exact model-visible input."""

    turn_index: int
    participant: Participant
    request: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    agent_trajectory: Optional[dict[str, Any]] = None


class EpisodeEvent(BaseModel):
    """One ordered participant output, state observation, or termination event."""

    sequence: int
    turn_index: int
    kind: EpisodeEventKind
    participant: Optional[Participant] = None
    data: dict[str, Any]


class EpisodeStatus(BaseModel):
    """Resources-server response used to stop an episode and expose shared state."""

    model_config = ConfigDict(extra="allow")

    terminated: bool = False
    reason: Optional[str] = None
    state: dict[str, Any] = Field(default_factory=dict)


class UserAssistantRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    user_responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class UserAssistantVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    assistant_trajectory: list[ParticipantTurn]
    user_trajectory: list[ParticipantTurn]
    episode_trajectory: list[EpisodeEvent]
    termination_reason: str
    turns_completed: int


class UserAssistantVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    assistant_trajectory: list[ParticipantTurn]
    user_trajectory: list[ParticipantTurn]
    episode_trajectory: list[EpisodeEvent]
    termination_reason: str
    turns_completed: int


class UserAssistantProcessorConfig(BaseProcessorConfig):
    assistant_agent: AgentServerRef
    user_agent: AgentServerRef
    resources_server: ResourcesServerRef
    max_turns: int = Field(8, ge=1)
    status_url_path: str = "/episode_status"
