# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native episode contracts shared by rollout collectors and processors."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


class TaskIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_source: str
    task_id: str


class EpisodeId(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_id: str
    attempt: NonNegativeInt = 0
    group_id: str | None = None
    member_index: NonNegativeInt | None = None
    group_size: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        group_fields = (self.group_id, self.member_index, self.group_size)
        if any(value is not None for value in group_fields):
            if any(value is None for value in group_fields):
                raise ValueError("group fields must be supplied together")
            if self.member_index >= self.group_size:
                raise ValueError("member_index must be less than group_size")
        return self


class EpisodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task: TaskIdentity
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    task_data: dict[str, JsonValue]
    deadline: datetime | None = None


class EpisodeSeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task: TaskIdentity
    task_data: dict[str, JsonValue]


class EpisodeSeedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resources_session_id: str


class EpisodeVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class ResourcesSessionCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resources_session_id: str


class ResourcesSessionCloseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resources_session_id: str


class EpisodeFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["invalid_request", "unavailable", "agent", "verification", "deadline", "internal"]
    message: str = Field(max_length=2000)
    retryable: bool


class EpisodeVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float = Field(allow_inf_nan=False)
    reward_components: dict[str, float] = Field(default_factory=dict)
    mask_sample: bool = False
    verifier_data: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reward_components(self) -> Self:
        if any(not isfinite(value) for value in self.reward_components.values()):
            raise ValueError("reward components must be finite")
        return self


class AgentTurn(BaseModel):
    """One ordered agent activation and its exact model-visible exchange."""

    model_config = ConfigDict(extra="forbid")

    sequence: NonNegativeInt
    agent_id: str
    request: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    observations: AgentObservationBundle | None = None
    state_after: JsonValue = None
    termination_reason: str | None = None


class EpisodeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task: TaskIdentity
    agent_turns: list[AgentTurn] = Field(default_factory=list)
    output_turn_sequence: NonNegativeInt | None = None
    verification: EpisodeVerification | None = None
    agent_observations: AgentObservationBundle | None = None
    failure: EpisodeFailure | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.verification is None) == (self.failure is None):
            raise ValueError("exactly one of verification or failure is required")
        sequences = [turn.sequence for turn in self.agent_turns]
        if sequences != list(range(len(self.agent_turns))):
            raise ValueError("agent turn sequences must be contiguous and ordered from zero")
        if self.verification is not None and self.output_turn_sequence is None:
            raise ValueError("a verified episode requires an output turn")
        if self.output_turn_sequence is not None and self.output_turn_sequence >= len(self.agent_turns):
            raise ValueError("output_turn_sequence must reference an agent turn")
        return self
