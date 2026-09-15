# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native episode contracts shared by rollout collectors and processors."""

from __future__ import annotations

from datetime import datetime
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


class AgentTurn(BaseModel):
    """One ordered participant activation and its exact model-visible exchange."""

    model_config = ConfigDict(extra="forbid")

    sequence: NonNegativeInt
    participant: str
    request: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    observations: AgentObservationBundle | None = None


class EpisodeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task: TaskIdentity
    response: NeMoGymResponse | None = None
    verification: EpisodeVerification | None = None
    agent_observations: AgentObservationBundle | None = None
    failure: EpisodeFailure | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.verification is None) == (self.failure is None):
            raise ValueError("exactly one of verification or failure is required")
        if self.verification is not None and self.response is None:
            raise ValueError("a verified episode requires a response")
        return self
