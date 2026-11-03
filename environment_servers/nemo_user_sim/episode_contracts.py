# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode contracts owned by the NeMo UserSim Environment Server."""

from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator

from nemo_gym.core import validate_failure_kind
from nemo_gym.server_runtime import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
from resources_servers.nemo_user_sim.episode_contracts import (
    USERSIM_EPISODE_PROTOCOL,
    UserSimInvocation,
    UserSimSimulationResult,
    UserSimTaskInput,
    UserSimVerification,
)


class UserSimEpisodeResult(BaseModel):
    """Successful UserSim episode output."""

    model_config = ConfigDict(extra="forbid")

    reward: float
    mask_sample: bool = False
    failure_kind: str | None = None
    failure_reason: str | None = None
    reward_components: dict[str, float]
    verification: UserSimVerification
    usersim_result: UserSimSimulationResult
    invocations: list[UserSimInvocation]
    episode_interaction_protocol: str = USERSIM_EPISODE_PROTOCOL

    @field_validator("failure_kind")
    @classmethod
    def validate_shared_failure_kind(cls, value: str | None) -> str | None:
        """Apply Gym's shared registered-or-namespaced failure-kind validation."""
        return validate_failure_kind(value)

    @classmethod
    def from_verification(
        cls,
        *,
        verification: UserSimVerification,
        usersim_result: UserSimSimulationResult,
        invocations: list[UserSimInvocation],
    ) -> Self:
        """Project verifier scoring fields onto Gym's persisted episode-result contract."""
        return cls(
            reward=verification.reward,
            mask_sample=verification.mask_sample,
            failure_kind=verification.failure_kind,
            failure_reason=verification.failure_reason,
            reward_components=verification.reward_components,
            verification=verification.model_copy(update={"usersim_result": None}),
            usersim_result=usersim_result,
            invocations=invocations,
        )


class UserSimEpisodeFailure(EpisodeFailure):
    """Use the shared episode failure-stage vocabulary."""


class UserSimEpisodeRequest(BaseEpisodeRequest[UserSimTaskInput]):
    """Request for the UserSim episode protocol."""


class UserSimEpisodeResponse(BaseEpisodeResponse[UserSimEpisodeResult]):
    """Response for the UserSim episode protocol."""

    failure: UserSimEpisodeFailure | None = None
