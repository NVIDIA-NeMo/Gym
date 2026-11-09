# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode contracts owned by the NeMo UserSim Environment Server."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.core import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure, validate_failure_kind
from resources_servers.nemo_user_sim.episode_contracts import (
    USERSIM_EPISODE_PROTOCOL,
    UserSimInvocation,
    UserSimSimulationResult,
    UserSimTaskInput,
    UserSimVerification,
    usersim_failure,
)


class UserSimEpisodeResult(BaseModel):
    """Successful UserSim episode output."""

    model_config = ConfigDict(extra="forbid")

    reward: float
    mask_sample: bool = False
    failure_kind: str | None = None
    failure_reason: str | None = None
    reward_components: dict[str, float]
    # None only when the episode skipped verification, which the flag below marks.
    verification: UserSimVerification | None
    usersim_result: UserSimSimulationResult
    invocations: list[UserSimInvocation]
    episode_interaction_protocol: str = USERSIM_EPISODE_PROTOCOL
    # Set when skip_verification replaced /verify, so anything reading the result can tell the fixed reward from a
    # measured one. Left out of the result otherwise.
    verification_skipped: bool = Field(default=False, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def verified_or_skipped(self) -> Self:
        """A result carries a verification exactly when /verify ran."""
        if (self.verification is None) != self.verification_skipped:
            raise ValueError("a UserSim episode result carries a verification exactly when it was not skipped")
        return self

    @field_validator("failure_kind")
    @classmethod
    def validate_shared_failure_kind(cls, value: str | None) -> str | None:
        """Apply Gym's shared registered-or-namespaced failure-kind validation."""
        return validate_failure_kind(value)

    @classmethod
    def skipped(
        cls,
        *,
        reward: float,
        usersim_result: UserSimSimulationResult,
        invocations: list[UserSimInvocation],
    ) -> Self:
        """The result of an episode whose verification skip_verification replaced with a fixed reward.

        A failed simulation does not earn the fixed reward. It scores 0.0 with the failure fields /verify gives it,
        masked unless the assistant model caused the failure, so skipping never hides a failure.
        """
        failure = usersim_failure(usersim_result.simulation_outcome)
        return cls(
            reward=reward if failure is None else 0.0,
            mask_sample=failure is not None and failure.mask_sample,
            failure_kind=failure.failure_kind if failure is not None else None,
            failure_reason=failure.failure_reason if failure is not None else None,
            reward_components={},
            verification=None,
            usersim_result=usersim_result,
            invocations=invocations,
            verification_skipped=True,
        )

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
