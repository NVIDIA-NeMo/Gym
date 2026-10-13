# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the NeMo UserSim episode protocol."""

import json
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.base_resources_server import (
    ResourcesSeedSessionResponse,
    ResourcesVerifyRequest,
)
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
from nemo_gym.failure_kinds import validate_failure_kind
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from resources_servers.nemo_user_sim.task_data import (
    TaskData as UserSimTaskInput,
)
from resources_servers.nemo_user_sim.task_data import (
    UserSimAgentRole,
)


USERSIM_EPISODE_PROTOCOL = "usersim.ConversationLoop"


class UserSimSeedResponse(ResourcesSeedSessionResponse):
    """Return the seeded Resources session identity."""


class UserSimSimulationResult(BaseModel):
    """Typed view of UserSim's Data Designer-compatible output columns."""

    model_config = ConfigDict(extra="allow")

    conversation_messages: list[dict[str, Any]]
    conversation_status: bool
    simulation_outcome: dict[str, Any] = Field(default_factory=dict)
    conversation_metadata: dict[str, Any] | None = None
    simulation_traces: list[dict[str, Any]] | None = None

    @field_validator(
        "conversation_messages",
        "simulation_outcome",
        "conversation_metadata",
        "simulation_traces",
        mode="before",
    )
    @classmethod
    def decode_json_columns(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value


class UserSimInvocation(BaseModel):
    """One ordered UserSim participant-Agent or support-model activation."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    role: UserSimAgentRole
    request: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    observations: AgentObservationBundle | None = None
    state_after: dict[str, Any] | None = None
    termination_reason: str | None = None


class UserSimVerificationInput(BaseModel):
    """Carry the completed protocol to the Resources Server verifier."""

    model_config = ConfigDict(extra="forbid")

    resolved_row: dict[str, Any]
    usersim_result: UserSimSimulationResult
    invocations: list[UserSimInvocation]
    episode_interaction_protocol: str = USERSIM_EPISODE_PROTOCOL


class UserSimVerifyRequest(ResourcesVerifyRequest[UserSimVerificationInput]):
    """Verify one completed UserSim episode."""


class UserSimVerification(BaseModel):
    """Typed verifier output preserved in the Environment Server result."""

    model_config = ConfigDict(extra="forbid")

    reward: float
    mask_sample: bool = False
    failure_kind: str | None = None
    failure_reason: str | None = None
    reward_components: dict[str, float]
    scenario_completed: bool
    verifier_data: dict[str, Any] = Field(default_factory=dict)
    usersim_result: UserSimSimulationResult | None = None

    @field_validator("failure_kind")
    @classmethod
    def validate_shared_failure_kind(cls, value: str | None) -> str | None:
        """Apply Gym's shared registered-or-namespaced failure-kind validation."""
        return validate_failure_kind(value)


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
