# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed and verification contracts for the NeMo UserSim Resources Server."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    is_namespaced,
    is_registered,
    validate_failure_kind,
)
from nemo_gym.server_runtime import (
    ResourcesSeedSessionResponse,
    ResourcesVerifyRequest,
)
from nemo_gym.trajectory import AgentObservationBundle
from resources_servers.nemo_user_sim.task_data import TaskData as UserSimTaskInput  # noqa: F401  (task contract)
from resources_servers.nemo_user_sim.task_data import (
    UserSimAgentRole,
)


USERSIM_EPISODE_PROTOCOL = "usersim.ConversationLoop"
# A failed simulation whose outcome names no failure_class.
SIMULATION_FAILED = "usersim:simulation_failed"


@dataclass(frozen=True)
class UserSimFailure:
    """The failure fields a UserSim episode records for a failure UserSim reported in its outcome."""

    mask_sample: bool
    failure_kind: str | None
    failure_reason: str | None


def usersim_failure(outcome: Mapping[str, Any]) -> UserSimFailure | None:
    """Classify a UserSim simulation outcome by who caused its failure, or return None when nothing failed.

    A failure attributed to the assistant model is the policy's own and stays in the score. Any other failed
    simulation (user model, judge, tool simulation, infrastructure) is masked. Verification and an episode that
    skips verification both apply this rule, so skipping never turns a failed simulation into an ordinary row.
    """
    failure_kind = _usersim_failure_kind(outcome.get("failure_class"))
    failure_reason = outcome.get("failure_detail") or outcome.get("failure_reason")
    if outcome.get("failure_attribution") == "assistant_model":
        return UserSimFailure(mask_sample=False, failure_kind=failure_kind, failure_reason=failure_reason)
    if outcome.get("status") != "failed":
        return None
    return UserSimFailure(
        mask_sample=True, failure_kind=failure_kind or SIMULATION_FAILED, failure_reason=failure_reason
    )


def _usersim_failure_kind(value: Any) -> str | None:
    """Map a UserSim failure_class to a registered or ``usersim:``-namespaced Gym failure kind."""
    if not value:
        return None
    name = str(value)
    if is_registered(name) or is_namespaced(name):
        return name
    normalized = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "simulation_failed"
    return f"usersim:{normalized}"


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
