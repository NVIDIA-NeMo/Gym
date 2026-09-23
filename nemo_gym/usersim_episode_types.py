# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the NeMo UserSim episode protocol."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.base_resources_server import (
    ResourcesSeedSessionResponse,
    ResourcesVerifyRequest,
)
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle


USERSIM_MODEL_ALIASES = frozenset(
    {"user_model", "assistant_model", "api_response_model", "judge_model", "summary_model"}
)
USERSIM_EPISODE_PROTOCOL = "usersim.ConversationLoop"


class UserSimScenario(BaseModel):
    """One fully resolved UserSim scenario."""

    model_config = ConfigDict(extra="allow")

    persona: dict[str, Any]
    probe_type: str = "general_open_ended"
    theme: dict[str, Any] | str
    goal: str = ""
    locale: str = "en_US"
    probe_data: dict[str, Any] = Field(default_factory=dict)


class UserSimTheme(BaseModel):
    """One selectable theme used to construct a scenario."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1)
    goal: str = Field(min_length=1)


class UserSimSamplingRequest(BaseModel):
    """Dataset-owned inputs used to select one replayable scenario."""

    model_config = ConfigDict(extra="forbid")

    locale: str = Field("en_US", pattern=r"^[A-Za-z0-9_]+$")
    seed: int
    probe_type: str | None = None


class UserSimProtocolConfig(BaseModel):
    """Run-wide subset of ``ConversationSimulatorConfig`` owned by the Environment Server."""

    model_config = ConfigDict(extra="forbid")

    max_query_attempts: int = Field(3, ge=1)
    max_assistant_attempts: int = Field(1, ge=1)
    enforce_user_language: bool = True
    user_language_min_script_compliance: float = Field(0.6, ge=0, le=1)
    user_language_min_letters: int = Field(8, ge=0)
    incremental_disclosure_ratio: float = Field(0.6, ge=0, le=1)
    persona_grounding_ratio: float = Field(1, ge=0, le=1)
    context_compression: bool = True
    compression_window: int = Field(1, ge=1)
    store_reasoning: bool = True
    random_seed: int | None = None
    verbosity: int = Field(1, ge=0, le=2)


class UserSimTaskInput(BaseModel):
    """Durable input loaded from one UserSim task row."""

    model_config = ConfigDict(extra="forbid")

    sampling: UserSimSamplingRequest
    probe_data: dict[str, Any] = Field(default_factory=dict)
    model_responses_create_params: dict[str, NeMoGymResponseCreateParamsNonStreaming] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_unknown_model_aliases(self) -> "UserSimTaskInput":
        unknown = set(self.model_responses_create_params) - USERSIM_MODEL_ALIASES
        if unknown:
            raise ValueError(f"model_responses_create_params contains unknown aliases: {sorted(unknown)}")
        return self


class ResolvedUserSimContext(BaseModel):
    """Selection provenance required to replay a resolved scenario."""

    model_config = ConfigDict(extra="forbid")

    locale: str
    seed: int
    personas_dataset_version: str
    personas_panel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    usersim_revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class UserSimSeedResponse(ResourcesSeedSessionResponse):
    """Return resources-session identity plus the resolved scenario."""

    scenario: UserSimScenario
    usersim_context: ResolvedUserSimContext
    assistant_tools: list[dict[str, Any]] = Field(default_factory=list)


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
    """One ordered UserSim model alias activation."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    alias: str
    executor: Literal["agent", "model"]
    request: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    observations: AgentObservationBundle | None = None
    state_after: dict[str, Any] | None = None
    termination_reason: str | None = None

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, alias: str) -> str:
        if alias not in USERSIM_MODEL_ALIASES:
            raise ValueError(f"unknown UserSim model alias: {alias}")
        return alias


class UserSimVerificationInput(BaseModel):
    """Carry the completed protocol to the Resources Server verifier."""

    model_config = ConfigDict(extra="forbid")

    scenario: UserSimScenario
    usersim_context: ResolvedUserSimContext
    usersim_result: UserSimSimulationResult
    invocations: list[UserSimInvocation]
    episode_interaction_protocol: str = USERSIM_EPISODE_PROTOCOL


class UserSimVerifyRequest(ResourcesVerifyRequest[UserSimVerificationInput]):
    """Verify one completed UserSim episode."""


class UserSimVerification(BaseModel):
    """Typed verifier output preserved in the Environment Server result."""

    model_config = ConfigDict(extra="forbid")

    reward: float
    reward_components: dict[str, float]
    scenario_completed: bool
    verifier_data: dict[str, Any] = Field(default_factory=dict)
    native_usersim_result: UserSimSimulationResult | None = None


class UserSimEpisodeResult(BaseModel):
    """Successful UserSim episode output."""

    model_config = ConfigDict(extra="forbid")

    verification: UserSimVerification
    usersim_result: UserSimSimulationResult
    invocations: list[UserSimInvocation]
    episode_interaction_protocol: str = USERSIM_EPISODE_PROTOCOL


class UserSimEpisodeFailure(EpisodeFailure):
    """Add the failing UserSim protocol stage."""

    stage: Literal["seed", "participant", "simulation", "verification", "cleanup"] | None = None


class UserSimEpisodeRequest(BaseEpisodeRequest[UserSimTaskInput]):
    """Native request for the UserSim episode protocol."""


class UserSimEpisodeResponse(BaseEpisodeResponse[UserSimEpisodeResult]):
    """Native response for the UserSim episode protocol."""

    failure: UserSimEpisodeFailure | None = None
