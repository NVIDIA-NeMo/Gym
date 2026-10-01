# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the NeMo UserSim episode protocol."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.base_resources_server import (
    ResourcesSeedSessionResponse,
    ResourcesVerifyRequest,
)
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle


UserSimAgentRole = Literal["user", "assistant", "judge", "summary"]
USERSIM_EPISODE_PROTOCOL = "usersim.ConversationLoop"


class UserSimTaskInput(BaseModel):
    """Fully resolved, provenance-pinned input loaded from one prepared task row."""

    model_config = ConfigDict(extra="forbid")

    resolved_row: dict[str, Any]
    responses_create_params: dict[UserSimAgentRole, NeMoGymResponseCreateParamsNonStreaming] = Field(
        default_factory=dict
    )


class UserSimSeedResponse(ResourcesSeedSessionResponse):
    """Return resources-session identity plus the unchanged resolved row."""

    resolved_row: dict[str, Any]
    assistant_tools: list[dict[str, Any]] = Field(default_factory=list)


class ActivationUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ActivationRequest(BaseModel):
    """One non-assistant model activation requested by ConversationRuntime."""

    model_config = ConfigDict(extra="forbid")

    activation_id: str = Field(min_length=1)
    role: Literal["user", "judge", "summary"]
    model_alias: str = Field(min_length=1)
    messages: list[dict[str, Any]]
    parameters: dict[str, Any]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tools_enabled: bool = False
    continues_turn: bool = False


class ActivationResult(BaseModel):
    """One externally executed non-assistant result."""

    model_config = ConfigDict(extra="forbid")

    activation_id: str = Field(min_length=1)
    response: dict[str, Any]
    usage: ActivationUsage | None = None


class AssistantLoopPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "single", "multi"]
    max_model_calls: int = Field(ge=1)
    max_tool_calls: int | None = Field(default=None, ge=0)
    final_synthesis: bool
    replay_reasoning: bool = False
    project_document_tool_results: bool = False


class ToolTurnContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(min_length=1)
    first_tool_turn_idx: int = Field(ge=0)
    max_tool_calls: int | None = Field(default=None, ge=0)
    state_snapshot: dict[str, Any]


class AssistantTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(min_length=1)
    role: Literal["assistant"] = "assistant"
    model_alias: Literal["assistant_model"] = "assistant_model"
    messages: list[dict[str, Any]]
    parameters: dict[str, Any]
    tools: list[dict[str, Any]]
    loop_policy: AssistantLoopPolicy
    tool_context: ToolTurnContext


class AssistantModelCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: dict[str, Any]
    usage: ActivationUsage | None = None


class ToolCallReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    raw_tool_call: dict[str, Any]
    payload: str
    turn_idx: int = Field(ge=0)
    call_idx: int = Field(ge=0)
    effect_state: dict[str, Any]


class ToolRoundReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    round_id: str
    receipts: list[ToolCallReceipt]
    limit_reached: bool = False


class CompletedTurnEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    rounds: list[ToolRoundReceipt]
    final_effect_state: dict[str, Any]

    @classmethod
    def from_unexecuted_turn(cls, context: ToolTurnContext) -> "CompletedTurnEvidence":
        """Build evidence for an Assistant turn that made no Resources calls."""
        return cls(
            turn_id=context.turn_id,
            rounds=[],
            final_effect_state={
                "metadata": context.state_snapshot.get("metadata", {}),
                "outcome": context.state_snapshot.get("outcome", {}),
            },
        )


class CompletedAssistantTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    transcript: list[dict[str, Any]]
    model_calls: list[AssistantModelCall]
    evidence: CompletedTurnEvidence


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


class UserSimEpisodeLifecycleComplete(BaseModel):
    """Terminal event from the Resources-owned native lifecycle."""

    model_config = ConfigDict(extra="forbid")

    complete: Literal[True] = True
    result: UserSimSimulationResult


UserSimLifecycleEvent = ActivationRequest | AssistantTurnRequest | UserSimEpisodeLifecycleComplete


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
