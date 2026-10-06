# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-neutral contracts for ordered, resumable agent activations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from nemo_gym.base_resources_server import ResourcesSeedSessionResponse
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeId
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnResult


class AgentContinuationRequirements(BaseModel):
    """Capabilities an interactive benchmark requires before candidate setup."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["native_conversation"] = "native_conversation"
    observations: list[str] = Field(default_factory=lambda: ["ordered_events", "timing"])


class AgentContinuationCapabilities(AgentContinuationRequirements):
    """Adapter promises, including runtime prerequisites and the meaning of limits."""

    runtime_prerequisites: dict[str, JsonValue] = Field(default_factory=dict)
    budget_semantics: dict[str, JsonValue] = Field(default_factory=dict)


class AgentActivationEvent(BaseModel):
    """An ordered native event; visible text and private reasoning remain distinct."""

    model_config = ConfigDict(extra="forbid")
    sequence: int = Field(ge=0)
    kind: Literal["step_start", "step_finish", "text", "tool_use", "reasoning", "compaction", "error"]
    text: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    arguments: JsonValue = None
    result: JsonValue = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AgentActivationObservation(BaseModel):
    """One activation's evidence; no benchmark-specific simulator projection is applied."""

    model_config = ConfigDict(extra="forbid")
    events: list[AgentActivationEvent] = Field(default_factory=list)
    raw_log: str | None = None
    elapsed_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    duration_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    harness_steps: int | None = Field(default=None, ge=0)
    provider_calls: int | None = Field(default=None, ge=0)
    agent_observations: AgentObservationBundle | None = None

    @model_validator(mode="after")
    def validate_event_order(self) -> "AgentActivationObservation":
        sequence = [event.sequence for event in self.events]
        if sequence != sorted(set(sequence)):
            raise ValueError("activation events must have strictly increasing sequence numbers")
        return self


class AgentActivationRequest(BaseModel):
    """Append exactly one input delta to a native conversation, indexed from zero."""

    model_config = ConfigDict(extra="forbid")
    agent_session_id: str = Field(min_length=1)
    episode_id: EpisodeId
    activation_id: int = Field(ge=0)
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class AgentActivationResponse(BaseModel):
    """Completion of an agent activation, independently of benchmark completion."""

    model_config = ConfigDict(extra="forbid")
    activation_id: int = Field(ge=0)
    response: NeMoGymResponse
    observation: AgentActivationObservation = Field(default_factory=AgentActivationObservation)
    turn_complete: bool = True
    stop_reason: str | None = None


class InteractiveAgentCloseReceipt(BaseModel):
    """Cumulative evidence returned only after adapter cleanup has completed."""

    model_config = ConfigDict(extra="forbid")
    agent_session_id: str
    agent_observations: AgentObservationBundle | None = None
    resources_cookies: dict[str, str] | None = None
    activations: list[AgentActivationResponse] = Field(default_factory=list)
    cleanup_confirmed: bool = False


class InteractiveResourcesSeedResponse(ResourcesSeedSessionResponse):
    """Resources prepares the candidate input and declares continuation requirements."""

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    continuation: AgentContinuationRequirements = Field(default_factory=AgentContinuationRequirements)


class ResourcesStepRequest(BaseModel):
    """Consult Resources once for each completed activation; identical retries must replay."""

    model_config = ConfigDict(extra="forbid")
    resources_session_id: str = Field(min_length=1)
    episode_id: EpisodeId
    activation: AgentActivationResponse


class ResourcesStepResponse(BaseModel):
    """Resources owns simulator messages, synthetic continuations, and stopping policy."""

    model_config = ConfigDict(extra="forbid")
    activation_id: int = Field(ge=0)
    continue_episode: bool
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None
    synthetic: bool = False
    stop_reason: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_next_input(self) -> "ResourcesStepResponse":
        if self.continue_episode != (self.responses_create_params is not None):
            raise ValueError("continuing requires next input; stopping must not provide next input")
        return self


class InteractiveVerificationInput(BaseModel):
    """Final candidate evidence, with confirmed cleanup before benchmark verification."""

    model_config = ConfigDict(extra="forbid")
    resources_session_id: str
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    activations: list[AgentActivationResponse]
    steps: list[ResourcesStepResponse]
    agent_close: InteractiveAgentCloseReceipt


class InteractiveAgentTaskInput(BaseModel):
    """Benchmark-owned materialized data; Resources produces the initial candidate prompt."""

    model_config = ConfigDict(extra="forbid")
    task_data: dict[str, JsonValue]


class InteractiveAgentRequest(BaseEpisodeRequest[InteractiveAgentTaskInput]):
    """Request one interactive episode."""


class InteractiveAgentResult(SingleAgentTurnResult):
    """Benchmark verdict and retained interactive evidence."""

    ng_activations: list[AgentActivationResponse] = Field(default_factory=list)
    ng_steps: list[ResourcesStepResponse] = Field(default_factory=list)
    ng_agent_close: InteractiveAgentCloseReceipt | None = None


class InteractiveAgentFailure(SingleAgentTurnFailure):
    """Retain completed work and lifecycle evidence when an episode cannot be measured."""

    activations: list[AgentActivationResponse] = Field(default_factory=list)
    steps: list[ResourcesStepResponse] = Field(default_factory=list)
    agent_close: InteractiveAgentCloseReceipt | None = None


class InteractiveAgentResponse(BaseEpisodeResponse[InteractiveAgentResult]):
    """A measured verdict or explicit lifecycle failure."""

    failure: InteractiveAgentFailure | None = None
