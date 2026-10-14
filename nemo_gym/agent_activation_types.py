# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-neutral contracts for ordered, resumable agent activations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle


class AgentContinuationRequirements(BaseModel):
    """Capabilities an interactive benchmark requires before candidate setup."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["native_conversation"] = "native_conversation"
    observations: list[str] = Field(default_factory=lambda: ["ordered_events", "timing"])
    requires_interaction_budget: bool = False


class AgentContinuationCapabilities(AgentContinuationRequirements):
    """Adapter promises, including runtime prerequisites and the meaning of limits."""

    supports_interaction_budget: bool = False


class InteractionBudget(BaseModel):
    """Immutable execution window shared across services with synchronized UTC clocks.

    Starts after setup and covers candidate activations plus environment work.
    Deadline owners cancel and await their work before returning a terminal checkpoint;
    an HTTP waiter timing out does not establish that the owned operation stopped.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    started_at_unix_seconds: float = Field(gt=0, allow_inf_nan=False)
    deadline_unix_seconds: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_deadline(self) -> "InteractionBudget":
        if self.deadline_unix_seconds <= self.started_at_unix_seconds:
            raise ValueError("interaction deadline must follow its start")
        return self


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
    interaction_budget: InteractionBudget | None = None
    activation_deadline_unix_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class AgentActivationResponse(BaseModel):
    """Completion of an agent activation, independently of benchmark completion."""

    model_config = ConfigDict(extra="forbid")
    activation_id: int = Field(ge=0)
    response: NeMoGymResponse
    observation: AgentActivationObservation = Field(default_factory=AgentActivationObservation)
    turn_complete: bool = True
    stop_reason: (
        Literal[
            "activation_timeout",
            "session_budget_exhausted",
            "model_budget_exhausted",
            "permission_denied",
            "execution_error",
        ]
        | None
    ) = None
