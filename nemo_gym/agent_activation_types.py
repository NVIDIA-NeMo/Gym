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


class AgentContinuationCapabilities(AgentContinuationRequirements):
    """Adapter promises, including runtime prerequisites and the meaning of limits."""


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
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)


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
            "model_budget_exhausted",
            "permission_denied",
            "content_filter",
            "interrupted",
            "execution_error",
        ]
        | None
    ) = None
