# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small shared contract for observations exposed by Agent integrations."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseInputItem


class ModelCallRef(BaseModel):
    """Stable identifiers an Agent integration can observe for one model call."""

    model_call_id: Optional[str] = None
    model_ref: Optional[ModelServerRef] = None
    response_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_join_key(self) -> "ModelCallRef":
        if not self.model_call_id and not (self.model_ref is not None and self.response_id):
            raise ValueError("model_call_id or both model_ref and response_id are required")
        return self


class AgentInvocation(BaseModel):
    """One root Agent or subagent conversation observed by a harness."""

    invocation_id: str
    parent_invocation_id: Optional[str] = None
    spawned_by_tool_call_id: Optional[str] = None
    status: Literal["completed", "failed", "incomplete", "unknown"] = Field(
        default="unknown", description="Harness-reported invocation outcome; unknown when not explicit."
    )
    model_calls: list[ModelCallRef] = Field(default_factory=list)
    conversation: list[NeMoGymResponseInputItem] = Field(
        default_factory=list,
        description="Normalized conversation items supported by this producer; gaps describe unavailable evidence.",
    )


class ToolCallObservation(BaseModel):
    """Timing observed for one tool call at an Agent-owned boundary."""

    invocation_id: str
    tool_call_id: str
    tool_name: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = None
    clock_id: Optional[str] = None
    timing_source: Optional[Literal["executor", "artifact"]] = None
    status: Literal["completed", "failed", "timeout", "incomplete", "unknown"] = "unknown"


class ContextCompactionObservation(BaseModel):
    """An explicit context-compaction event reported by the Agent harness."""

    invocation_id: str
    observed_at: Optional[float] = None
    trigger: Optional[str] = None
    tokens_before: Optional[int] = None
    tokens_after: Optional[int] = None


class ObservationGap(BaseModel):
    """A fact that the selected integration could not observe or join exactly."""

    code: str
    source: str
    invocation_id: Optional[str] = None
    detail: Optional[str] = None


class AgentObservationBundle(BaseModel):
    """Normalized observations returned by one Agent Server for one rollout."""

    source: str
    invocations: list[AgentInvocation] = Field(default_factory=list)
    tool_calls: list[ToolCallObservation] = Field(default_factory=list)
    compactions: list[ContextCompactionObservation] = Field(default_factory=list)
    gaps: list[ObservationGap] = Field(default_factory=list)


class AgentEpisode(BaseModel):
    """An Agent response and the observations available at its execution boundary."""

    response: NeMoGymResponse
    observations: AgentObservationBundle
