# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-neutral inputs and results for a Resources-managed user simulator."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from nemo_gym.agent_activation_types import AgentActivationResponse, AgentContinuationRequirements, InteractionBudget
from nemo_gym.agent_runtime_policy import AgentRuntimePolicy
from nemo_gym.base_resources_server import ResourcesSeedSessionRequest, ResourcesSeedSessionResponse
from nemo_gym.base_responses_api_agent import AgentCloseSessionResponse
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import TrajectoryRecord
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnResult


class InteractiveResourcesSeedRequest(ResourcesSeedSessionRequest):
    """Bound task preparation to the environment's setup deadline."""

    setup_deadline_unix_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class InteractiveResourcesSeedResponse(ResourcesSeedSessionResponse):
    """Resources prepares the candidate input and declares continuation requirements."""

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    continuation: AgentContinuationRequirements = Field(default_factory=AgentContinuationRequirements)
    supports_interaction_budget: bool = False
    runtime_policy: AgentRuntimePolicy | None = None


class NextUserTurnRequest(BaseModel):
    """Consult Resources once for each completed activation; identical retries must replay."""

    model_config = ConfigDict(extra="forbid")
    resources_session_id: str = Field(min_length=1)
    episode_id: EpisodeId
    activation: AgentActivationResponse
    interaction_budget: InteractionBudget | None = None


class NextUserTurnResponse(BaseModel):
    """Resources owns simulator messages, synthetic continuations, and stopping policy."""

    model_config = ConfigDict(extra="forbid")
    activation_id: int = Field(ge=0)
    kind: Literal["continue", "stop"]
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None
    synthetic: bool = False
    stop_reason: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_next_input(self) -> "NextUserTurnResponse":
        if (self.kind == "continue") != (self.responses_create_params is not None):
            raise ValueError("continuing requires next input; stopping must not provide next input")
        return self


class InteractiveVerificationInput(BaseModel):
    """Final candidate evidence, with confirmed cleanup before benchmark verification."""

    model_config = ConfigDict(extra="forbid")
    resources_session_id: str
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    activations: list[AgentActivationResponse]
    steps: list[NextUserTurnResponse]
    agent_close: AgentCloseSessionResponse
    grading_deadline_unix_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class InteractiveAgentTaskInput(BaseModel):
    """Benchmark-owned materialized data; Resources produces the initial candidate prompt."""

    model_config = ConfigDict(extra="forbid")
    task_data: dict[str, JsonValue]


class InteractiveAgentRequest(BaseEpisodeRequest[InteractiveAgentTaskInput]):
    """Request one interactive episode."""


class InteractiveAgentResult(SingleAgentTurnResult):
    """Benchmark verdict and retained interactive evidence."""

    ng_activations: list[AgentActivationResponse] = Field(default_factory=list)
    ng_steps: list[NextUserTurnResponse] = Field(default_factory=list)
    ng_agent_close: AgentCloseSessionResponse | None = None
    ng_trajectory: TrajectoryRecord | None = None


class InteractiveDependencyError(BaseModel):
    """Bounded dependency diagnostics, excluding transport headers and request metadata."""

    model_config = ConfigDict(extra="forbid")
    status_code: int
    body: str | None = Field(default=None, max_length=8192)
    body_truncated: bool = False


class InteractiveAgentFailure(SingleAgentTurnFailure):
    """Retain completed work and lifecycle evidence when an episode cannot be measured."""

    activations: list[AgentActivationResponse] = Field(default_factory=list)
    steps: list[NextUserTurnResponse] = Field(default_factory=list)
    agent_close: AgentCloseSessionResponse | None = None
    dependency_error: InteractiveDependencyError | None = None


class InteractiveAgentResponse(BaseEpisodeResponse[InteractiveAgentResult]):
    """A measured verdict or explicit lifecycle failure."""

    failure: InteractiveAgentFailure | None = None
