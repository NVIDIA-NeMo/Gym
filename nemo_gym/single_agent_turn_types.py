# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the built-in single-agent-turn protocol."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.episode_types import (
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeFailure,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord


class SingleAgentTurnTaskInput(BaseModel):
    """Input loaded for one single-agent turn."""

    model_config = ConfigDict(extra="forbid")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    task_data: dict[str, JsonValue]
    agent_timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Agent execution budget after setup; close the agent before grading its preserved state on timeout.",
    )


class SingleAgentTurnResult(BaseVerifyResponse):
    """Successful single-agent-turn output: the Resources verify response plus agent observations.

    The verify response fields stay at the top level so a stored rollout record has the same shape
    as an agent's `/run` result.
    Extra fields are allowed because each Resources Server returns its own benchmark-specific fields.
    """

    model_config = ConfigDict(extra="allow")

    ng_agent_observations: AgentObservationBundle | None = None
    ng_trajectory: TrajectoryRecord | None = None
    agent_timed_out: bool = False
    agent_timeout_seconds: float | None = None


class SingleAgentTurnFailure(EpisodeFailure):
    """Add the failing protocol stage and any usable agent response."""

    stage: Literal["seed", "agent", "verification", "cleanup"] | None = None
    partial_response: NeMoGymResponse | None = None
    ng_agent_observations: AgentObservationBundle | None = None
    ng_trajectory: TrajectoryRecord | None = None
    cleanup_error: str | None = None


class SingleAgentTurnRequest(BaseEpisodeRequest[SingleAgentTurnTaskInput]):
    """Request for one resources-backed agent turn."""


class SingleAgentTurnResponse(BaseEpisodeResponse[SingleAgentTurnResult]):
    """Response for one resources-backed agent turn."""

    failure: SingleAgentTurnFailure | None = None
