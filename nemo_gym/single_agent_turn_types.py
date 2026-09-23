# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the built-in single-agent-turn protocol."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from nemo_gym.base_resources_server import ResourcesVerifyRequest, ResourcesVerifyResponse
from nemo_gym.episode_types import (
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeFailure,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle


class SingleAgentTurnTaskInput(BaseModel):
    """Input loaded for one single-agent turn."""

    model_config = ConfigDict(extra="forbid")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    task_data: dict[str, JsonValue]


class SingleAgentTurnResult(BaseModel):
    """Successful single-agent-turn output."""

    model_config = ConfigDict(extra="forbid")

    verification: ResourcesVerifyResponse
    agent_observations: AgentObservationBundle | None = None


class SingleAgentTurnFailure(EpisodeFailure):
    """Add the failing protocol stage and any usable agent response."""

    stage: Literal["seed", "agent", "verification", "cleanup"] | None = None
    partial_response: NeMoGymResponse | None = None


class SingleAgentTurnRequest(BaseEpisodeRequest[SingleAgentTurnTaskInput]):
    """Request for one resources-backed agent turn."""


class SingleAgentTurnResponse(BaseEpisodeResponse[SingleAgentTurnResult]):
    """Response for one resources-backed agent turn."""

    failure: SingleAgentTurnFailure | None = None


class SingleAgentTurnVerificationInput(BaseModel):
    """Carry one Responses API activation to a resources server."""

    model_config = ConfigDict(extra="forbid")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class SingleAgentTurnResourcesVerifyRequest(ResourcesVerifyRequest[SingleAgentTurnVerificationInput]):
    """Verify one completed single-agent turn."""
