# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the built-in single-agent episode protocol."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from nemo_gym.base_resources_server import BaseVerifyResponse, ResourcesVerifyRequest
from nemo_gym.episode_types import (
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeFailure,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle


SINGLE_AGENT_TASK_INPUT_CONTRACT = "nemo_gym.single_agent.v1"


class SingleAgentTaskInput(BaseModel):
    """Input loaded from one single-agent task row."""

    model_config = ConfigDict(extra="forbid")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    task_data: dict[str, JsonValue]


class SingleAgentEpisodeResult(BaseModel):
    """Successful single-agent episode output."""

    model_config = ConfigDict(extra="forbid")

    verification: BaseVerifyResponse
    agent_observations: AgentObservationBundle | None = None


class SingleAgentEpisodeFailure(EpisodeFailure):
    """Add the failing protocol stage and any usable agent response."""

    stage: Literal["seed", "agent", "verification", "cleanup"] | None = None
    partial_response: NeMoGymResponse | None = None


class SingleAgentEpisodeRequest(BaseEpisodeRequest[SingleAgentTaskInput]):
    """Native request for the common resources-backed protocol."""


class SingleAgentEpisodeResponse(BaseEpisodeResponse[SingleAgentEpisodeResult]):
    """Native response for the common resources-backed protocol."""

    failure: SingleAgentEpisodeFailure | None = None


class ResponsesVerificationInput(BaseModel):
    """Carry one Responses API activation to a resources server."""

    model_config = ConfigDict(extra="forbid")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class ResponsesResourcesVerifyRequest(ResourcesVerifyRequest[ResponsesVerificationInput]):
    """Verify a completed single-agent Responses API activation."""
