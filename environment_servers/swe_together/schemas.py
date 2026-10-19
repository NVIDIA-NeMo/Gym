# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWE-Together episode inputs and retained evidence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from environment_servers.swe_together.task import TaskData
from nemo_gym.agent_activation_types import AgentActivationResponse
from nemo_gym.base_responses_api_agent import AgentCloseSessionResponse
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseUsage
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnResult


class SWETTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_data: TaskData
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None


class UserTurn(BaseModel):
    """The accepted simulator decision and the environment's stopping decision."""

    activation_id: int
    kind: Literal["continue", "stop"]
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None
    synthetic: bool = False
    stop_reason: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SWETResult(SingleAgentTurnResult):
    judge_score: float | None = None
    task_id: str
    ng_activations: list[AgentActivationResponse] = Field(default_factory=list)
    ng_steps: list[UserTurn] = Field(default_factory=list)
    ng_agent_close: AgentCloseSessionResponse | None = None


class SWETFailure(SingleAgentTurnFailure):
    """Execution failures retain their diagnostics without becoming candidate zeroes."""

    activations: list[AgentActivationResponse] = Field(default_factory=list)
    agent_close: AgentCloseSessionResponse | None = None
    cleanup_receipts: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_directory: str | None = None
    dependency_error: dict[str, JsonValue] | None = None


class SWETRequest(BaseEpisodeRequest[SWETTaskInput]):
    pass


class SWETResponse(BaseEpisodeResponse[SWETResult]):
    failure: SWETFailure | None = None


def episode_response(activations: list[AgentActivationResponse]) -> NeMoGymResponse:
    """Retain the final output and report usage over the whole collected episode.

    Each activation retains its own usage. A missing counter makes the episode
    total unknown, rather than silently reporting a partial sum as complete.
    """
    usages = [activation.response.usage for activation in activations]
    usage = NeMoGymResponseUsage.sum_from_list(usages) if all(item is not None for item in usages) else None
    return activations[-1].response.model_copy(update={"usage": usage}, deep=True)
