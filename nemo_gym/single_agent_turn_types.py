# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the built-in single-agent-turn protocol."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from nemo_gym.base_resources_server import BaseVerifyResponse
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

    @model_validator(mode="before")
    @classmethod
    def accept_flat_task_input(cls, value: object) -> object:
        """Accept flat task input while preserving the canonical ``task_data`` container.

        ``task_data`` is reserved for that container; non-conflicting flat fields are merged into it.
        """
        if not isinstance(value, Mapping):
            return value
        fields = dict(value)
        response_params = fields.pop("responses_create_params", None)
        task_data = fields.pop("task_data", {})
        if not isinstance(task_data, Mapping):
            return value  # Let the field validator report the malformed canonical container.
        for key in fields.keys() & task_data.keys():
            if fields[key] != task_data[key]:
                raise ValueError(f"Conflicting task field {key!r} inside and outside task_data")
        return {"responses_create_params": response_params, "task_data": dict(task_data) | fields}


class SingleAgentTurnResult(BaseVerifyResponse):
    """Successful single-agent-turn output: the Resources verify response plus agent observations.

    The verify response fields stay at the top level so a stored rollout record has the same shape
    as an agent's `/run` result.
    Extra fields are allowed because each Resources Server returns its own benchmark-specific fields.
    """

    model_config = ConfigDict(extra="allow")

    ng_agent_observations: AgentObservationBundle | None = None


class SingleAgentTurnFailure(EpisodeFailure):
    """Extend the shared failure with any usable agent response for diagnostics."""

    partial_response: NeMoGymResponse | None = None


class SingleAgentTurnRequest(BaseEpisodeRequest[SingleAgentTurnTaskInput]):
    """Request for one resources-backed agent turn."""


class SingleAgentTurnResponse(BaseEpisodeResponse[SingleAgentTurnResult]):
    """Response for one resources-backed agent turn."""

    failure: SingleAgentTurnFailure | None = None
