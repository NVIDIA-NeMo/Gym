# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the built-in single-agent-turn protocol."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from nemo_gym.core.episode_types import BaseEpisodeRequest, BaseEpisodeResponse
from nemo_gym.core.failure_records import EpisodeFailure
from nemo_gym.core.responses_api_types import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime.agent_context import AGENT_CONTEXT_KEY_NAME, AgentContext
from nemo_gym.server_runtime.base_resources_server import BaseVerifyResponse
from nemo_gym.trajectory.trajectory_record import AgentObservationBundle, TrajectoryRecord, VerifierInvocation


class SingleAgentTurnTaskInput(BaseModel):
    """Input loaded for one single-agent turn.

    ``agent_context`` is this environment's row convenience for its one agent session. Environments with
    several agents define their own task inputs and construct the context for each session explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    task_data: dict[str, JsonValue]
    # What the agent session receives beyond its model input. Kept out of task_data, which the Resources Server gets.
    agent_context: AgentContext | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="before")
    @classmethod
    def accept_flat_task_input(cls, value: object) -> object:
        """Accept flat task input while preserving the canonical ``task_data`` container.

        ``task_data`` is reserved for that container; non-conflicting flat fields are merged into it.
        A row's ``agent_context`` moves out of it into its own field.
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
        task_data = dict(task_data) | fields
        task_input = {"responses_create_params": response_params, "task_data": task_data}
        if (agent_context := task_data.pop(AGENT_CONTEXT_KEY_NAME, None)) is not None:
            task_input[AGENT_CONTEXT_KEY_NAME] = agent_context
        return task_input

    @field_validator("agent_context")
    @classmethod
    def omit_empty_agent_context(cls, agent_context: AgentContext | None) -> AgentContext | None:
        """An empty context is no context, so the task input dumps and digests as a row without the key."""
        return agent_context if agent_context is not None and agent_context.fields() else None


class SingleAgentTurnResult(BaseVerifyResponse):
    """Successful single-agent-turn output: the Resources verify response plus agent observations.

    The verify response fields stay at the top level so a stored rollout record has the same shape
    as an agent's `/run` result.
    Extra fields are allowed because each Resources Server returns its own benchmark-specific fields.
    """

    model_config = ConfigDict(extra="allow")

    ng_agent_observations: AgentObservationBundle | None = None
    # The harness's own turns from its session close. Rollout collection takes them over rebuilding turns from
    # the captured model calls; absent, the key stays out of the result as before.
    ng_trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)


class SingleAgentTurnFailure(EpisodeFailure):
    """Extend the shared failure with any usable agent response and agent evidence for diagnostics."""

    partial_response: NeMoGymResponse | None = None
    agent_observations: AgentObservationBundle | None = None
    # The harness's own turns from its session close, when the run failed after taking some: rollout collection
    # keeps them on the failure row as it does on a result, instead of rebuilding turns from the captured calls.
    trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)
    verifier_invocations: list[VerifierInvocation] | None = Field(default=None, exclude_if=lambda value: value is None)


class SingleAgentTurnRequest(BaseEpisodeRequest[SingleAgentTurnTaskInput]):
    """Request for one resources-backed agent turn."""


class SingleAgentTurnResponse(BaseEpisodeResponse[SingleAgentTurnResult]):
    """Response for one resources-backed agent turn."""

    failure: SingleAgentTurnFailure | None = None
