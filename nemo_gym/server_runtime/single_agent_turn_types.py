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


class SingleAgentTurnAgentContext(BaseModel):
    """A row's ``agent_context``, keyed by participant role; single_agent_turn has one role, ``agent``.

    Each role's context is what that participant's session receives, so the size limit applies per role.
    """

    model_config = ConfigDict(extra="forbid")

    agent: AgentContext | None = None

    @model_validator(mode="before")
    @classmethod
    def refuse_other_roles(cls, value: object) -> object:
        if isinstance(value, Mapping) and (others := sorted(set(value) - {"agent"})):
            raise ValueError(
                f"agent_context is keyed by participant role, and single_agent_turn has one role, 'agent'; got "
                f"{others}. Write the agent's context as {{'agent': {{'values': ..., 'artifacts': ...}}}}"
            )
        return value


class SingleAgentTurnTaskInput(BaseModel):
    """Input loaded for one single-agent turn."""

    model_config = ConfigDict(extra="forbid")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    task_data: dict[str, JsonValue]
    # What each participant's session receives beyond its model input, by role. Kept out of task_data, which the
    # Resources Server gets.
    agent_context: SingleAgentTurnAgentContext | None = Field(default=None, exclude_if=lambda value: value is None)

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
    def omit_empty_agent_context(
        cls, agent_context: SingleAgentTurnAgentContext | None
    ) -> SingleAgentTurnAgentContext | None:
        """An empty context is no context, so the task input dumps and digests as a row without the key."""
        if agent_context is None or agent_context.agent is None or not agent_context.agent.fields():
            return None
        return agent_context

    @property
    def agent_seed_context(self) -> AgentContext | None:
        """The context the ``agent`` role's session is seeded with, or None."""
        return self.agent_context.agent if self.agent_context is not None else None


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
