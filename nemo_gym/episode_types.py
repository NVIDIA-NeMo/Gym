# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for environment servers."""

import re
from collections.abc import Mapping
from typing import Generic, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

from nemo_gym.correlation import validate_agent_instance
from nemo_gym.failure_kinds import FailureStage, validate_failure_kind


_CAPTURE_KEY_ATTEMPT = re.compile(r"-a([1-9][0-9]*)$")


class EpisodeId(BaseModel):
    """Identify one physical attempt of a logical rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    attempt: int = Field(default=0, ge=0)

    @field_validator("rollout_id")
    @classmethod
    def reserve_attempt_suffix(cls, rollout_id: str) -> str:
        """Keep the derived capture key injective without changing existing keys."""
        if _CAPTURE_KEY_ATTEMPT.search(rollout_id):
            raise ValueError("rollout_id must not end with the reserved attempt suffix '-a<N>'")
        return rollout_id

    @property
    def capture_key(self) -> str:
        """Return the attempt-qualified key used by capture routes."""
        return self.rollout_id if self.attempt == 0 else f"{self.rollout_id}-a{self.attempt}"

    @classmethod
    def from_capture_key(cls, capture_key: str) -> Self:
        """Invert ``capture_key``; the reserved attempt suffix makes the mapping one-to-one."""
        match = _CAPTURE_KEY_ATTEMPT.search(capture_key)
        if match is None:
            return cls(rollout_id=capture_key)
        return cls(rollout_id=capture_key[: match.start()], attempt=int(match.group(1)))


class TaskId(BaseModel):
    """Identify one task within a run-qualified taskset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    taskset: str = Field(
        min_length=1,
        description="Run-unique taskset name; use the '<environment>:<taskset>' convention for blended runs.",
    )
    task_id: str = Field(min_length=1)


def is_materialized_task_row(row: Mapping[str, object]) -> bool:
    """Return whether a row has a materialized-task envelope.

    This identifies the routing shape, not whether the envelope is valid. A malformed
    ``task_input`` must still route as a materialized task and fail task validation instead
    of falling through to the legacy run-request path.
    """
    return isinstance(row.get("task_id"), Mapping) and "task_input" in row


class EpisodeFailure(BaseModel):
    """Describe a failure using the same fields on the wire and in saved records.

    The explanation is ``failure_reason``. Kind and stage are optional. The kind
    identifies what happened; it does not determine terminality. A collector
    observing a lost reply may not know the episode's stage.
    """

    model_config = ConfigDict(extra="forbid")

    failure_reason: str = Field(max_length=2000)
    terminal: bool = Field(description="Whether rollout collection must not attempt this episode again.")
    failure_kind: str | None = None
    stage: FailureStage | None = None

    @field_validator("failure_kind")
    @classmethod
    def _validate_failure_kind(cls, value: str | None) -> str | None:
        return validate_failure_kind(value)

    # A return annotation would replace the public JSON Schema with that type.
    @model_serializer(mode="wrap")
    def _serialize_failure(self, handler: SerializerFunctionWrapHandler):
        # Omit metadata when the producer cannot classify the failure.
        # Protocol subclasses still serialize their own diagnostic fields.
        result = handler(self)
        for key in ("failure_kind", "stage"):
            if result.get(key) is None:
                result.pop(key, None)
        return result


TaskInputT = TypeVar("TaskInputT", bound=BaseModel)
EpisodeResultT = TypeVar("EpisodeResultT")


class MaterializedTask(BaseModel, Generic[TaskInputT]):
    """Carry durable task identity and protocol-shaped task input."""

    model_config = ConfigDict(extra="forbid")

    task_id: TaskId
    task_input: TaskInputT


class BaseEpisodeRequest(BaseModel, Generic[TaskInputT]):
    """Carry environment-neutral identity and typed task input."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task: MaterializedTask[TaskInputT]


# The episode-response field and result-row key mapping each named agent instance to its own result.
AGENT_INSTANCES_KEY = "agent_instances"


class AgentInstanceResult(BaseModel):
    """One agent instance's part of an episode result.

    ``terminal_response_id`` is the served response the agent instance ended on: the reply to its last
    activation, filled by the environment server. ``reward`` is optional and set by the environment; without it
    the trainer chooses a fallback, such as the episode reward.
    """

    model_config = ConfigDict(extra="forbid")

    terminal_response_id: str | None = None
    reward: float | None = None

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler: SerializerFunctionWrapHandler):
        return {key: value for key, value in handler(self).items() if value is not None}


class BaseEpisodeResponse(BaseModel, Generic[EpisodeResultT]):
    """Return either a typed result or a handled failure.

    ``agent_instances`` reports each named agent instance of a multi-agent episode; it is omitted for a
    single-agent episode, whose result keeps its flat fields.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    result: EpisodeResultT | None = None
    failure: EpisodeFailure | None = None
    agent_instances: dict[str, AgentInstanceResult] | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.result is None) == (self.failure is None):
            raise ValueError("exactly one of result or failure is required")
        for agent_instance in self.agent_instances or {}:
            validate_agent_instance(agent_instance)
        return self

    @model_serializer(mode="wrap")
    def _omit_agent_instances(self, handler: SerializerFunctionWrapHandler):
        result = handler(self)
        if self.agent_instances is None:
            result.pop("agent_instances", None)
        return result
