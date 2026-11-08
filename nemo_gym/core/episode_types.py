# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode identity and the wire contracts for environment servers."""

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

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.core.failure_records import EpisodeFailure, FailureReport


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


class RolloutCoordinates(BaseModel):
    """Where one rollout sits in its run, beside its episode identity.

    Rollout collection numbers every row with its task and its repeat of that task. A caller that compares the
    rollouts of one task as a group, such as a GenRM cohort, also names the group and the attempt that group is on.
    Environment Servers pass these to the Resources Server's `/verify` as they arrived; each is None when the
    caller did not send it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Strict, so a boolean or a numeric string is refused rather than read as an index.
    task_index: int | None = Field(default=None, ge=0, strict=True)
    rollout_index: int | None = Field(default=None, ge=0, strict=True, description="The task's repeat, from zero.")
    group_id: str | None = Field(default=None, min_length=1, strict=True)
    group_attempt: int | None = Field(
        default=None,
        ge=0,
        strict=True,
        description="Shared by every member of a group; a replacement of the whole group raises it.",
    )

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Self | None:
        """Read the coordinates a flat dataset row carries as ``_ng_*`` keys, or None when it carries none."""
        # Imported here because global_config imports feature modules that import the episode identity.
        from nemo_gym.config.global_config import (
            GROUP_ATTEMPT_KEY_NAME,
            GROUP_ID_KEY_NAME,
            ROLLOUT_INDEX_KEY_NAME,
            TASK_INDEX_KEY_NAME,
        )

        keys = {
            "task_index": TASK_INDEX_KEY_NAME,
            "rollout_index": ROLLOUT_INDEX_KEY_NAME,
            "group_id": GROUP_ID_KEY_NAME,
            "group_attempt": GROUP_ATTEMPT_KEY_NAME,
        }
        fields = {name: row[key] for name, key in keys.items() if row.get(key) is not None}
        return cls.model_validate(fields) if fields else None


def is_materialized_task_row(row: Mapping[str, object]) -> bool:
    """Return whether a row has a materialized-task envelope.

    This identifies the routing shape, not whether the envelope is valid. A malformed
    ``task_input`` must still route as a materialized task and fail task validation instead
    of falling through to the legacy run-request path.
    """
    return isinstance(row.get("task_id"), Mapping) and "task_input" in row


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
    rollout_coordinates: RolloutCoordinates | None = Field(default=None, exclude_if=lambda value: value is None)


class BaseEpisodeResponse(BaseModel, Generic[EpisodeResultT]):
    """Return either a typed result or a handled failure."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    result: EpisodeResultT | None = None
    failure: EpisodeFailure | None = None
    failure_report: FailureReport | None = Field(
        default=None, description="Diagnostic failures of this attempt, beside ``failure`` or a recovered result."
    )

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.result is None) == (self.failure is None):
            raise ValueError("exactly one of result or failure is required")
        return self

    # A return annotation would replace the public JSON Schema with that type.
    @model_serializer(mode="wrap")
    def _serialize_response(self, handler: SerializerFunctionWrapHandler):
        result = handler(self)
        if result.get("failure_report") is None:
            result.pop("failure_report", None)
        return result


# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
__getattr__ = moved_names_getattr(__name__)
