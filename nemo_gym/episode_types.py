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


# A cause chain deeper than this is cut; the last kept record says ``cause_truncated``.
MAX_FAILURE_CAUSE_DEPTH = 8
# The most failure events one reply or row carries.
MAX_FAILURE_EVENTS = 64


class ForeignErrorDetails(BaseModel):
    """Immutable SDK diagnostics selected by a boundary adapter before it translates an error.

    The adapter supplies the observed type, a scalar SDK message and, when available, its HTTP status.
    These fields travel on ``NgFailureEvent`` and then ``FailureEventRecord`` through HTTP to Collection.
    They are diagnostic data, not another cause link or a retry/scoring instruction. SDK objects, request
    bodies, outputs and response payloads stay outside this model. Direct oversized values are rejected;
    automatic capture bounds selected strings with an explicit truncation marker before construction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    exception_type: str = Field(min_length=1, max_length=256)
    message: str = Field(max_length=2000)
    http_status: int | None = Field(default=None, ge=100, le=599)


class FailureEventRecord(BaseModel):
    """One failure observation, an entry of a ``failure_events`` map keyed by its ``event_id``.

    ``event_id`` is assigned once where the failure is raised and kept through forwarding and retries, so a
    reader counts each failure once. ``cause_id`` names the explicit cause (``raise ... from``) by its id in the
    same map. ``type`` is the boundary class (``ModelFailure``, ``ResServerFailure``, ...) or ``untyped``;
    ``exception_type`` is the raised class when it differs. Kinds and reasons are each boundary's own. Request
    bodies, outputs, conversations and tool results are never part of a record: ``operation_id`` and
    ``attempt_id`` reference the records that hold them when the producer has those identities.

    Shared helpers serialize Python exceptions into this model, forward its bounded map on replies, and
    reconstruct safe boundary exceptions on receipt. Collection retains the map beside its existing outcome
    records, including observations accompanying success. This model stores no execution history and chooses
    no outcome: ``EpisodeFailure.terminal`` and existing consumer policy remain authoritative.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=128)
    type: str = Field(min_length=1, max_length=256)
    exception_type: str | None = Field(default=None, max_length=256)
    failure_reason: str = Field(max_length=2000)
    reason_truncated: bool | None = None
    failure_kind: str | None = None
    operation: str | None = Field(default=None, max_length=256)
    operation_id: str | None = Field(default=None, max_length=128)
    attempt_id: str | None = Field(default=None, max_length=128)
    stage: FailureStage | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    foreign_error: ForeignErrorDetails | None = None
    cause_id: str | None = Field(default=None, max_length=128)
    cause_truncated: bool | None = None

    @field_validator("failure_kind")
    @classmethod
    def _validate_failure_kind(cls, value: str | None) -> str | None:
        return validate_failure_kind(value)

    @model_serializer(mode="wrap")
    def _serialize_event(self, handler: SerializerFunctionWrapHandler):
        return {key: value for key, value in handler(self).items() if value is not None}


def validate_failure_event_map(events: Mapping[str, FailureEventRecord]) -> Mapping[str, FailureEventRecord]:
    """Check that ``events`` is keyed by event id and its causes resolve in it, acyclic and bounded."""
    if len(events) > MAX_FAILURE_EVENTS:
        raise ValueError(f"more than {MAX_FAILURE_EVENTS} failure events")
    for key, record in events.items():
        if key != record.event_id:
            raise ValueError(f"failure event {record.event_id!r} is keyed as {key!r}")
    for record in events.values():
        seen, current, depth = {record.event_id}, record, 0
        while current.cause_id is not None:
            depth += 1
            if depth > MAX_FAILURE_CAUSE_DEPTH:
                raise ValueError(f"cause chain deeper than {MAX_FAILURE_CAUSE_DEPTH}")
            if current.cause_id in seen:
                raise ValueError(f"cause chain repeats event {current.cause_id!r}")
            if current.cause_id not in events:
                raise ValueError(f"cause {current.cause_id!r} is not in the failure events")
            seen.add(current.cause_id)
            current = events[current.cause_id]
    return events


class EpisodeFailure(BaseModel):
    """Describe a failure using the same fields on the wire and in saved records.

    An environment server returns this envelope for one failed episode attempt. Collection uses its
    existing ``terminal`` flag and retry budget; ``failure_kind`` describes the failure without deciding
    terminality. Kind and stage are optional because a collector observing a lost reply may not know them.

    ``failure_id`` selects this envelope's current failure from ``failure_events``. The same map may retain
    earlier recovered observations and explicit causes, each stored once under its origin ID. Producers and
    HTTP helpers supply that evidence; this envelope neither creates an execution-history store nor changes
    scoring. A successful reply forwards observations separately without a current failure ID.
    """

    model_config = ConfigDict(extra="forbid")

    failure_reason: str = Field(max_length=2000)
    terminal: bool = Field(description="Whether rollout collection must not attempt this episode again.")
    failure_kind: str | None = None
    stage: FailureStage | None = None
    # (``nemo_gym.typed_exceptions``). Taken from the exception chain by type, never from text.
    # This failure's own entry in ``failure_events``.
    failure_id: str | None = Field(default=None, max_length=128)
    # Failure observations this reply forwards, keyed by event id: earlier failures a retry recovered from and
    # this failure with its explicit causes. Diagnostic only.
    failure_events: dict[str, FailureEventRecord] | None = None
    # How many observations were left out to keep the reply within its bounds.
    failure_events_dropped: int | None = Field(default=None, ge=1)

    @field_validator("failure_kind")
    @classmethod
    def _validate_failure_kind(cls, value: str | None) -> str | None:
        return validate_failure_kind(value)

    @field_validator("failure_events")
    @classmethod
    def _validate_failure_events(cls, value: dict[str, FailureEventRecord] | None):
        return None if value is None else validate_failure_event_map(value)

    @model_validator(mode="after")
    def _failure_id_is_an_event(self) -> Self:
        if self.failure_id is not None and self.failure_id not in (self.failure_events or {}):
            raise ValueError(f"failure_id {self.failure_id!r} is not in failure_events")
        return self

    # A return annotation would replace the public JSON Schema with that type.
    @model_serializer(mode="wrap")
    def _serialize_failure(self, handler: SerializerFunctionWrapHandler):
        # Omit metadata when the producer cannot classify the failure.
        # Protocol subclasses still serialize their own diagnostic fields.
        result = handler(self)
        for key in ("failure_kind", "stage", "failure_id", "failure_events", "failure_events_dropped"):
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


class BaseEpisodeResponse(BaseModel, Generic[EpisodeResultT]):
    """Return either a typed result or a handled failure."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    result: EpisodeResultT | None = None
    failure: EpisodeFailure | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.result is None) == (self.failure is None):
            raise ValueError("exactly one of result or failure is required")
        return self
