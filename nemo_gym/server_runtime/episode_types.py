# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for environment servers."""

import uuid
from collections.abc import Mapping
from enum import StrEnum
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
from nemo_gym.core.episode import EpisodeId, TaskId
from nemo_gym.core.failure_kinds import FailureStage, require_failure_kind, validate_failure_kind


def is_materialized_task_row(row: Mapping[str, object]) -> bool:
    """Return whether a row has a materialized-task envelope.

    This identifies the routing shape, not whether the envelope is valid. A malformed
    ``task_input`` must still route as a materialized task and fail task validation instead
    of falling through to the legacy run-request path.
    """
    return isinstance(row.get("task_id"), Mapping) and "task_input" in row


class FailureBoundary(StrEnum):
    """The six places a failure is observed, as the failure-boundaries RFC names them.

    Each one reports what it observed; none of them chooses retries, terminality or scores.
    """

    MODEL = "model"
    """A model client's request or one HTTP attempt of it."""
    HARNESS = "harness"
    """The program an agent server drives in a sandbox."""
    AGENT = "agent"
    """An agent server route (``/run``, ``/v1/responses``, a session seed)."""
    RESOURCES_SERVER = "resources_server"
    """A resources server's verifier, tool or session (a judge call, for example)."""
    ENVIRONMENT = "environment"
    """An environment server preparing, running or verifying an episode."""
    COLLECTION = "collection"
    """Rollout collection dispatching an episode or reading its reply."""


class SdkErrorDetails(BaseModel):
    """Diagnostics an adapter selected from a third-party SDK error before it raised its own failure.

    The adapter chooses the message (for example the SDK's ``message`` field), because ``str()`` of an SDK
    error can contain request data. The SDK object, its request and its response body are never stored.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    exception_type: str = Field(min_length=1, max_length=256)
    message: str = Field(max_length=2000)
    http_status: int | None = Field(default=None, ge=100, le=599)


class FailureEvent(BaseModel):
    """One observed failure: who observed it, what failed, and which failure caused it.

    This is the validated wire projection of a ``NgFailureEvent``, whose fields and origin identity live on
    the exception itself. ``cause_id`` names another Gym failure in the same map. Foreign SDK facts live in
    ``foreign_error``; they are not boundary-less Gym events. Canonical projections validate kind and boundary
    on both failed and recovered reports, independently of the legacy outcome envelope.

    Readers count each failure once by ``event_id``. Outcome policy stays with ``EpisodeFailure.terminal`` and
    the collector's existing rules.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1, max_length=128)
    boundary: FailureBoundary
    exception_type: str | None = Field(default=None, max_length=256)
    failure_reason: str = Field(default="", max_length=2000)
    failure_kind: str
    operation: str | None = Field(default=None, max_length=256)
    operation_id: str | None = Field(default=None, max_length=128)
    attempt_id: str | None = Field(default=None, max_length=128)
    stage: FailureStage | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    request_upload_complete: bool | None = Field(default=None, strict=True)
    foreign_error: SdkErrorDetails | None = None
    cause_id: str | None = Field(default=None, max_length=128)

    @field_validator("failure_kind")
    @classmethod
    def _validate_failure_kind(cls, value: str) -> str:
        return require_failure_kind(value)

    @model_serializer(mode="wrap")
    def _serialize_event(self, handler: SerializerFunctionWrapHandler):
        return {key: value for key, value in handler(self).items() if value is not None}


def validate_failure_event_map(events: Mapping[str, FailureEvent]) -> Mapping[str, FailureEvent]:
    """Check that ``events`` is keyed by event id and that every ``cause_id`` resolves in it without a cycle."""
    for key, event in events.items():
        if key != event.event_id:
            raise ValueError(f"failure event {event.event_id!r} is keyed as {key!r}")
    for event in events.values():
        seen = {event.event_id}
        while event.cause_id is not None:
            if event.cause_id not in events:
                raise ValueError(f"cause {event.cause_id!r} is not in the failure events")
            if event.cause_id in seen:
                raise ValueError(f"cause chain repeats event {event.cause_id!r}")
            seen.add(event.cause_id)
            event = events[event.cause_id]
    return events


class FailureReport(BaseModel):
    """The failures one reply forwards: its own failure (``failure_id``, absent on success) and every failure
    observed while serving the request, each with its explicit causes.

    ``omitted`` says that a report on the way here was too large to send, so events are missing.
    ``nemo_gym.failure_transport`` carries it in one HTTP header; Collection saves it beside the outcome.
    """

    model_config = ConfigDict(extra="forbid")

    failure_id: str | None = Field(default=None, max_length=128)
    events: dict[str, FailureEvent] = Field(default_factory=dict)
    omitted: bool = False

    @field_validator("events", mode="before")
    @classmethod
    def _restore_event_ids(cls, value):
        # Map keys are the origin identities; decoding must never mint replacement ids.
        if isinstance(value, Mapping):
            return {
                key: ({"event_id": key, **event} if isinstance(event, Mapping) else event)
                for key, event in value.items()
            }
        return value

    @field_validator("events")
    @classmethod
    def _validate_events(cls, value: dict[str, FailureEvent]) -> dict[str, FailureEvent]:
        return dict(validate_failure_event_map(value))

    @model_validator(mode="after")
    def _failure_id_is_an_event(self) -> Self:
        if self.failure_id is not None and self.failure_id not in self.events:
            raise ValueError(f"failure_id {self.failure_id!r} is not in events")
        return self

    def merged(self, other: "FailureReport") -> "FailureReport":
        """This report with ``other``'s events added; ``failure_id`` stays this report's.

        An event id names one immutable record, so the first copy of an id is kept.
        """
        return FailureReport(
            failure_id=self.failure_id,
            events=other.events | self.events,
            omitted=self.omitted or other.omitted,
        )


class EpisodeFailure(BaseModel):
    """Describe a failure using the same fields on the wire and in saved records.

    An environment server returns this envelope for one failed episode attempt. Collection uses its
    existing ``terminal`` flag and retry budget; ``failure_kind`` describes the failure without deciding
    terminality. Kind and stage are optional because a collector observing a lost reply may not know them.

    The attempt's ``FailureReport`` travels beside this envelope (``BaseEpisodeResponse.failure_report``), so
    records that nest ``EpisodeFailure`` keep their saved format.
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
