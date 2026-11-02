# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The exception a failure boundary raises, and the conversion of its explicit cause chain to report data.

Flow: a boundary raises ``NgFailureEvent(...) from cause``. ``failure_report_for`` turns that
chain into a ``FailureReport`` (one ``FailureEvent`` per exception, linked by ``cause_id``).
``nemo_gym.failure_transport`` sends the report with the HTTP reply; the receiver rebuilds the chain with
``rebuild_failure`` and Collection saves it. Nothing here chooses retries, terminality or scores.
"""

from __future__ import annotations

import copyreg
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from aiohttp import ClientResponseError

from nemo_gym.core.failure_kinds import AGENT_ROUTE_FAILED, FailureStage, require_failure_kind
from nemo_gym.exception_inspection import bounded_text, error_to_http_status, explicit_causes, qualified_type_name
from nemo_gym.server_runtime.episode_types import FailureBoundary, FailureEvent, FailureReport, SdkErrorDetails


_EVENT_FIELDS = (
    "event_id",
    "boundary",
    "exception_type",
    "failure_reason",
    "failure_kind",
    "operation",
    "operation_id",
    "attempt_id",
    "stage",
    "http_status",
    "request_upload_complete",
    "foreign_error",
)


class NgFailureEvent(Exception):
    """One Gym failure: the exception a boundary raises and the data it serializes, in one object.

    The raiser names its own ``boundary`` and a registered ``failure_kind``; both are validated here, and
    ``event_id`` is fixed here and travels unchanged. Raise it ``from`` the error it reports. ``to_record``
    derives each ``cause_id`` from ``__cause__``; ``from_record`` rebuilds the chain with the same ids and
    boundaries and never imports a type from the payload (the original type name stays data).

    Pass ``message_args`` only when ``str()`` must differ from ``failure_reason``.
    """

    http_error: ClientResponseError | None = None
    """The HTTP response that carried a remote failure, separate from its original causal chain."""

    def __init__(
        self,
        *,
        boundary: FailureBoundary | str,
        failure_kind: str,
        failure_reason: str,
        http_status: int | None = None,
        request_upload_complete: bool | None = None,
        foreign_error: SdkErrorDetails | None = None,
        event_id: str | None = None,
        operation: str | None = None,
        operation_id: str | None = None,
        attempt_id: str | None = None,
        stage: FailureStage | None = None,
        exception_type: str | None = None,
        message_args: tuple[object, ...] = (),
    ) -> None:
        self._set_fields(
            boundary=FailureBoundary(boundary),
            failure_kind=require_failure_kind(failure_kind),
            failure_reason=failure_reason,
            http_status=http_status,
            request_upload_complete=request_upload_complete,
            foreign_error=foreign_error,
            event_id=event_id,
            operation=operation,
            operation_id=operation_id,
            attempt_id=attempt_id,
            stage=stage,
            exception_type=exception_type,
            message_args=message_args,
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("NgFailureEvent has no subclasses; select failures by boundary, failure_kind and operation")

    def _set_fields(self, *, message_args: tuple[object, ...] = (), **fields: object) -> None:
        fields["boundary"] = FailureBoundary(fields["boundary"])
        fields["failure_kind"] = require_failure_kind(fields.get("failure_kind"))
        if fields.get("event_id") is None:
            fields["event_id"] = uuid.uuid4().hex
        # Validate bounds once with the saved-record schema; the fields stay on this exception.
        validated = FailureEvent(**fields)
        for name in _EVENT_FIELDS:
            setattr(self, name, getattr(validated, name))
        Exception.__init__(self, *(message_args or (validated.failure_reason,)))

    @property
    def event(self) -> FailureEvent:
        """This failure's fields as its saved event, without ``cause_id`` (``to_record`` adds it)."""
        return FailureEvent(**{name: getattr(self, name) for name in _EVENT_FIELDS})

    def to_record(self, *, unravel_inner: bool = True) -> dict[str, object]:
        """``failure_id`` and ``failure_events`` for this failure.

        By default the map holds this failure and every explicit cause once, linked by ``cause_id``. With
        ``unravel_inner=False`` it holds only this failure and its ``cause_id``; combine it with the cause's
        records before transport or storage.
        """
        report = failure_report_for(self)
        if not unravel_inner:
            own = report.events[self.event_id]
            return {"failure_id": self.event_id, "failure_events": {self.event_id: own.model_dump(mode="json")}}
        return {
            "failure_id": report.failure_id,
            "failure_events": {event_id: event.model_dump(mode="json") for event_id, event in report.events.items()},
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "NgFailureEvent":
        """The failure ``record["failure_id"]`` names, with its causes as ``__cause__`` and the same ids."""
        report = FailureReport(failure_id=record.get("failure_id"), events=record.get("failure_events") or {})
        rebuilt = rebuild_failure(report)
        if rebuilt is None:
            raise ValueError("a failure record needs a failure_id")
        return rebuilt

    def __reduce__(self) -> tuple:
        # Project SDK facts before replacing foreign exceptions with the canonical Gym cause chain.
        # SDK request objects never cross the process interface.
        report = failure_report_for(self)
        own = report.events[self.event_id]
        cause = rebuild_failure(report.model_copy(update={"failure_id": own.cause_id}))
        state = self.__dict__ | {name: getattr(own, name) for name in _EVENT_FIELDS}
        return copyreg.__newobj__, (type(self), *self.args), state | {"__cause__": cause}


def agent_route_failure(cause: Exception, *, operation: str, stage: FailureStage | None = None) -> NgFailureEvent:
    """The agent boundary's failure for route ``operation``; raise it ``from cause``.

    It names the cause's type, not its text, and takes no kind from it. ``route_reply_error`` selects the cause
    as the HTTP reply.
    """
    return NgFailureEvent(
        boundary=FailureBoundary.AGENT,
        failure_kind=AGENT_ROUTE_FAILED,
        failure_reason=f"Agent operation {operation} failed: {type(cause).__name__}",
        operation=operation,
        stage=stage,
    )


def route_reply_error(error: BaseException) -> BaseException:
    """The error whose HTTP reply a route sends: an agent route failure answers with its cause's reply."""
    if (
        isinstance(error, NgFailureEvent)
        and error.boundary == FailureBoundary.AGENT
        and error.failure_kind == AGENT_ROUTE_FAILED
        and error.__cause__ is not None
    ):
        return error.__cause__
    return error


def find_boundary_failure(error: BaseException | None) -> NgFailureEvent | None:
    """The outermost failure in ``error``'s explicit cause chain that names a boundary, or ``None``."""
    if error is None:
        return None
    for current in explicit_causes(error):
        if isinstance(current, NgFailureEvent) and current.boundary is not None:
            return current
    return None


def failure_report_for(error: BaseException | None) -> FailureReport:
    """``error``'s Gym failures and their explicit causes as a ``FailureReport``.

    Only ``NgFailureEvent``s become events, each under its own id. An SDK or builtin exception in the chain is
    not an event: its type, bounded SDK-selected message and HTTP status become the
    ``foreign_error`` of the Gym failure directly above it, unless that failure already selected one. A Gym
    failure below a foreign intermediate stays linked by ``cause_id``. ``failure_id`` is the outermost Gym
    failure.
    """
    if error is None:
        return FailureReport()
    chain = list(explicit_causes(error))
    events: dict[str, FailureEvent] = {}
    previous: NgFailureEvent | None = None
    previous_event: FailureEvent | None = None
    for current in chain:
        if isinstance(current, NgFailureEvent):
            if current.event_id in events:
                break
            if previous_event is not None:
                events[previous_event.event_id] = previous_event.model_copy(update={"cause_id": current.event_id})
            previous, previous_event = current, current.event
            events[current.event_id] = previous_event
        elif previous_event is not None and previous_event.foreign_error is None and previous.__cause__ is current:
            previous_event = previous_event.model_copy(update={"foreign_error": sdk_error_details(current)})
            events[previous_event.event_id] = previous_event
    if not events:
        return FailureReport()
    return FailureReport(failure_id=next(iter(events)), events=events)


def rebuild_failure(report: FailureReport | None) -> NgFailureEvent | None:
    """The failure ``report.failure_id`` names, rebuilt with its causes as ``__cause__``; else ``None``.

    Every event becomes a ``NgFailureEvent`` with the same event, so serializing it again gives the same ids.
    """
    if report is None or report.failure_id is None:
        return None
    root: NgFailureEvent | None = None
    previous: NgFailureEvent | None = None
    event_id: str | None = report.failure_id
    while event_id is not None:
        event = report.events[event_id]
        # Strict: the wire names a boundary and a registered kind; no type is selected from the payload.
        failure = NgFailureEvent(**{name: getattr(event, name) for name in _EVENT_FIELDS})
        if previous is None:
            root = failure
        else:
            previous.__cause__ = failure
        previous, event_id = failure, event.cause_id
    return root


def sdk_error_details(error: BaseException, *, message: str | None = None) -> SdkErrorDetails:
    """The type, message and HTTP status of a third-party SDK error, without its request or body.

    ``message`` is the text the adapter selected for this SDK; by default the SDK's own ``message`` field. A
    missing or non-string message gives an empty one, so it never hides the original failure. Long text keeps
    a visible truncation marker.
    """
    try:
        text = getattr(error, "message", "") if message is None else message
    except Exception:
        text = ""
    return SdkErrorDetails(
        exception_type=bounded_text(qualified_type_name(error), 256),
        message=bounded_text(text, 2000) if isinstance(text, str) else "",
        http_status=error_to_http_status(error),
    )
