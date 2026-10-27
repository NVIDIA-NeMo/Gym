# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The exception a failure boundary raises, and the conversion of its explicit cause chain to report data.

Flow: a boundary raises ``BoundaryFailure(FailureEvent(...)) from cause``. ``failure_report_for`` turns that
chain into a ``FailureReport`` (one ``FailureEvent`` per exception, linked by ``cause_id``).
``nemo_gym.failure_transport`` sends the report with the HTTP reply; the receiver rebuilds the chain with
``rebuild_failure`` and Collection saves it. Nothing here chooses retries, terminality or scores.
"""

import copyreg
import uuid
from dataclasses import dataclass, field

from nemo_gym.episode_types import FailureBoundary, FailureEvent, FailureReport, SdkErrorDetails
from nemo_gym.exception_inspection import bounded_text, error_to_http_status, explicit_causes, qualified_type_name
from nemo_gym.failure_kinds import FailureStage


class BoundaryFailure(Exception):
    """A failure observed at one of the six ``FailureBoundary`` places, with its data in ``event``.

    Raise it ``from`` the error it reports, so the explicit cause stays linked. ``event.event_id`` is fixed here
    and travels unchanged. An existing exception type that is itself a boundary's failure (``JudgeCallFailed``,
    ``HarnessTerminationError``) subclasses this next to its old base, so its handlers keep working; pass the
    old ``str()`` text as ``args`` when it must differ from the event's ``failure_reason``.

    A receiver rebuilds a remote failure as this class from its event; the event keeps the original type name
    as data (``exception_type``), and no type is imported from the wire.
    """

    def __init__(self, event: FailureEvent, *args: object) -> None:
        super().__init__(*(args or (event.failure_reason,)))
        if event.exception_type is None and type(self) is not BoundaryFailure:
            event = event.model_copy(update={"exception_type": bounded_text(qualified_type_name(self), 256)})
        self.event = event

    def __reduce__(self) -> tuple:
        # Rebuild with Exception.__new__ and the saved state, so a subclass's own constructor signature
        # (``HarnessTerminationError(*, harness, receipt)``) is not called with ``args``.
        return copyreg.__newobj__, (type(self), *self.args), self.__dict__


@dataclass
class ForeignCauseIds:
    """The event ids given to explicit causes that are not ``BoundaryFailure`` (an SDK or builtin error).

    An id is assigned the first time a cause is captured and reused for that same exception object, so one
    SDK error observed before and after it is wrapped, or under two parents, is one event. The owner bounds
    its lifetime: a request's ``RequestFailureLog`` or a single ``failure_report_for`` call. It keeps a
    reference to each exception (exceptions do not support weak references) and stores nothing on them.
    """

    _ids: dict[int, tuple[BaseException, str]] = field(default_factory=dict)

    def event_id(self, error: BaseException) -> str:
        """The event id of ``error``, assigned now if this owner has not captured it before."""
        entry = self._ids.get(id(error))
        if entry is None or entry[0] is not error:
            entry = (error, uuid.uuid4().hex)
            self._ids[id(error)] = entry
        return entry[1]


class AgentRouteFailure(BoundaryFailure):
    """An agent server route failed; its HTTP reply is the reply its cause would get.

    The route wrapper raises it ``from`` the original error. The server middleware answers with the cause's
    body and status and sends this failure, and its cause, in the failure report. It names the cause's type,
    not its text, and takes no kind from it.
    """

    def __init__(self, cause: Exception, *, operation: str, stage: FailureStage | None = None) -> None:
        super().__init__(
            FailureEvent(
                boundary=FailureBoundary.AGENT,
                failure_reason=f"Agent operation {operation} failed: {type(cause).__name__}",
                operation=operation,
                stage=stage,
            )
        )

    @property
    def reply_error(self) -> BaseException:
        """The error whose HTTP reply this failure's route sends."""
        return self.__cause__ or self


def find_boundary_failure(error: BaseException | None) -> BoundaryFailure | None:
    """The outermost failure in ``error``'s explicit cause chain that names a boundary, or ``None``."""
    if error is None:
        return None
    for current in explicit_causes(error):
        if isinstance(current, BoundaryFailure) and current.event.boundary is not None:
            return current
    return None


def failure_report_for(error: BaseException | None, *, foreign_ids: ForeignCauseIds | None = None) -> FailureReport:
    """``error`` and its explicit causes as a ``FailureReport`` whose ``failure_id`` is ``error``'s event.

    The report is empty when the chain holds no boundary failure. A ``BoundaryFailure`` contributes its own
    event. Any other exception contributes its type and HTTP status only (its text may hold request data),
    under the id ``foreign_ids`` gives it; pass the request's ids so the same exception keeps one id.
    """
    if find_boundary_failure(error) is None:
        return FailureReport()
    foreign_ids = foreign_ids if foreign_ids is not None else ForeignCauseIds()
    events: dict[str, FailureEvent] = {}
    previous: FailureEvent | None = None
    for current in explicit_causes(error):
        if isinstance(current, BoundaryFailure):
            event = current.event
        else:
            event = FailureEvent(
                event_id=foreign_ids.event_id(current),
                exception_type=bounded_text(qualified_type_name(current), 256),
                http_status=error_to_http_status(current),
            )
        if event.event_id in events:
            break
        if previous is not None:
            events[previous.event_id] = previous.model_copy(update={"cause_id": event.event_id})
        events[event.event_id] = event
        previous = event
    return FailureReport(failure_id=next(iter(events)), events=events)


def rebuild_failure(report: FailureReport | None) -> BoundaryFailure | None:
    """The failure ``report.failure_id`` names, rebuilt with its causes as ``__cause__``; else ``None``.

    Every event becomes a ``BoundaryFailure`` with the same event, so serializing it again gives the same ids.
    """
    if report is None or report.failure_id is None:
        return None
    root: BoundaryFailure | None = None
    previous: BoundaryFailure | None = None
    event_id: str | None = report.failure_id
    while event_id is not None:
        event = report.events[event_id]
        failure = BoundaryFailure(event.model_copy(update={"cause_id": None}))
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
    text = getattr(error, "message", "") if message is None else message
    return SdkErrorDetails(
        exception_type=bounded_text(qualified_type_name(error), 256),
        message=bounded_text(text, 2000) if isinstance(text, str) else "",
        http_status=error_to_http_status(error),
    )
