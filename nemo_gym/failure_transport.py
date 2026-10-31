# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Carry ``FailureReport`` data between Gym servers and into Collection's rows.

Flow for one request a Gym server serves:

1. The server middleware opens a ``RequestFailureLog`` for the request.
2. While the request runs, ``observe_failure`` adds failures that a retry may still recover from (a model
   attempt), and ``observe_reply`` adds the report each reply from another Gym server forwarded.
3. When the request ends, the middleware merges the request's own failure (if it raised) with the log and
   sends the result in the ``FailureHeader``.
4. The caller's ``raise_boundary_failure`` raises the rebuilt failure with separate HTTP details;
   ``raise_for_status`` keeps the HTTP exception compatibility path. Collection saves the report with
   ``FailureReportRow``.
"""

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import ValidationError

from nemo_gym.server_runtime.episode_types import FailureReport
from nemo_gym.typed_exceptions import failure_report_for


LOGGER = logging.getLogger(__name__)


class FailureHeader:
    """Encode a ``FailureReport`` into one reply header and decode it on the other side.

    aiohttp's client rejects a reply whose header line (name and value) is over 8190 bytes, and the caller then
    sees an unreadable error instead of the real status and body. A report whose JSON is over ``MAX_BYTES`` is
    therefore not sent: the header says ``{"omitted": true}``, and every hop above keeps that flag.
    """

    NAME: ClassVar[str] = "X-NG-Failure-Report"
    MAX_BYTES: ClassVar[int] = 4096

    @classmethod
    def encode(cls, report: FailureReport) -> dict[str, str]:
        """The reply headers for ``report``; none when it holds nothing."""
        if not report.events and not report.omitted:
            return {}
        value = json.dumps(report.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":"))
        if len(value) > cls.MAX_BYTES:
            LOGGER.warning("Failure report of %d bytes not sent; %d events omitted", len(value), len(report.events))
            value = json.dumps({"omitted": True})
        return {cls.NAME: value}

    @classmethod
    def decode(cls, headers: Mapping[str, str] | None) -> FailureReport | None:
        """The report in ``headers``; ``None`` without one. An invalid value is reported as omitted."""
        value = headers.get(cls.NAME) if headers is not None else None
        if value is None:
            return None
        try:
            return FailureReport.model_validate_json(value)
        except ValidationError:
            LOGGER.warning("Invalid %s header; its failure events are omitted", cls.NAME)
            return FailureReport(omitted=True)


@dataclass
class RequestFailureLog:
    """The failures observed while one server request is served, other than the request's own failure.

    The server middleware opens one per request with ``capture``; code serving the request reaches it through
    ``current``, so a retry loop deep in a client can record an attempt without threading it through calls.
    """

    report: FailureReport = field(default_factory=FailureReport)

    _current: ClassVar[ContextVar["RequestFailureLog | None"]] = ContextVar("ng_request_failure_log", default=None)

    @classmethod
    @contextmanager
    def capture(cls) -> Iterator["RequestFailureLog"]:
        """Make a new log current for the request being served."""
        log = cls()
        token = cls._current.set(log)
        try:
            yield log
        finally:
            cls._current.reset(token)

    @classmethod
    def current(cls) -> "RequestFailureLog | None":
        """The log of the request being served, or ``None`` outside a Gym server request."""
        return cls._current.get()

    def add(self, report: FailureReport) -> None:
        """Add ``report``'s events as observations; the log never has a ``failure_id`` of its own."""
        self.report = self.report.merged(report.model_copy(update={"failure_id": None}))

    def reply_report(self, error: BaseException | None) -> FailureReport:
        """The report for this request's reply: its own failure (``error``, ``None`` on success) and the log."""
        return failure_report_for(error).merged(self.report)


def report_for_current_request(error: BaseException | None) -> FailureReport:
    """Project the explicit Gym cause chain, preserving its origin IDs."""
    return failure_report_for(error)


def observe_failure(error: BaseException) -> None:
    """Record ``error`` for the request being served, for example a model attempt a retry may replace."""
    log = RequestFailureLog.current()
    if log is not None:
        log.add(failure_report_for(error))


def observe_reply(headers: Mapping[str, str] | None) -> None:
    """Record the report a reply from another Gym server forwarded, success or failure."""
    log = RequestFailureLog.current()
    report = FailureHeader.decode(headers)
    if log is not None and report is not None:
        log.add(report)


def reply_failure_report(error: BaseException) -> FailureReport | None:
    """``error``'s explicit cause chain for the reply beside its ``EpisodeFailure``; ``None`` without events.

    The producer keeps its kind, reason, terminality and stage on the failure; a cause's kind stays on its event.
    """
    report = report_for_current_request(error)
    return report if report.events or report.omitted else None


class FailureReportRow:
    """The row keys under which Collection saves a ``FailureReport`` beside a rollout's outcome."""

    EVENTS: ClassVar[str] = "_ng_failure_events"
    FAILURE_ID: ClassVar[str] = "_ng_failure_id"
    OMITTED: ClassVar[str] = "_ng_failure_events_omitted"
    KEYS: ClassVar[tuple[str, ...]] = (EVENTS, FAILURE_ID, OMITTED)

    @classmethod
    def fields(cls, report: FailureReport) -> dict[str, Any]:
        """``report`` as row fields; none for an empty report."""
        fields: dict[str, Any] = {}
        if report.events:
            fields[cls.EVENTS] = {event_id: event.model_dump(mode="json") for event_id, event in report.events.items()}
        if report.failure_id is not None:
            fields[cls.FAILURE_ID] = report.failure_id
        if report.omitted:
            fields[cls.OMITTED] = True
        return fields

    @classmethod
    def report(cls, row: Mapping[str, Any]) -> FailureReport:
        """The report saved in ``row``; an invalid one is reported as omitted, never raised."""
        try:
            return FailureReport(
                failure_id=row.get(cls.FAILURE_ID),
                events=row.get(cls.EVENTS) or {},
                omitted=bool(row.get(cls.OMITTED)),
            )
        except ValidationError:
            LOGGER.warning("Invalid saved failure report; its failure events are omitted")
            return FailureReport(omitted=True)

    @classmethod
    def merged_into(cls, row: dict[str, Any], *reports: FailureReport) -> dict[str, Any]:
        """``row`` with ``reports`` merged into its saved report; ``row``'s own ``failure_id`` wins."""
        report = cls.report(row)
        for extra in reports:
            report = report.merged(extra)
            if report.failure_id is None and extra.failure_id is not None:
                report = report.model_copy(update={"failure_id": extra.failure_id})
        for key in cls.KEYS:
            row.pop(key, None)
        row.update(cls.fields(report))
        return row
