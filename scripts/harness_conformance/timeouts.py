# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timeout checks across the provider, caller, and saved outcome boundaries."""

import json
from collections.abc import Callable

from multidict import CIMultiDict
from pydantic import ValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nemo_gym.client.openai_client import MODEL_RESPONSE_TIMEOUT_REASON
from nemo_gym.core.failure_kinds import MODEL_REQUEST_ERROR
from nemo_gym.core.failure_records import FailureReport, ModelFailureDetails
from nemo_gym.core.failure_transport import FailureHeader, FailureReportRow, StreamFailureReportReader
from nemo_gym.rollouts.harness_capabilities.checks import BehavioralCheck
from nemo_gym.rollouts.harness_capabilities.results import Results


class FailureReportWitness:
    """Observe real reply diagnostics without altering headers, body, or request scope."""

    def __init__(self, app: ASGIApp, record: Callable[[FailureReport], None]) -> None:
        self.app = app
        self.record = record

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        report = None
        stream = StreamFailureReportReader()

        async def witness_reply(message: Message) -> None:
            nonlocal report
            if message["type"] == "http.response.start":
                headers = CIMultiDict((key.decode(), value.decode()) for key, value in message["headers"])
                report = FailureHeader.decode(headers)
            elif message["type"] == "http.response.body":
                stream.feed(message.get("body", b""))
            await send(message)

        await self.app(scope, receive, witness_reply)
        streamed = stream.finish()
        if streamed is not None:
            report = report.merged(streamed) if report is not None else streamed
        if report is not None:
            self.record(report)


def _report(value: object) -> FailureReport | None:
    try:
        return FailureReport.model_validate(value)
    except ValidationError:
        return None


def _one(values: object) -> dict | None:
    return values[0] if isinstance(values, list) and len(values) == 1 and isinstance(values[0], dict) else None


def _model_report(values: object, *, exhausted: bool) -> FailureReport | None:
    """Keep all observed calls, with only the last failed call selecting a terminal cause."""
    if not isinstance(values, list) or not values:
        return None
    reports = [_report(value) for value in values]
    if any(report is None for report in reports):
        return None
    merged = reports[-1] if exhausted else FailureReport()
    for report in reports:
        if any(key in merged.events and merged.events[key] != event for key, event in report.events.items()):
            return None
        merged = merged.merged(report)
    return merged


def _retains(report: FailureReport | None, origin: FailureReport | None) -> bool:
    return bool(
        report is not None
        and origin is not None
        and not report.omitted
        and not origin.omitted
        and origin.events
        and all(report.events.get(key) == event for key, event in origin.events.items())
    )


def _causes(report: FailureReport | None, target: str | None) -> bool:
    if report is None or target is None:
        return False
    current = report.failure_id
    while current is not None:
        if current == target:
            return True
        current = report.events[current].cause_id
    return False


def inspect_model_timeouts(witness: dict | None, record: dict | None, *, exhausted: bool) -> list[dict]:
    """Require real timeouts and retain their events through successful or failed delivery."""
    witness = witness or {}
    attempts = witness.get("timeout_attempts", [])
    model = _model_report(witness.get("model_server_failure_reports"), exhausted=exhausted)
    caller = _report(_one(witness.get("caller_failure_reports")))
    response = _one(witness.get("agent_responses")) or {}
    response_report = _report(response.get("failure_report"))
    saved = FailureReportRow.report(record) if record is not None else None
    wanted_timeouts = (
        sum(isinstance(attempt, dict) and attempt.get("timed_out") is True for attempt in attempts)
        if isinstance(attempts, list)
        else 0
    )
    expected_sequence = [True] * wanted_timeouts + ([] if exhausted else [False])
    events = list(model.events.values()) if model is not None else []
    model_events = [event for event in events if event.boundary == "model"]
    results = Results()

    def check(key: str, passed: bool, reason: str, *, location: str = "witness.json") -> None:
        results.run(
            BehavioralCheck(
                id=f"timeout.{key}",
                tier="P1",
                evidence=("TE-1", "TE-8"),
                location=location,
                reason=reason,
                predicate=lambda: passed,
            )
        )

    check(
        "attempts",
        isinstance(attempts, list)
        and wanted_timeouts > 0
        and all(
            isinstance(attempt, dict)
            and isinstance(attempt.get("request"), dict)
            and bool(attempt["request"])
            and isinstance(attempt.get("request_id"), str)
            and bool(attempt["request_id"])
            and type(attempt.get("timed_out")) is bool
            for attempt in attempts
        )
        # Gym creates X-Request-Id once per logical client call and reuses it
        # across transport retries. The provider observes these independently.
        and len({attempt["request_id"] for attempt in attempts}) == len(attempts)
        and [attempt.get("timed_out") for attempt in attempts] == expected_sequence
        and all(attempt.get("request") == attempts[0].get("request") for attempt in attempts[1:]),
        "provider attempts must have distinct model-call request IDs, retain the request, and show timeout(s) "
        "followed by recovery or only timeouts on failure",
    )
    check(
        "model_events",
        model is not None
        and not model.omitted
        and wanted_timeouts > 0
        and len(model_events) == wanted_timeouts
        and len({event.operation_id for event in model_events}) == wanted_timeouts
        and len({event.attempt_id for event in model_events}) == wanted_timeouts
        and all(
            event.failure_kind == MODEL_REQUEST_ERROR
            and event.operation == "model_request"
            and bool(event.operation_id)
            and bool(event.attempt_id)
            and event.failure_reason == MODEL_RESPONSE_TIMEOUT_REASON
            and isinstance(event.details, ModelFailureDetails)
            and event.details.request_upload_complete is True
            and event.foreign_error is not None
            and event.foreign_error.exception_type.rsplit(".", 1)[-1]
            in {"TimeoutError", "ServerTimeoutError", "SocketTimeoutError"}
            for event in model_events
        ),
        "every timed-out model attempt needs a distinct registered event with confirmed upload and timeout evidence",
    )
    check(
        "caller_events",
        _retains(caller, model),
        "the EnvironmentServer must receive every original model failure unchanged",
    )
    check(
        "saved_events",
        _retains(saved, model),
        "Collection must save every original model failure unchanged without omitted diagnostics",
        location="$._ng_failure_events",
    )
    if exhausted:
        check(
            "terminal_outcome",
            response.get("status") == "failed"
            and bool(response.get("error"))
            and _retains(response_report, model)
            and _causes(response_report, model.failure_id if model else None)
            and _causes(saved, model.failure_id if model else None)
            and not witness.get("verifications")
            and record is not None
            and record.get("evaluation_completed") is not True
            and record.get("reward") in (None, 0.0),
            "exhaustion must reach caller and Collection as failure with its model cause, without grading success",
        )
    else:
        check(
            "recovered_outcome",
            response.get("status") == "completed"
            and not response.get("error")
            and "CONFORMANCE_DONE" in json.dumps(response.get("output"))
            and (
                response.get("failure_report") is None
                or (response_report is not None and not response_report.omitted and response_report.failure_id is None)
            )
            and caller is not None
            and caller.failure_id is None
            and saved is not None
            and saved.failure_id is None
            and record is not None
            and record.get("reward") == 1.0
            and record.get("mask_sample") is False
            and record.get("evaluation_completed") is True,
            "recovery must preserve answer and rewarded outcome with no selected terminal failure",
        )
    return results.dump()
