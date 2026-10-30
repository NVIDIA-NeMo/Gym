# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure reports: explicit cause chain -> FailureEvent data -> one HTTP header -> rebuilt chain."""

import asyncio
import json
import socket

import aiohttp
import pytest
from aiohttp import web
from pydantic import ValidationError

from nemo_gym.agent_utils.supervisor_client import harness_termination_failure
from nemo_gym.core.episode import EpisodeId, TaskId
from nemo_gym.failure_transport import (
    FailureHeader,
    FailureReportRow,
    RequestFailureLog,
    observe_failure,
    observe_reply,
    reply_failure_report,
)
from nemo_gym.rollout_collection import _episode_record
from nemo_gym.sandbox.providers.base import sandbox_credentials_rejected
from nemo_gym.server_runtime.episode_types import (
    BaseEpisodeResponse,
    EpisodeFailure,
    FailureBoundary,
    FailureEvent,
    FailureReport,
)
from nemo_gym.typed_exceptions import (
    NgFailureEvent,
    failure_report_for,
    rebuild_failure,
    sdk_error_details,
)
from nemo_gym.verification.judge import judge_call_failed


def _model(reason: str = "HTTP 503 from the model server", **fields) -> NgFailureEvent:
    return NgFailureEvent(
        boundary=FailureBoundary.MODEL, failure_reason=reason, **({"failure_kind": "test:failure"} | fields)
    )


def _chained(outer: BaseException, cause: BaseException) -> BaseException:
    try:
        raise outer from cause
    except BaseException as error:
        return error


class _SdkError(Exception):
    """Shaped like OpenSandbox 0.1.15 SandboxException: the message is only in args[0]."""

    def __init__(self, message: str, *, status_code: int | None, request: str = "POST /sandboxes token=secret"):
        super().__init__(message)
        self.status_code = status_code
        self.request = request

    def __str__(self) -> str:
        return f"{self.args[0]} (request: {self.request})"


def test_a_chain_keeps_origin_ids_and_records_untyped_causes_by_type_only() -> None:
    sdk = _SdkError("refused", status_code=401)
    model = _chained(_model(), sdk)
    agent = _chained(
        NgFailureEvent(failure_kind="test:failure", boundary=FailureBoundary.AGENT, failure_reason="run failed"), model
    )

    report = failure_report_for(agent)

    own = report.events[report.failure_id]
    assert own.event_id == agent.event.event_id and own.boundary == "agent"
    model_event = report.events[own.cause_id]
    assert model_event.event_id == model.event.event_id and model_event.boundary == "model"
    assert model_event.cause_id is None
    assert model_event.foreign_error.exception_type == f"{__name__}._SdkError"
    assert model_event.foreign_error.http_status == 401
    assert model_event.foreign_error.message == ""
    assert len(report.events) == 2
    assert "secret" not in report.model_dump_json()
    assert failure_report_for(agent) == report  # origin Gym IDs stay stable


def test_only_explicit_causes_are_linked_and_a_chain_without_a_boundary_reports_nothing() -> None:
    try:
        try:
            raise ValueError("handled first")
        except ValueError:
            raise _model()
    except NgFailureEvent as raised:
        failure, report = raised, failure_report_for(raised)
    assert list(report.events) == [failure.event.event_id] and report.events[failure.event.event_id].cause_id is None
    assert failure_report_for(_chained(RuntimeError("x"), KeyError("y"))) == FailureReport()


def test_a_rebuilt_chain_reports_the_same_events() -> None:
    chain = _chained(_model(), _chained(_model("first"), TimeoutError()))
    report = failure_report_for(chain)

    rebuilt = rebuild_failure(report)

    assert isinstance(rebuilt, NgFailureEvent) and type(rebuilt.__cause__) is NgFailureEvent
    assert failure_report_for(rebuilt) == report


def test_merging_keeps_one_copy_per_event_id_and_the_first_reports_failure_id() -> None:
    first = failure_report_for(_model())
    second = failure_report_for(_chained(_model("outer"), rebuild_failure(first)))

    merged = second.merged(first)

    assert merged.failure_id == second.failure_id
    assert set(merged.events) == set(second.events) and len(merged.events) == 2


def test_the_header_codec_omits_an_oversized_report_and_keeps_the_flag_upward() -> None:
    small = failure_report_for(_model())
    assert FailureHeader.decode(FailureHeader.encode(small)) == small
    assert FailureHeader.encode(FailureReport()) == {}

    events: dict[str, FailureEvent] = {}
    for _ in range(4):
        event = FailureEvent(failure_kind="test:failure", boundary=FailureBoundary.MODEL, failure_reason="x" * 2000)
        events[event.event_id] = event
    oversized = FailureHeader.encode(FailureReport(events=events))
    assert FailureHeader.decode(oversized) == FailureReport(omitted=True)

    upward = FailureReport().merged(FailureHeader.decode(oversized))
    assert upward.omitted and not upward.events
    assert FailureHeader.decode({FailureHeader.NAME: "not json"}) == FailureReport(omitted=True)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def _status_over_a_real_socket(headers: dict[str, str]) -> int | str:
    async def handler(_: web.Request) -> web.Response:
        return web.json_response({"detail": "inner"}, status=503, headers=headers)

    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        async with aiohttp.ClientSession() as client:
            async with client.get(f"http://127.0.0.1:{port}/") as reply:
                return reply.status
    except aiohttp.ClientResponseError as error:
        return type(error).__name__
    finally:
        await runner.cleanup()


def test_aiohttp_rejects_a_large_header_and_the_encoded_report_keeps_the_real_status() -> None:
    events: dict[str, FailureEvent] = {}
    for _ in range(6):
        event = FailureEvent(failure_kind="test:failure", boundary=FailureBoundary.MODEL, failure_reason="x" * 2000)
        events[event.event_id] = event
    raw = FailureReport(events=events).model_dump_json()

    assert asyncio.run(_status_over_a_real_socket({FailureHeader.NAME: raw})) != 503
    assert asyncio.run(_status_over_a_real_socket(FailureHeader.encode(FailureReport(events=events)))) == 503


def test_the_request_log_forwards_a_recovered_attempt_without_a_failure_id() -> None:
    attempt = _model(operation_id="call", attempt_id="call-1", http_status=503)
    with RequestFailureLog.capture() as log:
        observe_failure(attempt)
        observe_failure(attempt)  # the same observation twice is one event
        observe_reply(FailureHeader.encode(failure_report_for(_model("relayed"))))
        success = log.reply_report(None)
    assert success.failure_id is None and len(success.events) == 2
    assert success.events[attempt.event.event_id].http_status == 503
    assert RequestFailureLog.current() is None
    observe_failure(attempt)  # outside a request: nothing to record, nothing raised


def test_an_exhausted_request_links_its_last_observed_attempt() -> None:
    last = _model(operation_id="call", attempt_id="call-3", http_status=503)
    with RequestFailureLog.capture() as log:
        observe_failure(_model(operation_id="call", attempt_id="call-2", http_status=503))
        observe_failure(last)
        error = _chained(aiohttp.ClientResponseError(None, (), status=503), last)
        failed = log.reply_report(
            _chained(
                NgFailureEvent(
                    failure_kind="test:failure", boundary=FailureBoundary.AGENT, failure_reason="test failure"
                ),
                error,
            )
        )
    own = failed.events[failed.failure_id]
    http = failed.events[own.cause_id]
    assert own.foreign_error.http_status == 503
    assert http.event_id == last.event_id and http.cause_id is None
    assert len(failed.events) == 3  # agent, HTTP error, last attempt and the earlier attempt as an observation


def test_judge_failed_is_its_own_kind_above_the_typed_cause() -> None:
    cause = _model(failure_kind="model_request_error", http_status=401)
    judge = _chained(judge_call_failed("judge call failed: unauthorized", operation="judge.request"), cause)

    report = failure_report_for(judge)

    assert type(judge) is NgFailureEvent
    own = report.events[report.failure_id]
    assert (own.boundary, own.failure_kind, own.stage) == ("resources_server", "judge_failed", "verification")
    assert report.events[own.cause_id].failure_kind == "model_request_error"


@pytest.mark.parametrize(
    ("status", "kind"), [(401, "provider_credentials_rejected"), (403, "provider_credentials_rejected")]
)
def test_sandbox_credentials_are_rejected_only_by_status(status: int, kind: str) -> None:
    sdk = _SdkError("Authentication credentials are invalid.", status_code=status)
    rejected = sandbox_credentials_rejected(
        sdk, boundary=FailureBoundary.ENVIRONMENT, operation="sandbox.setup", sdk_message=sdk.args[0]
    )
    assert rejected.__cause__ is sdk and rejected.event.failure_kind == kind
    assert rejected.event.foreign_error.message == "Authentication credentials are invalid."
    assert "secret" not in rejected.event.model_dump_json()


@pytest.mark.parametrize("status", [500, None])
def test_auth_looking_text_without_a_401_or_403_is_not_a_credentials_failure(status: int | None) -> None:
    sdk = _SdkError("Authentication credentials are invalid.", status_code=status)
    assert sandbox_credentials_rejected(sdk, boundary=FailureBoundary.ENVIRONMENT, operation="x") is None


def test_sdk_details_select_only_the_given_text_and_bound_it_visibly() -> None:
    sdk = _SdkError("y" * 3000, status_code=401)
    details = sdk_error_details(sdk, message=sdk.args[0])
    assert len(details.message) == 2000 and details.message.endswith(" [truncated]")
    assert sdk_error_details(sdk).message == ""  # no ``message`` field and none selected: blank, never str()
    sdk.message = 7
    assert sdk_error_details(sdk).message == ""  # a non-string diagnostic does not mask the failure


def test_outer_503_keeps_the_inner_sdk_401_below_the_credentials_failure() -> None:
    from fastapi import HTTPException

    sdk = _SdkError("denied", status_code=401)
    rejected = sandbox_credentials_rejected(
        sdk, boundary=FailureBoundary.ENVIRONMENT, operation="sandbox.setup", stage="seed"
    )
    outer = _chained(HTTPException(503, "Could not set up sandbox"), rejected)
    episode = _chained(
        NgFailureEvent(
            failure_kind="test:failure", boundary=FailureBoundary.ENVIRONMENT, failure_reason="seed failed"
        ),
        outer,
    )

    report = failure_report_for(episode)

    statuses = [event.http_status for event in report.events.values()]
    assert statuses == [None, 401]
    assert report.events[episode.event_id].foreign_error.http_status == 503
    assert report.events[rejected.event_id].foreign_error.http_status == 401
    assert [event.boundary for event in report.events.values()] == ["environment", "environment"]


def test_an_episode_failure_keeps_its_own_classification_and_its_report_travels_beside_it() -> None:
    error = _chained(_model(failure_kind="model_request_error"), TimeoutError())
    failure = EpisodeFailure(failure_reason="Episode timed out", terminal=False)
    report = reply_failure_report(error)
    assert report is not None
    assert report.events[error.event.event_id].failure_kind == "model_request_error"
    assert report.events[report.failure_id].foreign_error.exception_type == "TimeoutError"
    reply = BaseEpisodeResponse[dict](
        episode_id=EpisodeId(rollout_id="r", attempt=0),
        task_id=TaskId(taskset="test:source", task_id="t"),
        failure=failure,
        failure_report=report,
    )
    wire = json.loads(reply.model_dump_json())
    # The failure envelope keeps main's saved shape; events stay in the report beside it.
    assert set(wire["failure"]) == {"failure_reason", "terminal"}
    assert FailureReport.model_validate(wire["failure_report"]) == report
    assert reply_failure_report(KeyError("no boundary")) is None


def test_a_version_one_reader_of_episode_failure_refuses_event_maps() -> None:
    """A saved record that nests ``EpisodeFailure`` (#3302 ``RolloutFailureRecord`` v1) keeps that shape."""
    report = reply_failure_report(_chained(_model(failure_kind="model_request_error"), TimeoutError()))
    for extra in (
        {"failure_id": report.failure_id},
        {"failure_events": report.model_dump(mode="json")["events"]},
        {"failure_events_omitted": True},
    ):
        with pytest.raises(ValidationError):
            EpisodeFailure.model_validate({"failure_reason": "x", "terminal": False} | extra)
    assert set(EpisodeFailure.model_json_schema()["properties"]) == {
        "failure_reason",
        "terminal",
        "failure_kind",
        "stage",
    }


def test_collection_rows_save_and_merge_reports_without_replacing_the_outcome() -> None:
    body = failure_report_for(_chained(_model("own"), TimeoutError()))
    header = FailureReport().merged(failure_report_for(_model("recovered")))
    row = {"reward": 1.0, **FailureReportRow.fields(body)}

    merged = FailureReportRow.merged_into(row, header, FailureReport(omitted=True))

    assert merged["reward"] == 1.0 and merged[FailureReportRow.FAILURE_ID] == body.failure_id
    assert len(merged[FailureReportRow.EVENTS]) == 2 and merged[FailureReportRow.OMITTED] is True
    broken = FailureReportRow.merged_into({FailureReportRow.EVENTS: {"a": {"event_id": "b"}}})
    assert broken == {FailureReportRow.OMITTED: True}


def test_a_harness_termination_is_the_harness_boundary_and_keeps_its_runtime_error_text() -> None:
    error = harness_termination_failure(harness="codex", receipt={"cleanup_confirmed": False, "error": "pid 41"})
    assert type(error) is NgFailureEvent and str(error).endswith("pid 41")
    assert error.event.boundary == "harness" and "pid 41" not in error.event.failure_reason


def test_one_foreign_exception_keeps_one_event_id_before_and_after_wrapping() -> None:
    model = _model(http_status=503)
    foreign = _chained(RuntimeError("agent step failed"), model)
    with RequestFailureLog.capture() as log:
        observe_failure(foreign)
        wrapped = _chained(
            NgFailureEvent(failure_kind="test:failure", boundary=FailureBoundary.AGENT, failure_reason="test failure"),
            foreign,
        )
        reply = log.reply_report(wrapped)
    assert len(reply.events) == 2
    own = reply.events[wrapped.event_id]
    assert own.cause_id == model.event_id
    assert own.foreign_error.exception_type == "RuntimeError"
    assert reply.events[model.event_id] == model.event


def test_one_sdk_error_under_two_parents_is_one_event() -> None:
    sdk = _SdkError("refused", status_code=401)
    first = _chained(_model("first"), sdk)
    second = _chained(_model("second"), sdk)
    with RequestFailureLog.capture() as log:
        observe_failure(first)
        observe_failure(second)
    assert len(log.report.events) == 2
    for event in log.report.events.values():
        assert event.cause_id is None
        assert event.foreign_error.exception_type == f"{__name__}._SdkError"
        assert event.foreign_error.http_status == 401


def test_boundary_failures_pickle_with_their_event_args_and_existing_types() -> None:
    import pickle

    harness = harness_termination_failure(harness="codex", receipt={"cleanup_confirmed": False, "error": "pid 41"})
    judge = judge_call_failed("judge call failed", operation="judge.request")
    for error in (harness, judge, _model(), rebuild_failure(failure_report_for(_model()))):
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is type(error) and str(restored) == str(error) and restored.event == error.event
    assert pickle.loads(pickle.dumps(harness)).receipt == harness.receipt
    assert type(pickle.loads(pickle.dumps(judge))) is NgFailureEvent


@pytest.mark.parametrize("status", [401, 500, None])
async def test_selected_judge_details_and_http_error_identity_survive_observation_and_forwarding(status) -> None:
    import pickle
    from types import SimpleNamespace

    from aiohttp.test_utils import TestServer

    from nemo_gym.client.http_client import raise_for_status
    from nemo_gym.typed_exceptions import agent_route_failure
    from nemo_gym.verification.judge import _judge_error

    sdk = aiohttp.ClientResponseError(
        SimpleNamespace(real_url="http://provider.invalid/judge"), (), status=status, message="authentication failed"
    )
    sdk.request_payload = {"secret": "must not be serialized"}
    origin = _judge_error(sdk)
    origin.__cause__ = sdk
    before = failure_report_for(origin)
    details = origin.event.foreign_error
    assert (details.message, details.http_status) == ("authentication failed", status)
    assert "must not be serialized" not in before.model_dump_json()

    async def endpoint(request):
        return web.json_response({"error": "unavailable"}, status=503, headers=FailureHeader.encode(before))

    app = web.Application()
    app.router.add_get("/failure", endpoint)
    async with TestServer(app) as server, aiohttp.ClientSession() as client:
        with RequestFailureLog.capture() as log:
            async with client.get(server.make_url("/failure")) as reply:
                with pytest.raises(aiohttp.ClientResponseError) as caught:
                    await raise_for_status(reply)
            error = caught.value
            assert error.status == 503
            rebuilt = error.__cause__
            assert rebuilt.event.foreign_error == details and rebuilt.event.failure_kind == "judge_failed"
            assert failure_report_for(rebuilt) == before
            assert pickle.loads(pickle.dumps(rebuilt)).event.foreign_error == details
            observe_failure(error)
            observe_failure(error)
            agent = agent_route_failure(error, operation="agent.run")
            agent.__cause__ = error
            forwarded = log.reply_report(agent)

    assert len(forwarded.events) == 2
    own = forwarded.events[agent.event_id]
    assert own.foreign_error.http_status == 503
    assert own.cause_id == origin.event_id
    assert forwarded.events[origin.event_id].foreign_error == details
    assert FailureHeader.decode(FailureHeader.encode(forwarded)) == forwarded


def test_collection_saves_a_body_only_reply_report_on_failed_and_recovered_rows() -> None:
    """No header here: the report beside ``failure`` or ``result`` in the reply body alone reaches the row."""
    failed = reply_failure_report(_chained(_model(failure_kind="model_request_error"), TimeoutError()))
    task_id = {"taskset": "test:source", "task_id": "t"}
    row = _episode_record(
        {
            "task_id": task_id,
            "failure": {"failure_reason": "Episode timed out", "terminal": False},
            "failure_report": failed.model_dump(mode="json"),
        }
    )
    assert FailureReportRow.report(row) == failed
    assert row["_ng_failure_message"] == "Episode timed out"

    recovered = failed.model_copy(update={"failure_id": None})
    row = _episode_record(
        {"task_id": task_id, "result": {"reward": 1.0}, "failure_report": recovered.model_dump(mode="json")}
    )
    assert row["reward"] == 1.0 and FailureReportRow.FAILURE_ID not in row
    assert FailureReportRow.report(row).events == recovered.events


@pytest.mark.parametrize("serializer_name", ["pickle", "cloudpickle"])
def test_process_projection_preserves_generated_sdk_details(serializer_name):
    import importlib

    serializer = importlib.import_module(serializer_name)
    sdk = _SdkError("args contain request details", status_code=401)
    sdk.message = "selected SDK message"
    error = _model()
    error.__cause__ = sdk
    assert error.foreign_error is None
    before = failure_report_for(error)
    restored = serializer.loads(serializer.dumps(error))
    assert failure_report_for(restored) == before
    assert restored.foreign_error.message == "selected SDK message"
    assert restored.foreign_error.http_status == 401
    assert restored.__cause__ is None
    assert "secret" not in failure_report_for(restored).model_dump_json()


@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("invalid", [{"boundary": None}, {"failure_kind": None}, {"failure_kind": "unknown_kind"}])
def test_all_report_decode_paths_reject_invalid_canonical_fields(recovered, invalid):
    failure = _model()
    report = failure_report_for(failure).model_dump(mode="json")
    if recovered:
        report["failure_id"] = None
    report["events"][failure.event_id].update(invalid)
    with pytest.raises(ValidationError):
        FailureReport.model_validate(report)
    assert FailureHeader.decode({FailureHeader.NAME: json.dumps(report)}) == FailureReport(omitted=True)
    row = {FailureReportRow.EVENTS: report["events"]}
    if not recovered:
        row[FailureReportRow.FAILURE_ID] = failure.event_id
    assert FailureReportRow.report(row).omitted
    with pytest.raises((ValidationError, ValueError)):
        NgFailureEvent.from_record({"failure_id": failure.event_id, "failure_events": report["events"]})
