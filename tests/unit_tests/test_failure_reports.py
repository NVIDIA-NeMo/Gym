# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure reports: explicit cause chain -> FailureEvent data -> one HTTP header -> rebuilt chain."""

import asyncio
import socket

import aiohttp
import pytest
from aiohttp import web

from nemo_gym.agent_utils.supervisor_client import HarnessTerminationError
from nemo_gym.episode_types import EpisodeFailure, FailureBoundary, FailureEvent, FailureReport
from nemo_gym.failure_transport import (
    FailureHeader,
    FailureReportRow,
    RequestFailureLog,
    observe_failure,
    observe_reply,
    with_failure_report,
)
from nemo_gym.judge import JudgeCallFailed, JudgeError
from nemo_gym.sandbox.providers.base import SandboxCredentialsRejected
from nemo_gym.typed_exceptions import (
    BoundaryFailure,
    ForeignCauseIds,
    failure_report_for,
    rebuild_failure,
    sdk_error_details,
)


def _model(reason: str = "HTTP 503 from the model server", **fields) -> BoundaryFailure:
    return BoundaryFailure(FailureEvent(boundary=FailureBoundary.MODEL, failure_reason=reason, **fields))


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
    agent = _chained(BoundaryFailure(FailureEvent(boundary=FailureBoundary.AGENT, failure_reason="run failed")), model)

    ids = ForeignCauseIds()
    report = failure_report_for(agent, foreign_ids=ids)

    own = report.events[report.failure_id]
    assert own.event_id == agent.event.event_id and own.boundary == "agent"
    model_event = report.events[own.cause_id]
    assert model_event.event_id == model.event.event_id and model_event.boundary == "model"
    untyped = report.events[model_event.cause_id]
    assert (untyped.boundary, untyped.exception_type, untyped.http_status, untyped.failure_reason) == (
        None,
        f"{__name__}._SdkError",
        401,
        "",
    )
    assert "secret" not in report.model_dump_json()
    assert failure_report_for(agent, foreign_ids=ids) == report  # the same owner gives the same ids


def test_only_explicit_causes_are_linked_and_a_chain_without_a_boundary_reports_nothing() -> None:
    try:
        try:
            raise ValueError("handled first")
        except ValueError:
            raise _model()
    except BoundaryFailure as raised:
        failure, report = raised, failure_report_for(raised)
    assert list(report.events) == [failure.event.event_id] and report.events[failure.event.event_id].cause_id is None
    assert failure_report_for(_chained(RuntimeError("x"), KeyError("y"))) == FailureReport()


def test_a_rebuilt_chain_reports_the_same_events() -> None:
    chain = _chained(_model(), _chained(_model("first"), TimeoutError()))
    report = failure_report_for(chain)

    rebuilt = rebuild_failure(report)

    assert isinstance(rebuilt, BoundaryFailure) and type(rebuilt.__cause__) is BoundaryFailure
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
        event = FailureEvent(boundary=FailureBoundary.MODEL, failure_reason="x" * 2000)
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
        event = FailureEvent(boundary=FailureBoundary.MODEL, failure_reason="x" * 2000)
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
        failed = log.reply_report(_chained(BoundaryFailure(FailureEvent(boundary=FailureBoundary.AGENT)), error))
    own = failed.events[failed.failure_id]
    http = failed.events[own.cause_id]
    assert http.http_status == 503 and http.cause_id == last.event.event_id
    assert len(failed.events) == 4  # agent, HTTP error, last attempt and the earlier attempt as an observation


def test_judge_failed_is_its_own_kind_above_the_typed_cause() -> None:
    cause = _model(failure_kind="model_request_error", http_status=401)
    judge = _chained(JudgeCallFailed("judge call failed: unauthorized", operation="judge.request"), cause)

    report = failure_report_for(judge)

    assert isinstance(judge, JudgeError)
    own = report.events[report.failure_id]
    assert (own.boundary, own.failure_kind, own.stage) == ("resources_server", "judge_failed", "verification")
    assert report.events[own.cause_id].failure_kind == "model_request_error"


@pytest.mark.parametrize(
    ("status", "kind"), [(401, "provider_credentials_rejected"), (403, "provider_credentials_rejected")]
)
def test_sandbox_credentials_are_rejected_only_by_status(status: int, kind: str) -> None:
    sdk = _SdkError("Authentication credentials are invalid.", status_code=status)
    rejected = SandboxCredentialsRejected.from_provider_error(
        sdk, boundary=FailureBoundary.ENVIRONMENT, operation="sandbox.setup", sdk_message=sdk.args[0]
    )
    assert rejected.__cause__ is sdk and rejected.event.failure_kind == kind
    assert rejected.event.foreign_error.message == "Authentication credentials are invalid."
    assert "secret" not in rejected.event.model_dump_json()


@pytest.mark.parametrize("status", [500, None])
def test_auth_looking_text_without_a_401_or_403_is_not_a_credentials_failure(status: int | None) -> None:
    sdk = _SdkError("Authentication credentials are invalid.", status_code=status)
    assert (
        SandboxCredentialsRejected.from_provider_error(sdk, boundary=FailureBoundary.ENVIRONMENT, operation="x")
        is None
    )


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
    rejected = SandboxCredentialsRejected.from_provider_error(
        sdk, boundary=FailureBoundary.ENVIRONMENT, operation="sandbox.setup", stage="seed"
    )
    outer = _chained(HTTPException(503, "Could not set up sandbox"), rejected)
    episode = _chained(
        BoundaryFailure(FailureEvent(boundary=FailureBoundary.ENVIRONMENT, failure_reason="seed failed")), outer
    )

    report = failure_report_for(episode)

    statuses = [event.http_status for event in report.events.values()]
    assert statuses == [None, 503, 401, 401]
    assert [event.boundary for event in report.events.values()] == ["environment", None, "environment", None]


def test_an_episode_failure_keeps_its_own_classification_when_attaching_a_report() -> None:
    error = _chained(_model(failure_kind="model_request_error"), TimeoutError())
    failure = with_failure_report(EpisodeFailure(failure_reason="Episode timed out", terminal=False), error)
    assert (failure.failure_reason, failure.terminal, failure.failure_kind) == (
        "Episode timed out",
        False,
        None,
    )
    assert failure.failure_events[error.event.event_id].failure_kind == "model_request_error"
    assert EpisodeFailure.model_validate_json(failure.model_dump_json()) == failure
    plain = EpisodeFailure(failure_reason="x", terminal=True)
    assert with_failure_report(plain, KeyError("no boundary")) is plain


def test_collection_rows_save_and_merge_reports_without_replacing_the_outcome() -> None:
    body = failure_report_for(_chained(_model("own"), TimeoutError()))
    header = FailureReport().merged(failure_report_for(_model("recovered")))
    row = {"reward": 1.0, **FailureReportRow.fields(body)}

    merged = FailureReportRow.merged_into(row, header, FailureReport(omitted=True))

    assert merged["reward"] == 1.0 and merged[FailureReportRow.FAILURE_ID] == body.failure_id
    assert len(merged[FailureReportRow.EVENTS]) == 3 and merged[FailureReportRow.OMITTED] is True
    broken = FailureReportRow.merged_into({FailureReportRow.EVENTS: {"a": {"event_id": "b"}}})
    assert broken == {FailureReportRow.OMITTED: True}


def test_a_harness_termination_is_the_harness_boundary_and_keeps_its_runtime_error_text() -> None:
    error = HarnessTerminationError(harness="codex", receipt={"cleanup_confirmed": False, "error": "pid 41"})
    assert isinstance(error, RuntimeError) and str(error).endswith("pid 41")
    assert error.event.boundary == "harness" and "pid 41" not in error.event.failure_reason


def test_one_foreign_exception_keeps_one_event_id_before_and_after_wrapping() -> None:
    model = _model(http_status=503)
    foreign = _chained(RuntimeError("agent step failed"), model)
    with RequestFailureLog.capture() as log:
        observe_failure(foreign)
        wrapped = _chained(BoundaryFailure(FailureEvent(boundary=FailureBoundary.AGENT)), foreign)
        reply = log.reply_report(wrapped)
    foreign_ids = [event_id for event_id, event in reply.events.items() if event.exception_type == "RuntimeError"]
    assert len(foreign_ids) == 1 and len(reply.events) == 3
    assert reply.events[reply.failure_id].cause_id == foreign_ids[0]
    assert reply.events[foreign_ids[0]].cause_id == model.event.event_id


def test_one_sdk_error_under_two_parents_is_one_event() -> None:
    sdk = _SdkError("refused", status_code=401)
    first = _chained(_model("first"), sdk)
    second = _chained(_model("second"), sdk)
    with RequestFailureLog.capture() as log:
        observe_failure(first)
        observe_failure(second)
    sdk_events = [event for event in log.report.events.values() if event.boundary is None]
    assert len(sdk_events) == 1 and len(log.report.events) == 3
    assert {event.cause_id for event in log.report.events.values() if event.boundary} == {sdk_events[0].event_id}


def test_boundary_failures_pickle_with_their_event_args_and_existing_types() -> None:
    import pickle

    harness = HarnessTerminationError(harness="codex", receipt={"cleanup_confirmed": False, "error": "pid 41"})
    judge = JudgeCallFailed("judge call failed", operation="judge.request")
    for error in (harness, judge, _model(), rebuild_failure(failure_report_for(_model()))):
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is type(error) and str(restored) == str(error) and restored.event == error.event
    assert pickle.loads(pickle.dumps(harness)).receipt == harness.receipt
    assert isinstance(pickle.loads(pickle.dumps(judge)), JudgeError)


@pytest.mark.parametrize("status", [401, 500, None])
async def test_selected_judge_details_and_http_error_identity_survive_observation_and_forwarding(status) -> None:
    import pickle
    from types import SimpleNamespace

    from aiohttp.test_utils import TestServer

    from nemo_gym.judge import _judge_error
    from nemo_gym.server_utils import raise_for_status
    from nemo_gym.typed_exceptions import AgentRouteFailure

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
            agent = AgentRouteFailure(error, operation="agent.run")
            agent.__cause__ = error
            forwarded = log.reply_report(agent)

    http_events = [event for event in forwarded.events.values() if event.http_status == 503]
    assert len(http_events) == 1
    assert forwarded.events[agent.event.event_id].cause_id == http_events[0].event_id
    assert http_events[0].cause_id == origin.event.event_id
    assert FailureHeader.decode(FailureHeader.encode(forwarded)) == forwarded
