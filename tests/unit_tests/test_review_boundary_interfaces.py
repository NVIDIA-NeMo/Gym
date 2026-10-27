# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime, HTTP and process interfaces preserve boundary identity and local policy."""

import json
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import cloudpickle
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi.testclient import TestClient

import nemo_gym.server_utils as server_utils
from nemo_gym.base_environment_server import HandledEpisodeError, is_retryable_dependency_error
from nemo_gym.episode_types import BaseEpisodeResponse, EpisodeFailure, FailureBoundary, FailureEvent, FailureReport
from nemo_gym.failure_transport import FailureHeader, RequestFailureLog, report_for_current_request
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.typed_exceptions import BoundaryFailure, failure_report_for
from tests.unit_tests.test_environment_server import _environment_server, _EnvironmentServer, _request


class _EvidenceFailure(EpisodeFailure):
    partial_response: str | None = None
    agent_observations: list[str] | None = None
    trajectory: dict[str, str] | None = None


class _EvidenceResponse(BaseEpisodeResponse[str]):
    failure: _EvidenceFailure | None = None


@pytest.mark.parametrize("terminal", [False, True])
def test_an_environment_runtime_failure_is_handled_without_constructing_a_wire_model(terminal):
    provider = RuntimeError("provider disconnected")
    model = BoundaryFailure(FailureEvent(boundary=FailureBoundary.MODEL, http_status=503))
    model.__cause__ = provider
    event = FailureEvent(
        boundary=FailureBoundary.ENVIRONMENT,
        failure_reason="Verification did not complete",
        stage="verification",
        failure_kind="judge_failed",
    )
    error = HandledEpisodeError(
        event,
        terminal=terminal,
        evidence={"partial_response": "partial", "agent_observations": ["turn failed"], "trajectory": {"turn": "one"}},
    )

    class Environment(_EnvironmentServer):
        response_model = _EvidenceResponse

        async def run(self, request, cleanup):
            async def finish_evidence():
                error.evidence["agent_observations"].append("cleanup completed")

            cleanup.register_cleanup("finish evidence", finish_evidence)
            raise error from model

    template = _environment_server()
    server = Environment(config=template.config, server_client=template.server_client)
    with RequestFailureLog.capture(), TestClient(server.setup_webserver()) as client:
        response = client.post("/run", json=_request().model_dump(mode="json"))
    assert response.status_code == 200
    failure = _EvidenceFailure.model_validate(response.json()["failure"])
    report = FailureReport.model_validate(response.json()["failure_report"])
    report = FailureReport.model_validate(response.json()["failure_report"])
    report = FailureReport.model_validate(response.json()["failure_report"])
    assert failure.terminal is terminal and failure.stage == "verification"
    assert failure.failure_kind == "judge_failed" and failure.failure_reason == event.failure_reason
    assert report.failure_id == event.event_id
    assert report.events[event.event_id].cause_id == model.event.event_id
    assert report.events[model.event.event_id].http_status == 503
    assert failure.partial_response == "partial"
    assert failure.agent_observations == ["turn failed", "cleanup completed"]
    assert failure.trajectory == {"turn": "one"}
    assert not hasattr(error, "failure")


@pytest.mark.parametrize("header_kind", ["failure", "malformed", "recovered"])
@pytest.mark.parametrize("status", [422, 503])
async def test_the_typed_http_path_raises_the_remote_failure_and_keeps_http_details_separate(
    monkeypatch, header_kind, status
):
    model = BoundaryFailure(FailureEvent(boundary=FailureBoundary.MODEL, http_status=429))
    remote = BoundaryFailure(FailureEvent(boundary=FailureBoundary.RESOURCES_SERVER, failure_kind="judge_failed"))
    remote.__cause__ = model
    model.__cause__ = ValueError("private provider request")
    report = failure_report_for(remote)
    headers = FailureHeader.encode(report)
    if header_kind == "malformed":
        headers = {FailureHeader.NAME: "not-json"}
    elif header_kind == "recovered":
        headers = FailureHeader.encode(report.model_copy(update={"failure_id": None}))
    hits = 0

    async def endpoint(request):
        nonlocal hits
        hits += 1
        return web.Response(status=status, text="try later", headers=headers)

    app = web.Application()
    app.router.add_get("/verify", endpoint)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: session)
        for typed in [False, True]:
            response = await server_utils.request("GET", str(server.make_url("/verify")), _max_num_tries=1)
            raises = server_utils.raise_boundary_failure if typed else server_utils.raise_for_status
            if typed and header_kind == "failure":
                with pytest.raises(BoundaryFailure) as caught:
                    await raises(response)
                error = caught.value
                assert error.event.boundary is FailureBoundary.RESOURCES_SERVER
                assert failure_report_for(error) == report
                assert error.http_error.status == status and error.http_error.response_content == b"try later"
                assert error.http_error.__cause__ is None
                assert error.__cause__.event.event_id == model.event.event_id
                for serializer in (pickle, cloudpickle):
                    restored = serializer.loads(serializer.dumps(error))
                    assert failure_report_for(restored) == report
                    assert restored.http_error.status == status
                    assert restored.http_error.response_content == b"try later"
                    assert isinstance(restored.http_error.request_info.real_url, str)
                assert is_retryable_dependency_error(error) is (status == 503)
                server_owner = SimpleNamespace(get_session_middleware_key=lambda: "caller")
                typed_reply = server_utils.SimpleServer._failure_reply(server_owner, error)
                http_reply = server_utils.SimpleServer._failure_reply(server_owner, error.http_error)
                assert (typed_reply.status_code, typed_reply.body) == (http_reply.status_code, http_reply.body)
            else:
                with pytest.raises(aiohttp.ClientResponseError) as caught:
                    await raises(response)
                assert caught.value.status == status and caught.value.response_content == b"try later"
                if header_kind == "failure":
                    assert failure_report_for(caught.value.__cause__) == report
    assert hits == 2


@pytest.mark.parametrize("header_kind", ["none", "reason", "malformed", "recovered", "failure"])
async def test_only_a_valid_failed_reply_reference_can_replace_the_current_model_attempt(monkeypatch, header_kind):
    old = FailureEvent(boundary=FailureBoundary.MODEL, http_status=429, failure_reason="earlier attempt")
    final = FailureEvent(boundary=FailureBoundary.MODEL, http_status=503, failure_reason="remote final attempt")
    headers = {}
    if header_kind == "reason":
        headers = {FailureHeader.NAME: json.dumps({"failure_reason": "try later", "terminal": False})}
    elif header_kind == "malformed":
        headers = {FailureHeader.NAME: "{"}
    elif header_kind == "recovered":
        headers = FailureHeader.encode(FailureReport(events={old.event_id: old}))
    elif header_kind == "failure":
        headers = FailureHeader.encode(FailureReport(failure_id=final.event_id, events={final.event_id: final}))
    hits = 0

    async def endpoint(request):
        nonlocal hits
        hits += 1
        return web.Response(status=503, text="try later", headers=headers) if hits == 1 else web.Response(text="ok")

    app = web.Application()
    app.router.add_post("/responses", endpoint)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: session)
        monkeypatch.setattr(server_utils.asyncio, "sleep", AsyncMock())
        client = NeMoGymAsyncOpenAI(
            api_key="test", base_url=str(server.make_url("/")), max_http_attempts=2, max_rate_limit_wait_seconds=0
        )
        with RequestFailureLog.capture() as log:
            response = await client._request(method="POST", url=str(server.make_url("/responses")))
            assert response.status == 200 and await response.text() == "ok"
            report = log.reply_report(None)
    failed_attempts = [
        event
        for event in report.events.values()
        if event.boundary is FailureBoundary.MODEL and event.http_status == 503
    ]
    assert hits == 2 and report.failure_id is None
    assert len(failed_attempts) == 1
    if header_kind == "failure":
        assert failed_attempts[0].event_id == final.event_id
    else:
        assert failed_attempts[0].operation_id and failed_attempts[0].attempt_id
    if header_kind == "recovered":
        assert report.events[old.event_id] == old


class _SdkError(RuntimeError):
    def __init__(self):
        super().__init__("private SDK request")
        self.request = lambda: None  # Third-party request state must not cross the process interface.
        self.__cause__ = ValueError("private inner request")


@pytest.mark.parametrize("serializer", [pickle, cloudpickle], ids=["pickle", "cloudpickle"])
def test_process_serialization_keeps_nested_typed_and_foreign_causes_with_their_captured_ids(serializer):
    model = BoundaryFailure(FailureEvent(boundary=FailureBoundary.MODEL, http_status=429))
    model.__cause__ = _SdkError()
    agent = BoundaryFailure(FailureEvent(boundary=FailureBoundary.AGENT))
    agent.__cause__ = model
    with RequestFailureLog.capture():
        before = report_for_current_request(agent)
        restored = serializer.loads(serializer.dumps(agent))
    after = failure_report_for(restored)
    assert after == before and len(after.events) == 4
    assert restored.event.event_id == agent.event.event_id
    assert restored.__cause__.event.event_id == model.event.event_id
    assert restored.__cause__.event.http_status == 429
    foreign = [event for event in after.events.values() if event.boundary is None]
    assert {event.exception_type for event in foreign} == {f"{__name__}._SdkError", "ValueError"}
    assert all(event.failure_reason == "" for event in foreign)
    assert "private" not in after.model_dump_json()
