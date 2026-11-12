# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents import agent_tools
from nemo_gym.config.config_types import ResourcesServerRef
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.server_runtime.base_resources_server import BaseRunRequest


@pytest.mark.parametrize("host", ["localhost", "localhost.", "127.0.0.1", "127.0.0.2", "[::1]", "0.0.0.0", "[::]"])
def test_remote_endpoint_rejects_unreachable_bind(host, monkeypatch):
    monkeypatch.setattr(agent_tools, "get_server_url", lambda _: f"http://{host}:8000")
    with pytest.raises(ValueError, match="use_absolute_ip=true"):
        agent_tools.sandbox_server_url("model", require_reachable=True)


@pytest.mark.parametrize(
    "host,expected",
    [("127.0.0.1", "127.0.0.1"), ("localhost", "localhost"), ("0.0.0.0", "127.0.0.1"), ("[::]", "[::1]")],
)
def test_inherited_network_preserves_host_network_connectivity(host, expected, monkeypatch):
    monkeypatch.setattr(agent_tools, "get_server_url", lambda _: f"http://{host}:8000")
    assert agent_tools.sandbox_server_url("model") == f"http://{expected}:8000"


@pytest.mark.parametrize("reward,masked", [(1, False), (0, True)])
async def test_failure_zero_rejects_incompatible_verifier(reward, masked):
    original = NeMoGymResponse.model_validate(
        dict(
            id="x",
            created_at=0,
            model="test",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )
    )
    body = BaseRunRequest(responses_create_params={"input": "x"})
    reply = SimpleNamespace(
        ok=True,
        read=AsyncMock(return_value=json.dumps(dict(reward=reward, mask_sample=masked)).encode()),
        raise_for_status=lambda: None,
    )
    with pytest.raises(ValueError, match="unmasked zero"):
        await agent_tools.verify_agent_response(
            SimpleNamespace(post=AsyncMock(return_value=reply)),
            ResourcesServerRef(type="resources_servers", name="math"),
            body,
            original,
            {},
            force_zero_reward=True,
        )


@pytest.mark.parametrize("omitted", [False, True])
async def test_verification_transports_bounded_failure_report_without_changing_result(omitted):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from nemo_gym.core.failure_kinds import PROVIDER_IO_TIMEOUT, PROVIDER_OOM_KILLED
    from nemo_gym.core.failure_records import FailureReport, SandboxFailureDetails
    from nemo_gym.core.failure_transport import RequestFailureLog, observe_failure
    from nemo_gym.core.typed_exceptions import NgFailureEvent

    # Serialize through a verifier HTTP route, so the reference must survive the request wire format.
    verifier = FastAPI()
    received = []

    @verifier.post("/verify")
    async def verify(request: Request):
        payload = await request.json()
        received.append(payload)
        FailureReport.model_validate(payload["failure_report"])
        return {"reward": 1.0, "response": payload["response"]}

    failure = NgFailureEvent(
        boundary="sandbox",
        failure_kind=PROVIDER_OOM_KILLED,
        failure_reason="Current user container OOMKilled",
        details=SandboxFailureDetails(provider="opensandbox", sandbox_id="sandbox-1"),
    )
    cause = NgFailureEvent(boundary="sandbox", failure_kind=PROVIDER_IO_TIMEOUT, failure_reason="Read failed")
    failure.__cause__ = cause
    response = NeMoGymResponse.model_validate(
        dict(
            id="unchanged",
            created_at=0,
            model="test",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )
    )
    body = BaseRunRequest(responses_create_params={"input": "x"})
    # Simulate a run request subtype carrying an untrusted diagnostic field.
    dumped = body.model_dump(mode="json") | {"failure_report": {"failure_id": "forged"}, "task_field": "retained"}
    request_body = SimpleNamespace(model_dump=lambda **_: dumped)

    with TestClient(verifier) as wire_client:

        async def post(**kwargs):
            result = wire_client.post(kwargs["url_path"], json=kwargs["json"])
            return SimpleNamespace(ok=True, read=AsyncMock(return_value=result.content), raise_for_status=lambda: None)

        with RequestFailureLog.capture() as log:
            if omitted:
                log.MAX_EVENTS = 1
            observe_failure(failure)
            result = await agent_tools.verify_agent_response(
                SimpleNamespace(post=post),
                ResourcesServerRef(type="resources_servers", name="verifier"),
                request_body,
                response,
                {},
                force_zero_reward=False,
            )

    report = FailureReport.model_validate(received[0]["failure_report"])
    assert report.failure_id is None
    assert report.omitted is omitted
    assert received[0]["task_field"] == "retained"
    assert result["reward"] == 1.0
    assert result["response"] == response.model_dump(mode="json")
    if omitted:
        assert failure.event_id not in report.events
    else:
        assert report.events[failure.event_id].cause_id == cause.event_id
        assert report.events[cause.event_id].failure_kind == PROVIDER_IO_TIMEOUT


@pytest.mark.parametrize("overflow", [False, True])
def test_restored_response_without_request_scope_keeps_bounds_and_clears_selection(overflow):
    from nemo_gym.core.failure_kinds import PROVIDER_OOM_KILLED
    from nemo_gym.core.failure_records import FailureReport
    from nemo_gym.core.failure_transport import RequestFailureLog
    from nemo_gym.core.typed_exceptions import NgFailureEvent

    assert RequestFailureLog.current() is None
    errors = [
        NgFailureEvent(boundary="sandbox", failure_kind=PROVIDER_OOM_KILLED, failure_reason="OOMKilled")
        for _ in range(RequestFailureLog.MAX_EVENTS + 1 if overflow else 1)
    ]
    report = FailureReport(failure_id=errors[0].event_id, events={error.event_id: error.event for error in errors})
    response = NeMoGymResponse.model_validate(
        dict(
            id="saved",
            created_at=0,
            model="test",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
            failure_report=report.model_dump(mode="json"),
        )
    )
    forwarded = FailureReport.model_validate(agent_tools.verification_failure_fields(response)["failure_report"])
    assert forwarded.failure_id is None
    assert forwarded.omitted is overflow
    assert len(forwarded.events) == (0 if overflow else 1)
    assert response.failure_report["failure_id"] == errors[0].event_id
