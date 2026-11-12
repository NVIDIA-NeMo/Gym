# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary-selected facts survive raising, HTTP transport and Collection's saved reports."""

import json
import pickle

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from nemo_gym.client.http_client import raise_boundary_failure
from nemo_gym.core.failure_records import (
    FailureEvent,
    ModelFailureDetails,
    SandboxFailureDetails,
)
from nemo_gym.core.failure_transport import FailureHeader, FailureReportRow, RequestFailureLog, observe_failure
from nemo_gym.core.typed_exceptions import NgFailureEvent, failure_report_for
from nemo_gym.rollouts.rollout_collection import _episode_record


@pytest.mark.parametrize(
    "boundary,details",
    [
        ("sandbox", SandboxFailureDetails()),  # Creation failed before a sandbox ID existed.
        (
            "sandbox",
            SandboxFailureDetails(
                sandbox_id="sb-1",
                execution_id="exec-1",
                provider="opensandbox",
                provider_status_reason="OOMKilled",
                exit_code=137,
            ),
        ),
        ("model", ModelFailureDetails(provider_request_id="request-1", request_upload_complete=True)),
        ("model", None),
    ],
)
async def test_details_survive_exception_chain_http_and_collection(boundary, details, tmp_path):
    inner = NgFailureEvent(
        boundary=boundary,
        failure_kind="provider_unavailable" if boundary == "sandbox" else "model_request_error",
        failure_reason="Observed failure",
        operation="exec" if boundary == "sandbox" else "model_request",
        details=details,
    )
    try:
        raise inner
    except NgFailureEvent as caught:
        try:
            raise NgFailureEvent(
                boundary="agent", failure_kind="agent_route_failed", failure_reason="Agent failed"
            ) from caught
        except NgFailureEvent as outer:
            report = failure_report_for(outer)
            fragment = outer.to_record(unravel_inner=False)
            assert set(fragment["failure_events"]) == {outer.event_id}
            assert fragment["failure_events"][outer.event_id]["cause_id"] == inner.event_id
            restored = NgFailureEvent.from_record(json.loads(json.dumps(outer.to_record())))
            assert restored.event_id == outer.event_id != inner.event_id
    assert restored.__cause__.event_id == inner.event_id
    assert restored.__cause__.details == details
    assert type(restored.__cause__.details) is type(details)
    assert pickle.loads(pickle.dumps(restored)).to_record() == restored.to_record()

    async def endpoint(request):
        return web.json_response({"error": "failed"}, status=503, headers=FailureHeader.encode(report))

    app = web.Application()
    app.router.add_get("/failure", endpoint)
    async with TestServer(app) as server, aiohttp.ClientSession() as client:
        with RequestFailureLog.capture() as log:
            async with client.get(server.make_url("/failure")) as reply:
                with pytest.raises(NgFailureEvent) as caught:
                    await raise_boundary_failure(reply)
            assert caught.value.to_record() == restored.to_record()
            observe_failure(caught.value)
            observe_failure(caught.value)  # Forwarding does not create another event.
            recovered = log.reply_report(None)
    assert recovered.failure_id is None and len(recovered.events) == 2
    row = _episode_record(
        {
            "task_id": {"taskset": "test:source", "task_id": "t"},
            "result": {"reward": 1.0},
            "failure_report": recovered.model_dump(mode="json"),
        }
    )
    path = tmp_path / "rollout.jsonl"
    path.write_text(json.dumps(row) + "\n")
    saved = json.loads(path.read_text())
    assert saved["reward"] == 1.0
    assert FailureReportRow.report(saved) == recovered
    assert "_ng_failure_record" not in saved
    assert not any("retry" in key or "attempt" in key for key in saved)


@pytest.mark.parametrize(
    "boundary,details",
    [
        ("sandbox", ModelFailureDetails()),
        ("model", SandboxFailureDetails()),
        ("sandbox", {"provider_request_id": "r"}),
        ("model", {"sandbox_id": "s"}),
        ("agent", {}),
        ("model", {"attribution": "model"}),
        ("sandbox", {"retryable": True}),
        ("sandbox", {"cause": "oom"}),
        ("sandbox", {"provider_status_reason": "x" * 2001}),
        ("model", {"request_upload_complete": "true"}),
    ],
)
def test_details_are_validated_at_construction_and_decoding(boundary, details):
    fields = dict(boundary=boundary, failure_kind="model_request_error", failure_reason="observed", details=details)
    with pytest.raises(ValidationError):
        NgFailureEvent(**fields)
    # Dicts from transport go through the same boundary-selected schema.
    if isinstance(details, (SandboxFailureDetails, ModelFailureDetails)):
        return  # Empty wire objects have no variant tag: their boundary deliberately selects the type.
    with pytest.raises(ValidationError):
        NgFailureEvent.from_record({"failure_id": "f", "failure_events": {"f": fields}})


@pytest.mark.parametrize("boundary,expected", [("sandbox", SandboxFailureDetails), ("model", ModelFailureDetails)])
def test_empty_details_decode_using_boundary(boundary, expected):
    event = FailureEvent(boundary=boundary, failure_kind="provider_unavailable", details={})
    assert type(event.details) is expected
    assert type(FailureEvent.model_validate_json(event.model_dump_json()).details) is expected


def test_model_upload_evidence_has_one_location_and_provider_text_is_only_evidence():
    with pytest.raises(ValidationError, match="Extra inputs"):
        FailureEvent(boundary="model", failure_kind="model_request_error", request_upload_complete=True)
    failure = NgFailureEvent(
        boundary="sandbox",
        failure_kind="provider_unavailable",
        failure_reason="Sandbox unavailable",
        details=SandboxFailureDetails(provider_status_reason="possibly out of memory"),
    )
    rebuilt = NgFailureEvent.from_record(failure.to_record())
    assert rebuilt.failure_kind == "provider_unavailable"
    assert rebuilt.details == failure.details
