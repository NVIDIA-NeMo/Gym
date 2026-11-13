# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-body observations reach Collection independently of verification policy."""

import pytest
from fastapi.testclient import TestClient

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentOutcome, AgentOutcomePolicyConfig
from nemo_gym.core import FailureHeader, FailureReport, FailureReportRow, NgFailureEvent
from nemo_gym.rollouts.rollout_collection import _episode_record
from tests.helpers.single_agent_turn import _agent_response, _environment_server, _request, _Response


@pytest.mark.parametrize("path", ["fail", "retry", "skip", "verify", "close_failure"])
def test_agent_body_events_reach_collection_on_every_exit(path):
    environment, client = _environment_server()
    event = NgFailureEvent(
        boundary="sandbox",
        failure_kind="provider_oom_killed",
        failure_reason="Current workload OOMKilled",
        stage="agent",
    )
    # No selected failure: recovered observations and sibling events are not an exception chain.
    report = FailureReport(events={event.event_id: event.event})
    response = _agent_response().model_dump(mode="json")
    response.update(
        status="failed",
        failure_report=report.model_dump(mode="json"),
        metadata={
            AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=event.failure_kind).model_dump_json(),
        },
    )
    client.responses[2] = _Response(response)
    if path in {"fail", "retry"}:
        environment.config.agent_outcome_policy = AgentOutcomePolicyConfig(by_failure_kind={event.failure_kind: path})
        del client.responses[4]
    elif path == "skip":
        environment.config.skip_verification = True
        del client.responses[4]
    elif path == "close_failure":
        client.responses = [
            *client.responses[:3],
            TimeoutError("close timed out"),
            client.responses[3],
            client.responses[5],
        ]

    app = environment.setup_webserver()
    environment.setup_exception_middleware(app)
    with TestClient(app) as http:
        reply = http.post("/run", json=_request().model_dump(mode="json"))
    assert reply.status_code == 200
    header = FailureHeader.decode(reply.headers)
    assert header is not None and not header.omitted
    assert header.events[event.event_id] == event.event
    row = FailureReportRow.merged_into(_episode_record(reply.json()), header)
    saved = FailureReportRow.report(row)
    assert saved.events[event.event_id] == event.event
    verified = [kwargs for _, url, kwargs in client.calls if url == "/verify"]
    assert bool(verified) is (path == "verify")
    if path in {"fail", "retry", "close_failure"}:
        assert reply.json()["result"] is None
        assert reply.json()["failure"]["terminal"] is (path == "fail")
        assert saved.failure_id != event.event_id
    else:
        assert reply.json()["failure"] is None
        assert row["reward"] == (0.0 if path == "skip" else 1.0)
        assert saved.failure_id is None


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("damage", ["omitted", "invalid", "oversized"])
def test_agent_body_diagnostics_omission_does_not_change_outcome(damage, verify):
    environment, client = _environment_server()
    event = NgFailureEvent(
        boundary="sandbox", failure_kind="provider_oom_killed", failure_reason="x" * 2000, stage="agent"
    )
    sibling = NgFailureEvent(
        boundary="sandbox", failure_kind="provider_io_timeout", failure_reason="y" * 2000, stage="agent"
    )
    report = FailureReport(events={event.event_id: event.event, sibling.event_id: sibling.event}).model_dump(
        mode="json"
    )
    if damage == "omitted":
        report = {"omitted": True}
    elif damage == "invalid":
        report = {"failure_id": "missing", "events": {}}
    response = _agent_response().model_dump(mode="json")
    response.update(
        status="completed" if verify else "failed",
        failure_report=report,
        metadata={AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=event.failure_kind).model_dump_json()},
    )
    client.responses[2] = _Response(response)
    if not verify:
        environment.config.agent_outcome_policy = AgentOutcomePolicyConfig(
            by_failure_kind={event.failure_kind: "fail"}
        )
        del client.responses[4]
    app = environment.setup_webserver()
    environment.setup_exception_middleware(app)
    with TestClient(app) as http:
        reply = http.post("/run", json=_request().model_dump(mode="json"))
    assert reply.status_code == 200
    if verify:
        assert reply.json()["failure"] is None
        assert reply.json()["result"]["reward"] == 1.0
    else:
        assert reply.json()["failure"]["terminal"] is True
        assert reply.json()["failure"]["partial_response"]["failure_report"] == report
    assert any(url == "/verify" for _, url, _ in client.calls) is verify
    header = FailureHeader.decode(reply.headers)
    assert header is not None and header.omitted
    row = FailureReportRow.merged_into(_episode_record(reply.json()), header)
    assert row[FailureReportRow.OMITTED] is True
