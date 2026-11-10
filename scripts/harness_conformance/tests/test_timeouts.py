# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutation checks prevent timeout conformance from passing after evidence loss."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from scripts.harness_conformance.timeouts import inspect_model_timeouts

from nemo_gym.client.openai_client import MODEL_RESPONSE_TIMEOUT_REASON
from nemo_gym.core.failure_kinds import MODEL_POLICY_RESPONSE_TIMEOUT, MODEL_REQUEST_ERROR
from nemo_gym.core.failure_records import FailureReport
from nemo_gym.core.failure_transport import FailureReportRow


@pytest.mark.parametrize("streamed", [False, True])
async def test_failure_report_witness_preserves_reply_and_reads_terminal_stream(streamed):
    from scripts.harness_conformance.timeouts import FailureReportWitness

    from nemo_gym.core.failure_transport import FailureHeader

    report = FailureReport(events={"failure": {"boundary": "model", "failure_kind": "model_request_error"}})
    payload = {
        "type": "response.completed",
        FailureHeader.STREAM_FIELD: report.model_dump(mode="json"),
    }
    body = ("data: " + json.dumps(payload) + "\n\n").encode() if streamed else b'{"answer": "ok"}'
    headers = (
        []
        if streamed
        else [(key.lower().encode(), value.encode()) for key, value in FailureHeader.encode(report).items()]
    )
    messages = [
        {"type": "http.response.start", "status": 200, "headers": headers},
        {"type": "http.response.body", "body": body[:25], "more_body": True},
        {"type": "http.response.body", "body": body[25:]},
    ]

    async def app(scope, receive, send):
        for message in messages:
            await send(message)

    received, reports = [], []

    async def send(message):
        received.append(message)

    async def receive():
        return {"type": "http.disconnect"}

    await FailureReportWitness(app, reports.append)({"type": "http"}, receive, send)
    assert received == messages
    assert reports == [report]


@pytest.mark.parametrize("exhausted", [False, True])
def test_real_model_timeout_reaches_environment(tmp_path, exhausted):
    """Validate the one-shot fixture independently of the harness's conformance verdict."""
    from scripts.harness_conformance.runner import inspect_episode
    from scripts.harness_conformance.scenarios import SCENARIOS

    root = Path(__file__).resolve().parents[3]
    name = "model_timeout_exhausted" if exhausted else "model_timeout_recovered"
    scenario = next(s for s in SCENARIOS if s.name == name)
    with (tmp_path / "episode.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.harness_conformance.episode",
                "--harness",
                "simple",
                "--scenario",
                name,
                "--directory",
                str(tmp_path),
                "--timeout",
                "30",
            ],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root), "NEMO_GYM_EXTRA_ROOTS": str(root)},
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=45,
        )
    # Collection may exit 1 when the real timeout leaves no scored row. Require
    # its actual failure evidence below rather than a successful evaluation.
    assert result.returncode in (0, 1), (tmp_path / "episode.log").read_text()
    witness = json.loads((tmp_path / "witness.json").read_text())
    assert witness["seeded"] == witness["closed"] == 1
    (attempt,) = witness["timeout_attempts"]
    assert attempt["timed_out"] is True
    (model_report,) = witness["model_server_failure_reports"]
    model = FailureReport.model_validate(model_report)
    (event,) = model.events.values()
    assert not model.omitted and model.failure_id == event.event_id
    assert event.boundary == "model" and event.failure_kind == MODEL_REQUEST_ERROR
    assert event.operation_id and event.attempt_id
    assert event.failure_reason == MODEL_RESPONSE_TIMEOUT_REASON
    assert event.details.request_upload_complete is True
    assert event.foreign_error.exception_type.endswith("TimeoutError")

    (response,) = witness["agent_responses"]
    assert response["status"] == "failed" and response["error"]
    final = FailureReport.model_validate(response["failure_report"])
    assert final.events[final.failure_id].failure_kind == MODEL_POLICY_RESPONSE_TIMEOUT
    assert final.events[final.failure_id].cause_id == event.event_id
    (caller_report,) = witness["caller_failure_reports"]
    (record,) = [json.loads(line) for line in (tmp_path / "rollouts_failures.jsonl").read_text().splitlines()]
    for report in (FailureReport.model_validate(caller_report), final, FailureReportRow.report(record)):
        assert not report.omitted and report.events.get(event.event_id) == event
    assert record.get("reward") is None and record.get("evaluation_completed") is not True

    measured = inspect_episode(scenario, tmp_path, {"returncode": result.returncode, "timed_out": False})
    checks = {check["id"]: check for check in measured["checks"]}
    # The current SimpleAgent stops after its first confirmed timeout. Keep the
    # conformance failure visible; this test establishes that the fixture works.
    assert measured["verdict"] == "not_fulfilled"
    assert checks["timeout.attempts"]["status"] == "fail"
    if exhausted:
        assert measured["p0_verdict"] == "fulfilled", measured["issues"]
        assert {key for key, check in checks.items() if check["status"] == "fail"} == {
            "timeout.attempts",
            "timeout.model_events",
        }
    else:
        assert checks["timeout.recovered_outcome"]["status"] == "fail"


def case(exhausted):
    count = 3 if exhausted else 1
    events = {
        f"model-{index}": {
            "boundary": "model",
            "failure_kind": "model_request_error",
            "operation": "model_request",
            "operation_id": "model-operation",
            "attempt_id": f"attempt-{index}",
            "failure_reason": MODEL_RESPONSE_TIMEOUT_REASON,
            "details": {"request_upload_complete": True},
            "foreign_error": {"exception_type": "builtins.TimeoutError", "message": ""},
        }
        for index in range(count)
    }
    model = FailureReport(events=events, failure_id=f"model-{count - 1}" if exhausted else None)
    final = model.model_copy(deep=True)
    if exhausted:
        final = FailureReport(
            events={
                **events,
                "agent": {
                    "boundary": "agent",
                    "failure_kind": "model_policy_response_timeout",
                    "failure_reason": "Policy model timed out",
                    "cause_id": model.failure_id,
                },
            },
            failure_id="agent",
        )
    response = {
        "status": "failed" if exhausted else "completed",
        "error": {"code": "server_error"} if exhausted else None,
        "output": [] if exhausted else [{"content": [{"text": "CONFORMANCE_DONE"}]}],
        "failure_report": final.model_dump(mode="json"),
    }
    witness = {
        "timeout_attempts": [{"request": {"messages": ["request"]}, "timed_out": True} for _ in range(count)]
        + ([] if exhausted else [{"request": {"messages": ["request"]}, "timed_out": False}]),
        "model_server_failure_reports": [model.model_dump(mode="json")],
        "caller_failure_reports": [model.model_dump(mode="json")],
        "agent_responses": [response],
        "verifications": [],
    }
    record = {
        **FailureReportRow.fields(final),
        "reward": None if exhausted else 1.0,
        "mask_sample": exhausted,
        "evaluation_completed": not exhausted,
    }
    return witness, record


def failures(witness, record, *, exhausted):
    checks = inspect_model_timeouts(witness, record, exhausted=exhausted)
    assert checks and all(check["tier"] == "P1" for check in checks)
    return {check["id"] for check in checks if check["status"] == "fail"}


@pytest.mark.parametrize("exhausted", [False, True])
def test_complete_timeout_evidence(exhausted):
    assert not failures(*case(exhausted), exhausted=exhausted)


@pytest.mark.parametrize("target", ["caller_failure_reports", "model_server_failure_reports"])
def test_missing_recovered_event_is_not_a_pass(target):
    witness, record = case(False)
    witness[target][0]["events"] = {}
    assert failures(witness, record, exhausted=False)


@pytest.mark.parametrize("field,value", [("failure_kind", "model_transport_error"), ("attempt_id", "changed")])
def test_mutated_forwarded_fact_fails(field, value):
    witness, record = case(False)
    witness["caller_failure_reports"][0]["events"]["model-0"][field] = value
    assert "timeout.caller_events" in failures(witness, record, exhausted=False)


def test_changed_event_identity_fails():
    witness, record = case(False)
    events = witness["caller_failure_reports"][0]["events"]
    event = events.pop("model-0")
    event["event_id"] = "changed"
    events["changed"] = event
    assert "timeout.caller_events" in failures(witness, record, exhausted=False)


@pytest.mark.parametrize("exhausted", [False, True])
def test_additional_provider_retry_fails(exhausted):
    witness, record = case(exhausted)
    witness["timeout_attempts"].append(deepcopy(witness["timeout_attempts"][-1]))
    assert "timeout.attempts" in failures(witness, record, exhausted=exhausted)


def test_exhausted_earlier_attempt_must_survive_collection():
    witness, record = case(True)
    record["_ng_failure_events"].pop("model-0")
    assert "timeout.saved_events" in failures(witness, record, exhausted=True)


@pytest.mark.parametrize("boundary", ["model", "caller", "saved"])
def test_omission_cannot_certify_complete_forwarding(boundary):
    witness, record = case(False)
    if boundary == "saved":
        record["_ng_failure_events_omitted"] = True
    else:
        key = "model_server_failure_reports" if boundary == "model" else "caller_failure_reports"
        witness[key][0]["omitted"] = True
    assert failures(witness, record, exhausted=False)


def test_success_cannot_select_a_terminal_failure():
    witness, record = case(False)
    record["_ng_failure_id"] = "model-0"
    assert "timeout.recovered_outcome" in failures(witness, record, exhausted=False)


def test_terminal_response_must_retain_model_cause():
    witness, record = case(True)
    witness["agent_responses"][0]["failure_report"]["events"]["agent"].pop("cause_id")
    assert "timeout.terminal_outcome" in failures(witness, record, exhausted=True)


@pytest.mark.parametrize("field,value", [("operation_id", None), ("failure_reason", "connection timed out")])
def test_incomplete_timeout_evidence_cannot_manufacture_response_timeout(field, value):
    witness, record = case(False)
    witness["model_server_failure_reports"][0]["events"]["model-0"][field] = value
    assert "timeout.model_events" in failures(witness, record, exhausted=False)


def test_unknown_upload_is_not_confirmed_timeout():
    witness, record = case(False)
    witness["model_server_failure_reports"][0]["events"]["model-0"]["details"]["request_upload_complete"] = None
    assert "timeout.model_events" in failures(witness, record, exhausted=False)


def test_unavailable_artifacts_fail_closed():
    assert failures(None, None, exhausted=False)


def test_malformed_success_report_cannot_hide_terminal_failure():
    witness, record = case(False)
    witness["agent_responses"][0]["failure_report"] = {"failure_id": "missing"}
    assert "timeout.recovered_outcome" in failures(witness, record, exhausted=False)


def test_failed_episode_cannot_retain_successful_reward():
    witness, record = case(True)
    record["reward"] = 1.0
    assert "timeout.terminal_outcome" in failures(witness, record, exhausted=True)


def test_full_partial_response_does_not_replace_canonical_saved_event_history():
    witness, record = case(True)
    record["_ng_failure_partial_response"] = deepcopy(witness["agent_responses"][0])
    # The selected last failure and its causes survive, but earlier retries only
    # remain in the partial response. Consumers of Collection's event map lose them.
    record["_ng_failure_events"].pop("model-0")
    record["_ng_failure_events"].pop("model-1")
    assert "timeout.saved_events" in failures(witness, record, exhausted=True)


@pytest.mark.parametrize(
    "attempts",
    [
        None,
        {},
        [None],
        [3],
        [{"timed_out": True}, {"timed_out": False}],
        [{"request": {"model": "m"}, "timed_out": 1}, {"request": {"model": "m"}, "timed_out": 0}],
    ],
)
def test_malformed_attempts_fail_without_raising(attempts):
    witness, record = case(False)
    witness["timeout_attempts"] = attempts
    assert "timeout.attempts" in failures(witness, record, exhausted=False)
