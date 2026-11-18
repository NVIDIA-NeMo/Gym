# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native timeout failure propagation must not depend on a captured model turn."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from scripts.harness_conformance.native_timeouts import ReplyWitness, inspect_native_timeout
from scripts.harness_conformance.registry import HARNESSES
from scripts.harness_conformance.scenarios import SCENARIOS

from nemo_gym.core.failure_records import FailureReport
from nemo_gym.core.failure_transport import FailureHeader, FailureReportRow


def case():
    report = FailureReport(
        events={
            "earlier": {"boundary": "agent", "failure_kind": "agent_run_error", "failure_reason": "earlier attempt"},
            "native": {
                "boundary": "agent",
                "failure_kind": "model_transport_error",
                "failure_reason": "native timeout",
            },
        },
        failure_id="native",
    )
    wire = report.model_dump(mode="json")
    witness = {
        "attempts": [{"request": {"model": "m"}, "status_code": None, "response": None}],
        "native_timeout": {
            "seconds": 1,
            "stall_seconds": 120,
            "error_pattern": "operation timed out",
            "artifacts": [{"path": "native.db", "error": "TimeoutError: The operation timed out"}],
            "replies": [{"body": "The operation timed out", "report": wire, "elapsed_seconds": 2}],
            "caller_replies": [deepcopy(wire)],
        },
    }
    return witness, FailureReportRow.fields(report), {"timed_out": False, "returncode": 1}


def failures(witness, record, execution):
    return {check["id"] for check in inspect_native_timeout(witness, record, execution) if check["status"] == "fail"}


def test_complete_native_timeout_without_trajectory_or_model_capture():
    assert not failures(*case())


@pytest.mark.parametrize("where", ["caller", "saved"])
def test_every_original_event_survives_independent_of_selected_failure(where):
    witness, record, execution = case()
    events = (
        witness["native_timeout"]["caller_replies"][0]["events"] if where == "caller" else record["_ng_failure_events"]
    )
    events.pop("earlier")
    assert f"native_timeout.{where}_events" in failures(witness, record, execution)


@pytest.mark.parametrize("where", ["caller", "saved"])
def test_changed_original_fact_is_not_forwarding(where):
    witness, record, execution = case()
    events = (
        witness["native_timeout"]["caller_replies"][0]["events"] if where == "caller" else record["_ng_failure_events"]
    )
    events["native"]["failure_reason"] = "changed"
    assert f"native_timeout.{where}_events" in failures(witness, record, execution)


def test_missing_source_diagnostics_cannot_pass_vacuously():
    witness, record, execution = case()
    witness["native_timeout"]["replies"][0]["report"] = {}
    assert {"native_timeout.failure_event", "native_timeout.caller_events", "native_timeout.saved_events"} <= failures(
        witness, record, execution
    )


def test_native_timeout_cannot_manufacture_gym_response_timeout():
    witness, record, execution = case()
    witness["native_timeout"]["replies"][0]["report"]["events"]["native"]["failure_kind"] = (
        "model_policy_response_timeout"
    )
    assert "native_timeout.failure_event" in failures(witness, record, execution)


@pytest.mark.parametrize("mutation", ["outer", "server_response", "not_timeout", "server_deadline"])
def test_native_deadline_must_win(mutation):
    witness, record, execution = case()
    if mutation == "outer":
        execution["timed_out"] = True
    elif mutation == "server_response":
        witness["attempts"][0]["status_code"] = 504
    elif mutation == "not_timeout":
        witness["native_timeout"]["artifacts"][0]["error"] = "connection reset"
    else:
        witness["native_timeout"]["replies"][0]["elapsed_seconds"] = 121
    assert "native_timeout.trigger" in failures(witness, record, execution)


def test_canonical_saved_report_is_required_even_if_partial_response_has_events():
    witness, record, execution = case()
    record["_ng_failure_partial_response"] = {"failure_report": witness["native_timeout"]["replies"][0]["report"]}
    record["_ng_failure_events"] = {}
    assert "native_timeout.saved_events" in failures(witness, record, execution)


def test_terminal_failure_must_still_select_original_cause():
    witness, record, execution = case()
    record["_ng_failure_id"] = "earlier"
    assert "native_timeout.terminal_failure" in failures(witness, record, execution)


def test_other_harnesses_explicitly_report_unsupported(tmp_path):
    scenario = next(s for s in SCENARIOS if s.native_model_timeout)
    configured = HARNESSES["opencode"].settings(tmp_path, 90, scenario)
    assert not configured.get("unsupported_reason")
    assert (
        configured.probe_overrides.native_timeout_seconds * 1000 == configured.harness_overrides.model_call_timeout_ms
    )
    assert HARNESSES["simple"].settings(tmp_path, 90, scenario).unsupported_reason


async def test_wire_witness_preserves_reply_and_decodes_case_insensitive_headers():
    witness, _, _ = case()
    report = FailureReport.model_validate(witness["native_timeout"]["replies"][0]["report"])
    messages = [
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(key.lower().encode(), value.encode()) for key, value in FailureHeader.encode(report).items()],
        },
        {"type": "http.response.body", "body": b'{"status":"failed"}'},
    ]

    class Probe:
        replies = []

        def save(self):
            pass

    probe, received = Probe(), []

    async def app(scope, receive, send):
        for message in messages:
            await send(message)

    async def send(message):
        received.append(message)

    await ReplyWitness(app, probe)({"type": "http", "path": "/v1/responses"}, None, send)
    assert received == messages
    assert probe.replies[0]["report"] == report.model_dump(mode="json")


@pytest.mark.parametrize(
    "kind,boundary",
    [
        ("agent_timeout", "agent"),
        ("transport_timeout", "model"),
        ("model_request_error", "model"),
        ("custom:native_timeout", "model"),
    ],
)
def test_registered_or_namespaced_native_classification_is_preserved(kind, boundary):
    witness, record, execution = case()
    for events in (
        witness["native_timeout"]["replies"][0]["report"]["events"],
        witness["native_timeout"]["caller_replies"][0]["events"],
        record["_ng_failure_events"],
    ):
        events["native"].update(failure_kind=kind, boundary=boundary)
    assert "native_timeout.failure_event" not in failures(witness, record, execution)


@pytest.mark.parametrize("lose_event", [False, True])
def test_full_episode_verdict_uses_native_requirements_not_unavailable_capture(tmp_path, lose_event):
    from scripts.harness_conformance.runner import inspect_episode

    witness, record, execution = case()
    witness["seeded"] = 1
    if lose_event:
        record["_ng_failure_events"].pop("earlier")
    (tmp_path / "witness.json").write_text(json.dumps(witness))
    (tmp_path / "rollouts_failures.jsonl").write_text(json.dumps(record) + "\n")
    scenario = next(s for s in SCENARIOS if s.native_model_timeout)
    result = inspect_episode(scenario, tmp_path, execution)
    assert result["verdict"] == ("not_fulfilled" if lose_event else "fulfilled")
    assert result["p0_verdict"] == "fulfilled"
    diagnostics = {check["id"]: check for check in result["diagnostic_checks"]}
    assert diagnostics["model_calls.present"]["status"] == "fail"
    for key in (
        "model.finished",
        "verifier.outcome",
        "verifier.answer",
        "verifier.saved_reward",
        "verifier.saved_outcome",
    ):
        assert diagnostics[key]["status"] == "not_applicable"
    assert all(
        check["id"].startswith("native_timeout.")
        or check["id"] in {"episode.initialized", "model.reached", "model.protocol"}
        for check in result["checks"]
    )


def test_passing_native_episode_has_successful_suite_exit_and_visible_diagnostics(tmp_path, monkeypatch):
    from scripts.harness_conformance import runner

    def execute(command, *, directory, timeout):
        witness, record, execution = case()
        witness["seeded"] = 1
        (directory / "witness.json").write_text(json.dumps(witness))
        (directory / "rollouts_failures.jsonl").write_text(json.dumps(record) + "\n")
        return {**execution, "returncode": 0}

    monkeypatch.setattr(runner, "run_process", execute)
    scenario = next(s for s in SCENARIOS if s.native_model_timeout)
    output = tmp_path / "suite"
    summary, code = runner.run_suite(
        harnesses=["opencode"], scenarios=(scenario,), output=output, timeout=90, gate_tier="P1"
    )
    assert code == 0 and summary["verdict"] == "fulfilled"
    assert "Ancillary diagnostics (not scenario requirements)" in (output / "conformance_report.md").read_text()


async def test_stall_expiration_records_actual_provider_response(tmp_path):
    from scripts.harness_conformance.native_timeouts import NativeTimeoutProbe

    scenario = next(s for s in SCENARIOS if s.native_model_timeout)
    probe = NativeTimeoutProbe(
        scenario, tmp_path, responses=[{"kind": "stall", "delay_seconds": 0.001}], native_timeout_seconds=0.0001
    )
    response = await probe.model(SimpleNamespace(json=AsyncMock(return_value={"model": "m"})))
    assert response.status_code == 504
    witness = json.loads((tmp_path / "witness.json").read_text())
    assert witness["attempts"][0]["status_code"] == 504
    assert witness["attempts"][0]["response"] == json.loads(response.body)


@pytest.mark.parametrize("condition", ["native", "non_native", "setup", "outer"])
async def test_worker_collection_accepts_only_witnessed_native_no_score(tmp_path, monkeypatch, condition):
    from scripts.harness_conformance.episode import _collect_episode

    from nemo_gym.rollouts import RolloutCollectionHelper
    from nemo_gym.rollouts.rollout_recovery import IncompleteEvaluationError

    witness, _, _ = case()
    if condition == "setup":
        witness["attempts"] = []

    def save():
        (tmp_path / "witness.json").write_text(json.dumps(witness))

    probe = SimpleNamespace(save=save)

    async def collect(*args):
        if condition == "outer":
            await asyncio.Event().wait()
        raise IncompleteEvaluationError("no score")

    monkeypatch.setattr(RolloutCollectionHelper, "run_from_config", collect)
    kwargs = dict(directory=tmp_path, timeout=0.001, native_timeout=condition != "non_native")
    if condition == "native":
        await _collect_episode(None, probe, **kwargs)
    else:
        with pytest.raises(TimeoutError if condition == "outer" else IncompleteEvaluationError):
            await _collect_episode(None, probe, **kwargs)
