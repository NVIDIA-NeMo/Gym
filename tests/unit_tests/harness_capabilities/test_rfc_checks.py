# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise missing RFC requirements independently of producer defaults and copies."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from nemo_gym.harness_capabilities.checker import EvidenceScope, inspect_record
from nemo_gym.harness_capabilities.cli import inspect_bundle, main
from nemo_gym.harness_capabilities.reader import hydrate_record
from nemo_gym.harness_capabilities.registry import CHECKS
from tests.unit_tests.harness_capabilities.synthetic import evidence_record


def inspect(record, *, require_sandbox=False):
    return inspect_record(hydrate_record(record), scope=EvidenceScope(require_sandbox=require_sandbox))


def ids(result):
    return {f"{f['evidence']}.{f['assertion']}" for f in result["findings"]}


def test_registered_json_schemas_are_valid():
    for check in CHECKS.values():
        if check.schema is not None:
            Draft202012Validator.check_schema(check.schema)


def test_complete_fixture_passes_added_requirements():
    result = inspect(evidence_record())
    assert result["verdict"] == "fulfilled", result["findings"]
    assert not ids(result)


@pytest.mark.parametrize("field", ["started_at", "completed_at"])
@pytest.mark.parametrize("value", ["missing", None, -0.01])
def test_call_timestamps_are_required_at_canonical_path(field, value):
    record = evidence_record()
    call = record["ng_trajectory"]["model_calls"][0]
    if value == "missing":
        del call[field]
    else:
        call[field] = value
    result = inspect(record)
    assert "TE-1.rfc.timing" in ids(result)
    assert any(f["location"].endswith("/model_calls/0/" + field) for f in result["findings"])


@pytest.mark.parametrize(
    "collection,field,check",
    [
        ("model_calls", "model_call_id", "TE-1.rfc.call_id"),
        ("turns", "invocation_id", "TE-3.rfc.invocation_id"),
        ("tool_calls", "tool_call_id", "TE-5.rfc.tool_id"),
        ("tool_calls", "tool_name", "TE-5.rfc.tool_name"),
        ("tool_calls", "invocation_id", "TE-5.rfc.invocation_id"),
    ],
)
def test_whitespace_does_not_establish_identity(collection, field, check):
    record = evidence_record()
    record["ng_trajectory"][collection][0][field] = " \t"
    assert check in ids(inspect(record))


@pytest.mark.parametrize("field", ["task_id", "rollout_id"])
def test_container_identity_cannot_come_from_other_fields(field):
    record = evidence_record()
    record["ng_trajectory"].pop(field)
    assert f"TE-8.rfc.{field}" in ids(inspect(record))


def test_capture_alone_cannot_satisfy_canonical_call_requirements():
    record = hydrate_record(evidence_record())
    del record["ng_trajectory"]["model_calls"]
    result = inspect(record)
    for te in ("TE-1", "TE-2", "TE-4", "TE-7", "TE-8", "TE-9"):
        assert f"{te}.rfc.calls" in ids(result)
        assert result["evidence"][te]["verdict"] == "not_fulfilled"


@pytest.mark.parametrize("collection,te", [("turns", "TE-3"), ("tool_calls", "TE-5")])
def test_invocation_target_is_the_saved_canonical_collection(collection, te):
    record = evidence_record()
    # Observation ownership is still complete; the designated target is absent.
    record["ng_trajectory"]["invocations"] = []
    assert record["ng_agent_observations"]["records"][0]["invocation_id"] == "invocation"
    result = inspect(record)
    assert f"{te}.rfc.reference_target" in ids(result)
    assert any(
        f["assertion"] == "rfc.reference_target" and f["location"].endswith(f"/{collection}/0/invocation_id")
        for f in result["findings"]
    )


def test_negative_step_timestamp_fails():
    record = evidence_record()
    record["ng_trajectory"]["turns"][0]["timestamp"] = -1.0
    assert "TE-3.rfc.timestamp" in ids(inspect(record))


@pytest.mark.parametrize("value", [None, False, 42])
def test_tool_output_must_be_saved_at_its_designated_field(value):
    record = evidence_record()
    record["ng_trajectory"]["tool_calls"][0]["output"] = value
    assert "TE-5.rfc.output" in ids(inspect(record))


@pytest.mark.parametrize("value", ["", [], {}])
def test_empty_tool_content_is_valid_json_evidence(value):
    check = CHECKS["TE-5.rfc.output"]
    assert Draft202012Validator(check.schema).is_valid({"output": value})


@pytest.mark.parametrize("field", ["evaluation_completed", "mask_sample"])
def test_evaluation_flags_must_be_present(field):
    record = evidence_record()
    del record[field]
    result = inspect(record)
    assert f"TE-6.rfc.{field}" in ids(result)
    assert result["evidence"]["TE-6"]["verdict"] == "not_fulfilled"


def test_incomplete_verification_does_not_force_a_mask():
    record = evidence_record()
    record.update(
        evaluation_completed=False, mask_sample=False, failure_kind="skipped", failure_reason="no submission"
    )
    result = inspect(record)
    assert result["evidence"]["TE-6"]["verdict"] == "fulfilled"


def test_sandbox_presence_requires_explicit_run_selection():
    record = evidence_record()
    assert inspect(record)["evidence"]["TE-6"]["verdict"] == "fulfilled"
    assert "TE-6.rfc.sandbox_records" in ids(inspect(record, require_sandbox=True))
    record["ng_agent_observations"]["records"].append(
        {"kind": "sandbox", "sandbox_id": "sandbox-1", "role": "agent", "outcome": "completed"}
    )
    assert inspect(record, require_sandbox=True)["evidence"]["TE-6"]["verdict"] == "fulfilled"


@pytest.mark.parametrize(
    "patch,check",
    [
        ({"sandbox_id": None}, "sandbox_id"),
        ({"sandbox_id": "  "}, "sandbox_id"),
        ({"outcome": "unknown"}, "sandbox_outcome"),
        ({"outcome": "failed"}, "sandbox_error"),
        ({"outcome": "oom", "error_type": " "}, "sandbox_error"),
        ({"outcome": "sandbox_error"}, "sandbox_error"),
    ],
)
def test_saved_sandbox_records_are_checked_even_without_presence_option(patch, check):
    record = evidence_record()
    record["ng_agent_observations"]["records"].append(
        {"kind": "sandbox", "sandbox_id": "sandbox-1", "role": "agent", "outcome": "completed", **patch}
    )
    result = inspect(record)
    assert f"TE-6.rfc.{check}" in ids(result)
    assert result["evidence"]["TE-6"]["verdict"] == "not_fulfilled"


@pytest.mark.parametrize("outcome", ["completed", "failed", "timeout", "oom", "sandbox_error", "cancelled"])
def test_supported_sandbox_outcomes(outcome):
    record = evidence_record()
    record["ng_agent_observations"]["records"].append(
        {"kind": "sandbox", "sandbox_id": "sandbox-1", "role": "verifier", "outcome": outcome, "error_type": "example"}
    )
    assert inspect(record, require_sandbox=True)["evidence"]["TE-6"]["verdict"] == "fulfilled"


def test_references_require_nonempty_canonical_ids():
    record = evidence_record()
    turn = record["ng_trajectory"]["turns"][0]
    turn["model_calls"] = []
    assert "TE-9.rfc.references" in ids(inspect(record))
    turn["model_calls"] = [copy.deepcopy(record["ng_trajectory"]["invocations"][0]["model_calls"][0])]
    turn["model_calls"][0].pop("model_call_id")
    # Native model permits model_ref + response_id; the RFC requires model_call_id.
    assert "TE-9.rfc.references" in ids(inspect(record))


def test_call_target_must_exist_in_the_same_saved_trajectory():
    record = hydrate_record(evidence_record())
    record["ng_trajectory"]["model_calls"].pop()
    assert len(record["ng_model_call_capture"]["calls"]) == 2
    assert "TE-9.rfc.reference_target" in ids(inspect(record))


@pytest.mark.parametrize("dialect", ["chat", "responses", "messages"])
def test_request_protocol_shapes(dialect):
    record = evidence_record()
    call = record["ng_trajectory"]["model_calls"][0]
    call["response_metadata"]["dialect"] = dialect
    key = "input" if dialect == "responses" else "messages"
    validator = Draft202012Validator(CHECKS["TE-4.rfc.request"].schema)
    call["request"] = {key: []}
    assert validator.is_valid(call)
    call["request"] = {key: "prompt"}
    assert validator.is_valid(call) == (dialect == "responses")
    call["request"] = {key: [42]}
    assert not validator.is_valid(call)


def test_chat_choices_require_an_assistant_message():
    record = evidence_record()
    call = record["ng_trajectory"]["model_calls"][0]
    call["response_metadata"]["dialect"] = "chat"
    validator = Draft202012Validator(CHECKS["TE-7.rfc.response"].schema)
    call["response"] = {"choices": [{}]}
    assert not validator.is_valid(call)
    call["response"] = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    assert validator.is_valid(call)


def test_empty_http_body_and_transport_failure_are_distinct():
    call = evidence_record()["ng_trajectory"]["model_calls"][0]
    presence = Draft202012Validator(CHECKS["TE-7.rfc.response_presence"].schema)
    response = Draft202012Validator(CHECKS["TE-7.rfc.response"].schema)
    call["response"] = None
    assert presence.is_valid(call) and response.is_valid(call)
    call["response_metadata"].update(status_code=None, error_category="timeout")
    assert presence.is_valid(call)
    call["response"] = {"error": "body without an HTTP response"}
    assert not presence.is_valid(call)
    del call["response"]
    assert not response.is_valid(call)


def test_existing_conflicting_checks_are_still_visible():
    record = evidence_record()
    record["ng_trajectory"]["tool_calls"][0]["status"] = "incomplete"
    result = inspect(record)
    assert "TE-5.tool.terminal" in ids(result)
    assert result["evidence"]["TE-5"]["verdict"] == "not_fulfilled"
    record = evidence_record()
    record["ng_trajectory"]["model_calls"][0]["response"] = None
    result = inspect(record)
    assert "TE-7.payload.response" in ids(result)
    assert "TE-7.rfc.response" not in ids(result)


def test_sandbox_cli_and_report_identity(tmp_path):
    bundle = tmp_path / "evaluator_rollouts.jsonl"
    bundle.write_text(json.dumps(evidence_record()) + "\n")
    ordinary, summary = inspect_bundle(bundle, output=tmp_path / "reports")
    sandboxed, other = inspect_bundle(bundle, output=tmp_path / "reports", scope=EvidenceScope(require_sandbox=True))
    assert ordinary != sandboxed
    assert summary["verdict"] == "fulfilled"
    assert other["evidence"]["TE-6"]["verdict"] == "not_fulfilled"
    assert other["applicability"]["require_sandbox"] is True
    assert main(["inspect", "--bundle", str(bundle), "--output", str(tmp_path / "cli"), "--require-sandbox"]) == 1
    assert (
        main(["matrix", "--harness", f"example={bundle}", "--output", str(tmp_path / "matrix"), "--require-sandbox"])
        == 1
    )


def test_diagnostics_do_not_copy_payloads():
    record = evidence_record()
    secret = "payload-canary-do-not-report"
    record["ng_trajectory"]["model_calls"][0]["request"] = {"input": [secret]}
    result = inspect(record)
    assert "TE-4.rfc.request" in ids(result)
    assert secret not in json.dumps(result)


@pytest.mark.parametrize(
    "patch,expected",
    [
        ({"status_code": 200, "dialect": "chat", "finish_reason": "stop"}, True),
        ({"status_code": 200, "dialect": "messages", "finish_reason": "end_turn"}, True),
        ({"status_code": 200, "dialect": "chat", "finish_reason": " "}, False),
        ({"status_code": 200, "dialect": "responses", "response_status": "incomplete"}, True),
        ({"status_code": 200, "dialect": "responses", "response_status": "queued"}, False),
        ({"status_code": 400, "error_category": "http_error"}, True),
        ({"status_code": 400}, False),
        ({"status_code": None, "error_category": "timeout"}, True),
        ({"status_code": None, "error_category": " "}, False),
        ({"status_code": 999, "error_category": "http_error"}, False),
    ],
)
def test_registered_outcome_alternatives(patch, expected):
    validator = Draft202012Validator(CHECKS["TE-1.rfc.outcome"].schema)
    assert validator.is_valid({"response_metadata": patch}) is expected


@pytest.mark.parametrize("name", ["", " \t", None])
def test_canonical_model_reference_requires_nonblank_name(name):
    record = evidence_record()
    call = record["ng_trajectory"]["model_calls"][0]
    call["response_metadata"]["model_ref"]["name"] = name
    validator = Draft202012Validator(CHECKS["TE-1.rfc.model"].schema)
    assert not validator.is_valid(call)


def test_response_id_is_required_for_success_but_optional_for_error():
    validator = Draft202012Validator(CHECKS["TE-1.rfc.response_id"].schema)
    assert not validator.is_valid({"response_metadata": {}})
    assert not validator.is_valid({"response_metadata": {"response_id": " "}})
    assert validator.is_valid({"response_metadata": {"response_id": "r"}})
    assert validator.is_valid({"response_metadata": {"error_category": "timeout"}})
    assert validator.is_valid({"response_metadata": {"error_category": "timeout", "response_id": None}})


def test_missing_timestamp_errors_are_not_duplicated():
    record = evidence_record()
    call = record["ng_trajectory"]["model_calls"][0]
    del call["started_at"], call["completed_at"]
    result = inspect(record)
    findings = [f for f in result["findings"] if f["assertion"] == "rfc.timing"]
    assert [f["location"] for f in findings] == [
        "record/ng_trajectory/model_calls/0/started_at",
        "record/ng_trajectory/model_calls/0/completed_at",
    ]
