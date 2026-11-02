# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise missing RFC requirements independently of producer defaults and copies."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from nemo_gym.harness_capabilities.checker import EvidenceScope, Inspector, inspect_record
from nemo_gym.harness_capabilities.checks import Check, SchemaCheck
from nemo_gym.harness_capabilities.cli import inspect_bundle, main
from nemo_gym.harness_capabilities.results import Results
from tests.unit_tests.harness_capabilities.synthetic import evidence_record


def item_schema(check_id):
    """Exercise the schemas on actual check definitions, not separate copies."""
    definitions = {}

    class CollectingResults(Results):
        def run(self, check: Check) -> bool:
            definitions[check.id] = check
            return super().run(check)

    inspector = Inspector(evidence_record(), EvidenceScope())
    inspector.results = CollectingResults()
    inspector.model_calls()
    inspector.structure()
    check = definitions[check_id]
    assert isinstance(check, SchemaCheck)
    return check.schema["items"]


def inspect(record, *, require_sandbox=False):
    return inspect_record(record, scope=EvidenceScope(require_sandbox=require_sandbox))


def ids(result):
    return {c["id"] for c in result["checks"] if c["status"] == "fail"}


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
    assert "calls.timing" in ids(result)
    assert any("$.ng_trajectory.model_calls" in f["location"] for f in result["findings"])


@pytest.mark.parametrize(
    "collection,field,check",
    [
        ("model_calls", "model_call_id", "calls.identity"),
        ("turns", "invocation_id", "steps.invocation"),
        ("tool_calls", "tool_call_id", "tools.identity"),
        ("tool_calls", "tool_name", "tools.name"),
        ("tool_calls", "invocation_id", "tools.invocation"),
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
    assert f"trajectory.{field}" in ids(inspect(record))


def test_negative_step_timestamp_fails():
    record = evidence_record()
    record["ng_trajectory"]["turns"][0]["timestamp"] = -1.0
    assert "steps.timestamp" in ids(inspect(record))


@pytest.mark.parametrize("field,check", [("request", "calls.request"), ("response", "calls.response")])
def test_turn_content_cannot_replace_missing_canonical_call_content(field, check):
    record = evidence_record()
    # Content on the turn still cannot satisfy the designated model-call fields.
    assert record["ng_trajectory"]["turns"][0]["question"]
    assert record["ng_trajectory"]["turns"][0]["answer"]
    record["ng_trajectory"]["model_calls"][0].pop(field)
    result = inspect(record)
    assert result["evidence"]["TE-3"]["verdict"] == "fulfilled"
    assert check in ids(result)


@pytest.mark.parametrize("value", [None, False, 42])
def test_tool_output_must_be_saved_at_its_designated_field(value):
    record = evidence_record()
    record["ng_trajectory"]["tool_calls"][0]["output"] = value
    assert "tools.output" in ids(inspect(record))


@pytest.mark.parametrize("value", ["", [], {}])
def test_empty_tool_content_is_valid_json_evidence(value):
    assert Draft202012Validator(item_schema("tools.output")).is_valid({"output": value})


@pytest.mark.parametrize("field", ["evaluation_completed", "mask_sample"])
def test_evaluation_flags_must_be_present(field):
    record = evidence_record()
    del record[field]
    result = inspect(record)
    assert f"evaluation.{field}" in ids(result)
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
    assert "sandbox.present" in ids(inspect(record, require_sandbox=True))
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
    assert {"sandbox_id": "sandbox.identity", "sandbox_outcome": "sandbox.outcome", "sandbox_error": "sandbox.error"}[
        check
    ] in ids(result)
    assert result["evidence"]["TE-6"]["verdict"] == "not_fulfilled"


@pytest.mark.parametrize("outcome", ["completed", "failed", "timeout", "oom", "sandbox_error", "cancelled"])
def test_supported_sandbox_outcomes(outcome):
    record = evidence_record()
    record["ng_agent_observations"]["records"].append(
        {"kind": "sandbox", "sandbox_id": "sandbox-1", "role": "verifier", "outcome": outcome, "error_type": "example"}
    )
    assert inspect(record, require_sandbox=True)["evidence"]["TE-6"]["verdict"] == "fulfilled"


def test_turns_require_nonempty_call_references():
    record = evidence_record()
    turn = record["ng_trajectory"]["turns"][0]
    turn["model_calls"] = []
    assert "steps.references" in ids(inspect(record))


@pytest.mark.parametrize("form", ["call_id", "response_id", "response_id_null_call_id", "all"])
def test_supported_call_reference_forms_resolve(form):
    record = evidence_record()
    for turn in record["ng_trajectory"]["turns"]:
        ref = turn["model_calls"][0]
        if form == "call_id":
            ref["model_ref"] = ref["response_id"] = None
        elif form == "response_id":
            del ref["model_call_id"]
        elif form == "response_id_null_call_id":
            ref["model_call_id"] = None
    result = inspect(record)
    assert result["evidence"]["TE-9"]["verdict"] == "fulfilled", result["findings"]


@pytest.mark.parametrize(
    "ref",
    [
        {},
        {"response_id": "response-1"},
        {"model_ref": {"type": "responses_api_models", "name": "synthetic-model"}},
        {"model_call_id": " \t"},
        {"model_ref": {"type": "responses_api_models", "name": " "}, "response_id": "response-1"},
        {"model_ref": {"type": "responses_api_models", "name": "synthetic-model"}, "response_id": " "},
        {"model_call_id": "attempt-1", "response_id": ""},
    ],
)
def test_incomplete_or_blank_call_reference_fails_schema(ref):
    validator = Draft202012Validator(item_schema("steps.references"))
    assert not validator.is_valid({"model_calls": [ref]})


@pytest.mark.parametrize("response_reference", [False, True])
def test_call_target_must_exist_in_the_same_saved_trajectory(response_reference):
    record = evidence_record()
    if response_reference:
        for turn in record["ng_trajectory"]["turns"]:
            turn["model_calls"][0].pop("model_call_id")
    record["ng_trajectory"]["model_calls"].pop()
    assert len(record["ng_model_call_capture"]["calls"]) == 2
    assert "steps.call_target" in ids(inspect(record))


@pytest.mark.parametrize("field", ["model_call_id", "response_id", "model_ref"])
def test_conflicting_supplied_call_identifiers_do_not_fall_back(field):
    record = evidence_record()
    ref = record["ng_trajectory"]["turns"][0]["model_calls"][0]
    ref[field] = {"type": "responses_api_models", "name": "other-model"} if field == "model_ref" else "missing"
    result = inspect(record)
    assert "steps.references" not in ids(result)
    assert "steps.call_target" in ids(result)


@pytest.mark.parametrize("response_reference", [False, True])
def test_ambiguous_canonical_call_reference_fails(response_reference):
    record = evidence_record()
    trajectory = record["ng_trajectory"]
    if response_reference:
        trajectory["turns"][0]["model_calls"][0].pop("model_call_id")
    duplicate = copy.deepcopy(trajectory["model_calls"][0])
    if response_reference:
        duplicate["model_call_id"] = "another-attempt-with-same-response-id"
    trajectory["model_calls"].append(duplicate)
    assert "steps.call_target" in ids(inspect(record))


@pytest.mark.parametrize("other_server", [True, False])
def test_previous_response_lookup_is_scoped_to_model_server(other_server):
    record = evidence_record()
    trajectory = record["ng_trajectory"]
    calls = trajectory["model_calls"]
    calls[1]["request"]["previous_response_id"] = calls[0]["response_metadata"]["response_id"]
    assert not ids(inspect(record))
    other = copy.deepcopy(calls[0])
    other["model_call_id"] = "another-call"
    if other_server:
        other["response_metadata"]["model_ref"]["name"] = "other-model"
    calls.insert(1, other)
    reference = {"model_call_id": other["model_call_id"]}
    trajectory["invocations"][0]["model_calls"].append(reference)
    turn = copy.deepcopy(trajectory["turns"][0])
    turn.update(turn_no=3, step_count=3, model_calls=[reference])
    trajectory["turns"].append(turn)
    assert ids(inspect(record)) == (set() if other_server else {"content.previous_response"})


def test_response_id_is_scoped_to_the_model_server():
    record = evidence_record()
    trajectory = record["ng_trajectory"]
    trajectory["turns"][0]["model_calls"][0].pop("model_call_id")
    other = copy.deepcopy(trajectory["model_calls"][0])
    other["model_call_id"] = "other-server-call"
    other["response_metadata"]["model_ref"]["name"] = "other-model"
    trajectory["model_calls"].append(other)
    assert "steps.call_target" not in ids(inspect(record))


@pytest.mark.parametrize("dialect", ["chat", "responses", "messages"])
def test_request_protocol_shapes(dialect):
    record = evidence_record()
    call = record["ng_trajectory"]["model_calls"][0]
    call["response_metadata"]["dialect"] = dialect
    key = "input" if dialect == "responses" else "messages"
    validator = Draft202012Validator(item_schema("calls.request"))
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
    validator = Draft202012Validator(item_schema("calls.response"))
    call["response"] = {"choices": [{}]}
    assert not validator.is_valid(call)
    call["response"] = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    assert validator.is_valid(call)


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
    assert "calls.request" in ids(result)
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
def test_outcome_alternatives(patch, expected):
    validator = Draft202012Validator(item_schema("calls.outcome"))
    assert validator.is_valid({"response_metadata": patch}) is expected


@pytest.mark.parametrize("name", ["", " \t", None])
def test_canonical_model_reference_requires_nonblank_name(name):
    record = evidence_record()
    call = record["ng_trajectory"]["model_calls"][0]
    call["response_metadata"]["model_ref"]["name"] = name
    validator = Draft202012Validator(item_schema("calls.model"))
    assert not validator.is_valid(call)


def test_response_id_is_required_for_success_but_optional_for_error():
    validator = Draft202012Validator(item_schema("calls.response_id"))
    assert not validator.is_valid({"response_metadata": {}})
    assert not validator.is_valid({"response_metadata": {"response_id": " "}})
    assert validator.is_valid({"response_metadata": {"response_id": "r"}})
    assert validator.is_valid({"response_metadata": {"error_category": "timeout"}})
    assert validator.is_valid({"response_metadata": {"error_category": "timeout", "response_id": None}})


def test_no_response_transport_error_and_returned_body_are_distinct():
    call = evidence_record()["ng_trajectory"]["model_calls"][0]
    validator = Draft202012Validator(item_schema("calls.response"))
    call["response"] = None
    assert not validator.is_valid(call)
    call["response_metadata"].update(status_code=None, error_category="timeout")
    assert validator.is_valid(call)
    call["response"] = {"error": "body without status"}
    assert not validator.is_valid(call)
    call["response_metadata"]["status_code"] = 500
    assert validator.is_valid(call)


@pytest.mark.parametrize("field", ["error_category", "response_status", "finish_reason"])
def test_supplied_optional_outcome_strings_are_nonblank(field):
    metadata = {"status_code": 200, "dialect": "responses", "response_status": "completed"}
    metadata[field] = " "
    assert not Draft202012Validator(item_schema("calls.outcome")["properties"]["response_metadata"]).is_valid(metadata)
