# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from nemo_gym.rollouts.harness_capabilities.compaction import inspect_compaction
from tests.unit_tests.rollouts.harness_capabilities.synthetic import evidence_and_witness


@pytest.fixture
def compaction_evidence():
    record, witness = evidence_and_witness()
    reference = {
        "model_call_id": "summary-call",
        "response_id": "summary-response",
        "model_ref": {"type": "responses_api_models", "name": "synthetic-model"},
    }
    summary = "WITNESSED_SUMMARY: continue the task."
    response = {"id": "summary-response", "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}}
    helper = {
        "model_call_id": "summary-call",
        "response_metadata": {"response_id": "summary-response", "model_ref": reference["model_ref"]},
    }
    record["ng_trajectory"]["model_calls"].insert(1, helper)
    record["ng_trajectory"]["invocations"][0]["model_calls"].append(deepcopy(reference))
    record["ng_trajectory"]["model_calls"][2]["request"]["input"].append({"role": "assistant", "content": summary})
    record["ng_agent_observations"]["records"].append(
        {
            "kind": "context_compaction",
            "invocation_id": "invocation",
            "trigger": "automatic",
            "outcome": "completed",
            "summary": summary,
            "model_calls": [reference],
        }
    )
    record["response"]["usage"] = {"input_tokens": 27, "output_tokens": 12, "total_tokens": 39}
    witness["attempts"].insert(1, {"request": {}, "response": response, "status_code": 200})
    witness["compaction"] = {"summary": summary, "response_ids": ["summary-response"]}
    return record, witness


def checks(record, witness, *, expected=True):
    return {c["id"]: c for c in inspect_compaction(witness, record, expected=expected, policy_responses=2)}


def test_completed_compaction_retains_helper_without_policy_turn(compaction_evidence):
    record, witness = compaction_evidence
    result = checks(record, witness)
    assert len(result) == 6
    assert {c["status"] for c in result.values()} == {"pass"}
    assert {c["tier"] for c in result.values()} == {"P1"}


@pytest.mark.parametrize("reverse_owners", [False, True])
def test_ground_truth_separates_turn_exclusion_from_conflicting_ownership(compaction_evidence, reverse_owners):
    record, witness = compaction_evidence
    invocations = record["ng_trajectory"]["invocations"]
    invocations.append({"invocation_id": "other", "model_calls": [{"model_call_id": "summary-call"}]})
    if reverse_owners:
        invocations.reverse()

    result = checks(record, witness)

    assert result["compaction.helper_references"]["status"] == "fail"
    # The independent helper IDs still establish that the saved turns exclude
    # compaction, even though the retained ownership evidence is contradictory.
    assert result["compaction.policy_turns"]["status"] == "pass"
    assert result["compaction.usage_retained"]["status"] == "pass"


@pytest.mark.parametrize(
    "mutation,failed",
    [
        ("event_missing", "event"),
        ("event_duplicate", "event"),
        ("summary_changed", "summary_reuse"),
        ("summary_not_reused", "summary_reuse"),
        ("helper_reference_changed", "helper_references"),
        ("helper_call_missing", "helper_references"),
        ("wrong_owner", "helper_references"),
        ("helper_is_turn", "policy_turns"),
        ("policy_turn_missing", "policy_turns"),
        ("helper_usage_omitted", "usage_retained"),
        ("helper_usage_doubled", "usage_retained"),
        ("invalid_event", "schema"),
    ],
)
def test_compaction_rejects_saved_artifact_regressions(compaction_evidence, mutation, failed):
    record, witness = compaction_evidence
    events = record["ng_agent_observations"]["records"]
    event = events[-1]
    trajectory = record["ng_trajectory"]
    if mutation == "event_missing":
        events.pop()
    elif mutation == "event_duplicate":
        events.append(deepcopy(event))
    elif mutation == "summary_changed":
        event["summary"] = "different"
    elif mutation == "summary_not_reused":
        trajectory["model_calls"][2]["request"]["input"].pop()
    elif mutation == "helper_reference_changed":
        event["model_calls"][0]["model_call_id"] = "missing"
    elif mutation == "helper_call_missing":
        trajectory["model_calls"].pop(1)
    elif mutation == "wrong_owner":
        event["invocation_id"] = "other"
    elif mutation == "helper_is_turn":
        trajectory["turns"].append({"invocation_id": "invocation", "model_calls": deepcopy(event["model_calls"])})
    elif mutation == "policy_turn_missing":
        trajectory["turns"].pop()
    elif mutation == "helper_usage_omitted":
        record["response"]["usage"] = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
    elif mutation == "helper_usage_doubled":
        record["response"]["usage"] = {"input_tokens": 34, "output_tokens": 14, "total_tokens": 48}
    elif mutation == "invalid_event":
        event["tokens_before"] = -1
    assert checks(record, witness)[f"compaction.{failed}"]["status"] == "fail"


@pytest.mark.parametrize("missing", ["witness", "record", "records", "helper_witness"])
def test_missing_evidence_is_failure_not_non_applicability(compaction_evidence, missing):
    record, witness = compaction_evidence
    if missing == "witness":
        witness = None
    elif missing == "record":
        record = None
    elif missing == "records":
        record["ng_agent_observations"].pop("records")
    else:
        witness.pop("compaction")
    result = checks(record, witness)
    assert result["compaction.event"]["status"] == "fail"
    assert all(c["status"] != "not_applicable" for c in result.values())


def test_no_compaction_control_requires_absence(compaction_evidence):
    record, witness = compaction_evidence
    assert checks(record, witness, expected=False)["compaction.event"]["status"] == "fail"
    record["ng_agent_observations"]["records"].pop()
    record["ng_trajectory"]["model_calls"].pop(1)
    record["ng_trajectory"]["invocations"][0]["model_calls"].pop()
    record["response"]["usage"] = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
    witness["attempts"].pop(1)
    witness["compaction"]["response_ids"] = []
    result = checks(record, witness, expected=False)
    assert result["compaction.summary_reuse"]["status"] == "not_applicable"
    assert all(c["status"] == "pass" for name, c in result.items() if name != "compaction.summary_reuse")


def test_compaction_checks_do_not_fall_back_to_duplicate_artifacts(compaction_evidence):
    record, witness = compaction_evidence
    record["ng_model_call_capture"] = {"calls": []}
    record["ng_trajectory"]["invocations"][0]["conversation"] = []
    record["ng_agent_observations"]["records"][0]["model_calls"] = []
    assert {c["status"] for c in checks(record, witness).values()} == {"pass"}


@pytest.mark.parametrize("truncate", [False, True])
def test_compaction_preserves_complete_summary_inside_runtime_wrapper(compaction_evidence, truncate):
    record, witness = compaction_evidence
    summary = witness["compaction"]["summary"]
    wrapped = "Continue from this summary:\n" + (summary[:-1] if truncate else summary) + "\nResume work."
    record["ng_agent_observations"]["records"][-1]["summary"] = wrapped
    record["ng_trajectory"]["model_calls"][2]["request"]["input"][-1]["content"] = wrapped
    assert checks(record, witness)["compaction.summary_reuse"]["status"] == ("fail" if truncate else "pass")


def test_compaction_usage_includes_messages_cache_reads_and_writes(compaction_evidence):
    record, witness = compaction_evidence
    witness["attempts"][1]["response"]["usage"].update(cache_read_input_tokens=3, cache_creation_input_tokens=2)
    witness["attempts"][1]["response"]["usage"].pop("total_tokens")
    record["response"]["usage"].update(input_tokens=32, total_tokens=44)
    assert checks(record, witness)["compaction.usage_retained"]["status"] == "pass"


@pytest.mark.parametrize("mutation", [None, "missing_error", "invented_zero", "missing_gap", "reported_usage"])
def test_overflow_recovery_retains_error_and_measured_usage(compaction_evidence, mutation):
    record, witness = compaction_evidence
    error = {"error": {"code": "context_length_exceeded"}}
    attempt = {"request": {"input": "too large"}, "status_code": 400, "response": error}
    witness["attempts"].insert(1, attempt)
    failed = {
        "model_call_id": "overflow",
        "request": attempt["request"],
        "response": error,
        "response_metadata": {"status_code": 400},
        "token_stats": {},
    }
    record["ng_trajectory"]["model_calls"].append(failed)
    record["ng_trajectory"]["invocations"][0]["model_calls"].append({"model_call_id": "overflow"})
    record["ng_agent_observations"]["gaps"] = [{"code": "model_call_usage_unavailable"}]
    if mutation == "missing_error":
        record["ng_trajectory"]["model_calls"].pop()
    elif mutation == "invented_zero":
        failed["token_stats"]["tokens_in"] = 0
    elif mutation == "missing_gap":
        record["ng_agent_observations"]["gaps"] = []
    elif mutation == "reported_usage":
        error["usage"] = {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
        failed["token_stats"].update(tokens_in=3, tokens_out=1, tokens_total=4)
        record["response"]["usage"].update(input_tokens=30, output_tokens=13, total_tokens=43)
    result = checks(record, witness)
    if mutation in (None, "reported_usage"):
        assert {c["status"] for c in result.values()} == {"pass"}
    elif mutation == "missing_gap":
        assert result["compaction.usage_retained"]["status"] == "fail"
    else:
        assert result["compaction.failed_calls_retained"]["status"] == "fail"


def test_successful_compaction_call_cannot_omit_usage(compaction_evidence):
    record, witness = compaction_evidence
    witness["attempts"][1]["response"].pop("usage")
    record["ng_agent_observations"]["gaps"] = [{"code": "model_call_usage_unavailable"}]
    assert checks(record, witness)["compaction.usage_retained"]["status"] == "fail"
