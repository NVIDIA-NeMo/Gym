# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for authority, dependency isolation and retained relationships."""

from copy import deepcopy

import pytest

from nemo_gym.harness_capabilities.checker import TOKEN_FIELDS, EvidenceScope, inspect_record
from tests.unit_tests.harness_capabilities.synthetic import evidence_record


def checks(record, **kwargs):
    return {c["id"]: c for c in inspect_record(record, **kwargs)["checks"]}


def test_bad_step_does_not_erase_valid_model_evidence():
    record = evidence_record()
    record["ng_trajectory"]["turns"][0]["resolved"] = "true"
    result = checks(record)
    assert result["steps.resolution"]["status"] == "fail"
    assert result["calls.outcome"]["status"] == "pass"
    assert result["tokens.prompt_tokens"]["status"] == "pass"
    assert result["steps.call_target"]["status"] == "pass"


def test_shared_prerequisite_runs_once_and_blocks_only_dependents():
    record = evidence_record()
    record["ng_trajectory"]["model_calls"] = []
    result = checks(record)
    assert result["model_calls.present"]["status"] == "fail"
    assert result["calls.outcome"]["status"] == "fail"
    assert result["tokens.prompt_tokens"]["blocked_by"] == ["model_calls.present"]
    assert result["steps.resolution"]["status"] == "pass"
    assert result["tools.output"]["status"] == "pass"
    assert not any(k.startswith("TE-") for k in result)


@pytest.mark.parametrize("duplicate", ["ng_model_call_capture", "ng_agent_observations"])
def test_malformed_unused_copy_cannot_invalidate_authoritative_evidence(duplicate):
    record = evidence_record()
    record[duplicate] = {"calls": [False], "records": [False]}
    assert inspect_record(record)["verdict"] == "fulfilled"


def test_no_fallback_from_observations_or_capture():
    record = evidence_record()
    record["ng_trajectory"].pop("model_calls")
    record["ng_trajectory"].pop("invocations")
    result = checks(record)
    assert result["model_calls.present"]["status"] == "fail"
    assert result["invocations.present"]["status"] == "fail"
    assert result["steps.invocation_target"]["status"] == "fail"


@pytest.mark.parametrize("field", TOKEN_FIELDS)
@pytest.mark.parametrize("value", [-1, True, "10", 1.5])
def test_bad_normalized_count_is_rejected(field, value):
    record = evidence_record()
    record["ng_trajectory"]["model_calls"][0]["token_stats"][field] = value
    assert checks(record)["tokens." + field]["status"] == "fail"


@pytest.mark.parametrize("value", [None, 0, 100])
def test_optional_counts_are_not_compared_with_raw_provider_usage(value):
    record = evidence_record()
    record["ng_trajectory"]["model_calls"][0]["token_stats"]["prompt_tokens"] = value
    result = inspect_record(record)
    assert result["evidence"]["TE-2"]["verdict"] == "fulfilled"
    assert result["token_availability"]["prompt_tokens"]["available"] == (1 if value is None else 2)


def test_gym_owned_copy_consistency_and_clock_order_are_not_rechecked():
    record = evidence_record()
    record["ng_trajectory"]["model_calls"][0]["completed_at"] = 0
    record["ng_trajectory"]["turns"][0]["rollout_id"] = "ignored-copy"
    assert inspect_record(record)["verdict"] == "fulfilled"


def test_missing_rollout_fails_applicable_checks():
    result = checks(None)
    assert {c["status"] for c in result.values()} <= {"fail", "not_applicable"}
    assert result["calls.outcome"]["status"] == "fail"


def test_failure_artifact_is_not_silently_used_as_a_normal_rollout():
    result = checks({"failure": "no rollout"})
    assert result["model_calls.present"]["status"] == "fail"
    assert result["calls.outcome"]["status"] == "fail"


def test_retry_attempt_and_compaction_helper_accounting():
    record = evidence_record()
    trajectory = record["ng_trajectory"]
    call = deepcopy(trajectory["model_calls"][0])
    call["model_call_id"] = "auxiliary"
    call["response_metadata"]["response_id"] = "aux-response"
    trajectory["model_calls"].append(call)
    trajectory["invocations"][0]["model_calls"].append({"model_call_id": "auxiliary"})
    assert checks(record)["steps.attempt_accounting"]["status"] == "fail"
    # Same attempt as retry: must be assigned to the existing step.
    trajectory["turns"][0]["model_calls"].append({"model_call_id": "auxiliary"})
    assert checks(record)["steps.attempt_accounting"]["status"] == "pass"
    # A declared compaction helper has a separate purpose and must not be counted as a policy attempt.
    record["ng_agent_observations"]["records"].append(
        {"kind": "context_compaction", "model_calls": [{"model_call_id": "auxiliary"}]}
    )
    assert checks(record)["steps.attempt_accounting"]["status"] == "fail"
    trajectory["turns"][0]["model_calls"].pop()
    assert checks(record)["steps.attempt_accounting"]["status"] == "pass"
    trajectory["gaps"].append({"code": "turn_model_call_scope_incomplete"})
    assert checks(record)["steps.accounting_gap"]["status"] == "fail"


def test_ambiguous_reference_is_a_relationship_failure_not_a_global_schema_failure():
    record = evidence_record()
    record["ng_trajectory"]["model_calls"].append(deepcopy(record["ng_trajectory"]["model_calls"][0]))
    result = checks(record)
    assert result["calls.identity"]["status"] == "pass"
    assert result["ownership.call_target"]["status"] == "fail"
    assert result["steps.call_target"]["status"] == "fail"
    assert result["ownership.call_owner"]["status"] == "fail"


@pytest.mark.parametrize("value", [None, False, [], "wrong", [None]])
def test_malformed_collection_does_not_crash_or_poison_unrelated_checks(value):
    record = evidence_record()
    record["ng_trajectory"]["turns"] = value
    result = checks(record)
    assert result["turns.present"]["status"] == "fail"
    assert result["steps.number"]["status"] == "fail"
    assert result["calls.outcome"]["status"] == "pass"


def test_explicit_applicability_does_not_depend_on_empty_collection():
    record = evidence_record()
    record["ng_trajectory"]["tool_calls"] = []
    record["ng_agent_observations"]["records"] = [
        r for r in record["ng_agent_observations"]["records"] if r["kind"] != "tool_call"
    ]
    result = inspect_record(record)
    assert result["evidence"]["TE-5"]["verdict"] == "not_fulfilled"
    assert next(c for c in result["checks"] if c["id"] == "tool_calls.present")["status"] == "fail"
    assert checks(record, scope=EvidenceScope(tools=False))["tool_calls.present"]["status"] == "not_applicable"


def test_parent_reference_is_resolved_without_global_native_validation():
    record = evidence_record()
    record["ng_trajectory"]["invocations"][0]["parent_invocation_id"] = "invocation"
    result = checks(record)
    assert result["invocations.parent"]["status"] == "fail"
    assert result["calls.outcome"]["status"] == "pass"


def test_missing_step_call_references_block_p0():
    record = evidence_record()
    for turn in record["ng_trajectory"]["turns"]:
        turn.pop("model_calls")
    result = inspect_record(record)
    by_id = {c["id"]: c for c in result["checks"]}
    assert result["verdict"] == "not_fulfilled"
    assert result["evidence"]["TE-9"]["verdict"] == "not_fulfilled"
    assert by_id["steps.references"]["tier"] == "P0"
    assert by_id["steps.references"]["status"] == "fail"
    assert by_id["steps.call_target"]["blocked_by"] == ["steps.references"]
    assert by_id["steps.number"]["tier"] == "P0"
    assert by_id["steps.number"]["status"] == "pass"


def test_missing_invocations_still_block_p0_step_and_tool_relationships():
    record = evidence_record()
    record["ng_trajectory"].pop("invocations")
    result = inspect_record(record)
    by_id = {c["id"]: c for c in result["checks"]}
    assert result["verdict"] == "not_fulfilled"
    for check_id in (
        "invocations.present",
        "invocations.identity",
        "steps.invocation_target",
        "tools.invocation_target",
    ):
        assert by_id[check_id]["tier"] == "P0"
        assert by_id[check_id]["status"] == "fail"
