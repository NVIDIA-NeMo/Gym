# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutation tests: schema-shaped lies must not become passing artifacts."""

import copy
import json

import pytest

from nemo_gym.harness_capabilities.checker import EvidenceScope, inspect_record
from nemo_gym.harness_capabilities.cli import inspect_bundle, main
from nemo_gym.harness_capabilities.reader import hydrate_record
from tests.unit_tests.harness_capabilities.synthetic import evidence_record


@pytest.fixture
def record():
    return evidence_record()


def verdict(record, capability):
    return inspect_record(hydrate_record(record))["evidence"][capability]["verdict"]


def test_retained_model_evidence_and_tools_pass(record):
    result = inspect_record(hydrate_record(record))
    assert all(result["evidence"][c]["verdict"] == "fulfilled" for c in (f"TE-{i}" for i in range(1, 10))), result[
        "findings"
    ]
    assert result["verdict"] == "fulfilled"
    assert result["is_behavioral_qualification"] is False


@pytest.mark.parametrize(
    "field",
    [
        "tokens_in",
        "tokens_out",
        "tokens_reasoning",
        "tokens_total",
        "cached_tokens",
    ],
)
@pytest.mark.parametrize("value", [None, -1, True, 1.5])
def test_bad_token_values_fail(record, field, value):
    record["ng_model_call_capture"]["calls"][0][field] = value
    assert verdict(record, "TE-2") == "not_fulfilled"


def test_omitted_usage_cannot_become_zero(record):
    call = record["ng_model_call_capture"]["calls"][0]
    call["cached_tokens"] = 0
    del record["ng_trajectory"]["model_calls"][0]["response"]["usage"]["input_tokens_details"]["cached_tokens"]
    assert verdict(record, "TE-2") == "not_fulfilled"


def test_legitimate_zero_is_available(record):
    record["ng_model_call_capture"]["calls"][0]["cached_tokens"] = 0
    record["ng_trajectory"]["model_calls"][0]["token_stats"]["cached_tokens"] = 0
    record["ng_trajectory"]["model_calls"][0]["response"]["usage"]["input_tokens_details"]["cached_tokens"] = 0
    assert verdict(record, "TE-2") == "fulfilled"


def test_missing_response_usage_cannot_be_invented(record):
    record["ng_trajectory"]["model_calls"][0]["response"]["usage"] = None
    assert verdict(record, "TE-2") == "not_fulfilled"


def test_external_media_is_not_a_complete_payload(record):
    record["ng_trajectory"]["model_calls"][0]["request"]["input"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "https://example.invalid/missing.png",
                }
            ],
        }
    ]
    assert verdict(record, "TE-4") == "not_fulfilled"
    assert verdict(record, "TE-7") == "not_fulfilled"


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "wrong_session", "bad_reference", "cycle"],
)
def test_ownership_is_not_guessed(record, mutation):
    bundle = record["ng_agent_observations"]
    invocation = next(r for r in bundle["records"] if r["kind"] == "agent_invocation")
    if mutation == "missing":
        invocation["model_calls"] = []
    elif mutation == "duplicate":
        invocation["model_calls"] *= 2
    elif mutation == "wrong_session":
        invocation["invocation_id"] = "other"
    elif mutation == "bad_reference":
        invocation["model_calls"][0]["response_id"] = "different"
    else:
        invocation["parent_invocation_id"] = invocation["invocation_id"]
    assert verdict(record, "TE-8") == "not_fulfilled"


def test_failed_attempt_without_response_id_still_requires_owner(record):
    failed = copy.deepcopy(record["ng_model_call_capture"]["calls"][0])
    failed.update(
        model_call_id="failure",
        response_id=None,
        status_code=None,
        error_category="timeout",
        response=None,
    )
    failed["request"] = {"input": []}
    record["ng_model_call_capture"]["calls"].append(failed)
    assert verdict(record, "TE-8") == "not_fulfilled"
    record["ng_agent_observations"]["records"][0]["model_calls"].append({"model_call_id": "failure"})
    assert verdict(record, "TE-8") == "fulfilled"
    assert verdict(record, "TE-7") == "fulfilled"


def test_payload_removal_fails_even_with_complete_conversation(record):
    record["ng_trajectory"]["model_calls"][0]["request"] = None
    assert verdict(record, "TE-4") == "not_fulfilled"
    assert verdict(record, "TE-7") == "not_fulfilled"


def test_schema_valid_duplicate_attempt_fails(record):
    record["ng_model_call_capture"]["calls"] *= 2
    assert verdict(record, "TE-1") == "not_fulfilled"


def test_nullable_provider_model_keeps_explicit_server_identity(record):
    call = record["ng_model_call_capture"]["calls"][0]
    call["model"] = None
    assert verdict(record, "TE-1") == "fulfilled"
    call["model_ref"] = None
    assert verdict(record, "TE-1") == "not_fulfilled"


def test_sidecar_does_not_launder_payload_conflict(record, tmp_path):
    call = record["ng_model_call_capture"]["calls"][0]
    sidecar = {**call, "request": {"input": "different"}}
    (tmp_path / "0-0.capture.jsonl").write_text(json.dumps(sidecar) + "\n")
    hydrated = hydrate_record(record, capture_dir=tmp_path)
    assert inspect_record(hydrated)["evidence"]["TE-1"]["verdict"] == "not_fulfilled"


def test_tool_request_is_not_execution(record):
    record["ng_agent_observations"]["records"] = [
        r for r in record["ng_agent_observations"]["records"] if r["kind"] != "tool_call"
    ]
    record["ng_trajectory"]["tool_calls"] = []
    assert verdict(record, "TE-5") == "not_fulfilled"


def test_cli_atomic_replay_and_exit_codes(record, tmp_path):
    bundle = tmp_path / "input.jsonl"
    bundle.write_text(json.dumps(record) + "\n")
    output = tmp_path / "reports"
    args = ["inspect", "--bundle", str(bundle), "--output", str(output)]
    assert main(args) == 0
    assert main(args) == 0
    assert len(list(output.iterdir())) == 1
    report = next(output.glob("*/evidence_summary.json"))
    summary = json.loads(report.read_text())
    assert summary["checker_status"] == "completed"
    assert summary["is_behavioral_qualification"] is False
    record["ng_trajectory"]["model_calls"][0]["request"] = None
    bundle.write_text(json.dumps(record) + "\n")
    assert main(args) == 1
    assert len(list(output.iterdir())) == 2
    bundle.write_text('{"private prompt":')
    assert main(args) == 2
    assert len(list(output.iterdir())) == 2
    assert report.is_file()


def test_empty_and_duplicate_rollouts_fail(record, tmp_path):
    bundle = tmp_path / "input.jsonl"
    bundle.write_text("")
    with pytest.raises(ValueError, match="no rollout records"):
        inspect_bundle(bundle, output=tmp_path / "out", profile="gym-p0/v1")
    bundle.write_text((json.dumps(record) + "\n") * 2)
    _, report = inspect_bundle(bundle, output=tmp_path / "out", profile="gym-p0/v1")
    assert report["verdict"] == "not_fulfilled"


def test_missing_p0_steps_or_tools_cannot_pass(record):
    record["ng_trajectory"]["turns"] = []
    record["ng_trajectory"]["tool_calls"] = []
    record["ng_agent_observations"]["records"] = [
        r for r in record["ng_agent_observations"]["records"] if r["kind"] != "tool_call"
    ]
    result = inspect_record(hydrate_record(record))
    assert result["verdict"] == "not_fulfilled"
    assert result["evidence"]["TE-3"]["verdict"] == "not_fulfilled"
    assert result["evidence"]["TE-5"]["verdict"] == "not_fulfilled"


def test_provider_omission_is_preserved_and_reported(record):
    for c in record["ng_model_call_capture"]["calls"]:
        c["cached_tokens"] = c["tokens_reasoning"] = None
    for c in record["ng_trajectory"]["model_calls"]:
        c["token_stats"]["cached_tokens"] = c["token_stats"]["reasoning_tokens"] = None
        c["response"]["usage"].pop("input_tokens_details", None)
        c["response"]["usage"].pop("output_tokens_details", None)
    result = inspect_record(hydrate_record(record))
    assert result["evidence"]["TE-2"]["verdict"] == "fulfilled"
    assert result["token_availability"]["cached_tokens"]["available"] == 0
    assert result["token_availability"]["tokens_reasoning"]["available"] == 0


@pytest.mark.parametrize("field", ["response_status", "finish_reason"])
def test_terminal_metadata_conflict_fails(record, field):
    record["ng_trajectory"]["model_calls"][0]["response_metadata"][field] = "contradiction"
    assert verdict(record, "TE-1") == "not_fulfilled"


def test_terminal_response_metadata_required(record):
    for c in record["ng_model_call_capture"]["calls"]:
        c["response_status"] = c["finish_reason"] = None
    for c in record["ng_trajectory"]["model_calls"]:
        c["response_metadata"]["response_status"] = c["response_metadata"]["finish_reason"] = None
        c["response"].pop("status", None)
        c["response"].pop("incomplete_details", None)
    assert verdict(record, "TE-1") == "not_fulfilled"


def test_truthful_failure_is_evidence_not_a_health_verdict(record):
    c = record["ng_model_call_capture"]["calls"][0]
    c.update(status_code=503, response_status=None, finish_reason=None)
    t = record["ng_trajectory"]["model_calls"][0]
    t["response_metadata"].update(status_code=503, response_status=None, finish_reason=None)
    # Preserve usage reported on the failed attempt, as well as its error body.
    t["response"] = {"error": {"message": "failed"}, "usage": t["response"]["usage"]}
    assert verdict(record, "TE-1") == "fulfilled"
    assert verdict(record, "TE-2") == "fulfilled"
    c["tokens_in"] += 1
    t["token_stats"]["prompt_tokens"] = c["tokens_in"]
    assert verdict(record, "TE-2") == "not_fulfilled"


def test_tool_record_does_not_require_p1_clocks(record):
    for tool in record["ng_trajectory"]["tool_calls"] + record["ng_agent_observations"]["records"]:
        if tool.get("kind") == "tool_call":
            for key in ("started_at", "completed_at", "duration_ms", "timing_source"):
                tool.pop(key, None)
    assert verdict(record, "TE-5") == "fulfilled"


def test_partial_tool_execution_loss_fails(record):
    invocation = record["ng_agent_observations"]["records"][0]
    invocation["conversation"].append({"type": "function_call_output", "call_id": "lost-tool", "output": "ok"})
    assert verdict(record, "TE-5") == "not_fulfilled"


def test_na_requires_explicit_scope_and_rejects_contradictions(record):
    scope = EvidenceScope(tools=False, verifier=False, steps=False)
    assert inspect_record(hydrate_record(record), scope=scope)["evidence"]["TE-5"]["verdict"] == "not_fulfilled"
    record["ng_trajectory"]["turns"] = []
    record["ng_trajectory"]["tool_calls"] = []
    record["ng_agent_observations"]["records"] = [
        r for r in record["ng_agent_observations"]["records"] if r["kind"] != "tool_call"
    ]
    for inv in record["ng_agent_observations"]["records"]:
        inv["conversation"] = []
    record.pop("reward", None)
    record.pop("response", None)
    record["ng_trajectory"]["gaps"] = []
    result = inspect_record(hydrate_record(record), scope=scope)
    assert all(result["evidence"][key]["verdict"] == "not_applicable" for key in ("TE-3", "TE-5", "TE-6", "TE-9"))
    assert result["verdict"] == "fulfilled"


def test_step_join_can_qualify_without_invocation_refs():
    record = evidence_record()
    for inv in record["ng_agent_observations"]["records"]:
        if inv["kind"] == "agent_invocation":
            inv["model_calls"] = []
    result = inspect_record(hydrate_record(record))
    assert result["evidence"]["TE-8"]["verdict"] == "not_fulfilled"
    assert result["evidence"]["TE-9"]["verdict"] == "fulfilled"
    assert result["verdict"] == "fulfilled"


def test_duplicated_step_ref_fails_without_closure_requirement():
    record = evidence_record()
    assert verdict(record, "TE-9") == "fulfilled"
    record["ng_trajectory"]["turns"][0]["model_calls"] *= 2
    assert verdict(record, "TE-9") == "not_fulfilled"


@pytest.mark.parametrize("reward", [0.0, 1.0, -0.5])
def test_masked_numeric_reward_is_valid_evidence(record, reward):
    record.update(mask_sample=True, failure_kind="judge_failed", failure_reason="judge unavailable", reward=reward)
    assert verdict(record, "TE-6") == "fulfilled"


@pytest.mark.asyncio
async def test_judge_failsafe_response_satisfies_te6(record):
    from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
    from nemo_gym.judge import JudgeError, judge_failsafe

    @judge_failsafe
    async def verify(body):
        raise JudgeError("judge unavailable")

    body = BaseVerifyRequest(
        responses_create_params={"input": "task"},
        response={
            "id": "response-1",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    )
    data = json.loads((await verify(body)).body)
    verified = BaseVerifyResponse.model_validate(data)
    assert verified.reward == 0.0 and verified.mask_sample is True
    record.update(data)
    assert verdict(record, "TE-6") == "fulfilled"


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("reward", [None, True, "0.0", float("nan"), float("inf")])
def test_verifier_reward_must_be_numeric_even_when_masked(record, masked, reward):
    record.update(mask_sample=masked, failure_kind="judge_failed", failure_reason="judge unavailable", reward=reward)
    assert verdict(record, "TE-6") == "not_fulfilled"


@pytest.mark.parametrize(
    "signal", [{"mask_sample": True}, {"evaluation_completed": False}, {"verification_error": "error"}]
)
@pytest.mark.parametrize("field", ["failure_kind", "failure_reason"])
@pytest.mark.parametrize("value", [None, "", " \t", False])
def test_unavailable_verification_requires_both_failure_fields(record, signal, field, value):
    record.update(signal, failure_kind="judge_failed", failure_reason="judge unavailable")
    assert verdict(record, "TE-6") == "fulfilled"
    if value is None:
        record.pop(field)
    else:
        record[field] = value
    result = inspect_record(hydrate_record(record))
    assert result["evidence"]["TE-6"]["verdict"] == "not_fulfilled"
    assert any(f["assertion"] == "verifier.error" and f["location"] == "record/" + field for f in result["findings"])


@pytest.mark.parametrize("field", ["mask_sample", "evaluation_completed"])
@pytest.mark.parametrize("value", [None, 0, "false"])
def test_verification_flags_must_be_boolean_when_supplied(record, field, value):
    record[field] = value
    assert verdict(record, "TE-6") == "not_fulfilled"


def test_unmasked_diagnostic_metadata_does_not_invalidate_reward(record):
    record.update(mask_sample=False, failure_kind="judge_failed")
    assert verdict(record, "TE-6") == "fulfilled"


def test_masked_result_cannot_be_declared_verifier_free(record):
    record.update(mask_sample=True, reward=None)
    result = inspect_record(hydrate_record(record), scope=EvidenceScope(verifier=False))
    assert result["evidence"]["TE-6"]["verdict"] == "not_fulfilled"


def test_binary_resolution_does_not_replace_reward(record):
    record.pop("reward", None)
    for turn in record["ng_trajectory"]["turns"]:
        turn["resolved"] = True
    assert verdict(record, "TE-6") == "not_fulfilled"


def test_canonical_only_delivery_does_not_require_capture_middleware():
    record = evidence_record()
    del record["ng_model_call_capture"]
    for call in record["ng_trajectory"]["model_calls"]:
        call["started_at"] = call["completed_at"] = call["duration_ms"] = None
    assert inspect_record(hydrate_record(record))["verdict"] == "fulfilled"


def test_missing_prior_response_history_fails(record):
    record["ng_trajectory"]["model_calls"][0]["request"]["previous_response_id"] = "unretained-server-history"
    assert verdict(record, "TE-4") == "not_fulfilled"


def test_failed_response_without_response_id_preserves_status(record):
    capture = record["ng_model_call_capture"]["calls"][0]
    capture.update(response_id=None, response_status="failed")
    call = record["ng_trajectory"]["model_calls"][0]
    call["response_metadata"].update(response_id=None, response_status="failed")
    call["response"].update(id=None, status="failed", error={"code": "server_error"})
    assert verdict(record, "TE-1") == "fulfilled"


def test_retry_and_compaction_attempts_have_distinct_accounting():
    record = evidence_record()
    record = hydrate_record(record)
    retry = copy.deepcopy(record["ng_model_call_capture"]["calls"][0])
    retry.update(model_call_id="retry", response_id="retry-response")
    record["ng_model_call_capture"]["calls"].append(retry)
    record["ng_agent_observations"]["records"][0]["model_calls"].append({"model_call_id": "retry"})
    # A retry does not create a new step; its own attempt ref belongs on the existing one.
    assert inspect_record(record)["evidence"]["TE-9"]["verdict"] == "not_fulfilled"
    record["ng_trajectory"]["turns"][0]["model_calls"].append({"model_call_id": "retry"})
    assert inspect_record(record)["evidence"]["TE-9"]["verdict"] == "fulfilled"
    helper = copy.deepcopy(retry)
    helper.update(model_call_id="helper", response_id="helper-response")
    record["ng_model_call_capture"]["calls"].append(helper)
    record["ng_agent_observations"]["records"][0]["model_calls"].append({"model_call_id": "helper"})
    record["ng_agent_observations"]["records"].append(
        {
            "kind": "context_compaction",
            "invocation_id": record["ng_trajectory"]["turns"][0]["invocation_id"],
            "model_calls": [{"model_call_id": "helper"}],
        }
    )
    assert inspect_record(record)["evidence"]["TE-9"]["verdict"] == "fulfilled"
    record["ng_trajectory"]["turns"][0]["model_calls"].append({"model_call_id": "helper"})
    assert inspect_record(record)["evidence"]["TE-9"]["verdict"] == "not_fulfilled"


def test_conflicting_run_reference_cannot_be_hidden_by_step_join():
    record = evidence_record()
    record["ng_agent_observations"]["records"][0]["model_calls"] = [{"model_call_id": "missing"}]
    assert inspect_record(hydrate_record(record))["verdict"] == "not_fulfilled"


def test_anthropic_cache_prompt_and_derived_total(record):
    call = record["ng_model_call_capture"]["calls"][0]
    trajectory = record["ng_trajectory"]["model_calls"][0]
    call.update(
        dialect="messages", tokens_in=15, tokens_out=5, tokens_total=20, tokens_reasoning=None, cached_tokens=3
    )
    trajectory["response_metadata"]["dialect"] = "messages"
    trajectory["token_stats"].update(
        prompt_tokens=15, completion_tokens=5, total_tokens=20, reasoning_tokens=None, cached_tokens=3
    )
    trajectory["response"]["usage"] = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 2,
    }
    assert verdict(record, "TE-2") == "fulfilled"
    call["tokens_in"] = trajectory["token_stats"]["prompt_tokens"] = 10
    assert verdict(record, "TE-2") == "not_fulfilled"


def test_boolean_provider_usage_cannot_equal_integer_count(record):
    call = record["ng_model_call_capture"]["calls"][0]
    trajectory = record["ng_trajectory"]["model_calls"][0]
    call["cached_tokens"] = 1
    trajectory["token_stats"]["cached_tokens"] = 1
    trajectory["response"]["usage"]["input_tokens_details"]["cached_tokens"] = True
    assert verdict(record, "TE-2") == "not_fulfilled"


def test_response_identity_must_match_payload(record):
    record["ng_model_call_capture"]["calls"][0]["response_id"] = "wrong"
    record["ng_trajectory"]["model_calls"][0]["response_metadata"]["response_id"] = "wrong"
    assert verdict(record, "TE-1") == "not_fulfilled"


def test_tool_output_conflict_fails(record):
    record["ng_trajectory"]["tool_calls"][0]["output"] = "different-from-model-visible-result"
    assert verdict(record, "TE-5") == "not_fulfilled"


@pytest.mark.parametrize(
    ("parent", "field", "index", "mutation", "error_path"),
    [
        ("ng_trajectory", "turns", 0, {"step_count": -1}, "/step_count"),
        ("ng_trajectory", "turns", 0, {"step_count": "1"}, "/step_count"),
        ("ng_trajectory", "turns", 0, {"unexpected": "private-value"}, "/unexpected"),
        ("ng_trajectory", "turns", 0, {"model_calls": [{}]}, "/model_calls/0"),
        ("ng_trajectory", "invocations", 0, {"status": "invalid"}, "/status"),
        ("ng_trajectory", "tool_calls", 0, {"duration_ms": -1}, "/duration_ms"),
        ("ng_trajectory", "tool_calls", 0, {"started_at": 2, "completed_at": 1}, ""),
        ("ng_trajectory", "model_calls", 0, {"response_metadata": []}, "/response_metadata"),
        ("ng_trajectory", "model_calls", 0, {"token_stats": {"prompt_tokens": -1}}, "/token_stats/prompt_tokens"),
        ("ng_model_call_capture", "calls", 0, {"call_index": "0"}, "/call_index"),
        ("ng_agent_observations", "records", 0, {"status": "invalid"}, "/status"),
        ("ng_agent_observations", "records", 1, {"timing_source": "invalid"}, "/timing_source"),
        ("ng_agent_observations", "records", 1, {"output": "private-value"}, "/output"),
    ],
)
def test_shared_models_reject_invalid_objects_at_exact_paths(record, parent, field, index, mutation, error_path):
    record[parent][field][index].update(mutation)
    result = inspect_record(hydrate_record(record))
    assert result["verdict"] == "not_fulfilled"
    assert any(
        finding["assertion"].startswith("model.")
        and finding["location"] == f"record/{parent}/{field}/{index}{error_path}"
        for finding in result["findings"]
    )
    assert "private-value" not in json.dumps(result)


@pytest.mark.parametrize(("parent", "field"), [("ng_trajectory", "turns"), ("ng_model_call_capture", "calls")])
@pytest.mark.parametrize("value", [None, {}, [None]])
def test_invalid_evidence_collections_are_reported(record, parent, field, value, tmp_path):
    record[parent][field] = value
    bundle = tmp_path / "invalid.jsonl"
    bundle.write_text(json.dumps(record) + "\n")
    assert main(["inspect", "--bundle", str(bundle), "--output", str(tmp_path / "report")]) == 1
    result = json.loads(next((tmp_path / "report").glob("*/evidence_results.jsonl")).read_text())
    assert any(f["location"].startswith(f"invalid.jsonl:1/{parent}/{field}") for f in result["findings"])


@pytest.mark.parametrize(
    ("parent", "field", "capability"),
    [("ng_trajectory", "turns", "TE-3"), ("ng_model_call_capture", "calls", "TE-1")],
)
def test_valid_objects_at_unrecognized_paths_do_not_count(record, parent, field, capability):
    record["somewhere_else"] = record[parent].pop(field)
    assert verdict(record, capability) == "not_fulfilled"


@pytest.mark.parametrize(("field", "assertion"), [("question", "turn.question"), ("resolved", "turn.resolved")])
def test_optional_turn_fields_are_required_by_te3(record, field, assertion):
    from nemo_gym.rollout_observability import TrajectoryTurn

    turn = record["ng_trajectory"]["turns"][0]
    del turn[field]
    TrajectoryTurn.model_validate(turn, strict=True)
    result = inspect_record(hydrate_record(record))
    assert result["evidence"]["TE-3"]["verdict"] == "not_fulfilled"
    assert any(f["assertion"] == assertion for f in result["findings"])


def test_null_question_is_insufficient_even_when_turn_model_valid(record):
    record["ng_trajectory"]["turns"][0]["question"] = None
    assert verdict(record, "TE-3") == "not_fulfilled"


@pytest.mark.parametrize("answer", [None, "answer", []])
@pytest.mark.parametrize("reasoning", [None, "reasoning"])
def test_te3_requires_answer_or_reasoning_beyond_model_validity(record, answer, reasoning):
    from nemo_gym.rollout_observability import TrajectoryTurn

    turn = record["ng_trajectory"]["turns"][0]
    turn.update(answer=answer, reasoning_content=reasoning)
    TrajectoryTurn.model_validate(turn, strict=True)
    expected = "fulfilled" if answer is not None or reasoning is not None else "not_fulfilled"
    assert verdict(record, "TE-3") == expected


@pytest.mark.parametrize("field", ["tool_name", "status"])
def test_optional_tool_fields_remain_required_by_te5(record, field):
    from nemo_gym.rollout_observability import TrajectoryToolCall

    tool = record["ng_trajectory"]["tool_calls"][0]
    del tool[field]
    TrajectoryToolCall.model_validate(tool, strict=True)
    assert verdict(record, "TE-5") == "not_fulfilled"


def test_invocation_and_tool_fallbacks_use_observation_paths(record):
    record["ng_trajectory"].pop("invocations")
    record["ng_trajectory"].pop("tool_calls")
    assert inspect_record(hydrate_record(record))["verdict"] == "fulfilled"
    record["ng_agent_observations"]["records"][1]["status"] = "unknown"
    result = inspect_record(hydrate_record(record))
    assert any(
        f["assertion"] == "tool.terminal" and f["location"] == "record/ng_agent_observations/records/1/status"
        for f in result["findings"]
    )


def test_invocations_can_be_supplied_at_trajectory_path(record):
    del record["ng_agent_observations"]
    assert inspect_record(hydrate_record(record))["verdict"] == "fulfilled"


def test_model_call_reference_rejects_extra_fields(record):
    record["ng_agent_observations"]["records"][0]["model_calls"][0]["unexpected"] = True
    result = inspect_record(hydrate_record(record))
    assert any(
        f["assertion"] == "model.extra_forbidden"
        and f["location"] == "record/ng_agent_observations/records/0/model_calls/0/unexpected"
        for f in result["findings"]
    )


@pytest.mark.parametrize(
    ("parent", "collection", "field"),
    [("ng_trajectory", "turns", "step_count"), ("ng_model_call_capture", "calls", "call_index")],
)
def test_required_model_fields_cannot_be_omitted(record, parent, collection, field):
    del record[parent][collection][0][field]
    result = inspect_record(hydrate_record(record))
    assert any(
        f["assertion"] == "model.missing" and f["location"] == f"record/{parent}/{collection}/0/{field}"
        for f in result["findings"]
    )


@pytest.mark.parametrize(
    ("field", "value", "assertion"),
    [("dialect", "unknown", "dialect.supported"), ("status_code", 600, "status.range")],
)
def test_model_valid_call_still_requires_te1_semantics(record, field, value, assertion):
    from nemo_gym.base_responses_api_model import ModelCallRecord

    call = record["ng_model_call_capture"]["calls"][0]
    call[field] = value
    record["ng_trajectory"]["model_calls"][0]["response_metadata"][field] = value
    ModelCallRecord.model_validate(call, strict=True)
    result = inspect_record(hydrate_record(record))
    assert any(f["evidence"] == "TE-1" and f["assertion"] == assertion for f in result["findings"])
