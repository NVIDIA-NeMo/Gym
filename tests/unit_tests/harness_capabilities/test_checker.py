# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutation tests: schema-shaped lies must not become passing artifacts."""

import json

import pytest

from nemo_gym.harness_capabilities.checker import inspect_record
from nemo_gym.harness_capabilities.cli import main
from tests.unit_tests.harness_capabilities.synthetic import evidence_record


@pytest.fixture
def record():
    return evidence_record()


def verdict(record, capability):
    return inspect_record(record)["evidence"][capability]["verdict"]


def test_retained_model_evidence_and_tools_pass(record):
    result = inspect_record(record)
    assert all(result["evidence"][c]["verdict"] == "fulfilled" for c in (f"TE-{i}" for i in range(1, 10))), result[
        "findings"
    ]
    assert result["verdict"] == "fulfilled"
    assert all(c["status"] != "fail" for c in result["checks"])
    assert result["is_behavioral_qualification"] is False


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


def test_payload_removal_fails_even_with_complete_conversation(record):
    record["ng_trajectory"]["model_calls"][0]["request"] = None
    assert verdict(record, "TE-4") == "not_fulfilled"
    assert verdict(record, "TE-7") == "not_fulfilled"


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


def test_missing_p0_steps_or_tools_cannot_pass(record):
    record["ng_trajectory"]["turns"] = []
    record["ng_trajectory"]["tool_calls"] = []
    record["ng_agent_observations"]["records"] = [
        r for r in record["ng_agent_observations"]["records"] if r["kind"] != "tool_call"
    ]
    result = inspect_record(record)
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
    result = inspect_record(record)
    assert result["evidence"]["TE-2"]["verdict"] == "fulfilled"
    assert result["token_availability"]["cached_tokens"]["available"] == 0
    assert result["token_availability"]["reasoning_tokens"]["available"] == 0


def test_terminal_response_metadata_required(record):
    for c in record["ng_model_call_capture"]["calls"]:
        c["response_status"] = c["finish_reason"] = None
    for c in record["ng_trajectory"]["model_calls"]:
        c["response_metadata"]["response_status"] = c["response_metadata"]["finish_reason"] = None
        c["response"].pop("status", None)
        c["response"].pop("incomplete_details", None)
    assert verdict(record, "TE-1") == "not_fulfilled"


def test_tool_record_does_not_require_p1_clocks(record):
    for tool in record["ng_trajectory"]["tool_calls"] + record["ng_agent_observations"]["records"]:
        if tool.get("kind") == "tool_call":
            for key in ("started_at", "completed_at", "duration_ms", "timing_source"):
                tool.pop(key, None)
    assert verdict(record, "TE-5") == "fulfilled"


def test_missing_call_ownership_blocks_p0():
    record = evidence_record()
    for inv in record["ng_trajectory"]["invocations"]:
        if inv["kind"] == "agent_invocation":
            inv["model_calls"] = []
    result = inspect_record(record)
    assert result["evidence"]["TE-8"]["verdict"] == "not_fulfilled"
    assert result["evidence"]["TE-9"]["verdict"] == "fulfilled"
    assert result["verdict"] == "not_fulfilled"
    check = next(c for c in result["checks"] if c["id"] == "ownership.call_owner")
    assert check["tier"] == "P0" and check["status"] == "fail"


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


@pytest.mark.parametrize("field", ["mask_sample", "evaluation_completed"])
@pytest.mark.parametrize("value", [None, 0, "false"])
def test_verification_flags_must_be_boolean_when_supplied(record, field, value):
    record[field] = value
    assert verdict(record, "TE-6") == "not_fulfilled"


def test_unmasked_diagnostic_metadata_does_not_invalidate_reward(record):
    record.update(mask_sample=False, failure_kind="judge_failed")
    assert verdict(record, "TE-6") == "fulfilled"


def test_binary_resolution_does_not_replace_reward(record):
    record.pop("reward", None)
    for turn in record["ng_trajectory"]["turns"]:
        turn["resolved"] = True
    assert verdict(record, "TE-6") == "not_fulfilled"


def test_canonical_only_delivery_does_not_require_capture_middleware():
    record = evidence_record()
    del record["ng_model_call_capture"]
    assert inspect_record(record)["verdict"] == "fulfilled"
    for call in record["ng_trajectory"]["model_calls"]:
        call["started_at"] = call["completed_at"] = call["duration_ms"] = None
    result = inspect_record(record)
    assert result["verdict"] == "not_fulfilled"
    assert any(f["assertion"] == "calls.timing" for f in result["findings"])


def test_missing_prior_response_history_fails(record):
    record["ng_trajectory"]["model_calls"][0]["request"]["previous_response_id"] = "unretained-server-history"
    assert verdict(record, "TE-4") == "not_fulfilled"


def test_failed_response_without_response_id_preserves_status(record):
    capture = record["ng_model_call_capture"]["calls"][0]
    capture.update(response_id=None, response_status="failed")
    call = record["ng_trajectory"]["model_calls"][0]
    call["response_metadata"].update(response_id=None, response_status="failed")
    call["response"].update(id=None, status="failed", error={"code": "server_error"})
    assert verdict(record, "TE-1") == "not_fulfilled"
    capture["error_category"] = call["response_metadata"]["error_category"] = "server_error"
    assert verdict(record, "TE-1") == "fulfilled"


def test_turn_resolution_is_required_by_te3(record):
    from nemo_gym.rollout_observability import TrajectoryTurn

    turn = record["ng_trajectory"]["turns"][0]
    del turn["resolved"]
    TrajectoryTurn.model_validate(turn, strict=True)
    result = inspect_record(record)
    assert result["evidence"]["TE-3"]["verdict"] == "not_fulfilled"
    assert any(f["assertion"] == "steps.resolution" for f in result["findings"])


@pytest.mark.parametrize("missing", [False, True])
def test_turns_do_not_need_copies_of_model_content(record, missing):
    for turn in record["ng_trajectory"]["turns"]:
        for field in ("question", "answer", "reasoning_content"):
            if missing:
                turn.pop(field, None)
            else:
                turn[field] = None
    result = inspect_record(record)
    for te in ("TE-3", "TE-4", "TE-7", "TE-9"):
        assert result["evidence"][te]["verdict"] == "fulfilled", result["findings"]


@pytest.mark.parametrize("field", ["tool_name", "status"])
def test_optional_tool_fields_remain_required_by_te5(record, field):
    from nemo_gym.rollout_observability import TrajectoryToolCall

    tool = record["ng_trajectory"]["tool_calls"][0]
    del tool[field]
    TrajectoryToolCall.model_validate(tool, strict=True)
    assert verdict(record, "TE-5") == "not_fulfilled"


def test_invocations_can_be_supplied_at_trajectory_path(record):
    del record["ng_agent_observations"]
    assert inspect_record(record)["verdict"] == "fulfilled"
