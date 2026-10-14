# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check core behavioral comparisons without launching or importing harnesses."""

import copy
import json

import pytest

from nemo_gym.harness_capabilities.behavior import inspect_behavior, model_checks, tool_checks
from tests.unit_tests.harness_capabilities.synthetic import evidence_and_witness


def fingerprint(request: dict, status: int, response: dict) -> str:
    # The core accepts an injected comparison key; protocol normalization belongs to the runner.
    return json.dumps([request, status, response], sort_keys=True)


@pytest.fixture
def behavior_episode() -> tuple[dict, dict]:
    return evidence_and_witness()


def failed_model_checks(record: dict, attempts: list[dict]) -> set[str]:
    return {c["id"] for c in model_checks(record, attempts, fingerprint=fingerprint) if c["status"] == "fail"}


@pytest.mark.parametrize(
    "mutation", ["missing_calls", "response_id", "finish_reason", "prompt_tokens", "missing_tokens", "request_order"]
)
def test_canonical_behavioral_checks_use_the_independent_witness(behavior_episode, mutation):
    raw, witness = behavior_episode
    assert not failed_model_checks(raw, witness["attempts"])
    call = raw["ng_trajectory"]["model_calls"][1]
    if mutation == "missing_calls":
        del raw["ng_trajectory"]["model_calls"]
    elif mutation in ("response_id", "finish_reason"):
        call["response_metadata"][mutation] = "fabricated"
    elif mutation == "prompt_tokens":
        call["token_stats"]["prompt_tokens"] = 999
    elif mutation == "missing_tokens":
        del call["token_stats"]
    else:
        call["request"]["input"].reverse()
    issues = failed_model_checks(raw, witness["attempts"])
    assert issues
    assert issues <= {"model.saved_exchanges", "model.response_metadata", "model.token_counts"}


def test_canonical_error_response_cannot_invent_an_id():
    request = {"input": []}
    response = {"error": {"message": "prescribed HTTP 400"}}
    attempts = [{"request": request, "response": response, "status_code": 400}]
    metadata = {"status_code": 400, "response_id": None, "response_status": None, "finish_reason": None}
    call = {"request": request, "response": response, "response_metadata": metadata}
    record = {"ng_trajectory": {"model_calls": [call]}}
    assert not failed_model_checks(record, attempts)
    metadata["response_id"] = "invented"
    assert failed_model_checks(record, attempts) == {"model.response_metadata"}


@pytest.mark.parametrize(
    "field", ["prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cached_tokens"]
)
def test_canonical_omitted_usage_must_stay_unavailable(field):
    request = {"messages": [{"role": "user", "content": "hello"}]}
    response = {"id": "r", "object": "chat.completion", "created": 123, "choices": [{"finish_reason": "stop"}]}
    call = {
        "request": request,
        "response": copy.deepcopy(response),
        "response_metadata": {"status_code": 200, "response_id": "r", "finish_reason": "stop"},
    }
    record = {"ng_trajectory": {"model_calls": [call]}}
    attempts = [{"request": request, "response": response, "status_code": 200}]
    assert not failed_model_checks(record, attempts)
    call["token_stats"] = {field: 0}
    assert failed_model_checks(record, attempts) == {"model.token_counts"}


def test_canonical_counts_cannot_be_swapped_between_calls(behavior_episode):
    record, witness = behavior_episode
    calls = record["ng_trajectory"]["model_calls"]
    # Give the calls distinct, independently known counts.
    for i, call in enumerate(calls):
        call["token_stats"]["prompt_tokens"] = 10 + i
        witness["attempts"][i]["response"]["usage"]["input_tokens"] = 10 + i
        call["response"]["usage"]["input_tokens"] = 10 + i
    assert not failed_model_checks(record, witness["attempts"])
    calls[0]["token_stats"], calls[1]["token_stats"] = calls[1]["token_stats"], calls[0]["token_stats"]
    assert failed_model_checks(record, witness["attempts"]) == {"model.token_counts"}


def test_missing_tool_request_blocks_only_request_comparison(behavior_episode):
    record, witness = behavior_episode
    inv = record["ng_trajectory"]["invocations"][0]
    inv["conversation"] = [i for i in inv["conversation"] if i.get("call_id") != "tool-1"]
    checks = {c["id"]: c for c in tool_checks(record, witness["tool_calls"])}
    assert checks["tools.witness_join"]["status"] == "fail"
    assert checks["tools.witness_request"]["status"] == "fail"
    assert checks["tools.witness_status"]["status"] == "pass"
    assert checks["tools.witness_output"]["status"] == "pass"


def test_missing_output_witness_cannot_qualify_artifacts(behavior_episode):
    record, witness = behavior_episode
    options = dict(http_errors=(), terminal_error=False, tool_steps=2, expected_reward=0.0, fingerprint=fingerprint)
    assert all(c["status"] != "fail" for c in inspect_behavior(witness, record, **options))
    del witness["tool_calls"][0]["outputs"]
    checks = {c["id"]: c for c in inspect_behavior(witness, record, **options)}
    assert checks["tools.witness_output"]["status"] == "fail"
    assert checks["tools.witness_output"]["reasons"] == [
        "retained tool output differs from the independent tool witness"
    ]
