# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutation tests: schema-shaped lies must not become passing artifacts."""

import copy
import json
from pathlib import Path

import pytest

from nemo_gym.harness_capabilities.checker import P0, inspect_record
from nemo_gym.harness_capabilities.cli import inspect_bundle, main
from nemo_gym.harness_capabilities.reader import hydrate_record


@pytest.fixture(params=["opencode", "miniswe"])
def record(request):
    return json.loads((Path(__file__).parent / f"fixtures/{request.param}.json").read_text())


def verdict(record, capability):
    return inspect_record(hydrate_record(record))["capabilities"][capability]["verdict"]


def test_opencode_native_projection_passes_p0(record):
    result = inspect_record(hydrate_record(record))
    assert all(result["capabilities"][c]["verdict"] == "fulfilled" for c in P0), result["findings"]
    assert result["capabilities"]["C5"]["verdict"] == "fulfilled"
    assert not any(v["behavioral_qualification"] for v in result["capabilities"].values())


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
    assert verdict(record, "C2") == "not_fulfilled"


def test_omitted_usage_cannot_become_zero(record):
    call = record["ng_model_call_capture"]["calls"][0]
    call["cached_tokens"] = 0
    del record["ng_trajectory"]["model_calls"][0]["response"]["usage"]["input_tokens_details"]["cached_tokens"]
    assert verdict(record, "C2") == "not_fulfilled"


def test_legitimate_zero_is_available(record):
    record["ng_model_call_capture"]["calls"][0]["cached_tokens"] = 0
    record["ng_trajectory"]["model_calls"][0]["token_stats"]["cached_tokens"] = 0
    record["ng_trajectory"]["model_calls"][0]["response"]["usage"]["input_tokens_details"]["cached_tokens"] = 0
    assert verdict(record, "C2") == "fulfilled"


def test_missing_response_usage_cannot_be_invented(record):
    record["ng_trajectory"]["model_calls"][0]["response"]["usage"] = None
    assert verdict(record, "C2") == "not_fulfilled"


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
    assert verdict(record, "C4") == "not_fulfilled"
    assert verdict(record, "C9") == "not_fulfilled"


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
    assert verdict(record, "C10") == "not_fulfilled"


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
    assert verdict(record, "C10") == "not_fulfilled"
    record["ng_agent_observations"]["records"][0]["model_calls"].append({"model_call_id": "failure"})
    assert verdict(record, "C10") == "fulfilled"
    assert verdict(record, "C9") == "fulfilled"


def test_payload_removal_fails_even_with_complete_conversation(record):
    record["ng_trajectory"]["model_calls"][0]["request"] = None
    assert verdict(record, "C4") == "not_fulfilled"
    assert verdict(record, "C9") == "not_fulfilled"


def test_schema_valid_duplicate_attempt_fails(record):
    record["ng_model_call_capture"]["calls"] *= 2
    assert verdict(record, "C1") == "not_fulfilled"


def test_nullable_provider_model_keeps_explicit_server_identity(record):
    call = record["ng_model_call_capture"]["calls"][0]
    call["model"] = None
    assert verdict(record, "C1") == "fulfilled"
    call["model_ref"] = None
    assert verdict(record, "C1") == "not_fulfilled"


def test_sidecar_does_not_launder_payload_conflict(record, tmp_path):
    call = record["ng_model_call_capture"]["calls"][0]
    sidecar = {**call, "request": {"input": "different"}}
    (tmp_path / "0-0.capture.jsonl").write_text(json.dumps(sidecar) + "\n")
    hydrated = hydrate_record(record, capture_dir=tmp_path)
    assert inspect_record(hydrated)["capabilities"]["C0"]["verdict"] == "not_fulfilled"


def test_tool_request_is_not_execution(record):
    record["ng_agent_observations"]["records"] = [
        r for r in record["ng_agent_observations"]["records"] if r["kind"] != "tool_call"
    ]
    record["ng_trajectory"]["tool_calls"] = []
    assert verdict(record, "C5") == "not_fulfilled"


def test_independent_witness_requirements_never_pass_from_shape(record):
    for capability in ("C6", "C7", "C8", "C11"):
        assert verdict(record, capability) == "not_fulfilled"


def test_cli_atomic_replay_and_exit_codes(record, tmp_path):
    bundle = tmp_path / "input.jsonl"
    bundle.write_text(json.dumps(record) + "\n")
    output = tmp_path / "reports"
    args = ["inspect", "--bundle", str(bundle), "--output", str(output)]
    assert main(args) == 0
    assert main(args) == 0
    assert len(list(output.iterdir())) == 1
    report = next(output.glob("*/capability_summary.json"))
    summary = json.loads(report.read_text())
    assert summary["checker_status"] == "completed"
    assert summary["is_behavioral_qualification"] is False
    record["ng_agent_observations"]["records"][0]["model_calls"] = []
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
        inspect_bundle(bundle, output=tmp_path / "out", profile="gym-artifacts-p0/v1")
    bundle.write_text((json.dumps(record) + "\n") * 2)
    _, report = inspect_bundle(bundle, output=tmp_path / "out", profile="gym-artifacts-p0/v1")
    assert report["verdict"] == "not_fulfilled"
