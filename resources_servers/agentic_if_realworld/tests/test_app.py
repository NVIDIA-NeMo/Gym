# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import tempfile
from copy import deepcopy

import pytest

from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.agentic_if_realworld.app import (
    VERIFIER_FIXTURE,
    FixtureVerifier,
    Verifier,
    VerifyRequest,
    project_gym_response,
)
from resources_servers.agentic_if_realworld.benchmark import scoring
from resources_servers.agentic_if_realworld.benchmark.core import digest, read, write
from resources_servers.agentic_if_realworld.benchmark.synthetic import delivery


@pytest.fixture
def verifier(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    instance = FixtureVerifier()
    try:
        yield instance
    finally:
        instance.work.cleanup()


def test_verifier_fixture():
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="stochastic",
        )
    )


@pytest.mark.parametrize(
    ("text", "finish", "status", "verdict", "reward", "masked"),
    [
        ("OK.", "stop", "scored", "pass", 1.0, False),
        ("OK..", "stop", "scored", "fail", 0.0, False),
        ("partial", "length", "not_scored", None, 0.0, True),
    ],
)
def test_production_verifier_result_accounting(verifier, text, finish, status, verdict, reward, masked):
    body = delivery(verifier.dataset, verifier.receipt_dir, text, finish)
    result = asyncio.run(verifier.verify(VerifyRequest.model_validate(body)))
    assert result.evaluation_completed == (status == "scored")
    assert (
        result.scoring_status,
        result.item_verdict,
        result.reward,
        result.mask_sample,
    ) == (
        status,
        verdict,
        reward,
        masked,
    )


@pytest.mark.parametrize(
    ("damage", "error"),
    [
        ("raw", "Raw model delivery changed"),
        ("request", "Actual upstream prompt differs"),
        ("request_hash", "Actual request hash differs"),
    ],
)
def test_production_verifier_rejects_damaged_delivery(verifier, damage, error):
    body = delivery(verifier.dataset, verifier.receipt_dir, "OK.")
    receipt_id = body["response"]["metadata"]["receipt_id"]
    path = verifier.receipt_dir / "model" / (receipt_id + ".json")
    receipt = read(path)
    if damage == "raw":
        from pathlib import Path

        Path(receipt["raw_attempts"][0]["path"]).write_text("changed")
    elif damage == "request":
        receipt["request_body"]["messages"][0]["content"] = "changed"
    else:
        receipt["request_sha256"] = "0" * 64
    write(path, receipt)
    with pytest.raises(ValueError, match=error):
        asyncio.run(verifier.verify(VerifyRequest.model_validate(body)))


@pytest.mark.parametrize("outcome", ["outage", "malformed"])
@pytest.mark.parametrize("response_source", ["raw_receipt", "gym_response"])
def test_production_verifier_masks_judge_technical_failures(verifier, monkeypatch, outcome, response_source):
    calls = []

    async def wire(body, *, path, **kwargs):
        calls.append(1)
        if outcome == "outage":
            return {"status": "delivery_error"}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]}
        raw = path.with_suffix(".http")
        write(raw, response)
        receipt = {
            "status": "generated",
            "response": response,
            "content_type": "application/json",
            "raw_attempts": [{"path": str(raw), "sha256": digest(raw)}],
        }
        write(path, receipt)
        return receipt

    monkeypatch.setattr(scoring, "wire_call", wire)
    production = Verifier(
        dataset=verifier.dataset,
        receipt_dir=verifier.receipt_dir,
        endpoint="http://synthetic.invalid",
        response_source=response_source,
    )
    body = delivery(verifier.dataset, verifier.receipt_dir, "OK.")
    if response_source == "gym_response":
        body["response"]["metadata"] = {}
        body["response"]["status"] = "completed"
    result = asyncio.run(production.verify(VerifyRequest.model_validate(body)))
    assert result.scoring_status == "not_scored"
    assert result.item_verdict is None and result.mask_sample is True
    assert result.reward == 0.0 and len(calls) == 2


@pytest.fixture
def native(verifier, tmp_path):
    body = delivery(verifier.dataset, verifier.receipt_dir, "OK.")
    body["response"]["metadata"] = {}
    body["response"]["status"] = "completed"
    verifier.response_source = "gym_response"
    # No model receipt exists at this location: a trainer supplies the response.
    verifier.receipt_dir = tmp_path / "native-receipts"
    return verifier, body


@pytest.mark.parametrize("text,verdict,reward", [("OK.", "pass", 1), ("OK..", "fail", 0)])
@pytest.mark.parametrize("refusal", [False, True])
def test_native_response_rewards_without_model_receipts(native, text, verdict, reward, refusal):
    verifier, body = native
    body["response"]["output"][0]["content"] = [
        {"type": "refusal", "refusal": text} if refusal else {"type": "output_text", "text": text}
    ]
    result = asyncio.run(verifier.verify(VerifyRequest.model_validate(body)))
    assert (result.item_verdict, result.reward, result.mask_sample) == (
        verdict,
        reward,
        False,
    )
    assert not verifier.receipt_dir.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "truncated",
        "message_incomplete",
        "empty",
        "reasoning_only",
        "native_action",
        "two_messages",
        "error",
    ],
)
def test_native_technical_failures_are_masked(native, damage):
    verifier, body = native
    response = body["response"]
    if damage == "truncated":
        response["status"] = "incomplete"
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    elif damage == "message_incomplete":
        response["output"][0]["status"] = "incomplete"
    elif damage == "empty":
        response["output"][0]["content"][0]["text"] = "  "
    elif damage == "reasoning_only":
        response["output"] = [
            {
                "id": "r",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Thinking"}],
            }
        ]
    elif damage == "native_action":
        response["output"].append(
            {
                "id": "c",
                "type": "function_call",
                "name": "forbidden",
                "call_id": "c",
                "arguments": "{}",
            }
        )
    elif damage == "two_messages":
        response["output"].append(deepcopy(response["output"][0]))
    elif damage == "error":
        response["error"] = {"code": "server_error", "message": "Unavailable"}
    result = asyncio.run(verifier.verify(VerifyRequest.model_validate(body)))
    assert (result.scoring_status, result.reward, result.mask_sample) == (
        "not_scored",
        0,
        True,
    )
    assert result.item_verdict is None


def test_native_preserves_training_metadata_and_judges_separate_projection(native):
    verifier, body = native
    response = body["response"]
    response["usage"] = {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    response["output"][0].update(
        prompt_token_ids=[1, 2],
        generation_token_ids=[3, 4],
        generation_log_probs=[-0.1, -0.2],
    )
    response["output"].insert(
        0,
        {
            "id": "r",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Thinking"}],
        },
    )
    request = VerifyRequest.model_validate(body)
    saved = request.response.model_dump()
    projection = project_gym_response(request.response)
    assert projection["message"]["reasoning_content"] == "Thinking"
    assert "generation_token_ids" not in projection["message"]
    result = asyncio.run(verifier.verify(request))
    assert result.reward == 1
    assert result.response.model_dump() == saved == request.response.model_dump()


@pytest.mark.parametrize(
    "field,value",
    [
        ("instructions", "extra context"),
        ("previous_response_id", "resp_old"),
        (
            "tools",
            [
                {
                    "type": "function",
                    "name": "forbidden",
                    "parameters": {},
                    "strict": False,
                }
            ],
        ),
    ],
)
def test_native_rejects_additional_actor_context(native, field, value):
    verifier, body = native
    body["responses_create_params"][field] = value
    with pytest.raises(ValueError, match="Model-visible input changed"):
        asyncio.run(verifier.verify(VerifyRequest.model_validate(body)))


def test_native_rejects_changed_binding(native):
    verifier, body = native
    body["verifier_metadata"]["input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Verifier input provenance mismatch"):
        asyncio.run(verifier.verify(VerifyRequest.model_validate(body)))


def test_adaptive_still_requires_receipt(native):
    verifier, body = native
    verifier.response_source = "raw_receipt"
    with pytest.raises(ValueError, match="Missing trusted model delivery receipt"):
        asyncio.run(verifier.verify(VerifyRequest.model_validate(body)))
