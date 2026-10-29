# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import tempfile
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import nemo_gym.server_utils
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.agentic_if_realworld.app import (
    VERIFIER_FIXTURE,
    FixtureVerifier,
    ResourcesConfig,
    ResourcesServer,
    Verifier,
    VerifyRequest,
    project_gym_response,
)
from resources_servers.agentic_if_realworld.benchmark import scoring
from resources_servers.agentic_if_realworld.benchmark.core import digest, read, write
from resources_servers.agentic_if_realworld.benchmark.synthetic import delivery, review
from resources_servers.agentic_if_realworld.benchmark.transport import RetryableTransportError


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
    "endpoint,model,error",
    [
        ("", "synthetic", "Endpoint"),
        ("ftp://synthetic.invalid", "synthetic", "Endpoint"),
        ("http://synthetic.invalid:bad", "synthetic", "Port"),
        ("http://secret@synthetic.invalid", "synthetic", "Endpoint"),
        ("http://synthetic.invalid", " ", "Judge model"),
    ],
)
def test_judge_startup_validates_routing_without_changing_dry_run(endpoint, model, error, monkeypatch):
    monkeypatch.delenv("SYNTHETIC_MISSING_KEY", raising=False)
    server = ResourcesServer.model_construct(
        config=SimpleNamespace(judge_endpoint=endpoint, judge_model=model, judge_api_key_env="SYNTHETIC_MISSING_KEY")
    )
    with pytest.raises(ValueError, match=error):
        server.setup_webserver()


def test_judge_startup_requires_key_only_when_serving(monkeypatch):
    monkeypatch.delenv("SYNTHETIC_MISSING_KEY", raising=False)
    server = ResourcesServer.model_construct(
        config=SimpleNamespace(
            judge_endpoint="http://synthetic.invalid",
            judge_model="synthetic",
            judge_api_key_env="SYNTHETIC_MISSING_KEY",
        )
    )
    with pytest.raises(ValueError, match="SYNTHETIC_MISSING_KEY is not set"):
        server.setup_webserver()


def test_http_verify_outage_is_an_error_then_recovers_saved_answer(verifier, monkeypatch):
    from resources_servers.agentic_if_realworld.tests.test_transport import Response

    calls = []

    async def request(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return Response(401, {"error": "bad credentials"})
        payload = json.loads(kwargs["json"]["messages"][0]["content"].split("UNTRUSTED_EVALUATION_PAYLOAD:\n", 1)[1])
        return Response(
            200, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(review(payload))}}]}
        )

    monkeypatch.setattr(nemo_gym.server_utils, "request", request)
    monkeypatch.setenv("SYNTHETIC", "bad-key")
    config = ResourcesConfig(
        host="",
        port=0,
        entrypoint="app.py",
        name="agentic_if_realworld_resources",
        dataset_dir=str(verifier.dataset.directory),
        synthetic=True,
        receipt_dir=str(verifier.receipt_dir),
        judge_endpoint="http://synthetic.invalid",
        judge_api_key_env="SYNTHETIC",
    )
    server = ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    body = delivery(verifier.dataset, verifier.receipt_dir, "OK.")
    saved = {path: path.read_bytes() for path in (verifier.receipt_dir / "model").iterdir()}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        response = client.post("/verify", json=body)
        assert response.status_code == 500 and len(calls) == 1
        assert not list(verifier.receipt_dir.glob("judges/*/result.json"))
        monkeypatch.setenv("SYNTHETIC", "corrected-key")
        response = client.post("/verify", json=body)
        assert response.status_code == 200 and response.json()["item_verdict"] == "pass"
        assert len(calls) == 2 and all(path.read_bytes() == value for path, value in saved.items())


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
def test_production_verifier_distinguishes_judge_outage_from_delivered_invalid_review(
    verifier, monkeypatch, outcome, response_source
):
    calls = []

    async def wire(body, *, path, **kwargs):
        calls.append(1)
        if outcome == "outage":
            return {"status": "delivery_error", "http_status": 401}
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
    if outcome == "outage":
        with pytest.raises(RetryableTransportError):
            asyncio.run(production.verify(VerifyRequest.model_validate(body)))
        assert len(calls) == 1
        return
    result = asyncio.run(production.verify(VerifyRequest.model_validate(body)))
    assert result.scoring_status == "not_scored"
    assert result.item_verdict is None and result.mask_sample is True
    assert result.reward == 0.0 and len(calls) == 2


class JudgeContent:
    def __init__(self, data: bytes):
        self.data = data

    async def iter_chunked(self, size):
        yield self.data


class JudgeResponse:
    status = 200
    headers = {"content-type": "application/json"}

    def __init__(self, data: bytes):
        self.content = JudgeContent(data)

    def release(self):
        pass


@pytest.mark.parametrize("first_outcomes", [["transport", "transport"], ["malformed", "transport"]])
def test_ambiguous_judge_dispatch_stays_blocked_on_resume(verifier, monkeypatch, first_outcomes):
    calls = []

    async def request(method, url, **kwargs):
        calls.append(1)
        outcome = first_outcomes[len(calls) - 1] if len(calls) <= len(first_outcomes) else "graded"
        if outcome == "transport":
            raise ConnectionError("judge unreachable")
        content = "not JSON"
        if outcome == "graded":
            prompt = kwargs["json"]["messages"][0]["content"]
            content = json.dumps(review(json.loads(prompt.rsplit("UNTRUSTED_EVALUATION_PAYLOAD:\n", 1)[1])))
        choice = {"finish_reason": "stop", "message": {"content": content}}
        return JudgeResponse(json.dumps({"id": "judge", "choices": [choice]}).encode())

    monkeypatch.setattr(nemo_gym.server_utils, "request", request)
    monkeypatch.setenv("AGENTIC_IF_JUDGE_API_KEY", "synthetic")
    production = Verifier(
        dataset=verifier.dataset, receipt_dir=verifier.receipt_dir, endpoint="http://synthetic.invalid"
    )
    body = VerifyRequest.model_validate(delivery(verifier.dataset, verifier.receipt_dir, "OK."))
    with pytest.raises(ValueError, match="Ambiguous judge dispatch"):
        asyncio.run(production.verify(body))
    dispatched = first_outcomes.index("transport") + 1
    assert len(calls) == dispatched
    saved = {path: digest(path) for path in verifier.receipt_dir.glob("judges/*/attempt*.*")}
    assert len(list(verifier.receipt_dir.glob("judges/*/attempt*.json"))) == dispatched
    for _ in range(2):
        with pytest.raises(ValueError, match="Ambiguous judge dispatch"):
            asyncio.run(production.verify(body))
        assert len(calls) == dispatched
        assert {path: digest(path) for path in verifier.receipt_dir.glob("judges/*/attempt*.*")} == saved


@pytest.mark.parametrize("response_source", ["raw_receipt", "gym_response"])
@pytest.mark.parametrize(
    "damage", [None, "payload", "schema", "request", "request_hash", "endpoint", "raw", "started"]
)
def test_pending_judge_cache_validation_is_readonly(verifier, monkeypatch, response_source, damage):
    calls = []

    async def request(*args, **kwargs):
        calls.append(1)
        response = JudgeResponse(b'{"error":"synthetic rejected credentials"}')
        response.status = 401
        return response

    monkeypatch.setattr(nemo_gym.server_utils, "request", request)
    monkeypatch.setenv("AGENTIC_IF_JUDGE_API_KEY", "synthetic")
    production = Verifier(
        dataset=verifier.dataset,
        receipt_dir=verifier.receipt_dir,
        endpoint="http://synthetic.invalid",
        response_source=response_source,
    )
    row = delivery(verifier.dataset, verifier.receipt_dir, "OK.")
    if response_source == "gym_response":
        row["response"]["metadata"] = {}
        row["response"]["status"] = "completed"
    validate = VerifyRequest.model_validate({**row, "validate_judge_cache": True})

    def saved_files():
        return {path: path.read_bytes() for path in verifier.receipt_dir.rglob("*") if path.is_file()}

    # A saved answer may not have reached its first judge dispatch yet.
    before = saved_files()
    result = asyncio.run(production.verify(validate))
    assert result.scoring_status == "not_scored" and result.mask_sample is True
    assert result.reason == "judge_cache_validated_without_scoring"
    assert not calls and saved_files() == before
    assert not (verifier.receipt_dir / "judges").exists()

    with pytest.raises(RetryableTransportError):
        asyncio.run(production.verify(VerifyRequest.model_validate(row)))
    assert len(calls) == 1
    work = next((verifier.receipt_dir / "judges").iterdir())
    assert len(list(work.glob("attempt*.json"))) == 1
    assert not (work / "result.json").exists()
    attempt_path = work / "attempt0.json"
    attempt = read(attempt_path)
    if damage == "payload":
        write(work / "payload.json", {"changed": True})
    elif damage == "schema":
        write(work / "transport_schema.json", {"changed": True})
    elif damage == "raw":
        next(work.glob("attempt0.a*.http")).write_bytes(b"changed")
    elif damage:
        if damage == "request":
            attempt["request_body"]["model"] = "changed"
        elif damage == "request_hash":
            attempt["request_sha256"] = "0" * 64
        elif damage == "endpoint":
            attempt["endpoint"] = "http://changed.invalid"
        else:
            attempt["status"] = "started"
        write(attempt_path, attempt)
    before = saved_files()
    if damage:
        with pytest.raises(ValueError):
            asyncio.run(production.verify(validate))
    else:
        for _ in range(2):
            result = asyncio.run(production.verify(validate))
            assert result.scoring_status == "not_scored" and result.mask_sample is True
    assert len(calls) == 1 and saved_files() == before


def test_judge_cache_validation_rechecks_successful_review(verifier, monkeypatch):
    calls = []

    async def request(*args, **kwargs):
        calls.append(1)
        prompt = kwargs["json"]["messages"][0]["content"]
        payload = json.loads(prompt.rsplit("UNTRUSTED_EVALUATION_PAYLOAD:\n", 1)[1])
        choice = {"finish_reason": "stop", "message": {"content": json.dumps(review(payload))}}
        return JudgeResponse(json.dumps({"choices": [choice]}).encode())

    monkeypatch.setattr(nemo_gym.server_utils, "request", request)
    monkeypatch.setenv("AGENTIC_IF_JUDGE_API_KEY", "synthetic")
    production = Verifier(
        dataset=verifier.dataset, receipt_dir=verifier.receipt_dir, endpoint="http://synthetic.invalid"
    )
    row = delivery(verifier.dataset, verifier.receipt_dir, "OK.")
    assert asyncio.run(production.verify(VerifyRequest.model_validate(row))).reward == 1
    validate = VerifyRequest.model_validate({**row, "validate_judge_cache": True})
    saved = {path: path.read_bytes() for path in verifier.receipt_dir.rglob("*") if path.is_file()}
    result = asyncio.run(production.verify(validate))
    assert result.scoring_status == "scored" and result.reward == 1 and result.mask_sample is False
    assert len(calls) == 1
    assert {path: path.read_bytes() for path in saved} == saved
    result_path = next(verifier.receipt_dir.glob("judges/*/result.json"))
    cached = read(result_path)
    cached["raw_review"]["item_verdict"] = "fail"
    write(result_path, cached)
    damaged = result_path.read_bytes()
    with pytest.raises(ValueError, match="Cached grade differs"):
        asyncio.run(production.verify(validate))
    assert len(calls) == 1 and result_path.read_bytes() == damaged


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
        ("prompt", {"id": "pmpt_existing"}),
        ("conversation", "conv_existing"),
        ("conversation", {"id": "conv_existing"}),
        ("conversation", ""),
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
    verifier.evaluate = AsyncMock(side_effect=AssertionError("Additional context must not reach the judge"))
    body["responses_create_params"][field] = value
    with pytest.raises(ValueError, match="Model-visible input changed"):
        asyncio.run(verifier.verify(VerifyRequest.model_validate(body)))
    verifier.evaluate.assert_not_called()


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
