# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact native context validation precedes durable capture admission."""

import json

import pytest
from openai import BadRequestError, OpenAI

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.token_id_capture.sink import CAPTURE_PARENT_HEADER, NG_CAPTURE_FIELD
from responses_api_models.vllm_model.tests.test_framework_context import (
    HISTORY,
    assert_clean,
    make_capture_harness,
)


@pytest.mark.parametrize("dialect", ["responses", "chat/completions", "messages"])
@pytest.mark.parametrize("count", [99, 100, 101])
def test_prompt_boundary_rejects_before_intent_and_valid_retry_can_commit(monkeypatch, tmp_path, dialect, count):
    harness = make_capture_harness(monkeypatch, tmp_path)
    measured = []

    async def tokenize(client, **body):
        manifest = await harness.ledger.manifest("attempt")
        measured.append((body, manifest))
        return {"prompt_token_count": count, "max_model_len": 100, "ng_context_measured": True}

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    before = harness.manifest("attempt")
    body = {"input": "hello"} if dialect == "responses" else {"messages": [{"role": "user", "content": "hello"}]}
    if dialect == "messages":
        body["max_tokens"] = 8192
        body["model"] = "test-model"
    response = harness.client.post(f"/ng-rollout/attempt/training-token-capture/v1/{dialect}", json=body)
    if dialect == "messages":
        # Preserve the existing framework-capture dialect restriction.
        assert response.status_code == 422, response.text
        assert "requires Responses or Chat" in response.text
        assert not measured and not harness.worker_calls
        assert harness.manifest("attempt") == before
        return
    assert measured and measured[0][0][NG_CAPTURE_FIELD]["mode"] == "candidate", response.text
    assert not measured[0][1]["attempted_call_ids"]
    if count < 100:
        assert_clean(response)
        assert len(harness.manifest("attempt").records) == 1
        return
    assert response.status_code == 400, response.text
    assert "maximum context length is 100" in response.text
    assert f"{count} input tokens" in response.text
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert harness.manifest("attempt") == before
    assert not harness.worker_calls and not harness.sink.records
    # A genuine rejection acquired no worker custody. A compacted request
    # under the SAME rollout owner can proceed through ordinary capture.
    count = 2
    assert_clean(harness.post("attempt", [{"role": "user", "content": "summary"}]))
    after = harness.manifest("attempt")
    assert len(after.records) == len(after.attempted_call_ids) == 1
    assert not after.failures and not after.pending_call_ids


@pytest.mark.parametrize("send_parent_hint", [False, True])
def test_rejected_extension_retains_accepted_parent_and_same_endpoint(monkeypatch, tmp_path, send_parent_hint):
    harness = make_capture_harness(monkeypatch, tmp_path)
    first = assert_clean(harness.post("attempt", HISTORY))
    before = harness.manifest("attempt")
    measured = []
    limit = 100

    async def tokenize(client, **body):
        measured.append((client.base_url, client.retry_requests, body, await harness.ledger.manifest("attempt")))
        return {"prompt_token_count": 100, "max_model_len": limit, "ng_context_measured": True}

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    history = HISTORY + first["output"] + [{"role": "user", "content": "next"}]
    headers = {CAPTURE_PARENT_HEADER: json.dumps(first["id"])} if send_parent_hint else {}
    response = harness.client.post(
        "/ng-rollout/attempt/training-token-capture/v1/responses", json={"input": history}, headers=headers
    )
    assert response.status_code == 400, response.text
    assert harness.manifest("attempt") == before
    endpoint, retries, payload, during = measured[0]
    assert endpoint == "http://worker.test/v1" and retries is False
    assert payload[NG_CAPTURE_FIELD]["parent_call_id"] == before.records[0].model_call_id
    assert during == before.model_dump(mode="json")
    assert len(harness.worker_calls) == 1
    limit = 101
    assert_clean(
        harness.client.post(
            "/ng-rollout/attempt/training-token-capture/v1/responses", json={"input": history}, headers=headers
        )
    )
    after = harness.manifest("attempt")
    assert len(after.records) == 2 and not after.pending_call_ids and not after.failures
    assert measured[-1][2][NG_CAPTURE_FIELD]["parent_call_id"] == before.records[0].model_call_id
    assert harness.worker_calls[-1][0].parent_call_id == before.records[0].model_call_id


@pytest.mark.parametrize(
    "bad_result",
    [
        {"prompt_token_count": 2, "max_model_len": 100},
        {"prompt_token_count": True, "max_model_len": 100, "ng_context_measured": True},
        {"prompt_token_count": -1, "max_model_len": 100, "ng_context_measured": True},
        {"prompt_token_count": 2, "ng_context_measured": True},
        {"prompt_token_count": 2, "max_model_len": True, "ng_context_measured": True},
        {"prompt_token_count": 2, "max_model_len": 0, "ng_context_measured": True},
    ],
)
def test_unacknowledged_measurement_never_creates_a_capture_intent(monkeypatch, tmp_path, bad_result):
    harness = make_capture_harness(monkeypatch, tmp_path)
    before = harness.manifest("attempt")

    async def tokenize(client, **body):
        return bad_result

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    response = harness.post("attempt", HISTORY)
    assert response.status_code == 502, response.text
    assert harness.manifest("attempt") == before
    assert not harness.worker_calls and not harness.sink.records


def test_measurement_transport_failure_keeps_attempt_recoverable(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    before = harness.manifest("attempt")
    fail = True

    async def tokenize(client, **body):
        if fail:
            raise RuntimeError("measurement transport failed before generation")
        return {"prompt_token_count": 2, "max_model_len": 100, "ng_context_measured": True}

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    assert harness.post("attempt", HISTORY).status_code == 500
    assert harness.manifest("attempt") == before
    assert not harness.worker_calls
    fail = False
    assert_clean(harness.post("attempt", HISTORY))
    assert len(harness.manifest("attempt").records) == 1


def test_valid_prompt_preserves_requested_output_length(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    original_worker = NeMoGymAsyncOpenAI.create_chat_completion
    generated = []

    async def tokenize(client, **body):
        assert "max_tokens" not in body and "max_completion_tokens" not in body
        return {"prompt_token_count": 99, "max_model_len": 100, "ng_context_measured": True}

    async def worker(client, **body):
        generated.append(body)
        return await original_worker(client, **body)

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", worker)
    response = harness.client.post(
        "/ng-rollout/attempt/training-token-capture/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8192},
    )
    assert_clean(response)
    assert generated[0]["max_tokens"] == 8192


def test_openai_client_recognizes_context_error_without_poisoning_capture(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    before = harness.manifest("attempt")

    async def tokenize(client, **body):
        return {"prompt_token_count": 282833, "max_model_len": 262144, "ng_context_measured": True}

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    client = OpenAI(
        base_url="http://testserver/ng-rollout/attempt/training-token-capture/v1",
        api_key="test-key",
        max_retries=0,
        http_client=harness.client,
    )
    with pytest.raises(BadRequestError) as caught:
        client.chat.completions.create(model="test-model", messages=[{"role": "user", "content": "hello"}])
    assert caught.value.code == "context_length_exceeded"
    assert caught.value.body["param"] == "input_tokens"
    assert "282833 input tokens" in str(caught.value)
    assert harness.manifest("attempt") == before
    assert not harness.worker_calls and not harness.sink.records
    client.close()
