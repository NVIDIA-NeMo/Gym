# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only context probes use real capture custody, not another call lifecycle."""

import json

import pytest

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.token_id_capture import current_capture_context
from nemo_gym.token_id_capture.sink import CAPTURE_PARENT_HEADER, NG_CAPTURE_FIELD
from responses_api_models.vllm_model.tests.test_segment_capture import HISTORY, make_capture_harness


def _probe(harness, rollout, items, *, parent=None, **body):
    return harness.client.post(
        f"/context/{rollout}/measure",
        json={"input": items, **body},
        headers={CAPTURE_PARENT_HEADER: json.dumps(parent)},
    )


def test_probe_uses_real_parent_lookup_without_call_or_ledger_mutation(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    requests = []

    async def tokenize(client, **body):
        assert current_capture_context() is None
        assert NG_CAPTURE_FIELD not in body
        assert "required_prefix_token_ids" not in body
        assert CAPTURE_PARENT_HEADER not in client.default_headers
        requests.append(body)
        return {"prompt_token_count": body["ng_prefix_len"] + 2, "ng_prefix_len": body["ng_prefix_len"]}

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    root = _probe(harness, "g_g0_s0", HISTORY)
    assert root.status_code == 200, root.text
    assert root.json() == {"prompt_token_count": 2}
    assert requests[-1]["ng_prefix_staging_chain"] == []
    assert harness.manifest("g_g0_s0").records == []
    assert harness.worker_calls == []

    first = harness.post("g_g0_s0", HISTORY, parent=None).json()
    continued = HISTORY + first["output"] + [{"role": "user", "content": "another observation"}]
    before = harness.manifest("g_g0_s0")
    staged = dict(harness.sink.records)
    for _ in range(2):
        measured = _probe(harness, "g_g0_s0", continued, parent=first["id"])
        assert measured.status_code == 200, measured.text
        assert measured.json() == {"prompt_token_count": before.records[0].cum_len + 2}
    assert requests[-1]["ng_prefix_staging_chain"] == [before.records[0].staging_key]
    assert requests[-1]["ng_prefix_len"] == before.records[0].cum_len
    assert harness.manifest("g_g0_s0") == before
    assert harness.sink.records == staged
    assert len(harness.worker_calls) == 1

    # An explicit fresh root, including a definite-response root retry, has no old prefix.
    assert _probe(harness, "g_g0_s0", HISTORY).json() == {"prompt_token_count": 2}
    assert _probe(harness, "g_g0_s1", continued).json() == {"prompt_token_count": 2}
    assert harness.manifest("g_g0_s1").records == []
    assert harness.post("g_g0_s0", continued, parent=first["id"]).status_code == 200


@pytest.mark.parametrize("failure", ["missing", "foreign", "mutated_history"])
def test_rejected_probe_never_poisons_a_later_valid_generation(monkeypatch, tmp_path, failure):
    harness = make_capture_harness(monkeypatch, tmp_path)
    first = harness.post("g_g0_s0", HISTORY, parent=None).json()
    continued = HISTORY + first["output"]
    before = harness.manifest("g_g0_s0")
    rollout, parent, items = "g_g0_s0", first["id"], continued
    if failure == "missing":
        parent = "not-served"
    elif failure == "foreign":
        rollout = "g_g1_s0"
    else:
        items = [{"role": "user", "content": "unexplained rewrite"}] + first["output"]

    async def unexpected_tokenize(*args, **kwargs):
        pytest.fail("unresolved probes must not reach the backend")

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", unexpected_tokenize)
    assert _probe(harness, rollout, items, parent=parent).status_code == 409
    assert harness.manifest("g_g0_s0") == before
    assert harness.manifest("g_g1_s0").failures == []
    assert len(harness.worker_calls) == 1
    assert harness.post("g_g0_s0", continued, parent=first["id"]).status_code == 200
    assert harness.manifest("g_g0_s0").failures == []


@pytest.mark.parametrize("hint", ["not-json", "false", "1", '""', "[]", "{}", None])
def test_probe_requires_well_formed_explicit_parent(monkeypatch, tmp_path, hint):
    harness = make_capture_harness(monkeypatch, tmp_path)
    response = harness.client.post(
        "/context/g_g0_s0/measure",
        json={"input": HISTORY},
        headers={} if hint is None else {CAPTURE_PARENT_HEADER: hint},
    )
    assert response.status_code == 422
    assert harness.worker_calls == []
    assert harness.manifest("g_g0_s0").records == []
    assert harness.manifest("g_g0_s0").failures == []


@pytest.mark.parametrize(
    "payload",
    [
        {"tokens": [1, 2], "count": 2},
        {"prompt_token_count": 2},
        {"prompt_token_count": 2, "ng_prefix_len": 9},
        {"prompt_token_count": True, "ng_prefix_len": 0},
        {"prompt_token_count": -1, "ng_prefix_len": 0},
        {"prompt_token_count": 2, "ng_prefix_len": False},
    ],
)
def test_probe_requires_worker_feature_acknowledgement(monkeypatch, tmp_path, payload):
    harness = make_capture_harness(monkeypatch, tmp_path)

    async def tokenize(*args, **kwargs):
        return payload

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    assert _probe(harness, "g_g0_s0", HISTORY).status_code == 502
    assert harness.worker_calls == []
    assert harness.manifest("g_g0_s0").failures == []


def test_probe_preserves_converter_preprocessing_and_prompt_options(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    seen = []

    async def tokenize(client, **body):
        seen.append(body)
        return {"prompt_token_count": 8, "ng_prefix_len": 0}

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    response = _probe(
        harness,
        "g_g0_s0",
        [{"role": "user", "content": "hello"}],
        instructions="keep this instruction",
        reasoning={"effort": "high"},
        tools=[{"type": "function", "name": "observe", "parameters": {"type": "object"}, "strict": True}],
        metadata={
            "chat_template_kwargs": '{"enable_thinking": false}',
            "extra_body": '{"add_special_tokens":true,"chat_template":"literal-template","media_io_kwargs":{"image":{}},"documents":[{"text":"source"}]}',
        },
    )
    assert response.status_code == 200, response.text
    assert seen[0]["model"] == "test-model"
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0]["add_special_tokens"] is True
    assert seen[0]["chat_template"] == "literal-template"
    assert seen[0]["media_io_kwargs"] == {"image": {}}
    assert seen[0]["reasoning_effort"] == "high"
    assert seen[0]["documents"] == [{"text": "source"}]
    assert seen[0]["tools"][0]["function"]["name"] == "observe"
    assert "strict" not in seen[0]["tools"][0]["function"]
    assert any(message["content"] == "keep this instruction" for message in seen[0]["messages"])
    assert harness.manifest("g_g0_s0").records == []


def test_cc_rejects_engine_prompt_truncation_before_probe_or_generation(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    body = {"input": HISTORY, "metadata": {"extra_body": '{"truncate_prompt_tokens": 10}'}}
    for path in ["/context/g_g0_s0/measure", "/ng-rollout/g_g0_s0/training-token-capture/v1/responses"]:
        response = harness.client.post(path, json=body, headers={CAPTURE_PARENT_HEADER: "null"})
        assert response.status_code == 422, response.text
    assert harness.worker_calls == []
