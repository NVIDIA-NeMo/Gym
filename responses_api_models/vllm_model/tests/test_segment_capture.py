# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP segment capture through ordinary worker staging and the model ledger."""

import asyncio
import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponse, ClientResponseError, ServerDisconnectedError
from fastapi.testclient import TestClient

import nemo_gym.server_utils
from nemo_gym.base_responses_api_model import _CaptureMiddleware
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.sink import CAPTURE_PARENT_HEADER, NG_CAPTURE_FIELD, NG_COMMIT_COORDS_FIELD
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
from nemo_gym.token_id_capture.staging.rebuild import verify_and_linearize
from nemo_gym.token_id_capture.staging.records import (
    CaptureAdmission,
    RolloutManifest,
    RolloutReceipt,
    StageResult,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


_INFER = object()
HISTORY = [
    {"role": "user", "content": "old question"},
    {"role": "assistant", "content": "historical answer"},
    {"role": "user", "content": "continue the task"},
]


class MemoryStagingSink:
    def __init__(self):
        self.records = {}
        self.lose_ack = False

    def stage(self, record):
        key = f"{record.rollout_id}/{record.model_call_id}"
        self.records[key] = record.model_copy(deep=True)
        if self.lose_ack:
            return StageResult(ok=False, error="write completed but acknowledgement lost")
        return StageResult(ok=True, staging_key=key)

    def prefix(self, keys):
        return [token for key in keys for token in self.records[key].token_ids_delta]


@dataclass
class CaptureHarness:
    client: TestClient
    ledger: object
    sink: object
    worker_calls: list = field(default_factory=list)
    omit_coords: bool = False

    def post(self, rollout_id, input_items, *, parent=_INFER):
        path = f"/ng-rollout/{rollout_id}/training-token-capture/v1/responses" if rollout_id else "/v1/responses"
        headers = {} if parent is _INFER else {CAPTURE_PARENT_HEADER: json.dumps(parent)}
        return self.client.post(path, json={"input": input_items}, headers=headers)

    def manifest(self, rollout_id):
        return RolloutManifest.model_validate(asyncio.run(self.ledger.manifest(rollout_id)))


def make_capture_harness(
    monkeypatch, tmp_path, *, sink=None, fetch_prefix=None, transport_failure=None, root_prompt=None, reasoning=False
):
    """Allow paired RL tests to inject their staging transport and prefix reader."""
    sink = sink if sink is not None else MemoryStagingSink()
    fetch_prefix = fetch_prefix if fetch_prefix is not None else sink.prefix
    capture = RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 7)
    config = {
        "token_id_capture": {
            "enabled": True,
            "external_staging": True,
            "rebuild_response": False,
            "lineage_store": "nemo_gym.token_id_capture.lineage:FileLineageStore",
            "lineage_store_kwargs": {"root": str(tmp_path / "lineage")},
        }
    }
    monkeypatch.setenv("NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN", "test-control")
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", lambda: config)
    model = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=8081,
            base_url="http://worker.test/v1",
            api_key="test-key",
            model="test-model",
            entrypoint="",
            name="policy",
            return_token_id_information=False,
            uses_reasoning_parser=reasoning,
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict=config),
    )
    app = model.setup_webserver()
    middleware = next(item for item in app.user_middleware if item.cls is _CaptureMiddleware)
    harness = CaptureHarness(TestClient(app), middleware.kwargs["lineage_store"], sink)

    async def worker(client, **body):
        # A real client survives request-scoped model_copy; only its backend is replaced.
        assert CAPTURE_PARENT_HEADER not in client.default_headers
        assert CAPTURE_PARENT_HEADER not in json.dumps(body).lower()
        index = len(harness.worker_calls) + 1
        admission = CaptureAdmission.model_validate(body[NG_CAPTURE_FIELD]) if NG_CAPTURE_FIELD in body else None
        harness.worker_calls.append((admission, client.retry_requests))
        payload = {
            "id": f"chatcmpl-{index}",
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}],
        }
        if reasoning:
            payload["choices"][0]["message"]["reasoning"] = f"reasoning {index}"
        if admission is not None:
            prefix = fetch_prefix(admission.staging_chain) if admission.staging_chain else []
            active = capture.begin_call(admission, prefix_token_ids=prefix)
            prompt = prefix + [10 * index, 10 * index + 1]
            if admission.mode == "text" and root_prompt is not None:
                prompt = list(root_prompt)
            generated = [1000 + index]
            coords = capture.complete_call(
                active, prompt_token_ids=prompt, generated_token_ids=generated, generated_logprobs=[-0.25]
            )
            if not harness.omit_coords:
                payload[NG_COMMIT_COORDS_FIELD] = coords.model_dump(mode="json")
            # These transport-only arrays must disappear on the served agent response.
            payload["prompt_token_ids"] = prompt
            payload["choices"][0].update(token_ids=generated, logprobs={"content": []})
            payload["choices"][0]["message"].update(
                prompt_token_ids=prompt, generation_token_ids=generated, generation_log_probs=[-0.25]
            )
        return payload

    create_chat_completion = NeMoGymAsyncOpenAI.create_chat_completion

    async def worker_through_transport(client, **body):
        async def send(**kwargs):
            assert kwargs["method"] == "POST"
            assert kwargs["url"] == "http://worker.test/v1/chat/completions"
            assert CAPTURE_PARENT_HEADER not in kwargs["headers"]
            payload = await worker(client, **json.loads(kwargs["data"]))
            response = MagicMock(spec=ClientResponse, status=200, ok=True)
            response.read = AsyncMock(return_value=json.dumps(payload).encode())
            # A second attempt succeeds so accidentally restored retries fail
            # the test promptly instead of hanging in the ordinary retry loop.
            if len(harness.worker_calls) == 1:
                failure = ServerDisconnectedError("worker staged the tokens but its response was lost")
                if transport_failure == "request":
                    raise failure
                response.read.side_effect = failure
            return response

        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", lambda: MagicMock(request=send))
        return await create_chat_completion(client, **body)

    monkeypatch.setattr(
        NeMoGymAsyncOpenAI, "create_chat_completion", worker_through_transport if transport_failure else worker
    )
    return harness


@pytest.fixture
def harness(monkeypatch, tmp_path):
    result = make_capture_harness(monkeypatch, tmp_path)
    yield result
    result.client.close()
    asyncio.run(result.ledger.close())


def assert_clean(response, status=200):
    assert response.status_code == status, response.text
    for key in (
        NG_CAPTURE_FIELD,
        NG_COMMIT_COORDS_FIELD,
        "prompt_token_ids",
        "generation_token_ids",
        "generation_log_probs",
    ):
        assert key not in response.text
    return response.json()


def selected_row(harness, rollout_id, response_id):
    manifest = harness.manifest(rollout_id)
    assert manifest.failures == []
    terminal = next(record for record in manifest.records if record.response_id == response_id)
    receipt = RolloutReceipt(
        rollout_id=rollout_id,
        manifest=manifest.records,
        terminal_model_call_id=terminal.model_call_id,
        terminal_selection="declared",
    )
    return verify_and_linearize(receipt, [harness.sink.records[record.staging_key] for record in manifest.records])


def test_initial_root_selected_continuation_and_new_segment(harness):
    assert_clean(harness.post(None, [{"role": "user", "content": "ordinary"}]))
    assert harness.worker_calls == [(None, True)]
    assert_clean(harness.post(None, HISTORY, parent=None), 409)

    root = assert_clean(harness.post("L_s0", HISTORY, parent=None))
    continuation = HISTORY + root["output"] + [{"role": "user", "content": "next"}]
    child = assert_clean(harness.post("L_s0", continuation, parent=root["id"]))
    root_admission, child_admission = [call[0] for call in harness.worker_calls[1:]]
    assert root_admission.mode == "text" and root_admission.prev_len == 0
    assert child_admission.mode == "token_in"
    assert child_admission.parent_call_id == root_admission.model_call_id
    assert child_admission.prev_len == 3
    row = selected_row(harness, "L_s0", child["id"])
    assert row.token_ids == [20, 21, 1002, 30, 31, 1003]
    assert row.token_mask == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert all(not retry for _, retry in harness.worker_calls[1:])

    next_root = assert_clean(harness.post("L_s1", continuation, parent=None))
    next_admission = harness.worker_calls[-1][0]
    assert next_admission.mode == "text" and next_admission.staging_chain == []
    assert next_admission.parent_call_id is None
    assert selected_row(harness, "L_s1", next_root["id"]).token_ids == [40, 41, 1004]
    assert_clean(harness.post(None, [{"role": "user", "content": "ordinary again"}]))
    assert harness.worker_calls[-1] == (None, True), "CC must not change the cached ordinary client's retry policy"


def test_definite_root_retry_uses_selected_response_chain(harness):
    discarded = assert_clean(harness.post("L_s0", HISTORY, parent=None))
    selected = assert_clean(harness.post("L_s0", HISTORY, parent=None))
    assert discarded["id"] != selected["id"]
    continued = assert_clean(harness.post("L_s0", HISTORY + selected["output"], parent=selected["id"]))
    row = selected_row(harness, "L_s0", continued["id"])
    assert row.token_ids == [20, 21, 1002, 30, 31, 1003]
    assert row.model_call_ids == [call[0].model_call_id for call in harness.worker_calls[1:]]
    assert len(harness.manifest("L_s0").records) == 3


@pytest.mark.parametrize("failure", ["lost_ack", "missing_coords"])
@pytest.mark.parametrize("explicit", [False, True])
def test_capture_failure_fails_cc_and_preserves_ordinary_poison_and_serve(harness, failure, explicit):
    harness.sink.lose_ack = failure == "lost_ack"
    harness.omit_coords = failure == "missing_coords"
    response = harness.post("L_s0", HISTORY, **({"parent": None} if explicit else {}))
    assert_clean(response, 502 if explicit else 200)
    assert len(harness.worker_calls) == 1
    assert harness.worker_calls[0][1] is not explicit
    assert len(harness.sink.records) == 1, "the failed acknowledgement must follow a completed staging write"
    manifest = harness.manifest("L_s0")
    assert manifest.records == []
    assert manifest.failures


@pytest.mark.parametrize("transport_failure", ["request", "body"])
def test_lost_worker_response_stages_once_through_real_model_client(monkeypatch, tmp_path, transport_failure):
    harness = make_capture_harness(monkeypatch, tmp_path, transport_failure=transport_failure)
    try:
        # TestClient propagates the server's transport exception. No successful
        # Responses API completion may escape after the worker has mutated storage.
        with pytest.raises(ServerDisconnectedError, match="worker staged the tokens"):
            harness.post("L_s0", HISTORY, parent=None)

        assert len(harness.worker_calls) == len(harness.sink.records) == 1
        staged = next(iter(harness.sink.records.values()))
        assert staged.token_ids_delta == [10, 11, 1001]
        manifest = harness.manifest("L_s0")
        assert manifest.records == []
        assert [(failure.model_call_id, failure.reason) for failure in manifest.failures] == [
            (staged.model_call_id, "request_finished_without_staged_coordinates")
        ]
    finally:
        harness.client.close()
        asyncio.run(harness.ledger.close())


@pytest.mark.parametrize("explicit", [False, True])
def test_adapter_context_length_return_still_requires_committed_cc_capture(harness, monkeypatch, explicit):
    upstream_error = ClientResponseError(request_info=MagicMock(), history=(), status=400)
    upstream_error.response_content = b'{"message":"This model maximum context length was exceeded"}'
    generation = AsyncMock(side_effect=upstream_error)
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", generation)

    response = harness.post("L_s0", HISTORY, **({"parent": None} if explicit else {}))
    payload = assert_clean(response, 502 if explicit else 200)
    if explicit:
        assert payload == {"detail": "CC worker capture did not commit"}
    else:
        assert len(payload["output"]) == 1
        assert payload["output"][0]["content"][0]["text"] == ""
    generation.assert_awaited_once()
    assert harness.sink.records == {}
    manifest = harness.manifest("L_s0")
    assert manifest.records == []
    assert {failure.reason for failure in manifest.failures} == {
        "worker_response_missing_commit_coordinates",
        "request_finished_without_staged_coordinates",
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_lost_custody_ack_retains_commit_and_fails_cc(harness, monkeypatch, explicit):
    record = harness.ledger.record
    commits = []

    async def commit_then_lose_ack(commit):
        await record(commit)
        commits.append(commit)
        raise OSError("custody committed but acknowledgement lost")

    monkeypatch.setattr(harness.ledger, "record", commit_then_lose_ack)
    response = harness.post("L_s0", HISTORY, **({"parent": None} if explicit else {}))
    assert_clean(response, 502 if explicit else 200)
    assert len(commits) == len(harness.worker_calls) == len(harness.sink.records) == 1
    manifest = harness.manifest("L_s0")
    assert manifest.records == [commits[0].record]
    assert manifest.records[0].staging_key in harness.sink.records
    assert "invalid_worker_commit_coordinates" in {failure.reason for failure in manifest.failures}


def test_selected_response_cannot_cross_segment_namespace(harness):
    root = assert_clean(harness.post("L_s0", HISTORY, parent=None))
    assert_clean(harness.post("L_s1", HISTORY + root["output"], parent=root["id"]), 409)
    assert len(harness.worker_calls) == 1
    assert harness.manifest("L_s1").records == []
    assert harness.manifest("L_s1").failures


@pytest.mark.parametrize("value", ["not-json", '""', "123", "true", "[]", "{}"])
def test_invalid_parent_hint_rejected_before_generation(harness, value):
    response = harness.client.post(
        "/ng-rollout/L_s0/training-token-capture/v1/responses",
        json={"input": HISTORY},
        headers={CAPTURE_PARENT_HEADER: value},
    )
    assert response.status_code == 422
    assert harness.worker_calls == []


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_parent_hint_cannot_silently_fall_back_to_chat_capture(harness, capture, stream):
    path = "/v1/chat/completions"
    if capture:
        path = "/ng-rollout/L_s0/training-token-capture" + path
    response = harness.client.post(
        path,
        json={"messages": HISTORY, **({"stream": True} if stream else {})},
        headers={CAPTURE_PARENT_HEADER: '"missing-parent"'},
    )
    assert response.status_code == 422
    assert harness.worker_calls == []
