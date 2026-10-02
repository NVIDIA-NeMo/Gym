# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP segment capture through ordinary worker staging and the model ledger."""

import asyncio
import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponse, ServerDisconnectedError
from fastapi.testclient import TestClient

import nemo_gym.server_utils
from nemo_gym.base_responses_api_model import _CaptureMiddleware
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.sink import CAPTURE_PARENT_HEADER, NG_CAPTURE_FIELD, NG_COMMIT_COORDS_FIELD
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
from nemo_gym.token_id_capture.staging.records import (
    CaptureAdmission,
    RolloutManifest,
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
    model: object = None

    def post(self, rollout_id, input_items, *, parent=_INFER):
        path = f"/ng-rollout/{rollout_id}/training-token-capture/v1/responses" if rollout_id else "/v1/responses"
        headers = {} if parent is _INFER else {CAPTURE_PARENT_HEADER: json.dumps(parent)}
        return self.client.post(path, json={"input": input_items}, headers=headers)

    def manifest(self, rollout_id):
        return RolloutManifest.model_validate(asyncio.run(self.ledger.manifest(rollout_id)))


def make_capture_harness(
    monkeypatch,
    tmp_path,
    *,
    sink=None,
    fetch_prefix=None,
    transport_failure=None,
    root_prompt=None,
    reasoning=False,
    model_config=None,
    decide_input=None,
):
    """Allow paired RL tests to inject their staging transport and prefix reader."""
    sink = sink if sink is not None else MemoryStagingSink()
    fetch_prefix = fetch_prefix if fetch_prefix is not None else sink.prefix
    capture = RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 7)
    config = {
        "token_id_capture": {
            "enabled": True,
            "external_staging": True,
            "framework_owned_context": True,
            "rebuild_response": False,
            "lineage_store": "nemo_gym.token_id_capture.lineage:FileLineageStore",
            "lineage_store_kwargs": {"root": str(tmp_path / "lineage")},
        }
    }
    monkeypatch.setenv("NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN", "test-control")
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", lambda: config)
    model = VLLMModel(
        config=model_config
        or VLLMModelConfig(
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
    harness = CaptureHarness(
        TestClient(app, raise_server_exceptions=False), middleware.kwargs["lineage_store"], sink, model=model
    )

    async def worker(client, **body):
        # A real client survives request-scoped model_copy; only its backend is replaced.
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
            # Deterministic full prompts stand in for the model renderer in Gym-only tests.
            decision = decide_input(admission, messages=body["messages"]) if decide_input else None
            prefix = fetch_prefix(decision.serving.staging_chain) if decision and decision.serving else []
            admission = (
                decision.storage
                if decision
                else CaptureAdmission(
                    rollout_id=admission.rollout_id, model_call_id=admission.model_call_id, mode="text"
                )
            )
            active = capture.begin_call(admission, prefix_token_ids=prefix if admission.mode == "token_in" else None)
            prompt = prefix + (
                list(root_prompt) if root_prompt is not None and not prefix else [10 * index, 10 * index + 1]
            )
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


def assert_clean(response, status=200):
    assert response.status_code == status, response.text
    assert NG_CAPTURE_FIELD not in response.text
    assert NG_COMMIT_COORDS_FIELD not in response.text
    assert "prompt_token_ids" not in response.text
    return response.json()


def test_sequential_calls_share_attempt_and_store_independent_roots(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    first = assert_clean(harness.post("attempt", HISTORY))
    assert_clean(harness.post("attempt", HISTORY + first["output"] + [{"role": "user", "content": "next"}]))
    manifest = harness.manifest("attempt")
    assert len(manifest.records) == 2 and not manifest.failures and not manifest.pending_call_ids
    assert all(record.parent_call_id is None for record in manifest.records)
    assert len(manifest.attempted_call_ids) == 2
    assert all(retry is False for _, retry in harness.worker_calls)
    assert harness.worker_calls[1][0].parent_call_id == manifest.records[0].model_call_id


def test_measurement_uses_same_candidate_without_ledger_writes(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    first = assert_clean(harness.post("attempt", HISTORY))
    assert_clean(harness.post("attempt", HISTORY, parent=None))  # A newer rejected response.
    before = harness.manifest("attempt")
    calls = []

    async def tokenize(client, **body):
        calls.append(body)
        return {"prompt_token_count": 19, "ng_context_measured": True}

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    response = harness.client.post(
        "/context/attempt/measure",
        json={"input": HISTORY + first["output"] + [{"role": "user", "content": "next"}]},
        headers={CAPTURE_PARENT_HEADER: json.dumps(first["id"])},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"prompt_token_count": 19}
    assert calls[0][NG_CAPTURE_FIELD]["mode"] == "candidate"
    assert calls[0][NG_CAPTURE_FIELD]["parent_call_id"] == before.records[0].model_call_id
    assert harness.manifest("attempt") == before


@pytest.mark.parametrize("parent", ["not-json", '""', "3", "{}", "[]", "true"])
def test_invalid_candidate_hint_is_rejected_before_generation(monkeypatch, tmp_path, parent):
    harness = make_capture_harness(monkeypatch, tmp_path)
    response = harness.client.post(
        "/ng-rollout/attempt/training-token-capture/v1/responses",
        json={"input": HISTORY},
        headers={CAPTURE_PARENT_HEADER: parent},
    )
    assert response.status_code == 422
    assert not harness.worker_calls


def test_lost_capture_ack_is_terminal_and_preserves_attempt_identity(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    harness.omit_coords = True
    assert harness.post("attempt", HISTORY).status_code == 500
    manifest = harness.manifest("attempt")
    assert len(manifest.attempted_call_ids) == len(manifest.pending_call_ids) == 1
    assert len(harness.sink.records) == 1
    assert harness.post("attempt", HISTORY).status_code == 500
    assert len(harness.worker_calls) == 1


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_call_keeps_intent_without_claiming_worker_custody(monkeypatch, tmp_path, cancelled):
    harness = make_capture_harness(monkeypatch, tmp_path)
    first = assert_clean(harness.post("attempt", HISTORY))

    async def failed_worker(client, **body):
        if cancelled:
            raise asyncio.CancelledError()
        raise RuntimeError("engine failed before staging")

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", failed_worker)
    assert harness.post("attempt", HISTORY + first["output"]).status_code == 500
    manifest = harness.manifest("attempt")
    assert len(manifest.records) == len(harness.sink.records) == 1
    assert len(manifest.attempted_call_ids) == 2
    assert len(manifest.pending_call_ids) == 1
    assert manifest.failures[0].model_call_id == manifest.pending_call_ids[0]
    # The framework can mask this attempt. A failure row must not fabricate a
    # successful acknowledgement or permit another call under the same owner.
    assert harness.post("attempt", HISTORY).status_code == 500
    assert harness.manifest("attempt") == manifest
    harness.client.close()
    asyncio.run(harness.ledger.close())


@pytest.mark.parametrize(
    "dialect, body",
    [
        ("chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True}),
        ("chat/completions", {"messages": [{"role": "user", "content": "hi"}], "n": 2}),
        ("responses", {"input": "hi", "previous_response_id": "opaque"}),
        ("responses", {"input": "hi", "stream": True, "previous_response_id": "opaque"}),
    ],
)
def test_unsupported_request_does_not_reach_generation(monkeypatch, tmp_path, dialect, body):
    harness = make_capture_harness(monkeypatch, tmp_path)
    response = harness.client.post(f"/ng-rollout/attempt/training-token-capture/v1/{dialect}", json=body)
    assert response.status_code == 422, response.text
    assert not harness.worker_calls


def test_tool_arguments_beyond_the_64_bit_range_still_admit(monkeypatch, tmp_path):
    """A tool call whose arguments carry an integer beyond 64 bits (a kernel
    task's bitmask) must canonicalize for the replay digest instead of failing
    the call's capture and poisoning the rest of the attempt."""
    harness = make_capture_harness(monkeypatch, tmp_path)
    history = HISTORY + [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "shell",
            "arguments": '{"mask": 340282366920938463463374607431768211455}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
    ]
    first = assert_clean(harness.post("attempt", history))
    assert_clean(harness.post("attempt", history + first["output"] + [{"role": "user", "content": "next"}]))
    manifest = harness.manifest("attempt")
    assert len(manifest.records) == 2 and not manifest.failures and not manifest.pending_call_ids


def test_an_unexplained_history_rewrite_roots_a_new_segment(monkeypatch, tmp_path):
    """A request whose items extend a stored prefix with something other than
    user/tool observations (a harness compaction that keeps the leading
    context, or an echo shape no candidate explains) still admits as a
    candidate proposing the latest record; the worker decides it cannot
    splice and roots a new segment, and the ledger ends clean with a
    parentless second record instead of a failed call."""
    harness = make_capture_harness(monkeypatch, tmp_path)
    first = assert_clean(harness.post("attempt", HISTORY))
    rewritten = HISTORY + [
        {"role": "assistant", "content": "summary of earlier work (compacted)"},
        {"role": "user", "content": "continue from the summary"},
    ]
    assert first["output"]
    assert_clean(harness.post("attempt", rewritten))
    manifest = harness.manifest("attempt")
    assert len(manifest.records) == 2 and not manifest.failures and not manifest.pending_call_ids
    # The gateway proposed its latest record; rooting is the worker's decision.
    assert harness.worker_calls[1][0].mode == "candidate"
    assert harness.worker_calls[1][0].parent_call_id == manifest.records[0].model_call_id
    # The rewrite rooted a new segment instead of chaining onto the first call.
    assert manifest.records[1].parent_call_id is None


@pytest.mark.parametrize("stream", [False, True])
def test_an_engine_refusal_resolves_the_intent_as_a_failure(monkeypatch, tmp_path, stream):
    """A request the engine refuses (a context-window overflow) must leave a FAILURE
    row behind, not a pending intent: a pending intent makes the attempt's terminal
    receipt unreconcilable and the trainer refuses to seal it, killing the run."""
    import json as _json

    from aiohttp import ClientResponseError

    async def refusing_worker(client, **body):
        request_info = MagicMock(real_url="http://worker.test/v1/chat/completions")
        error = ClientResponseError(request_info, (), status=400, message="Bad Request")
        error.response_content = _json.dumps(
            {"object": "error", "message": "maximum context length is 32768 tokens", "code": 400}
        ).encode()
        raise error

    harness = make_capture_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", refusing_worker)
    response = harness.client.post(
        "/ng-rollout/attempt/training-token-capture/v1/responses",
        json={"input": HISTORY, "stream": stream},
    )
    if stream:
        # The streaming contract reports the refusal as a terminal event.
        assert response.status_code == 200, response.text
        assert "response.failed" in response.text
    else:
        # Without propagate_context_overflow_errors the refusal surfaces as a
        # plain server error; the ledger contract below is what matters.
        assert response.status_code >= 400, response.text
    manifest = harness.manifest("attempt")
    assert not manifest.pending_call_ids, manifest
    assert [f.reason for f in manifest.failures] == ["engine_refused_request"]


def test_streaming_responses_capture_like_their_non_streaming_twin(monkeypatch, tmp_path):
    """SSE-only Responses harnesses (the Codex CLI) send ``stream: true``. The dispatch
    makes exactly one non-streaming worker call through capture and replays the finished
    response as synthesized SSE, so the ledger records the call like any other."""
    harness = make_capture_harness(monkeypatch, tmp_path)
    response = harness.client.post(
        "/ng-rollout/attempt/training-token-capture/v1/responses",
        json={"input": HISTORY, "stream": True},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "response.completed" in response.text
    # Capture transport fields never reach the served stream.
    assert NG_CAPTURE_FIELD not in response.text
    assert NG_COMMIT_COORDS_FIELD not in response.text
    assert "prompt_token_ids" not in response.text
    # One admitted worker call, retries off, recorded in the manifest.
    assert len(harness.worker_calls) == 1
    admission, retry_requests = harness.worker_calls[0]
    assert admission is not None
    assert retry_requests is False
    manifest = harness.manifest("attempt")
    assert len(manifest.records) == 1 and not manifest.failures and not manifest.pending_call_ids


@pytest.mark.parametrize("failure", ["request", "read"])
def test_transport_loss_never_replays_inference(monkeypatch, tmp_path, failure):
    harness = make_capture_harness(monkeypatch, tmp_path, transport_failure=failure)
    assert harness.post("attempt", HISTORY).status_code == 500
    assert len(harness.worker_calls) == len(harness.sink.records) == 1
    assert harness.worker_calls[0][1] is False
    manifest = harness.manifest("attempt")
    assert len(manifest.attempted_call_ids) == len(manifest.pending_call_ids) == 1
