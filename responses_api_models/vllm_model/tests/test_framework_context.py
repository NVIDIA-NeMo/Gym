# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP segment capture through ordinary worker staging and the model ledger."""

import asyncio
import json
from dataclasses import dataclass, field
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientConnectorError, ClientOSError, ClientResponse, ClientResponseError, ServerDisconnectedError
from fastapi.testclient import TestClient

import nemo_gym.base_responses_api_model
import nemo_gym.server_utils
from nemo_gym._checkpoint.control import CheckpointRequest
from nemo_gym.base_responses_api_model import _CaptureMiddleware, _reconstruct_streamed_response
from nemo_gym.chat_streaming import SSE_KEEPALIVE
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_streaming import is_responses_lifecycle_chunk
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.protocols import CAPTURE_REFUSAL_HEADER
from nemo_gym.token_id_capture.sink import NG_CAPTURE_FIELD, NG_COMMIT_COORDS_FIELD
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
from nemo_gym.token_id_capture.staging.records import (
    INVALID_COMMIT_COORDS_REASON,
    CaptureAdmission,
    RolloutManifest,
    StageResult,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
from responses_api_models.vllm_model.tests.test_external_capture_streaming import _request


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
        key = f"{record.capture_key}/{record.model_call_id}"
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

    def post(self, rollout_id, input_items):
        path = f"/ng-rollout/{rollout_id}/training-token-capture/v1/responses" if rollout_id else "/v1/responses"
        return self.client.post(path, json={"input": input_items})

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
    parent_resolution_mode="source_history",
    checkpoint=False,
):
    """Allow paired RL tests to inject their staging transport and prefix reader."""
    sink = sink if sink is not None else MemoryStagingSink()
    fetch_prefix = fetch_prefix if fetch_prefix is not None else sink.prefix
    capture = RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 7)
    config = {
        "token_id_capture": {
            "enabled": True,
            "external_staging": True,
            "parent_resolution": parent_resolution_mode,
            "rebuild_response": False,
            "lineage_store": "nemo_gym.token_id_capture.lineage:FileLineageStore",
            "lineage_store_kwargs": {"root": str(tmp_path / "lineage")},
        }
    }
    if checkpoint:
        config["checkpoint"] = {"enabled": True, "control_auth_token": "test-control"}
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
            checkpoint={"policy": checkpoint},
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict=config),
    )
    app = model.setup_webserver()
    model.setup_exception_middleware(app)
    middleware = next(item for item in app.user_middleware if item.cls is _CaptureMiddleware)
    harness = CaptureHarness(
        TestClient(app, raise_server_exceptions=False), middleware.kwargs["lineage_store"], sink, model=model
    )

    async def worker(client, *, _resend_after_disconnect=True, **body):
        # A real client survives request-scoped model_copy; only its backend is replaced.
        index = len(harness.worker_calls) + 1
        admission = CaptureAdmission.model_validate(body[NG_CAPTURE_FIELD]) if NG_CAPTURE_FIELD in body else None
        harness.worker_calls.append((admission, _resend_after_disconnect))
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
                    capture_key=admission.capture_key, model_call_id=admission.model_call_id, mode="text"
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

    async def worker_through_transport(client, *, _resend_after_disconnect=True, **body):
        async def send(**kwargs):
            assert kwargs["method"] == "POST"
            assert kwargs["url"] == "http://worker.test/v1/chat/completions"
            payload = await worker(
                client, _resend_after_disconnect=_resend_after_disconnect, **json.loads(kwargs["data"])
            )
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
        return await create_chat_completion(client, _resend_after_disconnect=_resend_after_disconnect, **body)

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
    assert_clean(harness.post("attempt", HISTORY))  # A newer rejected response.
    before = harness.manifest("attempt")
    calls = []

    async def tokenize(client, **body):
        assert client.max_http_attempts == nemo_gym.server_utils.MAX_NUM_TRIES
        calls.append(body)
        return {"prompt_token_count": 19, "ng_context_measured": True}

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    response = harness.client.post(
        "/ng-rollout/attempt/training-token-capture/context/measure",
        json={"input": HISTORY + first["output"] + [{"role": "user", "content": "next"}]},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"prompt_token_count": 19}
    assert calls[0][NG_CAPTURE_FIELD]["mode"] == "candidate"
    assert calls[0][NG_CAPTURE_FIELD]["parent_call_id"] == before.records[0].model_call_id
    assert harness.manifest("attempt") == before


@pytest.mark.parametrize("local_sink", [False, True])
@pytest.mark.parametrize("prior_call", [False, True])
def test_measurement_uses_rollout_routing_without_capture_writes(monkeypatch, tmp_path, local_sink, prior_call):
    harness = make_capture_harness(monkeypatch, tmp_path)
    if local_sink:
        observer = MagicMock(mark_incomplete=AsyncMock())
        middleware = next(item for item in harness.client.app.user_middleware if item.cls is _CaptureMiddleware)
        middleware.kwargs["configured_sink"] = observer
    first_client = harness.model._clients[0]
    harness.model._clients.append(first_client.model_copy(update={"base_url": "http://other-worker.test/v1"}))
    # Seed a cookie whose endpoint differs from this rollout's deterministic endpoint.
    assert_clean(harness.post(None, HISTORY))
    session_id = next(iter(harness.model._session_id_to_client))
    rollout_id = next(
        f"rollout-{i}"
        for i in range(100)
        if harness.model._hashed_index(f"rollout-{i}", 2) != harness.model._hashed_index(session_id, 2)
    )
    history = list(HISTORY)
    if prior_call:
        first = assert_clean(harness.post(rollout_id, history))
        history += first["output"] + [{"role": "user", "content": "next"}]
    root = tmp_path / "lineage"
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    measured = []

    async def tokenize(client, **body):
        assert current_rollout_id() == rollout_id
        measured.append((client.base_url, CaptureAdmission.model_validate(body[NG_CAPTURE_FIELD])))
        return {"prompt_token_count": 19, "ng_context_measured": True}

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    path = f"/ng-rollout/{rollout_id}/training-token-capture/context/measure"
    assert_clean(harness.client.post(path, json={"input": history}))
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert measured[0][1].mode == ("candidate" if prior_call else "text")
    expected_client = harness.model._clients[harness.model._hashed_index(rollout_id, 2)]
    assert measured[0][0] == expected_client.base_url
    # A caller that drops cookies still reaches the same endpoint for generation.
    original_worker = NeMoGymAsyncOpenAI.create_chat_completion
    generated = []

    async def generate(client, **body):
        generated.append(client.base_url)
        return await original_worker(client, **body)

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", generate)
    fresh_client = TestClient(harness.client.app, raise_server_exceptions=False)
    assert_clean(
        fresh_client.post(f"/ng-rollout/{rollout_id}/training-token-capture/v1/responses", json={"input": history})
    )
    assert generated == [measured[0][0]]
    manifest = harness.manifest(rollout_id)
    assert len(manifest.records) == len(manifest.attempted_call_ids) == int(prior_call) + 1
    assert not manifest.failures and not manifest.pending_call_ids
    if local_sink:
        observer.mark_incomplete.assert_not_awaited()


def test_measurement_without_rollout_prefix_is_rejected_without_files(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    tokenize = AsyncMock()
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    response = harness.client.post("/context/measure", json={"input": "question"})
    assert response.status_code == 422, response.text
    tokenize.assert_not_awaited()
    assert not harness.worker_calls and list((tmp_path / "lineage").iterdir()) == []


@pytest.mark.parametrize("echo", ["empty", "omitted", "edited_text"])
def test_chat_continuation_tolerates_omitted_empty_tool_calls(monkeypatch, tmp_path, echo):
    harness = make_capture_harness(monkeypatch, tmp_path)
    original_worker = NeMoGymAsyncOpenAI.create_chat_completion

    async def worker(client, **body):
        response = await original_worker(client, **body)
        response["choices"][0]["message"]["tool_calls"] = []
        return response

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", worker)
    path = "/ng-rollout/attempt/training-token-capture/v1/chat/completions"
    history = [{"role": "user", "content": "question"}]
    first = assert_clean(harness.client.post(path, json={"messages": history}))
    message = first["choices"][0]["message"]
    assert message["tool_calls"] == []
    # Harnesses omit empty optional fields rather than returning explicit nulls.
    echoed = {"role": message["role"], "content": message["content"]}
    if echo == "empty":
        echoed["tool_calls"] = []
    elif echo == "edited_text":
        echoed["content"] = "edited answer"
    first_record = harness.manifest("attempt").records[0]
    assert_clean(harness.client.post(path, json={"messages": history + [echoed, {"role": "user", "content": "next"}]}))
    admission = harness.worker_calls[-1][0]
    assert admission.mode == ("text" if echo == "edited_text" else "candidate")
    assert admission.parent_call_id == (None if echo == "edited_text" else first_record.model_call_id)
    assert admission.staging_chain == ([] if echo == "edited_text" else [first_record.staging_key])
    manifest = harness.manifest("attempt")
    assert len(manifest.records) == 2 and not manifest.failures and not manifest.pending_call_ids


@pytest.mark.parametrize("with_parent", [False, True])
@pytest.mark.parametrize(
    "reporting_fields", [{}, {"include": ["reasoning.encrypted_content"]}, {"reasoning": {"summary": "auto"}}]
)
def test_measurement_matches_generation_prompt_and_options(monkeypatch, tmp_path, with_parent, reporting_fields):
    harness = make_capture_harness(monkeypatch, tmp_path)
    harness.model.config.drop_unrepresentable_request_fields = True
    harness.model.config.replace_developer_role_with_system = True
    harness.model.config.chat_template_kwargs = {"enable_thinking": True, "source": "config"}
    harness.model.config.extra_body = {"mm_processor_kwargs": {"max_pixels": 1024}}
    history = [{"role": "developer", "content": "Be concise"}, {"role": "user", "content": "question"}]
    body = {
        "metadata": {"chat_template_kwargs": '{"source":"request"}'},
        "tools": [
            {"type": "function", "name": "lookup", "strict": False, "parameters": {"type": "object", "properties": {}}}
        ],
        **reporting_fields,
    }
    if with_parent:
        first = assert_clean(
            harness.client.post(
                "/ng-rollout/attempt/training-token-capture/v1/responses", json={"input": history, **body}
            )
        )
        history += first["output"] + [{"role": "user", "content": "next"}]
    before = harness.manifest("attempt")
    measured, generated = [], []

    async def tokenize(client, **body):
        measured.append(body)
        return {"prompt_token_count": 19, "ng_context_measured": True}

    original_generate = NeMoGymAsyncOpenAI.create_chat_completion

    async def generate(client, **body):
        generated.append(body)
        return await original_generate(client, **body)

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", generate)
    body["input"] = history
    assert_clean(harness.client.post("/ng-rollout/attempt/training-token-capture/context/measure", json=body))
    assert harness.manifest("attempt") == before
    assert_clean(harness.client.post("/ng-rollout/attempt/training-token-capture/v1/responses", json=body))
    measured_admission = measured[0].pop(NG_CAPTURE_FIELD)
    generated_prompt = harness.model._get_tokenize_chat_body(generated[0])
    generated_admission = generated_prompt.pop(NG_CAPTURE_FIELD)
    assert measured[0] == generated_prompt
    assert measured[0]["messages"][0]["role"] == "system"
    assert measured[0]["tools"][0]["function"]["name"] == "lookup"
    assert measured[0]["chat_template_kwargs"] == {"enable_thinking": True, "source": "request"}
    assert measured[0]["mm_processor_kwargs"] == {"max_pixels": 1024}
    for admission in (measured_admission, generated_admission):
        admission.pop("model_call_id")
    assert measured_admission == generated_admission
    assert measured_admission["mode"] == ("candidate" if with_parent else "text")
    assert len(harness.worker_calls) == int(with_parent) + 1


@pytest.mark.parametrize("effort", ["max", "high"])
def test_measurement_renders_the_reasoning_effort_generation_forwards(monkeypatch, tmp_path, effort):
    # Codex sends effort `max`, which the SDK does not list. Measuring must accept it and render the
    # prompt with the same reasoning_effort generation sends, since chat templates can depend on it.
    harness = make_capture_harness(monkeypatch, tmp_path)
    harness.model.config.forward_unvalidated_reasoning_effort = True
    harness.model.config.drop_unrepresentable_request_fields = True
    body = {
        "input": [{"role": "user", "content": "question"}],
        "reasoning": {"effort": effort, "context": "all_turns"},
        "include": ["reasoning.encrypted_content"],
    }
    measured, generated = [], []

    async def tokenize(client, **body):
        measured.append(body)
        return {"prompt_token_count": 19, "ng_context_measured": True}

    original_generate = NeMoGymAsyncOpenAI.create_chat_completion

    async def generate(client, **body):
        generated.append(body)
        return await original_generate(client, **body)

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", generate)
    assert_clean(harness.client.post("/ng-rollout/attempt/training-token-capture/context/measure", json=body))
    assert_clean(harness.client.post("/ng-rollout/attempt/training-token-capture/v1/responses", json=body))
    measured[0].pop(NG_CAPTURE_FIELD)
    generated_prompt = harness.model._get_tokenize_chat_body(generated[0])
    generated_prompt.pop(NG_CAPTURE_FIELD)
    assert generated_prompt["reasoning_effort"] == effort
    assert measured[0] == generated_prompt


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"metadata": {"chat_template_kwargs": "{"}},
        {"metadata": {"extra_body": '{"n":2}'}},
        {"metadata": {"extra_body": '{"stream":true}'}},
        {"include": ["reasoning.encrypted_content"]},
    ],
)
def test_measurement_render_errors_are_client_errors_without_capture_writes(monkeypatch, tmp_path, invalid_fields):
    harness = make_capture_harness(monkeypatch, tmp_path)
    before = harness.manifest("attempt")
    tokenize = AsyncMock()
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    response = harness.client.post(
        "/ng-rollout/attempt/training-token-capture/context/measure", json={"input": "question", **invalid_fields}
    )
    assert response.status_code == 422, response.text
    assert harness.manifest("attempt") == before
    assert not harness.worker_calls and not harness.sink.records
    tokenize.assert_not_awaited()


@pytest.mark.parametrize("setting", ["is_responses_native", "use_completions_api"])
def test_framework_measurement_cannot_use_an_alternative_render_backend(monkeypatch, tmp_path, setting):
    harness = make_capture_harness(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="chat-backed Responses|does not support use_completions_api"):
        make_capture_harness(
            monkeypatch, tmp_path, model_config=harness.model.config.model_copy(update={setting: True})
        )


@pytest.mark.parametrize("original_question", ["unchanged", "omitted", "changed"])
def test_string_input_continuity_matches_measurement_and_generation(monkeypatch, tmp_path, original_question):
    harness = make_capture_harness(monkeypatch, tmp_path)
    first = assert_clean(harness.post("attempt", "old question"))
    before = harness.manifest("attempt")
    history = first["output"] + [{"role": "user", "content": "continue"}]
    if original_question != "omitted":
        question = "old question" if original_question == "unchanged" else "different question"
        history.insert(0, {"role": "user", "content": question})
    measured = []

    async def tokenize(client, **body):
        measured.append(CaptureAdmission.model_validate(body[NG_CAPTURE_FIELD]))
        return {"prompt_token_count": 19, "ng_context_measured": True}

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    assert_clean(
        harness.client.post("/ng-rollout/attempt/training-token-capture/context/measure", json={"input": history})
    )
    assert harness.manifest("attempt") == before
    assert_clean(harness.post("attempt", history))
    generated = harness.worker_calls[-1][0]
    expected_parent = before.records[0].model_call_id if original_question == "unchanged" else None
    for admission in [measured[0], generated]:
        assert admission.mode == ("candidate" if expected_parent is not None else "text")
        assert admission.parent_call_id == expected_parent
        assert admission.prev_len == (before.records[0].cum_len if expected_parent is not None else 0)
    after = harness.manifest("attempt")
    assert not after.failures and not after.pending_call_ids


def test_lost_capture_ack_is_retained_while_an_independent_call_finishes(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    harness.omit_coords = True
    assert harness.post("attempt", HISTORY).status_code == 500
    manifest = harness.manifest("attempt")
    assert len(manifest.attempted_call_ids) == len(manifest.pending_call_ids) == 1
    assert len(harness.sink.records) == 1
    harness.omit_coords = False
    assert_clean(harness.post("attempt", [{"role": "user", "content": "independent task"}]))
    after = harness.manifest("attempt")
    assert after.pending_call_ids == manifest.pending_call_ids
    assert after.failures == manifest.failures
    assert len(after.records) == 1 and len(after.attempted_call_ids) == 2
    assert len(harness.worker_calls) == len(harness.sink.records) == 2
    assert after.records[0].parent_call_id is None


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("continuation", [False, True])
def test_failed_call_keeps_intent_without_blocking_other_calls(monkeypatch, tmp_path, cancelled, continuation):
    harness = make_capture_harness(monkeypatch, tmp_path)
    first = assert_clean(harness.post("attempt", HISTORY))
    worker = NeMoGymAsyncOpenAI.create_chat_completion

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
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", worker)
    history = (
        HISTORY + first["output"] + [{"role": "user", "content": "continue healthy branch"}]
        if continuation
        else [{"role": "user", "content": "independent task"}]
    )
    assert_clean(harness.post("attempt", history))
    after = harness.manifest("attempt")
    assert after.pending_call_ids == manifest.pending_call_ids
    assert after.failures == manifest.failures
    assert len(after.records) == len(harness.sink.records) == 2
    candidate = harness.worker_calls[-1][0]
    assert candidate.parent_call_id == (manifest.records[0].model_call_id if continuation else None)
    harness.client.close()
    asyncio.run(harness.ledger.close())


@pytest.mark.parametrize(
    "dialect, body",
    [
        ("chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True, "n": 2}),
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


@pytest.mark.parametrize("route", ["responses", "responses-stream", "chat/completions", "measure"])
@pytest.mark.parametrize("prior_call", [False, True])
@pytest.mark.parametrize(
    "extra_body",
    [
        '{"truncate_prompt_tokens":32}',
        '{"truncate_prompt_tokens":0}',
        '{"continue_final_message":true}',
        "{",
        "[]",
    ],
)
def test_invalid_metadata_is_rejected_before_capture_intent(monkeypatch, tmp_path, route, prior_call, extra_body):
    harness = make_capture_harness(monkeypatch, tmp_path)
    if prior_call:
        assert_clean(harness.post("attempt", HISTORY))
    before = harness.manifest("attempt")
    path = f"/ng-rollout/attempt/training-token-capture/v1/{route}"
    body = {"messages" if route == "chat/completions" else "input": HISTORY, "metadata": {"extra_body": extra_body}}
    if route == "responses-stream":
        path = path.removesuffix("-stream")
        body["stream"] = True
    if route == "measure":
        path = "/ng-rollout/attempt/training-token-capture/context/measure"
    response = harness.client.post(path, json=body)
    assert response.status_code == 422, response.text
    assert harness.manifest("attempt") == before
    assert len(harness.worker_calls) == len(harness.sink.records) == int(prior_call)
    assert_clean(harness.post("attempt", [{"role": "user", "content": "valid retry"}]))
    after = harness.manifest("attempt")
    assert not after.pending_call_ids and not after.failures


@pytest.mark.parametrize("route", ["responses", "responses-stream", "measure"])
@pytest.mark.parametrize("prior_call", [False, True])
@pytest.mark.parametrize(
    "body",
    [
        {"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "file-test"}]}]},
        {"input": "task", "tool_choice": "required"},
    ],
)
def test_conversion_error_is_rejected_before_intent(monkeypatch, tmp_path, route, prior_call, body):
    harness = make_capture_harness(monkeypatch, tmp_path)
    if prior_call:
        assert_clean(harness.post("attempt", HISTORY))
    before = harness.manifest("attempt")
    path = "/ng-rollout/attempt/training-token-capture/v1/responses"
    if route == "measure":
        path = "/ng-rollout/attempt/training-token-capture/context/measure"
    response = harness.client.post(path, json=body | {"stream": route == "responses-stream"})
    assert response.status_code == 422, response.text
    assert harness.manifest("attempt") == before
    assert len(harness.worker_calls) == len(harness.sink.records) == int(prior_call)
    assert_clean(harness.post("attempt", [{"role": "user", "content": "valid retry"}]))
    after = harness.manifest("attempt")
    assert not after.pending_call_ids and not after.failures


@pytest.mark.parametrize("route", ["responses", "responses-stream", "measure"])
@pytest.mark.parametrize("prior_call", [False, True])
@pytest.mark.parametrize(
    "item",
    [
        {"type": "web_search_call", "id": "ws1", "status": "completed", "action": {"type": "search", "query": "q"}},
        {"type": "custom_tool_call", "id": "ct1", "call_id": "tool1", "name": "tool", "input": "{}"},
        {"type": "custom_tool_call_output", "call_id": "tool1", "output": "result"},
        {"type": "compaction", "id": "compact1", "encrypted_content": "opaque"},
    ],
    ids=lambda item: item["type"],
)
def test_unsupported_replay_item_is_rejected_before_intent(monkeypatch, tmp_path, route, prior_call, item):
    harness = make_capture_harness(monkeypatch, tmp_path)
    history = list(HISTORY)
    if prior_call:
        first = assert_clean(harness.post("attempt", history))
        history += first["output"]
    history += [item, {"role": "user", "content": "next"}]
    # This is valid Responses input; the rejection must come from replay support.
    NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": history})
    before = harness.manifest("attempt")
    root = tmp_path / "lineage"
    before_files = {path.name: path.read_bytes() for path in root.iterdir()}
    tokenize = AsyncMock()
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    path = "/ng-rollout/attempt/training-token-capture/v1/responses"
    body = {"input": history}
    if route == "responses-stream":
        body["stream"] = True
    elif route == "measure":
        path = "/ng-rollout/attempt/training-token-capture/context/measure"
    response = harness.client.post(path, json=body)
    assert response.status_code == 422, response.text
    # Conversion or replay validation may reject first; both must identify the kind.
    assert item["type"] in response.json()["detail"]
    assert harness.manifest("attempt") == before
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before_files
    assert len(harness.worker_calls) == len(harness.sink.records) == int(prior_call)
    tokenize.assert_not_awaited()
    # Correcting the request can reuse the attempt; no phantom failed call remains.
    assert_clean(harness.post("attempt", [{"role": "user", "content": "valid retry"}]))
    after = harness.manifest("attempt")
    assert len(after.records) == int(prior_call) + 1 and not after.pending_call_ids and not after.failures


@pytest.mark.parametrize(
    "overrides",
    [
        {"continue_final_assistant_message": True},
        {"extra_body": {"truncate_prompt_tokens": 32}},
        {"extra_body": {"continue_final_message": True}},
        {"sampling_overrides": {"truncate_prompt_tokens": 32}},
        {"sampling_overrides": {"continue_final_message": True}},
    ],
)
@pytest.mark.parametrize("framework_owned", [False, True])
def test_framework_rejects_unsupported_server_options_at_startup(monkeypatch, tmp_path, overrides, framework_owned):
    harness = make_capture_harness(monkeypatch, tmp_path)
    config = harness.model.config.model_copy(update=overrides)
    if framework_owned:
        with pytest.raises(ValueError, match="truncation|prefill|continue_final_assistant_message"):
            make_capture_harness(monkeypatch, tmp_path, model_config=config)
    else:
        make_capture_harness(monkeypatch, tmp_path, model_config=config, parent_resolution_mode="fingerprint")


@pytest.mark.parametrize("dialect", ["responses", "chat/completions"])
@pytest.mark.parametrize(
    "options",
    [
        {"truncate_prompt_tokens": None},
        {"continue_final_message": False},
        {"max_tokens": 4},
    ],
)
def test_supported_metadata_options_still_generate(monkeypatch, tmp_path, dialect, options):
    harness = make_capture_harness(monkeypatch, tmp_path)
    body = {
        "messages" if dialect == "chat/completions" else "input": HISTORY,
        "metadata": {"extra_body": json.dumps(options)},
    }
    assert_clean(harness.client.post(f"/ng-rollout/attempt/training-token-capture/v1/{dialect}", json=body))
    manifest = harness.manifest("attempt")
    assert len(manifest.records) == 1 and not manifest.failures and not manifest.pending_call_ids


@pytest.mark.parametrize("dialect", ["responses", "chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
async def test_checkpoint_gate_delivers_framework_capture(monkeypatch, tmp_path, dialect, stream):
    harness = make_capture_harness(monkeypatch, tmp_path, checkpoint=True)
    messages = await _request(
        harness.client.app,
        f"/ng-rollout/attempt/training-token-capture/v1/{dialect}",
        {"messages" if dialect == "chat/completions" else "input": HISTORY, "stream": stream},
    )
    assert messages[0]["status"] == 200
    raw = b"".join(message.get("body", b"") for message in messages)
    wire_dialect = "chat_completions" if dialect == "chat/completions" else "responses"
    response = _reconstruct_streamed_response(raw, wire_dialect) if stream else json.loads(raw)
    assert response is not None
    if dialect == "responses":
        assert response["status"] == "completed"
    manifest = RolloutManifest.model_validate(await harness.ledger.manifest("attempt"))
    assert len(manifest.records) == len(harness.sink.records) == 1
    assert manifest.records[0].response_id == response["id"]
    assert not manifest.pending_call_ids and not manifest.failures
    assert not harness.client.app.state.nemo_gym_policy_gate.tickets


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("dialect", ["responses", "chat/completions"])
async def test_checkpoint_gate_keeps_framework_capture_pending_until_delivery(
    monkeypatch, tmp_path, stream, cancelled, dialect
):
    harness = make_capture_harness(monkeypatch, tmp_path, checkpoint=True)
    gate = harness.client.app.state.nemo_gym_policy_gate
    staged, release, delivering = asyncio.Event(), asyncio.Event(), asyncio.Event()
    worker, deliver = NeMoGymAsyncOpenAI.create_chat_completion, gate.deliver_response
    sent = []

    async def held_worker(client, **body):
        payload = await worker(client, **body)
        staged.set()
        await release.wait()
        return payload

    async def held_delivery(ticket):
        delivering.set()
        await deliver(ticket)

    def answer(body) -> bool:
        # Keep-alive comments and Responses lifecycle events are liveness: they may leave before the row.
        return bool(body) and body != SSE_KEEPALIVE and not is_responses_lifecycle_chunk(body)

    async def observe(message):
        if answer(message.get("body")):
            manifest = RolloutManifest.model_validate(await harness.ledger.manifest("attempt"))
            assert len(manifest.records) == 1 and not manifest.pending_call_ids
        sent.append(message)

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", held_worker)
    monkeypatch.setattr(gate, "deliver_response", held_delivery)
    call = asyncio.create_task(
        _request(
            harness.client.app,
            f"/ng-rollout/attempt/training-token-capture/v1/{dialect}",
            {"messages" if dialect == "chat/completions" else "input": HISTORY, "stream": stream},
            send=observe,
        )
    )
    try:
        await asyncio.wait_for(staged.wait(), timeout=5)
        await gate.close(CheckpointRequest(checkpoint_id="held", deadline_ts=time() + 5))
        release.set()
        await asyncio.wait_for(delivering.wait(), timeout=5)
        context = next(iter(gate.tickets)).capture
        assert context.deferred_commit is not None and not context.committed
        before = RolloutManifest.model_validate(await harness.ledger.manifest("attempt"))
        assert not before.records and not before.failures
        assert before.pending_call_ids == [context.model_call_id]
        assert not any(answer(message.get("body")) for message in sent)
        if cancelled:
            call.cancel()
            await asyncio.gather(call, return_exceptions=True)
            assert not context.committed
            assert RolloutManifest.model_validate(await harness.ledger.manifest("attempt")) == before
        else:
            gate.open()
            await asyncio.wait_for(call, timeout=5)
            assert context.committed
            assert any(answer(message.get("body")) for message in sent)
    finally:
        release.set()
        gate.open()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)
    assert not gate.tickets


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("dialect", ["responses", "chat/completions"])
async def test_checkpoint_publication_failure_cannot_serve_a_successful_response(
    monkeypatch, tmp_path, stream, dialect
):
    harness = make_capture_harness(monkeypatch, tmp_path, checkpoint=True)
    sent = []

    async def fail_publication(commit):
        raise OSError("ledger unavailable")

    async def observe(message):
        sent.append(message)

    monkeypatch.setattr(harness.ledger, "record", fail_publication)
    await asyncio.gather(
        _request(
            harness.client.app,
            f"/ng-rollout/attempt/training-token-capture/v1/{dialect}",
            {"messages" if dialect == "chat/completions" else "input": HISTORY, "stream": stream},
            send=observe,
        ),
        return_exceptions=True,
    )
    manifest = RolloutManifest.model_validate(await harness.ledger.manifest("attempt"))
    assert len(harness.sink.records) == len(manifest.pending_call_ids) == 1
    assert not manifest.records
    assert any(failure.reason == INVALID_COMMIT_COORDS_REASON for failure in manifest.failures)
    raw = b"".join(message.get("body", b"") for message in sent)
    assert b"response.completed" not in raw and b"answer" not in raw and b"[DONE]" not in raw
    assert not harness.client.app.state.nemo_gym_policy_gate.tickets


def test_framework_checkpoint_restores_a_closed_attempt_and_accepts_its_next_call(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path / "live", checkpoint=True)
    first = assert_clean(harness.post("episode", HISTORY))
    first_record = harness.manifest("episode").records[0]
    asyncio.run(harness.ledger.close_admissions("episode"))
    headers = {"Authorization": "Bearer test-control"}
    directory = str(tmp_path / "checkpoint")

    def control(client, operation, checkpoint_id, **fields):
        response = client.post(
            f"/ng-control/v1/checkpoint/{operation}",
            headers=headers,
            json={"checkpoint_id": checkpoint_id, "deadline_ts": time() + 5, **fields},
        )
        assert response.status_code == 200, response.text

    control(harness.client, "prepare", "saved")
    control(harness.client, "commit", "saved", checkpoint_dir=directory, episode_ids=[{"rollout_id": "episode"}])
    control(harness.client, "resume", "saved")
    restored = make_capture_harness(monkeypatch, tmp_path / "restored", checkpoint=True, sink=harness.sink)
    # Keep synthetic response IDs unique across the simulated server restart.
    restored.worker_calls.extend(harness.worker_calls)
    control(restored.client, "restore", "resumed", checkpoint_dir=directory, episode_ids=[{"rollout_id": "episode"}])
    control(restored.client, "resume", "resumed")
    assert_clean(restored.post("episode-a1", HISTORY + first["output"] + [{"role": "user", "content": "next"}]))
    manifest = restored.manifest("episode-a1")
    assert len(manifest.records) == 2 and not manifest.pending_call_ids and not manifest.failures
    assert manifest.records[0].source_capture_key == "episode"
    candidate = restored.worker_calls[-1][0]
    assert candidate.parent_call_id == first_record.model_call_id
    assert candidate.staging_chain == [first_record.staging_key]
    assert harness.post("episode", HISTORY).status_code == 409


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
    """Gym roots source rewrites before they reach the worker."""
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
    assert harness.worker_calls[1][0].mode == "text"
    assert harness.worker_calls[1][0].parent_call_id is None
    # The rewrite rooted a new segment instead of chaining onto the first call.
    assert manifest.records[1].parent_call_id is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [400, 429, 500])
def test_engine_errors_preserve_unacknowledged_intents(monkeypatch, tmp_path, stream, status):
    """An HTTP error alone does not acknowledge that staging stopped."""
    import json as _json

    from aiohttp import ClientResponseError

    async def refusing_worker(client, **body):
        request_info = MagicMock(real_url="http://worker.test/v1/chat/completions")
        error = ClientResponseError(request_info, (), status=status, message="Engine error")
        error.response_content = _json.dumps(
            {"object": "error", "message": "maximum context length is 32768 tokens", "code": status}
        ).encode()
        raise error

    harness = make_capture_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", refusing_worker)
    response = harness.client.post(
        "/ng-rollout/attempt/training-token-capture/v1/responses",
        json={"input": HISTORY, "stream": stream},
    )
    assert response.status_code == status, response.text
    if status == 400:
        assert response.json()["error"]["code"] == "context_length_exceeded"
    manifest = harness.manifest("attempt")
    assert len(manifest.pending_call_ids) == 1, manifest
    assert [f.reason for f in manifest.failures] == ["request_finished_without_staged_coordinates"]


@pytest.mark.parametrize("dialect", ["responses", "chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("acknowledgement", ["correct", "missing", "wrong-call"])
def test_context_refusal_settles_capture_only_with_worker_ack(monkeypatch, tmp_path, dialect, stream, acknowledgement):
    harness = make_capture_harness(monkeypatch, tmp_path)
    first = assert_clean(harness.post("attempt", HISTORY))
    before = harness.manifest("attempt")
    worker = NeMoGymAsyncOpenAI.create_chat_completion
    refused_calls = []

    async def refuse(client, **body):
        admission = CaptureAdmission.model_validate(body[NG_CAPTURE_FIELD])
        refused_calls.append(admission.model_call_id)
        headers = {}
        if acknowledgement != "missing":
            headers[CAPTURE_REFUSAL_HEADER] = (
                admission.model_call_id if acknowledgement == "correct" else "another-call"
            )
        error = ClientResponseError(MagicMock(real_url="http://worker.test"), (), status=400, headers=headers)
        error.response_content = b'{"error":{"message":"maximum context length is 32768 tokens","code":400}}'
        raise error

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", refuse)
    history = HISTORY + first["output"] + [{"role": "user", "content": "too much context"}]
    body = {"input": history} if dialect == "responses" else {"messages": HISTORY}
    response = harness.client.post(
        f"/ng-rollout/attempt/training-token-capture/v1/{dialect}", json={**body, "stream": stream}
    )
    assert_clean(response, status=400)
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert response.json()["error"]["message"] == "maximum context length is 32768 tokens"
    assert CAPTURE_REFUSAL_HEADER not in response.headers
    assert len(harness.sink.records) == 1  # Refusal generated and staged nothing.
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", worker)
    compacted = [{"role": "user", "content": "summary of prior work; continue"}]
    if acknowledgement != "correct":
        manifest = harness.manifest("attempt")
        assert manifest.pending_call_ids == refused_calls
        assert manifest.failures
        # Serving may continue, but this attempt must remain untrainable.
        assert_clean(harness.post("attempt", compacted))
        after = harness.manifest("attempt")
        assert after.pending_call_ids == manifest.pending_call_ids
        assert after.failures == manifest.failures
        assert len(after.records) == 2
        return

    assert harness.manifest("attempt") == before
    assert_clean(harness.post("attempt", compacted))
    manifest = harness.manifest("attempt")
    assert len(manifest.records) == 2
    assert not manifest.pending_call_ids and not manifest.failures
    assert manifest.attempted_call_ids == [record.model_call_id for record in manifest.records]
    assert manifest.records[1].parent_call_id is None
    assert harness.sink.prefix([manifest.records[1].staging_key]) == [20, 21, 1002]


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("keepalive", [False, True])
async def test_streaming_chat_continuation_commits_before_answer(monkeypatch, tmp_path, reasoning, keepalive):
    harness = make_capture_harness(monkeypatch, tmp_path, reasoning=reasoning)
    worker = NeMoGymAsyncOpenAI.create_chat_completion
    live = asyncio.Event()
    if keepalive:
        monkeypatch.setattr(nemo_gym.base_responses_api_model, "_CHAT_KEEPALIVE_SECONDS", 0.001)

    async def generate(client, **body):
        assert body["stream"] is False and "stream_options" not in body
        payload = await worker(client, **body)
        if keepalive:
            await live.wait()
        return payload

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", generate)
    history = list(HISTORY)
    for turn in range(1, 3):
        live.clear()

        async def observe(message):
            chunk = message.get("body")
            if chunk == SSE_KEEPALIVE:
                live.set()
            elif chunk and chunk.startswith(b"data:"):
                manifest = RolloutManifest.model_validate(await harness.ledger.manifest("attempt"))
                assert len(manifest.records) == turn and not manifest.pending_call_ids and not manifest.failures

        sent = await asyncio.wait_for(
            _request(
                harness.client.app,
                "/ng-rollout/attempt/training-token-capture/v1/chat/completions",
                {"messages": history, "stream": True, "stream_options": {"include_usage": True}},
                send=observe,
            ),
            timeout=5,
        )
        assert sent[0]["status"] == 200
        raw = b"".join(message.get("body", b"") for message in sent)
        assert b"[DONE]" in raw
        for field in (NG_CAPTURE_FIELD, NG_COMMIT_COORDS_FIELD, "prompt_token_ids", "generation_token_ids"):
            assert field.encode() not in raw
        response = _reconstruct_streamed_response(raw, "chat_completions")
        assert response is not None and response["choices"][0]["finish_reason"] == "stop"
        message = response["choices"][0]["message"]
        assert "answer" in message["content"]
        if reasoning:
            assert f"reasoning {turn}" in message["content"]
        manifest = RolloutManifest.model_validate(await harness.ledger.manifest("attempt"))
        assert manifest.records[-1].response_id == response["id"]
        history += [message, {"role": "user", "content": "next"}]
    assert len(harness.worker_calls) == len(harness.sink.records) == 2
    assert harness.worker_calls[-1][0].parent_call_id == manifest.records[0].model_call_id
    assert all(resend is False for _, resend in harness.worker_calls)


@pytest.mark.parametrize("delay_failure_write", [False, True])
async def test_cancelled_chat_keepalive_retains_uncertain_capture(monkeypatch, tmp_path, delay_failure_write):
    import threading

    harness = make_capture_harness(monkeypatch, tmp_path)
    failure_written = threading.Event()
    record_failure = harness.ledger._record_failure

    def persist_failure(*args):
        record_failure(*args)
        failure_written.set()

    monkeypatch.setattr(harness.ledger, "_record_failure", persist_failure)
    if delay_failure_write:
        record_failure_async = harness.ledger.record_failure

        async def delayed_failure(*args):
            # Exercise cancellation before the ledger submits work to its executor.
            await asyncio.sleep(0.01)
            await record_failure_async(*args)

        monkeypatch.setattr(harness.ledger, "record_failure", delayed_failure)
    worker = NeMoGymAsyncOpenAI.create_chat_completion
    live, staged, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sent = []
    monkeypatch.setattr(nemo_gym.base_responses_api_model, "_CHAT_KEEPALIVE_SECONDS", 0.001)

    async def held_worker(client, **body):
        await worker(client, **body)
        staged.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def observe(message):
        sent.append(message)
        if message.get("body") == SSE_KEEPALIVE:
            live.set()

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", held_worker)
    call = asyncio.create_task(
        _request(
            harness.client.app,
            "/ng-rollout/attempt/training-token-capture/v1/chat/completions",
            {"messages": HISTORY, "stream": True},
            send=observe,
        )
    )
    try:
        await asyncio.wait_for(asyncio.gather(live.wait(), staged.wait()), timeout=5)
    finally:
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
    assert cancelled.is_set()
    # Cancellation can stop awaiting the ledger write while its worker thread finishes it.
    assert await asyncio.to_thread(failure_written.wait, 5)
    manifest = RolloutManifest.model_validate(await harness.ledger.manifest("attempt"))
    assert len(harness.worker_calls) == len(harness.sink.records) == len(manifest.pending_call_ids) == 1
    assert not manifest.records and manifest.failures
    assert not any(message.get("body") not in (None, b"", SSE_KEEPALIVE) for message in sent)


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
    # One admitted worker call, resending disabled, recorded in the manifest.
    assert len(harness.worker_calls) == 1
    admission, resend_after_disconnect = harness.worker_calls[0]
    assert admission is not None
    assert resend_after_disconnect is False
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


@pytest.mark.parametrize(
    "failure",
    [
        "connect",
        "connect_exhausted",
        "connect_exhausted_custom",
        "disconnect",
        "socket",
        "timeout",
        "429",
        "503",
        "503_after_stage",
        "quota",
        "auth",
    ],
)
@pytest.mark.parametrize("internal", [False, True])
def test_capture_transport_retries_only_before_connection(monkeypatch, tmp_path, failure, internal):
    real_generate = NeMoGymAsyncOpenAI.create_chat_completion
    harness = make_capture_harness(monkeypatch, tmp_path)
    harness.model._clients[0].internal = internal
    if failure == "connect_exhausted_custom":
        harness.model._clients[0].max_connection_retries = 2
    fake_worker = NeMoGymAsyncOpenAI.create_chat_completion
    wire_calls = []
    refused = ClientConnectorError(
        SimpleNamespace(host="worker.test", port=80, ssl=False), ConnectionRefusedError("worker restarting")
    )

    async def send(**kwargs):
        wire_calls.append(kwargs)
        payload_body = json.loads(kwargs["data"])
        assert "_resend_after_disconnect" not in payload_body
        if failure.startswith("connect_exhausted") or failure == "connect" and len(wire_calls) == 1:
            if len(wire_calls) > nemo_gym.server_utils.MAX_NUM_TRIES:
                raise RuntimeError("connection retry limit was not applied")
            raise refused
        if failure in ("429", "503", "503_after_stage", "quota", "auth") and len(wire_calls) == 1:
            if failure == "503_after_stage":
                await fake_worker(harness.model._clients[0], **payload_body)
            status = {"429": 429, "503": 503, "503_after_stage": 503, "quota": 429, "auth": 401}[failure]
            content = json.dumps(
                {"error": {"code": {"quota": "insufficient_quota", "auth": "invalid_api_key"}.get(failure, "busy")}}
            ).encode()
            response = MagicMock(spec=ClientResponse, status=status, ok=False, headers={})
            response.content.read = AsyncMock(return_value=content)
            response.read = AsyncMock(return_value=content)
            response.raise_for_status.side_effect = ClientResponseError(
                SimpleNamespace(url=kwargs["url"], real_url=kwargs["url"], method="POST", headers={}),
                (),
                status=status,
                message="worker refused",
                headers={},
            )
            return response
        payload = await fake_worker(harness.model._clients[0], **payload_body)
        if failure in ("disconnect", "socket", "timeout"):
            raise {"disconnect": ServerDisconnectedError, "socket": ClientOSError, "timeout": TimeoutError}[failure](
                "worker executed; reply lost"
            )
        response = MagicMock(spec=ClientResponse, status=200, ok=True)
        response.read = AsyncMock(return_value=json.dumps(payload).encode())
        return response

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", real_generate)
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", lambda: MagicMock(request=send))
    monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())
    response = harness.post("attempt", HISTORY)
    if failure == "connect":
        assert_clean(response)
        assert len(wire_calls) == 2 and len(harness.sink.records) == len(harness.worker_calls) == 1
        assert wire_calls[0]["data"] == wire_calls[1]["data"]
        manifest = harness.manifest("attempt")
        assert len(manifest.records) == 1 and not manifest.pending_call_ids and not manifest.failures
        assert len(manifest.attempted_call_ids) == 1
    else:
        assert response.status_code != 200
        if failure in ("429", "503", "503_after_stage", "quota", "auth"):
            assert (
                response.status_code
                == {"429": 429, "503": 503, "503_after_stage": 503, "quota": 429, "auth": 401}[failure]
            )
        expected_calls = {"connect_exhausted": nemo_gym.server_utils.MAX_NUM_TRIES, "connect_exhausted_custom": 2}.get(
            failure, 1
        )
        assert len(wire_calls) == expected_calls
        manifest = harness.manifest("attempt")
        assert not manifest.records and len(manifest.pending_call_ids) == len(manifest.failures) == 1
        assert len(harness.sink.records) == int(failure in ("disconnect", "socket", "timeout", "503_after_stage"))
    if failure in ("quota", "auth"):
        assert harness.post("attempt", HISTORY).status_code != 200
        assert len(wire_calls) == 1  # The cached client remembers a permanent endpoint failure.
    assert harness.model._clients[0].max_http_attempts == nemo_gym.server_utils.MAX_NUM_TRIES


def test_closed_capture_refuses_dispatch_without_poisoning(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    assert_clean(harness.post("attempt", [{"role": "user", "content": "task"}]))
    path = "/training-token-capture/control/rollouts/attempt/close-admission"
    assert harness.client.post(path).status_code == 401
    headers = {"Authorization": "Bearer test-control"}
    assert harness.client.post(path.removesuffix("-admission"), headers=headers).status_code == 404
    response = harness.client.post(path, headers=headers)
    assert response.status_code == 200
    snapshot = response.json()
    assert harness.client.post(path, headers=headers).json() == snapshot
    refused = harness.post("attempt", [{"role": "user", "content": "late"}])
    assert refused.status_code == 409
    assert "closed" in refused.json()["detail"]
    assert len(harness.worker_calls) == 1
    assert harness.manifest("attempt").model_dump() == snapshot
    assert_clean(harness.post("other", [{"role": "user", "content": "other rollout"}]))


@pytest.mark.parametrize("first", [0, 1])
def test_http_calls_overlap_and_drain_after_close(monkeypatch, tmp_path, first):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    harness = make_capture_harness(monkeypatch, tmp_path)
    entered, release = [Event(), Event()], [Event(), Event()]
    lock, admitted = Lock(), []
    worker = NeMoGymAsyncOpenAI.create_chat_completion

    async def paused_worker(client, **body):
        with lock:
            index = len(admitted)
            admitted.append(body[NG_CAPTURE_FIELD]["model_call_id"])
        entered[index].set()
        assert await asyncio.to_thread(release[index].wait, 5)
        return await worker(client, **body)

    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", paused_worker)
    with harness.client, ThreadPoolExecutor(max_workers=2) as executor:
        calls = []
        try:
            for index in range(2):
                calls.append(executor.submit(harness.post, "attempt", [{"role": "user", "content": f"task {index}"}]))
                assert entered[index].wait(5), "second call failed to reach the worker concurrently"
            response = harness.client.post(
                "/training-token-capture/control/rollouts/attempt/close-admission",
                headers={"Authorization": "Bearer test-control"},
            )
            assert response.status_code == 200
            assert set(response.json()["pending_call_ids"]) == set(admitted)
            for index in (first, 1 - first):
                release[index].set()
                assert_clean(calls[index].result(timeout=5))
            manifest = harness.manifest("attempt")
            assert [r.model_call_id for r in manifest.records] == [admitted[first], admitted[1 - first]]
            assert not manifest.pending_call_ids and not manifest.failures
        finally:
            for event in release:
                event.set()


@pytest.mark.parametrize("route", ["responses", "responses-stream", "chat/completions", "measure"])
@pytest.mark.parametrize("prior_failure", [False, True])
def test_ambiguous_capture_roots_without_choosing_a_parent(monkeypatch, tmp_path, route, prior_failure):
    harness = make_capture_harness(monkeypatch, tmp_path)
    if route == "chat/completions":
        path = "/ng-rollout/attempt/training-token-capture/v1/chat/completions"
        first = assert_clean(harness.client.post(path, json={"messages": HISTORY}))
        assert_clean(harness.client.post(path, json={"messages": HISTORY}))
        output = [{"role": "assistant", "content": first["choices"][0]["message"]["content"]}]
    else:
        first = assert_clean(harness.post("attempt", HISTORY))
        assert_clean(harness.post("attempt", HISTORY))
        output = [{**item, "id": "exporter-local-id"} for item in first["output"]]
    if prior_failure:
        asyncio.run(harness.ledger.record_failure("attempt", "failed-call", "lost acknowledgement"))
    before = harness.manifest("attempt")
    # Equal source history from two generations supplies no occurrence identity.
    history = HISTORY + output + [{"role": "user", "content": "next"}]
    path = f"/ng-rollout/attempt/training-token-capture/v1/{route}"
    if route == "measure":
        path = "/ng-rollout/attempt/training-token-capture/context/measure"
        tokenize = AsyncMock(return_value={"prompt_token_count": 19, "ng_context_measured": True})
        monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    body = {"messages" if route == "chat/completions" else "input": history}
    if route == "responses-stream":
        path = path.removesuffix("-stream")
        body["stream"] = True
    response = harness.client.post(path, json=body)
    assert response.status_code == 200, response.text
    if route == "measure":
        admission = CaptureAdmission.model_validate(tokenize.call_args.kwargs[NG_CAPTURE_FIELD])
        assert harness.manifest("attempt") == before
        assert len(harness.worker_calls) == 2
    else:
        admission = harness.worker_calls[-1][0]
        after = harness.manifest("attempt")
        assert len(after.records) == len(harness.sink.records) == 3
        assert after.records[:2] == before.records
        assert after.failures == before.failures and not after.pending_call_ids
        assert after.records[-1].parent_call_id is None
        assert after.records[-1].root_reason == "ambiguous_candidate"
    assert admission.mode == "text"
    assert admission.parent_call_id is None and admission.staging_chain == []
    assert admission.prev_len == 0 and admission.root_reason == "ambiguous_candidate"


@pytest.mark.parametrize("route", ["responses", "responses-stream", "chat/completions", "measure"])
@pytest.mark.parametrize(
    "fault,reason",
    [
        ("missing_replay", "missing source replay evidence"),
        ("missing_ancestor", "Candidate storage ancestor is missing"),
        ("cycle", "Cyclic candidate storage chain"),
    ],
)
def test_unusable_restored_history_is_refused_without_mutating_capture(monkeypatch, tmp_path, route, fault, reason):
    harness = make_capture_harness(monkeypatch, tmp_path)
    first = assert_clean(
        harness.client.post(
            "/ng-rollout/attempt/training-token-capture/v1/chat/completions", json={"messages": HISTORY}
        )
    )
    rows = harness.ledger.export_rows("attempt")
    record = next(row for row in rows if row.get("staging_key"))
    if fault == "missing_replay":
        record.pop("replay")
    else:
        record.update(
            parent_call_id=record["model_call_id"] if fault == "cycle" else "missing-call",
            prev_len=1,
            delta_len=record["cum_len"] - 1,
            mode="token_in",
        )
    # Import the damaged history through the actual checkpoint ledger surface.
    harness.ledger.import_rows("attempt", rows)
    before = harness.manifest("attempt")
    root = tmp_path / "lineage"
    before_files = {path.name: path.read_bytes() for path in root.iterdir()}
    tokenize = AsyncMock()
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_tokenize", tokenize)
    output = {"role": "assistant", "content": first["choices"][0]["message"]["content"]}
    history = HISTORY + [output, {"role": "user", "content": "next"}]
    path = f"/ng-rollout/attempt/training-token-capture/v1/{route}"
    body = {"messages" if route == "chat/completions" else "input": history}
    if route == "measure":
        path = "/ng-rollout/attempt/training-token-capture/context/measure"
    elif route == "responses-stream":
        path = path.removesuffix("-stream")
        body["stream"] = True
    for _ in range(2):
        response = harness.client.post(path, json=body)
        assert response.status_code == 409, response.text
        assert reason in response.json()["detail"]
        assert harness.manifest("attempt") == before
        assert {path.name: path.read_bytes() for path in root.iterdir()} == before_files
    tokenize.assert_not_awaited()
    assert len(harness.worker_calls) == len(harness.sink.records) == 1


@pytest.mark.parametrize("n", [None, 1])
@pytest.mark.parametrize("stream", [False, True])
def test_chat_accepts_single_choice_default(monkeypatch, tmp_path, n, stream):
    harness = make_capture_harness(monkeypatch, tmp_path)
    response = harness.client.post(
        "/ng-rollout/attempt/training-token-capture/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "task"}], "n": n, "stream": stream},
    )
    if stream:
        assert response.status_code == 200, response.text
        assert _reconstruct_streamed_response(response.content, "chat_completions") is not None
    else:
        assert_clean(response)
    assert len(harness.manifest("attempt").records) == 1


def test_internal_admission_fault_is_not_a_client_refusal(monkeypatch, tmp_path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(harness.ledger, "begin_call", AsyncMock(side_effect=ValueError("ledger bug")))
    assert harness.post("attempt", HISTORY).status_code == 500
    assert not harness.worker_calls


@pytest.mark.parametrize("change", ["dropped_template", "forwarded_template", "overridden_model"])
def test_parent_choice_uses_effective_options_after_gateway_overrides(monkeypatch, tmp_path, change):
    harness = make_capture_harness(monkeypatch, tmp_path)
    harness.model.config.forward_request_chat_template_kwargs = change == "forwarded_template"
    path = "/ng-rollout/attempt/training-token-capture/v1/chat/completions"
    first = assert_clean(harness.client.post(path, json={"model": "ignored", "messages": HISTORY}))
    message = first["choices"][0]["message"]
    items = HISTORY + [{"role": message["role"], "content": message["content"]}, {"role": "user", "content": "next"}]
    payload = {"model": "different-ignored-model", "messages": items}
    if change != "overridden_model":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    assert_clean(harness.client.post(path, json=payload))
    admission = harness.worker_calls[1][0]
    root = change == "forwarded_template"
    assert admission.mode == ("text" if root else "candidate")
    assert admission.parent_call_id == (None if root else harness.manifest("attempt").records[0].model_call_id)
    assert bool(admission.staging_chain) is not root


@pytest.mark.parametrize("path", ["v1/responses", "v1/chat/completions", "measure"])
def test_retired_capture_refuses_dispatch_and_measurement(monkeypatch, tmp_path, path):
    harness = make_capture_harness(monkeypatch, tmp_path)
    assert_clean(harness.post("attempt", HISTORY))
    response = harness.client.post(
        "/training-token-capture/control/rollouts/retire",
        json={"rollout_ids": ["attempt"]},
        headers={"Authorization": "Bearer test-control"},
    )
    assert response.status_code == 200, response.text
    url = (
        "/ng-rollout/attempt/training-token-capture/context/measure"
        if path == "measure"
        else f"/ng-rollout/attempt/training-token-capture/{path}"
    )
    body = {"messages" if path.endswith("completions") else "input": HISTORY}
    response = harness.client.post(url, json=body)
    assert response.status_code == 409, response.text
    assert "retired" in response.json()["detail"]
    assert len(harness.worker_calls) == 1
