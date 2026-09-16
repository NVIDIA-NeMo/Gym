# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Buffered SSE through real model routes, worker staging, and a file ledger."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym import chat_streaming, responses_streaming
from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.base_responses_api_model import _reconstruct_streamed_response
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.records import UNCOMMITTED_CALL_REASON
from nemo_gym.token_id_capture.sink import current_capture_context
from nemo_gym.token_id_capture.staging import resolve_terminal
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
from nemo_gym.token_id_capture.staging.rebuild import verify_and_linearize
from nemo_gym.token_id_capture.staging.records import (
    CaptureAdmission,
    RolloutManifest,
    RolloutReceipt,
    StagedCallBaseSnapshot,
    StageResult,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
from responses_api_models.vllm_model_with_compaction.app import VLLMModelWithCompaction


PREFIX = "/ng-rollout/r1/training-token-capture"
DIALECTS = ["chat/completions", "responses", "messages", "compaction"]


class _Worker:
    def __init__(self):
        self.records = {}
        self.requests = []
        self.context = None
        self.tool_call = False
        self.reasoning = False
        self.refusal = False
        self.capture = RolloutTokenCapture(sink=self, weight_version_fn=lambda: 7, adapter=VLLMCaptureAdapter())

    def stage(self, record):
        self.records[record.staging_key] = record
        return StageResult(ok=True, staging_key=record.staging_key)

    def fetch(self, keys):
        return [
            StagedCallBaseSnapshot.model_validate(self.records[key].model_dump(exclude={"extras"})) for key in keys
        ]

    async def create_chat_completion(self, **body):
        self.requests.append(body)
        self.context = current_capture_context()
        turn = len(self.requests)
        payload = {
            "id": f"completion-{turn}",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": f"answer {turn}",
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if self.reasoning:
            payload["choices"][0]["message"]["reasoning_content"] = "Check the requested calculation."
        if self.refusal:
            payload["choices"][0]["message"].update(content=None, refusal="I cannot help with that.")
        if self.tool_call and turn == 1:
            payload["choices"][0]["finish_reason"] = "tool_calls"
            payload["choices"][0]["message"].update(
                content=None,
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": body["tools"][0]["function"]["name"],
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            )
        if "ng_capture" not in body:
            return payload
        admission = CaptureAdmission.model_validate(body["ng_capture"])
        prefix = [token for key in admission.staging_chain for token in self.records[key].token_ids_delta]
        call = self.capture.begin_call(admission, prefix_token_ids=prefix, stream=body["stream"])
        prompt = prefix + [turn * 10]
        payload["choices"][0]["message"].update(
            prompt_token_ids=prompt,
            generation_token_ids=[turn * 10 + 1],
            generation_log_probs=[-0.25],
            routed_experts=[[[0]]] * (len(prompt) + 1),
        )
        coords = await asyncio.to_thread(self.capture.complete_call_from_response, call, payload)
        payload["ng_commit_coords"] = coords.model_dump()
        return payload


@pytest.fixture
def make_harness(tmp_path, monkeypatch):
    def make(dialect="responses", evaluation=False, reasoning=False, **overrides):
        root = tmp_path / dialect.replace("/", "-")
        global_config = {
            "token_id_capture": {
                "enabled": True,
                "external_staging": True,
                "rebuild_response": False,
                "lineage_store": "nemo_gym.token_id_capture.lineage:FileLineageStore",
                "lineage_store_kwargs": {"root": str(root)},
            }
        }
        if evaluation:
            global_config.update(observability_enabled=True, model_call_capture_dir=str(root))
        monkeypatch.setenv("NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN", "test-control-token")
        cls = VLLMModelWithCompaction if dialect == "compaction" else VLLMModel
        config = VLLMModelConfig(
            host="localhost",
            port=8080,
            entrypoint="",
            name="test",
            model="test",
            base_url="http://worker/v1",
            api_key="unused",
            return_token_id_information=False,
            uses_reasoning_parser=reasoning,
            **overrides,
        )
        model = cls(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict=global_config))
        worker = _Worker()
        worker.reasoning = reasoning
        model._clients = [worker]
        monkeypatch.setattr(model, "_finalize_external_capture", AsyncMock(wraps=model._finalize_external_capture))
        return SimpleNamespace(model=model, worker=worker, app=model.setup_webserver(), ledger=FileLineageStore(root))

    return make


def _body(dialect, stream=True):
    body = {"model": "test", "stream": stream}
    body["messages" if dialect in ("chat/completions", "messages") else "input"] = [
        {"role": "user", "content": "hello"}
    ]
    if dialect == "messages":
        body["max_tokens"] = 32
    if dialect == "chat/completions" and stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _path(dialect):
    return PREFIX + "/v1/" + ("responses" if dialect == "compaction" else dialect)


async def _request(app, path, body, send=None):
    messages = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        await asyncio.Event().wait()

    async def collect(message):
        if send is not None:
            await send(message)
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "server": ("test", 80),
        "client": ("test", 1),
    }
    await app(scope, receive, collect)
    return messages


def _events(messages):
    raw = b"".join(message.get("body", b"") for message in messages).decode()
    return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("evaluation", [False, True])
async def test_external_capture_routes(make_harness, dialect, stream, evaluation):
    h = make_harness(dialect, evaluation)

    async def check_send(message):
        manifest = RolloutManifest.model_validate(await h.ledger.manifest("r1"))
        assert len(manifest.records) == 1 and not manifest.failures
        assert h.worker.context.committed

    messages = await _request(h.app, _path(dialect), _body(dialect, stream), check_send)
    assert h.model._finalize_external_capture.await_count == 1
    assert messages[0]["status"] == 200
    raw = b"".join(message.get("body", b"") for message in messages).decode()
    assert "answer 1" in raw and "completion-1" in raw
    for internal in (
        "ng_commit_coords",
        "prompt_token_ids",
        "generation_token_ids",
        "generation_log_probs",
        "routed_experts",
    ):
        assert internal not in raw
    assert h.worker.requests[0]["stream"] is False
    assert "stream_options" not in h.worker.requests[0]
    with TestClient(h.app) as client:
        url = "/training-token-capture/control/rollouts/r1/manifest"
        assert client.get(url).status_code == 401
        manifest = client.get(url, headers={"Authorization": "Bearer test-control-token"}).json()
    record = manifest["records"][0]
    receipt = RolloutReceipt(
        rollout_id="r1",
        manifest=[record],
        terminal_model_call_id=record["model_call_id"],
        terminal_selection="declared",
    )
    row = verify_and_linearize(receipt, h.worker.fetch([record["staging_key"]]))
    assert row.token_ids == [10, 11]
    assert row.token_mask == [0.0, 1.0]
    assert row.logprobs == [0.0, -0.25]
    if stream:
        assert dict(messages[0]["headers"])[b"content-type"].startswith(b"text/event-stream")
        events = _events(messages)
        if dialect in ("responses", "compaction"):
            assert events[-1]["type"] == "response.completed"


@pytest.mark.parametrize("override", ["extra_body", "sampling_overrides"])
def test_static_streaming_override_rejected(make_harness, override):
    with pytest.raises(ValueError, match="non-streaming backend"):
        make_harness(**{override: {"stream": True}})


@pytest.mark.parametrize("dialect", ["responses", "compaction"])
@pytest.mark.parametrize("evaluation", [False, True])
async def test_runtime_override_is_scoped_to_captured_calls(make_harness, dialect, evaluation):
    h = make_harness(dialect, evaluation)
    body = _body(dialect) | {"metadata": {"extra_body": json.dumps({"stream": True})}}
    failures_at_send = []

    async def check_send(message):
        if b"event: response.failed" in message.get("body", b""):
            manifest = await h.ledger.manifest("r1")
            assert not manifest["records"]
            assert len(manifest["failures"]) == len(failures_at_send) + 1
            failures_at_send.append(manifest["failures"])

    messages = await _request(h.app, _path(dialect), body, check_send)
    assert not h.worker.requests
    assert _events(messages)[-1]["type"] == "response.failed"
    # Existing ledger rows make this unresolvable continuation unadmitted.
    body["input"] += [{"role": "assistant", "content": "unrecorded"}, {"role": "user", "content": "continue"}]
    messages = await _request(h.app, _path(dialect), body, check_send)
    assert not h.worker.requests
    assert _events(messages)[-1]["type"] == "response.failed"
    assert len((await h.ledger.manifest("r1"))["failures"]) == 2
    assert len(failures_at_send) == 2
    await _request(h.app, "/v1/responses", body)
    assert h.worker.requests[0]["stream"] is True


@pytest.mark.parametrize(
    "dialect,content_kind",
    [(dialect, "reasoning") for dialect in DIALECTS] + [(dialect, "refusal") for dialect in DIALECTS],
)
@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.parametrize("evaluation", [False, True])
async def test_two_call_continuation_from_served_sse(make_harness, dialect, content_kind, tool_call, evaluation):
    h = make_harness(dialect, evaluation, reasoning=content_kind == "reasoning")
    h.worker.tool_call = tool_call
    h.worker.refusal = content_kind == "refusal"
    body = _body(dialect)
    if tool_call:
        function = {
            "name": "weather",
            "strict": False,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
        if dialect == "messages":
            body["tools"] = [{"name": "weather", "input_schema": function["parameters"]}]
        elif dialect == "chat/completions":
            body["tools"] = [{"type": "function", "function": function}]
        else:
            body["tools"] = [{"type": "namespace", "name": "functions", "tools": [{"type": "function", **function}]}]

    async def complete():
        messages = await _request(h.app, _path(dialect), body)
        raw = b"".join(message.get("body", b"") for message in messages)
        expected = b"I cannot help with that." if h.worker.refusal else b"Check the requested calculation."
        assert expected in raw
        wire_dialect = {"chat/completions": "chat_completions", "compaction": "responses"}.get(dialect, dialect)
        response = _reconstruct_streamed_response(raw, wire_dialect)
        if dialect == "chat/completions" and h.worker.refusal:
            assert any(
                choice["delta"].get("refusal") == expected.decode()
                for event in _events(messages)
                for choice in event.get("choices", [])
            )
            assert response["choices"][0]["message"]["refusal"] == expected.decode()
        return response

    first = await complete()
    if dialect in ("responses", "compaction"):
        body["input"].extend(first["output"])
        body["input"].append(
            {"type": "function_call_output", "call_id": "call-1", "output": "sunny"}
            if tool_call
            else {"role": "user", "content": "continue"}
        )
        if tool_call:
            call = next(item for item in first["output"] if item["type"] == "function_call")
            assert call["namespace"] == "functions"
            assert call["name"] == "weather"
            assert h.model._finalize_external_capture.await_args.args[0]["output"] == first["output"]
            manifest = RolloutManifest.model_validate(await h.ledger.manifest("r1"))
            # Neither a declaration nor an envelope ID may bypass content attribution.
            attribution = resolve_terminal(manifest.records, {**first, "id": ""})
            assert attribution.attributed and attribution.method == "content"
            assert attribution.model_call_id == manifest.records[0].model_call_id
            altered = {
                **first,
                "id": "",
                "output": [{**item, "namespace": "other"} if item is call else item for item in first["output"]],
            }
            assert not resolve_terminal(manifest.records, altered).attributed
    else:
        assistant = (
            first["choices"][0]["message"]
            if dialect == "chat/completions"
            else {"role": "assistant", "content": first["content"]}
        )
        body["messages"].append(assistant)
        if tool_call and dialect == "chat/completions":
            body["messages"].append({"role": "tool", "tool_call_id": "call-1", "content": "sunny"})
        elif tool_call:
            body["messages"].append(
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "sunny"}]}
            )
        else:
            body["messages"].append({"role": "user", "content": "continue"})
    second = await complete()
    manifest = RolloutManifest.model_validate(await h.ledger.manifest("r1"))
    assert not manifest.failures and len(manifest.records) == 2
    parent, child = manifest.records
    admission = h.worker.requests[1]["ng_capture"]
    assert admission["parent_call_id"] == parent.model_call_id
    assert admission["staging_chain"] == [parent.staging_key]
    assert child.response_id == second["id"]
    assert h.model._finalize_external_capture.await_count == 2
    receipt = RolloutReceipt(
        rollout_id="r1",
        manifest=manifest.records,
        terminal_model_call_id=child.model_call_id,
        terminal_selection="declared",
    )
    row = verify_and_linearize(receipt, h.worker.fetch([record.staging_key for record in manifest.records]))
    assert row.token_ids == [10, 11, 20, 21]
    assert row.token_mask == [0.0, 1.0, 0.0, 1.0]
    assert row.logprobs == [0.0, -0.25, 0.0, -0.25]


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize(
    "dialect,failure,stream",
    [(dialect, "conversion", True) for dialect in ("responses", "messages", "compaction")]
    + [(dialect, "serialization", True) for dialect in DIALECTS]
    + [("compaction", failure, False) for failure in ("conversion", "serialization")],
)
async def test_response_preparation_failure_does_not_commit(
    make_harness, monkeypatch, dialect, failure, stream, evaluation
):
    h = make_harness(dialect, evaluation)

    def fail_conversion(*args, **kwargs):
        raise ValueError(f"injected {failure} failure")

    if failure == "conversion":
        if dialect == "messages":
            monkeypatch.setattr(AnthropicConverter, "responses_to_anthropic_response", fail_conversion)
        else:
            monkeypatch.setattr(type(h.model._converter), "chat_completion_to_response", fail_conversion)
    elif not stream:
        monkeypatch.setattr(
            "responses_api_models.vllm_model_with_compaction.app._orjson_dispatch_response", fail_conversion
        )
    elif dialect in ("responses", "compaction"):
        original = responses_streaming._sse_event

        def serialize(payload):
            if payload["type"] == "response.output_item.done":
                raise ValueError("injected serialization failure")
            return original(payload)

        monkeypatch.setattr(responses_streaming, "_sse_event", serialize)
    elif dialect == "chat/completions":
        original = chat_streaming._sse_data

        def serialize(payload):
            if any(choice["delta"].get("content") for choice in payload["choices"]):
                raise ValueError("injected serialization failure")
            return original(payload)

        monkeypatch.setattr(chat_streaming, "_sse_data", serialize)
    else:
        original = AnthropicConverter._sse_event

        def serialize(self, event, payload):
            if event == "content_block_start":
                raise ValueError("injected serialization failure")
            return original(self, event, payload)

        monkeypatch.setattr(AnthropicConverter, "_sse_event", serialize)

    sent = []

    async def check_send(message):
        sent.append(message)
        if b"event: response.failed" in message.get("body", b""):
            manifest = await h.ledger.manifest("r1")
            assert not manifest["records"]
            assert any(row["reason"] == UNCOMMITTED_CALL_REASON for row in manifest["failures"])

    if stream and dialect in ("responses", "compaction"):
        await _request(h.app, _path(dialect), _body(dialect), check_send)
        assert _events(sent)[-1]["type"] == "response.failed"
        assert all(event["type"] != "response.output_item.done" for event in _events(sent))
    else:
        with pytest.raises(ValueError, match=f"injected {failure} failure"):
            await _request(h.app, _path(dialect), _body(dialect, stream), check_send)
        assert sent[0]["status"] == 500

    assert len(h.worker.records) == 1, "the failure must occur after worker staging succeeds"
    h.model._finalize_external_capture.assert_not_awaited()
    manifest = await h.ledger.manifest("r1")
    assert not manifest["records"]
    assert any(row["reason"] == UNCOMMITTED_CALL_REASON for row in manifest["failures"])
    assert b"ng_commit_coords" not in b"".join(message.get("body", b"") for message in sent)
