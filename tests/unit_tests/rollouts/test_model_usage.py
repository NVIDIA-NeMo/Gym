# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from functools import partial
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from nemo_gym.models.model_call_capture import CaptureStore, ModelCallCaptureConfig, install_model_call_capture
from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence
from nemo_gym.rollouts.model_usage import ModelUsageCapture, aggregate_model_usage
from nemo_gym.trajectory.model_call_record import ModelCallRecord


def _exchange(prompt: int = 10, completion: int = 3) -> dict:
    return {"response": {"usage": {"input_tokens": prompt, "output_tokens": completion}}}


async def _capture(tmp_path: Path, rollout_id: str = "rollout") -> ModelUsageCapture:
    capture = await ModelUsageCapture.start(
        {"observability_enabled": True, "model_call_capture_dir": tmp_path}, rollout_id=rollout_id
    )
    assert capture is not None
    return capture


@pytest.mark.parametrize(
    ("dialect", "streaming"),
    [
        ("chat", False),
        ("chat", True),
        ("responses", False),
        ("responses", True),
        ("raw_responses", True),
        ("messages", False),
        ("messages", True),
    ],
)
async def test_counts_model_server_json_and_sse_without_harness_reported_usage(tmp_path, dialect, streaming):
    from nemo_gym.models.anthropic_converter import AnthropicConverter
    from nemo_gym.models.chat_streaming import synthesize_chat_completion_sse
    from nemo_gym.models.responses_streaming import synthesize_responses_sse

    usage = {"input_tokens": 10, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 4}}
    if dialect == "chat":
        path = "/v1/chat/completions"
        payload = {
            "id": "repeated-provider-id",
            "model": "policy",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}},
        }
        events = synthesize_chat_completion_sse(payload, include_usage=True)
    elif dialect in {"responses", "raw_responses"}:
        path = "/v1/responses"
        payload = {"id": "repeated-provider-id", "status": "incomplete", "output": [], "usage": usage}
        events = (
            [
                "event: response.incomplete\ndata: "
                + json.dumps({"type": "response.incomplete", "response": payload})
                + "\n\n"
            ]
            if dialect == "raw_responses"
            else synthesize_responses_sse(payload)
        )
    else:
        path = "/v1/messages"
        payload = {
            "id": "repeated-provider-id",
            "type": "message",
            "role": "assistant",
            "model": "policy",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "max_tokens",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 4,
                "output_tokens": 3,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 2,
            },
        }
        events = AnthropicConverter().anthropic_response_to_sse(payload)

    # Materialize once: the provider reuses its response ID on a retry, but both
    # physical exchanges consumed tokens and must count.
    events = list(events)
    app = FastAPI()

    @app.post(path)
    async def model():
        return StreamingResponse(iter(events), media_type="text/event-stream") if streaming else payload

    install_model_call_capture(
        app, ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path)
    )
    capture = await _capture(tmp_path)
    with TestClient(app) as client:
        for _ in range(2):
            assert client.post(f"/ng-rollout/rollout{path}", json={}).status_code == 200
        assert client.post(f"/ng-rollout/unrelated{path}", json={}).status_code == 200
    result = await capture.usage()
    assert result is not None
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (20, 6, 26)
    assert result.input_tokens_details.cached_tokens == 8
    # Gym's synthesized Responses wire schema supplies zero; raw provider omission remains unknown.
    expected_reasoning = 0 if dialect == "responses" and streaming else None
    assert result.output_tokens_details.reasoning_tokens == expected_reasoning


async def test_capture_is_scoped_to_agent_lifecycle_and_snapshot_is_stable(tmp_path):
    store = CaptureStore(tmp_path)
    store.record("rollout", _exchange(1000))  # Prior attempt or Resources seed.
    capture = await _capture(tmp_path)
    store.record("rollout", _exchange())
    store.record("rollout", _exchange(20, 7))  # Delegate/compaction/close-time reply.
    result = await capture.usage()
    store.record("rollout", _exchange(2000))  # Judge runs after usage is finalized.
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (30, 10, 40)
    assert result.input_tokens_details.cached_tokens is None


@pytest.mark.parametrize("failure", ["missing", "missing_usage", "damaged", "incomplete", "truncated", "deleted"])
async def test_incomplete_capture_never_turns_into_a_partial_or_zero_total(tmp_path, failure):
    store = CaptureStore(tmp_path)
    if failure in {"truncated", "deleted"}:
        store.record("rollout", _exchange(1000))
    capture = await _capture(tmp_path)
    if failure != "missing":
        store.record("rollout", _exchange())
    if failure == "missing_usage":
        store.record("rollout", {"response": {}})
    elif failure == "damaged":
        with store.path_for("rollout").open("ab") as handle:
            handle.write(b"broken json\n")
    elif failure == "incomplete":
        store.mark_incomplete("rollout")
    elif failure == "truncated":
        store.path_for("rollout").write_bytes(b"")
    elif failure == "deleted":
        store.path_for("rollout").unlink()
    assert await capture.usage() is None


async def test_disabled_capture_does_not_create_a_store(tmp_path):
    assert await ModelUsageCapture.start({"model_call_capture_dir": tmp_path}, rollout_id="rollout") is None
    assert not list(tmp_path.iterdir())


def test_zero_is_measured_but_unknown_details_and_truncated_streams_are_not():
    first = ModelCallRecord(
        call_index=0, tokens_in=0, tokens_out=0, tokens_total=0, cached_tokens=0, tokens_reasoning=0
    )
    zero = aggregate_model_usage([first])
    assert (
        zero.total_tokens
        == zero.input_tokens_details.cached_tokens
        == zero.output_tokens_details.reasoning_tokens
        == 0
    )
    second = first.model_copy(update={"call_index": 1, "cached_tokens": None, "tokens_reasoning": None})
    total = aggregate_model_usage([first, second])
    assert total.total_tokens == 0
    assert total.input_tokens_details.cached_tokens is None
    assert total.output_tokens_details.reasoning_tokens is None
    assert aggregate_model_usage([first, second.model_copy(update={"error_category": "stream_truncated"})]) is None
    assert aggregate_model_usage([]) is None


@pytest.mark.parametrize("status_code", [429, 500, 400])
async def test_a_failed_call_without_a_response_contributes_nothing(tmp_path, status_code):
    store = CaptureStore(tmp_path)
    capture = await _capture(tmp_path)
    store.record("rollout", {"status_code": status_code, "response": {"error": {"message": "rejected"}}})
    assert (await capture.usage()).total_tokens == 0  # Every call was rejected: zero, not unknown.
    store.record("rollout", _exchange())  # The agent's retry finished.
    result = await capture.usage()
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (10, 3, 13)


async def test_verifier_calls_under_the_rollout_are_not_agent_usage(tmp_path):
    store = CaptureStore(tmp_path)
    capture = await _capture(tmp_path)
    store.record("rollout", _exchange())
    store.record("rollout", _exchange(500, 50) | {"verifier_invocation_id": "verify-1"})
    assert (await capture.usage()).total_tokens == 13


async def test_cache_write_tokens_are_summed(tmp_path):
    store = CaptureStore(tmp_path)
    capture = await _capture(tmp_path)
    usage = {"input_tokens": 10, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 4}}
    store.record("rollout", {"response": {"usage": usage}})
    store.record("rollout", {"response": {"usage": usage}})
    assert (await capture.usage()).input_tokens_details.cache_write_tokens is None
    anthropic = {"input_tokens": 6, "output_tokens": 3, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 2}
    capture = await _capture(tmp_path)
    store.record("rollout", {"dialect": "messages", "response": {"usage": anthropic}})
    result = await capture.usage()
    assert (result.input_tokens, result.input_tokens_details.cached_tokens) == (12, 4)
    assert result.input_tokens_details.cache_write_tokens == 2


async def test_a_capture_write_still_pending_waits_then_counts(tmp_path):
    store = CaptureStore(tmp_path)
    capture = await _capture(tmp_path)
    store.record("rollout", _exchange())
    store.mark_pending("rollout", "call-2")  # A cancelled call finishes its capture write detached.

    async def land_write() -> None:
        await asyncio.sleep(0.2)
        store.record("rollout", _exchange(20, 7))
        store.clear_pending("rollout", "call-2")

    writer = asyncio.create_task(land_write())
    result = await capture.usage()
    await writer
    assert result.total_tokens == 40


async def test_a_capture_write_never_landing_makes_usage_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nemo_gym.rollouts.model_usage.await_model_call_capture_quiescence",
        partial(await_model_call_capture_quiescence, timeout_s=0.1),
    )
    store = CaptureStore(tmp_path)
    capture = await _capture(tmp_path)
    store.record("rollout", _exchange())
    store.mark_pending("rollout", "call-2")
    assert await capture.usage() is None


def test_offset_reads_from_an_append_boundary(tmp_path):
    store = CaptureStore(tmp_path)
    assert store.offset("rollout") == 0
    store.record("rollout", _exchange(1000))
    end = store.offset("rollout")
    assert store.read_available("rollout", start_offset=end) == ([], 0)
    store.record("rollout", _exchange())
    assert store.read_available("rollout", start_offset=end) == ([(0, _exchange())], 0)
    with pytest.raises(OSError, match="truncated"):
        store.read_available("rollout", start_offset=store.offset("rollout") + 1)
    store.path_for("rollout").unlink()
    with pytest.raises(FileNotFoundError):
        store.read_available("rollout", start_offset=end)
