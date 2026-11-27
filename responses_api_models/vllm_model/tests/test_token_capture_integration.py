# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import NeMoGymAsyncOpenAI, ServerClient
from nemo_gym.models import AnthropicConverter, VLLMWorkerCaptureHandler
from nemo_gym.models.model_call_capture import _CaptureMiddleware
from nemo_gym.models.token_capture import (
    NG_CAPTURE_FIELD,
    NG_COMMIT_COORDS_FIELD,
    UNCOMMITTED_CALL_REASON,
    CaptureContext,
    FileLineageStore,
    InMemoryLineageStore,
    LineageResolver,
    ParentResolutionStatus,
    TokenCaptureSnapshot,
    TokenCaptureStore,
    TokenEntry,
    TokenSink,
    TokenSource,
    reset_token_sink,
    set_token_sink,
    trajectories_from_source,
)
from nemo_gym.models.token_capture.harness_notes.opencode import OPENCODE_MAX_STEPS_PROMPT
from nemo_gym.models.token_capture.staging import CaptureAdmission, RolloutTokenCapture, StageResult
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
from responses_api_models.vllm_model.tests.test_app import COLLABORATION_TOOLS


def _model(client: NeMoGymAsyncOpenAI, **config_overrides: Any) -> VLLMModel:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vllm_model",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy_model",
        return_token_id_information=True,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
        supply_prefix_token_ids=True,
    )
    config = config.model_copy(update=config_overrides)
    model = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    model._clients = [client]
    return model


@pytest.mark.parametrize(
    "endpoint,reasoning_field,tool_call,last_turn_edit",
    [
        (endpoint, field, tool_call, "text")
        for endpoint, field in [
            (endpoint, field)
            for endpoint in ("responses", "chat/completions", "messages")
            for field in (None, "reasoning", "reasoning_content", "inline")
        ]
        + [("responses", "refusal"), ("messages", "refusal")]
        for tool_call in (False, True)
    ]
    + [
        ("responses", field, True, edit)
        for field in (None, "reasoning", "reasoning_content", "inline", "refusal")
        for edit in ("arguments", "name", "id", "add", "remove")
    ],
)
def test_external_staging_resolves_served_reasoning_history(
    tmp_path, endpoint, reasoning_field, tool_call, last_turn_edit
) -> None:
    """Echoing served output must preserve capture ancestry across API conversion."""
    records = {}

    class Sink:
        def stage(self, record, *, attachments=None):
            assert attachments is None
            key = f"{record.capture_key}/{record.model_call_id}"
            records[key] = record
            return StageResult(ok=True, staging_key=key)

    capture = RolloutTokenCapture(sink=Sink(), weight_version_fn=lambda: 7)
    admissions = []
    consumed_prefixes = []

    async def complete(**body):
        admission = CaptureAdmission.model_validate(body[NG_CAPTURE_FIELD]) if NG_CAPTURE_FIELD in body else None
        admissions.append(admission)
        index = len(admissions)
        payload = _completion([], [], f"answer {index}")
        message = payload["choices"][0]["message"]
        if reasoning_field == "refusal":
            message.update(content=None, refusal=f"refusal {index}")
        elif reasoning_field == "inline":
            message["content"] = f"<think>reasoning {index}</think>" + message["content"]
        elif reasoning_field:
            message[reasoning_field] = f"reasoning {index}"
        if tool_call:
            message["tool_calls"] = [
                {"id": f"tool-{index}", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]
        if admission is not None:
            prefix = [token for key in admission.staging_chain for token in records[key].token_ids_delta]
            active = capture.begin_call(admission, prefix_token_ids=prefix)
            consumed_prefixes.append(prefix)
            coords = capture.complete_call(
                active,
                prompt_token_ids=prefix + [10 * index],
                generated_token_ids=[100 + index],
                generated_logprobs=[-0.1],
            )
            payload[NG_COMMIT_COORDS_FIELD] = coords.model_dump(mode="json")
        return payload

    backend = MagicMock(spec=NeMoGymAsyncOpenAI)
    backend.create_chat_completion = AsyncMock(side_effect=complete)
    model = _model(
        backend,
        return_token_id_information=False,
        supply_prefix_token_ids=False,
        uses_reasoning_parser=reasoning_field is not None,
    )
    model._external_capture_handler = VLLMWorkerCaptureHandler()
    ledger = FileLineageStore(tmp_path)
    ledger.record = AsyncMock(wraps=ledger.record)
    history = [{"role": "user", "content": "question"}]
    app = _CaptureMiddleware(
        model.setup_webserver(),
        store=None,
        model_server_name="vllm_model",
        lineage_store=ledger,
        external_staging=True,
        token_capture_enabled=True,
    )
    try:
        with TestClient(app) as client:
            for index in range(1, 4):
                body = {"input" if endpoint == "responses" else "messages": history}
                if endpoint == "messages":
                    body.update(model="dummy_model", max_tokens=16)
                response = client.post(
                    f"/ng-rollout/reasoning-rollout/training-token-capture/v1/{endpoint}", json=body
                )
                assert response.status_code == 200, response.text
                manifest = asyncio.run(ledger.manifest("reasoning-rollout"))
                assert not manifest["failures"], manifest
                assert len(manifest["records"]) == index
                # Messages delegates through Responses and Chat, but commits only once.
                assert ledger.record.await_count == index
                assert admissions[-1] is not None
                assert admissions[-1].parent_call_id == (None if index == 1 else admissions[-2].model_call_id)
                assert admissions[-1].prev_len == 2 * (index - 1)
                assert NG_COMMIT_COORDS_FIELD not in response.text
                assert "prompt_token_ids" not in response.text
                served = response.json()
                assert manifest["records"][-1]["response_id"] == served["id"]
                if endpoint == "responses":
                    output = served["output"]
                elif endpoint == "messages":
                    output = [{"role": "assistant", "content": served["content"]}]
                else:
                    output = [
                        {key: value for key, value in served["choices"][0]["message"].items() if value is not None}
                    ]
                history.extend(output)
                if tool_call:
                    if endpoint == "responses":
                        history.append(
                            {"type": "function_call_output", "call_id": f"tool-{index}", "output": "result"}
                        )
                    elif endpoint == "messages":
                        history.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "tool_result", "tool_use_id": f"tool-{index}", "content": "result"}
                                ],
                            }
                        )
                    else:
                        history.append({"role": "tool", "tool_call_id": f"tool-{index}", "content": "result"})
                history.append({"role": "user", "content": "continue"})

        manifest = asyncio.run(ledger.manifest("reasoning-rollout"))
        assert len(manifest["records"]) == 3
        assert not manifest["failures"]
        # Only argument rewrites may reuse the last served turn's original tokens.
        last_answer = history[-3 if tool_call else -2]
        if last_turn_edit == "arguments":
            last_answer["arguments"] = '{"changed": true}'
        elif last_turn_edit == "name":
            last_answer["name"] = "different_tool"
        elif last_turn_edit == "id":
            last_answer["call_id"] = "different_id"
        elif last_turn_edit == "add":
            history.insert(-2, {"role": "assistant", "content": "uncaptured answer"})
        elif last_turn_edit == "remove":
            history.remove(last_answer)
        elif endpoint == "responses":
            # Edit the text/refusal item even when the response ends in a tool call.
            last_message = next(item for item in reversed(history) if item.get("role") == "assistant")
            last_message["content"] = [{"type": "output_text", "text": "different answer"}]
        else:
            last_answer["content"] = "different answer"
        original_parent = admissions[-1].model_call_id
        original_keys = tuple(records)
        with TestClient(app) as client:
            response = client.post(f"/ng-rollout/reasoning-rollout/training-token-capture/v1/{endpoint}", json=body)
        assert response.status_code == 200, response.text
        manifest = asyncio.run(ledger.manifest("reasoning-rollout"))
        if last_turn_edit == "arguments":
            assert admissions[-1] is not None
            assert admissions[-1].mode == "token_in"
            assert admissions[-1].parent_call_id == original_parent
            assert tuple(admissions[-1].staging_chain) == original_keys
            assert admissions[-1].prev_len == 6
            # The worker used all three original staged segments as the prompt prefix;
            # no tokens reconstructed from the rewritten arguments replaced them.
            assert consumed_prefixes[-1] == [10, 101, 20, 102, 30, 103]
            assert [token for key in original_keys for token in records[key].token_ids_delta] == consumed_prefixes[-1]
            assert records[f"reasoning-rollout/{admissions[-1].model_call_id}"].token_ids_delta == [40, 104]
            assert len(manifest["records"]) == 4
            assert not manifest["failures"]
        else:
            assert admissions[-1] is None
            assert len(consumed_prefixes) == 3
            assert len(manifest["records"]) == 3
            assert manifest["failures"]
    finally:
        asyncio.run(ledger.close())


@pytest.mark.parametrize(
    "endpoint,failure",
    [("responses", "conversion"), ("messages", "conversion")]
    + [(endpoint, "serialization") for endpoint in ("responses", "chat/completions", "messages")],
)
def test_external_staging_conversion_failure_does_not_commit(tmp_path, monkeypatch, endpoint, failure) -> None:
    """Worker staging can succeed while the final API conversion fails."""
    staged = []

    class Sink:
        def stage(self, record, *, attachments=None):
            assert attachments is None
            staged.append(record)
            return StageResult(ok=True, staging_key=f"{record.capture_key}/{record.model_call_id}")

    capture = RolloutTokenCapture(sink=Sink(), weight_version_fn=lambda: 7)

    async def complete(**body):
        admission = CaptureAdmission.model_validate(body[NG_CAPTURE_FIELD])
        coords = capture.complete_call(
            capture.begin_call(admission, prefix_token_ids=[]),
            prompt_token_ids=[10],
            generated_token_ids=[100],
            generated_logprobs=[-0.1],
        )
        payload = _completion([], [], "answer")
        payload[NG_COMMIT_COORDS_FIELD] = coords.model_dump(mode="json")
        return payload

    backend = MagicMock(spec=NeMoGymAsyncOpenAI)
    backend.create_chat_completion = AsyncMock(side_effect=complete)
    model = _model(backend, return_token_id_information=False, supply_prefix_token_ids=False)
    model._external_capture_handler = VLLMWorkerCaptureHandler()

    def fail_conversion(*args, **kwargs):
        raise ValueError("injected final conversion failure")

    if failure == "serialization":
        monkeypatch.setattr("nemo_gym.models.base_responses_api_model._orjson_dispatch_response", fail_conversion)
    elif endpoint == "responses":
        monkeypatch.setattr(type(model._converter), "chat_completion_to_response", fail_conversion)
    else:
        monkeypatch.setattr(AnthropicConverter, "responses_to_anthropic_response", fail_conversion)
    ledger = FileLineageStore(tmp_path)
    ledger.record = AsyncMock(wraps=ledger.record)
    app = _CaptureMiddleware(
        model.setup_webserver(),
        store=None,
        model_server_name="vllm_model",
        lineage_store=ledger,
        external_staging=True,
        token_capture_enabled=True,
    )
    body = {"input" if endpoint == "responses" else "messages": [{"role": "user", "content": "question"}]}
    if endpoint == "messages":
        body.update(model="dummy_model", max_tokens=16)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                f"/ng-rollout/conversion-failure/training-token-capture/v1/{endpoint}",
                json=body,
            )
        assert response.status_code == 500
        assert len(staged) == 1, "failure must occur after worker staging"
        ledger.record.assert_not_awaited()
        manifest = asyncio.run(ledger.manifest("conversion-failure"))
        assert not manifest["records"]
        assert any(row["reason"] == UNCOMMITTED_CALL_REASON for row in manifest["failures"])
        assert NG_COMMIT_COORDS_FIELD not in response.text
    finally:
        asyncio.run(ledger.close())


@pytest.mark.parametrize("note_content", ["string", "text_parts"])
def test_external_staging_chains_calls_across_opencode_max_steps_notes(tmp_path, note_content) -> None:
    """OpenCode's step-limit note is harness text: the model reads it as a user message, and every call chains.

    OpenCode 1.17.11 adds the note as a final assistant message to each request from the limit on, and drops
    it again from the next request. Here the model keeps calling tools after the limit, so two requests carry it.
    """
    records = {}

    class Sink:
        def stage(self, record, *, attachments=None):
            key = f"{record.capture_key}/{record.model_call_id}"
            records[key] = record
            return StageResult(ok=True, staging_key=key)

    capture = RolloutTokenCapture(sink=Sink(), weight_version_fn=lambda: 7)
    admissions = []
    forwarded = []
    consumed_prefixes = []

    async def complete(**body):
        # A call capture refused (a poisoned call) reaches the worker without an admission.
        admission = CaptureAdmission.model_validate(body[NG_CAPTURE_FIELD]) if NG_CAPTURE_FIELD in body else None
        admissions.append(admission)
        forwarded.append(body["messages"])
        index = len(admissions)
        payload = _completion([], [], f"answer {index}")
        payload["choices"][0]["message"]["tool_calls"] = [
            {"id": f"tool-{index}", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ]
        if admission is None:
            return payload
        prefix = [token for key in admission.staging_chain for token in records[key].token_ids_delta]
        consumed_prefixes.append(prefix)
        active = capture.begin_call(admission, prefix_token_ids=prefix)
        coords = capture.complete_call(
            active,
            prompt_token_ids=prefix + [10 * index],
            generated_token_ids=[100 + index],
            generated_logprobs=[-0.1],
        )
        payload[NG_COMMIT_COORDS_FIELD] = coords.model_dump(mode="json")
        return payload

    backend = MagicMock(spec=NeMoGymAsyncOpenAI)
    backend.create_chat_completion = AsyncMock(side_effect=complete)
    model = _model(backend, return_token_id_information=False, supply_prefix_token_ids=False)
    model._external_capture_handler = VLLMWorkerCaptureHandler()
    ledger = FileLineageStore(tmp_path)
    app = _CaptureMiddleware(
        model.setup_webserver(),
        store=None,
        model_server_name="vllm_model",
        lineage_store=ledger,
        external_staging=True,
        token_capture_enabled=True,
    )
    note = (
        OPENCODE_MAX_STEPS_PROMPT
        if note_content == "string"
        else [
            {"type": "text", "text": OPENCODE_MAX_STEPS_PROMPT[:40]},
            {"type": "text", "text": OPENCODE_MAX_STEPS_PROMPT[40:]},
        ]
    )
    history = [{"role": "user", "content": "question"}]
    try:
        with TestClient(app) as client:
            for index in range(1, 4):
                # Calls 2 and 3 are past the step limit: each request ends with the note, which the next one drops.
                messages = history + ([{"role": "assistant", "content": note}] if index > 1 else [])
                response = client.post(
                    "/ng-rollout/opencode-rollout/training-token-capture/v1/chat/completions",
                    json={"messages": messages},
                )
                assert response.status_code == 200, response.text
                manifest = asyncio.run(ledger.manifest("opencode-rollout"))
                assert not manifest["failures"], manifest
                assert admissions[-1] is not None
                assert admissions[-1].parent_call_id == (None if index == 1 else admissions[-2].model_call_id)
                assert admissions[-1].mode == ("text" if index == 1 else "token_in")
                served = response.json()["choices"][0]["message"]
                history.append({key: value for key, value in served.items() if value is not None})
                history.append({"role": "tool", "tool_call_id": f"tool-{index}", "content": "result"})
        # The worker reads the note as the last user turn, so it splices the parent's tokens at the parent's own
        # output, and the tool result before the note stays in the prompt.
        for messages in forwarded[1:]:
            assert messages[-1] == {"role": "user", "content": OPENCODE_MAX_STEPS_PROMPT}
            assert [message["role"] for message in messages[-2:]] == ["tool", "user"]
            assert all(message.get("content") != note for message in messages if message["role"] == "assistant")
        # Call 3 extends call 2's tokens, which include the first note, although its own request dropped that note.
        assert consumed_prefixes == [[], [10, 101], [10, 101, 20, 102]]
    finally:
        asyncio.run(ledger.close())


def test_an_assistant_message_that_only_resembles_the_note_stays_model_output(tmp_path) -> None:
    """Only the exact note text is harness text. Any other final assistant message keeps the old handling."""
    admissions = []

    async def complete(**body):
        admissions.append(body.get(NG_CAPTURE_FIELD))
        return _completion([], [], "answer")

    backend = MagicMock(spec=NeMoGymAsyncOpenAI)
    backend.create_chat_completion = AsyncMock(side_effect=complete)
    model = _model(backend, return_token_id_information=False, supply_prefix_token_ids=False)
    model._external_capture_handler = VLLMWorkerCaptureHandler()
    ledger = FileLineageStore(tmp_path)
    app = _CaptureMiddleware(
        model.setup_webserver(),
        store=None,
        model_server_name="vllm_model",
        lineage_store=ledger,
        external_staging=True,
        token_capture_enabled=True,
    )
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": OPENCODE_MAX_STEPS_PROMPT + " "},
    ]
    try:
        with TestClient(app) as client:
            response = client.post(
                "/ng-rollout/prefill-rollout/training-token-capture/v1/chat/completions", json={"messages": messages}
            )
        assert response.status_code == 200, response.text
        assert backend.create_chat_completion.await_args.kwargs["messages"][-1]["role"] == "assistant"
    finally:
        asyncio.run(ledger.close())


def _completion(prompt: list[int], generation: list[int], content: str) -> dict[str, Any]:
    return {
        "id": f"completion-{content}",
        "object": "chat.completion",
        "created": 0,
        "model": "dummy_model",
        "prompt_token_ids": prompt,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "token_ids": generation,
                "message": {"role": "assistant", "content": content},
                "logprobs": {
                    "content": [
                        {
                            "token": f"token_id:{token_id}",
                            "logprob": -0.1,
                            "bytes": None,
                            "top_logprobs": [],
                        }
                        for token_id in generation
                    ]
                },
            }
        ],
    }


class _ExternalBackend:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, TokenEntry]] = {}
        self.incomplete: set[str] = set()
        self.frozen: set[str] = set()
        self.versions: dict[str, int] = {}
        self.lineage = InMemoryLineageStore()

    async def commit(self, entry: TokenEntry) -> None:
        if entry.capture_key in self.frozen:
            raise RuntimeError(f"Token capture for rollout {entry.capture_key} is already frozen")
        rollout = self.entries.setdefault(entry.capture_key, {})
        previous = rollout.get(entry.model_call_id)
        if previous is not None and previous != entry:
            raise ValueError(f"Conflicting model call {entry.model_call_id}")
        if previous is None:
            rollout[entry.model_call_id] = entry
            await self.lineage.put(entry)
            self.versions[entry.capture_key] = self.versions.get(entry.capture_key, 0) + 1


class _ExternalSink:
    def __init__(self, backend: _ExternalBackend) -> None:
        self.backend = backend

    async def put(self, entry: TokenEntry) -> None:
        await self.backend.commit(entry)

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "", *, identity=None) -> None:
        self.backend.incomplete.add(rollout_id)

    async def close(self) -> None:
        pass


class _ExternalLineageStore:
    def __init__(self, backend: _ExternalBackend) -> None:
        self.backend = backend

    async def resolve(self, rollout_id: str, request_items: list[dict], *, identity=None):
        return await self.backend.lineage.resolve(rollout_id, request_items, identity=identity)

    def is_process_shared(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class _ExternalSource:
    def __init__(self, backend: _ExternalBackend) -> None:
        self.backend = backend

    async def freeze(self, rollout_id: str) -> TokenCaptureSnapshot:
        self.backend.frozen.add(rollout_id)
        return TokenCaptureSnapshot(
            capture_key=rollout_id,
            entries=tuple(self.backend.entries.get(rollout_id, {}).values()),
            incomplete=rollout_id in self.backend.incomplete,
            snapshot_id=f"snapshot-{rollout_id}",
            version=self.backend.versions.get(rollout_id, 0),
        )

    async def drop(self, rollout_id: str, *, snapshot_id: str, version: int) -> bool:
        if snapshot_id != f"snapshot-{rollout_id}" or version != self.backend.versions.get(rollout_id, 0):
            return False
        self.backend.entries.pop(rollout_id, None)
        return True

    async def close(self) -> None:
        pass


def _simulate_two_workers(
    sink_factory: Callable[[], TokenSink],
    lineage_factory: Callable[[], LineageResolver],
    source: TokenSource,
    read_entries: Callable[[], list[TokenEntry]],
) -> None:
    rollout_id = "simulated-rollout"
    outbound_requests: list[dict[str, Any]] = []
    completions = [
        _completion([11, 12], [13, 14], "first answer"),
        _completion([11, 12, 13, 14, 21], [22], "second answer"),
    ]

    async def create_chat_completion(**kwargs):
        outbound_requests.append(kwargs)
        return completions[len(outbound_requests) - 1]

    def worker() -> TestClient:
        client = MagicMock(spec=NeMoGymAsyncOpenAI)
        client.create_chat_completion = AsyncMock(side_effect=create_chat_completion)
        client.create_tokenize = AsyncMock()
        return TestClient(_model(client).setup_webserver())

    worker_a = worker()
    worker_b = worker()

    def serve(client: TestClient, call_id: str, messages: list[dict]) -> tuple[dict, CaptureContext]:
        context = CaptureContext(
            capture_key=rollout_id,
            model_call_id=call_id,
            token_sink=sink_factory(),
            lineage_store=lineage_factory(),
        )
        token = set_token_sink(context)
        try:
            response = client.post("/v1/chat/completions", json={"messages": messages})
        finally:
            reset_token_sink(token)
        assert response.status_code == 200, response.text
        return response.json(), context

    first_request = [{"role": "user", "content": "first question"}]
    first_response, first_context = serve(worker_a, "call-a", first_request)
    first_message = first_response["choices"][0]["message"]
    first_answer = {"role": first_message["role"], "content": first_message["content"]}

    second_request = first_request + [first_answer, {"role": "user", "content": "second question"}]
    _, second_context = serve(worker_b, "call-b", second_request)

    entries = read_entries()
    assert len(entries) == 2
    assert entries[0].parent_resolution == ParentResolutionStatus.ROOT
    assert entries[0].prefix_requested is False
    assert entries[0].prefix_supplied is False
    assert entries[1].parent_resolution == ParentResolutionStatus.RESOLVED
    assert entries[1].parent_call_id == entries[0].model_call_id
    assert entries[1].prefix_requested is True
    assert entries[1].prefix_supplied is True
    # The engine's chat finish reason survives the vLLM chat path into the record.
    assert [entry.finish_reason for entry in entries] == ["stop", "stop"]

    assert first_context.parent_resolution is not None
    assert first_context.parent_resolution.status == ParentResolutionStatus.ROOT
    assert second_context.parent_resolution is not None
    assert second_context.parent_resolution.status == ParentResolutionStatus.RESOLVED
    assert second_context.parent_tokens == [11, 12, 13, 14]
    assert outbound_requests[0].get("required_prefix_token_ids") is None
    assert outbound_requests[1]["required_prefix_token_ids"] == [11, 12, 13, 14]

    built = asyncio.run(trajectories_from_source(rollout_id, source))
    assert built is not None
    assert built["mask_sample"] is False
    assert built["metrics"]["roots"] == 1
    assert built["metrics"]["chains"] == 1
    assert built["metrics"]["delivered_fraction"] == 1.0
    assert built["metrics"]["unresolved_parent_calls"] == 0

    output = built["rebuilt_response"]["output"]
    assert len(output) == 2
    assert output[0]["prompt_token_ids"] == [11, 12]
    assert output[0]["generation_token_ids"] == [13, 14]
    assert output[1]["prompt_token_ids"] == [11, 12, 13, 14, 21]
    assert output[1]["generation_token_ids"] == [22]

    snapshot = built["_capture_snapshot"]
    assert asyncio.run(
        source.drop(
            rollout_id,
            snapshot_id=snapshot["snapshot_id"],
            version=snapshot["version"],
        )
    )


def _text_message(role: str, text: str) -> dict[str, Any]:
    """Return a message of ``role`` that holds ``text`` in one ``input_text`` part."""
    return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}


def _agent_message(author: str, recipient: str, message_type: str, payload: str) -> dict[str, Any]:
    """Return a message between two Codex agents as the CLI sends it in a parent's or a subagent's request."""
    text = f"Message Type: {message_type}\nTask name: {recipient}\nSender: {author}\nPayload:\n{payload}"
    return {
        "type": "agent_message",
        "author": author,
        "recipient": recipient,
        "content": [{"type": "input_text", "text": text}],
    }


@pytest.mark.parametrize(
    ("renders_anew", "kept_wait"),
    [(False, True), (False, False), (True, True)],
    ids=["answer-after-the-wait", "answer-after-a-discarded-response", "engine-renders-every-request-anew"],
)
def test_subagent_rollout_delivers_the_parent_chain_or_is_masked(
    tmp_path, renders_anew: bool, kept_wait: bool
) -> None:
    """A Codex rollout with one subagent delivers the parent's chain of calls, under two conditions.

    By default the subagent's first request holds its parent's conversation, so its call continues the parent's
    call that spawned it, here the parent's first call, and forms a branch. The parent's chain continues across
    the request that carries the subagent's answer. When the answer arrives while Codex reads a parent response,
    Codex discards the tool call of that response and keeps its reasoning, and the next request continues the
    call before it.

    The build delivers the parent's chain when the verified response names its last call and the engine's prompt
    for each call starts with the tokens of the call it continues. The first two cases use an engine that
    extends the supplied prefix. The third uses an engine that renders every request anew and leaves out the
    reasoning of the assistant turns before the last user message. The answer is such a user message, so the
    prompt of the request that carries it does not start with the parent's tokens, and the rollout is masked.
    """
    rollout_id = "subagent-rollout"
    replies: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    rendered: dict[tuple, int] = {}

    def turn_token(message: dict[str, Any], with_reasoning: bool) -> int:
        """Return the one token that stands for a chat message, which differs with and without its reasoning."""
        content = message.get("content")
        text = content if isinstance(content, str) else "".join(part["text"] for part in content or [])
        calls = tuple(call["function"]["name"] for call in message.get("tool_calls") or [])
        reasoning = message.get("reasoning_content") if with_reasoning else None
        return rendered.setdefault((message["role"], text, calls, reasoning), 100 + len(rendered))

    def render(messages: list[dict[str, Any]]) -> list[int]:
        last_user = max(index for index, message in enumerate(messages) if message["role"] == "user")
        return [turn_token(message, index > last_user) for index, message in enumerate(messages)]

    async def create_chat_completion(**kwargs):
        requests.append(kwargs)
        number = len(requests)
        reply = replies[number - 1]
        if renders_anew:
            prompt, generated = render(kwargs["messages"]), [turn_token(reply, True)]
        else:
            # Call N extends the supplied prefix by the prompt token 10 * N and generates 10 * N + 1.
            prompt, generated = [*(kwargs.get("required_prefix_token_ids") or []), 10 * number], [10 * number + 1]
        completion = _completion(prompt, generated, "")
        completion["choices"][0]["message"] = reply
        return completion

    backend = MagicMock(spec=NeMoGymAsyncOpenAI)
    backend.create_chat_completion = AsyncMock(side_effect=create_chat_completion)
    backend.create_tokenize = AsyncMock()
    model = _model(
        backend,
        uses_reasoning_parser=True,
        uses_interleaved_reasoning=True,
        supply_prefix_token_ids=not renders_anew,
    )
    client = TestClient(model.setup_webserver())

    def reply(reasoning: str, content: str | None = None, tool: str | None = None) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        if tool:
            function = {"name": f"collaboration__{tool}", "arguments": "{}"}
            message["tool_calls"] = [{"id": f"call_{tool}", "type": "function", "function": function}]
        return message

    def serve(call_id: str, input_items: list[dict], message: dict[str, Any]) -> dict[str, Any]:
        """Serve one streaming request of the rollout and return the response that the stream ends with."""
        replies.append(message)
        context = CaptureContext(
            capture_key=rollout_id,
            model_call_id=call_id,
            token_sink=TokenCaptureStore(tmp_path),
            lineage_store=FileLineageStore(tmp_path),
        )
        token = set_token_sink(context)
        try:
            body = {"stream": True, "input": input_items, "tools": [COLLABORATION_TOOLS]}
            response = client.post("/v1/responses", json=body)
        finally:
            reset_token_sink(token)
        assert response.status_code == 200, response.text
        events = [
            json.loads(line[len("data: ") :]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert events[-1]["type"] == "response.completed", events[-1]
        return events[-1]["response"]

    def echoed(response: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the reasoning item and the function call of a response with the fields that Codex sends back."""
        reasoning, call = response["output"]
        return [
            {"type": "reasoning", "id": reasoning["id"], "summary": reasoning["summary"]},
            {key: call[key] for key in ("type", "id", "status", "namespace", "name", "arguments", "call_id")},
        ]

    def output(call_id: str, text: str) -> dict[str, Any]:
        return {"type": "function_call_output", "call_id": call_id, "output": text}

    start = [_text_message("developer", "You are Codex."), _text_message("user", "Check the result with a subagent.")]
    spawn = echoed(serve("parent-1", start, reply("I delegate the check.", tool="spawn_agent")))
    thread = [*start, *spawn, output("call_spawn_agent", '{"task_name":"/root/check"}')]
    wait = echoed(serve("parent-2", thread, reply("I wait for the subagent.", tool="wait_agent")))
    subagent = [
        *start,
        *spawn,
        output("call_spawn_agent", "aborted"),
        _text_message("developer", "You are now a newly spawned child agent."),
        _agent_message("/root", "/root/check", "NEW_TASK", "Check the result."),
    ]
    serve("subagent-1", subagent, reply("I check the result.", content="The result holds."))
    answer = _agent_message("/root/check", "/root", "FINAL_ANSWER", "The result holds.")
    if kept_wait:
        thread = [*thread, *wait, output("call_wait_agent", '{"message":"Wait completed.","timed_out":false}'), answer]
    else:
        thread = [*thread, wait[0], answer]
    final = serve(
        "parent-3", thread, reply("The subagent has answered.", content="The result holds, says the subagent.")
    )

    # The engine reads the task and the answer as the last message of the request that carries them.
    last_messages = [request["messages"][-1] for request in requests[2:]]
    assert [message["role"] for message in last_messages] == ["user", "user"]
    assert [message["content"][0]["text"].split("\n")[0] for message in last_messages] == [
        "Message Type: NEW_TASK",
        "Message Type: FINAL_ANSWER",
    ]
    # The reasoning that Codex keeps of a discarded response is an assistant turn before the answer.
    assert [message["role"] for message in requests[3]["messages"][-3:]] == (
        ["assistant", "tool", "user"] if kept_wait else ["tool", "assistant", "user"]
    )

    # At request time each call names the call it continues.
    entries = {entry.model_call_id: entry for entry in TokenCaptureStore(tmp_path).read_entries(rollout_id)}
    assert {call_id: entry.parent_call_id for call_id, entry in entries.items()} == {
        "parent-1": None,
        "parent-2": "parent-1",
        "subagent-1": "parent-1",
        "parent-3": "parent-2" if kept_wait else "parent-1",
    }
    assert all(entry.parent_resolution == ParentResolutionStatus.RESOLVED for entry in list(entries.values())[1:])

    built = asyncio.run(trajectories_from_source(rollout_id, TokenCaptureStore(tmp_path), verified_response=final))
    if renders_anew:
        # Neither the subagent's prompt nor the prompt with the answer starts with the tokens of its parent call.
        assert built["metrics"]["parent_link_failures"] == {"parent_digest_mismatch": 2}
        assert built["metrics"]["terminal_attribution"]["chain"] == "broken"
        assert built["mask_sample"] is True
        return

    # The subagent's request branches off after the parent's first call, and the request with the answer extends
    # the parent's last kept call.
    prefixes = [request.get("required_prefix_token_ids") or [] for request in requests]
    assert prefixes == [[], [10, 11], [10, 11], [10, 11, 20, 21] if kept_wait else [10, 11]]
    assert built["mask_sample"] is False
    assert built["metrics"]["terminal_attribution"]["chain"] == "delivered"
    assert built["metrics"]["chains"] == (2 if kept_wait else 3)
    delivered = [item.get("generation_token_ids") for item in built["rebuilt_response"]["output"]]
    assert [tokens for tokens in delivered if tokens] == ([[11], [21], [41]] if kept_wait else [[11], [41]])
    assert built["metrics"]["delivered_fraction"] == (0.75 if kept_wait else 0.5)

    # Without the verified response nothing names the parent's chain among the branches, so the rollout is masked.
    unattributed = asyncio.run(trajectories_from_source(rollout_id, TokenCaptureStore(tmp_path)))
    assert unattributed["mask_sample"] is True


def test_local_store_capture_supply_and_rebuild_one_safe_trajectory(tmp_path) -> None:
    _simulate_two_workers(
        sink_factory=lambda: TokenCaptureStore(tmp_path),
        lineage_factory=lambda: FileLineageStore(tmp_path),
        source=TokenCaptureStore(tmp_path),
        read_entries=lambda: TokenCaptureStore(tmp_path).read_entries("simulated-rollout"),
    )


def test_external_protocols_capture_supply_and_rebuild_one_safe_trajectory() -> None:
    backend = _ExternalBackend()
    _simulate_two_workers(
        sink_factory=lambda: _ExternalSink(backend),
        lineage_factory=lambda: _ExternalLineageStore(backend),
        source=_ExternalSource(backend),
        read_entries=lambda: list(backend.entries["simulated-rollout"].values()),
    )
