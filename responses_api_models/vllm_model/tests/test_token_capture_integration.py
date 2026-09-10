# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import _CaptureMiddleware
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture import (
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
from nemo_gym.token_id_capture.sink import NG_CAPTURE_FIELD, NG_COMMIT_COORDS_FIELD
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
from nemo_gym.token_id_capture.staging.records import CaptureAdmission, StageResult
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


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


@pytest.mark.parametrize("endpoint", ["responses", "chat/completions", "messages"])
@pytest.mark.parametrize("reasoning_field", [None, "reasoning", "reasoning_content", "inline"])
@pytest.mark.parametrize("tool_call", [False, True])
def test_external_staging_resolves_served_reasoning_history(tmp_path, endpoint, reasoning_field, tool_call) -> None:
    """Echoing served output must preserve capture ancestry across API conversion."""
    records = {}

    class Sink:
        def stage(self, record):
            key = f"{record.rollout_id}/{record.model_call_id}"
            records[key] = record
            return StageResult(ok=True, staging_key=key)

    capture = RolloutTokenCapture(sink=Sink(), weight_version_fn=lambda: 7)
    admissions = []

    async def complete(**body):
        admission = CaptureAdmission.model_validate(body[NG_CAPTURE_FIELD]) if NG_CAPTURE_FIELD in body else None
        admissions.append(admission)
        index = len(admissions)
        payload = _completion([], [], f"answer {index}")
        message = payload["choices"][0]["message"]
        if reasoning_field == "inline":
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
    model._external_capture_enabled = True
    ledger = FileLineageStore(tmp_path)
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
                assert admissions[-1] is not None
                assert admissions[-1].parent_call_id == (None if index == 1 else admissions[-2].model_call_id)
                assert admissions[-1].prev_len == 2 * (index - 1)
                assert NG_COMMIT_COORDS_FIELD not in response.text
                assert "prompt_token_ids" not in response.text
                served = response.json()
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
        # Parent verification must still reject edits to the last served answer.
        last_answer = history[-3 if tool_call else -2]
        if last_answer.get("type") == "function_call":
            last_answer["arguments"] = '{"changed": true}'
        else:
            last_answer["content"] = "different answer"
        with TestClient(app) as client:
            response = client.post(f"/ng-rollout/reasoning-rollout/training-token-capture/v1/{endpoint}", json=body)
        assert response.status_code == 200, response.text
        assert admissions[-1] is None
        manifest = asyncio.run(ledger.manifest("reasoning-rollout"))
        assert len(manifest["records"]) == 3
        assert manifest["failures"]
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
        if entry.rollout_id in self.frozen:
            raise RuntimeError(f"Token capture for rollout {entry.rollout_id} is already frozen")
        rollout = self.entries.setdefault(entry.rollout_id, {})
        previous = rollout.get(entry.model_call_id)
        if previous is not None and previous != entry:
            raise ValueError(f"Conflicting model call {entry.model_call_id}")
        if previous is None:
            rollout[entry.model_call_id] = entry
            await self.lineage.put(entry)
            self.versions[entry.rollout_id] = self.versions.get(entry.rollout_id, 0) + 1


class _ExternalSink:
    def __init__(self, backend: _ExternalBackend) -> None:
        self.backend = backend

    async def put(self, entry: TokenEntry) -> None:
        await self.backend.commit(entry)

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        self.backend.incomplete.add(rollout_id)

    async def close(self) -> None:
        pass


class _ExternalLineageStore:
    def __init__(self, backend: _ExternalBackend) -> None:
        self.backend = backend

    async def resolve(self, rollout_id: str, request_items: list[dict]):
        return await self.backend.lineage.resolve(rollout_id, request_items)

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
            rollout_id=rollout_id,
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
            rollout_id=rollout_id,
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
