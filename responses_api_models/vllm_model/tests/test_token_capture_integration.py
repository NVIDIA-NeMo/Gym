# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture import (
    CaptureContext,
    FileLineageStore,
    InMemoryLineageStore,
    LineageStore,
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
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def _model(client: NeMoGymAsyncOpenAI) -> VLLMModel:
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
    model = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    model._clients = [client]
    return model


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
    lineage_factory: Callable[[], LineageStore],
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
        assert response.status_code == 200
        return response.json(), context

    first_request = [{"role": "user", "content": "first question"}]
    first_response, first_context = serve(worker_a, "call-a", first_request)
    first_answer = first_response["choices"][0]["message"]

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
