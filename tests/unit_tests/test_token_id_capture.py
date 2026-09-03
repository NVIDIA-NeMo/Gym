# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test training-token capture records, stores, and sources.

Served-path tests build a real ``SimpleResponsesAPIModel``.
Middleware mints a ``model_call_id``.
Middleware sets a request-scoped token sink.
The model server records a ``TokenEntry``.
Consumers read records through ``TokenSource.freeze``.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import subprocess
import sys
from time import time
from unittest.mock import MagicMock, patch
from uuid import uuid4

import orjson
import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    CaptureStore,
    SimpleResponsesAPIModel,
    _request_messages,
    read_model_call_records,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture import (
    TOKEN_ENTRY_MIN_SCHEMA_VERSION,
    TOKEN_ENTRY_RECORD_SCHEMA_VERSION,
    TOKEN_FIELDS,
    CaptureContext,
    LineageResolution,
    ParentResolutionStatus,
    TokenCaptureStore,
    TokenEntry,
    TokenIdCaptureConfig,
    capture_tokens,
    commit_entry,
    compute_digest,
    cumulative_tokens,
    current_capture_context,
    extract_token_fields,
    install_token_sink,
    register_call_intent,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
    stamp_continuation,
    stamp_lineage,
)
from nemo_gym.token_id_capture.config import token_id_capture_enabled_for_agent
from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest
from nemo_gym.token_id_capture.lineage import (
    FileLineageStore,
    LineageIndex,
    RolloutLineage,
)
from nemo_gym.token_id_capture.protocols import TokenSource
from nemo_gym.token_id_capture.store import make_token_store


_ASSISTANT_TURN = {
    "role": "assistant",
    "content": "checking",
    "tool_calls": [{"function": {"name": "search", "arguments": '{"q":"alpha"}'}}],
}

_MSG_ARGS = {"model": "downstream-model", "max_tokens": 64}

PTOKS = [1, 2, 3]
GTOKS = [4, 5]
LPS = [-0.1, -0.2]


# --- schema / extractor -------------------------------------------------------


def test_extract_token_fields_responses_shape():
    payload = {
        "output": [
            {"type": "message", "prompt_token_ids": PTOKS, "generation_token_ids": GTOKS, "generation_log_probs": LPS}
        ]
    }
    assert extract_token_fields(payload) == {
        "prompt_token_ids": PTOKS,
        "generation_token_ids": GTOKS,
        "generation_log_probs": LPS,
        "routed_experts": None,
    }


def test_extract_token_fields_chat_shape():
    payload = {
        "choices": [
            {"message": {"prompt_token_ids": [1], "generation_token_ids": [7], "generation_log_probs": [-0.3]}}
        ]
    }
    got = extract_token_fields(payload)
    assert got["generation_token_ids"] == [7] and got["prompt_token_ids"] == [1]


def test_extract_token_fields_absent_returns_none():
    assert extract_token_fields({"output": [{"type": "message"}]}) is None
    assert extract_token_fields({}) is None


def test_extract_token_fields_rejects_partial_metadata():
    with pytest.raises(ValueError, match="prompt_token_ids"):
        extract_token_fields(
            {
                "output": [
                    {
                        "type": "message",
                        "generation_token_ids": GTOKS,
                        "generation_log_probs": LPS,
                    }
                ]
            }
        )


def test_extract_token_fields_rejects_multiple_carriers():
    carrier = {
        "prompt_token_ids": PTOKS,
        "generation_token_ids": GTOKS,
        "generation_log_probs": LPS,
    }
    with pytest.raises(ValueError, match="multiple response items"):
        extract_token_fields({"output": [carrier, carrier]})


def test_token_entry_rejects_mismatched_generation_arrays():
    with pytest.raises(ValidationError, match="same length"):
        TokenEntry(
            rollout_id="r0",
            model_call_id="c0",
            prompt_token_ids=PTOKS,
            generation_token_ids=GTOKS,
            generation_log_probs=[-0.1],
        )


def test_token_entry_rejects_inconsistent_parent_resolution():
    common = {
        "rollout_id": "r0",
        "model_call_id": "c0",
        "prompt_token_ids": PTOKS,
        "generation_token_ids": GTOKS,
        "generation_log_probs": LPS,
    }
    with pytest.raises(ValidationError, match="requires parent_call_id"):
        TokenEntry(**common, parent_resolution=ParentResolutionStatus.RESOLVED)
    with pytest.raises(ValidationError, match="cannot carry parent_call_id"):
        TokenEntry(
            **common,
            parent_resolution=ParentResolutionStatus.ROOT,
            parent_call_id="parent",
        )


# --- store --------------------------------------------------------------------


def test_token_store_round_trip(tmp_path):
    store = TokenCaptureStore(tmp_path)
    entry = TokenEntry(
        rollout_id="t0-r0",
        model_call_id="abc",
        model="m",
        prompt_token_ids=PTOKS,
        generation_token_ids=GTOKS,
        generation_log_probs=LPS,
    )
    store.append(entry)
    store.append(entry.model_copy(update={"model_call_id": "def"}))
    read = store.read_entries("t0-r0")
    assert [e.model_call_id for e in read] == ["abc", "def"]
    assert read[0].prompt_token_ids == PTOKS
    assert store.read_entries("missing") == []


def test_token_store_put_is_idempotent_and_conflicts_fail_closed(tmp_path):
    store = TokenCaptureStore(tmp_path)
    entry = TokenEntry(
        rollout_id="r0",
        model_call_id="c0",
        prompt_token_ids=PTOKS,
        generation_token_ids=GTOKS,
        generation_log_probs=LPS,
    )
    asyncio.run(store.put(entry))
    asyncio.run(store.put(entry))
    assert store.read_entries("r0") == [entry]

    with pytest.raises(ValueError, match="reused with a different payload"):
        asyncio.run(store.put(entry.model_copy(update={"generation_token_ids": [8, 9]})))
    assert store.is_incomplete("r0")
    assert store.read_entries("r0") == [entry]


def test_token_store_append_uses_the_compact_entry_index(tmp_path, monkeypatch):
    store = TokenCaptureStore(tmp_path)
    entry = TokenEntry(
        rollout_id="r0",
        model_call_id="c0",
        prompt_token_ids=PTOKS,
        generation_token_ids=GTOKS,
        generation_log_probs=LPS,
    )
    store.append(entry)

    def fail_if_rescanned(_rollout_id):
        raise AssertionError("append rescanned prior token records")

    monkeypatch.setattr(store, "_read_entries_unlocked", fail_if_rescanned)
    store.append(entry.model_copy(update={"model_call_id": "c1"}))
    store.append(entry)


def test_token_store_recovers_an_unindexed_durable_tail(tmp_path):
    store = TokenCaptureStore(tmp_path)
    entry = TokenEntry(
        rollout_id="r0",
        model_call_id="c0",
        prompt_token_ids=PTOKS,
        generation_token_ids=GTOKS,
        generation_log_probs=LPS,
    )
    payload = orjson.dumps(entry.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS) + b"\n"
    store.path_for("r0").write_bytes(payload)

    store.append(entry)

    assert store.read_entries("r0") == [entry]
    state = orjson.loads(store.state_path_for("r0").read_bytes())
    assert state["indexed_size"] == len(payload)
    assert set(state["entry_digests"]) == {"c0"}


def test_dangling_call_intent_marks_frozen_snapshot_incomplete(tmp_path):
    store = TokenCaptureStore(tmp_path)
    asyncio.run(store.begin_call("lost", "c1"))

    snapshot = store.freeze_now("lost")

    assert snapshot.entries == ()
    assert snapshot.incomplete is True


def test_committed_call_satisfies_its_intent(tmp_path):
    store = TokenCaptureStore(tmp_path)
    asyncio.run(store.begin_call("complete", "c1"))
    entry = TokenEntry(
        rollout_id="complete",
        model_call_id="c1",
        prompt_token_ids=PTOKS,
        generation_token_ids=GTOKS,
        generation_log_probs=LPS,
    )
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    asyncio.run(store.put(entry))

    snapshot = store.freeze_now("complete")

    assert [entry.model_call_id for entry in snapshot.entries] == ["c1"]
    assert snapshot.incomplete is False


def test_register_call_intent_uses_optional_sink_extension():
    calls: list[tuple[str, str]] = []

    class Sink:
        async def begin_call(self, rollout_id: str, model_call_id: str) -> None:
            calls.append((rollout_id, model_call_id))

    token = set_token_sink(CaptureContext(rollout_id="r", model_call_id="c", token_sink=Sink()))
    try:
        asyncio.run(register_call_intent())
    finally:
        reset_token_sink(token)

    assert calls == [("r", "c")]


def test_token_store_freeze_is_atomic_and_conditional_drop_is_race_safe(tmp_path):
    store = TokenCaptureStore(tmp_path)
    entry = TokenEntry(
        rollout_id="r0",
        model_call_id="c0",
        prompt_token_ids=PTOKS,
        generation_token_ids=GTOKS,
        generation_log_probs=LPS,
    )
    asyncio.run(store.put(entry))
    snapshot = asyncio.run(store.freeze("r0"))
    assert snapshot.entries == (entry,)
    assert asyncio.run(store.freeze("r0")) == snapshot

    with pytest.raises(RuntimeError, match="already frozen"):
        asyncio.run(store.put(entry.model_copy(update={"model_call_id": "late"})))
    asyncio.run(store.mark_incomplete("r0", "late"))
    assert not asyncio.run(store.drop("r0", snapshot_id=snapshot.snapshot_id, version=snapshot.version))
    updated = asyncio.run(store.freeze("r0"))
    assert updated.incomplete
    assert asyncio.run(store.drop("r0", snapshot_id=updated.snapshot_id, version=updated.version))
    assert store.read_entries("r0") == []
    state = orjson.loads(store.state_path_for("r0").read_bytes())
    assert state["retired"] is True
    assert state["indexed_size"] == 0
    assert state["entry_digests"] == {}
    with pytest.raises(RuntimeError, match="retired"):
        asyncio.run(store.freeze("r0"))
    with pytest.raises(RuntimeError, match="retired"):
        asyncio.run(store.put(entry.model_copy(update={"model_call_id": "late-after-drop"})))

    store.delete("r0")
    replacement = entry.model_copy(update={"model_call_id": "replacement"})
    asyncio.run(store.put(replacement))
    assert store.read_entries("r0") == [replacement]


def test_token_store_sweeps_only_old_retired_tombstones(tmp_path):
    store = TokenCaptureStore(tmp_path)
    for rollout_id in ("old", "recent", "live"):
        entry = TokenEntry(
            rollout_id=rollout_id,
            model_call_id=f"{rollout_id}-c1",
            prompt_token_ids=PTOKS,
            generation_token_ids=GTOKS,
            generation_log_probs=LPS,
        )
        stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
        store.append(entry)
    for rollout_id in ("old", "recent"):
        snapshot = store.freeze_now(rollout_id)
        assert asyncio.run(store.drop(rollout_id, snapshot_id=snapshot.snapshot_id, version=snapshot.version))

    old = time() - 3600
    os.utime(store.state_path_for("old"), (old, old))

    assert store.sweep_retired(older_than_seconds=600) == 1
    assert not store.state_path_for("old").exists()
    assert store.state_path_for("recent").exists()
    assert store.path_for("live").exists()


def test_token_store_recovers_state_lag_from_the_durable_jsonl_tail(tmp_path):
    store = TokenCaptureStore(tmp_path)
    first = TokenEntry(
        rollout_id="lag",
        model_call_id="c1",
        prompt_token_ids=PTOKS,
        generation_token_ids=GTOKS,
        generation_log_probs=LPS,
    )
    stamp_lineage(first, None, parent_resolution=ParentResolutionStatus.ROOT)
    store.append(first)
    state_after_first = store.state_path_for("lag").read_bytes()
    second = first.model_copy(update={"model_call_id": "c2"})
    store.append(second)

    store.state_path_for("lag").write_bytes(state_after_first)
    snapshot = store.freeze_now("lag")

    assert {entry.model_call_id for entry in snapshot.entries} == {"c1", "c2"}
    assert snapshot.incomplete is False


# --- config -------------------------------------------------------------------


def _block(**kwargs) -> dict:
    return {
        "token_id_capture": {
            "enabled": True,
            "rebuild_response": False,
            "allow_unresolved_continuations": True,
            **kwargs,
        }
    }


def test_config_disabled_needs_no_dir():
    cfg = TokenIdCaptureConfig.model_validate({})
    assert cfg.enabled is False
    assert make_token_store({}) is None


def test_config_enabled_requires_absolute_dir(tmp_path):
    """Reject a relative capture directory.

    Relative paths depend on the server working directory.
    """
    with pytest.raises(ValueError):
        TokenIdCaptureConfig.model_validate(_block(dir="relative/dir"))
    cfg = TokenIdCaptureConfig.model_validate(_block(dir=str(tmp_path)))
    assert cfg.resolved_dir() == tmp_path


def test_config_falls_back_to_model_call_capture_dir(tmp_path):
    cfg = TokenIdCaptureConfig.model_validate(_block() | {"model_call_capture_dir": str(tmp_path)})
    assert cfg.resolved_dir() == tmp_path


def test_config_keeps_settings_when_capture_is_off(tmp_path):
    """Allow inactive settings in templated configurations.

    A run may toggle only ``enabled``.
    """
    cfg = TokenIdCaptureConfig.model_validate({"token_id_capture": {"enabled": False, "dir": str(tmp_path)}})
    assert cfg.enabled is False
    assert cfg.build_sink() is None


def test_mask_fraction_limit_defaults_off_and_parses():
    default = TokenIdCaptureConfig.model_validate(_block(dir="/tmp/token-capture"))
    configured = TokenIdCaptureConfig.model_validate(_block(dir="/tmp/token-capture", max_mask_fraction=0.5))

    assert default.token_id_capture.max_mask_fraction is None
    assert configured.token_id_capture.max_mask_fraction == 0.5
    assert configured.token_id_capture.mask_fraction_min_samples == 50


def test_agent_capture_selection_uses_static_agent_config_or_all_agents():
    config = {
        "token_id_capture": {"enabled": True, "rebuild_response": False, "allow_unresolved_continuations": True},
        "captured": {"responses_api_agents": {"implementation": {"token_id_capture": True}}},
        "ordinary": {"responses_api_agents": {"implementation": {"token_id_capture": False}}},
    }

    assert token_id_capture_enabled_for_agent(config, "captured")
    assert not token_id_capture_enabled_for_agent(config, "ordinary")
    assert not token_id_capture_enabled_for_agent(config, "missing")
    assert token_id_capture_enabled_for_agent(
        {**config, "token_id_capture": {**config["token_id_capture"], "all_agents": True}},
        "ordinary",
    )
    assert not token_id_capture_enabled_for_agent(
        {**config, "token_id_capture": {**config["token_id_capture"], "enabled": False, "all_agents": True}},
        "captured",
    )


def test_config_warns_rather_than_fails_on_a_sink_beside_a_directory(caplog):
    """Warn when a custom sink replaces the configured directory."""
    with caplog.at_level(logging.WARNING):
        cfg = TokenIdCaptureConfig.model_validate(_block(sink=f"{__name__}:_ConfiguredSink", dir="/tmp/x"))
    assert cfg.enabled is True
    assert "will not be written to" in caplog.text


def test_config_rejects_an_unknown_key():
    """A typo in this block silently disables capture, so it is refused at startup."""
    with pytest.raises(ValueError):
        TokenIdCaptureConfig.model_validate({"token_id_capture": {"enabled": True, "dirr": "/tmp/x"}})


def test_config_accepts_a_sink_without_constructing_the_consumer_source():
    config = TokenIdCaptureConfig.model_validate(
        {
            "token_id_capture": {
                "enabled": True,
                "sink": f"{__name__}:_ConfiguredSink",
                "allow_unresolved_continuations": True,
            }
        }
    )
    assert config.token_id_capture.sink == f"{__name__}:_ConfiguredSink"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("source", "framework.capture:Source"),
        ("source_kwargs", {"endpoint": "transport://tokens"}),
    ],
)
def test_config_rejects_framework_source_construction(key, value):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        TokenIdCaptureConfig.model_validate(
            {"token_id_capture": {"enabled": True, "rebuild_response": False, key: value}}
        )


# --- source / readers ---------------------------------------------------------


def _training_response(text: str, model: str = "downstream-model") -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model=model,
        object="response",
        output=[
            {
                "type": "message",
                "id": f"msg_{uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "prompt_token_ids": PTOKS,
                "generation_token_ids": GTOKS,
                "generation_log_probs": LPS,
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


def _training_chat_completion(model: str = "downstream-model") -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": f"chatcmpl_{uuid4().hex}",
            "created": int(time()),
            "model": model,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "prompt_token_ids": PTOKS,
                        "generation_token_ids": GTOKS,
                        "generation_log_probs": LPS,
                    },
                }
            ],
        }
    )


class _CapturingModel(SimpleResponsesAPIModel):
    config: BaseResponsesAPIModelConfig
    model_config = {"arbitrary_types_allowed": True}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        return _training_response("hi from responses")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        return _training_chat_completion()


def _server(global_config_dict, *, num_workers: int | None = None) -> SimpleResponsesAPIModel:
    return _CapturingModel(
        config=BaseResponsesAPIModelConfig(
            host="0.0.0.0",
            port=8099,
            entrypoint="",
            name="srv",
            num_workers=num_workers,
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict=global_config_dict),
    )


def _both_enabled(tmp_path) -> dict:
    return {
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path),
        "token_id_capture": {"enabled": True, "dir": str(tmp_path)},
    }


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/responses", {"input": "hi"}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ],
)
def test_disabled_capture_does_not_serialize_request_messages(path, body):
    client = TestClient(_server({}).setup_webserver())
    with patch(
        "nemo_gym.base_responses_api_model._request_messages",
        side_effect=AssertionError("disabled capture serialized request messages"),
    ):
        response = client.post(path, json=body)
    assert response.status_code == 200


def test_responses_call_captures_tokens_joined_to_eval_record(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post("/ng-rollout/task0-roll0/training-token-capture/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200

    tokens = TokenCaptureStore(tmp_path).read_entries("task0-roll0")
    assert len(tokens) == 1
    assert tokens[0].generation_token_ids == GTOKS and tokens[0].prompt_token_ids == PTOKS

    records = read_model_call_records(CaptureStore(tmp_path), "task0-roll0")
    assert len(records) == 1
    # The training entry joins its eval record by the middleware-minted model_call_id.
    assert tokens[0].model_call_id == records[0].model_call_id


def test_captured_entry_carries_content(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    client.post("/ng-rollout/task0-rollC/training-token-capture/v1/responses", json={"input": "hi"})
    tokens = TokenCaptureStore(tmp_path).read_entries("task0-rollC")
    assert len(tokens) == 1
    # Not token-only: the captured record carries the content-bearing output items.
    assert tokens[0].output_items
    text = tokens[0].output_items[-1]["content"][0]["text"]
    assert text == "hi from responses"


def test_token_arrays_are_stored_once(tmp_path):
    """Store token arrays once and preserve response content.

    Served responses carry arrays on an output item.
    Captured records move them to the entry.
    Chained trajectories rebuild each item's running prompt.
    """
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    client.post("/ng-rollout/task0-rollDedup/training-token-capture/v1/responses", json={"input": "hi"})
    entry = TokenCaptureStore(tmp_path).read_entries("task0-rollDedup")[0]
    assert entry.generation_token_ids == GTOKS
    for item in entry.output_items:
        assert not any(field in item for field in TOKEN_FIELDS)
    # Content is kept; only the arrays move off.
    assert entry.output_items[-1]["content"][0]["text"] == "hi from responses"
    # Which item they came off, so a consumer can put the chain-correct values back.
    assert entry.token_item_index == len(entry.output_items) - 1


def test_messages_call_captures_tokens(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post(
        "/ng-rollout/task0-roll1/training-token-capture/v1/messages",
        json={"model": "claude-x", "max_tokens": 16, "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    # The Anthropic response on the wire never carries token ids.
    assert "generation_token_ids" not in resp.text
    tokens = TokenCaptureStore(tmp_path).read_entries("task0-roll1")
    assert len(tokens) == 1 and tokens[0].generation_token_ids == GTOKS


def test_chat_completions_call_captures_tokens(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post(
        "/ng-rollout/task0-roll2/training-token-capture/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    tokens = TokenCaptureStore(tmp_path).read_entries("task0-roll2")
    assert len(tokens) == 1 and tokens[0].generation_token_ids == GTOKS


def test_tokens_captured_even_when_eval_capture_disabled(tmp_path):
    config = {"token_id_capture": {"enabled": True, "dir": str(tmp_path)}}
    client = TestClient(_server(config).setup_webserver())
    resp = client.post("/ng-rollout/task1-roll0/training-token-capture/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    assert len(TokenCaptureStore(tmp_path).read_entries("task1-roll0")) == 1
    # No eval capture file was written.
    assert read_model_call_records(CaptureStore(tmp_path), "task1-roll0") == []


def test_observability_prefix_does_not_enable_training_token_capture(tmp_path):
    """Rollout correlation is neutral; token capture requires explicit path intent."""
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post("/ng-rollout/observed-only/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    assert TokenCaptureStore(tmp_path).read_entries("observed-only") == []
    assert len(read_model_call_records(CaptureStore(tmp_path), "observed-only")) == 1


def test_uncorrelated_call_captures_nothing(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post("/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    # No rollout prefix -> nothing recorded, no file created.
    assert list(tmp_path.glob("*.tokens.jsonl")) == []


def test_package_is_dependency_free_leaf():
    """Keep token capture independent of Gym's server stack.

    Framework inference workers import the record and protocols.
    They must not import Ray, FastAPI, or uvicorn through this package.
    A subprocess isolates this check from earlier test imports.
    """
    heavy = ("ray", "fastapi", "uvicorn", "aiohttp", "requests", "torch")
    program = (
        f"import sys; import nemo_gym.token_id_capture; print(','.join(m for m in {heavy!r} if m in sys.modules))"
    )
    proc = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"leaf package pulled in: {proc.stdout.strip()}"


def test_streamed_messages_capture_tokens_absent_from_the_stream(tmp_path):
    """Capture tokens before streaming Anthropic messages.

    Token ids exist only on the assembled response.
    Anthropic conversion omits them from SSE.
    This test covers the complete served path.
    """
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    with client.stream(
        "POST",
        "/ng-rollout/stream0-roll0/training-token-capture/v1/messages",
        json={
            "model": "claude-x",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    # Nothing on the wire carries token ids.
    assert "generation_token_ids" not in body
    assert "prompt_token_ids" not in body
    # ...yet the record is complete.
    entries = TokenCaptureStore(tmp_path).read_entries("stream0-roll0")
    assert len(entries) == 1
    assert entries[0].generation_token_ids == GTOKS
    assert entries[0].prompt_token_ids == PTOKS
    assert entries[0].output_items, "content must be captured alongside the tokens"


def test_capture_failure_marks_the_rollout_incomplete(tmp_path, monkeypatch):
    """Mark a rollout incomplete when capture loses a call.

    A bad payload must not break the model call.
    Consumers must still detect the missing record.
    """
    store = TokenCaptureStore(tmp_path)

    async def boom(self, entry):
        raise RuntimeError("sink is down")

    monkeypatch.setattr(TokenCaptureStore, "put", boom)
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    # The model call still succeeds.
    resp = client.post("/ng-rollout/fail0-roll0/training-token-capture/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    assert store.read_entries("fail0-roll0") == []
    assert store.is_incomplete("fail0-roll0")


class _SilentModel(_CapturingModel):
    """A model server that answers normally but returns no token ids."""

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        response = _training_response("hi with no tokens")
        for field in ("prompt_token_ids", "generation_token_ids", "generation_log_probs"):
            setattr(response.output[0], field, None)
        return response


def _silent_server(global_config_dict) -> SimpleResponsesAPIModel:
    return _SilentModel(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="srv"),
        server_client=MagicMock(spec=ServerClient, global_config_dict=global_config_dict),
    )


def test_a_response_without_token_ids_marks_the_rollout_incomplete(tmp_path):
    """Treat missing token ids as an incomplete capture.

    Silent omission makes the rollout look complete.
    Generated tokens may then enter the next prompt with mask zero.
    """
    client = TestClient(_silent_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post("/ng-rollout/silent0-roll0/training-token-capture/v1/responses", json={"input": "hi"})
    # The model call itself still succeeds; capture never breaks the harness's run.
    assert resp.status_code == 200

    store = TokenCaptureStore(tmp_path)
    assert store.read_entries("silent0-roll0") == []
    assert store.is_incomplete("silent0-roll0")


def _external_mode(tmp_path) -> dict:
    """Capture on, no destination in this process: records are staged elsewhere."""
    return {
        "observability_enabled": False,
        "token_id_capture": {"enabled": True, "rebuild_response": False, "allow_unresolved_continuations": True},
    }


def test_external_mode_still_mints_identity_for_a_correlated_call(tmp_path):
    """Create capture identity without a local destination.

    Framework inference workers use the identity minted here.
    Local destination availability must not gate that identity.
    """
    seen = {}

    class _Peek(_CapturingModel):
        async def responses(self, request: Request, body=Body()) -> NeMoGymResponse:
            context = current_capture_context()
            seen["rollout_id"] = context.rollout_id if context else None
            seen["model_call_id"] = context.model_call_id if context else None
            seen["sink"] = context.token_sink if context else "no context"
            return _training_response("hi")

    model = _Peek(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="srv"),
        server_client=MagicMock(spec=ServerClient, global_config_dict=_external_mode(tmp_path)),
    )
    assert (
        TestClient(model.setup_webserver())
        .post("/ng-rollout/ext0-r0/training-token-capture/v1/responses", json={"input": "hi"})
        .status_code
        == 200
    )

    assert seen["rollout_id"] == "ext0-r0"
    assert seen["model_call_id"], "a call id has to be minted for the staged record to key on"
    assert seen["sink"] is None, "no destination in this process"


def test_external_mode_does_not_mark_a_token_less_response_incomplete(tmp_path):
    """Leave completeness to external staging without a local destination.

    This process cannot distinguish a lost call from normal external capture.
    Marking locally would mask every rollout.
    """
    client = TestClient(_silent_server(_external_mode(tmp_path)).setup_webserver())
    assert (
        client.post("/ng-rollout/ext1-r0/training-token-capture/v1/responses", json={"input": "hi"}).status_code == 200
    )
    assert list(tmp_path.glob("**/*.incomplete")) == []


def test_a_committed_call_is_not_marked_even_without_token_ids(tmp_path):
    """A caller that had the arrays when this process did not has already accounted for the call."""
    store = TokenCaptureStore(tmp_path)
    context = CaptureContext(rollout_id="cm0-r0", model_call_id="c1", token_sink=store)
    token = set_token_sink(context)
    try:
        asyncio.run(
            commit_entry(
                TokenEntry(
                    rollout_id="cm0-r0",
                    model_call_id="c1",
                    prompt_token_ids=PTOKS,
                    generation_token_ids=GTOKS,
                    generation_log_probs=LPS,
                )
            )
        )
        assert context.committed is True
        asyncio.run(capture_tokens({"output": [{"type": "message"}]}))
    finally:
        reset_token_sink(token)

    assert not store.is_incomplete("cm0-r0"), "the call was recorded, so it is not a hole"


def test_untagged_traffic_without_token_ids_marks_nothing(tmp_path):
    """No rollout prefix means no sink, so there is no rollout to call incomplete."""
    client = TestClient(_silent_server(_both_enabled(tmp_path)).setup_webserver())
    assert client.post("/v1/responses", json={"input": "hi"}).status_code == 200
    assert list(tmp_path.glob("**/*.incomplete")) == []


def test_delete_removes_records_and_marker(tmp_path):
    store = TokenCaptureStore(tmp_path)
    store.append(
        TokenEntry(
            rollout_id="gone-0",
            model_call_id="c",
            prompt_token_ids=PTOKS,
            generation_token_ids=GTOKS,
            generation_log_probs=LPS,
        )
    )
    asyncio.run(store.mark_incomplete("gone-0", "c"))
    assert store.path_for("gone-0").exists() and store.is_incomplete("gone-0")
    store.delete("gone-0")
    assert not store.path_for("gone-0").exists()
    assert not store.is_incomplete("gone-0")
    # Idempotent: consuming a rollout twice must not raise.
    store.delete("gone-0")


def test_concurrent_appends_to_one_rollout_stay_intact(tmp_path):
    """Keep concurrent writers from interleaving partial records.

    The exclusive file lock covers threads and processes.
    """
    import concurrent.futures

    store = TokenCaptureStore(tmp_path)
    entries = [
        TokenEntry(
            rollout_id="r0",
            model_call_id=f"call-{i}",
            prompt_token_ids=list(range(200)),
            generation_token_ids=[i] * 64,
            generation_log_probs=[-0.1] * 64,
        )
        for i in range(32)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(store.append, entries))

    read_back = store.read_entries("r0")
    assert len(read_back) == 32
    assert sorted(e.model_call_id for e in read_back) == sorted(e.model_call_id for e in entries)
    # Every line parsed, so no write landed inside another.
    assert all(len(e.generation_token_ids) == 64 for e in read_back)


# --- framework-owned sink: the documented extension point ---------------------


class _RecordingSink:
    """Implement ``TokenSink`` without a file store.

    Framework transports may keep no local files.
    The capture path must accept this protocol-only implementation.
    """

    def __init__(self) -> None:
        self.entries: list[TokenEntry] = []
        self.incomplete: list[tuple[str, str]] = []

    async def put(self, entry: TokenEntry) -> None:
        self.entries.append(entry)

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        self.incomplete.append((rollout_id, model_call_id))

    async def close(self) -> None:
        pass


@pytest.fixture
def installed_sink():
    sink = _RecordingSink()
    install_token_sink(sink)
    try:
        yield sink
    finally:
        install_token_sink(None)


def test_installed_sink_receives_entries_without_a_capture_dir(installed_sink):
    """The framework path: capture on, no directory anywhere, records still arrive."""
    config = {"token_id_capture": {"enabled": True, "rebuild_response": False, "allow_unresolved_continuations": True}}
    client = TestClient(_server(config).setup_webserver())
    resp = client.post("/ng-rollout/task0-sink0/training-token-capture/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    assert len(installed_sink.entries) == 1
    assert installed_sink.entries[0].generation_token_ids == GTOKS
    assert installed_sink.entries[0].rollout_id == "task0-sink0"


def test_config_allows_no_directory_when_a_sink_is_installed(installed_sink):
    """Requiring a directory would block the sink-only deployment the docstring describes."""
    assert TokenIdCaptureConfig.model_validate(_block()).resolved_dir() is None


def test_config_allows_capture_with_no_destination_at_all():
    """Allow external staging without a local store.

    This process still resolves the capture identity.
    """
    cfg = TokenIdCaptureConfig.model_validate(_block())
    assert cfg.enabled is True
    assert cfg.resolved_dir() is None
    assert cfg.build_sink() is None


def test_installed_sink_is_marked_incomplete_through_the_protocol(installed_sink, monkeypatch):
    """Send incomplete state through the ``TokenSink`` protocol.

    Capture code must not require concrete store attributes.
    """

    async def boom(entry):
        raise RuntimeError("transport down")

    monkeypatch.setattr(installed_sink, "put", boom)
    client = TestClient(
        _server(
            {"token_id_capture": {"enabled": True, "rebuild_response": False, "allow_unresolved_continuations": True}}
        ).setup_webserver()
    )
    resp = client.post("/ng-rollout/task0-sink1/training-token-capture/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200  # capture never fails the model call
    assert installed_sink.incomplete == [("task0-sink1", installed_sink.incomplete[0][1])]


def test_a_sink_without_mark_incomplete_is_refused_at_install():
    """Reject a sink that cannot report a lost call.

    Without ``mark_incomplete``, a failed write could leave no durable evidence.
    Installation rejects that sink before it can silently lose capture records.
    """

    class _PutOnlySink:
        async def put(self, entry):
            raise RuntimeError("transport down")

    from nemo_gym.token_id_capture import installed_token_sink

    with pytest.raises(TypeError, match="mark_incomplete"):
        install_token_sink(_PutOnlySink())
    assert installed_token_sink() is None


def test_commit_entry_records_a_call_with_no_token_fields_on_the_response(installed_sink):
    """Allow engine-side capture to commit an existing entry.

    Engine-side callers already have the token arrays.
    They should share the standard durability path.
    """
    entry = TokenEntry(
        rollout_id="task0-sink3",
        model_call_id="mc-1",
        prompt_token_ids=PTOKS,
        generation_token_ids=GTOKS,
        generation_log_probs=LPS,
    )
    token = set_token_sink(CaptureContext(rollout_id="task0-sink3", model_call_id="mc-1", token_sink=installed_sink))
    try:
        asyncio.run(commit_entry(entry))
    finally:
        reset_token_sink(token)
    assert len(installed_sink.entries) == 1
    # The commit step stamps lineage even when the caller skips extraction.
    assert installed_sink.entries[0].cum_len == len(PTOKS) + len(GTOKS)
    assert installed_sink.entries[0].digest


def test_records_carry_a_schema_version():
    """Writer and reader are different processes and may be different repositories."""
    assert TOKEN_ENTRY_RECORD_SCHEMA_VERSION == 1
    entry = TokenEntry(
        rollout_id="r",
        model_call_id="c",
        prompt_token_ids=[1],
        generation_token_ids=[2],
        generation_log_probs=[-0.1],
    )
    assert entry.schema_version == TOKEN_ENTRY_RECORD_SCHEMA_VERSION
    assert "schema_version" in entry.model_dump_json()


def test_a_malformed_token_payload_does_not_fail_the_model_call(installed_sink):
    """Guard record construction failures.

    ``capture_tokens`` runs on the model response path.
    Invalid token fields must not fail the model call.
    The rollout must still be marked incomplete.
    """
    entry_ctor = TokenEntry

    def _bad_entry(**kwargs):
        # Stand in for a payload that fails validation, e.g. token ids that are not integers.
        raise ValueError("prompt_token_ids: not a list of ints")

    with patch("nemo_gym.token_id_capture.sink.TokenEntry", _bad_entry):
        client = TestClient(
            _server(
                {
                    "token_id_capture": {
                        "enabled": True,
                        "rebuild_response": False,
                        "allow_unresolved_continuations": True,
                    }
                }
            ).setup_webserver()
        )
        resp = client.post("/ng-rollout/task0-bad0/training-token-capture/v1/responses", json={"input": "hi"})

    assert resp.status_code == 200, "a malformed token payload must not fail the model call"
    assert installed_sink.entries == [], "nothing should have been written"
    assert [r for r, _ in installed_sink.incomplete] == ["task0-bad0"], (
        "the rollout lost a call and must not look complete"
    )
    assert entry_ctor is TokenEntry  # patch scoped


def test_capture_stamps_cum_len_and_digest(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    client.post("/ng-rollout/lineage0-roll0/training-token-capture/v1/responses", json={"input": "hi"})
    (entry,) = TokenCaptureStore(tmp_path).read_entries("lineage0-roll0")
    assert entry.cum_len == len(PTOKS) + len(GTOKS)
    assert entry.digest == compute_digest(PTOKS + GTOKS)
    # A missing parent link makes the builder match strict token prefixes.
    assert entry.parent_call_id is None


def test_digest_round_trip_and_stamp_lineage():
    entry = TokenEntry(
        rollout_id="r",
        model_call_id="c",
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.5],
    )
    stamp_lineage(entry, "parent-1")
    assert cumulative_tokens(entry) == [1, 2, 3]
    assert entry.cum_len == 3
    assert entry.parent_call_id == "parent-1"
    assert entry.digest == compute_digest([1, 2, 3])
    # Distinct sequences must not collide.
    # The empty sequence has a stable digest.
    assert compute_digest([1, 2, 3]) != compute_digest([1, 2, 4])
    assert compute_digest([]) == compute_digest([])
    with pytest.raises(ValueError):
        compute_digest([-1])


def test_fingerprint_ignores_non_assistant_turns():
    """Use only model-authored turns for lineage lookup."""
    a = assistant_fingerprint([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    b = assistant_fingerprint([{"role": "user", "content": "DIFFERENT"}, {"role": "assistant", "content": "a"}])
    assert a == b != ""
    # A request without an assistant turn starts a new conversation.
    assert assistant_fingerprint([{"role": "user", "content": "q"}]) == ""


def test_fingerprint_survives_tool_argument_reserialization():
    """Match tool arguments across equivalent JSON serializations."""
    compact = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": '{"b":1,"a":2}'}}]}
    ]
    pretty = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": '{\n  "a": 2,\n  "b": 1\n}'}}],
        }
    ]
    assert assistant_fingerprint(compact) == assistant_fingerprint(pretty)


def test_lineage_resolves_the_parent_across_a_turn():
    lineage = RolloutLineage()
    first_request = [{"role": "user", "content": "hello"}]
    lineage.record("call-1", first_request + [{"role": "assistant", "content": "hi"}], [1, 2, 3], "d1")

    # The next request echoes the assistant turn.
    second_request = first_request + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}]
    parent = lineage.resolve(second_request)
    assert parent.status == ParentResolutionStatus.RESOLVED
    assert parent.match is not None and parent.match.model_call_id == "call-1"
    assert parent.match.cumulative_token_ids == (1, 2, 3)


def test_lineage_record_is_idempotent():
    lineage = RolloutLineage()
    messages = [{"role": "assistant", "content": "hi"}]
    lineage.record("call-1", messages, [1, 2, 3], "d1")
    lineage.record("call-1", messages, [1, 2, 3], "d1")

    parent = lineage.resolve(messages)
    assert parent.status == ParentResolutionStatus.RESOLVED
    assert parent.match is not None and parent.match.model_call_id == "call-1"


def test_lineage_rejects_a_conflicting_call_identity():
    lineage = RolloutLineage()
    lineage.record("call-1", [{"role": "assistant", "content": "a"}], [1], "d1")

    with pytest.raises(ValueError, match="conflicting lineage record"):
        lineage.record("call-1", [{"role": "assistant", "content": "b"}], [2], "d2")


def test_lineage_misses_on_a_rewritten_history():
    """Treat compacted or rewritten model history as unresolved."""
    lineage = RolloutLineage()
    lineage.record("call-1", [{"role": "assistant", "content": "hi"}], [1, 2, 3], "d1")
    assert (
        lineage.resolve([{"role": "assistant", "content": "a summary of the above"}]).status
        == ParentResolutionStatus.UNRESOLVED
    )


def test_lineage_refuses_an_ambiguous_parent():
    """Refuse to guess between calls with identical output."""
    lineage = RolloutLineage()
    messages = [{"role": "assistant", "content": "same"}]
    lineage.record("call-a", messages, [1, 2], "da")
    lineage.record("call-b", messages, [3, 4], "db")
    assert lineage.resolve(messages).status == ParentResolutionStatus.UNRESOLVED


def test_lineage_is_a_tree_so_forks_get_the_parent_not_the_previous_call():
    """Resolve both branches to their shared parent.

    A running cursor would give the second branch the first branch's generation.
    Exact prefix supply would then consume the wrong cumulative tokens.
    """
    lineage = RolloutLineage()
    shared = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "plan"}]
    lineage.record("parent", shared, [1, 2, 3], "dp")
    lineage.record(
        "branch-a",
        shared + [{"role": "user", "content": "a"}, {"role": "assistant", "content": "A"}],
        [1, 2, 3, 4],
        "da",
    )

    # The second branch continues the shared parent.
    second = shared + [{"role": "user", "content": "b"}]
    parent = lineage.resolve(second)
    assert parent.status == ParentResolutionStatus.RESOLVED
    assert parent.match is not None and parent.match.model_call_id == "parent"
    assert parent.match.cumulative_token_ids == (1, 2, 3)


def test_lineage_index_is_bounded():
    """Bound worker-local lineage for abandoned rollouts."""
    index = LineageIndex(max_rollouts=3)
    for i in range(10):
        index.for_rollout(f"r{i}")
    assert len(index) == 3


def _put_shared_file_entry(
    root: str,
    rollout_id: str = "process-shared",
    model_call_id: str = "child-process-call",
    response_text: str = "hi",
) -> None:
    request = [{"role": "user", "content": "hello"}]
    entry = TokenEntry(
        rollout_id=rollout_id,
        model_call_id=model_call_id,
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.1],
        output_items=[{"role": "assistant", "content": response_text}],
    )
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    stamp_continuation(entry, request)
    TokenCaptureStore(root).append(entry)


def _put_shared_file_delta_chain(root: str, depth: int) -> tuple[list[dict], list[int]]:
    """Write a resolved delta chain and return its next request and cumulative tokens."""
    rollout_id = "delta-chain"
    request = [{"role": "user", "content": "start"}]
    cumulative: list[int] = []
    store = TokenCaptureStore(root)
    for index in range(depth):
        output = [{"role": "assistant", "content": f"answer-{index}"}]
        parent_call_id = f"call-{index - 1}" if index else None
        parent_resolution = ParentResolutionStatus.RESOLVED if index else ParentResolutionStatus.ROOT
        prompt_tokens = [10 + index]
        generation_tokens = [100 + index]
        cumulative = cumulative + prompt_tokens + generation_tokens
        entry = TokenEntry(
            rollout_id=rollout_id,
            model_call_id=f"call-{index}",
            prompt_token_ids=prompt_tokens,
            generation_token_ids=generation_tokens,
            generation_log_probs=[-0.1],
            output_items=output,
            parent_call_id=parent_call_id,
            parent_resolution=parent_resolution,
            prompt_is_delta=index > 0,
        )
        stamp_continuation(entry, request)
        stamp_lineage(
            entry,
            parent_call_id,
            parent_resolution=parent_resolution,
            cumulative=cumulative,
        )
        store.append(entry)
        request.extend(output)
        request.append({"role": "user", "content": f"continue-{index}"})
    return request, cumulative


async def test_file_lineage_resolves_across_independent_worker_instances(tmp_path):
    reader = FileLineageStore(tmp_path)
    request = [{"role": "user", "content": "hello"}]
    response = [{"role": "assistant", "content": "hi"}]
    _put_shared_file_entry(str(tmp_path), "shared-rollout", "call-1")

    parent = await reader.resolve("shared-rollout", request + response + [{"role": "user", "content": "next"}])

    assert parent.status == ParentResolutionStatus.RESOLVED
    assert parent.match is not None
    assert parent.match.model_call_id == "call-1"
    assert parent.match.cumulative_token_ids == (1, 2, 3)


async def test_file_lineage_batch_loads_and_flattens_a_delta_chain(tmp_path):
    request, expected_tokens = _put_shared_file_delta_chain(str(tmp_path), depth=100)
    resolver = FileLineageStore(tmp_path)

    with patch.object(resolver, "_load_entries", wraps=resolver._load_entries) as load_entries:
        resolution = await resolver.resolve("delta-chain", request)

    assert resolution.status == ParentResolutionStatus.RESOLVED
    assert resolution.match is not None
    assert resolution.match.model_call_id == "call-99"
    assert resolution.match.cumulative_token_ids == tuple(expected_tokens)
    load_entries.assert_called_once()
    assert len(load_entries.call_args.args[1]) == 100


async def test_file_lineage_reuses_the_latest_materialized_parent(tmp_path):
    request, expected_tokens = _put_shared_file_delta_chain(str(tmp_path), depth=100)
    resolver = FileLineageStore(tmp_path)
    assert (await resolver.resolve("delta-chain", request)).status == ParentResolutionStatus.RESOLVED

    output = [{"role": "assistant", "content": "answer-100"}]
    expected_tokens.extend([110, 200])
    entry = TokenEntry(
        rollout_id="delta-chain",
        model_call_id="call-100",
        prompt_token_ids=[110],
        generation_token_ids=[200],
        generation_log_probs=[-0.1],
        output_items=output,
        parent_call_id="call-99",
        parent_resolution=ParentResolutionStatus.RESOLVED,
        prompt_is_delta=True,
    )
    stamp_continuation(entry, request)
    stamp_lineage(
        entry,
        "call-99",
        parent_resolution=ParentResolutionStatus.RESOLVED,
        cumulative=expected_tokens,
    )
    TokenCaptureStore(tmp_path).append(entry)
    request.extend(output)
    request.append({"role": "user", "content": "continue-100"})

    with patch.object(resolver, "_load_entries", wraps=resolver._load_entries) as load_entries:
        resolution = await resolver.resolve("delta-chain", request)

    assert resolution.match is not None
    assert resolution.match.cumulative_token_ids == tuple(expected_tokens)
    assert len(load_entries.call_args.args[1]) == 1


async def test_file_lineage_cache_is_lru_and_keeps_nodes_metadata_only(tmp_path):
    resolver = FileLineageStore(tmp_path, max_cached_rollouts=2)
    continuation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "next"},
    ]
    for rollout_id in ("r-a", "r-b", "r-c"):
        _put_shared_file_entry(str(tmp_path), rollout_id, f"{rollout_id}-c1")

    await resolver.resolve("r-a", continuation)
    await resolver.resolve("r-b", continuation)
    await resolver.resolve("r-a", continuation)
    await resolver.resolve("r-c", continuation)

    assert "r-a" in resolver._cache
    assert "r-b" not in resolver._cache
    node = resolver._cache["r-a"][2].by_call_id["r-a-c1"]
    assert node.cum_tokens is None
    assert node.entry_offset >= 0

    cold = await resolver.resolve("r-b", continuation)
    assert cold.status == ParentResolutionStatus.RESOLVED
    assert cold.match is not None
    assert cold.match.cumulative_token_ids == (1, 2, 3)


async def test_file_lineage_materialized_cache_has_a_global_token_bound(tmp_path):
    resolver = FileLineageStore(tmp_path, max_cached_tokens=4)
    continuation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "next"},
    ]
    for rollout_id in ("r-a", "r-b"):
        _put_shared_file_entry(str(tmp_path), rollout_id, f"{rollout_id}-c1")
        assert (await resolver.resolve(rollout_id, continuation)).status == ParentResolutionStatus.RESOLVED

    assert "r-a" not in resolver._materialized
    assert resolver._materialized["r-b"][1] == (1, 2, 3)
    assert resolver._materialized_tokens == 3


def test_file_lineage_uses_bounded_striped_locks(tmp_path):
    resolver = FileLineageStore(tmp_path)

    for index in range(10_000):
        resolver._rollout_lock(f"r-{index}")

    assert len(resolver._rollout_locks) == 256


async def test_file_lineage_appends_without_rewriting_prior_records(tmp_path):
    _put_shared_file_entry(str(tmp_path), "shared-rollout", "call-1", "first")
    path = tmp_path / "shared-rollout.tokens.jsonl"
    first_payload = path.read_bytes()
    first_inode = path.stat().st_ino

    _put_shared_file_entry(str(tmp_path), "shared-rollout", "call-2", "second")

    payload = path.read_bytes()
    assert path.stat().st_ino == first_inode
    assert payload.startswith(first_payload)
    assert len(payload.splitlines()) == 2


async def test_file_lineage_resolves_across_spawned_worker_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_put_shared_file_entry, args=(str(tmp_path),))
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0

    store = FileLineageStore(tmp_path)
    parent = await store.resolve(
        "process-shared",
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "next"},
        ],
    )
    assert parent.status == ParentResolutionStatus.RESOLVED
    assert parent.match is not None and parent.match.model_call_id == "child-process-call"


async def test_failed_token_commit_never_becomes_resolver_visible(tmp_path, monkeypatch):
    token_store = TokenCaptureStore(tmp_path)
    resolver = FileLineageStore(tmp_path)
    request = [{"role": "user", "content": "hello"}]
    entry = TokenEntry(
        rollout_id="failed-publication",
        model_call_id="call-1",
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.1],
        output_items=[{"role": "assistant", "content": "hi"}],
    )
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    stamp_continuation(entry, request)

    monkeypatch.setattr(token_store, "append", MagicMock(side_effect=RuntimeError("write failed")))
    with pytest.raises(RuntimeError, match="write failed"):
        await token_store.put(entry)

    resolution = await resolver.resolve(
        entry.rollout_id,
        request + entry.output_items + [{"role": "user", "content": "next"}],
    )
    assert resolution.status == ParentResolutionStatus.UNRESOLVED


async def test_retirement_invalidates_warm_worker_indexes(tmp_path):
    _put_shared_file_entry(str(tmp_path), "retired-rollout", "call-1")
    resolver = FileLineageStore(tmp_path)
    continuation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "next"},
    ]
    assert (await resolver.resolve("retired-rollout", continuation)).status == ParentResolutionStatus.RESOLVED

    token_store = TokenCaptureStore(tmp_path)
    lock_inode = token_store.lock_path_for("retired-rollout").stat().st_ino
    snapshot = await token_store.freeze("retired-rollout")
    assert await token_store.drop(
        "retired-rollout",
        snapshot_id=snapshot.snapshot_id,
        version=snapshot.version,
    )

    assert (await resolver.resolve("retired-rollout", continuation)).status == ParentResolutionStatus.UNRESOLVED
    assert token_store.lock_path_for("retired-rollout").stat().st_ino == lock_inode
    with pytest.raises(RuntimeError, match="retired"):
        _put_shared_file_entry(str(tmp_path), "retired-rollout", "late-call")


def test_served_calls_link_to_their_parent(tmp_path):
    """Record the echoed first call as the second call's parent."""
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    first = [{"role": "user", "content": "hello"}]
    client.post("/ng-rollout/lin0-roll0/training-token-capture/v1/chat/completions", json={"messages": first})
    entries = TokenCaptureStore(tmp_path).read_entries("lin0-roll0")
    assert len(entries) == 1 and entries[0].parent_call_id is None
    assert entries[0].parent_resolution == ParentResolutionStatus.ROOT
    assert entries[0].continuation_fingerprint
    assert entries[0].continuation_context_digest

    content = entries[0].output_items[0]["content"]
    served_text = content if isinstance(content, str) else content[0]["text"]
    second = first + [{"role": "assistant", "content": served_text}, {"role": "user", "content": "more"}]
    client.post("/ng-rollout/lin0-roll0/training-token-capture/v1/chat/completions", json={"messages": second})

    entries = TokenCaptureStore(tmp_path).read_entries("lin0-roll0")
    assert len(entries) == 2
    assert entries[1].parent_call_id == entries[0].model_call_id
    assert entries[1].parent_resolution == ParentResolutionStatus.RESOLVED


def test_separate_model_worker_instances_share_committed_lineage(tmp_path):
    config = _both_enabled(tmp_path)
    worker_a = TestClient(_server(config, num_workers=2).setup_webserver())
    worker_b = TestClient(_server(config, num_workers=2).setup_webserver())
    first = [{"role": "user", "content": "hello"}]

    worker_a.post("/ng-rollout/two-workers/training-token-capture/v1/chat/completions", json={"messages": first})
    first_entry = TokenCaptureStore(tmp_path).read_entries("two-workers")[0]
    content = first_entry.output_items[0]["content"]
    served_text = content if isinstance(content, str) else content[0]["text"]
    second = first + [{"role": "assistant", "content": served_text}, {"role": "user", "content": "more"}]
    worker_b.post("/ng-rollout/two-workers/training-token-capture/v1/chat/completions", json={"messages": second})

    entries = TokenCaptureStore(tmp_path).read_entries("two-workers")
    assert len(entries) == 2
    assert entries[1].parent_resolution == ParentResolutionStatus.RESOLVED
    assert entries[1].parent_call_id == entries[0].model_call_id


def test_served_calls_do_not_link_across_a_changed_system_prompt(tmp_path):
    """Reject lineage across a changed system prompt.

    Anthropic sends the system prompt beside the message list.
    Reusing the old prefix would continue instructions that the harness did not send.
    """
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    store = TokenCaptureStore(tmp_path)

    def call(rollout, system, messages):
        client.post(
            f"/ng-rollout/{rollout}/training-token-capture/v1/messages",
            json={"system": system, "messages": messages, **_MSG_ARGS},
        )

    first = [{"role": "user", "content": "hello"}]
    call("sys0", "SYSTEM ONE", first)
    served = store.read_entries("sys0")[0]
    content = served.output_items[0]["content"]
    echoed = content if isinstance(content, str) else content[0]["text"]
    second = first + [{"role": "assistant", "content": echoed}, {"role": "user", "content": "more"}]

    # Same conversation, different instructions.
    call("sys0", "SYSTEM TWO", second)
    entries = store.read_entries("sys0")
    assert len(entries) == 2
    assert entries[1].parent_call_id is None
    assert entries[1].parent_resolution == ParentResolutionStatus.UNRESOLVED

    # Unchanged instructions preserve the link.
    call("sys1", "SYSTEM ONE", first)
    call("sys1", "SYSTEM ONE", second)
    linked = store.read_entries("sys1")
    assert len(linked) == 2
    assert linked[1].parent_call_id == linked[0].model_call_id
    assert linked[1].parent_resolution == ParentResolutionStatus.RESOLVED


async def test_lineage_lookup_failure_is_persisted_as_unresolved(tmp_path):
    class _FailingResolver:
        async def resolve(self, rollout_id, request_items):
            raise RuntimeError("resolver unavailable")

        def is_process_shared(self):
            return True

        async def close(self):
            pass

    store = TokenCaptureStore(tmp_path)
    context = CaptureContext(
        rollout_id="lookup-failure",
        model_call_id="call-1",
        token_sink=store,
        lineage_store=_FailingResolver(),
    )
    request = [{"role": "assistant", "content": "prior output"}, {"role": "user", "content": "continue"}]
    token = set_token_sink(context)
    try:
        await resolve_parent(request)
        await capture_tokens(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "next output",
                        "prompt_token_ids": PTOKS,
                        "generation_token_ids": GTOKS,
                        "generation_log_probs": LPS,
                    }
                ]
            },
            request_messages=request,
        )
    finally:
        reset_token_sink(token)

    entry = store.read_entries("lookup-failure")[0]
    assert entry.parent_resolution == ParentResolutionStatus.UNRESOLVED
    assert entry.parent_call_id is None


def test_a_changed_tool_schema_breaks_the_link():
    """Reject lineage across a changed tool schema."""
    turn = [{"role": "user", "content": "hi"}, _ASSISTANT_TURN]
    with_search = _request_messages({"messages": turn, "tools": [{"name": "search"}]})
    with_bash = _request_messages({"messages": turn, "tools": [{"name": "bash"}]})

    lineage = RolloutLineage()
    lineage.record("call-1", with_search, [1, 2, 3], "d1")
    assert (
        lineage.resolve(with_search + [{"role": "user", "content": "next"}]).status == ParentResolutionStatus.RESOLVED
    )
    assert (
        lineage.resolve(with_bash + [{"role": "user", "content": "next"}]).status == ParentResolutionStatus.UNRESOLVED
    )


def test_the_envelope_does_not_change_the_lookup_key():
    """Exclude the request envelope from the assistant fingerprint."""
    turn = [{"role": "user", "content": "hi"}, _ASSISTANT_TURN]
    plain = _request_messages({"messages": turn})
    enveloped = _request_messages({"messages": turn, "instructions": "be brief"})
    assert len(enveloped) == len(plain) + 1
    assert assistant_fingerprint(enveloped) == assistant_fingerprint(plain) != ""


def test_the_envelope_is_stable_across_dict_and_model_tools():
    """Normalize equivalent dictionary and model tool schemas."""

    class _Tool(BaseModel):
        name: str

    as_dicts = _request_messages({"messages": [], "tools": [{"name": "search"}]})
    as_models = _request_messages({"messages": [], "tools": [_Tool(name="search")]})
    assert as_dicts == as_models


def test_fingerprint_matches_across_openai_and_anthropic_tool_shapes():
    """Match equivalent OpenAI and Anthropic tool-call shapes.

    OpenAI records calls in ``tool_calls``.
    Anthropic echoes calls in ``tool_use`` content blocks.
    """
    recorded = [
        {
            "role": "assistant",
            "content": "Let me compute that.",
            "tool_calls": [{"id": "c1", "function": {"name": "Bash", "arguments": '{"command":"echo 6"}'}}],
        }
    ]
    echoed = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me compute that."},
                {"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "echo 6"}},
            ],
        }
    ]
    assert assistant_fingerprint(recorded) == assistant_fingerprint(echoed) != ""


def test_fingerprint_agrees_across_all_three_dialects():
    """Hash equivalent turns identically across all dialects.

    Chat puts tool calls on the message.
    Anthropic nests tool calls in content blocks.
    Responses emits roleless ``function_call`` items.
    """

    anthropic = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "Bash", "input": {"cmd": "ls"}}]},
    ]
    chat = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "Bash", "arguments": '{"cmd":"ls"}'}}],
        },
    ]
    responses = [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "function_call", "name": "Bash", "arguments": '{"cmd":"ls"}', "call_id": "c1"},
    ]

    assert assistant_fingerprint(anthropic) == assistant_fingerprint(chat) == assistant_fingerprint(responses)
    assert assistant_fingerprint(responses) != ""


def test_responses_tool_calls_are_distinguished():
    """Distinguish Responses turns with different tool arguments."""

    def turn(cmd):
        return [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "checking"}]},
            {"type": "function_call", "name": "Bash", "arguments": '{"cmd":"%s"}' % cmd, "call_id": "c1"},
        ]

    assert assistant_fingerprint(turn("ls")) != assistant_fingerprint(turn("rm -rf /"))


def test_tool_call_identity_changes_the_fingerprint():
    def turn(call_id):
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "function": {"name": "Bash", "arguments": '{"cmd":"ls"}'},
                    }
                ],
            }
        ]

    assert assistant_fingerprint(turn("call-a")) != assistant_fingerprint(turn("call-b"))


def test_multimodal_content_changes_the_conversation_digest():
    def request(url):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    {"type": "input_image", "image_url": url},
                ],
            }
        ]

    assert conversation_digest(request("https://example/a.png")) != conversation_digest(
        request("https://example/b.png")
    )


def test_non_object_request_items_fail_closed():
    with pytest.raises(ValueError, match="not an object"):
        conversation_digest([{"role": "user", "content": "ok"}, "unsupported"])


def test_lineage_resolves_a_tool_using_turn_echoed_in_anthropic_shape():
    lineage = RolloutLineage()
    produced = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "Bash", "arguments": '{"command":"factor 420"}'}}],
    }
    lineage.record("call-1", [{"role": "user", "content": "factor 420"}, produced], [1, 2, 3], "d1")

    # The harness echoes the turn as Anthropic blocks.
    # The harness then appends the tool result.
    next_request = [
        {"role": "user", "content": "factor 420"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "factor 420"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "420: 2 2 3 5 7"}],
        },
    ]
    parent = lineage.resolve(next_request)
    assert parent.status == ParentResolutionStatus.RESOLVED
    assert parent.match is not None and parent.match.model_call_id == "call-1"
    assert parent.match.cumulative_token_ids == (1, 2, 3)


@pytest.mark.parametrize("bad", ["", "a/b", "../escape", "a b"])
def test_an_unsafe_rollout_id_is_rejected(tmp_path, bad):
    """Reject rollout ids that could escape the store directory."""
    with pytest.raises(ValueError):
        TokenCaptureStore(tmp_path).path_for(bad)


def test_a_record_is_readable_as_soon_as_put_returns(tmp_path):
    """Make ``put`` durable before it returns.

    Consumers may run in another process after rollout completion.
    Conditional deletion requires all writes to be finished.
    """
    store = TokenCaptureStore(tmp_path)
    entry = TokenEntry(
        rollout_id="r0",
        model_call_id="c1",
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.1],
    )
    asyncio.run(store.put(entry))
    assert [e.model_call_id for e in store.read_entries("r0")] == ["c1"]


def test_a_rollout_that_lost_a_call_is_distinguishable_from_a_complete_one(tmp_path):
    """Expose incomplete capture to consumers.

    Capture failures do not fail model calls.
    Surviving records may otherwise look contiguous.
    """
    store = TokenCaptureStore(tmp_path)
    entry = TokenEntry(
        rollout_id="r0",
        model_call_id="c1",
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.1],
    )
    asyncio.run(store.put(entry))
    assert not store.is_incomplete("r0")
    asyncio.run(store.mark_incomplete("r0", "c2"))
    assert store.is_incomplete("r0")


# --- where records go, and surviving multiple server workers -------------------


class _ConfiguredSink:
    """Constructed by dotted path, so every server process builds its own."""

    entries: list = []

    async def put(self, entry) -> None:
        type(self).entries.append(entry)

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        pass

    async def close(self) -> None:
        pass


class _ConfiguredLineage:
    def __init__(self, namespace: str = "") -> None:
        self.namespace = namespace

    async def resolve(self, rollout_id: str, request_items: list[dict]):
        return LineageResolution(ParentResolutionStatus.ROOT)

    def is_process_shared(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class _NotASink:
    async def put(self, entry) -> None:
        pass

    async def close(self) -> None:
        pass


class _NotCallableSink:
    put = "not a method"

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        pass

    async def close(self) -> None:
        pass


class _KwargSink:
    def __init__(self, endpoint: str, shard: int = 0) -> None:
        self.endpoint, self.shard = endpoint, shard

    async def put(self, entry) -> None:
        pass

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        pass

    async def close(self) -> None:
        pass


def test_a_configured_sink_receives_entries(tmp_path):
    _ConfiguredSink.entries = []
    config = {
        "token_id_capture": {
            "enabled": True,
            "rebuild_response": False,
            "sink": f"{__name__}:_ConfiguredSink",
            "allow_unresolved_continuations": True,
        }
    }
    client = TestClient(_server(config).setup_webserver())

    assert (
        client.post("/ng-rollout/task0-cfg0/training-token-capture/v1/responses", json={"input": "hi"}).status_code
        == 200
    )

    assert [e.rollout_id for e in _ConfiguredSink.entries] == ["task0-cfg0"]
    assert _ConfiguredSink.entries[0].generation_token_ids == GTOKS


def test_a_configured_sink_wins_over_an_installed_one(installed_sink):
    """Both routes exist; the configured one is preferred because it survives extra workers."""
    _ConfiguredSink.entries = []
    config = {
        "token_id_capture": {
            "enabled": True,
            "rebuild_response": False,
            "sink": f"{__name__}:_ConfiguredSink",
            "allow_unresolved_continuations": True,
        }
    }
    client = TestClient(_server(config).setup_webserver())

    assert (
        client.post("/ng-rollout/task0-cfg1/training-token-capture/v1/responses", json={"input": "hi"}).status_code
        == 200
    )

    assert len(_ConfiguredSink.entries) == 1
    assert installed_sink.entries == []


def test_a_sink_receives_its_configured_kwargs():
    """Require explicit constructor wiring for a transport sink."""
    config = TokenIdCaptureConfig.model_validate(
        _block(sink=f"{__name__}:_KwargSink", sink_kwargs={"endpoint": "https://dp", "shard": 3})
    )
    sink = config.build_sink()
    assert (sink.endpoint, sink.shard) == ("https://dp", 3)


def test_a_lineage_store_receives_its_configured_kwargs():
    config = TokenIdCaptureConfig.model_validate(
        _block(
            lineage_store=f"{__name__}:_ConfiguredLineage",
            lineage_store_kwargs={"namespace": "training-run"},
        )
    )
    lineage = config.build_lineage_store()
    assert lineage.namespace == "training-run"


def test_multi_worker_custom_capture_requires_shared_lineage():
    config = _block(sink=f"{__name__}:_ConfiguredSink")
    with pytest.raises(ValueError, match="process-shared lineage resolver"):
        _server(config, num_workers=2).setup_webserver()


def test_multi_worker_custom_capture_accepts_configured_shared_lineage():
    config = _block(
        sink=f"{__name__}:_ConfiguredSink",
        lineage_store=f"{__name__}:_ConfiguredLineage",
    )
    _server(config, num_workers=2).setup_webserver()


def test_multi_worker_file_capture_uses_process_shared_lineage(tmp_path):
    _server(_both_enabled(tmp_path), num_workers=2).setup_webserver()


def test_a_sink_given_kwargs_it_cannot_take_is_refused_at_startup():
    config = TokenIdCaptureConfig.model_validate(_block(sink=f"{__name__}:_KwargSink", sink_kwargs={"nope": 1}))
    with pytest.raises(ValueError, match="sink_kwargs"):
        config.build_sink()


def test_a_sink_that_cannot_report_failures_is_refused_at_startup():
    """Reject sinks that cannot mark incomplete capture."""
    config = TokenIdCaptureConfig.model_validate(
        {
            "token_id_capture": {
                "enabled": True,
                "rebuild_response": False,
                "sink": f"{__name__}:_NotASink",
                "allow_unresolved_continuations": True,
            }
        }
    )
    with pytest.raises(ValueError, match="mark_incomplete"):
        config.build_sink()


def test_a_sink_whose_protocol_member_is_not_callable_is_refused():
    """Require callable methods for the ``TokenSink`` protocol.

    Attribute presence alone is insufficient.
    Derive the checks from the protocol.
    """
    config = TokenIdCaptureConfig.model_validate(
        {
            "token_id_capture": {
                "enabled": True,
                "rebuild_response": False,
                "sink": f"{__name__}:_NotCallableSink",
                "allow_unresolved_continuations": True,
            }
        }
    )
    with pytest.raises(ValueError, match="put"):
        config.build_sink()


@pytest.mark.parametrize(
    "target, expected",
    [("no_colon", "module.path:ClassName"), ("nemo_gym.token_id_capture:Nope", "could not load")],
)
def test_a_malformed_sink_path_is_refused_at_startup(target, expected):
    config = TokenIdCaptureConfig.model_validate(
        {
            "token_id_capture": {
                "enabled": True,
                "rebuild_response": False,
                "sink": target,
                "allow_unresolved_continuations": True,
            }
        }
    )
    with pytest.raises(ValueError, match=expected):
        config.build_sink()


def test_a_programmatically_installed_sink_does_not_reach_a_spawned_worker():
    """Build configured sinks inside spawned workers.

    Uvicorn workers re-import the app module.
    They do not inherit launcher process globals.
    Each worker must construct its configured sink.
    """
    ctx = multiprocessing.get_context("spawn")  # the context uvicorn uses
    queue = ctx.Queue()
    process = ctx.Process(target=_report_installed_sink, args=(queue,))
    process.start()
    process.join(timeout=60)

    assert queue.get(timeout=10) == "None"


def _report_installed_sink(queue) -> None:
    # Runs in the spawned process, which re-imports rather than inheriting.
    from nemo_gym.token_id_capture import installed_token_sink

    queue.put(repr(installed_token_sink()))


def test_the_store_is_a_token_source(tmp_path):
    """Use the file store as the local ``TokenSource``.

    A separate local reader would only forward each call.
    """
    store = TokenCaptureStore(tmp_path)
    assert isinstance(store, TokenSource)

    store.append(
        TokenEntry(
            rollout_id="r0",
            model_call_id="c1",
            prompt_token_ids=[1],
            generation_token_ids=[2],
            generation_log_probs=[-0.1],
        )
    )
    assert [e.model_call_id for e in store.read_entries("r0")] == ["c1"]

    # A colocated source can detect a capture failure.
    # This prevents training on an incomplete rollout.
    assert store.is_incomplete("r0") is False
    asyncio.run(store.mark_incomplete("r0", "c2"))
    assert store.is_incomplete("r0") is True


def _entry_fields(**overrides):
    return dict(
        rollout_id="r0",
        model_call_id="c1",
        prompt_token_ids=[1],
        generation_token_ids=[2],
        generation_log_probs=[-0.1],
        **overrides,
    )


def test_a_record_below_the_schema_floor_is_refused():
    with pytest.raises(ValidationError, match="below the supported minimum"):
        TokenEntry(**_entry_fields(schema_version=TOKEN_ENTRY_MIN_SCHEMA_VERSION - 1))


def test_omitted_optional_schema_fields_use_safe_defaults():
    entry = TokenEntry(**_entry_fields())

    assert entry.prompt_is_delta is False
    assert entry.prefix_requested is False
    assert entry.prefix_supplied is False


def test_a_record_newer_than_this_reader_is_refused():
    """Reject newer records hidden by ``extra="allow"``."""
    with pytest.raises(ValidationError, match="this reader understands up to"):
        TokenEntry(**_entry_fields(schema_version=TOKEN_ENTRY_RECORD_SCHEMA_VERSION + 1))


def test_a_newer_record_in_the_store_fails_the_read_rather_than_being_skipped(tmp_path):
    """Fail loudly instead of training on a partial newer record."""
    store = TokenCaptureStore(tmp_path)
    store.append(TokenEntry(**_entry_fields()))
    path = next(tmp_path.glob("*.tokens.jsonl"))
    record = json.loads(path.read_text().splitlines()[0])
    record["schema_version"] = TOKEN_ENTRY_RECORD_SCHEMA_VERSION + 1
    path.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValidationError):
        store.read_entries("r0")


def test_digest_and_cum_len_are_filled_for_every_entry():
    """Stamp cumulative length and digest on every entry."""
    empty = TokenEntry(
        rollout_id="r",
        model_call_id="e",
        prompt_token_ids=[],
        generation_token_ids=[],
        generation_log_probs=[],
    )
    stamp_lineage(empty, None)
    assert empty.cum_len == 0 and empty.digest == compute_digest([])

    normal = TokenEntry(
        rollout_id="r",
        model_call_id="n",
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.1],
    )
    stamp_lineage(normal, None)
    assert normal.cum_len == 3 and normal.digest == compute_digest([1, 2, 3])


def test_a_rewritten_conversation_does_not_resolve_to_the_original_call():
    """Reject an assistant match when the earlier conversation changed."""
    lineage = RolloutLineage()
    original = [{"role": "user", "content": "solve task ALPHA"}]
    lineage.record("call-1", original + [_ASSISTANT_TURN], cum_tokens=[1, 2, 3], digest="d1")

    compacted = [{"role": "user", "content": "SUMMARY: we were working on task BETA"}, _ASSISTANT_TURN]

    assert lineage.resolve(compacted).status == ParentResolutionStatus.UNRESOLVED


def test_appending_a_tool_result_still_resolves():
    """Resolve a continuation after it appends a tool result."""
    lineage = RolloutLineage()
    sent = [{"role": "user", "content": "q"}]
    lineage.record("call-1", sent + [_ASSISTANT_TURN], cum_tokens=[1, 2, 3], digest="d1")

    continuation = sent + [_ASSISTANT_TURN, {"role": "tool", "content": "search result"}]

    resolved = lineage.resolve(continuation)
    assert resolved.status == ParentResolutionStatus.RESOLVED
    assert resolved.match is not None and resolved.match.model_call_id == "call-1"


def test_two_calls_with_identical_output_resolve_to_neither():
    """Resolve neither call when their outputs are identical."""
    lineage = RolloutLineage()
    messages = [{"role": "user", "content": "q"}, _ASSISTANT_TURN]
    lineage.record("call-1", messages, cum_tokens=[1, 2], digest="d1")
    lineage.record("call-2", messages, cum_tokens=[9, 9], digest="d2")

    assert lineage.resolve(messages).status == ParentResolutionStatus.UNRESOLVED


def test_identical_retries_with_the_same_tokens_share_one_parent():
    lineage = RolloutLineage()
    messages = [{"role": "user", "content": "q"}, _ASSISTANT_TURN]
    lineage.record("call-2", messages, cum_tokens=[1, 2], digest="same")
    lineage.record("call-1", messages, cum_tokens=[1, 2], digest="same")

    resolved = lineage.resolve(messages)

    assert resolved.status == ParentResolutionStatus.RESOLVED
    assert resolved.match is not None
    assert resolved.match.model_call_id == "call-1"
    assert resolved.match.cumulative_token_ids == (1, 2)


def test_a_conversation_with_no_model_turn_starts_a_new_root():
    """Start a new root when the request has no model-authored turn."""
    lineage = RolloutLineage()
    lineage.record("call-1", [{"role": "user", "content": "q"}, _ASSISTANT_TURN], cum_tokens=[1], digest="d")

    assert lineage.resolve([{"role": "user", "content": "a brand new task"}]).status == ParentResolutionStatus.ROOT
    assert assistant_fingerprint([{"role": "user", "content": "q"}]) == ""


def test_two_forks_of_one_call_both_resolve_to_it():
    """Resolve two forks to the same parent and cumulative tokens."""
    lineage = RolloutLineage()
    base = [{"role": "user", "content": "q"}, _ASSISTANT_TURN]
    lineage.record("parent", base, cum_tokens=[1, 2, 3], digest="dp")

    a = lineage.resolve(base + [{"role": "tool", "content": "branch A"}])
    b = lineage.resolve(base + [{"role": "tool", "content": "branch B"}])

    assert a.status == b.status == ParentResolutionStatus.RESOLVED
    assert a.match is not None and b.match is not None
    assert a.match.model_call_id == b.match.model_call_id == "parent"
    assert a.match.cumulative_token_ids == b.match.cumulative_token_ids == (1, 2, 3)


def test_recording_a_child_does_not_mutate_its_parent():
    """Keep a parent immutable while recording children."""
    lineage = RolloutLineage()
    base = [{"role": "user", "content": "q"}, _ASSISTANT_TURN]
    lineage.record("parent", base, cum_tokens=[1, 2, 3], digest="dp")
    before = list(lineage.by_call_id["parent"].cum_tokens)

    for i in range(5):
        lineage.record(f"child-{i}", base + [{"role": "tool", "content": str(i)}], [7, 7], "dc")

    assert lineage.by_call_id["parent"].cum_tokens == before


def test_an_evicted_rollout_resolves_to_nothing_rather_than_to_another_rollout():
    """Resolve an evicted rollout to nothing."""
    index = LineageIndex(max_rollouts=2, max_tokens=10_000_000)
    for name in ("r1", "r2", "r3"):
        index.for_rollout(name).record(name, [{"role": "user", "content": "q"}, _ASSISTANT_TURN], [1], "d")

    assert (
        index.for_rollout("r1").resolve([{"role": "user", "content": "q"}, _ASSISTANT_TURN]).status
        == ParentResolutionStatus.UNRESOLVED
    )


def test_the_last_rollout_is_kept_even_over_budget():
    """Keep the only rollout even when it exceeds the token budget."""
    index = LineageIndex(max_rollouts=1, max_tokens=1)
    messages = [{"role": "user", "content": "q"}, _ASSISTANT_TURN]
    index.for_rollout("r1").record("c1", messages, [1] * 100, "d")

    assert index.for_rollout("r1").resolve(messages).status == ParentResolutionStatus.RESOLVED


def test_a_response_echoed_as_several_items_still_resolves():
    """Resolve a response echoed as several items.

    A Responses harness can echo assistant text and a tool call as separate items.
    Indexing the served items preserves that shape.
    """
    served = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "let me look"}]},
        {"type": "function_call", "name": "search", "arguments": '{"q":"x"}'},
    ]
    sent = [{"role": "user", "content": "find x"}]
    lineage = RolloutLineage()
    lineage.record("call-1", sent + served, cum_tokens=[1, 2, 3], digest="d", context_len=len(sent))

    continuation = sent + served + [{"type": "function_call_output", "output": "42"}]

    resolved = lineage.resolve(continuation)
    assert resolved.status == ParentResolutionStatus.RESOLVED
    assert resolved.match is not None and resolved.match.model_call_id == "call-1"


def test_reasoning_the_harness_drops_does_not_break_resolution():
    """Ignore standalone reasoning that the harness does not echo."""
    served = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}]
    sent = [{"role": "user", "content": "q"}]
    lineage = RolloutLineage()
    lineage.record(
        "call-1",
        sent + [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]}] + served,
        cum_tokens=[1, 2],
        digest="d",
        context_len=len(sent),
    )

    resolved = lineage.resolve(sent + served)
    assert resolved.status == ParentResolutionStatus.RESOLVED
    assert resolved.match is not None and resolved.match.model_call_id == "call-1"


@pytest.mark.parametrize(
    "before, after",
    [
        # Responses stores the payload under ``output``.
        (
            [{"type": "function_call_output", "call_id": "c1", "output": "42 files"}],
            [{"type": "function_call_output", "call_id": "c1", "output": "[truncated]"}],
        ),
        # Anthropic stores the payload under ``content``.
        (
            [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "42 files"}]}],
            [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "[truncated]"}]}],
        ),
        # Chat stores the payload as plain content.
        (
            [{"role": "tool", "tool_call_id": "c1", "content": "42 files"}],
            [{"role": "tool", "tool_call_id": "c1", "content": "[truncated]"}],
        ),
    ],
)
def test_a_rewritten_tool_result_changes_the_conversation_digest(before, after):
    """Change the digest when an earlier tool result changes."""
    assert conversation_digest(before) != conversation_digest(after)


def test_the_fingerprint_still_ignores_tool_results():
    """Exclude appended tool results from the lookup fingerprint."""
    turn = [{"role": "assistant", "content": "ok"}]
    assert assistant_fingerprint(turn) == assistant_fingerprint(
        turn + [{"type": "function_call_output", "output": "42 files"}]
    )
