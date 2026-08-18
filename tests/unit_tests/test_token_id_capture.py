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
There is no HTTP token reader.
"""

import asyncio
import json
import logging
import multiprocessing
import subprocess
import sys
from time import time
from unittest.mock import MagicMock, patch
from uuid import uuid4

import orjson
import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    CaptureStore,
    SimpleResponsesAPIModel,
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
    TOKEN_ENTRY_RECORD_SCHEMA_VERSION,
    TOKEN_FIELDS,
    CaptureContext,
    TokenCaptureStore,
    TokenEntry,
    TokenIdCaptureConfig,
    capture_tokens,
    commit_entry,
    current_capture_context,
    extract_token_fields,
    install_token_sink,
    reset_token_sink,
    set_token_sink,
)
from nemo_gym.token_id_capture.config import token_id_capture_enabled_for_agent
from nemo_gym.token_id_capture.protocols import TokenSource
from nemo_gym.token_id_capture.store import make_token_store


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
    retired = asyncio.run(store.freeze("r0"))
    assert retired.entries == ()
    assert retired.snapshot_id == updated.snapshot_id
    assert retired.version == updated.version
    with pytest.raises(RuntimeError, match="already frozen"):
        asyncio.run(store.put(entry.model_copy(update={"model_call_id": "late-after-drop"})))

    store.delete("r0")
    replacement = entry.model_copy(update={"model_call_id": "replacement"})
    asyncio.run(store.put(replacement))
    assert store.read_entries("r0") == [replacement]


# --- config -------------------------------------------------------------------


def _block(**kwargs) -> dict:
    return {"token_id_capture": {"enabled": True, "rebuild_response": False, **kwargs}}


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


def test_agent_capture_selection_uses_static_agent_config_or_all_agents():
    config = {
        "token_id_capture": {"enabled": True, "rebuild_response": False},
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
        {"token_id_capture": {"enabled": True, "sink": f"{__name__}:_ConfiguredSink"}}
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


def _server(global_config_dict) -> SimpleResponsesAPIModel:
    return _CapturingModel(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="srv"),
        server_client=MagicMock(spec=ServerClient, global_config_dict=global_config_dict),
    )


def _both_enabled(tmp_path) -> dict:
    return {
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path),
        "token_id_capture": {"enabled": True, "dir": str(tmp_path)},
    }


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
        "token_id_capture": {"enabled": True, "rebuild_response": False},
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
    config = {"token_id_capture": {"enabled": True, "rebuild_response": False}}
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
    client = TestClient(_server({"token_id_capture": {"enabled": True, "rebuild_response": False}}).setup_webserver())
    resp = client.post("/ng-rollout/task0-sink1/training-token-capture/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200  # capture never fails the model call
    assert installed_sink.incomplete == [("task0-sink1", installed_sink.incomplete[0][1])]


def test_a_sink_without_mark_incomplete_is_logged_not_swallowed(caplog):
    """The signal cannot be lost quietly: that is the outcome the failure path exists to stop."""

    class _PutOnlySink:
        async def put(self, entry):
            raise RuntimeError("transport down")

    install_token_sink(_PutOnlySink())
    try:
        client = TestClient(
            _server({"token_id_capture": {"enabled": True, "rebuild_response": False}}).setup_webserver()
        )
        with caplog.at_level(logging.ERROR):
            resp = client.post("/ng-rollout/task0-sink2/training-token-capture/v1/responses", json={"input": "hi"})
        assert resp.status_code == 200
        assert any("does not implement mark_incomplete" in r.message for r in caplog.records)
    finally:
        install_token_sink(None)


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
    assert installed_sink.entries[0].generation_token_ids == GTOKS


def test_records_carry_a_schema_version():
    """Writer and reader are different processes and may be different repositories."""
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
            _server({"token_id_capture": {"enabled": True, "rebuild_response": False}}).setup_webserver()
        )
        resp = client.post("/ng-rollout/task0-bad0/training-token-capture/v1/responses", json={"input": "hi"})

    assert resp.status_code == 200, "a malformed token payload must not fail the model call"
    assert installed_sink.entries == [], "nothing should have been written"
    assert [r for r, _ in installed_sink.incomplete] == ["task0-bad0"], (
        "the rollout lost a call and must not look complete"
    )
    assert entry_ctor is TokenEntry  # patch scoped


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
    assert [e.model_call_id for e in asyncio.run(store.tokens_for("r0"))] == ["c1"]


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
        {"token_id_capture": {"enabled": True, "rebuild_response": False, "sink": target}}
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
    assert [e.model_call_id for e in asyncio.run(store.tokens_for("r0"))] == ["c1"]

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


def test_a_record_older_than_this_reader_is_accepted():
    """Use defaults for fields absent from older records."""
    entry = TokenEntry(**_entry_fields(schema_version=TOKEN_ENTRY_RECORD_SCHEMA_VERSION - 1))
    assert entry.generation_token_ids == [2]


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
