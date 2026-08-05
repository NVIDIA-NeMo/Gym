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
"""Training-token capture: schema, store, readers, source, and the served path.

The served-path tests build a real ``SimpleResponsesAPIModel`` so the full chain runs:
the capture middleware mints a ``model_call_id`` and sets a per-request token sink, the
model server records a ``TokenEntry`` from its complete response, and the entry is read
back through the store, the HTTP route, and a ``TokenSource``.
"""

import asyncio
import logging
import multiprocessing
import subprocess
import sys
from time import time
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient

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
    commit_entry,
    extract_token_fields,
    install_token_sink,
    reset_token_sink,
    set_token_sink,
)
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


# --- config -------------------------------------------------------------------


def _block(**kwargs) -> dict:
    return {"token_id_capture": {"enabled": True, **kwargs}}


def test_config_disabled_needs_no_dir():
    cfg = TokenIdCaptureConfig.model_validate({})
    assert cfg.enabled is False
    assert make_token_store({}) is None


def test_config_enabled_requires_absolute_dir(tmp_path):
    with pytest.raises(ValueError):
        TokenIdCaptureConfig.model_validate(_block())
    with pytest.raises(ValueError):
        TokenIdCaptureConfig.model_validate(_block(dir="relative/dir"))
    cfg = TokenIdCaptureConfig.model_validate(_block(dir=str(tmp_path)))
    assert cfg.resolved_dir() == tmp_path


def test_config_falls_back_to_model_call_capture_dir(tmp_path):
    cfg = TokenIdCaptureConfig.model_validate(_block() | {"model_call_capture_dir": str(tmp_path)})
    assert cfg.resolved_dir() == tmp_path


def test_config_keeps_settings_when_capture_is_off(tmp_path):
    """Templated configs set a directory unconditionally and toggle `enabled` per run, so the
    rest of the block is left alone rather than rejected."""
    cfg = TokenIdCaptureConfig.model_validate({"token_id_capture": {"enabled": False, "dir": str(tmp_path)}})
    assert cfg.enabled is False
    assert cfg.build_sink() is None


def test_config_warns_rather_than_fails_on_a_sink_beside_a_directory(caplog):
    """Nothing is lost, the directory is just never read, but someone expecting files on disk
    will not find any."""
    with caplog.at_level(logging.WARNING):
        cfg = TokenIdCaptureConfig.model_validate(_block(sink=f"{__name__}:_ConfiguredSink", dir="/tmp/x"))
    assert cfg.enabled is True
    assert "will not be written to" in caplog.text


def test_config_rejects_an_unknown_key():
    """A typo in this block silently disables capture, so it is refused at startup."""
    with pytest.raises(ValueError):
        TokenIdCaptureConfig.model_validate({"token_id_capture": {"enabled": True, "dirr": "/tmp/x"}})


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
    resp = client.post("/ng-rollout/task0-roll0/v1/responses", json={"input": "hi"})
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
    client.post("/ng-rollout/task0-rollC/v1/responses", json={"input": "hi"})
    tokens = TokenCaptureStore(tmp_path).read_entries("task0-rollC")
    assert len(tokens) == 1
    # Not token-only: the captured record carries the content-bearing output items.
    assert tokens[0].output_items
    text = tokens[0].output_items[-1]["content"][0]["text"]
    assert text == "hi from responses"


def test_token_arrays_are_stored_once(tmp_path):
    """The served response carries the arrays on an output item; the record does not repeat them.

    Storing them again per item roughly doubles a record, and the per-item copy is not the
    value a trainer reads: an item's prompt in a chained trajectory is the running sequence.
    """
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    client.post("/ng-rollout/task0-rollDedup/v1/responses", json={"input": "hi"})
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
        "/ng-rollout/task0-roll1/v1/messages",
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
        "/ng-rollout/task0-roll2/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200
    tokens = TokenCaptureStore(tmp_path).read_entries("task0-roll2")
    assert len(tokens) == 1 and tokens[0].generation_token_ids == GTOKS


def test_tokens_captured_even_when_eval_capture_disabled(tmp_path):
    config = {"token_id_capture": {"enabled": True, "dir": str(tmp_path)}}
    client = TestClient(_server(config).setup_webserver())
    resp = client.post("/ng-rollout/task1-roll0/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    assert len(TokenCaptureStore(tmp_path).read_entries("task1-roll0")) == 1
    # No eval capture file was written.
    assert read_model_call_records(CaptureStore(tmp_path), "task1-roll0") == []


def test_uncorrelated_call_captures_nothing(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post("/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    # No rollout prefix -> nothing recorded, no file created.
    assert list(tmp_path.glob("*.tokens.jsonl")) == []


def test_package_is_dependency_free_leaf():
    """``nemo_gym.token_id_capture`` must import without Gym's server stack.

    A training framework's inference worker imports the record, the protocols,
    and the capture core so it can write into its own data plane (see
    ``protocols.py``). If the package drags in ray/fastapi/uvicorn, that is not
    possible. Run in a subprocess so this test is unaffected by whatever the
    rest of the suite has already imported.
    """
    heavy = ("ray", "fastapi", "uvicorn", "aiohttp", "requests", "torch")
    program = (
        f"import sys; import nemo_gym.token_id_capture; print(','.join(m for m in {heavy!r} if m in sys.modules))"
    )
    proc = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"leaf package pulled in: {proc.stdout.strip()}"


def test_streamed_messages_capture_tokens_absent_from_the_stream(tmp_path):
    """The Claude Code shape: streamed /v1/messages.

    Token ids exist only on the assembled response, before it is converted to
    Anthropic and split into SSE. This is the case the whole design turns on, so
    it is asserted end to end rather than only through the non-streamed path.
    """
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    with client.stream(
        "POST",
        "/ng-rollout/stream0-roll0/v1/messages",
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
    """A lost call must not leave the rollout looking complete.

    Capture stays best-effort so a bad payload cannot break the harness's run,
    but delivery has to be able to tell "10 of 10 captured" from "9 of 10".
    """
    store = TokenCaptureStore(tmp_path)

    async def boom(self, entry):
        raise RuntimeError("sink is down")

    monkeypatch.setattr(TokenCaptureStore, "put", boom)
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    # The model call still succeeds.
    resp = client.post("/ng-rollout/fail0-roll0/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    assert store.read_entries("fail0-roll0") == []
    assert store.is_incomplete("fail0-roll0")


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
    store.mark_incomplete("gone-0", "c")
    assert store.path_for("gone-0").exists() and store.is_incomplete("gone-0")
    store.delete("gone-0")
    assert not store.path_for("gone-0").exists()
    assert not store.is_incomplete("gone-0")
    # Idempotent: consuming a rollout twice must not raise.
    store.delete("gone-0")


def test_concurrent_appends_to_one_rollout_stay_intact(tmp_path):
    """Writes take an exclusive file lock, which is what keeps two writers from interleaving a
    partial line. Under sharding the writers are separate processes, so the lock has to hold there
    too; this covers the same code path with threads."""
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
    """A sink that is only a ``TokenSink``: no file store, no directory.

    Deliberately not a ``TokenCaptureStore`` subclass. A training framework whose sink is
    its own transport has nothing on disk, and this is the shape the capture path has to
    accept for ``install_token_sink`` to mean anything.
    """

    def __init__(self) -> None:
        self.entries: list[TokenEntry] = []
        self.incomplete: list[tuple[str, str]] = []

    async def put(self, entry: TokenEntry) -> None:
        self.entries.append(entry)

    def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        self.incomplete.append((rollout_id, model_call_id))


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
    config = {"token_id_capture": {"enabled": True}}
    client = TestClient(_server(config).setup_webserver())
    resp = client.post("/ng-rollout/task0-sink0/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    assert len(installed_sink.entries) == 1
    assert installed_sink.entries[0].generation_token_ids == GTOKS
    assert installed_sink.entries[0].rollout_id == "task0-sink0"


def test_config_allows_no_directory_when_a_sink_is_installed(installed_sink):
    """Requiring a directory would block the sink-only deployment the docstring describes."""
    assert TokenIdCaptureConfig.model_validate(_block()).resolved_dir() is None


def test_config_still_requires_a_directory_with_no_sink_installed():
    with pytest.raises(ValueError):
        TokenIdCaptureConfig.model_validate(_block())


def test_installed_sink_is_marked_incomplete_through_the_protocol(installed_sink, monkeypatch):
    """A protocol-only sink must receive the incomplete signal.

    Reaching for a concrete store attribute here would raise inside the failure path and be
    swallowed, leaving a rollout that lost a call looking complete.
    """

    async def boom(entry):
        raise RuntimeError("transport down")

    monkeypatch.setattr(installed_sink, "put", boom)
    client = TestClient(_server({"token_id_capture": {"enabled": True}}).setup_webserver())
    resp = client.post("/ng-rollout/task0-sink1/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200  # capture never fails the model call
    assert installed_sink.incomplete == [("task0-sink1", installed_sink.incomplete[0][1])]


def test_a_sink_without_mark_incomplete_is_logged_not_swallowed(caplog):
    """The signal cannot be lost quietly: that is the outcome the failure path exists to stop."""

    class _PutOnlySink:
        async def put(self, entry):
            raise RuntimeError("transport down")

    install_token_sink(_PutOnlySink())
    try:
        client = TestClient(_server({"token_id_capture": {"enabled": True}}).setup_webserver())
        with caplog.at_level(logging.ERROR):
            resp = client.post("/ng-rollout/task0-sink2/v1/responses", json={"input": "hi"})
        assert resp.status_code == 200
        assert any("does not implement mark_incomplete" in r.message for r in caplog.records)
    finally:
        install_token_sink(None)


def test_commit_entry_records_a_call_with_no_token_fields_on_the_response(installed_sink):
    """Engine-side capture: the caller has the arrays, the served response does not.

    The commit half has to be reachable on its own, otherwise a framework in that position
    forks the durability ordering rather than sharing it.
    """
    entry = TokenEntry(
        rollout_id="task0-sink3",
        model_call_id="mc-1",
        prompt_token_ids=PTOKS,
        generation_token_ids=GTOKS,
        generation_log_probs=LPS,
    )
    token = set_token_sink(CaptureContext(rollout_id="task0-sink3", model_call_id="mc-1", sink=installed_sink))
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
    """Building the record is guarded, not just writing it.

    ``capture_tokens`` is awaited directly on the model server's response path, so anything
    it raises fails the model call. A payload whose token fields do not validate has to be
    treated like any other capture failure: the call succeeds and the rollout is marked.
    """
    entry_ctor = TokenEntry

    def _bad_entry(**kwargs):
        # Stand in for a payload that fails validation, e.g. token ids that are not integers.
        raise ValueError("prompt_token_ids: not a list of ints")

    with patch("nemo_gym.token_id_capture.sink.TokenEntry", _bad_entry):
        client = TestClient(_server({"token_id_capture": {"enabled": True}}).setup_webserver())
        resp = client.post("/ng-rollout/task0-bad0/v1/responses", json={"input": "hi"})

    assert resp.status_code == 200, "a malformed token payload must not fail the model call"
    assert installed_sink.entries == [], "nothing should have been written"
    assert [r for r, _ in installed_sink.incomplete] == ["task0-bad0"], (
        "the rollout lost a call and must not look complete"
    )
    assert entry_ctor is TokenEntry  # patch scoped


@pytest.mark.parametrize("bad", ["", "a/b", "../escape", "a b"])
def test_an_unsafe_rollout_id_is_rejected(tmp_path, bad):
    """The id names the capture file, so it has to be a safe filename component: a separator
    would let a rollout id write outside the store directory."""
    with pytest.raises(ValueError):
        TokenCaptureStore(tmp_path).path_for(bad)


def test_a_record_is_readable_as_soon_as_put_returns(tmp_path):
    """``put`` is awaited rather than backgrounded, so the record is on disk before the model
    call returns. A reader in another process runs after the rollout and has no way to wait for
    a writer, and delete-on-consume is only safe because nothing is still in flight."""
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
    """Capture failures do not fail the model call, so nothing downstream would otherwise know
    a turn is missing. The chain built from what survived can look perfectly contiguous."""
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
    store.mark_incomplete("r0", "c2")
    assert store.is_incomplete("r0")


# --- where records go, and surviving multiple server workers -------------------


class _ConfiguredSink:
    """Constructed by dotted path, so every server process builds its own."""

    entries: list = []

    async def put(self, entry) -> None:
        type(self).entries.append(entry)

    def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        pass


class _NotASink:
    async def put(self, entry) -> None:
        pass


class _NotCallableSink:
    put = "not a method"

    def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        pass


class _KwargSink:
    def __init__(self, endpoint: str, shard: int = 0) -> None:
        self.endpoint, self.shard = endpoint, shard

    async def put(self, entry) -> None:
        pass

    def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        pass


def test_a_configured_sink_receives_entries(tmp_path):
    _ConfiguredSink.entries = []
    config = {"token_id_capture": {"enabled": True, "sink": f"{__name__}:_ConfiguredSink"}}
    client = TestClient(_server(config).setup_webserver())

    assert client.post("/ng-rollout/task0-cfg0/v1/responses", json={"input": "hi"}).status_code == 200

    assert [e.rollout_id for e in _ConfiguredSink.entries] == ["task0-cfg0"]
    assert _ConfiguredSink.entries[0].generation_token_ids == GTOKS


def test_a_configured_sink_wins_over_an_installed_one(installed_sink):
    """Both routes exist; the configured one is preferred because it survives extra workers."""
    _ConfiguredSink.entries = []
    config = {"token_id_capture": {"enabled": True, "sink": f"{__name__}:_ConfiguredSink"}}
    client = TestClient(_server(config).setup_webserver())

    assert client.post("/ng-rollout/task0-cfg1/v1/responses", json={"input": "hi"}).status_code == 200

    assert len(_ConfiguredSink.entries) == 1
    assert installed_sink.entries == []


def test_a_sink_receives_its_configured_kwargs():
    """A sink for a real transport needs an endpoint and a client; a zero-argument one could only
    get them from ambient state."""
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
    """Without mark_incomplete an incomplete rollout looks complete, so this fails at startup
    rather than at whichever step first loses a call."""
    config = TokenIdCaptureConfig.model_validate(
        {"token_id_capture": {"enabled": True, "sink": f"{__name__}:_NotASink"}}
    )
    with pytest.raises(ValueError, match="mark_incomplete"):
        config.build_sink()


def test_a_sink_whose_protocol_member_is_not_callable_is_refused():
    """isinstance against a Protocol only checks that the attributes exist, so callability is
    checked too. Both are derived from the protocol rather than a list written out here, so the
    check keeps up if TokenSink gains a method."""
    config = TokenIdCaptureConfig.model_validate(
        {"token_id_capture": {"enabled": True, "sink": f"{__name__}:_NotCallableSink"}}
    )
    with pytest.raises(ValueError, match="put"):
        config.build_sink()


@pytest.mark.parametrize(
    "target, expected",
    [("no_colon", "module.path:ClassName"), ("nemo_gym.token_id_capture:Nope", "could not load")],
)
def test_a_malformed_sink_path_is_refused_at_startup(target, expected):
    config = TokenIdCaptureConfig.model_validate({"token_id_capture": {"enabled": True, "sink": target}})
    with pytest.raises(ValueError, match=expected):
        config.build_sink()


def test_a_programmatically_installed_sink_does_not_reach_a_spawned_worker():
    """A model server with num_workers > 1 is launched by uvicorn with an app string and
    workers=N, and uvicorn spawns those workers (multiprocessing "spawn"), re-importing the app
    module rather than inheriting the launcher's memory. ``install_token_sink`` sets a process
    global, so it does not cross that boundary and capture silently falls back to the file store,
    or writes nothing when no directory is set.

    This is why ``token_id_capture.sink`` is configuration rather than only a function call: it is
    constructed inside each worker. The test pins the limitation so the reason for the config key
    does not get lost.
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
    """Records are read back through a TokenSource, and the file store is one. There is no
    separate local reader: a wrapper over the store would only forward every call."""
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

    # A colocated source can tell that a call failed to capture, which is what keeps an
    # incomplete rollout from being trained on.
    assert store.is_incomplete("r0") is False
    store.mark_incomplete("r0", "c2")
    assert store.is_incomplete("r0") is True
