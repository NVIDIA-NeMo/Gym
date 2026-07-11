# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import fcntl
import multiprocessing as mp
import threading
from pathlib import Path
from types import MappingProxyType
from unittest.mock import MagicMock

import orjson
import pytest
from fastapi import Body, FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict
from pytest import MonkeyPatch

import nemo_gym.base_responses_api_model as obs
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModel,
    BaseResponsesAPIModelConfig,
    CaptureStore,
    ModelCallCaptureConfig,
    ModelCallRecord,
    SimpleResponsesAPIModel,
    aggregate_model_call_records,
    clear_model_call_captures_for_rollouts,
    maybe_rollout_id_from_run_body,
    merge_model_call_capture_into_record,
    model_call_capture_dirs_from_config,
    read_model_call_records,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import (
    ServerClient,
)


class TestBaseResponsesAPIModel:
    def test_BaseResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")
        BaseResponsesAPIModel(config=config)
        assert "observability_enabled" not in BaseResponsesAPIModelConfig.model_fields
        assert "model_call_capture_dir" not in BaseResponsesAPIModelConfig.model_fields

    def test_SimpleResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")

        class TestSimpleResponsesAPIModel(SimpleResponsesAPIModel):
            async def chat_completions(
                self, request: NeMoGymResponseCreateParamsNonStreaming
            ) -> NeMoGymChatCompletion:
                raise NotImplementedError

            async def responses(self, request: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
                raise NotImplementedError

        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        model = TestSimpleResponsesAPIModel(config=config, server_client=server_client)
        model.setup_webserver()


def _capture_config(tmp_path, *, enabled: bool = True) -> ModelCallCaptureConfig:
    return ModelCallCaptureConfig(
        observability_enabled=enabled,
        model_call_capture_dir=tmp_path if enabled else None,
    )


def _install_capture(app, tmp_path, *, model_server_name: str = "srv") -> None:
    server_mock = MagicMock()
    server_mock.config.name = model_server_name
    SimpleResponsesAPIModel.install_model_call_capture(
        server_mock,
        app,
        _capture_config(tmp_path),
    )


def test_capture_store_preserves_valid_rollout_id(tmp_path):
    store = CaptureStore(tmp_path)

    assert store.path_for("task-1_a.2").name == "task-1_a.2.capture.jsonl"


TEST_ROLLOUT_ID = "my-test-rollout-id"


def _create_test_model_call_record() -> ModelCallRecord:
    return ModelCallRecord(
        rollout_id=TEST_ROLLOUT_ID,
        route="my-test-route",
        timestamp_start=0.0,
        timestamp_end=0.0,
        model_ref=ModelServerRef(type="responses_api_models", name="my-server-name"),
        request=NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    role="user",
                    content=[{"type": "input_text", "text": "hello"}],
                    type="message",
                ),
            ]
        ),
        response=NeMoGymResponse(
            id="resp_123",
            created_at=1691418000,
            model="dummy_model",
            tools=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            text="hi :) how are you?",
                            type="output_text",
                            annotations=[],
                        )
                    ],
                )
            ],
            object="response",
        ),
        error_response=None,
        raw_request=None,
        raw_response=None,
    )


class TestSimpleResponsesAPIModel(SimpleResponsesAPIModel):
    async def chat_completions(self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()):
        raise NotImplementedError

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()):
        return _create_test_model_call_record().response


def _create_test_app_with_model_call_capture(
    tmp_path: Path, TestSimpleResponsesAPIModel_cls: type[SimpleResponsesAPIModel] = TestSimpleResponsesAPIModel
) -> FastAPI:
    mock_server_client = MagicMock(spec=ServerClient)
    mock_server_client.global_config_dict = OmegaConf.create(
        {"observability_enabled": True, "model_call_capture_dir": str(tmp_path)}
    )

    model_server = TestSimpleResponsesAPIModel_cls(
        config=BaseResponsesAPIModelConfig(
            host="",
            port=0,
            entrypoint="",
            name="my-test-server-name",
        ),
        server_client=mock_server_client,
    )

    return model_server.setup_webserver()


def test_capture_store_orjson_round_trip_preserves_unicode_and_blank_lines(tmp_path: Path):
    store = CaptureStore(tmp_path)
    store.path_for(TEST_ROLLOUT_ID).write_bytes(b"\n")

    record = _create_test_model_call_record()
    record.request.input[0].content[0]["text"] = "Unicode payload: café 東京"

    store.record(record)

    assert store.read(TEST_ROLLOUT_ID) == [record.model_dump()]

    record2 = record.model_copy(deep=True)
    record2.request.input[0].content[0]["text"] = "second"

    store.record(record2)
    assert read_model_call_records(store, TEST_ROLLOUT_ID) == [record.model_dump(), record2.model_dump()]


def test_capture_store_raises_on_malformed_nonblank_json(tmp_path: Path):
    store = CaptureStore(tmp_path)
    store.path_for("rollout-1").write_bytes(b'{"request": {}}\n{not-json}\n')

    with pytest.raises(orjson.JSONDecodeError):
        store.read("rollout-1")


def test_raised_call_is_captured_then_reraised(tmp_path: Path):
    """A model call that raises (not just a non-2xx) is captured with an exception category and the
    error is re-raised (response unchanged for the caller)."""

    class MyTestSimpleResponsesAPIModel(TestSimpleResponsesAPIModel):
        async def responses(self, body: dict = Body()) -> dict:
            raise RuntimeError("kaboom")

    app = _create_test_app_with_model_call_capture(tmp_path, MyTestSimpleResponsesAPIModel)

    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/ng-rollout/r-raise/responses", json={"input": "x"})
    assert r.status_code == 500  # error propagated, response unchanged

    calls = read_model_call_records(CaptureStore(tmp_path), "r-raise")
    assert len(calls) == 1
    assert calls[0].response is None
    assert calls[0].error_response is not None and "RuntimeError" in calls[0].error_response


def test_per_rollout_url_prefix_correlates_and_is_openai_compatible(tmp_path: Path):
    """A caller attributes calls through the model base URL. The prefix is stripped before routing,
    while an ordinary unprefixed request remains unobserved."""
    app = _create_test_app_with_model_call_capture(tmp_path)

    client = TestClient(app)

    # Prefixed base_url: routes to /v1/responses and correlates capture by the path id.
    r = client.post("/v1/ng-rollout/task7-roll2/responses", json={"input": "hi"})
    assert r.status_code == 200
    calls = read_model_call_records(CaptureStore(tmp_path), "task7-roll2")
    assert len(calls) == 1

    # Prefixed base_url: routes to /v1/responses and correlates capture by the path id.
    r = client.post("/v1/responses/ng-rollout/task7-roll2/", json={"input": "hi"})
    assert r.status_code == 200
    calls = read_model_call_records(CaptureStore(tmp_path), "task7-roll2")
    assert len(calls) == 2

    # Plain /v1 URL is routed normally but is not captured without an explicit rollout prefix.
    r2 = client.post("/v1/responses", json={"input": "hi"})
    assert r2.status_code == 200
    assert read_model_call_records(CaptureStore(tmp_path), "rollout") == []


def test_per_rollout_prefix_not_stripped_for_non_observed_paths_too(tmp_path: Path):
    """A prefixed but non-observed path (e.g. /v1/models) is not stripped and not found,
    and is not captured (composes with arbitrary endpoints, not just the observed dialects)."""
    app = _create_test_app_with_model_call_capture(tmp_path)

    @app.get("/v1/models")
    async def _models() -> dict:
        return {"object": "list", "data": []}

    client = TestClient(app)

    r = client.get("/v1/ng-rollout/abc/models")
    assert r.status_code == 404
    assert CaptureStore(tmp_path).read("abc") == []


def test_maybe_rollout_id_from_run_body_reads_canonical_indices():
    """The shared accessor agents use to derive the rollout id from a /run request body."""

    mapping = MappingProxyType({TASK_INDEX_KEY_NAME: 3, ROLLOUT_INDEX_KEY_NAME: 1})
    assert maybe_rollout_id_from_run_body(mapping) == "3-1"
    assert maybe_rollout_id_from_run_body({TASK_INDEX_KEY_NAME: 3}) is None  # partial -> None
    assert maybe_rollout_id_from_run_body({}) is None
    assert maybe_rollout_id_from_run_body(None) is None

    # The shape agents actually receive: a run-request model with extra="allow".
    class _Body(BaseModel):
        model_config = ConfigDict(extra="allow")

    body = _Body.model_validate({TASK_INDEX_KEY_NAME: 5, ROLLOUT_INDEX_KEY_NAME: 2})
    assert maybe_rollout_id_from_run_body(body) == "5-2"


# --- capture-store config + init failure ---
def test_model_call_capture_keys_are_reserved_global_config():
    assert {"observability_enabled", "model_call_capture_dir"} <= set(NEMO_GYM_RESERVED_TOP_LEVEL_KEYS)


def test_model_call_capture_config_requires_absolute_dir_when_enabled(tmp_path: Path, monkeypatch: MonkeyPatch):
    with pytest.raises(ValueError, match="required"):
        ModelCallCaptureConfig(observability_enabled=True)
    with pytest.raises(ValueError, match="absolute"):
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir="relative")

    global_config = OmegaConf.create({"observability_enabled": True, "model_call_capture_dir": str(tmp_path)})
    config = ModelCallCaptureConfig.model_validate(global_config)
    store = CaptureStore(config.model_call_capture_dir)
    assert store is not None and store.root == tmp_path
    assert model_call_capture_dirs_from_config(global_config) == [store.root]

    monkeypatch.setenv("NEMO_GYM_MODEL_CALL_CAPTURE_DIR", str(tmp_path))
    assert model_call_capture_dirs_from_config({}) == []
    nested_config = {"policy_model": {"responses_api_models": {"model": {"observability_enabled": True}}}}
    assert model_call_capture_dirs_from_config(nested_config) == []


def test_capture_records_non_json_response_as_error(tmp_path: Path):
    class MyTestSimpleResponsesAPIModel(TestSimpleResponsesAPIModel):
        async def responses(self, body: dict = Body()) -> PlainTextResponse:
            return PlainTextResponse("not json")

    app = _create_test_app_with_model_call_capture(tmp_path, MyTestSimpleResponsesAPIModel)

    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/ng-rollout/rnj/responses", json={"input": "x"})
    assert r.status_code == 500
    records = CaptureStore(tmp_path).read("rnj")
    assert len(records) == 1 and records[0].response is None
    assert records[0].error_response is not None


# --- base-agent correlation helpers ---
def test_base_agent_resolve_model_call_path():
    with_id = SimpleResponsesAPIAgent.resolve_model_call_path(
        None, base_url_or_path="http://my-test-url/v1", body={TASK_INDEX_KEY_NAME: 2}
    )
    assert with_id == "http://my-test-url/v1"

    with_id = SimpleResponsesAPIAgent.resolve_model_call_path(
        None, base_url_or_path="http://my-test-url/v1", body={TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 4}
    )
    assert with_id == "http://my-test-url/v1/ng-rollout/2-4"

    with_id = SimpleResponsesAPIAgent.resolve_model_call_path(
        None, base_url_or_path="/v1/responses", body={TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 4}
    )
    assert with_id == "/v1/responses/ng-rollout/2-4"


def test_maybe_rollout_id_from_run_body_attempt_suffix():
    base = {"_ng_task_index": 3, "_ng_rollout_index": 2}
    assert maybe_rollout_id_from_run_body(base) == "3-2"  # no attempt -> bare key
    assert maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": 0}) == "3-2"  # first attempt -> bare
    assert maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": 1}) == "3-2-a1"
    assert maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": "2"}) == "3-2-a2"  # coerced
    assert maybe_rollout_id_from_run_body({"_ng_rollout_index": 2}) is None  # missing task -> None
    with pytest.raises(ValueError):
        maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": "invalid"})


def _capture_exchange(dialect, model_server, usage, response):
    return {
        "dialect": dialect,
        "model_server": model_server,
        "latency_ms": 1.0,
        "status_code": 200,
        "error_category": None,
        "request": {"input": "hi"},
        "response": {"model": "m", "usage": usage, **response},
    }


def test_merge_capture_attaches_metrics_without_raw_payloads(tmp_path):
    store = CaptureStore(tmp_path)
    store.record(
        "0-0",
        _capture_exchange(
            "responses",
            "A",
            {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]},
        ),
    )

    record = {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0, "response": {"harness": "A"}}
    merge_model_call_capture_into_record(record, [tmp_path])

    capture = record["ng_model_call_capture"]
    assert set(capture) == {"rollout_id", "metrics", "calls"}
    assert capture["rollout_id"] == "0-0"
    assert capture["metrics"]["num_calls"] == 1
    assert capture["calls"][0]["tokens_in"] == 3
    assert "request" not in capture["calls"][0] and "response" not in capture["calls"][0]
    assert record["response"] == {"harness": "A"} and record["reward"] == 1.0


def test_merge_capture_noop_without_capture(tmp_path):
    rec = {"_ng_task_index": 9, "_ng_rollout_index": 9, "reward": 1.0}
    merge_model_call_capture_into_record(rec, [tmp_path])  # no capture file for 9-9
    assert "ng_model_call_capture" not in rec
    merge_model_call_capture_into_record(rec, [])  # no dirs
    assert "ng_model_call_capture" not in rec


def test_merge_capture_surfaces_malformed_data_only_when_active(tmp_path):
    store = CaptureStore(tmp_path)
    store.path_for("9-9").write_bytes(b"{not-json}\n")
    record = {"_ng_task_index": 9, "_ng_rollout_index": 9}

    merge_model_call_capture_into_record(record, [])
    with pytest.raises(orjson.JSONDecodeError):
        merge_model_call_capture_into_record(record, [tmp_path])


def test_clear_model_call_captures_for_rollouts_run_scoping(tmp_path, monkeypatch):
    store = CaptureStore(tmp_path)
    store.record("0-0", {"dialect": "chat", "request": {}, "response": {}})
    store.record("1-0", {"dialect": "chat", "request": {}, "response": {}})
    assert store.read("0-0") and store.read("1-0")

    # Clears only the rollout ids about to be (re)run; rows without indices are skipped, others stay.
    clear_model_call_captures_for_rollouts([{"_ng_task_index": 0, "_ng_rollout_index": 0}, {"no": "id"}], [tmp_path])
    assert store.read("0-0") == [] and store.read("1-0")
    clear_model_call_captures_for_rollouts([{"_ng_task_index": 1, "_ng_rollout_index": 0}], [])  # no dirs -> no-op
    assert store.read("1-0")

    # A stale-capture cleanup failure must be visible rather than mixing old and new calls.
    def _boom(_directory):
        raise OSError("cannot open")

    monkeypatch.setattr(obs, "CaptureStore", _boom)
    with pytest.raises(OSError, match="cannot open"):
        clear_model_call_captures_for_rollouts([{"_ng_task_index": 1, "_ng_rollout_index": 0}], [tmp_path])


def test_aggregate_model_call_records_sums_and_counts():
    calls = [
        ModelCallRecord(call_index=0, tokens_in=10, tokens_out=5, tokens_total=15, latency_total_ms=2.0),
        ModelCallRecord(call_index=1, tokens_in=20, tokens_out=3, tokens_total=23, latency_total_ms=1.0),
    ]
    agg = aggregate_model_call_records(calls)
    assert (agg["tokens_in"], agg["tokens_out"], agg["tokens_total"]) == (30, 8, 38)
    assert agg["latency_total_ms"] == 3.0 and agg["num_calls"] == 2
    # empty -> all-None totals but a well-formed shape (num_calls 0)
    assert aggregate_model_call_records([]) == {
        "tokens_in": None,
        "tokens_out": None,
        "tokens_reasoning": None,
        "tokens_total": None,
        "latency_total_ms": None,
        "num_calls": 0,
    }


def test_rollout_prefix_stripped_when_capture_disabled():
    # The /ng-rollout/<id> prefix must be stripped + routed even when capture is OFF (the default),
    # otherwise a default `gym eval` 404s on every prefixed model call.
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _cc() -> dict:
        return {"ok": True}

    mock_server = MagicMock()
    mock_server.config.name = ""
    SimpleResponsesAPIModel.install_model_call_capture(mock_server, app, ModelCallCaptureConfig())
    client = TestClient(app)
    assert client.post("/v1/chat/completions", json={}).status_code == 200
    assert client.post("/ng-rollout/3-0/v1/chat/completions", json={}).status_code == 200


def test_capture_store_concurrent_append_no_loss(tmp_path):
    store = CaptureStore(tmp_path)

    def _write(i: int) -> None:
        store.record("0-0", {"dialect": "chat", "request": {"i": i}, "response": {}})

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = store.read("0-0")
    assert len(rows) == 20  # flock + in-process lock: no lost or corrupted appends
    assert sorted(r["request"]["i"] for r in rows) == list(range(20))


def test_capture_store_read_waits_for_in_progress_append(tmp_path):
    store = CaptureStore(tmp_path)
    path = store.path_for("0-0")
    writer_ready = threading.Event()
    finish_write = threading.Event()
    reader_done = threading.Event()
    rows = []

    def _write() -> None:
        with path.open("ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(b'{"request":')
                handle.flush()
                writer_ready.set()
                assert finish_write.wait(timeout=5)
                handle.write(b'{},"response":{}}\n')
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read() -> None:
        rows.extend(store.read("0-0"))
        reader_done.set()

    writer = threading.Thread(target=_write)
    reader = threading.Thread(target=_read)
    writer.start()
    assert writer_ready.wait(timeout=5)
    reader.start()
    try:
        assert not reader_done.wait(timeout=0.1)
    finally:
        finish_write.set()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert rows == [{"request": {}, "response": {}}]


def _cross_process_writer(root: str, base: int) -> None:
    # Module-level so it is picklable under the "spawn" start method too.
    store = CaptureStore(root)
    for i in range(base, base + 100):
        store.record("0-0", {"dialect": "chat", "request": {"i": i}, "response": {}})


def test_capture_store_cross_process_append_no_loss(tmp_path):
    # The threads-only test above exercises the in-process lock; this exercises fcntl.flock across
    # *processes* -- the num_workers>1 case the in-process lock cannot coordinate.

    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_cross_process_writer, args=(str(tmp_path), b * 100)) for b in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    rows = CaptureStore(tmp_path).read("0-0")
    assert len(rows) == 400
    assert sorted(r["request"]["i"] for r in rows) == list(range(400))
