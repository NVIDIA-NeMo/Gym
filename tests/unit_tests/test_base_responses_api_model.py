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

from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModel,
    BaseResponsesAPIModelConfig,
    CaptureStore,
    ModelCallRecord,
    SimpleResponsesAPIModel,
    maybe_rollout_id_from_run_body,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import (
    ServerClient,
)


class TestBaseResponsesAPIModel:
    def test_BaseResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")
        BaseResponsesAPIModel(config=config)
        assert "should_capture_model_calls" not in BaseResponsesAPIModelConfig.model_fields
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


def test_capture_store_preserves_valid_rollout_id(tmp_path):
    store = CaptureStore(tmp_path)

    assert store.path_for("task-1_a.2").name == "task-1_a.2.capture.jsonl"


TEST_ROLLOUT_ID = "my-test-rollout-id"


def _create_test_model_call_record() -> ModelCallRecord:
    return ModelCallRecord(
        rollout_id=TEST_ROLLOUT_ID,
        status_code=200,
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
        raw_request=dict(),
        raw_response=dict(),
    )


class TestSimpleResponsesAPIModel(SimpleResponsesAPIModel):
    async def chat_completions(self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()):
        raise NotImplementedError

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()):
        return _create_test_model_call_record().response


def _create_test_app_with_model_call_capture(
    tmp_path: Path,
    TestSimpleResponsesAPIModel_cls: type[SimpleResponsesAPIModel] = TestSimpleResponsesAPIModel,
    should_capture_model_calls: bool = True,
) -> FastAPI:
    mock_server_client = MagicMock(spec=ServerClient)
    mock_server_client.global_config_dict = OmegaConf.create(
        {"should_capture_model_calls": should_capture_model_calls, "model_call_capture_dir": str(tmp_path)}
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

    assert store.read(TEST_ROLLOUT_ID) == [record]

    record2 = record.model_copy(deep=True)
    record2.request.input[0].content[0]["text"] = "second"

    store.record(record2)
    assert store.read(TEST_ROLLOUT_ID) == [record, record2]


def test_capture_store_raises_on_malformed_nonblank_json(tmp_path: Path):
    store = CaptureStore(tmp_path)
    store.path_for("rollout-1").write_bytes(b'{"request": {')

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

    calls = CaptureStore(tmp_path).read("r-raise")
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
    calls = CaptureStore(tmp_path).read("task7-roll2")
    assert len(calls) == 1

    # Prefixed base_url: routes to /v1/responses and correlates capture by the path id.
    r = client.post("/v1/responses/ng-rollout/task7-roll2/", json={"input": "hi"})
    assert r.status_code == 200
    calls = CaptureStore(tmp_path).read("task7-roll2")
    assert len(calls) == 2

    # Plain /v1 URL is routed normally but is not captured without an explicit rollout prefix.
    r2 = client.post("/v1/responses", json={"input": "hi"})
    assert r2.status_code == 200
    assert CaptureStore(tmp_path).read("rollout") == []


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
def test_capture_records_non_json_response_as_error_doesnt_record(tmp_path: Path):
    class MyTestSimpleResponsesAPIModel(TestSimpleResponsesAPIModel):
        async def responses(self, body: dict = Body()) -> PlainTextResponse:
            return PlainTextResponse("not json")

    app = _create_test_app_with_model_call_capture(tmp_path, MyTestSimpleResponsesAPIModel)

    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/ng-rollout/rnj/responses", json={"input": "x"})
    assert r.status_code == 500
    records = CaptureStore(tmp_path).read("rnj")
    assert len(records) == 0


# --- base-agent correlation helpers ---
def test_base_agent_resolve_model_call_path():
    mock_agent = MagicMock()
    mock_agent._capture_config.should_capture_model_calls = True

    with_id = SimpleResponsesAPIAgent.resolve_model_call_path(
        mock_agent, base_url_or_path="http://my-test-url/v1", body={TASK_INDEX_KEY_NAME: 2}
    )
    assert with_id == "http://my-test-url/v1"

    with_id = SimpleResponsesAPIAgent.resolve_model_call_path(
        mock_agent, base_url_or_path="http://my-test-url/v1", body={TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 4}
    )
    assert with_id == "http://my-test-url/v1/ng-rollout/2-4"

    with_id = SimpleResponsesAPIAgent.resolve_model_call_path(
        mock_agent, base_url_or_path="/v1/responses", body={TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 4}
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


def test_store_aggregate_sanity(tmp_path: Path):
    store = CaptureStore(tmp_path)

    record1 = _create_test_model_call_record()
    record1.rollout_id = "0-0"
    record1.response.usage = NeMoGymResponseUsage(
        input_tokens=1,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=2),
        output_tokens=3,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=4),
        total_tokens=4,
    )

    store.record(record1)

    aggregate_record = store.aggregate(
        rollout_id=maybe_rollout_id_from_run_body({"_ng_task_index": 0, "_ng_rollout_index": 0})
    )

    assert len(aggregate_record.records) == 1
    assert aggregate_record.records[0].rollout_id == "0-0"
    assert aggregate_record.records[0].response.usage.output_tokens == 3


def test_store_clear(tmp_path: Path, monkeypatch: MonkeyPatch):
    store = CaptureStore(tmp_path)

    record1 = _create_test_model_call_record()
    record1.rollout_id = "0-0"
    store.record(record1)
    record2 = _create_test_model_call_record()
    record2.rollout_id = "1-0"
    store.record(record2)

    assert store.read("0-0") and store.read("1-0")

    store.clear()
    assert store.read("0-0") == [] and store.read("1-0") == []


def test_rollout_prefix_not_stripped_when_capture_disabled(tmp_path: Path):
    app = _create_test_app_with_model_call_capture(tmp_path, should_capture_model_calls=False)

    client = TestClient(app)
    # Dummy input to just test it doesn't 404
    assert client.post("/v1/responses", json={}).status_code == 422
    assert client.post("/v1/ng-rollout/3-0/responses", json={}).status_code == 404


def test_capture_store_concurrent_append_no_loss(tmp_path):
    store = CaptureStore(tmp_path)

    def _write(i: int) -> None:
        store.record(_create_test_model_call_record())

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = store.read(TEST_ROLLOUT_ID)
    assert len(rows) == 20


def test_capture_store_read_waits_for_in_progress_append(tmp_path):
    store = CaptureStore(tmp_path)
    path = store.path_for("0-0")
    writer_ready = threading.Event()
    finish_write = threading.Event()
    reader_done = threading.Event()
    rows = []

    record = _create_test_model_call_record()
    record.rollout_id = "0-0"

    def _write() -> None:
        with path.open("ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

            record_str = record.model_dump_json().encode() + b"\n"
            try:
                handle.write(record_str[: len(record_str) // 2])
                handle.flush()
                writer_ready.set()
                assert finish_write.wait(timeout=5)
                handle.write(record_str[len(record_str) // 2 :])
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
    assert rows == [record]


def _cross_process_writer(root: str, base: int) -> None:
    # Module-level so it is picklable under the "spawn" start method too.
    store = CaptureStore(root)
    record = _create_test_model_call_record()
    record.rollout_id = "0-0"
    for _ in range(base, base + 100):
        store.record(record)


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
