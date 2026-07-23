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
from time import time
from unittest.mock import MagicMock
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
    CaptureTokenSource,
    HttpTokenReader,
    LocalTokenReader,
    TokenCaptureStore,
    TokenEntry,
    TokenIdCaptureConfig,
    extract_token_fields,
    make_token_store,
)
from nemo_gym.token_id_capture import reader as reader_module


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


@pytest.mark.parametrize("bad", ["", "a/b", "../x", "a b"])
def test_token_store_rejects_unsafe_rollout_ids(tmp_path, bad):
    with pytest.raises(ValueError):
        TokenCaptureStore(tmp_path).path_for(bad)


# --- config -------------------------------------------------------------------


def test_config_disabled_needs_no_dir():
    cfg = TokenIdCaptureConfig.model_validate({})
    assert cfg.token_id_capture_enabled is False
    assert make_token_store({}) is None


def test_config_enabled_requires_absolute_dir(tmp_path):
    with pytest.raises(ValueError):
        TokenIdCaptureConfig(token_id_capture_enabled=True)
    with pytest.raises(ValueError):
        TokenIdCaptureConfig(token_id_capture_enabled=True, token_id_capture_dir="relative/dir")
    cfg = TokenIdCaptureConfig(token_id_capture_enabled=True, token_id_capture_dir=str(tmp_path))
    assert cfg.resolved_dir() == tmp_path


def test_config_falls_back_to_model_call_capture_dir(tmp_path):
    cfg = TokenIdCaptureConfig(token_id_capture_enabled=True, model_call_capture_dir=str(tmp_path))
    assert cfg.resolved_dir() == tmp_path


# --- source / readers ---------------------------------------------------------


def test_capture_token_source_over_local_reader(tmp_path):
    store = TokenCaptureStore(tmp_path)
    store.append(
        TokenEntry(
            rollout_id="r",
            model_call_id="c",
            prompt_token_ids=PTOKS,
            generation_token_ids=GTOKS,
            generation_log_probs=LPS,
        )
    )
    source = CaptureTokenSource(LocalTokenReader(store))
    entries = asyncio.run(source.tokens_for("r"))
    assert len(entries) == 1 and entries[0].generation_token_ids == GTOKS


def test_http_token_reader_parses_ndjson(monkeypatch):
    entry = TokenEntry(
        rollout_id="r", model_call_id="c", prompt_token_ids=PTOKS, generation_token_ids=GTOKS, generation_log_probs=LPS
    )
    body = entry.model_dump_json() + "\n"

    class _FakeResp:
        async def text(self):
            return body

    async def _fake_request(method, url, **kwargs):
        assert method == "GET" and url.endswith("/ng-capture/tokens/r")
        return _FakeResp()

    async def _fake_raise(_resp):
        return None

    monkeypatch.setattr(reader_module, "request", _fake_request)
    monkeypatch.setattr(reader_module, "raise_for_status", _fake_raise)
    entries = asyncio.run(HttpTokenReader("http://model:9000").read("r"))
    assert len(entries) == 1 and entries[0].model_call_id == "c"


# --- served path (full model server) -----------------------------------------


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
        "token_id_capture_enabled": True,
        "token_id_capture_dir": str(tmp_path),
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
    config = {"token_id_capture_enabled": True, "token_id_capture_dir": str(tmp_path)}
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


def test_http_route_returns_tokens_and_404_when_disabled(tmp_path):
    # Enabled: the route serves the captured entries as ndjson.
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    client.post("/ng-rollout/task2-roll0/v1/responses", json={"input": "hi"})
    got = client.get("/ng-capture/tokens/task2-roll0")
    assert got.status_code == 200
    lines = [line for line in got.text.splitlines() if line.strip()]
    assert len(lines) == 1
    parsed = TokenEntry.model_validate_json(lines[0])
    assert parsed.generation_token_ids == GTOKS

    # Disabled: the route is not registered.
    disabled = TestClient(_server({}).setup_webserver())
    assert disabled.get("/ng-capture/tokens/task2-roll0").status_code == 404
