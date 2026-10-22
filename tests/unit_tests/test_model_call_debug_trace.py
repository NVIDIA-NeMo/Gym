# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from pathlib import Path
from time import time
from unittest.mock import MagicMock

import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    CaptureStore,
    SimpleResponsesAPIModel,
    read_model_call_records,
)
from nemo_gym.global_config import MODEL_CALL_DEBUG_TRACE_BLOCK, NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
from nemo_gym.model_call_debug_trace import (
    _CALL_TRACE,
    CallTrace,
    ModelCallDebugTrace,
    ModelCallDebugTraceSettings,
    _GlobalConfig,
    note_engine_response,
    request_engine_token_data,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture import NG_CAPTURE_FIELD, NG_DEBUG_TOKENS_FIELD


PROMPT_IDS = [101, 102, 103]
GENERATION_IDS = [7, 8]
GENERATION_LOG_PROBS = [-0.25, -1.5]
TOKENS = {
    "prompt_token_ids": PROMPT_IDS,
    "generation_token_ids": GENERATION_IDS,
    "generation_log_probs": GENERATION_LOG_PROBS,
}


def _response(text: str, *, with_tokens: bool) -> NeMoGymResponse:
    item = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    if with_tokens:
        item |= TOKENS
    return NeMoGymResponse(
        id="resp_1",
        created_at=int(time()),
        model="policy",
        object="response",
        output=[item],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


def _chat_completion(text: str, *, extra_message_fields: dict | None = None) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "policy",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text, **(extra_message_fields or {})},
            }
        ],
    }


class _TracedModel(SimpleResponsesAPIModel):
    """Serve fixed answers; ``engine_response`` stands in for a raw inference-engine reply."""

    config: BaseResponsesAPIModelConfig
    model_config = {"arbitrary_types_allowed": True}
    served_tokens: bool = True
    engine_response: dict | None = None
    fail: bool = False

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        if self.fail:
            raise RuntimeError("engine exploded")
        if self.engine_response is not None:
            note_engine_response(self.engine_response)
        return _response("the answer", with_tokens=self.served_tokens)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        return NeMoGymChatCompletion.model_validate(_chat_completion("chat answer", extra_message_fields=TOKENS))


def _server(
    tmp_path: Path, *, enabled: bool = True, observability: bool = False, **fields
) -> tuple[TestClient, _TracedModel]:
    global_config = {"model_call_debug_trace": {"enabled": enabled, "dir": str(tmp_path / "traces")}}
    if observability:
        global_config |= {"observability_enabled": True, "model_call_capture_dir": str(tmp_path / "calls")}
    server = _TracedModel(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="policy_model"),
        server_client=MagicMock(spec=ServerClient, global_config_dict=global_config),
        **fields,
    )
    return TestClient(server.setup_webserver(), raise_server_exceptions=False), server


def _records(server: _TracedModel, tmp_path: Path) -> list[dict]:
    server._model_call_debug_trace.flush()
    lines = []
    for path in sorted((tmp_path / "traces").glob("policy_model.*.jsonl")):
        lines.extend(json.loads(line) for line in path.read_text().splitlines())
    return lines


RESPONSES_BODY = {"input": [{"role": "user", "content": "question"}]}


class TestSettings:
    def test_block_is_a_reserved_global_config_key(self) -> None:
        # A non-reserved top-level block would be parsed as a server instance.
        assert MODEL_CALL_DEBUG_TRACE_BLOCK in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
        assert MODEL_CALL_DEBUG_TRACE_BLOCK in _GlobalConfig.model_fields

    def test_enabled_requires_absolute_dir(self) -> None:
        with pytest.raises(ValidationError, match="absolute"):
            ModelCallDebugTraceSettings(enabled=True)
        with pytest.raises(ValidationError, match="absolute"):
            ModelCallDebugTraceSettings(enabled=True, dir="relative/traces")

    def test_disabled_builds_no_tracer(self) -> None:
        assert ModelCallDebugTrace.from_global_config({}, server_name="srv") is None
        assert ModelCallDebugTrace.from_global_config(None, server_name="srv") is None

    def test_reads_an_omegaconf_global_config(self, tmp_path: Path) -> None:
        config = OmegaConf.create({"model_call_debug_trace": {"enabled": True, "dir": str(tmp_path)}, "other": 1})
        assert isinstance(ModelCallDebugTrace.from_global_config(config, server_name="srv"), ModelCallDebugTrace)


class TestModelServerRecords:
    def test_responses_call_records_text_and_served_tokens_once(self, tmp_path: Path) -> None:
        client, server = _server(tmp_path)
        response = client.post("/ng-rollout/3-1/v1/responses", json=RESPONSES_BODY)
        assert response.status_code == 200
        # The client still receives the token fields that the server serves.
        assert response.json()["output"][0]["generation_token_ids"] == GENERATION_IDS

        (record,) = _records(server, tmp_path)
        assert record["rollout_id"] == "3-1"
        assert record["model_server"] == "policy_model"
        assert record["path"] == "/v1/responses"
        assert record["request"]["input"] == RESPONSES_BODY["input"]
        assert record["response"]["output"][0]["content"][0]["text"] == "the answer"
        assert record["tokens"] == TOKENS
        assert record["tokens_source"] == "served_response"
        assert "generation_token_ids" not in record["response"]["output"][0]
        assert record["error"] is None

    def test_record_joins_the_model_call_capture_record(self, tmp_path: Path) -> None:
        client, server = _server(tmp_path, observability=True)
        for _ in range(2):
            assert client.post("/ng-rollout/5-0/v1/responses", json=RESPONSES_BODY).status_code == 200

        records = _records(server, tmp_path)
        captured = read_model_call_records(CaptureStore(tmp_path / "calls"), "5-0")
        assert len(records) == len(captured) == 2
        assert {record["model_call_id"] for record in records} == {call.model_call_id for call in captured}
        assert all(record["rollout_id"] == "5-0" for record in records)

    def test_anthropic_messages_call_keeps_tokens_the_conversion_drops(self, tmp_path: Path) -> None:
        client, server = _server(tmp_path)
        response = client.post(
            "/v1/messages",
            json={"model": "policy", "max_tokens": 16, "messages": [{"role": "user", "content": "question"}]},
        )
        assert response.status_code == 200
        assert "generation_token_ids" not in json.dumps(response.json())

        (record,) = _records(server, tmp_path)
        assert record["path"] == "/v1/messages"
        assert record["tokens"] == TOKENS

    def test_streaming_chat_call_is_recorded_once(self, tmp_path: Path) -> None:
        client, server = _server(tmp_path)
        response = client.post(
            "/v1/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": "question"}]}
        )
        assert response.status_code == 200
        assert "chat answer" in response.text

        (record,) = _records(server, tmp_path)
        assert record["path"] == "/v1/chat/completions"
        assert record["response"]["choices"][0]["message"]["content"] == "chat answer"
        assert record["tokens"] == TOKENS

    def test_worker_staged_echo_wins_over_the_served_response(self, tmp_path: Path) -> None:
        staged = {"prompt_token_ids": [9, 9], "generation_token_ids": [4], "generation_log_probs": [-0.1]}
        engine_response = _chat_completion("the answer") | {NG_DEBUG_TOKENS_FIELD: staged}
        client, server = _server(tmp_path, engine_response=engine_response, served_tokens=False)
        assert client.post("/v1/responses", json=RESPONSES_BODY).status_code == 200

        (record,) = _records(server, tmp_path)
        assert record["tokens"] == staged
        assert record["tokens_source"] == "engine_staged"
        assert NG_DEBUG_TOKENS_FIELD not in server.engine_response

    def test_failed_call_is_recorded_with_its_error(self, tmp_path: Path) -> None:
        client, server = _server(tmp_path, fail=True)
        assert client.post("/v1/responses", json=RESPONSES_BODY).status_code == 500

        (record,) = _records(server, tmp_path)
        assert record["error"] == {"type": "RuntimeError", "message": "engine exploded"}
        assert record["response"] is None
        assert record["tokens"] is None

    def test_disabled_trace_writes_nothing(self, tmp_path: Path) -> None:
        client, server = _server(tmp_path, enabled=False)
        assert client.post("/v1/responses", json=RESPONSES_BODY).status_code == 200
        assert server._model_call_debug_trace is None
        assert not (tmp_path / "traces").exists()


@pytest.fixture
def active_trace():
    trace = CallTrace(request_token_data=True)
    reset_token = _CALL_TRACE.set(trace)
    yield trace
    _CALL_TRACE.reset(reset_token)


class TestEngineRequest:
    def test_admitted_vllm_call_asks_the_worker_for_its_staged_arrays(self, active_trace: CallTrace) -> None:
        payload = {NG_CAPTURE_FIELD: {"rollout_id": "r"}, "logprobs": True, "top_logprobs": 0}
        request_engine_token_data(payload, megatron_engine=False)
        assert payload[NG_DEBUG_TOKENS_FIELD] is True
        assert "return_token_ids" not in payload

    def test_admitted_megatron_call_asks_through_offload_params(self, active_trace: CallTrace) -> None:
        payload = {"offload_params": {NG_CAPTURE_FIELD: {"rollout_id": "r"}}}
        request_engine_token_data(payload, megatron_engine=True)
        assert payload["offload_params"][NG_DEBUG_TOKENS_FIELD] is True
        assert NG_DEBUG_TOKENS_FIELD not in payload

    @pytest.mark.parametrize(
        ("megatron_engine", "token_ids_field"), [(False, "return_token_ids"), (True, "return_tokenized_data")]
    )
    def test_uncaptured_call_requests_engine_token_fields(
        self, active_trace: CallTrace, megatron_engine: bool, token_ids_field: str
    ) -> None:
        payload = {"messages": []}
        request_engine_token_data(payload, megatron_engine=megatron_engine)
        assert payload["logprobs"] is True
        assert payload["top_logprobs"] == 0
        assert payload["return_tokens_as_token_ids"] is True
        assert payload[token_ids_field] is True

    def test_client_requested_logprobs_are_left_alone(self, active_trace: CallTrace) -> None:
        payload = {"logprobs": True, "top_logprobs": 5}
        request_engine_token_data(payload, megatron_engine=False)
        assert payload["top_logprobs"] == 5
        assert "return_tokens_as_token_ids" not in payload
        assert active_trace.added_request_fields == {"return_token_ids"}

    def test_request_is_unchanged_without_an_opted_in_trace(self, active_trace: CallTrace) -> None:
        payload = {"n": 2}
        request_engine_token_data(payload, megatron_engine=False)
        assert payload == {"n": 2}
        active_trace.request_token_data = False
        payload = {"messages": []}
        request_engine_token_data(payload, megatron_engine=False)
        assert payload == {"messages": []}

    def test_untraced_call_is_unchanged(self) -> None:
        payload = {"messages": []}
        request_engine_token_data(payload, megatron_engine=False)
        note_engine_response(_chat_completion("x"))
        assert payload == {"messages": []}


class TestEngineResponse:
    def test_reads_and_removes_the_token_data_tracing_added(self, active_trace: CallTrace) -> None:
        request_engine_token_data({"messages": []}, megatron_engine=False)
        # NeMo RL's vLLM server returns message token fields alongside vLLM's own token-id fields.
        response = _chat_completion("hi", extra_message_fields=TOKENS) | {"prompt_token_ids": PROMPT_IDS}
        response["choices"][0] |= {"token_ids": GENERATION_IDS, "logprobs": {"content": []}}

        note_engine_response(response)

        assert active_trace.engine_tokens == TOKENS
        assert active_trace.engine_tokens_source == "engine_response"
        assert response == _chat_completion("hi")

    def test_parses_plain_vllm_token_id_logprobs(self, active_trace: CallTrace) -> None:
        response = _chat_completion("hi") | {"prompt_token_ids": PROMPT_IDS}
        response["choices"][0]["logprobs"] = {
            "content": [
                {"token": f"token_id:{token_id}", "logprob": log_prob}
                for token_id, log_prob in zip(GENERATION_IDS, GENERATION_LOG_PROBS)
            ]
        }
        note_engine_response(response)
        assert active_trace.engine_tokens == TOKENS
        # The client asked for these logprobs, so they stay on the response.
        assert response["choices"][0]["logprobs"]["content"]

    def test_unparseable_logprobs_are_reported_not_raised(self, active_trace: CallTrace) -> None:
        response = _chat_completion("hi")
        response["choices"][0]["logprobs"] = {"content": [{"token": "hello", "logprob": -0.1}]}
        note_engine_response(response)
        assert active_trace.engine_tokens is None
        assert active_trace.token_errors and active_trace.token_errors[0].startswith("engine_response: ValueError")

    def test_staged_echo_error_is_reported(self, active_trace: CallTrace) -> None:
        response = _chat_completion("hi") | {NG_DEBUG_TOKENS_FIELD: {"error": "capture failed"}}
        note_engine_response(response)
        assert active_trace.engine_tokens is None
        assert active_trace.token_errors == ["engine_staged: capture failed"]
        assert NG_DEBUG_TOKENS_FIELD not in response
