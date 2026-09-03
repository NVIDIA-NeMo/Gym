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
"""Test the token-metadata envelope wire contract."""

import math
from time import time
from unittest.mock import MagicMock
from uuid import uuid4

import orjson
import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    SimpleResponsesAPIModel,
    _encode_token_metadata_in_place,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessageForTraining,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture import TokenCaptureStore
from nemo_gym.token_id_capture.records import extract_token_fields, strip_token_fields
from nemo_gym.token_metadata_codec import (
    TOKEN_ENVELOPE_PREFIX,
    decode_output_item_token_fields,
    decode_token_list,
    encode_output_item_token_fields,
    encode_token_list,
    is_token_envelope,
)


PTOKS = [101, 102, 103]
GTOKS = [7, 9]
LPS = [-0.125, -0.5]


class TestCodecRoundTrip:
    @pytest.mark.parametrize(
        ("values", "dtype", "expected"),
        [
            ([0, 1, -1, 2**31 - 1, -(2**31)], "i32", "ngtok1:i32:AAAAAAEAAAD/////////fwAAAIA="),
            ([0.0, -0.5, 1.5], "f64", "ngtok1:f64:AAAAAAAAAAAAAAAAAADgvwAAAAAAAPg/"),
        ],
    )
    def test_protocol_golden_vectors(self, values, dtype, expected) -> None:
        assert encode_token_list(values, dtype) == expected
        assert decode_token_list(expected) == values

    def test_i32_round_trip_is_bit_exact(self) -> None:
        values = [0, 1, 151_936, 2**31 - 1, -(2**31)]
        envelope = encode_token_list(values, "i32")
        assert envelope.startswith(f"{TOKEN_ENVELOPE_PREFIX}i32:")
        assert decode_token_list(envelope) == values

    def test_f64_round_trip_is_bit_exact(self) -> None:
        values = [0.0, -0.123456789012345, -math.pi, 1e-300, -1e300]
        envelope = encode_token_list(values, "f64")
        assert envelope.startswith(f"{TOKEN_ENVELOPE_PREFIX}f64:")
        assert decode_token_list(envelope) == values

    def test_empty_list_round_trips(self) -> None:
        for dtype in ("i32", "f64"):
            assert decode_token_list(encode_token_list([], dtype)) == []

    def test_decode_passes_plain_lists_through(self) -> None:
        values = [1, 2, 3]
        assert decode_token_list(values) is values

    def test_is_token_envelope(self) -> None:
        assert is_token_envelope(encode_token_list([1], "i32"))
        assert not is_token_envelope([1, 2, 3])
        assert not is_token_envelope(None)
        assert not is_token_envelope("nrlre1:int16:2x1x2:AAABAAIAAwA=")
        assert not is_token_envelope("plain text")


class TestCodecErrors:
    def test_encode_rejects_unknown_dtype(self) -> None:
        with pytest.raises(ValueError, match="unsupported token-metadata dtype"):
            encode_token_list([1], "i64")

    def test_encode_rejects_out_of_range_ints(self) -> None:
        with pytest.raises(ValueError, match="cannot encode token metadata as i32"):
            encode_token_list([2**40], "i32")

    def test_encode_rejects_non_numeric_values(self) -> None:
        with pytest.raises(ValueError, match="cannot encode token metadata as i32"):
            encode_token_list(["a"], "i32")

    def test_decode_rejects_non_list_non_envelope(self) -> None:
        for bad in (42, None, "not an envelope"):
            with pytest.raises(ValueError, match="expected a list or an"):
                decode_token_list(bad)

    def test_decode_rejects_unknown_dtype(self) -> None:
        with pytest.raises(ValueError, match="unsupported token-metadata dtype"):
            decode_token_list(f"{TOKEN_ENVELOPE_PREFIX}f32:AAAA")

    def test_decode_rejects_incomplete_header(self) -> None:
        with pytest.raises(ValueError, match="header must end"):
            decode_token_list(f"{TOKEN_ENVELOPE_PREFIX}i32")

    def test_decode_rejects_malformed_base64(self) -> None:
        with pytest.raises(ValueError, match="malformed base64"):
            decode_token_list(f"{TOKEN_ENVELOPE_PREFIX}i32:!!!not-base64!!!")

    def test_decode_rejects_misaligned_payload(self) -> None:
        # Two bytes cannot hold an int32.
        with pytest.raises(ValueError, match="not a multiple of"):
            decode_token_list(f"{TOKEN_ENVELOPE_PREFIX}i32:AAA=")


class TestItemHelpers:
    def _item(self) -> dict:
        return {
            "type": "message",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            "prompt_token_ids": list(PTOKS),
            "generation_token_ids": list(GTOKS),
            "generation_log_probs": list(LPS),
            "routed_experts": "nrlre1:int16:2x1x2:AAABAAIAAwA=",
        }

    def test_encode_then_decode_restores_lists_in_place(self) -> None:
        item = self._item()
        encode_output_item_token_fields(item)
        assert is_token_envelope(item["prompt_token_ids"])
        assert is_token_envelope(item["generation_token_ids"])
        assert item["generation_log_probs"].startswith(f"{TOKEN_ENVELOPE_PREFIX}f64:")
        # The routed_experts envelope and content are not this codec's to touch.
        assert item["routed_experts"] == "nrlre1:int16:2x1x2:AAABAAIAAwA="
        assert item["content"][0]["text"] == "hi"

        decode_output_item_token_fields(item)
        assert item["prompt_token_ids"] == PTOKS
        assert item["generation_token_ids"] == GTOKS
        assert item["generation_log_probs"] == LPS

    def test_encode_uses_f64_and_is_idempotent(self) -> None:
        item = self._item()
        encode_output_item_token_fields(item)
        assert item["generation_log_probs"].startswith(f"{TOKEN_ENVELOPE_PREFIX}f64:")
        encoded_once = dict(item)
        encode_output_item_token_fields(item)
        assert item == encoded_once

    def test_encode_skips_absent_fields(self) -> None:
        item = {"type": "message", "content": []}
        encode_output_item_token_fields(item)
        assert item == {"type": "message", "content": []}
        decode_output_item_token_fields(item)
        assert item == {"type": "message", "content": []}

    def test_decode_enforces_field_dtypes(self) -> None:
        item = self._item()
        item["prompt_token_ids"] = encode_token_list([1.0], "f64")
        with pytest.raises(ValueError, match="dtype must be"):
            decode_output_item_token_fields(item)

    def test_decode_rejects_malformed_field_envelope(self) -> None:
        item = self._item()
        item["generation_token_ids"] = "ngtok1:i32"
        with pytest.raises(ValueError, match="header must end"):
            decode_output_item_token_fields(item)


def _envelope_fields() -> dict:
    return {
        "prompt_token_ids": encode_token_list(PTOKS, "i32"),
        "generation_token_ids": encode_token_list(GTOKS, "i32"),
        "generation_log_probs": encode_token_list(LPS, "f64"),
    }


def _response_dict(token_fields: dict) -> dict:
    return {
        "id": "resp_1",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                **token_fields,
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "tools": [],
    }


class TestModelValidation:
    def test_response_output_item_with_envelope_strings_validates(self) -> None:
        response = NeMoGymResponse.model_validate(_response_dict(_envelope_fields()))
        [item] = response.output
        assert isinstance(item, NeMoGymResponseOutputMessageForTraining)
        assert decode_token_list(item.prompt_token_ids) == PTOKS
        assert decode_token_list(item.generation_token_ids) == GTOKS
        # The envelope strings survive serialization untouched.
        dumped = response.model_dump(mode="json")
        assert dumped["output"][0]["prompt_token_ids"] == item.prompt_token_ids

    def test_chat_message_with_envelope_strings_validates(self) -> None:
        completion = NeMoGymChatCompletion.model_validate(
            {
                "id": "chatcmpl_1",
                "created": 0,
                "model": "m",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi", **_envelope_fields()},
                    }
                ],
            }
        )
        message = completion.choices[0].message
        assert decode_token_list(message.generation_token_ids) == GTOKS

    def test_atomic_token_metadata_holds_for_envelope_form(self) -> None:
        # A replayed input item carrying only part of the metadata must still fail,
        # whether the present field is a list or an envelope.
        partial = {
            "role": "assistant",
            "content": "hi",
            "generation_token_ids": encode_token_list(GTOKS, "i32"),
        }
        with pytest.raises(ValidationError, match="must include all required fields"):
            NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": [partial]})
        complete = {"role": "assistant", "content": "hi", **_envelope_fields()}
        NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": [complete]})

    def test_response_rejects_partial_envelope_metadata(self) -> None:
        fields = _envelope_fields()
        del fields["prompt_token_ids"]

        with pytest.raises(ValidationError, match="must include all required fields"):
            NeMoGymResponse.model_validate(_response_dict(fields))

    def test_chat_params_accept_envelope_training_messages(self) -> None:
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
            {"messages": [{"role": "assistant", "content": "hi", **_envelope_fields()}]}
        )

    def test_mixed_list_and_envelope_fields_validate(self) -> None:
        fields = _envelope_fields()
        fields["prompt_token_ids"] = PTOKS
        response = NeMoGymResponse.model_validate(_response_dict(fields))
        [item] = response.output
        assert item.prompt_token_ids == PTOKS
        assert decode_token_list(item.generation_token_ids) == GTOKS

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("prompt_token_ids", "ngtok1:i32"),
            ("prompt_token_ids", "ngtok1:f32:AAAAAA=="),
            ("generation_token_ids", "ngtok1:f64:AAAAAAAAAAA="),
            ("generation_log_probs", "ngtok1:i32:AAAAAA=="),
        ],
    )
    def test_envelope_headers_and_field_dtypes_are_validated(self, field, value) -> None:
        fields = _envelope_fields()
        fields[field] = value
        with pytest.raises(ValidationError):
            NeMoGymResponse.model_validate(_response_dict(fields))

    def test_non_envelope_strings_are_rejected(self) -> None:
        # The string wire form is reserved for "ngtok1:" envelopes; an arbitrary
        # string stays malformed token metadata.
        from nemo_gym.openai_utils import TokenIDLogProbMixin

        with pytest.raises(ValidationError, match="ngtok1"):
            TokenIDLogProbMixin.model_validate(
                {
                    "prompt_token_ids": [1],
                    "generation_token_ids": "not-a-list",
                    "generation_log_probs": [-0.1],
                }
            )


class TestCaptureAcceptsBothForms:
    def test_extract_token_fields_decodes_envelopes(self) -> None:
        info = extract_token_fields(_response_dict(_envelope_fields()))
        assert info is not None
        assert info["prompt_token_ids"] == PTOKS
        assert info["generation_token_ids"] == GTOKS
        assert info["generation_log_probs"] == LPS

    def test_extract_token_fields_still_passes_lists_through(self) -> None:
        info = extract_token_fields(
            _response_dict({"prompt_token_ids": PTOKS, "generation_token_ids": GTOKS, "generation_log_probs": LPS})
        )
        assert info is not None
        assert info["generation_log_probs"] == LPS

    def test_strip_token_fields_finds_envelope_bearing_item(self) -> None:
        items = [{"type": "reasoning"}, _response_dict(_envelope_fields())["output"][0]]
        stripped, index = strip_token_fields(items)
        assert index == 1
        assert "generation_token_ids" not in stripped[1]

    def test_rollout_carries_token_ids_accepts_envelopes(self) -> None:
        from nemo_gym.token_id_capture.delivery import rollout_carries_token_ids

        assert rollout_carries_token_ids({"response": _response_dict(_envelope_fields())})
        empty = {"generation_token_ids": encode_token_list([], "i32")}
        assert not rollout_carries_token_ids({"response": {"output": [empty]}})
        malformed = {"generation_token_ids": "ngtok1:i32"}
        assert not rollout_carries_token_ids({"response": {"output": [malformed]}})
        assert rollout_carries_token_ids({"response": _response_dict({"generation_token_ids": GTOKS})})


class _EncodingModel(SimpleResponsesAPIModel):
    """A minimal model server that always serves one training-shaped response."""

    config: BaseResponsesAPIModelConfig
    model_config = {"arbitrary_types_allowed": True}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        return NeMoGymResponse.model_validate(
            _response_dict(
                {
                    "prompt_token_ids": list(PTOKS),
                    "generation_token_ids": list(GTOKS),
                    "generation_log_probs": list(LPS),
                }
            )
            | {"id": f"resp_{uuid4().hex}", "created_at": int(time())}
        )

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        return NeMoGymChatCompletion.model_validate(
            {
                "id": f"chatcmpl_{uuid4().hex}",
                "created": int(time()),
                "model": "m",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "hi",
                            "prompt_token_ids": list(PTOKS),
                            "generation_token_ids": list(GTOKS),
                            "generation_log_probs": list(LPS),
                        },
                    }
                ],
            }
        )


def _client(tmp_path, encoding: str) -> TestClient:
    server = _EncodingModel(
        config=BaseResponsesAPIModelConfig(
            host="0.0.0.0",
            port=8099,
            entrypoint="",
            name="srv",
            token_metadata_encoding=encoding,
        ),
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={"token_id_capture": {"enabled": True, "dir": str(tmp_path)}},
        ),
    )
    return TestClient(server.setup_webserver())


class TestEndpointEncoding:
    def test_dictionary_response_is_encoded_in_place(self) -> None:
        response = _response_dict(
            {"prompt_token_ids": PTOKS, "generation_token_ids": GTOKS, "generation_log_probs": LPS}
        )
        _encode_token_metadata_in_place(response, "base64")
        [item] = response["output"]
        assert decode_token_list(item["prompt_token_ids"], ("i32",)) == PTOKS
        assert decode_token_list(item["generation_log_probs"], ("f64",)) == LPS

    def test_responses_route_serves_envelopes_while_capture_retains_lists(self, tmp_path) -> None:
        client = _client(tmp_path, "base64")
        resp = client.post("/ng-rollout/enc-a/training-token-capture/v1/responses", json={"input": "hi"})
        assert resp.status_code == 200
        [item] = resp.json()["output"]
        assert item["prompt_token_ids"] == encode_token_list(PTOKS, "i32")
        assert item["generation_token_ids"] == encode_token_list(GTOKS, "i32")
        assert item["generation_log_probs"] == encode_token_list(LPS, "f64")
        assert decode_token_list(item["generation_log_probs"]) == LPS
        assert item["content"][0]["text"] == "hi"

        # Capture ran before encoding, so the recorded entry holds the raw lists.
        [entry] = TokenCaptureStore(tmp_path).read_entries("enc-a")
        assert entry.prompt_token_ids == PTOKS
        assert entry.generation_token_ids == GTOKS
        assert entry.generation_log_probs == LPS

    def test_chat_route_serves_envelopes(self, tmp_path) -> None:
        client = _client(tmp_path, "base64")
        resp = client.post(
            "/ng-rollout/enc-c/training-token-capture/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        message = resp.json()["choices"][0]["message"]
        assert message["prompt_token_ids"] == encode_token_list(PTOKS, "i32")
        assert decode_token_list(message["generation_token_ids"]) == GTOKS
        [entry] = TokenCaptureStore(tmp_path).read_entries("enc-c")
        assert entry.generation_token_ids == GTOKS

    def test_default_json_encoding_serves_plain_lists(self, tmp_path) -> None:
        expected = _response_dict(
            {"prompt_token_ids": PTOKS, "generation_token_ids": GTOKS, "generation_log_probs": LPS}
        )
        expected_bytes = orjson.dumps(expected)
        _encode_token_metadata_in_place(expected, "json")
        assert orjson.dumps(expected) == expected_bytes

        client = _client(tmp_path, "json")
        resp = client.post("/ng-rollout/enc-d/training-token-capture/v1/responses", json={"input": "hi"})
        assert resp.status_code == 200
        [item] = resp.json()["output"]
        assert item["prompt_token_ids"] == PTOKS
        assert item["generation_token_ids"] == GTOKS
        assert item["generation_log_probs"] == LPS

    @pytest.mark.parametrize("encoding", ["base64_f32", "base64_f64"])
    def test_precision_specific_modes_are_rejected(self, encoding) -> None:
        with pytest.raises(ValidationError):
            BaseResponsesAPIModelConfig(
                host="",
                port=0,
                entrypoint="",
                name="",
                token_metadata_encoding=encoding,
            )

    def test_encoding_defaults_to_json(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, entrypoint="", name="")
        assert config.token_metadata_encoding == "json"
