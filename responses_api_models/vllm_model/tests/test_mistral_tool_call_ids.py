# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from unittest.mock import MagicMock

from fastapi import Request

from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


MISTRAL_ID = re.compile(r"^[A-Za-z0-9]{9}$")


def make_model(**overrides):
    return VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="test",
            base_url="http://unused/v1",
            api_key="dummy",  # pragma: allowlist secret
            model="test",
            return_token_id_information=False,
            uses_reasoning_parser=True,
            **overrides,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def call(call_id, name="f"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def history(*call_ids):
    """One assistant turn calling every id, then one tool result per id."""
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [call(i) for i in call_ids]},
        *({"role": "tool", "tool_call_id": i, "content": "ok"} for i in call_ids),
    ]


def preprocess(model, messages):
    body = {"model": "test", "messages": messages}
    return model._preprocess_chat_completion_create_params(Request({"type": "http", "headers": []}), body)


def ids_of(body):
    calls = [c["id"] for m in body["messages"] for c in m.get("tool_calls") or []]
    results = [m["tool_call_id"] for m in body["messages"] if m.get("role") == "tool"]
    return calls, results


def test_tool_call_ids_are_unchanged_by_default():
    body = preprocess(make_model(), history("call_0_conv_0001_2", "call_0_conv_0001_3"))
    assert ids_of(body) == (["call_0_conv_0001_2", "call_0_conv_0001_3"], ["call_0_conv_0001_2", "call_0_conv_0001_3"])


def test_ids_become_distinct_nine_character_alphanumerics_when_enabled():
    # Mistral's tokenizer accepts only 9-character [A-Za-z0-9] ids; vLLM keeps the last 9
    # characters of a longer id, so these two collide as "nv_0001_2" style duplicates.
    original = ["call_0_conv_0001_2", "call_1_conv_0001_2", "call_0_conv_0001_3"]
    calls, results = ids_of(preprocess(make_model(mistral_tool_call_ids=True), history(*original)))
    assert all(MISTRAL_ID.match(i) for i in calls)
    assert len(set(calls)) == len(original)
    # Each tool result still answers its own call.
    assert results == calls


def test_valid_mistral_ids_are_kept():
    calls, results = ids_of(preprocess(make_model(mistral_tool_call_ids=True), history("aB3dE6gH9", "call_x_1")))
    assert calls[0] == "aB3dE6gH9" and results[0] == "aB3dE6gH9"
    assert MISTRAL_ID.match(calls[1]) and calls[1] != "aB3dE6gH9"


def test_mapping_is_deterministic_across_requests():
    # Multi-turn rollouts resend the same history; the same ids must map the same way.
    ids = ["call_0_conv_0001_2", "call_0_conv_0001_3"]
    model = make_model(mistral_tool_call_ids=True)
    assert ids_of(preprocess(model, history(*ids))) == ids_of(preprocess(model, history(*ids)))


def test_a_generated_id_never_takes_an_id_already_in_the_request():
    model = make_model(mistral_tool_call_ids=True)
    first, _ = ids_of(preprocess(model, history("call_long_id_1")))
    calls, _ = ids_of(preprocess(model, history(first[0], "call_long_id_1")))
    assert calls[0] == first[0]
    assert calls[1] != first[0] and MISTRAL_ID.match(calls[1])
