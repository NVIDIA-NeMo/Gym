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

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from pydantic import ValidationError
from starlette.requests import Request

import nemo_gym.client.http_client as http_client
from nemo_gym.client import ServerClient, get_global_aiohttp_client
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming, NeMoGymResponseCreateParamsNonStreaming
from responses_api_models.external_model.app import (
    ExternalModel,
    ExternalModelClient,
    ExternalModelConfig,
    served_cache_salt,
)
from responses_api_models.external_model.tests.test_conversion import TOOLS, ChatML


@pytest.fixture
async def endpoint(monkeypatch):
    # Each test owns its event loop and pool; restore any prior process client afterward.
    monkeypatch.setattr(http_client, "_GLOBAL_AIOHTTP_CLIENT", None)
    monkeypatch.setattr(http_client, "_GLOBAL_AIOHTTP_CLIENT_LOOP", None)
    received = []
    state = {"text": "answer", "finish": "stop", "on_request": None}

    async def generate(request):
        body = await request.json()
        received.append((body, request.headers.get("Authorization")))
        if state["on_request"]:
            state["on_request"]()
        ids = ChatML().encode(state["text"]) + [1]
        return web.json_response(
            {
                "id": "completion-test",
                "created": 1,
                "model": "test-model",
                "usage": {
                    "prompt_tokens": len(body["prompt"]),
                    "completion_tokens": len(ids),
                    "total_tokens": len(body["prompt"]) + len(ids),
                },
                "choices": [
                    {
                        "text": state["text"],
                        "finish_reason": state["finish"],
                        "logprobs": {"tokens": [f"token_id:{i}" for i in ids], "token_logprobs": [-0.25] * len(ids)},
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_post("/v1/completions", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    try:
        yield url, received, state
    finally:
        await get_global_aiohttp_client().close()
        await runner.cleanup()


def config(tmp_path: Path, url: str, **overrides) -> ExternalModelConfig:
    return ExternalModelConfig(
        **{
            "host": "127.0.0.1",
            "port": 8080,
            "name": "policy_model",
            "entrypoint": "app.py",
            "base_url": url,
            "model": "test-model",
            "tokenizer_path": str(tmp_path),
            "max_model_len": 4096,
            "cache_salt_file": str(tmp_path / "served_identity"),
            **overrides,
        }
    )


def model(tmp_path: Path, url: str, monkeypatch) -> ExternalModel:
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: ChatML())
    return ExternalModel(
        config=config(tmp_path, url), server_client=MagicMock(spec=ServerClient, global_config_dict={})
    )


def request() -> Request:
    return Request({"type": "http", "session": {"session_id": "test-session"}, "headers": []})


async def test_two_turn_responses_preserve_exact_tool_tokens(tmp_path, endpoint, monkeypatch):
    url, received, state = endpoint
    (tmp_path / "served_identity").write_text("run-v1")
    server = model(tmp_path, url, monkeypatch)
    tool_text = '<tool_call><function=search><parameter=queries>["x"]</parameter></function></tool_call>'
    state["text"] = tool_text
    first_input = [{"role": "user", "content": "find x"}]
    tools = [{"type": "function", "strict": False, **TOOLS[0]["function"]}]
    first = await server.responses(
        request(),
        NeMoGymResponseCreateParamsNonStreaming(
            input=first_input,
            tools=tools,
            max_output_tokens=200,
        ),
    )
    output = [item.model_dump(exclude_none=True) for item in first.output]
    call = next(item for item in output if item["type"] == "function_call")
    assert call["name"] == "search"
    exact = ChatML().encode(tool_text) + [1]
    assert call["generation_token_ids"] == exact
    assert call["generation_log_probs"] == [-0.25] * len(exact)
    state["text"] = "found it"
    second = await server.responses(
        request(),
        NeMoGymResponseCreateParamsNonStreaming(
            input=first_input
            + output
            + [{"type": "function_call_output", "call_id": call["call_id"], "output": "result x"}],
            tools=tools,
            max_output_tokens=200,
        ),
    )
    prefix = received[0][0]["prompt"] + exact
    assert received[1][0]["prompt"][: len(prefix)] == prefix
    last = second.output[-1]
    assert last.prompt_token_ids == received[1][0]["prompt"]
    assert last.generation_token_ids == ChatML().encode("found it") + [1]
    assert all(body["nvext"] == {"cache_salt": "run-v1"} for body, _ in received)
    assert last.generation_log_probs == [-0.25] * len(last.generation_token_ids)


async def test_auth_and_checkpoint_are_resolved_each_request(tmp_path, endpoint):
    url, received, _ = endpoint
    identity, key_file = tmp_path / "identity", tmp_path / "key"
    identity.write_text("v1")
    key_file.write_text("test-only-key-one")
    client = ExternalModelClient(
        base_url=url,
        api_key="unused",
        tokenizer=ChatML(),
        max_model_len=1000,
        cache_salt_file=str(identity),
        api_key_file=str(key_file),
    )
    params = {"model": "test-model", "messages": [{"role": "user", "content": "q"}], "max_tokens": 10}
    await client.create_chat_completion(**params)
    identity.write_text("v2")
    key_file.write_text("test-only-key-two")
    await client.create_chat_completion(**params)
    assert [body["nvext"]["cache_salt"] for body, _ in received] == ["v1", "v2"]
    assert [auth for _, auth in received] == ["Bearer test-only-key-one", "Bearer test-only-key-two"]
    assert client.api_key == "unused"
    key_file.write_text("")
    with pytest.raises(RuntimeError, match="nonempty inference key"):
        await client.create_chat_completion(**params)
    identity.unlink()
    with pytest.raises(RuntimeError, match="sampling must remain paused"):
        await client.create_chat_completion(**params)
    assert len(received) == 2


async def test_checkpoint_change_discards_inflight_response(tmp_path, endpoint):
    url, received, state = endpoint
    identity = tmp_path / "identity"
    identity.write_text("v1")
    state["on_request"] = lambda: identity.write_text("v2")
    client = ExternalModelClient(
        base_url=url, api_key="unused", tokenizer=ChatML(), max_model_len=1000, cache_salt_file=str(identity)
    )
    with pytest.raises(RuntimeError, match="changed during generation"):
        await client.create_chat_completion(model="m", messages=[{"role": "user", "content": "q"}])
    assert len(received) == 1


async def test_full_context_returns_length_without_http(tmp_path, endpoint, monkeypatch):
    url, received, _ = endpoint
    (tmp_path / "served_identity").write_text("v1")
    server = model(tmp_path, url, monkeypatch)
    server.config.max_model_len = 1
    # The config is copied into each client at initialization.
    server._clients[0].max_model_len = 1
    result = await server.chat_completions(
        request(),
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[{"role": "user", "content": "hello"}],
        ),
    )
    assert result.choices[0].finish_reason == "length"
    assert received == []


@pytest.mark.parametrize("value", ["run-v3", "a/b", "x" * 129, "a@b", "a\\b", "a\0b"])
def test_cache_salt_matches_cwtrain(tmp_path, value):
    path = tmp_path / "identity"
    path.write_text(value)
    expected = value if value == "run-v3" else hashlib.sha256(value.encode()).hexdigest()
    assert served_cache_salt(str(path)) == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("api_key", "inline-key"),
        ("max_model_len", 0),
        ("supply_prefix_token_ids", True),
        ("is_responses_native", True),
        ("use_completions_api", True),
        ("extra_body", {"nvext": {"cache_salt": "unsafe"}}),
        ("default_headers", {"Authorization": "inline-key"}),
    ],
)
def test_rejects_unsupported_config_before_startup(tmp_path, field, value):
    with pytest.raises(ValidationError):
        config(tmp_path, "http://localhost/v1", **{field: value})
