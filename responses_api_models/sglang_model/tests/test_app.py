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
"""Orchestration tests for SGLangModel, for both transports.

Servers are built the same way `vllm_model/tests/test_app.py` builds them -- a real
`SGLangModel` with a mocked `ServerClient` -- so the *inherited* `VLLMModel` behavior is
genuinely exercised rather than stubbed. In particular the real
`_preprocess_chat_completion_create_params` runs, which is what locks in the contract between
it and this subclass.

All patching goes through pytest's `monkeypatch` fixture so module globals are restored
afterwards and cannot leak into later tests in the same process.

Pure-logic coverage lives in test_logic.py.
"""

from typing import Any, Dict
from unittest.mock import MagicMock

from pytest import MonkeyPatch, raises

import nemo_gym.server_utils
from nemo_gym.server_utils import ServerClient
from responses_api_models.sglang_model.app import SGLangModel, SGLangModelConfig


def _make_server(monkeypatch: MonkeyPatch, **overrides: Any) -> SGLangModel:
    config = SGLangModelConfig(
        host="0.0.0.0",
        port=8081,
        base_url=overrides.pop("base_url", "http://sglang-host:30000/v1"),
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy_model",
        entrypoint="",
        name="sglang_model",
        return_token_id_information=True,
        uses_reasoning_parser=False,
        **overrides,
    )
    get_global_config_dict_mock = MagicMock()
    get_global_config_dict_mock.return_value = dict()
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)
    return SGLangModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


def _choice(**overrides: Any) -> Dict[str, Any]:
    """An SGLang >= 0.5.13 chat choice with the native TITO extensions populated."""
    choice: Dict[str, Any] = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "the answer"},
        "prompt_token_ids": [1, 2, 3],
        "meta_info": {"output_token_logprobs": [[-0.5, 10, "a"], [-0.25, 11, "b"]]},
        "logprobs": {"content": [{"token": "a", "logprob": -0.5}]},
    }
    choice.update(overrides)
    return choice


# --------------------------- chat transport ---------------------------


class TestChatTransport:
    def test_preprocess_requests_sglang_tito_extensions(self, monkeypatch: MonkeyPatch) -> None:
        """The real inherited preprocess runs, then we swap vLLM's knob for SGLang's."""
        server = _make_server(monkeypatch)
        body: Dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}

        out = server._preprocess_chat_completion_create_params(MagicMock(), body)

        assert out["return_meta_info"] is True
        assert out["return_prompt_token_ids"] is True
        # vLLM's `token_id:NNN` encoding does not exist on SGLang; leaving it set would be a
        # silent no-op that misrepresents where the ids come from.
        assert "return_tokens_as_token_ids" not in out
        # ...and the inherited behavior is still in force.
        assert out["logprobs"] is True
        assert out["model"] == "dummy_model"

    def test_preprocess_honors_per_request_chat_template_kwargs(self, monkeypatch: MonkeyPatch) -> None:
        """Per-sample overrides must survive; dropping them tokenizes with the wrong template."""
        server = _make_server(monkeypatch, chat_template_kwargs={"enable_thinking": False})
        body: Dict[str, Any] = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"chat_template_kwargs": '{"enable_thinking": true}'},
        }

        out = server._preprocess_chat_completion_create_params(MagicMock(), body)

        assert out["chat_template_kwargs"] == {"enable_thinking": True}

    async def test_attach_reads_native_ids_and_logprobs(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server(monkeypatch)
        choice = _choice()

        await server._attach_token_id_information(choice, {}, MagicMock())

        assert choice["message"]["prompt_token_ids"] == [1, 2, 3]
        assert choice["message"]["generation_token_ids"] == [10, 11]
        assert choice["message"]["generation_log_probs"] == [-0.5, -0.25]
        # Non-OpenAI / duplicated fields are stripped so the response validates.
        for key in ("logprobs", "prompt_token_ids", "meta_info"):
            assert key not in choice

    async def test_attach_rejects_aborted_generation(self, monkeypatch: MonkeyPatch) -> None:
        """An abort is a truncated fragment; it must not enter a training batch as a `stop`."""
        server = _make_server(monkeypatch)

        with raises(RuntimeError, match="abort"):
            await server._attach_token_id_information(_choice(finish_reason="abort"), {}, MagicMock())

    async def test_attach_errors_when_server_predates_chat_tito(self, monkeypatch: MonkeyPatch) -> None:
        """Older SGLang ignores the extensions; say so instead of emitting empty token ids."""
        server = _make_server(monkeypatch)
        choice = _choice()
        choice.pop("prompt_token_ids")

        with raises(RuntimeError, match="0.5.13"):
            await server._attach_token_id_information(choice, {}, MagicMock())


# ------------------------- generate transport -------------------------


class _FakeTokenizer:
    def __init__(self, prompt_ids=(1, 2, 3, 4, 5), decoded="the answer"):
        self._prompt_ids = list(prompt_ids)
        self._decoded = decoded
        self.decode_calls: list = []
        self.template_calls: list = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict, **kw):
        assert tokenize is True and return_dict is False
        self.template_calls.append(kw)
        return list(self._prompt_ids)

    def decode(self, token_ids, skip_special_tokens=False):
        self.decode_calls.append({"token_ids": list(token_ids), "skip_special_tokens": skip_special_tokens})
        return self._decoded


class _FakeResp:
    def __init__(self, ok=True, status=200, body=b""):
        self.ok = ok
        self.status = status
        self._body = body

    async def read(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status}")


def _make_generate_server(monkeypatch: MonkeyPatch, tokenizer=None, **overrides: Any) -> SGLangModel:
    server = _make_server(
        monkeypatch, transport="generate", base_url="http://sglang-host:30000", **overrides
    )
    server._tokenizer = tokenizer or _FakeTokenizer()
    server._sglang_urls = ["http://sglang-host:30000"]
    return server


def _patch_http(monkeypatch: MonkeyPatch, *, result=None, resp=None) -> Dict[str, Any]:
    """Patch the module globals via monkeypatch so they are restored after the test."""
    import responses_api_models.sglang_model.app as sglang_app

    rec: Dict[str, Any] = {}

    async def fake_ng_request(method, url, json=None, **kw):
        rec.update(payload=json, url=url, headers=kw.get("headers"))
        return resp if resp is not None else _FakeResp(ok=True)

    async def fake_get_response_json(_resp):
        return result

    monkeypatch.setattr(sglang_app, "ng_request", fake_ng_request)
    monkeypatch.setattr(sglang_app, "get_response_json", fake_get_response_json)
    return rec


class _Body:
    def __init__(self, **fields: Any):
        self._fields = {"messages": [{"role": "user", "content": "hi"}], **fields}

    def model_dump(self, exclude_unset=True):
        return dict(self._fields)


_GENERATE_RESULT = {
    "meta_info": {
        "finish_reason": {"type": "stop"},
        "output_token_logprobs": [[-0.5, 10, "a"], [-0.25, 11, "b"]],
    }
}


class TestGenerateTransport:
    async def test_posts_input_ids_with_auth_header(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_generate_server(monkeypatch)
        rec = _patch_http(monkeypatch, result=_GENERATE_RESULT)

        res = await server.chat_completions(MagicMock(spec=[]), _Body())

        assert rec["url"] == "http://sglang-host:30000/generate"
        assert rec["payload"]["input_ids"] == [1, 2, 3, 4, 5]
        assert rec["payload"]["return_logprob"] is True
        # SGLang's --api-key middleware guards /generate too.
        assert rec["headers"] == {"Authorization": "Bearer dummy_key"}

        message = res.choices[0].message
        assert message.generation_token_ids == [10, 11]
        assert message.generation_log_probs == [-0.5, -0.25]

    async def test_graded_content_drops_special_tokens(self, monkeypatch: MonkeyPatch) -> None:
        tokenizer = _FakeTokenizer()
        server = _make_generate_server(monkeypatch, tokenizer=tokenizer)
        _patch_http(monkeypatch, result=_GENERATE_RESULT)

        await server.chat_completions(MagicMock(spec=[]), _Body())

        # Raw ids are kept for training, but the graded text must not carry a trailing
        # special token (it breaks strict parsers like structured_outputs json.loads).
        assert tokenizer.decode_calls[0]["skip_special_tokens"] is True
        assert tokenizer.decode_calls[0]["token_ids"] == [10, 11]

    async def test_overflowing_prompt_is_filterable_not_head_truncated(self, monkeypatch: MonkeyPatch) -> None:
        """Head-truncating would drop the newest turn and the generation cue silently."""
        server = _make_generate_server(monkeypatch, tokenizer=_FakeTokenizer(prompt_ids=range(50)), context_length=8)
        rec = _patch_http(monkeypatch, result=_GENERATE_RESULT)

        res = await server.chat_completions(MagicMock(spec=[]), _Body())

        assert res.choices[0].finish_reason == "length"
        assert not res.choices[0].message.content
        assert rec == {}, "no /generate call should be made for an overflowing prompt"

    async def test_tools_are_rendered_into_the_prompt(self, monkeypatch: MonkeyPatch) -> None:
        tokenizer = _FakeTokenizer()
        server = _make_generate_server(monkeypatch, tokenizer=tokenizer)
        _patch_http(monkeypatch, result=_GENERATE_RESULT)
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

        await server.chat_completions(MagicMock(spec=[]), _Body(tools=tools))

        assert tokenizer.template_calls[0]["tools"] == tools

    async def test_abort_is_rejected(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_generate_server(monkeypatch)
        _patch_http(monkeypatch, result={"meta_info": {"finish_reason": {"type": "abort"}}})

        with raises(RuntimeError, match="abort"):
            await server.chat_completions(MagicMock(spec=[]), _Body())

    async def test_http_error_carries_response_content(self, monkeypatch: MonkeyPatch) -> None:
        """nemo_gym's exception middleware asserts on `response_content` being present."""
        server = _make_generate_server(monkeypatch)
        _patch_http(monkeypatch, resp=_FakeResp(ok=False, status=400, body=b"boom"))

        with raises(Exception) as excinfo:
            await server.chat_completions(MagicMock(spec=[]), _Body())

        assert getattr(excinfo.value, "response_content", None) == b"boom"
