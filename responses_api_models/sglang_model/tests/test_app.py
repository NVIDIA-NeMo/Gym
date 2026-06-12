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
"""Orchestration tests for SGLangModel.chat_completions.

These exercise the full request->generate->response path with the SGLang HTTP call
(`ng_request`) and the HF tokenizer mocked out, so no live SGLang server / GPU / model
weights are required. They lock in the integration behaviors that the end-to-end smoke
runs surfaced:
  - the exact `/generate` payload (input_ids + return_logprob),
  - parsing token-ids + logprobs back out of the SGLang response,
  - attaching the training token fields only when return_token_id_information is set,
  - decoding graded `content` with skip_special_tokens=True (so trailing special tokens
    don't break strict parsers like structured_outputs json.loads),
  - the error path setting `response_content` on the raised exception (so the nemo_gym
    exception middleware doesn't itself assert).

Importing this needs nemo_gym + transformers (the per-server venv); it is skipped cleanly
otherwise. Pure-logic coverage lives in test_logic.py.
"""

import asyncio
import os
import sys
from types import SimpleNamespace


try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from responses_api_models.sglang_model import app as sglang_app

    _IMPORT_ERR = None
except Exception as e:  # nemo_gym / transformers / vllm_model not importable here
    _IMPORT_ERR = e

try:
    import pytest

    skip_if_no_framework = pytest.mark.skipif(
        _IMPORT_ERR is not None, reason=f"sglang_model app not importable: {_IMPORT_ERR}"
    )
except ImportError:  # allow standalone `python tests/test_app.py`
    pytest = None

    def skip_if_no_framework(fn):
        return fn


# ----------------------------- fakes -----------------------------
class _FakeTokenizer:
    """Records the prompt token-ids it returns and the decode kwargs it is called with."""

    def __init__(self, prompt_ids, decoded="the answer"):
        self._prompt_ids = prompt_ids
        self._decoded = decoded
        self.decode_calls = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict, **kw):
        assert tokenize is True and return_dict is False
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


def _make_self(prompt_ids=(1, 2, 3, 4, 5), return_token_ids=True, decoded="the answer"):
    """A minimal stand-in for an initialized SGLangModel (avoids loading real weights)."""
    cfg = SimpleNamespace(
        model="dummy-model",
        chat_template_kwargs=None,
        add_generation_prompt=True,
        context_length=4096,
        default_max_new_tokens=1024,
        return_token_id_information=return_token_ids,
    )
    me = SimpleNamespace(
        config=cfg,
        _tokenizer=_FakeTokenizer(prompt_ids, decoded=decoded),
        _sglang_urls=["http://sglang-host:30000"],
    )
    # _preprocess_chat_completion_create_params is inherited from VLLMModel; stub as passthrough.
    me._preprocess_chat_completion_create_params = lambda request, body_dict: body_dict
    return me


class _FakeBody:
    def __init__(self, **fields):
        self._fields = {"messages": [{"role": "user", "content": "hi"}], **fields}

    def model_dump(self, exclude_unset=True):
        return dict(self._fields)


def _patch_http(monkeypatch_like, *, result=None, resp=None):
    """Patch the module-level ng_request / get_response_json; return a call-recorder."""
    rec = {"payload": None, "url": None}

    async def fake_ng_request(method, url, json=None, **kw):
        rec["payload"] = json
        rec["url"] = url
        return resp if resp is not None else _FakeResp(ok=True)

    async def fake_get_response_json(_resp):
        return result

    sglang_app.ng_request = fake_ng_request
    sglang_app.get_response_json = fake_get_response_json
    return rec


def _run(coro):
    return asyncio.run(coro)


# ----------------------------- tests -----------------------------
@skip_if_no_framework
def test_chat_completions_happy_path_attaches_training_fields():
    me = _make_self(prompt_ids=(1, 2, 3, 4, 5), return_token_ids=True, decoded="the answer")
    result_json = {
        "meta_info": {
            "output_token_logprobs": [[-0.1, 10, "a"], [-0.2, 11, "b"]],
            "finish_reason": {"type": "stop"},
        }
    }
    rec = _patch_http(None, result=result_json, resp=_FakeResp(ok=True))
    body = _FakeBody(temperature=0.7, max_tokens=16)
    request = SimpleNamespace()  # no .session -> sid defaults to ""

    out = _run(sglang_app.SGLangModel.chat_completions(me, request, body))

    # the /generate payload uses input_ids + asks for logprobs
    assert rec["url"].endswith("/generate")
    assert rec["payload"]["input_ids"] == [1, 2, 3, 4, 5]
    assert rec["payload"]["return_logprob"] is True
    assert rec["payload"]["logprob_start_len"] == -1
    assert rec["payload"]["sampling_params"]["max_new_tokens"] == 16

    choice = out.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content == "the answer"
    # raw token ids + logprobs are attached for training
    assert list(choice.message.generation_token_ids) == [10, 11]
    assert list(choice.message.generation_log_probs) == [-0.1, -0.2]
    assert list(choice.message.prompt_token_ids) == [1, 2, 3, 4, 5]


@skip_if_no_framework
def test_content_decoded_with_skip_special_tokens():
    # regression: structured_outputs broke because content ended in a literal special token.
    me = _make_self(decoded="{}")
    _patch_http(None, result={"meta_info": {"output_token_logprobs": [[-0.1, 10, "x"]]}})
    _run(sglang_app.SGLangModel.chat_completions(me, SimpleNamespace(), _FakeBody()))
    assert me._tokenizer.decode_calls, "decode was not called"
    assert me._tokenizer.decode_calls[0]["skip_special_tokens"] is True


@skip_if_no_framework
def test_no_training_fields_when_flag_off():
    me = _make_self(return_token_ids=False)
    _patch_http(None, result={"meta_info": {"output_token_logprobs": [[-0.1, 10, "x"]]}})
    out = _run(sglang_app.SGLangModel.chat_completions(me, SimpleNamespace(), _FakeBody()))
    msg = out.choices[0].message
    assert getattr(msg, "generation_token_ids", None) is None
    assert getattr(msg, "prompt_token_ids", None) is None


@skip_if_no_framework
def test_length_finish_reason_mapped():
    me = _make_self()
    _patch_http(
        None,
        result={"meta_info": {"output_token_logprobs": [[-0.1, 10, "x"]], "finish_reason": {"type": "length"}}},
    )
    out = _run(sglang_app.SGLangModel.chat_completions(me, SimpleNamespace(), _FakeBody()))
    assert out.choices[0].finish_reason == "length"


@skip_if_no_framework
def test_error_response_sets_response_content_and_raises():
    # regression: a raw raise_for_status without response_content trips nemo_gym's middleware.
    me = _make_self()
    bad = _FakeResp(ok=False, status=400, body=b"input_ids out of range")
    _patch_http(None, result=None, resp=bad)
    raised = None
    try:
        _run(sglang_app.SGLangModel.chat_completions(me, SimpleNamespace(), _FakeBody()))
    except Exception as e:  # noqa: BLE001 - we assert on the captured exception below
        raised = e
    assert raised is not None, "expected an exception on a non-ok SGLang response"
    assert getattr(raised, "response_content", None) == b"input_ids out of range"


if __name__ == "__main__":
    if _IMPORT_ERR is not None:
        print(f"SKIP test_app.py: {_IMPORT_ERR}")
        sys.exit(0)
    npass = nfail = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                npass += 1
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                nfail += 1
                print(f"FAIL {name}: {e!r}")
    print(f"\n{npass} passed, {nfail} failed")
    sys.exit(1 if nfail else 0)
