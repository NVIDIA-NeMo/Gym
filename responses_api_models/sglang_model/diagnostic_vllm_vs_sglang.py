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
"""Diagnostic: *why* a dedicated sglang_model server is needed (vs. reusing vllm_model).

This is the "run SGLang through the existing vllm_model and see what breaks" experiment,
implemented as a self-contained, server-free probe. It drives the **real**
`vllm_model.VLLMModel.chat_completions` against two response shapes:

  - vLLM-shaped:  chat-completions logprobs whose `token` is the string ``"token_id:NNN"``
                  (vLLM with return_tokens_as_token_ids=True), plus a working `/tokenize`.
  - SGLang-shaped: chat-completions logprobs whose `token` is the actual token *text*
                  (the ordinary OpenAI convention SGLang follows), and no vLLM `/tokenize`.

It is NOT a pytest test (filename isn't test_*.py) so it never becomes a CI gate asserting
that another component is "broken" — it's an on-demand experiment that prints a report.

Run:  python responses_api_models/sglang_model/diagnostic_vllm_vs_sglang.py
Needs only the Gym framework (nemo_gym + aiohttp + fastapi). No GPU, no live server.

What it shows (and why sglang_model exists):
  #2 logprobs/token-ids — vllm_model recovers ids via `token.removeprefix("token_id:")` and a
     vLLM `/tokenize` call. On SGLang-shaped data the "ids" are silently the token *text*
     (no error raised) and `/tokenize` is absent. sglang_model sidesteps both by using SGLang
     native `/generate` (token-in / token-out).
  #4 max-seq-len — vllm_model detects overflow by string-matching the *vLLM* 400 message;
     SGLang's message differs, so it is missed and re-raised. sglang_model prevents overflow
     with cap_to_context instead.
"""

import asyncio
import os
import sys
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from aiohttp.client_exceptions import ClientResponseError

    from responses_api_models.vllm_model.app import VLLMModel

    _IMPORT_ERR = None
except Exception as e:  # nemo_gym / vllm_model not importable here
    _IMPORT_ERR = e


# ----------------------------- fakes (drive the real chat_completions) -----------------------------
class _FakeBody:
    def model_dump(self, exclude_unset=True):
        return {"messages": [{"role": "user", "content": "hi"}], "model": "m"}


class _MockClient:
    """Stands in for Gym's ServerClient: create_chat_completion + create_tokenize."""

    def __init__(self, *, completion=None, raise_chat=None, tokenize=None, raise_tokenize=None):
        self._completion = completion
        self._raise_chat = raise_chat
        self._tokenize = tokenize
        self._raise_tokenize = raise_tokenize

    async def create_chat_completion(self, **kw):
        if self._raise_chat is not None:
            raise self._raise_chat
        return self._completion

    async def create_tokenize(self, **kw):
        if self._raise_tokenize is not None:
            raise self._raise_tokenize
        return self._tokenize


def _make_vllm_self(client):
    cfg = SimpleNamespace(
        sequential_reasoning_allowed=True,
        uses_reasoning_parser=False,
        return_token_id_information=True,
        name="sglang_diag",
        model="m",
    )
    me = SimpleNamespace(config=cfg)
    me._preprocess_chat_completion_create_params = lambda request, body_dict: body_dict
    me._resolve_client = lambda request: client
    # reuse the real empty-completion builder (only touches me.config.model)
    me._create_empty_chat_completion = lambda: VLLMModel._create_empty_chat_completion(me)
    return me


def _completion(tokens):
    """A /v1/chat/completions response whose logprob `token` fields are `tokens`."""
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"},
                "logprobs": {"content": [{"token": t, "logprob": -0.1} for t in tokens]},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": len(tokens), "total_tokens": 3 + len(tokens)},
    }


def _http_400(body: bytes):
    err = ClientResponseError(request_info=None, history=(), status=400)
    err.response_content = body
    return err


def _call(client):
    me = _make_vllm_self(client)
    return asyncio.run(VLLMModel.chat_completions(me, SimpleNamespace(), _FakeBody()))


# ----------------------------- scenarios -----------------------------
def scenario_2_vllm_shaped():
    """CONTROL: vLLM emits `token_id:NNN` + a working /tokenize -> usable integer token-ids."""
    client = _MockClient(completion=_completion(["token_id:10", "token_id:11"]), tokenize={"tokens": [1, 2, 3]})
    out = _call(client)
    ids = out.choices[0].message.generation_token_ids
    ok = ids == [10, 11]
    return ok, f"recovered generation_token_ids={ids} (ints) — vllm_model works against vLLM"


def scenario_2_sglang_token_text():
    """BREAK (silent): SGLang logprob `token` is token TEXT, not `token_id:NNN`. vllm_model does
    `token.removeprefix("token_id:")`, which is a no-op on plain text -> the "token ids" are
    actually token strings. No exception is raised; training silently gets corrupt ids."""
    client = _MockClient(completion=_completion(["He", "llo"]), tokenize={"tokens": [1, 2, 3]})
    out = _call(client)
    ids = out.choices[0].message.generation_token_ids
    all_int = isinstance(ids, list) and all(isinstance(x, int) for x in ids)
    broke = not all_int  # SGLang -> token text strings, not integer ids
    return broke, f"generation_token_ids={ids} -> token TEXT, not integer ids (SILENT corruption, no error raised)"


def scenario_2_sglang_no_tokenize():
    """BREAK: SGLang has no vLLM-style /tokenize endpoint -> prompt_token_ids unrecoverable."""
    client = _MockClient(
        completion=_completion(["token_id:10"]),
        raise_tokenize=_http_400(b'{"object":"error","message":"Not Found","code":404}'),
    )
    try:
        _call(client)
        return False, "UNEXPECTED success: /tokenize did not break"
    except Exception as e:
        return True, f"{type(e).__name__}: client.create_tokenize() (vLLM /tokenize) is absent on SGLang"


def scenario_4_vllm_context_handled():
    """CONTROL: vLLM 400 'maximum context length...' -> detected, graceful empty (finish=length)."""
    msg = b'{"object":"error","message":"This model\'s maximum context length is 4096 tokens. ...","code":400}'
    client = _MockClient(raise_chat=_http_400(msg))
    out = _call(client)
    ok = out.choices[0].finish_reason == "length"
    return ok, f"detected overflow -> finish_reason={out.choices[0].finish_reason!r} (graceful)"


def scenario_4_sglang_context_missed():
    """BREAK: SGLang's 400 message differs -> not detected -> re-raised (not graceful)."""
    msg = b'{"object":"error","message":"Input length 5000 exceeds the maximum allowed length 4096","code":400}'
    client = _MockClient(raise_chat=_http_400(msg))
    try:
        _call(client)
        return False, "UNEXPECTED: overflow handled (string matched?)"
    except ClientResponseError:
        return True, "SGLang's 400 message lacks 'context length'/'max_tokens' -> undetected -> raised"


SCENARIOS = [
    ("#2 token-ids", "vLLM-shaped", "CONTROL (should work)", scenario_2_vllm_shaped),
    ("#2 token-ids", "SGLang-shaped", "EXPECTED BREAK (silent)", scenario_2_sglang_token_text),
    ("#2 /tokenize", "SGLang-shaped", "EXPECTED BREAK", scenario_2_sglang_no_tokenize),
    ("#4 overflow", "vLLM-shaped", "CONTROL (should work)", scenario_4_vllm_context_handled),
    ("#4 overflow", "SGLang-shaped", "EXPECTED BREAK", scenario_4_sglang_context_missed),
]


def main():
    if _IMPORT_ERR is not None:
        print(f"SKIP diagnostic: vllm_model/nemo_gym not importable here: {_IMPORT_ERR}")
        return 0
    print("=" * 96)
    print("DIAGNOSTIC: can the existing vllm_model serve SGLang? (drives real VLLMModel.chat_completions)")
    print("=" * 96)
    all_as_expected = True
    for req, shape, expectation, fn in SCENARIOS:
        as_expected, detail = fn()
        # CONTROL expects ok==True; EXPECTED BREAK expects the break to be observed (fn returns True when broken)
        verdict = "OK" if as_expected else "!! UNEXPECTED !!"
        all_as_expected = all_as_expected and as_expected
        print(f"\n[{req:13}] {shape:14} {expectation:22} -> {verdict}")
        print(f"    {detail}")
    print("\n" + "-" * 96)
    print("CONCLUSION: vllm_model recovers token-ids via the vLLM `token_id:NNN` logprob convention +")
    print("a vLLM `/tokenize` endpoint, and detects overflow via vLLM's 400 message — none of which")
    print("SGLang provides. => a dedicated sglang_model (native /generate, token-in/out, cap_to_context)")
    print("is required. #3 (Responses<->ChatCompletions converter) is reused unchanged; #1 (multi-turn")
    print("on-policy retokenization) is a nemo_rl-worker concern, out of scope for the Gym model server.")
    print("-" * 96)
    return 0 if all_as_expected else 1


if __name__ == "__main__":
    sys.exit(main())
