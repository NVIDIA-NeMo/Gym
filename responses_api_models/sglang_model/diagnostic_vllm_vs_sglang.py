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
"""Diagnostic (Brian's experiment): run the EXISTING vllm_model path against a REAL SGLang server
and see what breaks.

This sends **real HTTP requests to a live SGLang OpenAI endpoint** (no mocks, no canned responses)
and checks whether the server actually satisfies the assumptions `vllm_model` depends on. If no
server is provided it does NOT run and does NOT "pass" — it prints how to run it and exits non-zero.

The assumptions it probes (see responses_api_models/vllm_model/app.py):
  #2a  L324 sets `return_tokens_as_token_ids=True` and L511 does `token.removeprefix("token_id:")`
       -> vllm_model needs the chat-completions logprob `token` to be the literal string
          ``"token_id:NNN"``. We check what the real server returns.
  #2b  L526 calls `client.create_tokenize(...)` (a vLLM ``/tokenize`` endpoint) to recover the
       prompt token ids. We check whether that endpoint exists on SGLang.
  #4   L472 detects context overflow by string-matching the *vLLM* 400 message
       (``"context length"`` / ``"max_tokens"``). We trigger an overflow and inspect the message.

How to run (Brian's experiment, end to end):
  1) start a stock SGLang server for any model:
       python -m sglang.launch_server --model-path <model> --port 30000
  2) point this probe at it:
       SGLANG_BASE_URL=http://localhost:30000 SGLANG_MODEL=<model> \
           python responses_api_models/sglang_model/diagnostic_vllm_vs_sglang.py

Any BREAK below is a real vllm_model<->SGLang incompatibility that the sglang_model server avoids
(native /generate token-in/out, local tokenization, cap_to_context). The fix itself is verified in
tests/test_app.py and tests/test_logic.py — this file's job is only the "what breaks" experiment.
"""

import json
import os
import sys
import urllib.error
import urllib.request


BASE_URL = os.environ.get("SGLANG_BASE_URL", "").rstrip("/")
MODEL = os.environ.get("SGLANG_MODEL", "")


def _post(path, payload, timeout=120):
    """POST json to BASE_URL+path; return (status_or_None, parsed_or_text)."""
    url = BASE_URL + path
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # connection refused / timeout / DNS
        return None, f"{type(e).__name__}: {e}"


def probe_2a_token_id_format():
    """Does the real /v1/chat/completions return logprob tokens as `token_id:NNN`?"""
    status, body = _post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 5,
            "logprobs": True,
            "top_logprobs": 1,
            "return_tokens_as_token_ids": True,  # exactly what vllm_model sends (app.py L323)
        },
    )
    if status != 200:
        return "ERROR", f"chat/completions -> {status}: {str(body)[:140]}"
    try:
        tok = body["choices"][0]["logprobs"]["content"][0]["token"]
    except (KeyError, IndexError, TypeError) as e:
        return "BREAK", f"no logprobs.content[].token in response ({e!r}) -> vllm_model L504/511 cannot read ids"
    if str(tok).startswith("token_id:"):
        return "OK", f"logprob token={tok!r} -> 'token_id:' convention present; vllm_model id-recovery works"
    return (
        "BREAK",
        f"logprob token={tok!r} is token TEXT, not 'token_id:NNN' (server ignores "
        f"return_tokens_as_token_ids); removeprefix is a no-op -> generation_token_ids become token "
        f"strings: silent corruption",
    )


def probe_2b_tokenize_endpoint():
    """Does the vLLM-style /tokenize endpoint (used for prompt ids) exist on SGLang?"""
    status, body = _post("/tokenize", {"model": MODEL, "prompt": "hello world"})
    if status == 200:
        return "OK", "/tokenize exists -> vllm_model can recover prompt_token_ids"
    return "BREAK", f"/tokenize -> {status} (vLLM endpoint absent on SGLang) -> vllm_model L526 create_tokenize fails"


def probe_4_overflow_message():
    """Trigger a context overflow; does the 400 message match vllm_model's detector?"""
    huge = "word " * 100000  # force an over-context prompt
    status, body = _post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": huge}]})
    s = str(body)
    if status == 200:
        return "?", "no overflow error returned (server silently truncated?) -> can't confirm vllm_model's path"
    matched = ("context length" in s) or ("max_tokens" in s)
    if matched:
        return "OK", f"overflow {status} message matches vllm_model's string check"
    return (
        "BREAK",
        f"overflow {status} message lacks 'context length'/'max_tokens' -> vllm_model L472 misses it: {s[:120]}",
    )


PROBES = [
    ("#2a token-id fmt", probe_2a_token_id_format),
    ("#2b /tokenize", probe_2b_tokenize_endpoint),
    ("#4 overflow msg", probe_4_overflow_message),
]


def main():
    if not BASE_URL:
        print("NOT RUN — Brian's experiment needs a live SGLang server (this probe does not mock).")
        print("  1) python -m sglang.launch_server --model-path <model> --port 30000")
        print("  2) SGLANG_BASE_URL=http://localhost:30000 SGLANG_MODEL=<model> python <this file>")
        print("\n(No server provided -> not run. This is intentionally NOT a pass.)")
        return 2  # not-run sentinel: never let "no server" masquerade as success
    print(f"Brian's experiment: probing LIVE SGLang at {BASE_URL} (model={MODEL or '<unset>'})")
    print("against the assumptions the existing vllm_model server makes:\n")
    breaks = 0
    for name, fn in PROBES:
        verdict, detail = fn()
        if verdict == "BREAK":
            breaks += 1
        print(f"[{name:16}] {verdict:6} {detail}")
    print(f"\n{breaks} of {len(PROBES)} vllm_model assumptions broke against this SGLang server.")
    print("Each BREAK is what sglang_model avoids via native /generate (token-in/out), local")
    print("tokenization, and cap_to_context. (sglang_model's own behavior is verified in tests/.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
