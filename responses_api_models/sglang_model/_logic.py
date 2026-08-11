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
"""Pure, framework-free logic for the sglang_model Gym server.

Split out from app.py so it can be unit-tested without importing nemo_gym / the server
framework (which is only available inside the per-server venv). No third-party imports.
"""

from typing import Any, Dict, List, Tuple


def extract_generated_tokens_and_logprobs(
    result: Dict[str, Any],
) -> Tuple[List[int], List[float]]:
    """Parse an SGLang `meta_info` carrier into (generated_token_ids, logprobs).

    Works for both transports, because SGLang emits the same `meta_info` shape either way:
    the `/generate` response body, and (since 0.5.13, with `return_meta_info=true`) each
    `/v1/chat/completions` *choice*.

    Mirrors nemo_rl's sglang_worker: handles the dict-form and tuple-form
    `meta_info.output_token_logprobs`, plus the `output_token_logprobs_val/idx` fallback.
    Raises RuntimeError on malformed / missing logprobs.
    """
    meta = result.get("meta_info", {}) or {}
    otl = meta.get("output_token_logprobs", [])
    if otl:
        toks: List[int] = []
        lps: List[float] = []
        for item in otl:
            if isinstance(item, dict):
                tid = item.get("token_id", item.get("id"))
                lp = item.get("logprob")
            else:  # [logprob, token_id, (optional) text]
                lp, tid = item[0], item[1]
            if tid is None or lp is None:
                raise RuntimeError(f"Malformed SGLang output_token_logprobs entry: {item!r}")
            toks.append(int(tid))
            lps.append(float(lp))
        return toks, lps

    val = meta.get("output_token_logprobs_val", result.get("output_token_logprobs_val", []))
    idx = meta.get("output_token_logprobs_idx", result.get("output_token_logprobs_idx", []))
    ids = result.get("output_ids", meta.get("output_ids", []))
    if val:
        new = idx if idx else ids
        if len(new) != len(val):
            raise RuntimeError(f"SGLang mismatched gen logprob fields: {len(new)} ids vs {len(val)} logprobs")
        return [int(x) for x in new], [float(x) for x in val]

    raise RuntimeError(
        f"SGLang /generate returned no generation logprobs (keys={sorted(result)}, meta={sorted(meta)}). "
        "Ensure the request set return_logprob=true."
    )


def normalize_token_ids(rendered: Any) -> List[int]:
    """Normalize a chat-template tokenization result to a flat list[int].

    transformers 5.x `apply_chat_template(tokenize=True)` can return a dict / BatchEncoding
    (in which case `list(...)` would grab the KEYS), or a nested `[[...]]` for a single
    conversation. This collapses all of those to a flat list of python ints.
    """
    if isinstance(rendered, dict) or hasattr(rendered, "input_ids"):
        rendered = rendered["input_ids"]
    rendered = list(rendered)
    if rendered and isinstance(rendered[0], (list, tuple)):
        rendered = list(rendered[0])
    return [int(t) for t in rendered]


# OpenAI chat-completion param -> SGLang sampling_params key, for params that pass through
# unchanged when set. `n` is deliberately absent: /generate would return a single choice
# regardless, so a caller asking for n>1 must be told rather than silently under-served.
_PASSTHROUGH_SAMPLING_PARAMS = {
    "top_k": "top_k",
    "stop": "stop",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "repetition_penalty": "repetition_penalty",
    "min_p": "min_p",
}

# Params this transport cannot honor. Reported so a caller never silently gets a different
# sampling distribution than the one their recipe configured.
_UNSUPPORTED_SAMPLING_PARAMS = ("n", "seed", "response_format", "logit_bias")


def build_sampling_params(body_dict: Dict[str, Any], default_max_new_tokens: int) -> Dict[str, Any]:
    """Map OpenAI chat-completion params to SGLang sampling_params (``/generate`` transport).

    Only used by the legacy ``/generate`` transport; the ``chat`` transport forwards the
    request to SGLang's OpenAI-compatible endpoint verbatim, so it needs no mapping.
    """
    max_new = body_dict.get("max_completion_tokens")
    if max_new is None:
        max_new = body_dict.get("max_tokens")
    if max_new is None:
        max_new = default_max_new_tokens
    sp: Dict[str, Any] = {
        "temperature": body_dict.get("temperature", 1.0),
        "top_p": body_dict.get("top_p", 1.0),
        "max_new_tokens": int(max_new),
    }
    for openai_key, sglang_key in _PASSTHROUGH_SAMPLING_PARAMS.items():
        value = body_dict.get(openai_key)
        if value is not None and value != []:
            sp[sglang_key] = value
    return sp


def unsupported_sampling_params(body_dict: Dict[str, Any]) -> List[str]:
    """Names of request params the ``/generate`` transport cannot honor, for a loud warning.

    Silently ignoring these would make the realized rollout distribution diverge from the
    configured recipe -- which is invisible in the training data.
    """
    return [key for key in _UNSUPPORTED_SAMPLING_PARAMS if body_dict.get(key) is not None]


def cap_to_context(
    prompt_token_ids: List[int], sampling_params: Dict[str, Any], ctx: int
) -> Tuple[List[int], Dict[str, Any]]:
    """Keep the request within the context window: guarantee input_len + max_new_tokens < ctx
    while always leaving room for at least one generated token (SGLang /generate errors when a
    request exceeds the context). If the prompt alone is too long it is truncated. Mirrors
    nemo_rl's SGLang worker. Returns the (possibly truncated) ids and (possibly adjusted) params;
    does not mutate the inputs.
    """
    if not ctx:
        return prompt_token_ids, sampling_params
    if ctx < 2:
        # Below this there is no room for both a prompt and a generated token. Raising beats
        # silently POSTing a negative `max_new_tokens`, which SGLang rejects with an opaque 400.
        raise ValueError(f"context_length={ctx} is too small; need >= 2 to leave room for one generated token")
    # Cap the prompt at ctx-2 so that input + (>=1 generated token) <= ctx-1 < ctx. (A ctx-1
    # truncation combined with a max(1, ...) floor on `room` could yield input+gen == ctx, which
    # violates the bound and can overflow the context.)
    max_prompt_len = ctx - 2
    truncated = len(prompt_token_ids) > max_prompt_len
    if truncated:
        prompt_token_ids = prompt_token_ids[:max_prompt_len]
    room = ctx - 1 - len(prompt_token_ids)  # >= 1 (since len <= ctx-2); input + room == ctx-1 < ctx
    if sampling_params["max_new_tokens"] > room:
        sampling_params = {**sampling_params, "max_new_tokens": room}
    return prompt_token_ids, sampling_params


def would_truncate(prompt_token_ids: List[int], ctx: int) -> bool:
    """Whether `cap_to_context` would drop prompt tokens.

    Truncation keeps the prompt *head*, so it discards the newest user turn and the
    `add_generation_prompt` cue -- the model then generates from a malformed context. Callers
    use this to surface the rollout as degenerate instead of letting it enter the training set
    looking valid.
    """
    return bool(ctx) and ctx >= 2 and len(prompt_token_ids) > ctx - 2
