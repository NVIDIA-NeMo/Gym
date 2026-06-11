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
    """Parse an SGLang /generate response into (generated_token_ids, logprobs).

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


def build_sampling_params(body_dict: Dict[str, Any], default_max_new_tokens: int) -> Dict[str, Any]:
    """Map OpenAI chat-completion params to SGLang sampling_params."""
    max_new = body_dict.get("max_completion_tokens") or body_dict.get("max_tokens") or default_max_new_tokens
    sp: Dict[str, Any] = {
        "temperature": body_dict.get("temperature", 1.0),
        "top_p": body_dict.get("top_p", 1.0),
        "max_new_tokens": int(max_new),
    }
    if body_dict.get("top_k") is not None:
        sp["top_k"] = body_dict["top_k"]
    if body_dict.get("stop"):
        sp["stop"] = body_dict["stop"]
    return sp


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
    # Cap the prompt at ctx-2 so that input + (>=1 generated token) <= ctx-1 < ctx. (A ctx-1
    # truncation combined with a max(1, ...) floor on `room` could yield input+gen == ctx, which
    # violates the bound and can overflow the context.)
    max_prompt_len = ctx - 2
    if len(prompt_token_ids) > max_prompt_len:
        prompt_token_ids = prompt_token_ids[:max_prompt_len]
    room = ctx - 1 - len(prompt_token_ids)  # >= 1 (since len <= ctx-2); input + room == ctx-1 < ctx
    if sampling_params["max_new_tokens"] > room:
        sampling_params = {**sampling_params, "max_new_tokens": room}
    return prompt_token_ids, sampling_params
