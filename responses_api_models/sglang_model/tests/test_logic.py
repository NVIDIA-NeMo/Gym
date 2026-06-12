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
"""Tests for the sglang_model adapter.

L1: pure unit tests of _logic.py (no framework / no model needed).
L6: tokenization-parity tests against the real diffusion-model tokenizer (needs transformers).

Run standalone:  python tests/test_logic.py
Or via pytest:   pytest tests/test_logic.py
"""

import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, GYM_ROOT)

from responses_api_models.sglang_model._logic import (  # noqa: E402
    build_sampling_params,
    cap_to_context,
    extract_generated_tokens_and_logprobs,
    normalize_token_ids,
)


MODEL_PATH = os.environ.get("SGLANG_MODEL_PATH", "/linnanw/justGRPO/asset/Nemotron-Labs-Diffusion-3B")

# L6 tokenization-parity tests need the real diffusion-model tokenizer (a local path) + transformers,
# and the chat-template ones additionally need jinja2. None of those are guaranteed in CI, so L6 SKIPS
# cleanly when unavailable (per Gym's test-skip-guard convention). L1 is pure and always runs. Point at
# a model with SGLANG_MODEL_PATH to exercise L6 locally.
_HAS_MODEL = os.path.isdir(MODEL_PATH)
try:
    import transformers  # noqa: F401

    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False
try:
    import jinja2  # noqa: F401  (transformers.apply_chat_template requires it)

    _HAS_JINJA2 = True
except Exception:
    _HAS_JINJA2 = False

try:
    import pytest
except ImportError:  # allow standalone `python tests/test_logic.py`
    pytest = None


def _skip_unless(cond, reason):
    """Skip a test when cond is False — works under both pytest and the standalone runner."""

    def deco(fn):
        if not cond:
            fn._skip_reason = reason
        return pytest.mark.skipif(not cond, reason=reason)(fn) if pytest is not None else fn

    return deco


requires_real_model = _skip_unless(
    _HAS_MODEL and _HAS_TRANSFORMERS,
    f"L6 needs real model (present={_HAS_MODEL}, SGLANG_MODEL_PATH) + transformers={_HAS_TRANSFORMERS}",
)
requires_chat_template = _skip_unless(
    _HAS_MODEL and _HAS_TRANSFORMERS and _HAS_JINJA2,
    f"L6 chat-template needs real model + transformers + jinja2 (jinja2={_HAS_JINJA2})",
)


# ----------------------------- L1: extract_generated_tokens_and_logprobs -----------------------------
def test_extract_dict_form():
    r = {"meta_info": {"output_token_logprobs": [{"token_id": 5, "logprob": -0.1}, {"id": 7, "logprob": -0.2}]}}
    toks, lps = extract_generated_tokens_and_logprobs(r)
    assert toks == [5, 7] and lps == [-0.1, -0.2]


def test_extract_tuple_form():
    r = {"meta_info": {"output_token_logprobs": [[-0.5, 11, "a"], [-0.6, 12, "b"]]}}
    toks, lps = extract_generated_tokens_and_logprobs(r)
    assert toks == [11, 12] and lps == [-0.5, -0.6]


def test_extract_val_idx_fallback():
    r = {"meta_info": {"output_token_logprobs_val": [-0.1, -0.2], "output_token_logprobs_idx": [3, 4]}}
    toks, lps = extract_generated_tokens_and_logprobs(r)
    assert toks == [3, 4] and lps == [-0.1, -0.2]


def test_extract_empty_raises():
    for r in ({}, {"meta_info": {}}, {"meta_info": {"output_token_logprobs": []}}):
        try:
            extract_generated_tokens_and_logprobs(r)
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass


def test_extract_mismatch_raises():
    r = {"meta_info": {"output_token_logprobs_val": [-0.1, -0.2], "output_token_logprobs_idx": [3]}}
    try:
        extract_generated_tokens_and_logprobs(r)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_extract_malformed_entry_raises():
    r = {"meta_info": {"output_token_logprobs": [{"token_id": None, "logprob": -0.1}]}}
    try:
        extract_generated_tokens_and_logprobs(r)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


# ----------------------------- L1: normalize_token_ids -----------------------------
def test_normalize_flat_list():
    assert normalize_token_ids([1, 2, 3]) == [1, 2, 3]


def test_normalize_dict():
    # transformers 5.x bug: list(dict) would grab the KEYS -> this must extract input_ids
    assert normalize_token_ids({"input_ids": [9, 8, 7], "attention_mask": [1, 1, 1]}) == [9, 8, 7]


def test_normalize_batchencoding_like():
    class FakeBE:
        def __init__(self, ids):
            self._d = {"input_ids": ids}

        def __getitem__(self, k):
            return self._d[k]

        @property
        def input_ids(self):
            return self._d["input_ids"]

    assert normalize_token_ids(FakeBE([4, 5, 6])) == [4, 5, 6]


def test_normalize_nested():
    assert normalize_token_ids([[1, 2, 3]]) == [1, 2, 3]


def test_normalize_casts_to_int():
    class IntLike(int):
        pass

    out = normalize_token_ids([IntLike(1), IntLike(2)])
    assert out == [1, 2] and all(type(t) is int for t in out)


# ----------------------------- L1: build_sampling_params -----------------------------
def test_sp_default_when_absent():
    sp = build_sampling_params({}, 777)
    assert sp == {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 777}


def test_sp_precedence():
    # max_completion_tokens wins over max_tokens which wins over default
    assert build_sampling_params({"max_completion_tokens": 10, "max_tokens": 20}, 99)["max_new_tokens"] == 10
    assert build_sampling_params({"max_tokens": 20}, 99)["max_new_tokens"] == 20


def test_sp_optional_topk_stop():
    sp = build_sampling_params({"temperature": 0.7, "top_p": 0.9, "top_k": 40, "stop": ["</s>"]}, 50)
    assert sp["temperature"] == 0.7 and sp["top_p"] == 0.9 and sp["top_k"] == 40 and sp["stop"] == ["</s>"]
    # top_k None and falsy stop are omitted
    sp2 = build_sampling_params({"top_k": None, "stop": []}, 50)
    assert "top_k" not in sp2 and "stop" not in sp2


# ----------------------------- L1: cap_to_context -----------------------------
def test_cap_no_change_when_short():
    ids, sp = cap_to_context([1, 2, 3], {"max_new_tokens": 100}, 4096)
    assert ids == [1, 2, 3] and sp["max_new_tokens"] == 100


def test_cap_truncates_long_prompt():
    ids, sp = cap_to_context(list(range(5000)), {"max_new_tokens": 100}, 4096)
    # prompt alone exceeds ctx -> truncate to ctx-2, leaving room for >=1 gen token, total < ctx
    assert len(ids) == 4094 and ids == list(range(4094))
    assert len(ids) + sp["max_new_tokens"] < 4096


def test_cap_shrinks_max_new():
    # prompt 4000, ctx 4096 -> room = 4096-4000-1 = 95
    ids, sp = cap_to_context(list(range(4000)), {"max_new_tokens": 2048}, 4096)
    assert len(ids) == 4000 and sp["max_new_tokens"] == 95


def test_cap_room_floor_at_one():
    # prompt == ctx: truncate to ctx-2 and floor generation at 1 token, with input+gen still < ctx
    ids, sp = cap_to_context(list(range(4096)), {"max_new_tokens": 2048}, 4096)
    assert len(ids) == 4094 and sp["max_new_tokens"] == 1
    assert len(ids) + sp["max_new_tokens"] == 4095 < 4096


def test_cap_does_not_mutate_input():
    sp_in = {"max_new_tokens": 2048}
    _, sp_out = cap_to_context(list(range(4000)), sp_in, 4096)
    assert sp_in["max_new_tokens"] == 2048 and sp_out["max_new_tokens"] == 95


# ----------------------------- L6: tokenization parity (real tokenizer) -----------------------------
def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


@requires_chat_template
def test_l6_chat_template_parity():
    tok = _load_tokenizer()
    msgs = [{"role": "user", "content": "What is 2 + 2? Put the answer in \\boxed{}."}]
    ids_tok = normalize_token_ids(
        tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    )
    assert len(ids_tok) > 0 and all(isinstance(t, int) for t in ids_tok)
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids_txt = tok.encode(text, add_special_tokens=False)
    assert ids_tok == ids_txt, f"template tokenize != encode(template text): {len(ids_tok)} vs {len(ids_txt)}"


@requires_chat_template
def test_l6_dict_return_normalization_real():
    tok = _load_tokenizer()
    msgs = [{"role": "user", "content": "hello"}]
    flat = normalize_token_ids(
        tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    )
    as_dict = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)
    assert normalize_token_ids(as_dict) == flat  # normalizer handles the real dict/BatchEncoding


@requires_real_model
def test_l6_skip_special_tokens_strips():
    tok = _load_tokenizer()
    special_id = tok.eos_token_id
    if special_id is None and tok.all_special_ids:
        special_id = tok.all_special_ids[0]
    assert special_id is not None
    body = tok.encode("hello world", add_special_tokens=False)
    ids = body + [special_id]
    clean = tok.decode(ids, skip_special_tokens=True)
    raw = tok.decode(ids, skip_special_tokens=False)
    special_str = tok.convert_ids_to_tokens(special_id)
    # the special-token string is present raw but stripped from the clean (verifier) content
    assert clean != raw
    assert special_str not in clean


# ----------------------------- runner -----------------------------
def _run():
    import traceback

    items = sorted(globals().items())
    l1 = [k for k, v in items if k.startswith("test_") and not k.startswith("test_l6") and callable(v)]
    l6 = [k for k, v in items if k.startswith("test_l6") and callable(v)]
    npass = nfail = nskip = 0
    for name in l1 + l6:
        fn = globals()[name]
        reason = getattr(fn, "_skip_reason", None)
        if reason:
            print(f"  SKIP {name}: {reason}")
            nskip += 1
            continue
        try:
            fn()
            print(f"  PASS {name}")
            npass += 1
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            nfail += 1
    print(f"\n{npass} passed, {nfail} failed, {nskip} skipped  (L1={len(l1)}, L6={len(l6)})")
    return nfail


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
