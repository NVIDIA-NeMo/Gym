# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

from nemo_gym.rollout_prompt_storage import FORMAT, decode_rollout, encode_rollout


def rollout(prompts, field="messages"):
    return {
        "reward": 1.0,
        "verifier_metadata": {"expected": 750},
        "ng_trajectory": {
            "model_calls": [
                {
                    "model_call_id": f"call-{i}",
                    "request": {field: prompt, "temperature": 0.1, "tools": [{"name": "bash"}]},
                    "response": {"output": ["answer"], "generation_token_ids": [1, 2, 3]},
                    "token_stats": {"prompt_tokens": 100 + i, "completion_tokens": 3},
                }
                for i, prompt in enumerate(prompts)
            ],
            "verifier_invocations": [{"model_calls": [{"model_call_id": "call-3"}]}],
        },
    }


LONG_SYSTEM = {"role": "system", "content": "Instruction. " * 100}


@pytest.mark.parametrize("field", ["messages", "input"])
def test_append_reset_and_append_after_reset_preserve_entire_row(field):
    first = [LONG_SYSTEM, {"role": "user", "content": "task"}]
    second = first + [{"role": "assistant", "content": "answer"}, {"role": "tool", "content": "output"}]
    compacted = [LONG_SYSTEM, {"role": "user", "content": "summary"}]
    final = compacted + [{"role": "assistant", "content": "done"}]
    row = rollout([first, second, compacted, final], field)
    before = json.dumps(row)
    document = encode_rollout(row)
    prompts = [c["request"][field] for c in document["rollout"]["ng_trajectory"]["model_calls"]]
    assert prompts == [
        {"full": first},
        {"append_to": 0, "items": second[2:]},
        {"full": compacted},
        {"append_to": 2, "items": final[2:]},
    ]
    saved = json.dumps(document)
    assert json.dumps(decode_rollout(json.loads(saved))) == before
    assert json.dumps(row) == before
    assert json.dumps(document) == saved


@pytest.mark.parametrize("old,new", [(True, 1), (1, 1.0), (-0.0, 0.0), ({}, {"x": None})])
def test_equal_in_python_is_not_enough_to_reuse_prompt(old, new):
    row = rollout([[LONG_SYSTEM, old], [LONG_SYSTEM, new, "extra"]])
    document = encode_rollout(row)
    assert "full" in document["rollout"]["ng_trajectory"]["model_calls"][1]["request"]["messages"]
    assert json.dumps(decode_rollout(document)) == json.dumps(row)


def test_literals_missing_fields_and_identical_retry():
    row = rollout([None, "plain input", {"append_to": 0, "items": []}, [], [LONG_SYSTEM], [LONG_SYSTEM]], "input")
    row["ng_trajectory"]["model_calls"].extend([{"request": {"model": "other"}}, {"request": None}])
    document = encode_rollout(row)
    calls = document["rollout"]["ng_trajectory"]["model_calls"]
    assert calls[2]["request"]["input"] == {"full": {"append_to": 0, "items": []}}
    assert calls[3]["request"]["input"] == {"full": []}
    assert calls[5]["request"]["input"] == {"append_to": 4, "items": []}
    assert decode_rollout(document) == row


def test_interleaved_calls_reset_instead_of_assuming_the_last_call_was_the_parent():
    row = rollout([[LONG_SYSTEM], [{"role": "user", "content": "judge something"}], [LONG_SYSTEM, "continue"]])
    document = encode_rollout(row)
    assert "full" in document["rollout"]["ng_trajectory"]["model_calls"][2]["request"]["messages"]
    assert decode_rollout(document) == row


@pytest.mark.parametrize("base", [-1, 1, 99, True, 0.0, "0"])
def test_reject_invalid_reference(base):
    document = encode_rollout(rollout([[LONG_SYSTEM], [LONG_SYSTEM, "next"]]))
    document["rollout"]["ng_trajectory"]["model_calls"][1]["request"]["messages"]["append_to"] = base
    with pytest.raises(ValueError, match="earlier call"):
        decode_rollout(document)


@pytest.mark.parametrize("prompt", [[], {"full": [], "extra": 1}, {"append_to": 0, "items": "bad"}])
def test_reject_malformed_prompt(prompt):
    document = encode_rollout(rollout([[LONG_SYSTEM], [LONG_SYSTEM, "next"]]))
    document["rollout"]["ng_trajectory"]["model_calls"][1]["request"]["messages"] = prompt
    with pytest.raises(ValueError):
        decode_rollout(document)


def test_reject_non_array_base_and_unknown_format():
    document = encode_rollout(rollout(["text", []]))
    document["rollout"]["ng_trajectory"]["model_calls"][1]["request"]["messages"] = {"append_to": 0, "items": []}
    with pytest.raises(ValueError, match="arrays"):
        decode_rollout(document)
    document["format"] = "future-v2"
    with pytest.raises(ValueError, match="unsupported"):
        decode_rollout(document)


def test_no_calls_and_tiny_prompts_need_no_references():
    assert decode_rollout(encode_rollout({"reward": 0})) == {"reward": 0}
    row = rollout([[1], [1, 2]])
    document = encode_rollout(row)
    assert document["rollout"]["ng_trajectory"]["model_calls"][1]["request"]["messages"] == {"full": [1, 2]}
    assert decode_rollout(document) == row
    with pytest.raises(ValueError):
        decode_rollout(row)
    with pytest.raises(ValueError):
        encode_rollout(document)


def test_decoded_prompts_do_not_alias_their_bases():
    document = encode_rollout(rollout([[LONG_SYSTEM], [LONG_SYSTEM, "next"]]))
    saved = deepcopy(document)
    restored = decode_rollout(document)
    calls = restored["ng_trajectory"]["model_calls"]
    calls[1]["request"]["messages"][0]["content"] = "changed"
    assert calls[0]["request"]["messages"][0] == LONG_SYSTEM
    assert document == saved
    assert document["format"] == FORMAT
