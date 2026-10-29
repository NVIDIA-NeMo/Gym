# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

from nemo_gym.rollout_prompt_storage import FORMAT, STORAGE_KEY, decode_rollout, encode_rollout


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
    prompts = [c["request"][field] for c in document["ng_trajectory"]["model_calls"]]
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
    assert "full" in document["ng_trajectory"]["model_calls"][1]["request"]["messages"]
    assert json.dumps(decode_rollout(document)) == json.dumps(row)


def test_literals_missing_fields_and_identical_retry():
    row = rollout([None, "plain input", {"append_to": 0, "items": []}, [], [LONG_SYSTEM], [LONG_SYSTEM]], "input")
    row["ng_trajectory"]["model_calls"].extend([{"request": {"model": "other"}}, {"request": None}])
    document = encode_rollout(row)
    calls = document["ng_trajectory"]["model_calls"]
    assert calls[2]["request"]["input"] == {"full": {"append_to": 0, "items": []}}
    assert calls[3]["request"]["input"] == {"full": []}
    assert calls[5]["request"]["input"] == {"append_to": 4, "items": []}
    assert decode_rollout(document) == row


def test_interleaved_calls_reset_instead_of_assuming_the_last_call_was_the_parent():
    row = rollout([[LONG_SYSTEM], [{"role": "user", "content": "judge something"}], [LONG_SYSTEM, "continue"]])
    document = encode_rollout(row)
    assert "full" in document["ng_trajectory"]["model_calls"][2]["request"]["messages"]
    assert decode_rollout(document) == row


@pytest.mark.parametrize("base", [-1, 1, 99, True, 0.0, "0"])
def test_reject_invalid_reference(base):
    document = encode_rollout(rollout([[LONG_SYSTEM], [LONG_SYSTEM, "next"]]))
    document["ng_trajectory"]["model_calls"][1]["request"]["messages"]["append_to"] = base
    with pytest.raises(ValueError, match="earlier call"):
        decode_rollout(document)


@pytest.mark.parametrize("prompt", [[], {"full": [], "extra": 1}, {"append_to": 0, "items": "bad"}])
def test_reject_malformed_prompt(prompt):
    document = encode_rollout(rollout([[LONG_SYSTEM], [LONG_SYSTEM, "next"]]))
    document["ng_trajectory"]["model_calls"][1]["request"]["messages"] = prompt
    with pytest.raises(ValueError):
        decode_rollout(document)


def test_reject_non_array_base_and_unknown_format():
    document = encode_rollout(rollout(["text", []]))
    document["ng_trajectory"]["model_calls"][1]["request"]["messages"] = {"append_to": 0, "items": []}
    with pytest.raises(ValueError, match="arrays"):
        decode_rollout(document)
    document[STORAGE_KEY] = "future-v2"
    with pytest.raises(ValueError, match="unsupported"):
        decode_rollout(document)


def test_no_calls_and_tiny_prompts_need_no_references():
    assert decode_rollout(encode_rollout({"reward": 0})) == {"reward": 0}
    row = rollout([[1], [1, 2]])
    document = encode_rollout(row)
    assert document["ng_trajectory"]["model_calls"][1]["request"]["messages"] == {"full": [1, 2]}
    assert decode_rollout(document) == row
    assert decode_rollout(row) is row
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
    assert document[STORAGE_KEY] == FORMAT


@pytest.mark.asyncio
async def test_readers_expand_mixed_plain_and_compact_rows(tmp_path):
    import orjson

    from nemo_gym.rollout_collection import _read_jsonl, loads_jsonl_line
    from nemo_gym.rollout_compare.rows import _read_jsonl as compare_read
    from nemo_gym.rollout_health import _LineSlice, _read_record
    from nemo_gym.rollout_reverification import _yield_inputs_and_rollouts_paired

    row = rollout([[LONG_SYSTEM], [LONG_SYSTEM, "next"]])
    row.update({"_ng_task_index": 0, "_ng_rollout_index": 0})
    other = {**row, "_ng_task_index": 1}
    path = tmp_path / "rollouts.jsonl"
    encoded_line = orjson.dumps(encode_rollout(row)) + b"\n"
    path.write_bytes(encoded_line + orjson.dumps(other) + b"\n")
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_bytes(orjson.dumps(row) + b"\n" + orjson.dumps(other) + b"\n")
    assert await _read_jsonl(path) == [row, other]
    assert compare_read(path) == [row, other]
    assert loads_jsonl_line(encoded_line, path, 1) == row
    # Health's worker must inspect the expanded record.
    line = _LineSlice(path=str(path), offset=0, length=len(encoded_line), ordinal=0, source_index=0, line_number=1)
    assert _read_record(line) == (row, None)
    pairs = list(_yield_inputs_and_rollouts_paired(inputs, path))
    assert [pair.rollout for pair in pairs] == [row, other]


def test_malformed_storage_is_not_silently_consumed(tmp_path):
    from nemo_gym.config_types import ConfigError
    from nemo_gym.rollout_collection import loads_jsonl_line

    with pytest.raises(ConfigError, match="unsupported prompt-storage"):
        loads_jsonl_line(json.dumps({STORAGE_KEY: "future-v2"}), tmp_path / "bad.jsonl", 3)
