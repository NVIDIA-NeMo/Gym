# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from benchmarks.mrcr import prepare


@pytest.fixture
def tokenizer_path(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(path)
    return str(path)


def source_rows():
    return [
        {"prompt": json.dumps(messages), "answer": str(index), "random_string_to_prepend": "prefix", "n_needles": 2}
        for index, messages in enumerate(
            [
                [{"role": "user", "content": "hello " * 1000}],
                [{"role": "user", "content": "hello world"}],
                [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}],
            ]
        )
    ]


@pytest.mark.parametrize("cap", [None, 2])
def test_process_workers_preserve_exact_rows_order_and_length_filter(monkeypatch, tmp_path, tokenizer_path, cap):
    monkeypatch.setattr(prepare, "load_dataset", lambda *args, **kwargs: source_rows())
    serial = prepare.prepare(tokenizer_path, cap, tmp_path / "serial.jsonl", num_workers=1)
    parallel = prepare.prepare(tokenizer_path, cap, tmp_path / "parallel.jsonl", num_workers=2)
    assert parallel.read_bytes() == serial.read_bytes()
    rows = [json.loads(line) for line in parallel.read_text().splitlines()]
    assert [row["expected_answer"] for row in rows] == (["0", "1", "2"] if cap is None else ["1", "2"])
    assert [row["n_tokens"] for row in rows] == ([1000, 2, 2] if cap is None else [2, 2])
    assert rows[-1]["responses_create_params"]["input"] == json.loads(source_rows()[-1]["prompt"])


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_invalid_worker_count_is_rejected(workers, tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        prepare.prepare(output_fpath=tmp_path / "output.jsonl", num_workers=workers)


def test_worker_tokenizer_errors_reach_caller(monkeypatch, tmp_path):
    monkeypatch.setattr(prepare, "load_dataset", lambda *args, **kwargs: source_rows())
    with pytest.raises((ValueError, OSError)):
        prepare.prepare("/missing-mrcr-tokenizer", output_fpath=tmp_path / "out.jsonl", num_workers=2)
