# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from benchmarks.graphwalks import prepare


@pytest.fixture
def tokenizer_path(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(path)
    return str(path)


def source_rows():
    cases = [
        ("hello " * 1000, ["b", "a"], "bfs"),
        ("hello é", ["é", "a"], "bfs"),
        ("Perform a BFS from node root with depth 2", ["b", "a"], "bfs"),
        ("Find the parents of node target.", ["target", "b", "a"], "parents"),
        ("Find the parents of node target. " + "hello " * 1000, ["target", "b"], "parents"),
    ]
    return [
        {"prompt": prompt, "answer_nodes": nodes, "problem_type": kind, "prompt_chars": len(prompt)}
        for prompt, nodes, kind in cases
    ]


@pytest.mark.parametrize("cap", [None, 2])
def test_process_workers_preserve_bytes_corrections_and_filtering(monkeypatch, tmp_path, tokenizer_path, cap, capsys):
    source = source_rows()
    monkeypatch.setattr(prepare, "load_dataset", lambda *args, **kwargs: source)
    serial = prepare.prepare(tokenizer_path, cap, tmp_path / "serial.jsonl", num_workers=1)
    serial_summary = capsys.readouterr().out
    parallel = prepare.prepare(tokenizer_path, cap, tmp_path / "parallel.jsonl", num_workers=2)
    assert parallel.read_bytes() == serial.read_bytes()
    assert "cleaned 2 self-parent answers" in serial_summary
    assert "cleaned 2 self-parent answers" in capsys.readouterr().out
    assert source == source_rows()
    rows = [json.loads(line) for line in parallel.read_text().splitlines()]
    if cap is None:
        assert len(rows) == 5
        assert rows[0]["n_tokens"] == 1000
        assert rows[2]["responses_create_params"]["input"][0]["content"] == (
            "Perform a BFS from node root and return only the nodes at exactly depth 2 "
            "(not nodes at intermediate depths)"
        )
        assert json.loads(rows[3]["expected_answer"]) == ["a", "b"]
        assert json.loads(rows[4]["expected_answer"]) == ["b"]
        assert rows[2]["prompt_chars"] == len(source[2]["prompt"])
    else:
        assert len(rows) == 1
    unicode_row = rows[1] if cap is None else rows[0]
    assert unicode_row["n_tokens"] == 2
    assert unicode_row["responses_create_params"]["input"][0]["content"] == "hello é"
    assert json.loads(unicode_row["expected_answer"]) == ["a", "é"]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_invalid_worker_count_is_rejected(workers, tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        prepare.prepare(output_fpath=tmp_path / "output.jsonl", num_workers=workers)


def test_worker_tokenizer_errors_reach_caller(monkeypatch, tmp_path):
    monkeypatch.setattr(prepare, "load_dataset", lambda *args, **kwargs: source_rows())
    with pytest.raises((ValueError, OSError)):
        prepare.prepare("/missing-graphwalks-tokenizer", output_fpath=tmp_path / "out.jsonl", num_workers=2)
