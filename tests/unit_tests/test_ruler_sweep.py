# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.ruler import prepare_sweep


def test_prepare_combines_lengths_and_forwards_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def fake_prepare_helper(output_name: str, model: str, length: int, **kwargs) -> Path:
        calls.append((model, length, kwargs))
        output = tmp_path / output_name
        rows = [
            {"responses_create_params": {}, "outputs": ["qa"], "length": length, "subset": "qa_1"},
            {"responses_create_params": {}, "outputs": ["needle"], "length": length, "subset": "niah_single_1"},
        ]
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return output

    monkeypatch.setattr(prepare_sweep, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_sweep, "prepare_helper", fake_prepare_helper)

    output = prepare_sweep.prepare("test-model", [8192, 16384], data_format="default")
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert calls == [
        ("test-model", 8192, {"data_format": "default"}),
        ("test-model", 16384, {"data_format": "default"}),
    ]
    assert [(row["sequence_length"], row["subset"]) for row in rows] == [
        ("8k", "niah_single_1"),
        ("8k", "qa_1"),
        ("16k", "niah_single_1"),
        ("16k", "qa_1"),
    ]
    assert [row["sequence_length_tokens"] for row in rows] == [8192, 8192, 16384, 16384]
    assert [row["source_index"] for row in rows] == [0, 0, 0, 0]
    assert not list(tmp_path.glob(".ruler_sweep_*.jsonl"))


@pytest.mark.parametrize("lengths", [[], [0], [8192, 8192]])
def test_prepare_rejects_invalid_lengths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lengths: list[int]) -> None:
    monkeypatch.setattr(prepare_sweep, "DATA_DIR", tmp_path)

    with pytest.raises(ValueError):
        prepare_sweep.prepare("test-model", lengths)
