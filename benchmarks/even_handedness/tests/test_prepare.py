# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data-preparation tests."""

import csv
import json
from pathlib import Path

import pytest

from benchmarks.even_handedness import prepare as module


def test_convert_row_preserves_pair_and_source_provenance() -> None:
    row = {column: f"value-{column}" for column in module.REQUIRED_COLUMNS}
    row["prompt_a"] = "prompt A"
    row["prompt_b"] = "prompt B"
    converted = module._convert_row(7, row)
    assert converted["id"] == "even_handedness_0007"
    assert converted["prompt_a"] == "prompt A"
    assert converted["prompt_b"] == "prompt B"
    assert converted["responses_create_params"]["input"] == [{"role": "user", "content": "prompt A"}]
    assert converted["prompt_config"] is None
    assert converted["source_revision"] == module.UPSTREAM_REVISION
    assert converted["source_sha256"] == module.UPSTREAM_SOURCE_SHA256


def test_convert_row_rejects_an_empty_prompt() -> None:
    row = {column: "value" for column in module.REQUIRED_COLUMNS}
    row["prompt_a"] = ""
    with pytest.raises(ValueError, match="empty paired prompt"):
        module._convert_row(0, row)


def test_prepare_requires_complete_source_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "output.jsonl"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt_a", "prompt_b"])
        writer.writeheader()
        writer.writerow({"prompt_a": "A", "prompt_b": "B"})
    monkeypatch.setattr(module, "SOURCE_FPATH", source)
    monkeypatch.setattr(module, "OUTPUT_FPATH", output)
    monkeypatch.setattr(module, "_download_source", lambda: None)
    with pytest.raises(ValueError, match="missing columns"):
        module.prepare()


def test_committed_examples_are_the_first_five_public_rows() -> None:
    example_path = Path("resources_servers/even_handedness/data/example.jsonl")
    rows = [json.loads(line) for line in example_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == [f"even_handedness_{index:04d}" for index in range(5)]
    assert all(row["source_sha256"] == module.UPSTREAM_SOURCE_SHA256 for row in rows)
