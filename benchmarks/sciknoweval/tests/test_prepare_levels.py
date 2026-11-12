# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.sciknoweval import prepare


def _row(index: int, level: str, domain: str = "Chemistry") -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": f"Question {index}?"}]},
        "verifier_metadata": {
            "id": f"sciknoweval.mcq.{index:05d}",
            "answer_type": "mcq-4-choices",
            "expected_answer": "A",
            "task": "mcq",
            "domain": domain,
            "level": level,
            "subset_for_metrics": level,
        },
    }


ROWS = [_row(0, "L1"), _row(1, "L3"), _row(2, "L5", "Physics"), _row(3, "L2"), _row(4, "L3", "Biology")]


def _written(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_select_levels_accepts_python_and_hydra_spellings() -> None:
    assert prepare.select_levels(None) is None
    assert prepare.select_levels(("L3", "L5")) == ("L3", "L5")
    assert prepare.select_levels("L3,L5") == ("L3", "L5")
    assert prepare.select_levels("[L5, L3]") == ("L5", "L3")


@pytest.mark.parametrize("levels", ["", "L6", ["L3", "l5"], []])
def test_select_levels_rejects_unknown_or_empty_selections(levels) -> None:
    with pytest.raises(ValueError):
        prepare.select_levels(levels)


def test_prepare_keeps_only_the_selected_levels(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prepare, "build_rows", lambda: iter(ROWS))

    output = prepare.prepare(output_path=tmp_path / "subset.jsonl", levels="L3,L5")

    assert [row["verifier_metadata"]["id"] for row in _written(output)] == [
        "sciknoweval.mcq.00001",
        "sciknoweval.mcq.00002",
        "sciknoweval.mcq.00004",
    ]


def test_prepare_without_levels_keeps_every_row(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prepare, "build_rows", lambda: iter(ROWS))

    output = prepare.prepare(output_path=tmp_path / "full.jsonl")

    assert len(_written(output)) == len(ROWS)


def test_prepare_refuses_to_filter_an_explicit_input_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit input"):
        prepare.prepare(input_path=tmp_path / "in.jsonl", output_path=tmp_path / "out.jsonl", levels="L3")
