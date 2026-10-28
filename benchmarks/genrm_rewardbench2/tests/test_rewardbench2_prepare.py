# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RewardBench 2 dataset preparation tests."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.genrm_rewardbench2.prepare import COLUMNS, PARQUET_FPATH, REVISION, convert_rows, prepare


def record(**overrides: object) -> dict:
    return {
        "id": "7",
        "prompt": "Which answer is correct?",
        "chosen": ["The answer is 2."],
        "rejected": ["The answer is 3.", "The answer is 4.", "The answer is 5."],
        "subset": "Math",
    } | overrides


def test_best_of_four_record_becomes_three_comparisons_in_both_orders() -> None:
    rows = list(convert_rows([record()]))
    assert [row["id"] for row in rows] == [
        "Math:7:0:0",
        "Math:7:0:0-flipped",
        "Math:7:0:1",
        "Math:7:0:1-flipped",
        "Math:7:0:2",
        "Math:7:0:2-flipped",
    ]
    first, flipped = rows[0], rows[1]
    assert first["responses_create_params"]["input"] == [{"role": "user", "content": "Which answer is correct?"}]
    assert first["responses_create_params"]["metadata"] == {
        "response_1": "The answer is 2.",
        "response_2": "The answer is 3.",
    }
    assert first["verifier_metadata"] == {
        "pair_id": "Math:7:0:0",
        "example_id": "Math:7",
        "subset": "Math",
        "comparisons": 3,
        "correct_pairs": 0,
        "label": "A>B",
        "flipped": False,
    }
    assert flipped["verifier_metadata"]["label"] == "B>A"
    assert flipped["verifier_metadata"]["flipped"] is True
    assert flipped["responses_create_params"]["metadata"] == {
        "response_1": "The answer is 3.",
        "response_2": "The answer is 2.",
    }
    assert "label" not in first["responses_create_params"]["metadata"]
    assert "principle" not in first["responses_create_params"]["metadata"]  # Use template default.


def test_ties_record_pairs_every_correct_with_every_incorrect() -> None:
    ties = record(id="tied:0", subset="Ties", chosen=["1", "2"], rejected=["0", "11", "42"])
    rows = list(convert_rows([ties]))
    rejected_rows = [row for row in rows if row["verifier_metadata"]["label"] != "A=B"]
    assert len(rejected_rows) == 2 * 3 * 2
    assert rejected_rows[-2]["id"] == "Ties:tied:0:1:2"
    assert {row["verifier_metadata"]["comparisons"] for row in rows} == {6}
    assert rejected_rows[-2]["responses_create_params"]["metadata"] == {"response_1": "2", "response_2": "42"}


def test_ties_record_also_pairs_correct_answers_with_each_other() -> None:
    # The Ties score needs the spread among correct answers, which only these comparisons measure.
    ties = record(id="tied:0", subset="Ties", chosen=["1", "2", "3"], rejected=["0", "11"])
    rows = list(convert_rows([ties]))
    correct_rows = [row for row in rows if row["verifier_metadata"]["label"] == "A=B"]
    assert [row["id"] for row in correct_rows] == [
        "Ties:tied:0:c0:c1",
        "Ties:tied:0:c0:c1-flipped",
        "Ties:tied:0:c0:c2",
        "Ties:tied:0:c0:c2-flipped",
        "Ties:tied:0:c1:c2",
        "Ties:tied:0:c1:c2-flipped",
    ]
    assert correct_rows[0]["responses_create_params"]["metadata"] == {"response_1": "1", "response_2": "2"}
    assert correct_rows[1]["responses_create_params"]["metadata"] == {"response_1": "2", "response_2": "1"}
    # Both orders keep the A=B label: neither response is the better one.
    assert {row["verifier_metadata"]["label"] for row in correct_rows} == {"A=B"}
    assert {row["verifier_metadata"]["correct_pairs"] for row in rows} == {3}
    assert len(rows) == 2 * (3 * 2 + 3)


def test_single_chosen_response_yields_no_correct_answer_pairs() -> None:
    rows = list(convert_rows([record(id="ref:0", subset="Ties", rejected=["a", "b"])]))
    assert {row["verifier_metadata"]["correct_pairs"] for row in rows} == {0}
    assert not [row for row in rows if row["verifier_metadata"]["label"] == "A=B"]


def test_ids_are_qualified_by_subset() -> None:
    rows = list(convert_rows([record(subset="Factuality"), record(subset="Precise IF")]))
    assert len(rows) == 12
    assert {row["verifier_metadata"]["example_id"] for row in rows} == {"Factuality:7", "Precise IF:7"}


@pytest.mark.parametrize("defect", ["chosen", "rejected", "prompt"])
def test_incomplete_records_are_rejected(defect: str) -> None:
    with pytest.raises(ValueError, match="RewardBench 2 record Math:7"):
        list(convert_rows([record(**{defect: [] if defect != "prompt" else ""})]))


def test_duplicate_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        list(convert_rows([record(), record()]))


def test_local_preparation(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(record()) + "\n")
    output = prepare(str(tmp_path / "data/output.jsonl"), source_jsonl=str(source))
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert rows[1]["verifier_metadata"]["label"] == "B>A"


def test_huggingface_preparation(tmp_path: Path) -> None:
    table = type("Table", (), {"to_pylist": lambda self: [record()]})()
    with (
        patch("huggingface_hub.hf_hub_download", return_value="cached.parquet") as download,
        patch("pyarrow.parquet.read_table", return_value=table) as read_table,
    ):
        output = prepare(tmp_path / "data/output.jsonl")
    download.assert_called_once_with(
        "allenai/reward-bench-2", repo_type="dataset", filename=PARQUET_FPATH, revision=REVISION
    )
    read_table.assert_called_once_with("cached.parquet", columns=COLUMNS)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 6


@pytest.mark.parametrize("same_path", [False, True])
def test_preparation_preserves_existing_files(tmp_path: Path, same_path: bool) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in (record(), record())))
    original = source.read_text()
    output = source if same_path else tmp_path / "prepared.jsonl"
    if not same_path:
        output.write_text("existing dataset")
    with pytest.raises(ValueError):
        prepare(output, source_jsonl=source)
    assert source.read_text() == original
    assert output.read_text() == (original if same_path else "existing dataset")
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        ["source.jsonl"] if same_path else ["prepared.jsonl", "source.jsonl"]
    )
