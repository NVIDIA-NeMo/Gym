# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JudgeBench dataset preparation tests."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.genrm_judgebench.prepare import REVISION, convert_rows, prepare


def judge_pair() -> dict:
    return {
        "pair_id": "p",
        "original_id": "q",
        "source": "mmlu-pro-math",
        "question": "Which answer is correct?",
        "response_A": "The answer is 2.",
        "response_B": "The answer is 3.",
        "label": "A>B",
    }


def test_judgebench_native_request_and_flipped_label() -> None:
    first, flipped = convert_rows([judge_pair()])
    assert first["responses_create_params"]["input"] == [{"role": "user", "content": "Which answer is correct?"}]
    assert first["responses_create_params"]["metadata"] == {
        "response_1": "The answer is 2.",
        "response_2": "The answer is 3.",
    }
    assert first["verifier_metadata"]["label"] == "A>B"
    assert flipped["verifier_metadata"]["label"] == "B>A"
    assert flipped["responses_create_params"]["metadata"]["response_1"] == "The answer is 3."
    assert "label" not in first["responses_create_params"]["metadata"]
    assert "principle" not in first["responses_create_params"]["metadata"]  # Use template default.


def test_existing_judgebench_flip_is_not_duplicated() -> None:
    pair = judge_pair()
    flipped = pair | {
        "pair_id": "p-flipped",
        "response_A": pair["response_B"],
        "response_B": pair["response_A"],
        "label": "B>A",
    }
    assert len(list(convert_rows([pair, flipped]))) == 2
    with pytest.raises(ValueError, match="orientations"):
        list(convert_rows([pair, flipped, flipped]))


def test_local_preparation(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(judge_pair()) + "\n")
    output = prepare(str(tmp_path / "data/output.jsonl"), source_jsonl=str(source))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["verifier_metadata"]["label"] == "B>A"


def test_huggingface_preparation_split(tmp_path: Path) -> None:
    source = tmp_path / "claude.jsonl"
    source.write_text(json.dumps(judge_pair()) + "\n")
    with patch("huggingface_hub.hf_hub_download", return_value=str(source)) as download:
        output = prepare(tmp_path / "data/output.jsonl", split="claude")
    download.assert_called_once_with(
        "ScalerLab/JudgeBench", repo_type="dataset", filename="data/claude-00000-of-00001.jsonl", revision=REVISION
    )
    assert len(output.read_text().splitlines()) == 2


@pytest.mark.parametrize("defect", ["duplicate", "responses", "label"])
def test_invalid_existing_orientations(defect: str) -> None:
    pair = judge_pair()
    flipped = pair | {
        "pair_id": "p-flipped",
        "response_A": pair["response_B"],
        "response_B": pair["response_A"],
        "label": "B>A",
    }
    if defect == "duplicate":
        flipped = dict(pair)
    elif defect == "responses":
        flipped["response_A"] = pair["response_A"]
    else:
        flipped["label"] = pair["label"]
    with pytest.raises(ValueError, match="opposite response orders"):
        list(convert_rows([pair, flipped]))


@pytest.mark.parametrize("same_path", [False, True])
def test_preparation_preserves_existing_files(tmp_path: Path, same_path: bool) -> None:
    source = tmp_path / "source.jsonl"
    pair = judge_pair()
    source.write_text("\n".join(json.dumps(row) for row in (pair, pair | {"pair_id": "z"}, pair | {"pair_id": "z"})))
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
