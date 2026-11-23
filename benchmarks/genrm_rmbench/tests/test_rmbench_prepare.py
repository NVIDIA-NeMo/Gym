# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RM-Bench dataset preparation tests."""

import json
from pathlib import Path

import pytest

from benchmarks.genrm_rmbench.prepare import prepare


def test_rmbench_style_combinations_and_response_orders(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "prompt": "Question",
                    "domain": "math",
                    "chosen": ["A0", "A1", "A2"],
                    "rejected": ["B0", "B1", "B2"],
                }
            ]
        )
    )
    output = tmp_path / "prepared.jsonl"
    assert prepare(str(source), str(output)) == output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len({row["id"] for row in rows}) == 18
    for original, swapped in zip(rows[::2], rows[1::2], strict=True):
        first = original["responses_create_params"]["metadata"]
        second = swapped["responses_create_params"]["metadata"]
        assert first["response_1"] == second["response_2"]
        assert first["response_2"] == second["response_1"]
        assert original["verifier_metadata"]["correct"] == "first"
        assert swapped["verifier_metadata"]["correct"] == "second"
        assert (
            original["verifier_metadata"]["response_1_style_value"]
            == swapped["verifier_metadata"]["response_2_style_value"]
        )


@pytest.mark.parametrize("same_path", [False, True])
def test_preparation_preserves_existing_files(tmp_path: Path, same_path: bool) -> None:
    source = tmp_path / "source.json"
    row = {"id": 1, "domain": "chat", "prompt": "Q", "chosen": ["A"] * 3, "rejected": ["B"] * 3}
    source.write_text(json.dumps([row, row | {"chosen": []}]))
    original = source.read_text()
    output = source if same_path else tmp_path / "prepared.jsonl"
    if not same_path:
        output.write_text("existing dataset")
    with pytest.raises(ValueError):
        prepare(source, output)
    assert source.read_text() == original
    assert output.read_text() == (original if same_path else "existing dataset")
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        ["source.json"] if same_path else ["prepared.jsonl", "source.json"]
    )
