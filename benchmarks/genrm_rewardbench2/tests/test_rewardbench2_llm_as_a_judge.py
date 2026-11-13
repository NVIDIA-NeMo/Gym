# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM preparation preserves RewardBench 2 comparisons and scoring metadata."""

import json
from pathlib import Path

from benchmarks.genrm_rewardbench2 import prepare_llm_as_a_judge as module
from benchmarks.genrm_rewardbench2.metrics import compute_metrics
from benchmarks.genrm_rewardbench2.prepare import prepare


def test_preparation_preserves_ties_and_swapped_orders(tmp_path: Path, monkeypatch) -> None:
    records = [
        {
            "id": "tied:0",
            "subset": "Ties",
            "prompt": "What is 1 + 1?",
            "chosen": ["2", "Two"],
            "rejected": ["3"],
        },
        {
            "id": "ref:0",
            "subset": "Ties",
            "prompt": "What is 1 + 1?",
            "chosen": ["2"],
            "rejected": ["3"],
        },
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    genrm = prepare(tmp_path / "genrm.jsonl", source_jsonl=source)
    original = genrm.read_bytes()
    monkeypatch.setattr(module, "SOURCE", genrm)
    output = module.prepare(tmp_path / "llm.jsonl")
    before = [json.loads(line) for line in original.splitlines()]
    after = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(before) == len(after) == 8
    for genrm_row, llm_row in zip(before, after, strict=True):
        assert llm_row["id"] == genrm_row["id"]
        assert llm_row["verifier_metadata"] == genrm_row["verifier_metadata"]
        params = llm_row["responses_create_params"]
        assert params["max_output_tokens"] == 32768
        assert "metadata" not in params
        assert len(params["input"]) == 1
        assert params["input"][0]["role"] == "user"
        text = params["input"][0]["content"]
        assert "User: What is 1 + 1?" in text
        assert '"overall"' in text
        assert '"ranking": <1-6>' in text
        for number in (1, 2):
            answer = genrm_row["responses_create_params"]["metadata"][f"response_{number}"]
            assert f"[The Start of Response {number}]\n{answer}" in text
    assert genrm.read_bytes() == original
    assert {row["verifier_metadata"]["flipped"] for row in after} == {False, True}
    assert {row["verifier_metadata"]["label"] for row in after} == {"A>B", "B>A", "A=B"}

    def metrics(rows: list[dict]) -> dict:
        rankings = {"A>B": 1, "B>A": 6, "A=B": 3}
        return compute_metrics(
            [
                [
                    {
                        "verifier_metadata": row["verifier_metadata"],
                        "predicted_ranking": rankings[row["verifier_metadata"]["label"]],
                    }
                ]
                for row in rows
            ]
        )

    assert metrics(after) == metrics(before)
    assert metrics(after)["metrics"]["ties"] > 0.99
