# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check candidate ordering, image placement, and label separation."""

import json
from io import BytesIO

import pytest


pytest.importorskip("pyarrow")
Image = pytest.importorskip("PIL.Image")

from benchmarks.genrm_mmrb2.build_validation import convert, interleave, prepare


@pytest.fixture(autouse=True)
def mock_upstream_prompt(monkeypatch):
    from unittest.mock import Mock

    prompt = Mock(return_value="Synthetic judging instructions")
    monkeypatch.setattr("benchmarks.genrm_mmrb2.build_validation.load_prompt", prompt)
    return prompt


def image():
    stream = BytesIO()
    Image.new("RGB", (1, 1)).save(stream, format="PNG")
    return {"bytes": stream.getvalue(), "path": None}


@pytest.mark.parametrize("genrm", [False, True])
def test_orders_and_content(genrm, mock_upstream_prompt):
    row = {
        "pair_id": "pair",
        "chosen": "A",
        "prompt_source": "test",
        "prompt_text": "Look <image_0>",
        "prompt_images": [image()],
        "response_a_text": "Correct",
        "response_a_images": [],
        "response_b_text": "Wrong",
        "response_b_images": [],
    }
    forward, reverse = list(convert(row, genrm=genrm))
    assert forward["verifier_metadata"]["expected_label"] == "A"
    assert reverse["verifier_metadata"]["expected_label"] == "B"
    assert forward["verifier_metadata"]["pair_type"] == "chosen_text__rejected_text"
    assert forward["verifier_metadata"]["task"] == "reasoning"
    assert forward["verifier_metadata"]["response_modality"] == "text_only"
    params = forward["responses_create_params"]
    if genrm:
        mock_upstream_prompt.assert_not_called()
        assert params["metadata"]["response_1"] == "Correct"
        assert reverse["responses_create_params"]["metadata"]["response_1"] == "Wrong"
        assert forward["verifier_metadata"]["ground_truth_label"] == 3
        assert reverse["verifier_metadata"]["ground_truth_label"] == 4
    else:
        mock_upstream_prompt.assert_called_once()
        content = params["input"][0]["content"]
        assert content[0]["text"] == "Synthetic judging instructions"
        assert sum(part["type"] == "input_image" for part in content) == 1
        assert "verifier_metadata" not in json.dumps(params)


def test_no_silent_image_loss():
    with pytest.raises(ValueError, match="without text placeholders"):
        interleave("Question", [image()])
    with pytest.raises(ValueError, match="no corresponding image"):
        interleave("<image_0>", [])


@pytest.mark.parametrize(("task", "modality"), [("edit", "text_only"), ("reasoning", "all")])
def test_unsupported_scope_fails_before_reading(tmp_path, task, modality):
    with pytest.raises(ValueError, match="Only task=reasoning"):
        prepare(tmp_path, tmp_path / "out.jsonl", task=task, response_modality=modality)


def test_image_candidate_is_rejected():
    with pytest.raises(ValueError, match="text-only candidate"):
        list(convert({"response_a_images": [image()], "response_b_images": []}))


@pytest.mark.parametrize("genrm", [False, True])
def test_full_source_filters_to_350_pairs(tmp_path, genrm):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "pair_id": str(i),
            "chosen": "A",
            "prompt_source": "test",
            "prompt_text": "Question <image_0>",
            "prompt_images": [image()],
            "response_a_text": "Correct",
            "response_a_images": [],
            "response_b_text": "Wrong",
            "response_b_images": [] if i < 350 else [image()],
        }
        for i in range(1000)
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "test-0.parquet")
    output = tmp_path / "out.jsonl"
    prepare(tmp_path, output, genrm=genrm)
    tasks = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(tasks) == 700
    assert len({r["verifier_metadata"]["pair_id"] for r in tasks}) == 350
    assert {r["verifier_metadata"]["judge_mode"] for r in tasks} == {"genrm" if genrm else "llm_as_a_judge"}
    assert [r["verifier_metadata"]["order"] for r in tasks[:2]] == ["forward", "reverse"]

    previous = output.read_bytes()
    rows[349]["response_b_images"] = [image()]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "test-0.parquet")
    with pytest.raises(ValueError, match="Expected 350"):
        prepare(tmp_path, output, genrm=genrm)
    assert output.read_bytes() == previous
