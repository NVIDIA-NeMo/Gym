# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check source-side MMRB v1 conversion without downloading data or running a judge."""

import base64
import json

import pytest

from benchmarks.genrm_mmrb1.build_validation import convert, load_prompt, prepare, subcategory


@pytest.fixture
def source(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "image.png").write_bytes(b"image bytes")
    (tmp_path / "judge_prompt.yaml").write_text('judge_prompt: "Question: {question} A: {answer_a} B: {answer_b}"\n')
    return tmp_path


def row(**updates):
    return {
        "ID": "pairs_1",
        "Category": "safety",
        "Text": "Describe the image.",
        "Image": "image.png",
        "Output1": "First",
        "Output2": "Second",
        "Better": "Output2",
    } | updates


def test_regular_and_genrm(source):
    template = load_prompt(source / "judge_prompt.yaml")
    regular = convert(row(), source / "data", template)
    parts = regular["responses_create_params"]["input"][0]["content"]
    assert parts[0]["text"] == "Question: Describe the image. A: First B: Second"
    assert base64.b64decode(parts[1]["image_url"].split(",")[1]) == b"image bytes"
    assert regular["verifier_metadata"]["expected_label"] == "B"
    assert regular["verifier_metadata"]["subcategory"] == "safety/bias"
    genrm = convert(row(), source / "data", "unused", genrm=True)
    assert genrm["responses_create_params"]["input"][0]["content"][0]["text"] == row()["Text"]
    assert genrm["responses_create_params"]["metadata"] == {"response_1": "First", "response_2": "Second"}
    assert genrm["verifier_metadata"]["ground_truth_label"] == 4
    assert (
        convert(row(Better="Output1"), source / "data", "unused", genrm=True)["verifier_metadata"][
            "ground_truth_label"
        ]
        == 3
    )


@pytest.mark.parametrize(
    ("category", "id", "expected"),
    [
        ("reasoning", "mathvista_1", "reasoning/math"),
        ("reasoning", "EMMA-Coding_1", "reasoning/coding"),
        ("safety", "other_1", "safety/toxicity"),
        ("knowledge", "mmmu_1", "knowledge"),
    ],
)
def test_subcategory(category, id, expected):
    assert subcategory(row(Category=category, ID=id)) == expected


def test_preparation_and_missing_image_preserves_output(source):
    items = [row()]
    (source / "data/all_data.json").write_text(json.dumps(items))
    (source / "data/all_data_release.json").write_text(json.dumps(items))
    output = source / "result.jsonl"
    prepare(source, output, prompt_config=source / "judge_prompt.yaml")
    previous = output.read_bytes()
    assert len(output.read_text().splitlines()) == 1
    (source / "data/image.png").unlink()
    with pytest.raises(FileNotFoundError):
        prepare(source, output, prompt_config=source / "judge_prompt.yaml")
    assert output.read_bytes() == previous


def test_reject_incomplete_download(source):
    with pytest.raises(FileNotFoundError, match="Finish"):
        prepare(source, source / "result.jsonl")
    (source / "data/all_data.json").write_text(json.dumps([row()]))
    (source / "data/all_data_release.json").write_text(json.dumps([row(ID="other")]))
    with pytest.raises(ValueError, match="full public release"):
        prepare(source, source / "result.jsonl")


def test_repeated_question_ids_are_preserved(source):
    items = [row(), row(Better="Output1", Output1="Another candidate")]
    (source / "data/all_data.json").write_text(json.dumps(items))
    (source / "data/all_data_release.json").write_text(json.dumps(items))
    output = source / "result.jsonl"
    prepare(source, output, prompt_config=source / "judge_prompt.yaml")
    tasks = [json.loads(line) for line in output.read_text().splitlines()]
    assert [task["id"] for task in tasks] == ["pairs_1:0", "pairs_1:1"]
    assert [task["verifier_metadata"]["expected_label"] for task in tasks] == ["B", "A"]
    assert all(task["verifier_metadata"]["original_id"] == "pairs_1" for task in tasks)


@pytest.mark.parametrize("updates", [{"Better": "tie"}, {"Text": ""}, {"Image": "../outside.png"}])
def test_reject_bad_row(source, updates):
    with pytest.raises(ValueError):
        convert(row(**updates), source / "data", load_prompt(source / "judge_prompt.yaml"))
