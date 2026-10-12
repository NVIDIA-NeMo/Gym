# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from typing import get_args

import pytest
from PIL import Image, ImageDraw

from resources_servers.vision2code import evaluation, generic_rubric, rubrics


FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "golden.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[f"{c['dataset']}-{i}" for i, c in enumerate(CASES)])
def test_upstream_golden(case):
    actual = evaluation.aggregate_rating(
        case["parsed"], dataset_name=case["dataset"], candidate_inspection=case["inspection"]
    )
    assert {key: actual[key] for key in case["expected"]} == case["expected"]


def test_definitions_and_system_prompt():
    definitions = json.loads((FIXTURES / "definitions.json").read_text())
    assert {**rubrics.DATASET_RUBRICS, "generic": generic_rubric.GENERIC_RUBRIC} == definitions
    assert rubrics.RATER_SYSTEM_PROMPT == (FIXTURES / "system-prompt.txt").read_text()
    assert set(definitions) == set(get_args(evaluation.RubricName))
    assert len(definitions) == 17
    assert generic_rubric.GENERIC_RATER_SYSTEM_PROMPT == (FIXTURES / "generic-system-prompt.txt").read_text()


@pytest.mark.parametrize("name", get_args(evaluation.RubricName))
def test_failure_matches_upstream(name):
    expected = json.loads((FIXTURES / "failure-golden.json").read_text())[name]
    assert evaluation.build_render_failure_rating("code_error", dataset_name=name) == expected


def test_user_prompts(tmp_path):
    path = tmp_path / "ref.png"
    Image.new("RGB", (40, 40), "white").save(path)
    for name, expected in json.loads((FIXTURES / "user-prompts.json").read_text()).items():
        actual = evaluation.build_rater_user_content(
            source_image_path=path, rendered_image_path=path, dataset_name=name
        )
        for part in actual:
            if part["type"] == "image_url":
                assert part["image_url"]["url"].startswith("data:image/png;base64,")
                part["image_url"]["url"] = "IMAGE_DATA_URL"
        assert actual == expected


@pytest.mark.parametrize("filename", ["reference", "reference.jpg"])
def test_judge_image_mime_matches_contents(tmp_path, filename):
    path = tmp_path / filename
    Image.new("RGB", (40, 40), "white").save(path, format="PNG")
    assert rubrics.encode_image_to_data_url(path).startswith("data:image/png;base64,")


def test_sparse_is_diagnostic_and_blank_is_capped(tmp_path):
    path = tmp_path / "candidate.png"
    im = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(im).line((40, 200, 360, 200), fill="black", width=2)
    im.save(path)
    sparse = rubrics.inspect_candidate_image(path)
    assert sparse["checks_triggered"] == ["mostly_empty_canvas"]
    assert not sparse["is_blank_or_degenerate"]
    Image.new("RGB", (400, 400), "white").save(path)
    blank = rubrics.inspect_candidate_image(path)
    assert blank["is_blank_or_degenerate"]
    for name, rubric in rubrics.DATASET_RUBRICS.items():
        parsed = {"category_scores": {c["id"]: 5 for c in rubric["categories"]}, "rationales": {}}
        assert (
            rubrics.aggregate_rating_v2(parsed, dataset_name=name, candidate_inspection=sparse)["final_rating_0_to_5"]
            == 5
        )
        assert (
            rubrics.aggregate_rating_v2(parsed, dataset_name=name, candidate_inspection=blank)["final_rating_0_to_5"]
            == 0.5
        )


@pytest.mark.parametrize("name", get_args(evaluation.RubricName))
@pytest.mark.parametrize("score", [True, "NaN", "Infinity", -1, 5.1, None])
def test_invalid_scores(name, score):
    definition = generic_rubric.GENERIC_RUBRIC if name == "generic" else rubrics.DATASET_RUBRICS[name]
    parsed = {
        "category_scores": {c["id"]: score for c in definition["categories"]},
        "rationales": {},
    }
    with pytest.raises(ValueError):
        evaluation.aggregate_rating(parsed, dataset_name=name, candidate_inspection={})


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "<think>reasoning</think>{}", "Here is the result: {}"])
def test_json_parser(wrapper):
    obj = {"category_scores": {"x": 3}}
    assert rubrics.parse_json_object(wrapper.format(json.dumps(obj))) == obj
