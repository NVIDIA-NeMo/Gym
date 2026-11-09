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
ORIGINAL = json.loads((FIXTURES / "original-doc-eee.json").read_text())["rubrics"]
VARIANTS = {"DocVQA": "DocVQA-v2", "EEE-Bench": "EEE-Bench-v2"}


def historical_name(name):
    return VARIANTS.get(name, name)


def rename_dataset(value, name, replacement):
    return json.loads(json.dumps(value).replace(name, replacement))


@pytest.mark.parametrize("case", CASES, ids=[f"{c['dataset']}-{i}" for i, c in enumerate(CASES)])
def test_upstream_golden(case):
    actual = evaluation.aggregate_rating(
        case["parsed"], dataset_name=historical_name(case["dataset"]), candidate_inspection=case["inspection"]
    )
    expected = rename_dataset(case["expected"], case["dataset"], historical_name(case["dataset"]))
    assert {key: actual[key] for key in expected} == expected


def test_definitions_and_system_prompt():
    definitions = json.loads((FIXTURES / "definitions.json").read_text())
    for name, variant in VARIANTS.items():
        definitions[variant] = rename_dataset(definitions[name], name, variant)
        definitions[name] = ORIGINAL[name]["definition"]
    assert {**rubrics.DATASET_RUBRICS, "generic": generic_rubric.GENERIC_RUBRIC} == definitions
    assert rubrics.RATER_SYSTEM_PROMPT == (FIXTURES / "system-prompt.txt").read_text()
    assert set(definitions) == set(get_args(evaluation.RubricName))
    assert len(definitions) == 19
    assert generic_rubric.GENERIC_RATER_SYSTEM_PROMPT == (FIXTURES / "generic-system-prompt.txt").read_text()


@pytest.mark.parametrize("name", get_args(evaluation.RubricName))
def test_failure_matches_upstream(name):
    if name in ORIGINAL:
        expected = ORIGINAL[name]["failure"] | {"rubric_version": rubrics.RUBRIC_VERSION}
    else:
        source_name = next((key for key, value in VARIANTS.items() if value == name), name)
        expected = json.loads((FIXTURES / "failure-golden.json").read_text())[source_name]
        expected = rename_dataset(expected, source_name, name)
    assert evaluation.build_render_failure_rating("code_error", dataset_name=name) == expected


def test_user_prompts(tmp_path):
    path = tmp_path / "ref.png"
    Image.new("RGB", (40, 40), "white").save(path)
    prompts = json.loads((FIXTURES / "user-prompts.json").read_text())
    for name, variant in VARIANTS.items():
        prompts[variant] = rename_dataset(prompts[name], name, variant)
        # Only the existing Gym-wide rubric version label differs from the original prompt.
        prompts[name] = rename_dataset(
            ORIGINAL[name]["user_prompt"], "recreation_rubric_v4_decimal_per_dataset", rubrics.RUBRIC_VERSION
        )
    for name, expected in prompts.items():
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


# Exercise both scorers; dataset-specific weights and caps are covered by the golden cases.
@pytest.mark.parametrize("name", ["ChartQA", "generic"])
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


@pytest.mark.parametrize("name", ["DocVQA", "EEE-Bench"])
def test_original_rubric_weights_and_caps(name):
    for case in ORIGINAL[name]["golden"]:
        actual = evaluation.aggregate_rating(case["parsed"], dataset_name=name, candidate_inspection={})
        assert {key: actual[key] for key in case["expected"]} == case["expected"]


@pytest.mark.parametrize("image_format", ["TIFF", "BMP"])
def test_judge_image_encoding_converts_raster_formats_without_changing_artifact(tmp_path, image_format):
    import base64
    from io import BytesIO

    path = tmp_path / "output.png"
    Image.new("RGB", (80, 60), "red").save(path, format=image_format)
    original = path.read_bytes()
    url = rubrics.encode_image_to_data_url(path)
    assert url.startswith("data:image/png;base64,")
    with Image.open(BytesIO(base64.b64decode(url.split(",")[1]))) as decoded:
        assert decoded.size == (80, 60)
        assert decoded.getpixel((0, 0)) == (255, 0, 0, 255)
    assert path.read_bytes() == original
