# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden scores generated with the user's original image_metrics.py, not this port."""

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from resources_servers.vision2code.image_metrics import apply_blend_a, image_metrics


CASES = (
    "identical",
    "shift_2",
    "shift_6",
    "color",
    "smaller",
    "larger",
    "blank",
    "both_blank",
    "large",
    "alpha",
    "exif",
)


def write_images(directory: Path, case: str) -> tuple[Path, Path]:
    size = (1600, 1200) if case == "large" else (80, 60)
    reference = Image.new("RGB", size, "white")
    ImageDraw.Draw(reference).rectangle((10, 10, size[0] // 2, size[1] // 2), fill="black")
    candidate = reference.copy()
    if case.startswith("shift_"):
        offset = int(case.split("_")[1])
        candidate = Image.new("RGB", size, "white")
        ImageDraw.Draw(candidate).rectangle((10 + offset, 10, 40 + offset, 30), fill="black")
    elif case == "color":
        ImageDraw.Draw(candidate).rectangle((10, 10, 40, 30), fill="red")
    elif case in ("smaller", "larger"):
        candidate = candidate.crop((0, 0, 30, 25) if case == "smaller" else (0, 0, 100, 90))
    elif case in ("blank", "both_blank"):
        candidate = Image.new("RGB", size, "white")
        if case == "both_blank":
            reference = candidate.copy()
    elif case == "large":
        ImageDraw.Draw(candidate).rectangle((300, 100, 400, 500), fill="blue")
    elif case == "alpha":
        candidate = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(candidate).rectangle((10, 10, 40, 30), fill=(0, 0, 0, 255))
    elif case == "exif":
        candidate = candidate.transpose(Image.Transpose.ROTATE_90)
        candidate.getexif()[274] = 6
    paths = directory / "reference.png", directory / "candidate.png"
    reference.save(paths[0])
    candidate.save(paths[1])
    return paths


@pytest.mark.parametrize("case", CASES)
def test_upstream_golden(tmp_path, case):
    expected = json.loads((Path(__file__).parent / "fixtures" / "image-metrics-golden.json").read_text())[case]
    metrics = image_metrics(*write_images(tmp_path, case))
    assert metrics == pytest.approx(expected["metrics"])
    rating = {"execution_status": "ok", "raw_score_0_to_5": 3.8, "final_rating_0_to_5": 2.5}
    if case == "both_blank":
        with pytest.raises(ValueError, match="weights were not redistributed"):
            apply_blend_a(rating, metrics)
    else:
        blended = apply_blend_a(rating, metrics)
        assert blended["llm_score_0_to_5"] == 3.8
        assert blended["llm_final_rating_0_to_5"] == 2.5
        assert rating["final_rating_0_to_5"] == 2.5
        assert blended == expected["blend"]


@pytest.mark.parametrize("component", ["foreground_rgb_similarity", "foreground_tolerant_f1", "edge_tolerant_f1"])
@pytest.mark.parametrize("value", [None, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_metrics(component, value):
    metrics = dict.fromkeys(
        ("foreground_rgb_similarity", "foreground_tolerant_f1", "edge_tolerant_f1", "deterministic_weighted"), 1.0
    )
    metrics[component] = value
    with pytest.raises(ValueError):
        apply_blend_a({"execution_status": "ok", "raw_score_0_to_5": 5, "final_rating_0_to_5": 5}, metrics)
