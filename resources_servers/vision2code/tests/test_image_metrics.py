# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden scores generated from the upstream source pinned in fixtures/provenance.json."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from resources_servers.vision2code.image_metrics import _background_and_overlap, apply_blend_a, image_metrics


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


@pytest.mark.parametrize("caps, expected_judge", [([], 4.96), ([4.5, 2.5], 2.5), ([5.0], 4.96)])
def test_capped_blend_preserves_precision_and_uses_strictest_cap(caps, expected_judge):
    metrics = dict.fromkeys(
        ("foreground_rgb_similarity", "foreground_tolerant_f1", "edge_tolerant_f1", "deterministic_weighted"), 1.0
    )
    rating = {
        "execution_status": "ok",
        "raw_score_0_to_5": 4.96,
        "final_rating_0_to_5": round(min([4.96, *caps]), 1),
        "applied_caps": [{"cap_to": cap} for cap in caps],
    }
    result = apply_blend_a(rating, metrics, apply_judge_caps=True)
    assert result["llm_score_0_to_5"] == expected_judge
    assert result["final_rating_0_to_5"] == pytest.approx(2.5 + 0.5 * expected_judge)
    assert result["deterministic_scores"] == metrics
    assert rating["raw_score_0_to_5"] == 4.96


def test_capped_blend_render_failure_stays_zero():
    result = apply_blend_a(
        {"execution_status": "code_error", "raw_score_0_to_5": 0, "final_rating_0_to_5": 0},
        apply_judge_caps=True,
    )
    assert result["final_rating_0_to_5"] == 0
    assert result["scoring_method"] == "blend_a_capped"


def test_background_histogram_ties_and_even_medians():
    # Both bins occur four times: the lower bin wins, with independent channel medians.
    pixels = np.array([[[0, 1, 7], [1, 3, 6], [6, 4, 1], [7, 6, 0]], [[255, 255, 255]] * 4], dtype=np.uint8)
    image = Image.fromarray(pixels)
    background, overlap = _background_and_overlap(image, image)
    np.testing.assert_array_equal(background, [3.5, 3.5, 3.5])
    assert overlap == 1


@pytest.mark.parametrize("candidate_size", [(1000, 600), (700, 500), (1100, 700)])
def test_chunked_statistics_match_full_resolution_calculation(candidate_size):
    rng = np.random.default_rng(137)
    pixels = rng.integers(0, 256, (600, 1000, 3), dtype=np.uint8)
    # Give one bin many distinct values, including across chunk boundaries.
    pixels[::2] %= 8
    reference = Image.fromarray(pixels)
    candidate = reference.crop((0, 0, *candidate_size))
    actual_background, actual_overlap = _background_and_overlap(reference, candidate)
    bins = pixels.astype(np.int32) // 8
    labels = bins[..., 0] * 1024 + bins[..., 1] * 32 + bins[..., 2]
    peak = np.bincount(labels.ravel(), minlength=32768).argmax()
    np.testing.assert_array_equal(actual_background, np.median(pixels[labels == peak], axis=0))
    other = np.asarray(candidate)
    h, w = min(other.shape[0], pixels.shape[0]), min(other.shape[1], pixels.shape[1])
    assert actual_overlap == np.all(pixels[:h, :w] == other[:h, :w], axis=2).mean()
