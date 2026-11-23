# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from resources_servers.image_tools.app import (
    FailureCode,
    ImageToolsPivotResourcesServerConfig,
    _strip_ignored,
    canonical_tool_name,
    compute_argument_score,
    extract_expected_action,
    tool_family,
    verify_target_match,
    verify_tool_name_match,
)


TOL = 1e-6


def _zoom(bbox=(100, 100, 300, 300), img_idx=0, factor=3, label="x"):
    return {
        "name": "image_zoom_in_tool",
        "arguments": {
            "bbox_2d": list(bbox),
            "factor": factor,
            "img_idx": img_idx,
            "label": label,
        },
    }


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("image_zoom_in_tool", "bbox"),
        ("image_crop_tool", "bbox"),
        ("color_at_tool", "point"),
        ("find_color_tool", "color"),
        ("count_objects_tool", "color"),
        ("image_diff_tool", "pair"),
        ("image_overlay_tool", "pair"),
        ("image_side_by_side_tool", "multi"),
        ("image_rotate_tool", "scalar"),
        ("image_flip_tool", "scalar"),
        ("nope_tool", "unknown"),
    ],
)
def test_tool_family(name, expected):
    assert tool_family(name) == expected


# --------------------------------------------------------------------------
# Level 0
# --------------------------------------------------------------------------


def test_tool_name_match():
    assert verify_tool_name_match("image_zoom_in_tool", "image_zoom_in_tool")
    assert not verify_tool_name_match("color_at_tool", "image_zoom_in_tool")


def test_zoom_and_crop_are_the_same_decision():
    """crop IS zoom at factor 1, and `factor` is already ignored."""
    assert verify_tool_name_match("image_crop_tool", "image_zoom_in_tool")
    assert verify_tool_name_match("image_zoom_in_tool", "image_crop_tool")


def test_zoom_crop_unification_can_be_disabled():
    assert not verify_tool_name_match("image_crop_tool", "image_zoom_in_tool", False)
    assert verify_tool_name_match("image_zoom_in_tool", "image_zoom_in_tool", False)


@pytest.mark.parametrize(
    "a,b",
    [
        # Same tool_family(), but genuinely different decisions - these must
        # NOT be unified just because they share an argument-comparison path.
        ("find_color_tool", "count_objects_tool"),
        ("image_rotate_tool", "image_flip_tool"),
        ("image_diff_tool", "image_overlay_tool"),
    ],
)
def test_same_family_is_not_the_same_decision(a, b):
    assert not verify_tool_name_match(a, b)


# --------------------------------------------------------------------------
# Level 1
# --------------------------------------------------------------------------


def test_target_match_img_idx():
    e = _zoom(img_idx=0)["arguments"]
    assert verify_target_match("bbox", _zoom(img_idx=0)["arguments"], e)
    assert not verify_target_match("bbox", _zoom(img_idx=1)["arguments"], e)


def test_target_match_missing_index_defers():
    assert verify_target_match("bbox", {"bbox_2d": [0, 0, 1, 1]}, {"bbox_2d": [0, 0, 1, 1]})


def test_target_match_pair_and_multi():
    e = {"img_idx_a": 0, "img_idx_b": 1}
    assert verify_target_match("pair", {"img_idx_a": 0, "img_idx_b": 1}, e)
    assert not verify_target_match("pair", {"img_idx_a": 0, "img_idx_b": 2}, e)

    e = {"img_indices": [0, 1]}
    assert verify_target_match("multi", {"img_indices": [1, 0]}, e)  # order-insensitive
    assert not verify_target_match("multi", {"img_indices": [0, 2]}, e)


# --------------------------------------------------------------------------
# Level 2
# --------------------------------------------------------------------------


def test_bbox_iou_scoring():
    e = _zoom()["arguments"]
    same, _ = compute_argument_score("bbox", _zoom()["arguments"], e, TOL)
    assert same == pytest.approx(1.0)

    near, _ = compute_argument_score("bbox", _zoom(bbox=(110, 110, 310, 310))["arguments"], e, TOL)
    assert 0.5 < near < 1.0

    far, _ = compute_argument_score("bbox", _zoom(bbox=(600, 600, 800, 800))["arguments"], e, TOL)
    assert far == pytest.approx(0.0)


def test_label_and_factor_are_ignored():
    """A pivot is about the region, not the caption or the magnification."""
    e = _strip_ignored(_zoom(label="ground truth caption")["arguments"])
    r = _strip_ignored(_zoom(label="totally different", factor=6)["arguments"])
    score, _ = compute_argument_score("bbox", r, e, TOL)
    assert score == pytest.approx(1.0)


def test_bbox_unparseable():
    e = _zoom()["arguments"]
    score, detail = compute_argument_score("bbox", {"bbox_2d": "garbage"}, e, TOL)
    assert score == 0.0
    assert detail == "rollout_bbox_unparseable"


def test_point_scoring():
    e = {"point_2d": [500, 500], "img_idx": 0}
    exact, _ = compute_argument_score("point", {"point_2d": [500, 500]}, e, TOL)
    assert exact == pytest.approx(1.0)
    near, _ = compute_argument_score("point", {"point_2d": [510, 500]}, e, TOL)
    assert 0.9 < near < 1.0


def test_rotation_is_modular():
    e = {"degrees": 350, "img_idx": 0}
    same, _ = compute_argument_score("scalar", {"degrees": -10}, e, TOL)
    assert same == pytest.approx(1.0)
    diff, _ = compute_argument_score("scalar", {"degrees": 90}, e, TOL)
    assert diff == pytest.approx(0.0)


def test_flip_axis_string_compare():
    e = {"axis": "horizontal", "img_idx": 0}
    assert compute_argument_score("scalar", {"axis": "Horizontal"}, e, TOL)[0] == pytest.approx(1.0)
    assert compute_argument_score("scalar", {"axis": "vertical"}, e, TOL)[0] == pytest.approx(0.0)


def test_pair_alpha():
    e = {"img_idx_a": 0, "img_idx_b": 1, "alpha": 0.5}
    assert compute_argument_score("pair", dict(e), e, TOL)[0] == pytest.approx(1.0)
    off = dict(e, alpha=0.9)
    assert compute_argument_score("pair", off, e, TOL)[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# expected_action extraction
# --------------------------------------------------------------------------


class _Body:
    def __init__(self, **kw):
        self.expected_action = kw.get("expected_action")
        self.expected_answer = kw.get("expected_answer")
        self.metadata = kw.get("metadata")


def test_extract_expected_action_from_each_location():
    call = _zoom()
    assert extract_expected_action(_Body(expected_action=call))["name"] == call["name"]
    assert extract_expected_action(_Body(metadata={"expected_action": call}))["name"] == call["name"]
    assert extract_expected_action(_Body(expected_answer=json.dumps(call)))["name"] == call["name"]
    assert extract_expected_action(_Body()) is None


def test_extract_expected_action_string_arguments():
    call = {"name": "image_zoom_in_tool", "arguments": json.dumps({"bbox_2d": [1, 2, 3, 4]})}
    got = extract_expected_action(_Body(expected_action=call))
    assert got["arguments"]["bbox_2d"] == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# config defaults
# --------------------------------------------------------------------------


def test_config_defaults_are_binary_iou_half():
    cfg = ImageToolsPivotResourcesServerConfig(
        host="localhost",
        port=0,
        entrypoint="app.py",
    )
    assert cfg.reward_mode == "binary"
    assert cfg.argument_threshold == 0.5
    assert cfg.enable_target_match is True
    assert cfg.unify_region_inspect_tools is True
    assert FailureCode.NONE == "none"


def test_canonical_tool_name():
    canon = canonical_tool_name("image_zoom_in_tool")
    assert canonical_tool_name("image_crop_tool") == canon
    assert canonical_tool_name("  image_crop_tool  ") == canon
    assert canonical_tool_name("color_at_tool") != canon
    assert canonical_tool_name("image_crop_tool", False) == "image_crop_tool"


def test_crop_expected_zoom_rollout_scores_on_iou():
    """Level 0 now passes for zoom-vs-crop, so scoring falls through to IoU."""
    expected = {"bbox_2d": [100, 100, 300, 300], "img_idx": 0}
    assert verify_tool_name_match("image_zoom_in_tool", "image_crop_tool")
    assert verify_target_match("bbox", {"bbox_2d": [100, 100, 300, 300], "img_idx": 0}, expected)
    score, detail = compute_argument_score("bbox", {"bbox_2d": [105, 105, 305, 305]}, expected, TOL)
    assert detail == "iou"
    assert score > 0.5
