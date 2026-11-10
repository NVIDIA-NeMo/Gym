# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import cadquery as cq
import numpy as np
import pytest
from PIL import Image

from resources_servers.deepcad.dataset.prepare_gym import (
    construction_metrics,
    dimension_data,
    main,
    rank_split,
    render_diagonal_views,
    render_views,
    sample_view_mode,
    task_row,
)


def document(holes: int = 0) -> dict:
    loops = [{"profile_curves": [{"type": "Line3D"}] * 4}]
    loops += [{"profile_curves": [{"type": "Circle3D"}]} for _ in range(holes)]
    return {
        "sequence": [
            {"type": "Sketch", "entity": "s"},
            {"type": "ExtrudeFeature", "entity": "e"},
        ],
        "entities": {
            "s": {"profiles": {"p": {"loops": loops}, "unused": {"loops": loops * 100}}},
            "e": {
                "operation": "NewBodyFeatureOperation",
                "profiles": [{"sketch": "s", "profile": "p"}],
            },
        },
    }


def test_single_extrusion_many_holes_and_unused_profile() -> None:
    counts = construction_metrics(document(8))
    assert counts == {
        "extrusions": 1,
        "sketch_curves": 12,
        "additional_loops": 8,
        "cuts_and_intersections": 0,
        "arcs_and_circles": 8,
        "difficulty": 32,
    }


def test_reused_profile_cut_and_intersection() -> None:
    model = document()
    for name, operation in [
        ("cut", "CutFeatureOperation"),
        ("intersect", "IntersectFeatureOperation"),
    ]:
        model["entities"][name] = {
            **copy.deepcopy(model["entities"]["e"]),
            "operation": operation,
        }
        model["sequence"].append({"type": "ExtrudeFeature", "entity": name})
    counts = construction_metrics(model)
    assert counts["extrusions"] == 3
    assert counts["sketch_curves"] == 12
    assert counts["cuts_and_intersections"] == 2
    assert counts["difficulty"] == 32


def test_arc_count_has_no_extra_weight() -> None:
    model = document()
    model["entities"]["s"]["profiles"]["p"]["loops"][0]["profile_curves"][0] = {"type": "Arc3D"}
    counts = construction_metrics(model)
    assert counts["arcs_and_circles"] == 1 and counts["difficulty"] == 8


def test_multi_profile_feature() -> None:
    model = document()
    model["entities"]["s"]["profiles"]["p2"] = {"loops": [{"profile_curves": [{"type": "Circle3D"}]}]}
    model["entities"]["e"]["profiles"].append({"sketch": "s", "profile": "p2"})
    counts = construction_metrics(model)
    assert counts["extrusions"] == 1 and counts["sketch_curves"] == 5 and counts["difficulty"] == 9


def test_ranking_entire_split_and_stable_ties(tmp_path: Path) -> None:
    (tmp_path / "cad_json").mkdir()
    for identifier, holes in [("hard", 8), ("b", 0), ("a", 0)]:
        (tmp_path / "cad_json" / f"{identifier}.json").write_text(json.dumps(document(holes)))
    ranked, failures = rank_split(tmp_path, ["hard", "b", "missing", "a"])
    assert [record["task_id"] for record in ranked] == ["a", "b", "hard"]
    assert [record["difficulty"] for record in ranked] == [8, 8, 32]
    assert failures[0]["task_id"] == "missing"


def test_cli_sorts_before_limit_and_preserves_images(tmp_path: Path) -> None:
    (tmp_path / "cad_json").mkdir()
    for identifier, holes in [("hard", 8), ("easy", 0)]:
        (tmp_path / "cad_json" / f"{identifier}.json").write_text(json.dumps(document(holes)))
    (tmp_path / "train_val_test_split.json").write_text(json.dumps({"validation": ["hard", "easy"]}))
    image = tmp_path / "input.png"
    Image.new("RGB", (16, 16)).save(image)
    output = tmp_path / "tasks.jsonl"
    args = [
        "prepare_gym",
        "--data-root",
        str(tmp_path),
        "--output",
        str(output),
        "--reference-root",
        str(tmp_path / "refs"),
        "--limit",
        "1",
        "--diagonal-view-probability",
        "0",
    ]
    with (
        patch("sys.argv", args),
        patch(
            "resources_servers.deepcad.dataset.prepare_gym.reconstruct",
            return_value=cq.Workplane("XY").box(1, 2, 3).val(),
        ),
        patch(
            "resources_servers.deepcad.dataset.prepare_gym.render_views",
            return_value={k: image for k in ("front", "side", "top")},
        ),
    ):
        main()
    row = json.loads(output.read_text())
    assert row["verifier_metadata"]["task_id"] == "easy"
    assert row["verifier_metadata"]["difficulty"] == 8
    content = row["responses_create_params"]["input"][1]["content"]
    assert len([item for item in content if item["type"] == "input_image"]) == 3
    assert all(item["detail"] == "high" for item in content if item["type"] == "input_image")
    gym = pytest.importorskip("nemo_gym.server_runtime.base_resources_server")
    gym.BaseRunRequest.model_validate(row)
    assert "reference_step" not in json.dumps(row["responses_create_params"])
    assert len(output.with_suffix(".difficulty.jsonl").read_text().splitlines()) == 2


def test_manifest_requires_cad_history(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"task_id": "example"}) + "\n")
    with (
        patch(
            "sys.argv",
            [
                "prepare",
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--reference-root",
                str(tmp_path / "refs"),
            ],
        ),
        pytest.raises(ValueError, match="cad_json"),
    ):
        main()


def test_dimensions_use_final_geometry_and_convert_metres_to_mm(tmp_path: Path) -> None:
    # A drilled box exercises true radius and overall dimensions, with cap-edge duplicates.
    shape = cq.Workplane("XY").box(0.02, 0.03, 0.004).faces(">Z").workplane().hole(0.006).val()
    measured = dimension_data(shape)
    np.testing.assert_allclose(measured["size_mm"], [20, 30, 4], atol=1e-5)
    assert {round(c["radius"] * 1000, 6) for c in measured["circles"]} == {3}
    paths = render_views(shape, tmp_path)
    assert set(paths) == {"front", "side", "top"}
    for name, path in paths.items():
        image = Image.open(path)
        assert image.size == (256, 256)
        pixels = np.asarray(image)
        # View title, overall size labels and radius leaders occupy reserved margins.
        assert np.any(pixels[4:16, 4:212] < 180)
        assert np.any(pixels[226:242, 40:216] < 180)
        assert np.any(pixels[44:212, 226:244] < 180)
        if name == "top":
            assert np.any(pixels[16:32, 4:240] < 180)


def test_annotation_free_axis_render_removes_labels_and_leaders(tmp_path: Path) -> None:
    shape = cq.Workplane("XY").box(0.02, 0.03, 0.004).faces(">Z").workplane().hole(0.006).val()
    paths = render_views(shape, tmp_path, engineering_annotations=False)
    for path in paths.values():
        pixels = np.asarray(Image.open(path))
        assert np.all(pixels[:32] == 255) and np.all(pixels[218:] == 255)
        assert np.all(pixels[:, :32] == 255) and np.all(pixels[:, 218:] == 255)
        assert np.any(pixels[44:212, 44:212] < 200)


@pytest.mark.parametrize("probability", [0, 1])
def test_augment_fixed_annotations_updates_images_and_prompt(tmp_path: Path, probability: float) -> None:
    from resources_servers.deepcad.dataset.augment_views import augment

    shape = cq.Workplane("XY").box(0.02, 0.03, 0.004).val()
    cq.exporters.export(shape, str(tmp_path / "part.step"))
    images = render_views(shape, tmp_path / "original")
    source = task_row("part", "part.step", images, metrics=construction_metrics(document()), dimensions_mm=[20, 30, 4])
    result = augment((copy.deepcopy(source), tmp_path, tmp_path / "new", 0, 42, probability))
    assert result["verifier_metadata"]["engineering_annotations"] == (probability == 0)
    assert ("20, 30, 4" in str(result["responses_create_params"])) == (probability == 0)
    assert result == augment((copy.deepcopy(source), tmp_path, tmp_path / "new", 0, 42, probability))
    first_image = next(
        v for v in result["responses_create_params"]["input"][1]["content"] if v["type"] == "input_image"
    )
    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(first_image["image_url"].split(",", 1)[1]))))
    assert bool(np.any(pixels[:32] < 255)) == (probability == 0)


@pytest.mark.parametrize("probability,mode", [(0, "axis"), (1, "diagonal")])
def test_view_probability_extremes(probability: float, mode: str) -> None:
    assert all(sample_view_mode(str(index), diagonal_view_probability=probability) == mode for index in range(100))


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_view_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        sample_view_mode("part", diagonal_view_probability=probability)


def test_view_sampling_ratio_and_reproducibility() -> None:
    ids = [f"part-{index}" for index in range(2000)]
    selected = {identifier: sample_view_mode(identifier) for identifier in ids}
    assert 0.72 < sum(mode == "diagonal" for mode in selected.values()) / len(ids) < 0.78
    assert selected == {identifier: sample_view_mode(identifier) for identifier in reversed(ids)}
    assert selected != {identifier: sample_view_mode(identifier, seed=43) for identifier in ids}


def test_views_are_independent_of_bucket_selection() -> None:
    # The medium preparer selects the lowest candidate hashes using seed 42.
    ids = sorted(
        [f"candidate-{index}" for index in range(5000)],
        key=lambda identifier: hashlib.sha256(f"42:{identifier}".encode()).digest(),
    )[:200]
    ratio = sum(sample_view_mode(identifier) == "diagonal" for identifier in ids) / len(ids)
    assert 0.67 < ratio < 0.83


def test_diagonal_composite_and_prompt(tmp_path: Path) -> None:
    shape = cq.Workplane("XY").box(0.02, 0.03, 0.004).faces(">Z").workplane().hole(0.006).val()
    images = render_diagonal_views(shape, tmp_path)
    image = Image.open(images["diagonal"])
    assert image.size == (524, 524)
    pixels = np.asarray(image)
    assert np.all(pixels[:4] == 255) and np.all(pixels[:, :4] == 255)
    assert np.all(pixels[260:264] == 255) and np.all(pixels[:, 260:264] == 255)
    for x, y in ((4, 4), (264, 4), (4, 264), (264, 264)):
        tile = pixels[y : y + 256, x : x + 256]
        assert np.any(tile < 200)
        assert np.all(tile[:2] == 255) and np.all(tile[-2:] == 255)
        assert np.all(tile[:, :2] == 255) and np.all(tile[:, -2:] == 255)
    row = task_row("part", "private.step", images, metrics=construction_metrics(document()), dimensions_mm=[20, 30, 4])
    assert row["verifier_metadata"]["view_mode"] == "diagonal"
    prompt = row["responses_create_params"]
    assert "three input images" not in json.dumps(prompt)
    assert "Top-left:     [-1, -1, -1]" in json.dumps(prompt)
    assert "20, 30, 4" in json.dumps(prompt)
    assert "private.step" not in json.dumps(prompt)
    assert sum(item["type"] == "input_image" for item in prompt["input"][1]["content"]) == 1


@pytest.mark.parametrize("probability,mode,count,size", [(0, "axis", 3, (256, 256)), (1, "diagonal", 1, (524, 524))])
def test_augment_existing_row_preserves_target(
    tmp_path: Path, probability: float, mode: str, count: int, size: tuple[int, int]
) -> None:
    from resources_servers.deepcad.dataset.augment_views import augment
    from resources_servers.deepcad.task_data import TaskData

    shape = cq.Workplane("XY").box(0.02, 0.03, 0.004).val()
    cq.exporters.export(shape, str(tmp_path / "part.step"))
    images = render_views(shape, tmp_path / "original")
    row = task_row("part", "part.step", images, metrics=construction_metrics(document()))
    original = copy.deepcopy(row)
    result = augment((row, tmp_path, tmp_path / "new", probability, 42, 0))
    assert result["verifier_metadata"] == original["verifier_metadata"] | {"view_mode": mode}
    TaskData.model_validate(result["verifier_metadata"])
    inputs = [
        item for item in result["responses_create_params"]["input"][1]["content"] if item["type"] == "input_image"
    ]
    assert len(inputs) == count
    assert all(
        Image.open(io.BytesIO(base64.b64decode(item["image_url"].split(",", 1)[1]))).size == size for item in inputs
    )
    assert result["task_source"] == original["task_source"]


def test_plate_depth_with_mislabeled_outer_loops() -> None:
    model = document()
    points = [(0, 0), (0.02, 0), (0.02, 0.03), (0, 0.03)]
    curves = [
        {
            "type": "Line3D",
            "start_point": {"x": a[0], "y": a[1], "z": 0},
            "end_point": {"x": b[0], "y": b[1], "z": 0},
        }
        for a, b in zip(points, points[1:] + points[:1])
    ]
    model["entities"]["s"]["profiles"]["p"]["loops"] = [
        {
            "is_outer": True,
            "profile_curves": [
                {
                    "type": "Circle3D",
                    "radius": 0.003,
                    "center_point": {"x": 0.01, "y": 0.01, "z": 0},
                }
            ],
        },
        {"is_outer": True, "profile_curves": curves},
    ]
    model["entities"]["s"]["transform"] = {
        "origin": {"x": 0, "y": 0, "z": 0},
        "z_axis": {"x": 0, "y": 0, "z": 1},
    }
    model["entities"]["e"].update(
        {
            "extent_type": "OneSideFeatureExtentType",
            "extent_one": {"distance": {"value": 0.004}},
        }
    )
    measurements = dimension_data(cq.Workplane("XY").box(0.02, 0.03, 0.004).val(), model)
    assert len(measurements["plates"]) == 1
    assert measurements["plates"][0]["depth_mm"] == 4
    assert measurements["plates"][0]["axis"] == 2


def test_circle_dimensions_do_not_require_arc_length_inversion() -> None:
    shape = cq.Workplane("XY").box(0.02, 0.03, 0.004).faces(">Z").workplane().hole(0.006).val()
    with patch.object(cq.Edge, "positionAt", side_effect=RuntimeError("arc-length inversion failed")):
        measured = dimension_data(shape)
    np.testing.assert_allclose(measured["size_mm"], [20, 30, 4], atol=1e-5)
    assert measured["circles"]
    for circle in measured["circles"]:
        radial = circle["point"] - circle["center"]
        np.testing.assert_allclose(np.linalg.norm(radial), circle["radius"], atol=1e-9)
        np.testing.assert_allclose(np.dot(radial, circle["normal"]), 0, atol=1e-9)
