# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytic volume contracts, backend selection, and conservative kernel failures."""

import json
from pathlib import Path
from unittest.mock import patch

import cadquery as cq
import numpy as np
import pytest
from pydantic import ValidationError

from resources_servers.deepcad import geometry
from resources_servers.deepcad.app import DeepCADResourcesServerConfig


def test_brep_default_and_normalization() -> None:
    reference = cq.Workplane("XY").box(1, 2, 3).val()
    moved = reference.scale(1000).translate((20, -30, 50))
    with patch.object(geometry, "normalized_voxels", side_effect=AssertionError("Unexpected voxelization")):
        result = geometry.score(moved, reference)
    assert result["iou_backend"] == "brep" and result["scoring_method"] == "brep_iou"
    assert result["reward"] == pytest.approx(1)
    assert result["reference_volume"] == pytest.approx(2 / 9)
    assert "voxel_resolution" not in result
    for source in (reference, moved):
        expected = geometry.normalized_mesh(source)
        actual = geometry.normalized_mesh(geometry.normalized_solid(source))
        np.testing.assert_allclose(actual.bounds, expected.bounds, atol=1e-10)
    rotated = reference.rotate((0, 0, 0), (0, 0, 1), 90)
    assert geometry.score(rotated, reference)["reward"] == pytest.approx(1 / 3)


def test_brep_analytic_through_hole() -> None:
    box = cq.Workplane("XY").box(1, 1, 1).val()
    drilled = cq.Workplane("XY").box(1, 1, 1).faces(">Z").workplane().hole(0.01).val()
    result = geometry.score(drilled, box)
    assert result["reward"] == pytest.approx(1 - np.pi * 0.005**2, abs=1e-8)
    assert result["reward"] < 1
    assert result["intersection_volume"] == pytest.approx(drilled.Volume())


def test_overlapping_bodies_use_union_volume() -> None:
    a = cq.Workplane("XY").box(1, 1, 1).val()
    b = a.translate((0.5, 0, 0))
    compound = cq.Compound.makeCompound([a, b])
    fused = a.fuse(b)
    result = geometry.score(compound, fused)
    assert result["reward"] == pytest.approx(1)
    assert result["prediction_volume"] == pytest.approx(1.5 / 1.5**3)


def test_explicit_voxel_matches_previous_grid_scorer() -> None:
    a = cq.Workplane("XY").box(1, 2, 3).val()
    b = cq.Workplane("XY").box(2, 2, 3).val()
    previous = geometry.voxel_iou(geometry.normalized_voxels(a), geometry.normalized_voxels(b))
    result = geometry.score(a, b, iou_backend="voxel")
    assert result["iou_backend"] == "voxel"
    for key, value in previous.items():
        assert result[key] == value
    assert result["volume_iou"] == previous["reward"]


def test_unknown_backend_rejected() -> None:
    kwargs = dict(host="localhost", port=8080, entrypoint="app.py", name="deepcad")
    assert DeepCADResourcesServerConfig(**kwargs).iou_backend == "brep"
    with pytest.raises(ValidationError):
        DeepCADResourcesServerConfig(**kwargs, iou_backend="typo")
    with pytest.raises(ValueError, match="Unknown IoU backend"):
        geometry.score(None, None, iou_backend="typo")


def test_self_check_uses_independent_boundaries() -> None:
    shape = cq.Workplane("XY").box(1, 2, 3).val()
    real = geometry.brep_iou
    with patch.object(geometry, "brep_iou", wraps=real) as check:
        geometry.checked_normalized_solid(shape)
    first, second = check.call_args.args
    assert first is not second and not first.wrapped.IsSame(second.wrapped)


@pytest.mark.parametrize("failure", ["invalid", "bounds", "nonfinite", "self"])
def test_kernel_guards(failure: str) -> None:
    shape = cq.Workplane("XY").box(1, 1, 1).val()
    common = cq.Workplane("XY").box(1, 1, 1).val()
    if failure == "self":
        with patch.object(geometry, "brep_iou", return_value={"reward": 0.1}):
            with pytest.raises(ValueError, match="self-intersection"):
                geometry.checked_normalized_solid(shape)
        return
    with patch.object(cq.Shape, "intersect", return_value=common):
        if failure == "invalid":
            with patch.object(cq.Shape, "isValid", return_value=False):
                with pytest.raises(ValueError, match="Invalid B-rep"):
                    geometry.brep_iou(shape, shape)
        else:
            volumes = [1, 1, 2 if failure == "bounds" else float("nan")]
            with patch.object(cq.Shape, "Volume", side_effect=volumes):
                with pytest.raises(ValueError, match="outside valid bounds"):
                    geometry.brep_iou(shape, shape)


@pytest.mark.parametrize("backend", ["brep", "voxel"])
@pytest.mark.parametrize("mode", ["axis", "diagonal"])
def test_worker_selects_backend_and_keeps_components(tmp_path: Path, capsys, backend: str, mode: str) -> None:
    path = tmp_path / "box.step"
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(path))
    options = {
        "iou_backend": backend,
        "view_mode": mode,
        "reward_weights": {"voxel_iou": 1, "silhouette_iou": 0, "normal_alignment": 0, "hole_iou": 0},
    }
    with patch("sys.argv", ["geometry", str(path), str(path), "64", json.dumps(options)]):
        geometry.main()
    result = json.loads(capsys.readouterr().out)
    assert result["iou_backend"] == backend
    assert result["reward"] == result["volume_iou"] == result["voxel_iou"] == pytest.approx(1)
    assert set(result["reward_components"]) == {"voxel_iou"}
    assert sum(result["reward_components"].values()) == pytest.approx(1)


@pytest.mark.parametrize("reference_failure", [True, False])
def test_worker_masks_boolean_failures(tmp_path: Path, capsys, reference_failure: bool) -> None:
    path = tmp_path / "box.step"
    shape = cq.Workplane("XY").box(1, 2, 3).val()
    cq.exporters.export(shape, str(path))
    calls = (
        [ValueError("independent self-intersection failed")]
        if reference_failure
        else [shape, ValueError("invalid Boolean")]
    )
    with (
        patch("sys.argv", ["geometry", str(path), str(path)]),
        patch.object(geometry, "checked_normalized_solid", side_effect=calls),
    ):
        geometry.main()
    result = json.loads(capsys.readouterr().out)
    assert result["mask_sample"] and result["reward"] == 0
    assert result["failure_kind"] == ("deepcad:reference" if reference_failure else "deepcad:infrastructure")
