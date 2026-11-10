# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cadquery as cq
import numpy as np
import pytest

from resources_servers.deepcad import geometry, runner


@pytest.mark.parametrize(
    "code, valid",
    [
        ("import cadquery as cq\nresult=cq.Workplane('XY').box(1,2,3)", True),
        ("import cadquery as cq\nresult=cq.Workplane('XY').box(1,2,3).val()", True),
        ("import cadquery as cq\nresult=cq.Workplane('XY')", False),
        ("result=42", False),
        ("raise RuntimeError('bad program')", False),
        ("import cadquery as cq\nresult=cq.Workplane('XY').rect(1,2).wire()", False),
    ],
)
def test_runner_solid_contract(tmp_path: Path, code: str, valid: bool) -> None:
    (tmp_path / "model.py").write_text(code)
    output = tmp_path / "prediction.step"
    real_export = cq.exporters.export
    with patch.object(runner, "Path", side_effect=lambda path: tmp_path / Path(path).name):
        with patch.object(runner.cq.exporters, "export", side_effect=lambda shape, _: real_export(shape, str(output))):
            if valid:
                runner.main()
                assert geometry.load_solid(output).Volume() == pytest.approx(6)
            else:
                with pytest.raises((ValueError, RuntimeError)):
                    runner.main()
                assert not output.exists()


def test_runner_nonfinite_volume(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text("import cadquery as cq\nresult=cq.Workplane('XY').box(1,2,3).val()")
    with patch.object(runner, "Path", side_effect=lambda path: tmp_path / Path(path).name):
        with patch.object(cq.Shape, "Volume", return_value=float("nan")):
            with pytest.raises(ValueError, match="finite volume"):
                runner.main()


@pytest.mark.parametrize("missing", [None, "reference", "prediction"])
def test_worker_protocol(tmp_path: Path, capsys, missing: str | None) -> None:
    reference, prediction = tmp_path / "reference.step", tmp_path / "prediction.step"
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(reference))
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(prediction))
    if missing == "reference":
        reference = tmp_path / "missing.step"
    elif missing == "prediction":
        prediction = tmp_path / "missing.step"
    with patch("sys.argv", ["geometry", str(prediction), str(reference), "64"]):
        geometry.main()
    result = json.loads(capsys.readouterr().out)
    if missing:
        assert result["reward"] == 0
        assert result.get("mask_sample", False) == (missing == "reference")
    else:
        assert result["reward"] == pytest.approx(1)


def test_load_invalid_solid() -> None:
    imported = MagicMock()
    imported.val.return_value.Solids.return_value = [1]
    imported.val.return_value.Volume.return_value = 1
    imported.val.return_value.isValid.return_value = False
    with patch.object(cq.importers, "importStep", return_value=imported):
        with pytest.raises(ValueError, match="valid solids"):
            geometry.load_solid(Path("invalid.step"))


def test_empty_mesh() -> None:
    shape = MagicMock()
    shape.BoundingBox.return_value.DiagonalLength = 1
    shape.tessellate.return_value = ([], [])
    with pytest.raises(ValueError, match="Empty"):
        geometry.normalized_mesh(shape)


def test_degenerate_mesh() -> None:
    shape = MagicMock()
    shape.BoundingBox.return_value.DiagonalLength = 0
    shape.tessellate.return_value = ([cq.Vector(0, 0, 0)] * 3, [(0, 1, 2)])
    with pytest.raises(ValueError, match="Degenerate"):
        geometry.normalized_mesh(shape)


def test_independent_mesh_normalization_preserves_aspect_ratio() -> None:
    shape = cq.Workplane("XY").box(1, 2, 3).val()
    for source in (shape, shape.scale(1000).translate((20, -30, 50))):
        mesh = geometry.normalized_mesh(source)
        np.testing.assert_allclose(mesh.bounds.mean(axis=0), [0.5, 0.5, 0.5])
        np.testing.assert_allclose(mesh.extents, [1 / 3, 2 / 3, 1])
    assert geometry.score(cq.Workplane("XY").box(2, 2, 3).val(), shape)["reward"] < 0.7


def test_voxelization_fills_solids_and_preserves_through_holes() -> None:
    box = cq.Workplane("XY").box(1, 1, 1).val()
    filled = geometry.normalized_voxels(box)
    assert filled.shape == (68, 68, 68) and filled.dtype == bool
    assert filled[34, 34, 34] and np.count_nonzero(filled) > 64**3
    drilled = cq.Workplane("XY").box(1, 1, 1).faces(">Z").workplane().hole(0.5).val()
    hollow = geometry.normalized_voxels(drilled)
    assert not hollow[34, 34, 34]
    result = geometry.score(drilled, box, iou_backend="voxel")
    assert 0.7 < result["reward"] < 0.9
    assert result["reward"] == result["intersection_voxels"] / result["union_voxels"]
    assert result["prediction_voxels"] < result["reference_voxels"]


@pytest.mark.parametrize("resolution", [0, -1])
def test_invalid_voxel_resolution(resolution: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        geometry.normalized_voxels(cq.Workplane("XY").box(1, 1, 1).val(), resolution=resolution)


def test_empty_voxel_union() -> None:
    with pytest.raises(ValueError, match="Empty voxel union"):
        geometry.voxel_iou(np.zeros((4, 4, 4), dtype=bool), np.zeros((4, 4, 4), dtype=bool))


def test_reference_voxel_failure_is_masked(tmp_path: Path, capsys) -> None:
    path = tmp_path / "part.step"
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(path))
    with (
        patch("sys.argv", ["geometry", str(path), str(path), "64", json.dumps({"iou_backend": "voxel"})]),
        patch.object(geometry, "normalized_voxels", side_effect=ValueError("bad reference mesh")),
    ):
        geometry.main()
    result = json.loads(capsys.readouterr().out)
    assert result["reward"] == 0 and result["mask_sample"]
    assert result["failure_kind"] == "deepcad:reference"


def test_runner_requires_explicit_cadquery_import(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text("result=cq.Workplane().box(1,2,3)")
    with patch.object(runner, "Path", side_effect=lambda path: tmp_path / Path(path).name):
        with pytest.raises(NameError, match="cq"):
            runner.main()
    assert not (tmp_path / "prediction.step").exists()


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("explicit_import", [False, True])
def test_runner_optional_cq_namespace(tmp_path, monkeypatch, preload, explicit_import):
    code = ("import cadquery as cq\n" if explicit_import else "") + "result=cq.Workplane().box(1,2,3)"
    (tmp_path / "model.py").write_text(code)
    monkeypatch.setattr(runner.sys, "argv", ["runner.py"] + (["--preload-cq"] if preload else []))
    output = tmp_path / "prediction.step"
    real_export = cq.exporters.export
    with patch.object(runner, "Path", side_effect=lambda path: tmp_path / Path(path).name):
        with patch.object(runner.cq.exporters, "export", side_effect=lambda shape, _: real_export(shape, str(output))):
            if preload or explicit_import:
                runner.main()
                assert geometry.load_solid(output).Volume() == pytest.approx(6)
            else:
                with pytest.raises(NameError, match="cq"):
                    runner.main()
                assert not output.exists()
