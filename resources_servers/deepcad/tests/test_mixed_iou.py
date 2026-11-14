# SPDX-License-Identifier: Apache-2.0
"""Mixed scorer geometry contracts, caching, masking and Gym transport."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cadquery as cq
import numpy as np
import pytest
from pydantic import ValidationError

from resources_servers.deepcad import mixed_iou as scorer
from resources_servers.deepcad.app import DeepCADResourcesServer, DeepCADResourcesServerConfig
from resources_servers.deepcad.mixed_iou_config import MixedIoUConfig, MixedIoUWeights
from resources_servers.deepcad.operations import operation_bonus


def config(**kw):
    return DeepCADResourcesServerConfig(
        host="localhost",
        port=8080,
        entrypoint="app.py",
        name="deepcad",
        verification_metric="mixed_iou",
        mixed_iou={},
        voxel_resolution=128,
        reward_weights={"voxel_iou": 0},
        **kw,
    )


def write_step(path, shape):
    cq.exporters.export(shape, str(path))
    return path


def test_config():
    c = config()
    assert c.mixed_iou.weights.model_dump() == dict(brep_iou=0.3, octree_iou=0.3, boundary_iou=0.4)
    for weights in ({"brep_iou": float("nan")}, {"brep_iou": -0.1}, {"boundary_iou": 0.3}):
        with pytest.raises(ValidationError):
            MixedIoUWeights(**weights)
    with pytest.raises(ValidationError):
        config(alignment="procrustes")
    with pytest.raises(ValidationError):
        config(iou_backend="voxel")
    with pytest.raises(ValueError, match="depth"):
        MixedIoUConfig(octree_depth=8).validate_resolution(128)
    with pytest.raises(ValueError, match="band"):
        MixedIoUConfig(boundary_band_width=1 / 256).validate_resolution(128)


def test_boundary_and_octree_holes_symmetry_and_identity():
    solid = np.zeros((33,) * 3, bool)
    solid[4:29, 4:29, 4:29] = True
    drilled = solid.copy()
    drilled[14:19, 14:19, :] = False
    a = scorer.boundary_band(solid, 32, 1 / 32)
    b = scorer.boundary_band(drilled, 32, 1 / 32)
    assert b[13, 16, 16] and not a[13, 16, 16]  # inner hole wall
    assert a[4, 16, 16] and not a[5, 16, 16]
    assert scorer.mask_iou(a, b) < scorer.mask_iou(solid, drilled)
    assert scorer.octree_iou(solid, drilled, 5) < scorer.mask_iou(solid, drilled)
    assert scorer.octree_iou(solid, drilled, 5) == scorer.octree_iou(drilled, solid, 5)
    assert scorer.octree_iou(solid, solid, 5) == 1
    assert scorer.octree_iou(solid, np.zeros_like(solid), 5) == 0
    assert scorer.mask_iou(a, a) == 1
    # Physical band remains two voxels wide at 128, one at 64.
    full = np.ones((17,) * 3, bool)
    assert scorer.boundary_band(full, 128, 1 / 64)[1, 8, 8]
    assert not scorer.boundary_band(full, 64, 1 / 64)[1, 8, 8]


def test_shared_voxelization_and_reference_cache(tmp_path):
    box = cq.Workplane("XY").box(1, 2, 3).val()
    ref = write_step(tmp_path / "ref.step", box)
    pred = write_step(tmp_path / "pred.step", box.scale(10).translate((15, 2, 7)))
    settings = MixedIoUConfig(octree_depth=4, boundary_band_width=1 / 32)
    original = scorer.trimesh.Trimesh.voxelized
    calls = []

    def counted(mesh, *args, **kwargs):
        calls.append(1)
        return original(mesh, *args, **kwargs)

    with patch.object(scorer.trimesh.Trimesh, "voxelized", counted):
        first = scorer.score_paths(pred, ref, 32, settings, tmp_path / "cache")
        assert "failure_kind" not in first, first
        assert len(calls) == 2
        second = scorer.score_paths(pred, ref, 32, settings, tmp_path / "cache")
        assert "failure_kind" not in second, second
        assert len(calls) == 3  # one prediction grid, reused by both metrics
    assert first["reward"] == pytest.approx(1)
    assert second["reference_cache_hit"] and not first["reference_cache_hit"]
    assert second["reward"] == pytest.approx(first["reward"])
    for item in (first, second):
        assert item["reward"] == sum(item["reward_components"].values())
        assert item["volume_iou"] == item["brep_iou"]
    cachefile = next((tmp_path / "cache").glob("*/grids.npz"))
    cachefile.write_bytes(b"corrupt")
    rebuilt = scorer.score_paths(pred, ref, 32, settings, tmp_path / "cache")
    assert "failure_kind" not in rebuilt, rebuilt
    assert not rebuilt["reference_cache_hit"]
    assert box.isValid()


def test_analytic_hole_and_rotation(tmp_path):
    solid = cq.Workplane("XY").box(1, 1, 1).val()
    hole = cq.Workplane("XY").box(1, 1, 1).faces(">Z").workplane().hole(0.25).val()
    ref = write_step(tmp_path / "ref.step", hole)
    pred = write_step(tmp_path / "pred.step", solid)
    result = scorer.score_paths(pred, ref, 64, MixedIoUConfig())
    assert "failure_kind" not in result, result
    assert result["brep_iou"] == pytest.approx(1 - np.pi * 0.125**2, abs=1e-7)
    assert 0 < result["boundary_iou"] < 1
    assert 0 < result["octree_iou"] < 1
    assert result["reward"] == pytest.approx(
        0.3 * result["brep_iou"] + 0.3 * result["octree_iou"] + 0.4 * result["boundary_iou"]
    )
    bonus, _ = operation_bonus(result["reward"], ["extrude"], ["extrude"], max_fraction=0.2)
    assert result["reward"] + bonus == pytest.approx(result["reward"] * 1.2)
    rect = cq.Workplane("XY").box(1, 2, 3).val()
    write_step(ref, rect)
    write_step(pred, rect.rotate((0, 0, 0), (0, 0, 1), 90))
    result = scorer.score_paths(pred, ref, 32, MixedIoUConfig(octree_depth=4, boundary_band_width=1 / 32))
    assert result["brep_iou"] == pytest.approx(1 / 3)


def test_masking_and_normalization_failures(tmp_path):
    ref = write_step(tmp_path / "ref.step", cq.Workplane("XY").box(1, 1, 1).val())
    cfg = MixedIoUConfig(octree_depth=3, boundary_band_width=1 / 16)
    missing = tmp_path / "missing.step"
    result = scorer.score_paths(ref, missing, 16, cfg)
    assert result["mask_sample"] and result["failure_kind"] == "deepcad:reference"
    result = scorer.score_paths(missing, ref, 16, cfg)
    assert result["reward"] == 0 and not result.get("mask_sample")
    with patch.object(scorer, "brep_iou", return_value={"reward": 0}):
        result = scorer.score_paths(ref, ref, 16, cfg)
    assert result["mask_sample"] and "self-intersection" in result["failure_reason"]
    prepared = scorer.prepare(scorer.load_solid(ref), 16, cfg)
    with (
        patch.object(scorer, "reference_data", return_value=(*prepared, False)),
        patch.object(scorer, "prepare", side_effect=ValueError("normalization")),
    ):
        result = scorer.score_paths(ref, ref, 16, cfg)
    assert result["mask_sample"] and result["failure_kind"] == "deepcad:infrastructure"


@pytest.mark.asyncio
async def test_worker_dispatch():
    server = DeepCADResourcesServer.model_construct(config=config())
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b'{"reward":0.5}', b"")
    with patch("asyncio.create_subprocess_exec", return_value=process) as spawn:
        result = await server._score_geometry(Path("pred.step"), Path("ref.step"))
    args = spawn.call_args.args
    assert args[1].endswith("mixed_iou.py")
    assert args[4] == "128"
    assert json.loads(args[5])["mixed_iou"]["weights"]["boundary_iou"] == 0.4
    assert result["reward"] == 0.5


def test_cli_128(tmp_path):
    p = write_step(tmp_path / "box.step", cq.Workplane("XY").box(1, 2, 3).val())
    options = dict(mixed_iou={}, cache_root=str(tmp_path / "cache"), alignment="rescale", iou_backend="brep")
    with (
        patch.object(sys, "argv", ["mixed_iou.py", str(p), str(p), "128", json.dumps(options)]),
        patch("builtins.print") as output,
    ):
        scorer.main()
    result = json.loads(output.call_args.args[0])
    assert result["reward"] == pytest.approx(1), result
    assert result["voxel_resolution"] == 128


@pytest.mark.asyncio
async def test_verify_applies_bonus_once_and_stable_failure_schema():
    from unittest.mock import MagicMock

    from nemo_gym.server_runtime.base_server import ServerClient
    from resources_servers.deepcad.tests.test_app import request

    c = config(operation_bonus={"enabled": True, "max_fraction": 0.2})
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    server = DeepCADResourcesServer(config=c, server_client=client)
    body = request("import cadquery as cq\nresult=1")
    body.verifier_metadata.reference_operations = ["extrude", "fillet"]
    evaluated = dict(
        reward=0.5,
        mixed_iou=0.5,
        brep_iou=0.2,
        octree_iou=0.4,
        boundary_iou=0.8,
        volume_iou=0.2,
        voxel_iou=0.2,
        reward_components=dict(brep_iou=0.06, octree_iou=0.12, boundary_iou=0.32),
        executed_operations=["extrude"],
    )
    with patch.object(DeepCADResourcesServer, "_evaluate", AsyncMock(return_value=evaluated)):
        result = await server.verify(body)
    assert result.reward == pytest.approx(0.55)
    assert result.operation_bonus == pytest.approx(0.05)
    assert result.raw_metrics["operation_recall"] == 0.5
    assert result.raw_metrics["mixed_iou"] == 0.5
    assert result.raw_metrics["brep_iou"] == 0.2
    assert sum(result.reward_components.values()) == pytest.approx(result.reward)
    failed = await server.verify(request("if:"))
    assert failed.reward == 0 and not failed.mask_sample
    assert failed.reward_components.keys() == result.reward_components.keys()
    assert failed.raw_metrics.keys() == result.raw_metrics.keys()
