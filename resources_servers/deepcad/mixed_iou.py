# SPDX-License-Identifier: Apache-2.0
"""B-rep + symmetric adaptive-octree + inner-boundary IoU worker.

One normalized voxel grid per shape serves both discrete scores. No geometry
repair or fallback to voxel volume is permitted for failed Booleans. Optional
The three component scores share one normalized voxel grid.
The existing Gym operation bonus is applied once, after this worker returns.
"""

import fcntl
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt

from resources_servers.deepcad.geometry import brep_iou, load_solid
from resources_servers.deepcad.mixed_iou_config import MixedIoUConfig


SCORER_VERSION = "mixed-iou-v1"


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def boundary_band(grid: np.ndarray, resolution: int, width: float) -> np.ndarray:
    """Inner Euclidean band, including outside surfaces and internal hole walls."""
    distance = distance_transform_edt(np.pad(grid, 1))[1:-1, 1:-1, 1:-1]
    return grid & (distance <= width * resolution + 1e-9)


def octree_iou(a: np.ndarray, b: np.ndarray, depth: int) -> float:
    """Mean nonempty leaf IoU, averaged over A-guided and B-guided trees.

    Root depth is zero. A guiding block stops when homogeneous or at max depth.
    Leaves are equally weighted, rather than weighted by occupied volume.
    """

    def directed(guide, other):
        values = []

        def visit(x, y, level):
            count = np.count_nonzero(x)
            if not count and not y.any():
                return
            if level == depth or count in (0, x.size) or min(x.shape) < 2:
                values.append(mask_iou(x, y))
                return
            halves = [(slice(0, n // 2), slice(n // 2, n)) for n in x.shape]
            for i in halves[0]:
                for j in halves[1]:
                    for k in halves[2]:
                        visit(x[i, j, k], y[i, j, k], level + 1)

        visit(guide, other, 0)
        return float(np.mean(values)) if values else 1.0

    return (directed(a, b) + directed(b, a)) / 2


def prepare(shape: cq.Shape, resolution: int, config: MixedIoUConfig):
    # Meshing a shape can attach invalid polygon-on-triangulation caches to OCC
    # edges. Tessellate only a deep copy; Boolean operands remain pristine.
    bounds = shape.BoundingBox()
    scale = max(bounds.xlen, bounds.ylen, bounds.zlen)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate source bounding box")
    vertices, faces = shape.copy(mesh=False).tessellate(scale * 1e-4, 0.05)
    points = np.array([v.toTuple() for v in vertices])
    if not len(points) or not len(faces) or not np.isfinite(points).all():
        raise ValueError("Empty or non-finite tessellation")
    lower, upper = points.min(axis=0), points.max(axis=0)
    center, longest = (lower + upper) / 2, float((upper - lower).max())
    if not np.isfinite(longest) or longest <= 0:
        raise ValueError("Degenerate tessellated bounding box")

    def normalize():
        transformed = shape.copy(mesh=False).scale(1 / longest).translate(tuple(0.5 - center / longest))
        solids = transformed.Solids()
        if not solids:
            raise ValueError("No normalized solid bodies")
        union = solids[0].fuse(*solids[1:]) if len(solids) > 1 else solids[0]
        if not union.isValid() or not np.isfinite(union.Volume()) or union.Volume() <= 0:
            raise ValueError("Invalid normalized solid union")
        return union

    solid = normalize()
    if brep_iou(solid, normalize())["reward"] < 1 - 1e-6:
        raise ValueError("B-rep independent self-intersection failed")
    mesh = trimesh.Trimesh(vertices=(points - center) / longest + 0.5, faces=faces, process=False)
    points = mesh.voxelized(pitch=1 / resolution).fill().points
    indices = np.rint(points * resolution).astype(int)
    if not len(indices) or indices.min() < 0 or indices.max() > resolution:
        raise ValueError("Empty voxel grid or normalized voxels outside unit box")
    grid = np.zeros((resolution + 1,) * 3, dtype=bool)
    grid[tuple(indices.T)] = True
    return solid, grid, boundary_band(grid, resolution, config.boundary_band_width)


def reference_data(path: Path, resolution: int, config: MixedIoUConfig, cache_root: Path | None):
    if cache_root is None:
        return (*prepare(load_solid(path), resolution, config), False)
    settings = [SCORER_VERSION, cq.__version__, trimesh.__version__, resolution, config.boundary_band_width]
    digest = hashlib.sha256(path.read_bytes() + json.dumps(settings).encode()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / digest
    with (cache_root / (digest + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            try:
                solid = cq.Shape.importBrep(str(target / "solid.brep"))
                with np.load(target / "grids.npz", allow_pickle=False) as saved:
                    grid, band = saved["grid"], saved["band"]
                if (
                    grid.shape != (resolution + 1,) * 3
                    or band.shape != grid.shape
                    or grid.dtype != bool
                    or band.dtype != bool
                    or not grid.any()
                    or not band.any()
                    or np.any(band & ~grid)
                    or not solid.isValid()
                    or not np.isfinite(solid.Volume())
                    or solid.Volume() <= 0
                ):
                    raise ValueError("Invalid reference cache")
                return solid, grid, band, True
            except Exception:
                # Rebuild a corrupt/incompatible cache, never reuse partial output.
                shutil.rmtree(target)
        solid, grid, band = prepare(load_solid(path), resolution, config)
        with tempfile.TemporaryDirectory(prefix=digest + ".", dir=cache_root) as temp:
            temp = Path(temp)
            solid.exportBrep(str(temp / "solid.brep"))
            np.savez_compressed(temp / "grids.npz", grid=grid, band=band)
            temp.rename(target)
        return solid, grid, band, False


def score_paths(
    prediction: Path,
    reference: Path,
    resolution: int,
    config: MixedIoUConfig,
    cache_root: Path | None = None,
) -> dict:
    config.validate_resolution(resolution)
    started = time.perf_counter()
    try:
        ref_solid, ref_grid, ref_band, hit = reference_data(reference, resolution, config, cache_root)
    except Exception as exc:
        return dict(reward=0.0, mask_sample=True, failure_kind="deepcad:reference", failure_reason=str(exc))
    reference_time = time.perf_counter() - started
    try:
        shape = load_solid(prediction)
    except Exception as exc:
        return dict(reward=0.0, failure_kind="deepcad:geometry", failure_reason=str(exc))
    try:
        t = time.perf_counter()
        pred_solid, pred_grid, pred_band = prepare(shape, resolution, config)
        prediction_time = time.perf_counter() - t
        t = time.perf_counter()
        result = brep_iou(pred_solid, ref_solid)
        brep_time = time.perf_counter() - t
        result["brep_iou"] = result.pop("reward")
        t = time.perf_counter()
        result["octree_iou"] = octree_iou(pred_grid, ref_grid, config.octree_depth)
        octree_time = time.perf_counter() - t
        t = time.perf_counter()
        result["boundary_iou"] = mask_iou(pred_band, ref_band)
        boundary_time = time.perf_counter() - t
        result["reward_components"] = {k: w * result[k] for k, w in config.weights.model_dump().items()}
        result["reward"] = result["mixed_iou"] = sum(result["reward_components"].values())
        result.update(
            volume_iou=result["brep_iou"],
            voxel_iou=result["brep_iou"],
            iou_backend="brep",
            alignment="rescale",
            scoring_method="mixed_iou",
            scorer_version=SCORER_VERSION,
            voxel_resolution=resolution,
            reference_cache_hit=hit,
            reference_time_s=reference_time,
            prediction_prepare_time_s=prediction_time,
            brep_time_s=brep_time,
            octree_time_s=octree_time,
            boundary_time_s=boundary_time,
            scoring_time_s=time.perf_counter() - started,
        )
        return result
    except Exception as exc:
        return dict(reward=0.0, mask_sample=True, failure_kind="deepcad:infrastructure", failure_reason=str(exc))


def main():
    options = json.loads(sys.argv[4])
    if options.get("alignment", "rescale") != "rescale" or options.get("iou_backend", "brep") != "brep":
        raise ValueError("Mixed IoU requires rescale alignment and B-rep backend")
    result = score_paths(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        int(sys.argv[3]),
        MixedIoUConfig.model_validate(options["mixed_iou"]),
        Path(options["cache_root"]) if options.get("cache_root") else None,
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
