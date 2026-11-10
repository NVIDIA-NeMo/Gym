# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted geometry worker with fixed-orientation B-rep and voxel IoU."""

import json
import sys
import time
from pathlib import Path
from typing import Literal

import cadquery as cq
import numpy as np
import trimesh


def load_solid(path: Path) -> cq.Shape:
    """Read a STEP file and reject invalid, empty, or non-solid geometry."""
    shape = cq.importers.importStep(str(path)).val()
    volume = shape.Volume()
    if not shape.Solids() or not shape.isValid() or not np.isfinite(volume) or volume <= 0:
        raise ValueError("STEP must contain valid solids with positive finite volume")
    return shape


def tessellation_bounds(shape: cq.Shape) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Use the same tessellated bbox for both overlap representations."""
    vertices, faces = shape.tessellate(shape.BoundingBox().DiagonalLength * 0.002, 0.05)
    points = np.array([vertex.toTuple() for vertex in vertices], dtype=float)
    if not len(points) or not len(faces) or not np.isfinite(points).all():
        raise ValueError("Empty or non-finite mesh")
    lower, upper = points.min(axis=0), points.max(axis=0)
    longest = float(np.max(upper - lower))
    if not np.isfinite(longest) or longest <= 0:
        raise ValueError("Degenerate mesh bounding box")
    return points, np.asarray(faces), (lower + upper) / 2, longest


def normalized_mesh(shape: cq.Shape) -> trimesh.Trimesh:
    """Center the bbox at 0.5 and uniformly scale its longest dimension to one.

    Tessellation tolerance follows input size. Axes and orientation are preserved.
    """
    points, faces, center, longest = tessellation_bounds(shape)
    return trimesh.Trimesh(vertices=(points - center) / longest + 0.5, faces=faces, process=False)


def normalized_solid(shape: cq.Shape) -> cq.Shape:
    """Apply the current mesh bbox transform, then take the union of all bodies."""
    _, _, center, longest = tessellation_bounds(shape)
    transformed = shape.scale(1 / longest).translate(tuple(0.5 - center / longest))
    solids = transformed.Solids()
    if not solids:
        raise ValueError("No normalized solid bodies")
    union = solids[0].fuse(*solids[1:]) if len(solids) > 1 else solids[0]
    if not union.isValid() or not np.isfinite(union.Volume()) or union.Volume() <= 0:
        raise ValueError("Invalid normalized solid union")
    return union


def brep_iou(prediction: cq.Shape, reference: cq.Shape) -> dict:
    """Compute CAD Boolean volume IoU on already normalized solid unions."""
    pred_volume, ref_volume = prediction.Volume(), reference.Volume()
    if any(not np.isfinite(volume) or volume <= 0 for volume in (pred_volume, ref_volume)):
        raise ValueError("B-rep solids must have positive finite volume")
    common = prediction.intersect(reference)
    if common.Solids() and not common.isValid():
        raise ValueError("Invalid B-rep Boolean intersection")
    intersection = common.Volume() if common.Solids() else 0.0
    tolerance = 1e-7 * max(pred_volume, ref_volume)
    if (
        not np.isfinite(intersection)
        or intersection < -tolerance
        or intersection > min(pred_volume, ref_volume) + tolerance
    ):
        raise ValueError("B-rep intersection volume outside valid bounds")
    intersection = min(max(intersection, 0.0), pred_volume, ref_volume)
    union = pred_volume + ref_volume - intersection
    return {
        "reward": intersection / union,
        "scoring_method": "brep_iou",
        "intersection_volume": intersection,
        "union_volume": union,
        "prediction_volume": pred_volume,
        "reference_volume": ref_volume,
    }


def checked_normalized_solid(shape: cq.Shape) -> cq.Shape:
    """Reject kernel failures using independently transformed self-intersections.

    Intersecting a shape with itself uses an OCC identity fast path and can hide
    failures. Normalize the source twice, producing independent CAD boundaries.
    """
    normalized = normalized_solid(shape)
    check = brep_iou(normalized, normalized_solid(shape))
    if check["reward"] < 1 - 1e-6:
        raise ValueError("B-rep independent self-intersection failed")
    return normalized


def normalized_voxels(shape: cq.Shape, *, resolution: int = 64) -> np.ndarray:
    """Voxelize and fill at pitch 1/resolution, then center-pad like BenchCAD."""
    if resolution <= 0:
        raise ValueError("Voxel resolution must be positive")
    voxels = normalized_mesh(shape).voxelized(pitch=1.0 / resolution).fill()
    occupied = voxels.matrix.astype(bool)
    size = resolution + 4
    if any(dimension > size for dimension in occupied.shape):
        raise ValueError("Normalized voxel grid exceeds its padded bounds")
    grid = np.zeros((size, size, size), dtype=bool)
    start = (size - np.array(occupied.shape)) // 2
    slices = tuple(
        slice(int(offset), int(offset + length)) for offset, length in zip(start, occupied.shape, strict=True)
    )
    grid[slices] = occupied
    if not grid.any():
        raise ValueError("Empty voxelized solid")
    return grid


def voxel_iou(prediction: np.ndarray, reference: np.ndarray, *, resolution: int = 64) -> dict:
    """Compute filled-grid overlap without rotation, reflection or axis rescaling."""
    intersection = int(np.count_nonzero(prediction & reference))
    union = int(np.count_nonzero(prediction | reference))
    if not union:
        raise ValueError("Empty voxel union")
    return {
        "reward": intersection / union,
        "scoring_method": "voxel_iou",
        "voxel_resolution": resolution,
        "intersection_voxels": intersection,
        "union_voxels": union,
        "prediction_voxels": int(np.count_nonzero(prediction)),
        "reference_voxels": int(np.count_nonzero(reference)),
    }


def score(
    prediction: cq.Shape,
    reference: cq.Shape,
    *,
    resolution: int = 64,
    iou_backend: Literal["brep", "voxel"] = "brep",
    alignment: Literal["rescale"] = "rescale",
) -> dict:
    """Normalize each bounding box independently, preserving orientation."""
    if alignment != "rescale":
        raise ValueError(f"Unknown alignment: {alignment}")
    if iou_backend == "brep":
        result = brep_iou(checked_normalized_solid(prediction), checked_normalized_solid(reference))
    elif iou_backend == "voxel":
        result = voxel_iou(
            normalized_voxels(prediction, resolution=resolution),
            normalized_voxels(reference, resolution=resolution),
            resolution=resolution,
        )
    else:
        raise ValueError(f"Unknown IoU backend: {iou_backend}")
    result.update(alignment=alignment, iou_backend=iou_backend, volume_iou=result["reward"])
    return result


def main() -> None:
    """Keep invalid references masked and invalid generated geometry at zero."""
    resolution = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    options = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    backend = options.get("iou_backend", "brep")
    started = time.monotonic()
    try:
        reference = load_solid(Path(sys.argv[2]))
        target = (
            checked_normalized_solid(reference)
            if backend == "brep"
            else normalized_voxels(reference, resolution=resolution)
        )
    except Exception as exc:
        print(
            json.dumps(dict(reward=0.0, mask_sample=True, failure_kind="deepcad:reference", failure_reason=str(exc)))
        )
        return
    try:
        prediction = load_solid(Path(sys.argv[1]))
    except Exception as exc:
        print(json.dumps(dict(reward=0.0, failure_kind="deepcad:geometry", failure_reason=str(exc))))
        return
    try:
        if options.get("alignment", "rescale") != "rescale":
            raise ValueError("Only fixed-orientation rescale alignment is supported")
        if backend == "brep":
            result = brep_iou(checked_normalized_solid(prediction), target)
        elif backend == "voxel":
            result = voxel_iou(normalized_voxels(prediction, resolution=resolution), target, resolution=resolution)
        else:
            raise ValueError(f"Unknown IoU backend: {backend}")
        result.update(
            iou_backend=backend, alignment="rescale", volume_iou=result["reward"], voxel_iou=result["reward"]
        )
        result["reward_components"] = {
            "voxel_iou": options.get("reward_weights", {}).get("voxel_iou", 1.0) * result["reward"]
        }
        result["reward"] = sum(result["reward_components"].values())
        result["scoring_time_s"] = time.monotonic() - started
    except Exception as exc:
        result = dict(reward=0.0, mask_sample=True, failure_kind="deepcad:infrastructure", failure_reason=str(exc))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
