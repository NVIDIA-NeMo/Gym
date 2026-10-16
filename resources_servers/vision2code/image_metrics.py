# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image metrics and Blend A ported from vision2code/evaluation/image_metrics.py.

F1 matches sdg_api_vision2code/submission/scoring.py: reference canvas,
<=1024px analysis, 2px tolerance, no image registration. No external repo needed.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pydantic import JsonValue


BLEND_A_VERSION = "blend_a_v1_fg_rgb_union_mae_f1_1024_tol2"


def image_sha256(path: str | Path) -> str:
    """Hash the original image bytes for scoring provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_metrics(reference: str | Path, candidate: str | Path) -> dict[str, JsonValue]:
    """Compare images on the reference canvas using the upstream pixel protocol."""
    from scipy.ndimage import distance_transform_edt, sobel

    reference_path, candidate_path = reference, candidate

    def rgb(path: str | Path) -> Image.Image:
        with Image.open(path) as image:
            rgba = ImageOps.exif_transpose(image).convert("RGBA")
            return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")

    def tolerant_f1(a: np.ndarray, b: np.ndarray) -> float | None:
        na, nb = int(a.sum()), int(b.sum())
        if not na + nb:
            return None
        da = distance_transform_edt(~a) if na else np.full(a.shape, np.inf)
        db = distance_transform_edt(~b) if nb else np.full(b.shape, np.inf)
        precision = float(np.mean(da[b] <= 2)) if nb else 0.0
        recall = float(np.mean(db[a] <= 2)) if na else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    reference, candidate = rgb(reference), rgb(candidate)
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    height, width = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    overlap = float(np.all(a[:height, :width] == b[:height, :width], axis=2).mean())
    bins = a.astype(np.int32) // 8
    labels = bins[..., 0] * 1024 + bins[..., 1] * 32 + bins[..., 2]
    peak = np.bincount(labels.ravel(), minlength=32768).argmax()
    background = np.median(a[labels == peak], axis=0)
    canvas = Image.new("RGB", reference.size, tuple(int(v) for v in background))
    canvas.paste(candidate, (0, 0))
    scale = min(1.0, 1024 / max(reference.size))
    size = tuple(max(1, round(v * scale)) for v in reference.size)
    a = np.asarray(reference.resize(size, Image.Resampling.LANCZOS), dtype=float)
    b = np.asarray(canvas.resize(size, Image.Resampling.LANCZOS), dtype=float)
    foreground_a = np.linalg.norm(a - background, axis=2) > 32
    foreground_b = np.linalg.norm(b - background, axis=2) > 32
    union = foreground_a | foreground_b
    foreground_rgb = float(np.clip(1 - np.abs(a[union] - b[union]).mean() / 255, 0, 1)) if union.any() else None
    ga, gb = a @ np.array([0.2126, 0.7152, 0.0722]), b @ np.array([0.2126, 0.7152, 0.0722])
    edge_a = np.hypot(sobel(ga / 255, axis=0), sobel(ga / 255, axis=1)) > 0.15
    edge_b = np.hypot(sobel(gb / 255, axis=0), sobel(gb / 255, axis=1)) > 0.15
    scores = {
        "foreground_rgb_similarity": foreground_rgb,
        "foreground_tolerant_f1": tolerant_f1(foreground_a, foreground_b),
        "edge_tolerant_f1": tolerant_f1(edge_a, edge_b),
        "rgb_overlap": overlap,
        "same_dimensions": reference.size == candidate.size,
        "reference_sha256": image_sha256(reference_path),
        "candidate_sha256": image_sha256(candidate_path),
    }
    components = [scores[k] for k in ("foreground_rgb_similarity", "foreground_tolerant_f1", "edge_tolerant_f1")]
    scores["deterministic_weighted"] = (
        sum(w * v for w, v in zip((0.2, 0.4, 0.4), components)) if all(v is not None for v in components) else None
    )
    return scores


def apply_blend_a(rating: dict[str, JsonValue], metrics: dict[str, JsonValue] | None = None) -> dict[str, JsonValue]:
    """Preserve the uncapped LLM rubric and expose the blend as the final score."""
    result = dict(rating)
    failed = rating.get("execution_status") != "ok"
    llm = float(rating["raw_score_0_to_5"])
    if failed:
        metrics = dict.fromkeys(
            ("foreground_rgb_similarity", "foreground_tolerant_f1", "edge_tolerant_f1", "deterministic_weighted"), 0.0
        )
        llm = 0.0
    elif metrics is None or any(
        metrics.get(k) is None
        for k in ("foreground_rgb_similarity", "foreground_tolerant_f1", "edge_tolerant_f1", "deterministic_weighted")
    ):
        raise ValueError("Blend A unavailable: a foreground/edge metric is undefined; weights were not redistributed.")
    components = [metrics[k] for k in ("foreground_rgb_similarity", "foreground_tolerant_f1", "edge_tolerant_f1")]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in components) or not math.isfinite(llm) or not 0 <= llm <= 5:
        raise ValueError("Invalid Blend A component score")
    result.update(
        scoring_method="blend_a",
        score_version=BLEND_A_VERSION,
        llm_score_0_to_5=llm,
        llm_final_rating_0_to_5=rating["final_rating_0_to_5"],
        deterministic_scores=dict(metrics),
        deterministic_score_0_to_5=5 * metrics["deterministic_weighted"],
        score_weights={
            "foreground_rgb_similarity": 0.1,
            "foreground_tolerant_f1": 0.2,
            "edge_tolerant_f1": 0.2,
            "llm_score_0_to_5": 0.5,
        },
        final_rating_0_to_5=round(5 * sum(w * v for w, v in zip((0.1, 0.2, 0.2), components)) + 0.5 * llm, 4),
    )
    return result
