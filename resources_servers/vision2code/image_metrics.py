# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/image2code/vision2code (Apache-2.0).
# Source: vision2code/evaluation/image_metrics.py; base revision 000f2e162dd62fac5637c718e018d7123b3aff1e.
# Modified source SHA-256: 9d6b487e325d4ad7ad941140fddfe75948b7a60edf5c0c8e337bfc53771d6ebe (see NOTICE).
"""Foreground and edge similarity metrics with Blend A reward aggregation."""

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
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _background_and_overlap(reference: Image.Image, candidate: Image.Image) -> tuple[np.ndarray, float]:
    """Compute exact full-resolution statistics with bounded temporary arrays."""
    width, height = reference.size
    overlap_width, overlap_height = min(width, candidate.width), min(height, candidate.height)
    # About 256k pixels per chunk, independent of the full canvas dimensions.
    rows = max(1, 262_144 // width)
    histogram = np.zeros(32768, dtype=np.int64)
    matching = 0

    def labels(pixels: np.ndarray) -> np.ndarray:
        bins = pixels.astype(np.uint16) >> 3
        return (bins[..., 0] << 10) + (bins[..., 1] << 5) + bins[..., 2]

    for top in range(0, height, rows):
        bottom = min(top + rows, height)
        pixels = np.asarray(reference.crop((0, top, width, bottom)))
        histogram += np.bincount(labels(pixels).ravel(), minlength=32768)
        if top < overlap_height:
            other = np.asarray(candidate.crop((0, top, overlap_width, min(bottom, overlap_height))))
            matching += int(np.count_nonzero(np.all(pixels[: other.shape[0], :overlap_width] == other, axis=2)))

    # argmax preserves the original smallest-bin tie break. Per-channel value
    # histograms recover np.median exactly, including half-integer even medians.
    peak = int(histogram.argmax())
    channels = np.zeros((3, 256), dtype=np.int64)
    for top in range(0, height, rows):
        pixels = np.asarray(reference.crop((0, top, width, min(top + rows, height))))
        selected = pixels[labels(pixels) == peak]
        for channel in range(3):
            channels[channel] += np.bincount(selected[:, channel], minlength=256)
    count = int(histogram[peak])
    middle = [(count - 1) // 2 + 1, count // 2 + 1]
    background = np.array([np.searchsorted(channel.cumsum(), middle).mean() for channel in channels])
    return background, matching / (overlap_width * overlap_height)


def image_metrics(reference: str | Path, candidate: str | Path) -> dict[str, JsonValue]:
    """Compare images on the reference canvas using the upstream pixel protocol."""
    from scipy.ndimage import distance_transform_edt, sobel

    reference_path, candidate_path = reference, candidate

    def rgb(path: str | Path) -> Image.Image:
        with Image.open(path) as image:
            transposed = ImageOps.exif_transpose(image)
            if transposed.mode == "RGB":
                return transposed
            rgba = transposed.convert("RGBA")
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
    background, overlap = _background_and_overlap(reference, candidate)
    same_dimensions = reference.size == candidate.size
    # Pasting a same-size opaque RGB candidate covers the entire canvas. Reuse
    # it rather than allocating and copying another full-resolution image.
    canvas = candidate
    if not same_dimensions:
        canvas = Image.new("RGB", reference.size, tuple(int(v) for v in background))
        canvas.paste(candidate, (0, 0))
    scale = min(1.0, 1024 / max(reference.size))
    size = tuple(max(1, round(v * scale)) for v in reference.size)
    a = np.asarray(reference.resize(size, Image.Resampling.LANCZOS), dtype=float)
    reference.close()
    b = np.asarray(canvas.resize(size, Image.Resampling.LANCZOS), dtype=float)
    canvas.close()
    if canvas is not candidate:
        candidate.close()
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
        "same_dimensions": same_dimensions,
        "reference_sha256": image_sha256(reference_path),
        "candidate_sha256": image_sha256(candidate_path),
    }
    components = [scores[k] for k in ("foreground_rgb_similarity", "foreground_tolerant_f1", "edge_tolerant_f1")]
    scores["deterministic_weighted"] = (
        sum(w * v for w, v in zip((0.2, 0.4, 0.4), components)) if all(v is not None for v in components) else None
    )
    return scores


def apply_blend_a(
    rating: dict[str, JsonValue],
    metrics: dict[str, JsonValue] | None = None,
    *,
    apply_judge_caps: bool = False,
) -> dict[str, JsonValue]:
    """Blend pixel metrics with the raw judge score, optionally applying its rubric caps."""
    result = dict(rating)
    failed = rating.get("execution_status") != "ok"
    llm = float(rating["raw_score_0_to_5"])
    if apply_judge_caps:
        # Keep the judge term unrounded: substituting the final rating also changes
        # uncapped scores through its one-decimal rounding.
        llm = min([llm, *(float(cap["cap_to"]) for cap in rating.get("applied_caps", []))])
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
        scoring_method="blend_a_capped" if apply_judge_caps else "blend_a",
        score_version=BLEND_A_VERSION + ("_judge_caps" if apply_judge_caps else ""),
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
