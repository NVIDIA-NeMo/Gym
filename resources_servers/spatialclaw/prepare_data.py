# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize SpatialClaw benchmark samples in NeMo Gym Responses format."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _spatialclaw_root() -> Path:
    value = os.environ.get("SPATIALCLAW_ROOT", "")
    if not value:
        raise RuntimeError("SPATIALCLAW_ROOT must point to the pinned SpatialClaw checkout")
    root = Path(value).expanduser().resolve()
    if not (root / "spatial_agent" / "evals" / "factory.py").is_file():
        raise RuntimeError(f"Invalid SpatialClaw checkout: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _dataset_config_path(root: Path, dataset_config: str) -> Path:
    path = Path(dataset_config).expanduser()
    if not path.is_absolute():
        path = root / "spatial_agent" / "config" / "dataset" / path
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(f"SpatialClaw dataset config not found: {path}")
    return path.resolve()


def _portable_dataset_config(root: Path, config_path: Path) -> str:
    config_root = (root / "spatial_agent" / "config" / "dataset").resolve()
    try:
        return config_path.relative_to(config_root).as_posix()
    except ValueError:
        return str(config_path)


def _as_url(value: Any) -> str:
    text = str(value)
    if text.startswith(("data:", "file://", "http://", "https://")):
        return text
    return Path(text).expanduser().resolve().as_uri()


def _instruction(benchmark: Any, sample: Any) -> str:
    instruction = f"{sample.question}\n\n{benchmark.data_specific_prompt}"
    choices = getattr(sample, "choices", None)
    if isinstance(choices, dict):
        for letter, text in choices.items():
            instruction += f"\n{letter}. {text}"
    elif isinstance(choices, list):
        for index, text in enumerate(choices):
            instruction += f"\n{chr(65 + index)}. {text}"
    return instruction


def _media_and_metadata(sample: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    media: list[dict[str, Any]] = []

    video_sources = list(getattr(sample, "video_sources_per_video", None) or [])
    video = getattr(sample, "video", None)
    if not video_sources and video:
        video_sources = [video]
    if video_sources:
        media.extend({"type": "input_video", "video_url": _as_url(path)} for path in video_sources)
        metadata["video_sources_per_video"] = [_as_url(path) for path in video_sources]

    image_groups = getattr(sample, "image_groups", None)
    if image_groups:
        images = [image for group in image_groups for image in group]
        metadata["image_group_sizes"] = [len(group) for group in image_groups]
    else:
        images = list(getattr(sample, "images", None) or [])
    media.extend({"type": "input_image", "image_url": _as_url(path), "detail": "auto"} for path in images)

    for name in (
        "frame_indices",
        "frame_indices_groups",
        "fps",
        "total_video_frames",
        "duration_sec",
        "fps_per_video",
        "total_frames_per_video",
        "duration_per_video",
        "video_names",
    ):
        value = getattr(sample, name, None)
        if value not in (None, [], "", 0, 0.0):
            metadata[name] = value

    ref_images = list(getattr(sample, "ref_images", None) or [])
    if ref_images:
        metadata["ref_images"] = [_as_url(path) for path in ref_images]
    return media, metadata


def prepare_spatialclaw_benchmark(
    *,
    dataset_config: str,
    output_path: str | Path,
) -> Path:
    """Load one canonical SpatialClaw dataset and emit a Gym JSONL."""
    root = _spatialclaw_root()
    config_path = _dataset_config_path(root, dataset_config)
    portable_config = _portable_dataset_config(root, config_path)
    data_root = Path(os.environ.get("SPATIALCLAW_DATA_ROOT", str(root / "data"))).expanduser().resolve()

    from spatial_agent.config import SpatialAgentConfig, set_config
    from spatial_agent.evals.factory import BenchmarkFactory

    spatial_config = SpatialAgentConfig()
    spatial_config._load_from_envs()
    spatial_config.update_from_dataset_json(str(config_path))
    set_config(spatial_config)
    benchmark = BenchmarkFactory.create_benchmark(
        spatial_config.benchmark,
        data_root=str(data_root),
        question_type=spatial_config.question_type,
    )
    if benchmark is None:
        raise RuntimeError(f"No benchmark selected by {config_path}")

    limit_value = os.environ.get("SPATIALCLAW_PREPARE_LIMIT")
    samples = list(benchmark)
    if limit_value:
        limit = int(limit_value)
        if limit <= 0:
            raise ValueError("SPATIALCLAW_PREPARE_LIMIT must be positive")
        samples = samples[:limit]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            media, run_metadata = _media_and_metadata(sample)
            run_metadata.update(
                {
                    "dataset_config": portable_config,
                    "sample_id": str(sample.sample_id),
                    "benchmark": spatial_config.benchmark,
                }
            )
            content = [{"type": "input_text", "text": _instruction(benchmark, sample)}, *media]
            row = {
                "responses_create_params": {
                    "input": [{"role": "user", "type": "message", "content": content}],
                    "metadata": {
                        "spatialclaw": json.dumps(run_metadata, separators=(",", ":")),
                    },
                },
                "expected_answer": str(sample.answer),
                "benchmark": spatial_config.benchmark,
                "sample_id": str(sample.sample_id),
                "dataset_config": portable_config,
                "scoring_mode": "native",
                "verifier_metadata": {
                    "benchmark": spatial_config.benchmark,
                    "sample_id": str(sample.sample_id),
                    "dataset_config": portable_config,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} {spatial_config.benchmark} tasks to {output}")
    return output
