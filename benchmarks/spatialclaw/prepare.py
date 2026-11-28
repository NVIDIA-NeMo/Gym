# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a SpatialClaw-native dataset as portable NeMo Gym JSONL rows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from responses_api_agents.spatialclaw_agent.app import (
    SPATIALCLAW_COMMIT,
    SPATIALCLAW_URL,
    _checkout_lock,
    _config_path,
    _install_source_path,
    _validate_spatialclaw_checkout,
    ensure_spatialclaw_checkout,
)


BENCHMARK_DIR = Path(__file__).parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
SERVER_DIR = REPO_ROOT / "resources_servers" / "spatialclaw"


def _ensure_server_venv() -> Path:
    """Create the native scorer venv once so preparation has loader dependencies."""
    venv_python = SERVER_DIR / ".venv" / "bin" / "python"
    completion_marker = SERVER_DIR / ".venv" / ".spatialclaw-requirements-installed"
    with _checkout_lock(SERVER_DIR / ".venv.setup.lock"):
        if venv_python.exists() and completion_marker.exists():
            return venv_python
        if not venv_python.exists():
            subprocess.run(
                ["uv", "venv", "--python", sys.executable, ".venv"],
                check=True,
                cwd=SERVER_DIR,
            )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(venv_python), "-r", "requirements.txt"],
            check=True,
            cwd=SERVER_DIR,
        )
        completion_marker.touch()
    return venv_python


def _source_root(spatialclaw_root: str | None, source_cache_root: str | None) -> Path:
    configured = spatialclaw_root or os.environ.get("SPATIALCLAW_ROOT")
    if configured:
        return _validate_spatialclaw_checkout(Path(configured), SPATIALCLAW_COMMIT)
    cache_root = Path(source_cache_root).expanduser().resolve() if source_cache_root else BENCHMARK_DIR / ".sources"
    return ensure_spatialclaw_checkout(SPATIALCLAW_URL, SPATIALCLAW_COMMIT, cache_root)


def _portable_path(value: str, data_root: Path) -> str:
    if value.startswith(("data:", "http://", "https://", "file://")):
        return value
    path = Path(value).expanduser().resolve()
    try:
        return str(path.relative_to(data_root))
    except ValueError as exc:
        raise ValueError(f"SpatialClaw media path is outside data_root: {path}") from exc


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


def _json_metadata(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sample_metadata(sample: Any, data_root: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    video_path = getattr(sample, "video", None)
    video_paths = getattr(sample, "video_paths", None)
    image_groups = getattr(sample, "image_groups", None)
    images = getattr(sample, "images", None) or []

    if video_path:
        metadata["video_path"] = _portable_path(str(video_path), data_root)
    elif video_paths:
        metadata["video_paths"] = _json_metadata([_portable_path(str(path), data_root) for path in video_paths])
    elif image_groups:
        metadata["image_groups"] = _json_metadata(
            [[_portable_path(str(path), data_root) for path in group] for group in image_groups]
        )
    elif images:
        metadata["image_paths"] = _json_metadata([_portable_path(str(path), data_root) for path in images])

    ref_images = getattr(sample, "ref_images", None)
    if ref_images:
        metadata["ref_image_paths"] = _json_metadata([_portable_path(str(path), data_root) for path in ref_images])

    list_fields = {
        "frame_indices": int,
        "frame_indices_groups": lambda group: [int(value) for value in group],
        "fps_per_video": float,
        "total_frames_per_video": int,
        "duration_per_video": float,
        "video_names": str,
    }
    for name, converter in list_fields.items():
        value = getattr(sample, name, None)
        if value:
            metadata[name] = _json_metadata([converter(item) for item in value])

    return metadata


def _prepare_in_process(
    *,
    dataset_config: str,
    output_fpath: str | Path,
    spatialclaw_root: str | None = None,
    data_root: str | None = None,
    source_cache_root: str | None = None,
) -> Path:
    """Load one pinned SpatialClaw dataset and write the corresponding Gym rows."""
    root = _source_root(spatialclaw_root, source_cache_root)
    _install_source_path(root)

    from spatial_agent.config import SpatialAgentConfig, set_config
    from spatial_agent.evals.factory import BenchmarkFactory

    spatial_config = SpatialAgentConfig()
    spatial_config._load_from_envs()
    spatial_config.update_from_dataset_json(_config_path(root, dataset_config, "dataset"))
    set_config(spatial_config)

    configured_data_root = data_root or os.environ.get("SPATIALCLAW_DATA_ROOT")
    resolved_data_root = (
        Path(configured_data_root).expanduser().resolve() if configured_data_root else (root / "data").resolve()
    )
    benchmark = BenchmarkFactory.create_benchmark(
        spatial_config.benchmark,
        data_root=str(resolved_data_root),
        question_type=spatial_config.question_type,
    )
    if benchmark is None:
        raise RuntimeError(f"Dataset config selected no benchmark: {dataset_config}")

    destination = Path(output_fpath)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for sample in benchmark.data:
            row = {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": _instruction(benchmark, sample)}],
                        }
                    ],
                    "metadata": _sample_metadata(sample, resolved_data_root),
                },
                "sample_id": str(sample.sample_id),
                "answer": getattr(sample, "answer", None),
                "question_type": str(getattr(sample, "question_type", "")),
                "benchmark_name": spatial_config.benchmark,
                "dataset_config": dataset_config,
            }
            output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print(f"Wrote {len(benchmark.data)} SpatialClaw rows to {destination}")
    return destination


def prepare(
    *,
    dataset_config: str,
    output_fpath: str | Path,
    spatialclaw_root: str | None = None,
    data_root: str | None = None,
    source_cache_root: str | None = None,
    use_current_environment: bool = False,
) -> Path:
    """Prepare in the scorer venv, or directly when explicitly requested by tests."""
    kwargs = {
        "dataset_config": dataset_config,
        "output_fpath": str(output_fpath),
        "spatialclaw_root": spatialclaw_root,
        "data_root": data_root,
        "source_cache_root": source_cache_root,
    }
    if use_current_environment:
        return _prepare_in_process(**kwargs)

    venv_python = _ensure_server_venv()
    serialized = json.dumps(kwargs)
    helper = (
        "import json; "
        "from benchmarks.spatialclaw.prepare import _prepare_in_process; "
        f"_prepare_in_process(**json.loads({serialized!r}))"
    )
    subprocess.run([str(venv_python), "-c", helper], check=True, cwd=REPO_ROOT)
    return Path(output_fpath)
