# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from benchmarks.visualwebarena import prepare as visualwebarena_prepare


def _write_source(root: Path, count: int) -> tuple[Path, str]:
    image_path = root / "visualwebarena" / "shopping" / "task_0" / "input_0.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fixture")
    rows = [
        {
            "id": f"visualwebarena-{index}",
            "ques": f"Task {index}",
            "web_name": ["shopping"],
            "web": ["__SHOPPING__"],
            "image": ["visualwebarena/shopping/task_0/input_0.png"] if index == 0 else [],
            "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "fixture"}},
        }
        for index in range(count)
    ]
    source = root / "visualwebarena.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


def test_prepare_validates_images_and_writes_model_neutral_rows(tmp_path, monkeypatch) -> None:
    source, digest = _write_source(tmp_path, 908)
    output = tmp_path / "prepared.jsonl"
    monkeypatch.setattr(visualwebarena_prepare, "SOURCE_SHA256", digest)

    assert visualwebarena_prepare.prepare(source, output, tmp_path) == output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 908
    assert rows[0]["web_task"]["runtime_profile"] == "visual_browser"
    assert rows[0]["web_task"]["action_profile"] == "computer_use"
    assert rows[0]["web_task"]["input_images"] == ["visualwebarena/shopping/task_0/input_0.png"]
    assert rows[0]["responses_create_params"]["input"] == []
    assert "tools" not in rows[0]["responses_create_params"]


def test_prepare_rejects_a_different_task_population(tmp_path, monkeypatch) -> None:
    source, digest = _write_source(tmp_path, 1)
    monkeypatch.setattr(visualwebarena_prepare, "SOURCE_SHA256", digest)

    with pytest.raises(ValueError, match="exactly 908 tasks"):
        visualwebarena_prepare.prepare(source, tmp_path / "prepared.jsonl", tmp_path)


def test_prepare_rejects_missing_reference_images(tmp_path, monkeypatch) -> None:
    source, digest = _write_source(tmp_path, 908)
    monkeypatch.setattr(visualwebarena_prepare, "SOURCE_SHA256", digest)
    (tmp_path / "visualwebarena" / "shopping" / "task_0" / "input_0.png").unlink()

    with pytest.raises(FileNotFoundError, match="missing 1 referenced image"):
        visualwebarena_prepare.prepare(source, tmp_path / "prepared.jsonl", tmp_path)


def test_write_env_is_private_and_rejects_display_sharing(tmp_path) -> None:
    env_path = tmp_path / "env.yaml"
    assert visualwebarena_prepare.write_env(
        env_path,
        input_jsonl=tmp_path / "input.jsonl",
        output_jsonl=tmp_path / "output.jsonl",
        source_root=tmp_path / "source",
    )
    content = env_path.read_text()
    assert "benchmarks/visualwebarena/configs/nano_omni.yaml" in content
    assert "agent_name: visualwebarena_benchmark_agent" in content
    assert "visualwebarena_source_root:" in content
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600

    with pytest.raises(ValueError, match="one DISPLAY"):
        visualwebarena_prepare.write_env(
            tmp_path / "other.yaml",
            input_jsonl=tmp_path / "input.jsonl",
            output_jsonl=tmp_path / "output.jsonl",
            source_root=tmp_path / "source",
            concurrency=2,
        )
