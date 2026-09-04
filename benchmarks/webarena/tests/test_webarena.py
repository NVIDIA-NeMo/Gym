# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import stat

import pytest

from benchmarks.webarena import prepare as webarena_prepare


def _write_source(path, count: int) -> str:
    rows = [
        {
            "id": f"webarena-{index}",
            "ques": f"Task {index}",
            "web_name": ["wikipedia"],
            "web": ["__WIKIPEDIA__"],
            "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "fixture"}},
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_validates_denominator_and_writes_model_neutral_rows(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "prepared.jsonl"
    monkeypatch.setattr(webarena_prepare, "SOURCE_SHA256", _write_source(source, 812))

    assert webarena_prepare.prepare(source, output) == output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 812
    assert rows[0]["web_task"]["runtime_profile"] == "visual_browser"
    assert rows[0]["web_task"]["action_profile"] == "computer_use"
    assert rows[0]["responses_create_params"]["input"] == []
    assert "tools" not in rows[0]["responses_create_params"]


def test_prepare_rejects_a_different_task_population(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    monkeypatch.setattr(webarena_prepare, "SOURCE_SHA256", _write_source(source, 1))

    with pytest.raises(ValueError, match="exactly 812 tasks"):
        webarena_prepare.prepare(source, tmp_path / "prepared.jsonl")


def test_write_env_is_private_and_rejects_display_sharing(tmp_path) -> None:
    env_path = tmp_path / "env.yaml"
    assert webarena_prepare.write_env(
        env_path,
        input_jsonl=tmp_path / "input.jsonl",
        output_jsonl=tmp_path / "output.jsonl",
    )
    content = env_path.read_text()
    assert "benchmarks/webarena/configs/nano_omni.yaml" in content
    assert "agent_name: webarena_benchmark_agent" in content
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600

    with pytest.raises(ValueError, match="one DISPLAY"):
        webarena_prepare.write_env(
            tmp_path / "other.yaml",
            input_jsonl=tmp_path / "input.jsonl",
            output_jsonl=tmp_path / "output.jsonl",
            concurrency=2,
        )
