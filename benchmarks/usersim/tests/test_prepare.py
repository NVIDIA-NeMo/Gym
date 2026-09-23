# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.usersim import prepare as prepare_module


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{"first_name": "Morgan", "age": 42}]), path)


def test_prepare_invokes_usersim_panel_and_records_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "usersim.jsonl"
    example_path = tmp_path / "example.jsonl"
    example_path.write_text(
        '{"task_id":{"taskset":"usersim:example","task_id":"1042"},'
        '"task_input":{"sampling":{"locale":"en_US","seed":1042}}}\n'
    )
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)
    monkeypatch.setattr(prepare_module, "EXAMPLE_FPATH", example_path)
    monkeypatch.setattr(prepare_module.shutil, "which", lambda executable: f"/bin/{executable}")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_panel(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((command, kwargs))
        _write_parquet(Path(command[-1]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare_module.subprocess, "run", fake_panel)

    result = prepare_module.prepare(personas_cache_dir=tmp_path / "personas", personas_panel_size=1)

    assert result == output_path.absolute()
    assert json.loads(output_path.read_text())["task_id"]["taskset"] == "usersim:benchmark"
    command, kwargs = calls[0]
    assert command[:6] == ["/bin/usersim", "panel", "--locale", "en_US", "--num-personas", "1"]
    assert "env" not in kwargs
    panel_path = tmp_path / "personas" / "0.0.2" / "panels" / "en_US.parquet"
    manifest = json.loads(panel_path.with_suffix(".manifest.json").read_text())
    assert manifest["panel_rows"] == 1
    assert len(manifest["panel_sha256"]) == 64
    assert manifest["generator"] == "usersim panel"


def test_prepare_reuses_matching_panel_without_invoking_usersim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "usersim.jsonl"
    example_path = tmp_path / "example.jsonl"
    example_path.write_text(
        '{"task_id":{"taskset":"usersim:example","task_id":"1042"},'
        '"task_input":{"sampling":{"locale":"en_US","seed":1042}}}\n'
    )
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)
    monkeypatch.setattr(prepare_module, "EXAMPLE_FPATH", example_path)
    monkeypatch.setattr(prepare_module.shutil, "which", lambda executable: f"/bin/{executable}")

    def fake_panel(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        _write_parquet(Path(command[-1]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare_module.subprocess, "run", fake_panel)
    prepare_module.prepare(personas_cache_dir=tmp_path / "personas", personas_panel_size=1)
    monkeypatch.setattr(
        prepare_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("matching prepared panel must not invoke UserSim"),
    )

    prepare_module.prepare(personas_cache_dir=tmp_path / "personas", personas_panel_size=1)

    assert output_path.is_file()
    panel_path = tmp_path / "personas" / "0.0.2" / "panels" / "en_US.parquet"
    assert panel_path.with_suffix(".manifest.json").is_file()
