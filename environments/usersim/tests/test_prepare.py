# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from environments.usersim import prepare as prepare_module


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{"first_name": "Morgan", "age": 42}]), path)


def test_prepare_invokes_usersim_panel_and_records_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tasks_path = tmp_path / "example.jsonl"
    monkeypatch.setattr(prepare_module, "TASKS_FPATH", tasks_path)
    monkeypatch.setattr(prepare_module.shutil, "which", lambda executable: f"/bin/{executable}")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_panel(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((command, kwargs))
        _write_parquet(Path(command[-1]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare_module.subprocess, "run", fake_panel)

    result = prepare_module.prepare(personas_cache_dir=tmp_path / "personas", personas_panel_size=1)

    assert result == tasks_path.absolute()
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    assert len(rows) == 14
    assert all(row["task_id"]["taskset"] == "usersim:example" for row in rows)
    assert all(row["task_input"]["scenario"]["persona"]["first_name"] == "Morgan" for row in rows)
    assert {row["task_input"]["usersim_context"]["seed"] for row in rows} == set(range(1001, 1015))
    assert all(
        row["task_input"]["usersim_context"]["usersim_revision"] == prepare_module.USERSIM_REVISION for row in rows
    )
    command, kwargs = calls[0]
    assert command[:12] == [
        "/bin/uv",
        "run",
        "--no-config",
        "--no-project",
        "--isolated",
        "--with-requirements",
        str(prepare_module.PREPARE_REQUIREMENTS_FPATH),
        "usersim",
        "panel",
        "--locale",
        "en_US",
        "--num-personas",
    ]
    assert command[12] == "1"
    assert "env" not in kwargs
    assert Path(str(kwargs["cwd"])).name.startswith("usersim-panel-")
    panel_path = tmp_path / "personas" / "0.0.2" / "panels" / "en_US.parquet"
    manifest = json.loads(panel_path.with_suffix(".manifest.json").read_text())
    assert manifest["panel_rows"] == 1
    assert len(manifest["panel_sha256"]) == 64
    assert manifest["generator"] == "usersim panel"
    assert manifest["usersim_revision"] == prepare_module.USERSIM_REVISION


def test_prepare_reuses_matching_panel_without_invoking_usersim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example_path = tmp_path / "example.jsonl"
    monkeypatch.setattr(prepare_module, "TASKS_FPATH", example_path)
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

    assert example_path.is_file()
    assert len(example_path.read_text().splitlines()) == 14
    panel_path = tmp_path / "personas" / "0.0.2" / "panels" / "en_US.parquet"
    assert panel_path.with_suffix(".manifest.json").is_file()
