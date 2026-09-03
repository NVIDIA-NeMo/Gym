# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from resources_servers.workspace_bench.app import _directory_snapshot


def test_directory_snapshot_lists_relative_files(tmp_path: Path) -> None:
    (tmp_path / "answer.md").write_text("grounded answer", encoding="utf-8")

    snapshot = _directory_snapshot(tmp_path)

    assert "## answer.md" in snapshot
    assert "grounded answer" in snapshot


def test_directory_snapshot_respects_byte_budget(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("é" * 100, encoding="utf-8")
    (tmp_path / "b.md").write_text("é" * 100, encoding="utf-8")

    snapshot = _directory_snapshot(tmp_path, max_bytes=40)

    assert len(snapshot.encode("utf-8")) <= 40
    assert "## a.md" in snapshot
    assert "## b.md" in snapshot
