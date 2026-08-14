# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import zipfile
from pathlib import Path

import pytest

from resources_servers.apex_agents.artifacts import (
    artifact_changes_text,
    artifact_text,
    safe_extract_snapshot,
    snapshot_changes,
)


def _archive(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)


def test_extracts_and_labels_recursive_artifacts(tmp_path: Path) -> None:
    archive = tmp_path / "snapshot.zip"
    output = tmp_path / "out"
    output.mkdir()
    _archive(archive, {"filesystem/reports/final.txt": "answer", ".apps_data/mail/state.json": "{}"})

    files = safe_extract_snapshot(archive, output, max_files=10, max_uncompressed_bytes=1000)
    text, names = artifact_text(output, files, max_total_chars=1000, max_file_chars=1000)

    assert names == [".apps_data/mail/state.json", "filesystem/reports/final.txt"]
    assert "=== filesystem/reports/final.txt ===\nanswer" in text


def test_rejects_zip_slip(tmp_path: Path) -> None:
    archive = tmp_path / "snapshot.zip"
    output = tmp_path / "out"
    output.mkdir()
    _archive(archive, {"filesystem/../../rubric.txt": "leak"})

    with pytest.raises(ValueError, match="unsafe snapshot path"):
        safe_extract_snapshot(archive, output, max_files=10, max_uncompressed_bytes=1000)


def test_rejects_unexpected_root(tmp_path: Path) -> None:
    archive = tmp_path / "snapshot.zip"
    output = tmp_path / "out"
    output.mkdir()
    _archive(archive, {"grader/rubric.txt": "leak"})

    with pytest.raises(ValueError, match="unexpected snapshot root"):
        safe_extract_snapshot(archive, output, max_files=10, max_uncompressed_bytes=1000)


def test_renders_sqlite_application_state(tmp_path: Path) -> None:
    database = tmp_path / "mail.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE messages (recipient TEXT, subject TEXT)")
    connection.execute("INSERT INTO messages VALUES ('person@example.com', 'Completed')")
    connection.commit()
    connection.close()
    root = tmp_path / "root"
    target = root / ".apps_data" / "mail" / "mail.db"
    target.parent.mkdir(parents=True)
    target.write_bytes(database.read_bytes())

    rendered, paths = artifact_text(root, [target], max_total_chars=10_000, max_file_chars=10_000)

    assert paths == [".apps_data/mail/mail.db"]
    assert "Table: messages" in rendered
    assert "person@example.com | Completed" in rendered


def test_snapshot_changes_excludes_unchanged_and_renders_before_after(tmp_path: Path) -> None:
    initial_archive = tmp_path / "initial.zip"
    final_archive = tmp_path / "final.zip"
    _archive(
        initial_archive,
        {
            "filesystem/unchanged.txt": "same",
            "filesystem/modified.txt": "before",
            "filesystem/deleted.txt": "gone",
        },
    )
    _archive(
        final_archive,
        {
            "filesystem/unchanged.txt": "same",
            "filesystem/modified.txt": "after",
            "filesystem/added.txt": "new",
        },
    )
    initial_root = tmp_path / "initial"
    final_root = tmp_path / "final"
    initial_root.mkdir()
    final_root.mkdir()
    initial_files = safe_extract_snapshot(initial_archive, initial_root, max_files=10, max_uncompressed_bytes=1000)
    final_files = safe_extract_snapshot(final_archive, final_root, max_files=10, max_uncompressed_bytes=1000)

    changes = snapshot_changes(initial_root, initial_files, final_root, final_files)
    rendered = artifact_changes_text(changes, max_total_chars=10_000, max_file_chars=2000)

    assert [(change.path, change.change_type) for change in changes] == [
        ("filesystem/added.txt", "added"),
        ("filesystem/deleted.txt", "deleted"),
        ("filesystem/modified.txt", "modified"),
    ]
    assert "filesystem/unchanged.txt" not in rendered
    assert "--- BEFORE ---\nbefore" in rendered
    assert "--- AFTER ---\nafter" in rendered
