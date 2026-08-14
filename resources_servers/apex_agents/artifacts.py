# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe extraction and judge-oriented preprocessing for Apex artifact snapshots."""

from __future__ import annotations

import difflib
import hashlib
import shutil
import sqlite3
import stat
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


_SQLITE_HEADER = b"SQLite format 3\x00"
_MAX_SQLITE_BYTES = 64 * 1024 * 1024
_TEXT_EXTENSIONS = {".log", ".py", ".sh", ".sql", ".yaml", ".yml"}


@dataclass(frozen=True)
class ArtifactChange:
    """One added, modified, or deleted path from the snapshot pair."""

    path: str
    change_type: str
    before_path: Path | None
    after_path: Path | None


def safe_extract_snapshot(
    archive_path: Path,
    output_dir: Path,
    *,
    max_files: int,
    max_uncompressed_bytes: int,
) -> list[Path]:
    """Extract a harness snapshot while rejecting links, zip-slip, and zip bombs."""
    extracted: list[Path] = []
    total_size = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) > max_files:
            raise ValueError(f"snapshot has {len(members)} files; limit is {max_files}")
        for member in members:
            rel = PurePosixPath(member.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe snapshot path: {member.filename!r}")
            if not rel.parts or rel.parts[0] not in {"filesystem", ".apps_data"}:
                raise ValueError(f"unexpected snapshot root: {member.filename!r}")
            nested = rel.parts[1:]
            if rel.parts[0] == "filesystem" and any(part.startswith(".") for part in nested):
                continue
            if rel.parts[0] == ".apps_data" and nested and nested[-1].startswith("."):
                continue
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"snapshot links are not allowed: {member.filename!r}")
            total_size += member.file_size
            if total_size > max_uncompressed_bytes:
                raise ValueError(f"snapshot expands to more than {max_uncompressed_bytes} bytes")
            target = output_dir.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.append(target)
    return extracted


def _quote_sqlite_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _read_sqlite(path: Path) -> str:
    """Render bounded application state without executing writable SQL."""
    if path.stat().st_size > _MAX_SQLITE_BYTES:
        return f"[SQLite database, {path.stat().st_size} bytes; too large to render]"
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.enable_load_extension(False)
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT 21"
            )
        ]
        output: list[str] = []
        for table in tables[:20]:
            cursor = connection.execute(f"SELECT * FROM {_quote_sqlite_identifier(table)} LIMIT 101")
            columns = [str(description[0]) for description in cursor.description or []]
            rows = cursor.fetchall()
            output.append(f"Table: {table}\nColumns: {', '.join(columns)}")
            for row in rows[:100]:
                values = []
                for value in row:
                    if isinstance(value, bytes):
                        rendered = f"[blob: {len(value)} bytes]"
                    else:
                        rendered = str(value)
                    values.append(rendered[:2000])
                output.append(" | ".join(values))
            if len(rows) > 100:
                output.append("[...more rows omitted]")
        if len(tables) > 20:
            output.append("[...more tables omitted]")
        return "\n".join(output) or "[SQLite database has no user tables]"
    finally:
        connection.close()


def _extract_text(path: Path) -> str:
    from responses_api_agents.stirrup_agent.file_reader import _extract_text as gdpval_extract_text

    extension = path.suffix.lower()
    with path.open("rb") as stream:
        header = stream.read(len(_SQLITE_HEADER))
    if header == _SQLITE_HEADER:
        return _read_sqlite(path)
    if extension in _TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    if extension:
        return gdpval_extract_text(path, extension)
    data = path.read_bytes()
    if b"\x00" in data[:4096]:
        return f"[Binary file, {len(data)} bytes]"
    return data.decode("utf-8", errors="replace").strip()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_changes(
    initial_root: Path,
    initial_files: list[Path],
    final_root: Path,
    final_files: list[Path],
) -> list[ArtifactChange]:
    """Compare safely extracted snapshots and return only induced changes."""
    before = {path.relative_to(initial_root).as_posix(): path for path in initial_files}
    after = {path.relative_to(final_root).as_posix(): path for path in final_files}
    changes: list[ArtifactChange] = []
    for relative in sorted(before.keys() | after.keys()):
        before_path = before.get(relative)
        after_path = after.get(relative)
        if before_path is None:
            change_type = "added"
        elif after_path is None:
            change_type = "deleted"
        elif before_path.stat().st_size == after_path.stat().st_size and _digest(before_path) == _digest(after_path):
            continue
        else:
            change_type = "modified"
        changes.append(
            ArtifactChange(
                path=relative,
                change_type=change_type,
                before_path=before_path,
                after_path=after_path,
            )
        )
    return changes


def artifact_change_text(change: ArtifactChange, *, max_chars: int) -> str:
    """Render one artifact as before/after text plus a bounded unified diff."""

    def render(path: Path | None, missing: str) -> str:
        if path is None:
            return missing
        try:
            return _extract_text(path)
        except Exception as exc:
            return f"[Error reading artifact: {exc}]"

    before = render(change.before_path, "[File did not exist]")
    after = render(change.after_path, "[File was deleted]")
    diff = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"before/{change.path}",
            tofile=f"after/{change.path}",
            lineterm="",
        )
    )
    section = (
        f"Path: {change.path}\n"
        f"Change: {change.change_type}\n"
        f"--- BEFORE ---\n{before or '[Empty file]'}\n"
        f"--- AFTER ---\n{after or '[Empty file]'}\n"
        f"--- CHANGE DIFF ---\n{diff or '[No extractable textual diff]'}"
    )
    if len(section) > max_chars:
        return section[:max_chars] + "\n[...artifact change truncated]"
    return section


def artifact_changes_text(
    changes: list[ArtifactChange],
    *,
    max_total_chars: int,
    max_file_chars: int,
) -> str:
    """Render a bounded, path-labelled before/after change log."""
    parts: list[str] = []
    remaining = max_total_chars
    for index, change in enumerate(changes, 1):
        section = f"=== ARTIFACT {index} ===\n{artifact_change_text(change, max_chars=max_file_chars)}"
        if len(section) > remaining:
            if remaining > 200:
                parts.append(section[:remaining] + "\n[...all artifact changes truncated]")
            break
        parts.append(section)
        remaining -= len(section)
    return "\n\n".join(parts)


def artifact_text(
    root: Path,
    files: list[Path],
    *,
    max_total_chars: int,
    max_file_chars: int,
) -> tuple[str, list[str]]:
    """Render recursively extracted files into bounded, path-labelled judge text."""
    parts: list[str] = []
    names: list[str] = []
    remaining = max_total_chars
    for path in sorted(files):
        rel = path.relative_to(root).as_posix()
        names.append(rel)
        try:
            content = _extract_text(path)
        except Exception as exc:
            content = f"[Error reading artifact: {exc}]"
        if len(content) > max_file_chars:
            content = content[:max_file_chars] + "\n[...artifact truncated]"
        section = f"=== {rel} ===\n{content or '[Empty file]'}"
        if len(section) > remaining:
            if remaining > 200:
                parts.append(section[:remaining] + "\n[...all artifacts truncated]")
            break
        parts.append(section)
        remaining -= len(section)
    return "\n\n".join(parts), names


def visual_content_blocks(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    """Reuse GDP-Val's rendering pipeline after flattening recursive paths."""
    from responses_api_agents.stirrup_agent.file_reader import convert_deliverables_to_content_blocks

    staging = root / ".visual_staging" / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files):
        rel = path.relative_to(root).as_posix().replace("/", "__")
        shutil.copy2(path, staging / f"{index:04d}__{rel}")
    return convert_deliverables_to_content_blocks(str(staging))
