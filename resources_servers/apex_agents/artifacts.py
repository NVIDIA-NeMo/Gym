# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe extraction and judge-oriented preprocessing for Apex artifact snapshots.

The sub-artifact fingerprint, the title/hash/similarity matching order, the changed-file flattening and the content
tags are adapted from the Mercor Archipelago grader; snapshot handling, limits and persistence are NeMo Gym code.

Adapted source (https://github.com/Mercor-Intelligence/archipelago, revision 0cb5c476c219a9df637e0bd37fb86b2361f4ab89, Apache-2.0):
  grading/runner/helpers/snapshot_diff/main.py
  grading/runner/helpers/snapshot_diff/match_utils.py
The upstream files carry no per-file copyright notice. NVIDIA modifications are licensed under Apache-2.0.
"""

from __future__ import annotations

import difflib
import hashlib
import shutil
import sqlite3
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from resources_servers.apex_agents.file_extraction import (
    DocumentConverterError,
    SubArtifact,
    extract_file_content,
    extract_file_text,
    read_text_bounded,
    visual_blocks_for_file,
)


_SQLITE_HEADER = b"SQLite format 3\x00"
_MAX_SQLITE_BYTES = 64 * 1024 * 1024
_TRUNCATION_SENTINEL = "\n[...artifact content truncated]"
_DOCX_REDLINES_MARKER = "=== DOCUMENT REDLINES ==="
_DOCX_COMMENTS_MARKER = "=== DOCUMENT COMMENTS ==="


@dataclass(frozen=True)
class ArtifactChange:
    """One added, modified, or deleted path from the snapshot pair."""

    path: str
    change_type: str
    before_path: Path | None
    after_path: Path | None
    artifact_type: str = "file"
    index: int | None = None
    original_index: int | None = None
    title: str | None = None
    old_content: str | None = None
    new_content: str | None = None
    content_diff: str | None = None
    embedded_images_old: list[dict[str, Any]] | None = None
    embedded_images_new: list[dict[str, Any]] | None = None


class SnapshotLimitError(ValueError):
    """A snapshot breaks a content limit: too many files, too many bytes, or a link."""


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
            raise SnapshotLimitError(f"snapshot has {len(members)} files; limit is {max_files}")
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
                raise SnapshotLimitError(f"snapshot links are not allowed: {member.filename!r}")
            total_size += member.file_size
            if total_size > max_uncompressed_bytes:
                raise SnapshotLimitError(f"snapshot expands to more than {max_uncompressed_bytes} bytes")
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


def _extract_text(
    path: Path,
    *,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
) -> str:
    extension = path.suffix.lower()
    with path.open("rb") as stream:
        header = stream.read(4096)
    if header.startswith(_SQLITE_HEADER):
        return _read_sqlite(path)
    if extension:
        return extract_file_text(
            path,
            document_converter_image=document_converter_image,
            allow_host_document_tools=allow_host_document_tools,
        )
    if b"\x00" in header:
        return f"[Binary file, {path.stat().st_size} bytes]"
    return read_text_bounded(path)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sub_artifact_fingerprint(artifact: SubArtifact) -> str:
    """Port of Archipelago's content-and-image sub-artifact fingerprint."""
    image_keys = sorted(
        image.get("url", "") or image.get("hash", "") or str(image.get("caption", ""))
        for image in artifact.images
        if image
    )
    combined = artifact.content
    if image_keys:
        combined += "\n---IMAGES---\n" + "\n".join(image_keys)
    return hashlib.md5(combined.encode()).hexdigest()


def _match_sub_artifacts_by_content(
    original: list[SubArtifact],
    final: list[SubArtifact],
    *,
    similarity_threshold: float = 0.5,
) -> list[tuple[SubArtifact | None, SubArtifact | None, str]]:
    """Port Archipelago's title/hash/similarity matching order exactly."""
    matches: list[tuple[SubArtifact | None, SubArtifact | None, str]] = []
    unmatched_originals = list(original)
    unmatched_finals: list[SubArtifact] = []
    artifact_type = original[0].type if original else final[0].type if final else None

    if artifact_type == "sheet":
        originals_by_title: dict[str, SubArtifact] = {}
        for artifact in original:
            if artifact.title and artifact.title not in originals_by_title:
                originals_by_title[artifact.title] = artifact
        for artifact in final:
            matched = originals_by_title.get(artifact.title or "")
            if matched is not None and matched in unmatched_originals:
                match_type = (
                    "unchanged"
                    if _sub_artifact_fingerprint(matched) == _sub_artifact_fingerprint(artifact)
                    else "modified"
                )
                matches.append((matched, artifact, match_type))
                unmatched_originals.remove(matched)
            else:
                unmatched_finals.append(artifact)
    else:
        unmatched_finals = list(final)

    originals_by_hash: dict[str, list[SubArtifact]] = {}
    for artifact in unmatched_originals:
        originals_by_hash.setdefault(_sub_artifact_fingerprint(artifact), []).append(artifact)

    still_unmatched_finals: list[SubArtifact] = []
    for artifact in unmatched_finals:
        candidates = originals_by_hash.get(_sub_artifact_fingerprint(artifact), [])
        if candidates:
            matched = candidates.pop(0)
            matches.append((matched, artifact, "unchanged"))
            unmatched_originals.remove(matched)
        else:
            still_unmatched_finals.append(artifact)

    remaining_unmatched_finals: list[SubArtifact] = []
    for artifact in still_unmatched_finals:
        best_match: SubArtifact | None = None
        best_score = 0.0
        for candidate in unmatched_originals:
            score = difflib.SequenceMatcher(None, candidate.content, artifact.content).ratio()
            if score > best_score and score >= similarity_threshold:
                best_match = candidate
                best_score = score
        if best_match is None:
            remaining_unmatched_finals.append(artifact)
        else:
            matches.append((best_match, artifact, "modified"))
            unmatched_originals.remove(best_match)

    matches.extend((artifact, None, "deleted") for artifact in unmatched_originals)
    matches.extend((None, artifact, "created") for artifact in remaining_unmatched_finals)
    return matches


def _sub_artifact_changes(
    *,
    path: str,
    before_path: Path | None,
    after_path: Path | None,
    document_converter_image: str | None,
    allow_host_document_tools: bool = False,
) -> list[ArtifactChange] | None:
    """Flatten a changed presentation/workbook using Archipelago's diff semantics."""
    converter: dict[str, Any] = {
        "document_converter_image": document_converter_image,
        "allow_host_document_tools": allow_host_document_tools,
    }
    before = extract_file_content(before_path, **converter).sub_artifacts if before_path is not None else []
    after = extract_file_content(after_path, **converter).sub_artifacts if after_path is not None else []
    if (before_path is not None and not before) or (after_path is not None and not after):
        return None

    matches = _match_sub_artifacts_by_content(before, after)
    final_to_original = {
        final.index: original.index
        for original, final, match_type in matches
        if match_type in {"unchanged", "modified"} and original is not None and final is not None
    }
    sorted_final_indices = sorted(final_to_original)
    changes: list[ArtifactChange] = []
    for original, final, match_type in matches:
        if match_type == "unchanged":
            continue
        if match_type == "created":
            assert final is not None
            placed_after = next(
                (final_to_original[index] for index in reversed(sorted_final_indices) if index < final.index),
                None,
            )
            diff = "\n".join(
                difflib.unified_diff(
                    [],
                    final.content.splitlines(keepends=True),
                    fromfile="(new)",
                    tofile=f"final_{final.index}",
                    lineterm="",
                )
            )
            changes.append(
                ArtifactChange(
                    path=path,
                    change_type="added",
                    before_path=None,
                    after_path=after_path,
                    artifact_type=final.type,
                    index=final.index,
                    original_index=placed_after,
                    title=final.title,
                    new_content=final.content,
                    content_diff=diff or None,
                    embedded_images_new=final.images or None,
                )
            )
        elif match_type == "deleted":
            assert original is not None
            diff = "\n".join(
                difflib.unified_diff(
                    original.content.splitlines(keepends=True),
                    [],
                    fromfile=f"original_{original.index}",
                    tofile="(deleted)",
                    lineterm="",
                )
            )
            changes.append(
                ArtifactChange(
                    path=path,
                    change_type="deleted",
                    before_path=before_path,
                    after_path=None,
                    artifact_type=original.type,
                    index=original.index,
                    title=original.title,
                    old_content=original.content,
                    content_diff=diff or None,
                    embedded_images_old=original.images or None,
                )
            )
        else:
            assert original is not None and final is not None
            diff_lines = list(
                difflib.unified_diff(
                    original.content.splitlines(keepends=True),
                    final.content.splitlines(keepends=True),
                    fromfile=f"original_{original.index}",
                    tofile=f"final_{final.index}",
                    lineterm="",
                )
            )
            if not diff_lines and original.images == final.images:
                continue
            changes.append(
                ArtifactChange(
                    path=path,
                    change_type="modified",
                    before_path=before_path,
                    after_path=after_path,
                    artifact_type=final.type,
                    index=final.index,
                    original_index=original.index,
                    title=final.title or original.title,
                    old_content=original.content,
                    new_content=final.content,
                    content_diff="\n".join(diff_lines) or None,
                    embedded_images_old=original.images or None,
                    embedded_images_new=final.images or None,
                )
            )
    return changes


def snapshot_changes(
    initial_root: Path,
    initial_files: list[Path],
    final_root: Path,
    final_files: list[Path],
    *,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
) -> list[ArtifactChange]:
    """Compare safely extracted snapshots and return only induced changes.

    Each changed file is read here, once: its text, and for a modified file its diff, travel on the change, so
    grading every criterion re-reads nothing.
    """

    def text_of(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return _extract_text(
                path,
                document_converter_image=document_converter_image,
                allow_host_document_tools=allow_host_document_tools,
            )
        except DocumentConverterError:
            raise
        except Exception as exc:
            return f"[Error reading artifact: {exc}]"

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
        if Path(relative).suffix.lower() in {".pdf", ".pptx", ".xls", ".xlsm", ".xlsx"}:
            try:
                sub_changes = _sub_artifact_changes(
                    path=relative,
                    before_path=before_path,
                    after_path=after_path,
                    document_converter_image=document_converter_image,
                    allow_host_document_tools=allow_host_document_tools,
                )
            except DocumentConverterError:
                raise
            except Exception:
                sub_changes = None
            if sub_changes:
                changes.extend(sub_changes)
                continue
            # No slide, sheet or page differs by its text or pictures, yet the bytes do: speaker notes, formatting,
            # comments, hidden sheets and vector graphics are outside those fingerprints. Keep the file as one
            # change, so the judge still sees it and a file-type criterion is not failed for a missing deliverable.
        old_content = text_of(before_path)
        new_content = text_of(after_path)
        content_diff: str | None = None
        if change_type == "modified":
            before_body, _, _ = _split_docx_review_sections(before_path, old_content or "")
            after_body, _, _ = _split_docx_review_sections(after_path, new_content or "")
            content_diff = _file_diff(relative, before_body, after_body)
        changes.append(
            ArtifactChange(
                path=relative,
                change_type=change_type,
                before_path=before_path,
                after_path=after_path,
                old_content=old_content,
                new_content=new_content,
                content_diff=content_diff,
            )
        )
    return changes


def _file_diff(path: str, before: str, after: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"before/{path}",
            tofile=f"after/{path}",
            lineterm="",
        )
    )


def _split_docx_review_sections(path: Path | None, content: str) -> tuple[str, str | None, str | None]:
    if path is None or path.suffix.lower() != ".docx":
        return content, None, None
    redlines_at = content.find(_DOCX_REDLINES_MARKER)
    comments_at = content.find(_DOCX_COMMENTS_MARKER)
    starts = [position for position in (redlines_at, comments_at) if position >= 0]
    if not starts:
        return content, None, None

    body = content[: min(starts)].rstrip()
    redlines: str | None = None
    comments: str | None = None
    if redlines_at >= 0:
        redlines_end = comments_at if comments_at > redlines_at else len(content)
        redlines = content[redlines_at:redlines_end].strip()
    if comments_at >= 0:
        comments = content[comments_at:].strip()
    return body, redlines, comments


def _render_tagged_sections(sections: list[tuple[str, str]], *, max_chars: int) -> str:
    """Render Archipelago content tags while bounding each section without breaking XML."""

    def render(contents: list[str]) -> str:
        return "\n\n".join(
            f"<{tag}>\n{content}\n</{tag}>" for (tag, _), content in zip(sections, contents, strict=True)
        )

    contents = [content for _, content in sections]
    full = render(contents)
    if len(full) <= max_chars:
        return full

    empty_size = len(render(["" for _ in sections]))
    available = max(0, max_chars - empty_size)
    allocations = [0] * len(contents)
    pending = set(range(len(contents)))
    remaining = available
    while pending:
        share = remaining // len(pending)
        fitting = [index for index in pending if len(contents[index]) <= share]
        if not fitting:
            for offset, index in enumerate(sorted(pending)):
                allocations[index] = share + (1 if offset < remaining % len(pending) else 0)
            break
        for index in fitting:
            allocations[index] = len(contents[index])
            remaining -= allocations[index]
            pending.remove(index)

    bounded: list[str] = []
    for content, budget in zip(contents, allocations, strict=True):
        if len(content) <= budget:
            bounded.append(content)
        elif budget <= len(_TRUNCATION_SENTINEL):
            bounded.append(content[:budget])
        else:
            bounded.append(content[: budget - len(_TRUNCATION_SENTINEL)] + _TRUNCATION_SENTINEL)
    return render(bounded)


def _stored_or_read(stored: str | None, path: Path | None, missing: str) -> str:
    """A file-level change's text: what ``snapshot_changes`` read, else read now for a change built without it."""
    if path is None:
        return missing
    if stored is not None:
        return stored
    try:
        return _extract_text(path)
    except Exception as exc:
        return f"[Error reading artifact: {exc}]"


def artifact_change_text(change: ArtifactChange, *, max_chars: int) -> str:
    """Render one change with Archipelago's exact content-tag structure."""

    if change.artifact_type == "file":
        before = _stored_or_read(change.old_content, change.before_path, "")
        after = _stored_or_read(change.new_content, change.after_path, "")
        before_body, _, _ = _split_docx_review_sections(change.before_path, before)
        after_body, tracked_changes, comments = _split_docx_review_sections(change.after_path, after)
    else:
        before = change.old_content or ""
        after = change.new_content or ""
        before_body = before
        after_body = after
        tracked_changes = None
        comments = None

    sections: list[tuple[str, str]] = []
    if change.change_type in {"added", "modified"}:
        if tracked_changes:
            sections.append(("tracked_changes", tracked_changes))
        if comments:
            sections.append(("document_comments", comments))
    if change.change_type == "added":
        if after_body:
            sections.append(("created_content", after_body))
    elif change.change_type == "deleted":
        if before:
            sections.append(("deleted_content", before))
    else:
        diff = change.content_diff
        if diff is None:
            diff = _file_diff(change.path, before_body, after_body)
        if diff:
            sections.append(("diff", diff))
        if after_body:
            sections.append(("updated_content", after_body))
    return _render_tagged_sections(sections, max_chars=max_chars) if sections else ""


def artifact_change_content(change: ArtifactChange, *, max_chars: int | None = None) -> str:
    """Render the content Archipelago exposes to its artifact-selection step."""

    if change.artifact_type != "file":
        if change.change_type == "added":
            content = change.new_content or ""
        elif change.change_type == "deleted":
            content = change.old_content or ""
        else:
            content = change.content_diff or ""
    else:
        before = _stored_or_read(change.old_content, change.before_path, "[File did not exist]")
        after = _stored_or_read(change.new_content, change.after_path, "[File was deleted]")
        if change.change_type == "added":
            content = after
        elif change.change_type == "deleted":
            content = before
        elif change.content_diff is not None and change.after_path.suffix.lower() != ".docx":
            # The diff snapshot_changes computed; only a DOCX diff differs here, by its review sections.
            content = change.content_diff
        else:
            content = _file_diff(change.path, before, after)
    if max_chars is not None and len(content) > max_chars:
        return content[:max_chars] + "\n[...artifact selection content truncated]"
    return content


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


def visual_file_blocks(
    root: Path,
    path: Path,
    position: int,
    *,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
    changed_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Render one artifact, labelled by its position among the rendered files and its flattened path."""
    return visual_blocks_for_file(
        path,
        display_name=f"{position:04d}__{path.relative_to(root).as_posix().replace('/', '__')}",
        document_converter_image=document_converter_image,
        allow_host_document_tools=allow_host_document_tools,
        changed_indices=changed_indices,
    )
