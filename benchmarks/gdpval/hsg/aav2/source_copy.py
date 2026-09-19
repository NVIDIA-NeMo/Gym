# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize only visible selected tasks, retaining symlink provenance."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path


CACHE_NAME = re.compile(r"repeat_[0-9]+_verify_response(?:_(?:[0-9a-f]{12}|[0-9a-f]{16}))?\.json$")


def _scan(source: Path, task_names: set[str], destination: Path | None = None) -> dict:
    source = source.absolute()
    if not source.is_dir():
        raise ValueError(f"source must be a directory: {source}")
    manifest = {"source": str(source), "task_names": sorted(task_names), "links": [], "files": []}

    def visit(path: Path, relative: Path, ancestors: frozenset) -> None:
        if CACHE_NAME.fullmatch(path.name) and (
            len(relative.parts) == 2 or (len(relative.parts) == 3 and relative.parts[1] == "cache")
        ):
            return
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"unresolvable source link: {path}") from error
        if path.is_symlink():
            manifest["links"].append(
                {
                    "path": relative.as_posix(),
                    "target": os.readlink(path),
                    "resolved_target": str(resolved),
                }
            )
        before = resolved.stat()
        inode = before.st_dev, before.st_ino
        output = destination / relative if destination is not None else None
        if stat.S_ISDIR(before.st_mode):
            if inode in ancestors:
                raise ValueError(f"directory link cycle: {path}")
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
            for child in sorted(path.iterdir()):
                visit(child, relative / child.name, ancestors | {inode})
        elif stat.S_ISREG(before.st_mode):
            digest = hashlib.sha256()
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
            with resolved.open("rb") as stream:
                out = output.open("xb") if output is not None else None
                try:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                        if out is not None:
                            out.write(chunk)
                finally:
                    if out is not None:
                        out.close()
            after = resolved.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError(f"source changed while copying: {path}")
            manifest["files"].append(
                {
                    "path": relative.as_posix(),
                    "resolved_source": str(resolved),
                    "source_bytes": before.st_size,
                    "source_sha256": digest.hexdigest(),
                    "output_sha256": digest.hexdigest(),
                }
            )
        else:
            raise ValueError(f"unsupported special source file: {path}")

    root_stat = source.stat()
    ancestors = frozenset({(root_stat.st_dev, root_stat.st_ino)})
    if source.is_symlink():
        manifest["links"].append(
            {"path": ".", "target": os.readlink(source), "resolved_target": str(source.resolve())}
        )
    for name in sorted(task_names):
        if not name.startswith("task_") or Path(name).name != name or "\\" in name:
            raise ValueError(f"invalid task directory name: {name!r}")
        task = source / name
        if task.exists() or task.is_symlink():
            if not task.is_dir():
                raise ValueError(f"selected task must be a directory: {task}")
            visit(task, Path(name), ancestors)
    marker = source / "FILTER_MANIFEST.txt"
    if marker.exists() or marker.is_symlink():
        if not marker.is_file():
            raise ValueError("FILTER_MANIFEST.txt must be a regular file")
        visit(marker, Path(marker.name), ancestors)
    return manifest


def inventory(source: Path, task_names: set[str]) -> dict:
    """Return source identities and expected byte-copy hashes, following visible links only."""
    return _scan(source, task_names)


def validate_tree(source: Path, destination: Path, manifest: dict) -> None:
    if inventory(source, set(manifest["task_names"])) != manifest:
        raise ValueError("source tree changed since copying")
    actual = _scan(destination, set(manifest["task_names"]))
    if actual["links"] or {f["path"]: f["source_sha256"] for f in actual["files"]} != {
        f["path"]: f["output_sha256"] for f in manifest["files"]
    }:
        raise ValueError("copied tree changed")


def copy_tree(source: Path, destination: Path, task_names: set[str]) -> dict:
    """Copy into a fresh directory; caller owns atomic publication and failure cleanup."""
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(
        destination.resolve()
    ):
        raise ValueError("source and destination must be disjoint")
    destination.mkdir(parents=True)
    manifest = _scan(source, task_names, destination)
    validate_tree(source, destination, manifest)
    return manifest
