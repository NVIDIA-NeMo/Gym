# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable CVDP repository snapshots and replayable workspace changes.

The command-line entry point runs inside the agent sandbox using only the standard library.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
from pathlib import Path
from typing import TypedDict


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ChangeSummary(TypedDict):
    """Changes collected relative to the immutable starting project."""

    deleted: list[str]
    changed_count: int


class RepositoryChanges(ChangeSummary):
    """Portable, saved agent changes used to reconstruct verification input."""

    archive: str


def project_path(name: str) -> bool:
    """Match upstream heavy collection: omit hidden paths, prompt.json, and rundir."""
    parts = Path(name).parts
    return (
        bool(parts)
        and not Path(name).is_absolute()
        and all(part not in {"..", "rundir", "prompt.json"} and not part.startswith(".") for part in parts)
    )


def resolve_archive(directory: str, digest: str) -> Path:
    """Resolve a prepared snapshot by content hash, never a request-provided host path."""
    if not directory or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Heavy CVDP tasks require prepared_repository_dir and a SHA-256 snapshot identifier")
    prepared = Path(directory).expanduser()
    if not prepared.is_absolute():
        # Gym starts component servers in their own directories. Config paths
        # describe repository-relative artifacts, not process-relative paths.
        prepared = _REPOSITORY_ROOT / prepared
    path = prepared / f"{digest}.tar.gz"
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"CVDP repository snapshot checksum mismatch: {path}")
    return path


def extract_snapshot(archive: Path, destination: Path) -> None:
    """Restore project assets while rejecting unsafe archives and reserved harness paths."""
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if any(part in {".cvdp_harness", ".nv", ".agent_deps", ".home"} for part in Path(member.name).parts):
                raise ValueError("Repository snapshot contains a reserved runtime path")
            source.extract(member, destination, filter="data")


def snapshot(root: Path) -> dict[str, list[str | int]]:
    """Record regular file contents, executable bits, and symlink targets without following links."""
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        dirs[:] = [d for d in dirs if project_path((parent / d).relative_to(root).as_posix())]
        for name in files + [d for d in dirs if (parent / d).is_symlink()]:
            path = parent / name
            rel = path.relative_to(root).as_posix()
            if not project_path(rel):
                continue
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                result[rel] = ["link", os.readlink(path), stat.S_IMODE(mode)]
            elif stat.S_ISREG(mode):
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                result[rel] = ["file", digest, stat.S_IMODE(mode)]
    return result


def collect_changes(root: Path, before: dict[str, list[str | int]], output: Path) -> ChangeSummary:
    """Archive changes across the project, retaining binary data and file modes."""
    after = snapshot(root)
    changed = sorted(name for name in after if before.get(name) != after[name])
    deleted = sorted(set(before) - set(after))
    with tarfile.open(output, "w:gz") as archive:
        for name in changed:
            archive.add(root / name, arcname=name, recursive=False)
    return {"deleted": deleted, "changed_count": len(changed)}


def apply_changes(root: Path, changes: RepositoryChanges) -> None:
    """Apply a saved agent delta without letting it modify the grading harness."""
    deleted = changes.get("deleted", [])
    if not isinstance(deleted, list) or any(not isinstance(p, str) or not project_path(p) for p in deleted):
        raise ValueError("Invalid repository deletion paths")
    for name in deleted:
        path = root / name
        if not path.parent.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Repository deletion escapes workspace: {name}")
        if path.is_symlink() or path.is_file():
            path.unlink()
    payload = base64.b64decode(changes.get("archive", ""), validate=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive:
            if not project_path(member.name) or not (member.isfile() or member.issym()):
                raise ValueError(f"Invalid repository change: {member.name}")
            target = root / member.name
            if not target.parent.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Repository change escapes workspace: {member.name}")
            if target.is_symlink():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            archive.extract(member, root, filter="data")


def main() -> None:
    """Capture the initial snapshot or write the agent's delta outside the project tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["restore", "snapshot", "collect"])
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.operation == "restore":
        if args.archive is None:
            parser.error("restore requires --archive")
        extract_snapshot(args.archive, args.root)
        args.manifest.write_text(json.dumps(snapshot(args.root)))
    elif args.operation == "snapshot":
        args.manifest.write_text(json.dumps(snapshot(args.root)))
    else:
        if args.archive is None:
            parser.error("collect requires --archive")
        result = collect_changes(args.root, json.loads(args.manifest.read_text()), args.archive)
        args.archive.with_suffix(".json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
