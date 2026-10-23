# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archive and restore complete task roots, including files omitted by Git diffs."""

import hashlib
import json
import os
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath


FORBIDDEN_ROOTS = {"/", "/tmp", "/opt", "/home", "/root", "/usr", "/etc", "/var", "/proc", "/sys", "/dev"}


def task_roots(paths: list[str]) -> list[str]:
    """Choose only initial task roots, never broad system or harness directories."""
    roots = []
    for value in sorted(set(paths), key=lambda value: (len(PurePosixPath(value).parts), value)):
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or str(path) in FORBIDDEN_ROOTS:
            raise ValueError(f"Unsafe task snapshot root: {value}")
        if str(path).startswith(("/opt/gym-runtime", "/opt/gym-submission", "/tmp/nemo-gym-")):
            raise ValueError(f"Harness runtime cannot be a task snapshot root: {value}")
        if not any(path == PurePosixPath(root) or PurePosixPath(root) in path.parents for root in roots):
            roots.append(str(path))
    return roots


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def freeze(roots: list[str], directory: Path) -> dict:
    """Record every regular file, directory, hard link and symlink under the task roots."""
    roots = task_roots(roots)
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "state.tar.gz"
    with tarfile.open(archive_path, "w:gz", dereference=False) as archive:
        for root in roots:
            path = Path(root)
            if path.is_symlink():
                raise ValueError(f"Task root became a symlink: {root}")
            if path.exists():
                archive.add(path, arcname=root.lstrip("/"), recursive=True)
    manifest = {"roots": roots, "archive_sha256": digest(archive_path), "bytes": archive_path.stat().st_size}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def protected_directories(path: Path) -> list[Path]:
    """Find foreign directories that this image user cannot clear."""
    attributes = path.lstat()
    if not stat.S_ISDIR(attributes.st_mode):
        return []
    if attributes.st_uid != os.geteuid() and not os.access(path, os.W_OK | os.X_OK, effective_ids=True):
        return [path]
    return [protected for child in path.iterdir() for protected in protected_directories(child)]


def verify_protected_subtrees(archive: tarfile.TarFile, roots: list[str]) -> dict[Path, int]:
    """Retain protected baseline trees only when their complete archived state matches."""
    protected = [path for root in roots if Path(root).exists() for path in protected_directories(Path(root))]
    members = {Path("/") / item.name: item for item in archive.getmembers()}
    if len(members) != len(archive.getmembers()):
        raise ValueError("Submission contains duplicate member paths")
    for path, item in members.items():
        if item.islnk() and any(
            path.is_relative_to(root) or (Path("/") / item.linkname).is_relative_to(root) for root in protected
        ):
            raise ValueError(f"Protected baseline hardlink cannot be preserved: {path}")

    def inspect_tree(path: Path) -> dict[Path, os.stat_result]:
        attributes = path.lstat()
        entries = {path: attributes}
        if stat.S_ISDIR(attributes.st_mode):
            for child in path.iterdir():
                entries.update(inspect_tree(child))
        return entries

    verified = {}
    for root in protected:
        expected = {path: item for path, item in members.items() if path.is_relative_to(root)}
        actual = inspect_tree(root)
        if actual.keys() != expected.keys():
            mismatch = min(actual.keys() ^ expected.keys())
            raise ValueError(f"Protected baseline differs at {mismatch}: missing or extra member")
        for path, attributes in actual.items():
            item = expected[path]
            properties = {
                "type": (item.isdir() and stat.S_ISDIR(attributes.st_mode))
                or (item.isfile() and stat.S_ISREG(attributes.st_mode))
                or (item.issym() and stat.S_ISLNK(attributes.st_mode)),
                "mode": stat.S_IMODE(attributes.st_mode) == item.mode,
                "uid": attributes.st_uid == item.uid,
                "gid": attributes.st_gid == item.gid,
                # TarInfo uses seconds at the same float precision as stat.st_mtime.
                "mtime": attributes.st_mtime == item.mtime,
            }
            for name, matches in properties.items():
                if not matches:
                    raise ValueError(f"Protected baseline differs at {path}: {name}")
            if item.issym() and os.readlink(path) != item.linkname:
                raise ValueError(f"Protected baseline differs at {path}: symlink target")
            if item.isfile():
                if attributes.st_nlink != 1:
                    raise ValueError(f"Protected baseline hardlink cannot be preserved: {path}")
                with archive.extractfile(item) as source:
                    same_bytes = (
                        attributes.st_size == item.size
                        and digest(path) == hashlib.file_digest(source, "sha256").hexdigest()
                    )
                if not same_bytes:
                    raise ValueError(f"Protected baseline differs at {path}: file bytes")
        verified[root] = len(expected)
    return verified


def clear_directory(path: Path, preserved: set[Path]) -> bool:
    """Remove baseline contents, allowing changes to directories owned by this user."""
    if path in preserved:
        return True
    attributes = path.stat()
    if attributes.st_uid == os.geteuid():
        path.chmod(stat.S_IMODE(attributes.st_mode) | stat.S_IRWXU)
    retained = False
    for child in path.iterdir():
        if child.is_symlink() or not child.is_dir():
            child.unlink()
        else:
            if clear_directory(child, preserved):
                retained = True
            else:
                child.rmdir()
    return retained


def restore(directory: Path) -> dict:
    """Replace task contents in a fresh image, retaining existing root directories."""
    manifest = json.loads((directory / "manifest.json").read_text())
    roots = task_roots(manifest["roots"])
    archive_path = directory / "state.tar.gz"
    if digest(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Submission archive checksum mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for item in members:
            name = PurePosixPath(item.name)
            target = PurePosixPath("/") / name
            if (
                name.is_absolute()
                or ".." in name.parts
                or not any(target == PurePosixPath(root) or PurePosixPath(root) in target.parents for root in roots)
            ):
                raise ValueError("Submission member is outside the task roots")
            if not (item.isfile() or item.isdir() or item.issym() or item.islnk()):
                raise ValueError("Submission contains an unsupported filesystem entry")
        verified = verify_protected_subtrees(archive, roots)
        preserved = set(verified)
        archived_directories = {"/" + item.name.rstrip("/") for item in members if item.isdir()}
        for root in roots:
            path = Path(root)
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                # A task user can own /workspace without being allowed to unlink
                # it from /. Keep that directory and restore its archived attrs.
                clear_directory(path, preserved)
                if root not in archived_directories:
                    path.rmdir()

        # Absolute task symlinks are preserved, but never followed while extracting
        # another member. This allows normal venv links without writing through them.
        def checked_member(item: tarfile.TarInfo, destination: str) -> tarfile.TarInfo | None:
            target = Path(destination) / item.name
            if any(parent.is_symlink() for parent in target.parents):
                raise ValueError("Submission member would write through a symlink")
            if any(target.is_relative_to(root) for root in preserved):
                return None
            if item.islnk():
                link = PurePosixPath(item.linkname)
                if (
                    link.is_absolute()
                    or ".." in link.parts
                    or not any((PurePosixPath("/") / link).is_relative_to(root) for root in roots)
                ):
                    raise ValueError("Submission hardlink is outside task roots")
            return item

        # extractall delays directory ownership, modes and mtimes until children
        # exist, then restores deepest directories first (including read-only ones).
        archive.extractall(path="/", members=members, filter=checked_member)
    return {
        "archive_sha256": manifest["archive_sha256"],
        "restored_roots": roots,
        "members": len(members),
        "preserved_baseline": [{"root": str(root), "members": count} for root, count in verified.items()],
    }


if __name__ == "__main__":
    action, directory, *args = sys.argv[1:]
    try:
        result = freeze(json.loads(args[0]), Path(directory)) if action == "freeze" else restore(Path(directory))
    except PermissionError as exc:
        raise RuntimeError(
            f"Submission {action} denied for uid={os.geteuid()} gid={os.getegid()} at {exc.filename!r}: "
            f"{exc.strerror}. Task roots must be readable for freeze and writable for restore by the image user."
        ) from exc
    print(json.dumps(result))
