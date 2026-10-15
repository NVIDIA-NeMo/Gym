# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-folder digest of a Harbor task.

Same record format as Harbor's package publisher and Gym's Terminal Bench 4 loader:
SHA256 over sorted ``relative_path NUL file_sha256 LF`` records covering
``task.toml``, ``instruction.md``, ``README.md``, ``trajectory.json`` and everything
under ``environment/``, ``tests/``, ``solution/`` and ``steps/``. The digest travels in
``task_data`` under :data:`DIGEST_KEY` so the resources server can refuse a row whose
folder changed after materialization. :func:`content_hash` refuses directory symlinks because
this file-content format does not represent their targets.

:func:`context_digest` is the second digest here: Harbor's environment
id of a Dockerfile build context, which keys and names the images prepare builds. :func:`inputs_digest` is the
third: everything a build of that context reads, recorded with the built image so a context whose Harbor id
is unchanged but whose inputs are not (a mode, a symbolic link, an empty folder, a build setting) is rebuilt.
"""

import fnmatch
import hashlib
import os
import stat
from collections.abc import Mapping
from pathlib import Path

from nemo_gym.tasks.harbor.harbor_compat import environment_content_hash


DIGEST_KEY = "_ng_digest"

_SINGLE_FILES = ("task.toml", "instruction.md", "README.md", "trajectory.json")
_DIRECTORIES = ("environment", "tests", "solution", "steps")
_IGNORED = ("__pycache__", "*.pyc", ".DS_Store", "*.swp", "*.swo", "*~")


def is_ignored(relative: str) -> bool:
    """Whether :func:`content_hash` skips the task file at ``relative`` (a posix path under the task folder):
    editor leftovers and byte-code caches, matched on any path component."""
    parts = relative.split("/")
    return any(fnmatch.fnmatch(part, pattern) for part in parts for pattern in _IGNORED)


def _covered_files(path: Path) -> list[Path]:
    """The regular files under the resolved task folder ``path`` that a digest covers, in record order."""
    files = [path / name for name in _SINGLE_FILES if (path / name).is_file()]
    for name in _DIRECTORIES:
        files.extend(p for p in (path / name).rglob("*") if p.is_file())
    return sorted(files, key=lambda p: p.relative_to(path).as_posix())


def _record(path: Path, file: Path) -> bytes:
    """One ``relative_path NUL file_sha256 LF`` record of ``file`` under the resolved task folder ``path``."""
    relative = file.relative_to(path).as_posix()
    if not file.resolve().is_relative_to(path):
        raise ValueError(f"Task file escapes the task folder: {relative}")
    with file.open("rb") as stream:
        file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return f"{relative}\0{file_hash}\n".encode()


def archive_hash(path: Path) -> str:
    """Harbor's published content hash of the unpacked task package at ``path``.

    The same records as :func:`content_hash` over every covered file, with no exclusions: Harbor's
    packager applies the task's ``.gitignore`` (or its defaults, the same names as :data:`_IGNORED`)
    when it chooses the archive's files, and the published hash covers exactly those files. A
    downloaded archive therefore matches its advertised hash by this digest, whatever the publisher
    excluded or kept, while :func:`content_hash` stays the digest of a task folder on disk, where
    byte-code caches and editor leftovers appear after a run and must not change the task's identity.
    """
    path = Path(path).resolve()
    digest = hashlib.sha256()
    for file in _covered_files(path):
        digest.update(_record(path, file))
    return digest.hexdigest()


def content_hash(path: Path) -> str:
    """Digest the task folder at ``path``."""
    path = Path(path).resolve()
    # rglob does not descend into directory links, so check links before filtering to files
    # or ignoring cache paths. Reject unsupported links without changing the digest format.
    for link in path.rglob("*"):
        if not link.is_symlink():
            continue
        relative = link.relative_to(path).as_posix()
        target = link.resolve(strict=True)
        if not target.is_relative_to(path):
            raise ValueError(f"Task symlink escapes the task folder: {relative}")
        if target.is_dir():
            raise ValueError(
                f"Task directory symlinks are not supported: {relative}; replace it with a directory inside the task"
            )
    digest = hashlib.sha256()
    for file in _covered_files(path):
        relative = file.relative_to(path).as_posix()
        if is_ignored(relative):
            continue
        digest.update(_record(path, file))
    return digest.hexdigest()


def context_digest(path: Path) -> str:
    """Harbor's environment id of a Dockerfile build context (a task's ``environment/`` folder).

    :func:`~nemo_gym.tasks.harbor.harbor_compat.environment_content_hash` over the folder: 32 hex
    characters of SHA-256 over the sorted (relative path, bytes) of its regular files. Symbolic links and
    anything under a :data:`~nemo_gym.tasks.harbor.harbor_compat._CONTENT_HASH_IGNORE_NAMES` member
    (``__pycache__``, ``.git``, ``.DS_Store``) are skipped, and file modes are not part of it: parity with
    Harbor's image name is the contract, not a complete cache identity. An empty folder hashes to its
    name, as Harbor hashes it when the task declares no ``docker_image`` (one that does never reaches a
    build). Two contexts with the same files get the same id, so it keys the build record (see
    ``nemo_gym.tasks.harbor.image_build``); whether a recorded image is reused is :func:`inputs_digest`'s call.
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Not a build context folder: {path}")
    return environment_content_hash(path)


def inputs_digest(context: Path, settings: Mapping[str, str] | None = None) -> str:
    """Digest of everything an image build of the context folder at ``context`` reads.

    Where :func:`context_digest` is Harbor's name for the context, this is the build's identity: SHA-256 (64 hex
    characters) over every entry under ``context``, sorted by relative posix path, as
    ``relative_path NUL kind NUL st_mode NUL payload LF``. ``kind`` is ``f`` for a regular file (payload: the
    SHA-256 of its bytes), ``l`` for a symbolic link (payload: its target, read, not followed), ``d`` for a
    folder (no payload, so an empty folder counts) and ``o`` for anything else; ``st_mode`` is the full
    ``lstat`` mode in octal, permission bits included. Nothing is skipped: the names Harbor's hash ignores
    (``__pycache__``, ``.git``, ``.DS_Store``) and symbolic links are part of it, since a build tool reads them.
    ``settings`` are the build's inputs outside the tree (the Dockerfile path, the build arguments), appended
    as ``NUL key NUL value LF`` records sorted by key. Folder ownership is not part of it: a build does not
    read it, and it differs between hosts that hold the same folder.
    """
    context = Path(context)
    if not context.is_dir():
        raise ValueError(f"Not a build context folder: {context}")
    entries: list[tuple[str, Path]] = []
    for root, dirs, files in os.walk(context, followlinks=False):
        root_path = Path(root)
        for name in dirs + files:
            path = root_path / name
            entries.append((path.relative_to(context).as_posix(), path))
    digest = hashlib.sha256()
    for relative, path in sorted(entries, key=lambda item: item[0]):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            kind, payload = "l", os.readlink(path)
        elif stat.S_ISDIR(mode):
            kind, payload = "d", ""
        elif stat.S_ISREG(mode):
            with path.open("rb") as stream:
                kind, payload = "f", hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            kind, payload = "o", ""
        digest.update(f"{relative}\0{kind}\0{mode:o}\0{payload}\n".encode())
    for key, value in sorted((settings or {}).items()):
        digest.update(f"\0{key}\0{value}\n".encode())
    return digest.hexdigest()
