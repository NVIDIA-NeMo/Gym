# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned external DeepSWE checkout management."""

import asyncio
import fcntl
import hashlib
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _git_output(*args: str, cwd: Path | None = None) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return process.stdout.strip()


def _git_bytes(*args: str, cwd: Path) -> bytes:
    process = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        timeout=600,
    )
    return process.stdout


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def task_tree_digest(path: Path) -> str:
    """Return a stable SHA-256 over the pinned Git tree entries under ``tasks``."""
    path = path.expanduser().resolve()
    tree_entries = _git_bytes("ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", "tasks", cwd=path)
    return hashlib.sha256(tree_entries).hexdigest()


def materialize_task_snapshot(checkout: Path, destination_parent: Path) -> Path:
    """Materialize the committed task tree into a read-only per-process snapshot."""
    checkout = checkout.expanduser().resolve()
    destination_parent = destination_parent.expanduser().resolve()
    snapshot = Path(tempfile.mkdtemp(prefix="deep-swe-tasks-", dir=destination_parent))
    archive = snapshot / ".tasks.tar"
    try:
        process = subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", "HEAD", "tasks"],
            cwd=checkout,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if process.returncode != 0:
            raise RuntimeError(f"Could not materialize pinned DeepSWE tasks ({process.returncode})")
        with tarfile.open(archive, "r:") as task_archive:
            task_archive.extractall(snapshot, filter="data")
        archive.unlink()
        for path in snapshot.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"Pinned DeepSWE task snapshot contains a symlink: {path.relative_to(snapshot)}")
            if path.is_file():
                original_mode = stat.S_IMODE(path.stat().st_mode)
                os.chmod(path, 0o500 if original_mode & 0o100 else 0o400)
        for path in sorted((item for item in snapshot.rglob("*") if item.is_dir()), reverse=True):
            os.chmod(path, 0o500)
        os.chmod(snapshot, 0o500)
        return snapshot
    except BaseException:
        if archive.exists():
            archive.unlink()
        for root, directories, files in os.walk(snapshot):
            os.chmod(root, 0o700)
            for directory in directories:
                path = Path(root) / directory
                if not path.is_symlink():
                    os.chmod(path, 0o700)
            for filename in files:
                path = Path(root) / filename
                if not path.is_symlink():
                    os.chmod(path, 0o600)
        shutil.rmtree(snapshot)
        raise


def validate_checkout(path: Path, commit: str, expected_task_count: int | None) -> Path:
    """Validate that a checkout is exactly pinned and structurally complete."""
    path = path.expanduser().resolve()
    if not (path / ".git").exists():
        raise ValueError(f"DeepSWE checkout is not a git repository: {path}")
    actual_commit = _git_output("rev-parse", "HEAD", cwd=path)
    expected_commit = _git_output("rev-parse", f"{commit}^{{commit}}", cwd=path)
    if actual_commit != expected_commit:
        raise ValueError(f"DeepSWE checkout commit mismatch: expected {expected_commit}, got {actual_commit}")
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all", cwd=path)
    ignored_task_files = _git_output(
        "status",
        "--porcelain=v1",
        "--ignored",
        "--untracked-files=all",
        "--",
        "tasks",
        cwd=path,
    )
    if status or ignored_task_files:
        raise ValueError(f"DeepSWE checkout has modified or untracked content: {path}")
    tasks_dir = path / "tasks"
    if not tasks_dir.is_dir():
        raise ValueError(f"DeepSWE checkout has no tasks directory: {path}")
    task_dirs = sorted(candidate for candidate in tasks_dir.iterdir() if (candidate / "task.toml").is_file())
    if expected_task_count is not None and len(task_dirs) != expected_task_count:
        raise ValueError(
            f"DeepSWE task count mismatch at {actual_commit}: expected {expected_task_count}, found {len(task_dirs)}"
        )
    task_tree_digest(path)
    return path


def _clone_checkout(
    *,
    git_url: str,
    commit: str,
    destination: Path,
    expected_task_count: int | None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.lock"
    with _exclusive_lock(lock_path):
        if destination.exists():
            return validate_checkout(destination, commit, expected_task_count)

        with tempfile.TemporaryDirectory(prefix="deep-swe-clone-", dir=destination.parent) as tmp_dir:
            checkout = Path(tmp_dir) / "checkout"
            _git_output("clone", "--filter=blob:none", "--no-checkout", git_url, str(checkout))
            try:
                _git_output("fetch", "--depth", "1", "origin", commit, cwd=checkout)
                _git_output("checkout", "--detach", "FETCH_HEAD", cwd=checkout)
                validate_checkout(checkout, commit, expected_task_count)
                checkout.rename(destination)
            except BaseException:
                if checkout.exists():
                    shutil.rmtree(checkout)
                raise
        return validate_checkout(destination, commit, expected_task_count)


async def ensure_checkout(
    *,
    git_url: str,
    commit: str,
    cache_dir: Path,
    benchmark_path: Path | None,
    expected_task_count: int | None,
) -> Path:
    if benchmark_path is not None:
        return await asyncio.to_thread(
            validate_checkout,
            benchmark_path,
            commit,
            expected_task_count,
        )
    destination = cache_dir.expanduser().resolve() / commit
    return await asyncio.to_thread(
        _clone_checkout,
        git_url=git_url,
        commit=commit,
        destination=destination,
        expected_task_count=expected_task_count,
    )


def resolve_task(checkout: Path, task_id: str) -> Path:
    if TASK_ID_RE.fullmatch(task_id) is None:
        raise ValueError(f"Invalid DeepSWE task_id: {task_id!r}")
    tasks_dir = (checkout / "tasks").resolve()
    task_path = (tasks_dir / task_id).resolve()
    if not task_path.is_relative_to(tasks_dir) or not (task_path / "task.toml").is_file():
        raise ValueError(f"Unknown DeepSWE task_id: {task_id!r}")
    return task_path


__all__ = [
    "ensure_checkout",
    "materialize_task_snapshot",
    "resolve_task",
    "task_tree_digest",
    "validate_checkout",
]
