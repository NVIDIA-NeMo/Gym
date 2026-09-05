#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create and verify an immutable import of existing GDPVal deliverables.

The external tree is never modified.  Preparation hashes it, copies every byte
into a fresh checkpoint-e2e run, hashes the source again to close the scan/copy
race, and publishes an immutable provenance receipt.  Later verification
allows only new derived files (for example Office PDF sidecars); every imported
file and finish marker must remain byte-identical.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar


SCHEMA = "gdpval.existing-deliverables-import.v1"
INVENTORY_SCHEMA = "gdpval.existing-deliverables-inventory.v1"
PACKAGE_INVENTORY_SCHEMA = "gdpval.existing-judge-package-inventory.v1"
EXPECTED_TASKS = 220
FILE_MODE = 0o400
MARKER_NAME = "EXISTING_IMPORT_READY"
RECEIPT_NAME = "existing_import_receipt.json"
INVENTORY_NAME = "existing_import_inventory.json"
PACKAGE_INVENTORY_NAME = "existing_judge_package_inventory.json"
PACKAGE_DIR_NAME = "existing_judge_package"
DELIVERABLES_DIR_NAME = "deliverables"
ENVELOPE_SCHEMA = "gdpval.existing-judge-envelope.v1"
IO_RETRY_ATTEMPTS = 6
IO_RETRY_BASE_DELAY_SECONDS = 0.25
TRANSIENT_IO_ERRNOS = frozenset(
    getattr(errno, name)
    for name in (
        "EIO",
        "ESTALE",
        "ETIMEDOUT",
        "EAGAIN",
        "ENOTCONN",
        "ESHUTDOWN",
        "ENETDOWN",
        "ENETUNREACH",
        "ECONNRESET",
        "EHOSTDOWN",
        "EHOSTUNREACH",
    )
    if hasattr(errno, name)
)
T = TypeVar("T")


class ImportError(RuntimeError):
    """A fail-closed import or verification error."""


@dataclass(frozen=True)
class DatasetIdentity:
    path: Path
    sha256: str
    rows: tuple[dict[str, Any], ...]
    task_ids: tuple[str, ...]
    task_ids_sha256: str


def _fail(message: str) -> None:
    raise ImportError(message)


def _retry_io(operation: Callable[[], T], *, label: str) -> T:
    """Retry one complete read operation after known transient filesystem errors."""
    for attempt in range(1, IO_RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except OSError as exc:
            if exc.errno not in TRANSIENT_IO_ERRNOS or attempt == IO_RETRY_ATTEMPTS:
                raise
            delay = IO_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            error_name = errno.errorcode.get(exc.errno, str(exc.errno))
            print(
                f"EXISTING_IMPORT_IO_RETRY label={label} errno={error_name} "
                f"attempt={attempt}/{IO_RETRY_ATTEMPTS} delay={delay:g}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _read_bytes(path: Path) -> bytes:
    return _retry_io(path.read_bytes, label=f"read:{path}")


def _lstat(path: Path) -> os.stat_result:
    return _retry_io(path.lstat, label=f"lstat:{path}")


def _stat(path: Path) -> os.stat_result:
    return _retry_io(path.stat, label=f"stat:{path}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(document: Any) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _fsync_dir_once(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_dir(path: Path) -> None:
    _retry_io(lambda: _fsync_dir_once(path), label=f"fsync-directory:{path}")


def _mode(path: Path) -> int:
    return stat.S_IMODE(_lstat(path).st_mode)


def _require_regular(path: Path, *, mode: int | None = None) -> Path:
    try:
        metadata = _lstat(path)
    except OSError as exc:
        raise ImportError(f"required file is unavailable: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        _fail(f"path is not a regular non-symlink file: {path}")
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        _fail(f"file mode drift for {path}: {stat.S_IMODE(metadata.st_mode):04o} != {mode:04o}")
    return path


def _absolute_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        _fail(f"{label} must be an absolute path: {path}")
    try:
        metadata, resolved = _retry_io(
            lambda: (path.lstat(), path.resolve(strict=True)),
            label=f"resolve-directory:{path}",
        )
    except OSError as exc:
        raise ImportError(f"{label} is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        _fail(f"{label} must be a real directory: {path}")
    if resolved != path:
        _fail(f"{label} must already be resolved: {path} -> {resolved}")
    return resolved


def _absolute_regular(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        _fail(f"{label} must be an absolute path: {path}")
    try:
        metadata, resolved = _retry_io(
            lambda: (path.lstat(), path.resolve(strict=True)),
            label=f"resolve-file:{path}",
        )
    except OSError as exc:
        raise ImportError(f"{label} is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a regular non-symlink file: {path}")
    if resolved != path:
        _fail(f"{label} must already be resolved: {path} -> {resolved}")
    return resolved


def _hash_open_file_once(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"inventory entry is not a regular file: {path}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or total != before.st_size:
            _fail(f"file changed while it was being inventoried: {path}")
        return total, digest.hexdigest()
    finally:
        os.close(descriptor)


def _hash_open_file(path: Path) -> tuple[int, str]:
    return _retry_io(lambda: _hash_open_file_once(path), label=f"hash:{path}")


def _scan_tree(root: Path, *, ignored_directories: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    root = _absolute_directory(root, label="inventory root")
    records: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        before = _stat(directory)

        def list_directory() -> list[str]:
            with os.scandir(directory) as entries:
                return sorted(entry.name for entry in entries)

        names = _retry_io(list_directory, label=f"scandir:{directory}")
        for name in names:
            path = directory / name
            relative = path.relative_to(root).as_posix()
            metadata = _lstat(path)
            if stat.S_ISLNK(metadata.st_mode):
                _fail(f"symlinks are forbidden in imported trees: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                if name in ignored_directories:
                    continue
                records.append(
                    {"path": relative, "type": "directory", "mode": f"{stat.S_IMODE(metadata.st_mode):04o}"}
                )
                visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                size, digest = _hash_open_file(path)
                records.append(
                    {
                        "path": relative,
                        "type": "file",
                        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                        "bytes": size,
                        "sha256": digest,
                    }
                )
            else:
                _fail(f"special files are forbidden in imported trees: {path}")
        after = _stat(directory)
        if (before.st_dev, before.st_ino, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
        ):
            _fail(f"directory changed while it was being inventoried: {directory}")

    visit(root)
    return sorted(records, key=lambda row: (row["path"], row["type"]))


def _inventory_sha256(entries: Sequence[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_bytes(list(entries)))


def _content_identity(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in entries:
        if row["type"] == "directory":
            result.append({"path": row["path"], "type": "directory"})
        else:
            result.append(
                {
                    "path": row["path"],
                    "type": "file",
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
            )
    return sorted(result, key=lambda row: (row["path"], row["type"]))


def _read_dataset(path: Path, *, expected_tasks: int) -> DatasetIdentity:
    path = _absolute_regular(path, label="dataset")
    payload = _read_bytes(path)
    rows: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            _fail(f"dataset contains a blank line at {line_number}")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImportError(f"invalid dataset JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            _fail(f"dataset row {line_number} is not an object")
        task_id = row.get("task_id")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in {".", ".."}
            or "/" in task_id
            or "\\" in task_id
            or "\x00" in task_id
        ):
            _fail(f"dataset row {line_number} has an unsafe task_id")
        rows.append(row)
        task_ids.append(task_id)
    if len(rows) != expected_tasks:
        _fail(f"dataset has {len(rows)} rows, expected exactly {expected_tasks}")
    if len(set(task_ids)) != len(task_ids):
        _fail("dataset contains duplicate task IDs")
    task_ids_sha256 = _sha256_bytes(_canonical_bytes(sorted(task_ids)))
    return DatasetIdentity(
        path=path,
        sha256=_sha256_bytes(payload),
        rows=tuple(rows),
        task_ids=tuple(task_ids),
        task_ids_sha256=task_ids_sha256,
    )


def _validate_finish_markers(root: Path, dataset: DatasetIdentity) -> dict[str, Any]:
    root = _absolute_directory(root, label="deliverables snapshot")
    expected = set(dataset.task_ids)
    observed_task_dirs: set[str] = set()
    null_markers = 0
    object_markers = 0
    task_dirs = _retry_io(lambda: sorted(root.glob("task_*")), label=f"glob-tasks:{root}")
    for task_dir in task_dirs:
        metadata = _lstat(task_dir)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _fail(f"task path is not a real directory: {task_dir}")
        observed_task_dirs.add(task_dir.name.removeprefix("task_"))
    missing_dirs = sorted(expected - observed_task_dirs)
    extra_dirs = sorted(observed_task_dirs - expected)
    if missing_dirs or extra_dirs:
        _fail(f"task directory coverage mismatch: missing={missing_dirs} extra={extra_dirs}")

    for task_id in dataset.task_ids:
        marker = root / f"task_{task_id}" / "repeat_0" / "finish_params.json"
        _require_regular(marker)
        try:
            value = json.loads(_read_bytes(marker))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImportError(f"invalid finish marker {marker}: {exc}") from exc
        if value is None:
            null_markers += 1
        elif isinstance(value, dict):
            object_markers += 1
        else:
            _fail(f"finish marker is neither a JSON object nor null: {marker}")
    return {
        "completed": len(dataset.task_ids),
        "null_markers": null_markers,
        "object_markers": object_markers,
        "task_ids_sha256": dataset.task_ids_sha256,
    }


def _copy_file_once(source_path: Path, target: Path, row: dict[str, Any]) -> None:
    source_fd = os.open(source_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        target_fd = os.open(target, os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0))
        try:
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    if written <= 0:
                        _fail(f"short write while copying {source_path}")
                    view = view[written:]
                total += len(chunk)
                digest.update(chunk)
            if total != row["bytes"] or digest.hexdigest() != row["sha256"]:
                _fail(f"source changed while it was being copied: {source_path}")
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
    finally:
        os.close(source_fd)


def _copy_tree(source: Path, destination: Path, entries: Sequence[dict[str, Any]], *, package: bool) -> None:
    expected_identity = _content_identity(entries)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            _fail(f"snapshot destination is not a real directory: {destination}")
        current = _content_identity(_scan_tree(destination))
        if not current and not any(destination.iterdir()):
            destination.rmdir()
        elif current == expected_identity:
            return
        else:
            _fail(f"pre-existing snapshot differs from the immutable source: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.import.", dir=destination.parent))
    try:
        for row in entries:
            target = staging / row["path"]
            if row["type"] == "directory":
                target.mkdir(parents=True, exist_ok=False, mode=0o700)
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            source_path = source / row["path"]
            descriptor = _retry_io(
                lambda: os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                ),
                label=f"create-copy-target:{target}",
            )
            os.close(descriptor)
            executable = int(row["mode"], 8) & 0o111
            final_mode = 0o500 if package and executable else 0o400
            _retry_io(
                lambda: _copy_file_once(source_path, target, row),
                label=f"copy:{source_path}",
            )
            _retry_io(lambda: os.chmod(target, final_mode), label=f"chmod-copy-target:{target}")
        for path in sorted((item for item in staging.rglob("*") if item.is_dir()), reverse=True):
            path.chmod(0o500 if package else 0o700)
        staging.chmod(0o500 if package else 0o700)
        _fsync_dir(staging)
        os.replace(staging, destination)
        _fsync_dir(destination.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _publish_immutable(path: Path, payload: bytes, *, executable: bool = False) -> None:
    mode = 0o500 if executable else FILE_MODE
    if path.exists() or path.is_symlink():
        _require_regular(path, mode=mode)
        if _read_bytes(path) != payload:
            _fail(f"immutable receipt drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail(f"short write publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    _fsync_dir(path.parent)


def _identity_document(source: Path, dataset: DatasetIdentity, source_entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    source_sha = _inventory_sha256(source_entries)
    identity = {
        "schema": SCHEMA,
        "source": str(source),
        "source_inventory_sha256": source_sha,
        "dataset_sha256": dataset.sha256,
        "dataset_task_ids_sha256": dataset.task_ids_sha256,
        "expected_tasks": len(dataset.task_ids),
    }
    import_id = "import-" + _sha256_bytes(_canonical_bytes(identity))[:24]
    return {**identity, "import_id": import_id}


def identify(source: Path, dataset_path: Path, *, expected_tasks: int) -> dict[str, Any]:
    source = _absolute_directory(source, label="external deliverables source")
    dataset = _read_dataset(dataset_path, expected_tasks=expected_tasks)
    entries = _scan_tree(source)
    coverage = _validate_finish_markers(source, dataset)
    result = _identity_document(source, dataset, entries)
    result.update(
        {
            "source_files": sum(row["type"] == "file" for row in entries),
            "source_bytes": sum(row.get("bytes", 0) for row in entries),
            "coverage": coverage,
        }
    )
    return result


def identify_package(package: Path) -> dict[str, Any]:
    package = _absolute_directory(package, label="checkpoint_e2e package")
    entries = _scan_tree(package, ignored_directories=frozenset({"__pycache__"}))
    return {
        "schema": PACKAGE_INVENTORY_SCHEMA,
        "root": str(package),
        "inventory_sha256": _inventory_sha256(entries),
        "files": sum(row["type"] == "file" for row in entries),
        "bytes": sum(row.get("bytes", 0) for row in entries),
    }


def prepare(
    run_dir: Path,
    source: Path,
    dataset_path: Path,
    package: Path,
    *,
    expected_tasks: int,
    expected_import_id: str | None,
) -> dict[str, Any]:
    run_dir = _absolute_directory(run_dir, label="prepared campaign run")
    _require_regular(run_dir / "settings.env", mode=FILE_MODE)
    source = _absolute_directory(source, label="external deliverables source")
    package = _absolute_directory(package, label="checkpoint_e2e package")
    if run_dir == source or run_dir in source.parents or source in run_dir.parents:
        _fail("external source and fresh run must be disjoint")
    if run_dir == package or run_dir in package.parents:
        _fail("active package must not be nested below the fresh run")

    dataset = _read_dataset(dataset_path, expected_tasks=expected_tasks)
    source_before = _scan_tree(source)
    coverage = _validate_finish_markers(source, dataset)
    identity = _identity_document(source, dataset, source_before)
    if expected_import_id is not None and identity["import_id"] != expected_import_id:
        _fail(f"source identity changed before import: {identity['import_id']} != {expected_import_id}")
    package_before = _scan_tree(package, ignored_directories=frozenset({"__pycache__"}))

    snapshot = run_dir / DELIVERABLES_DIR_NAME
    package_snapshot = run_dir / PACKAGE_DIR_NAME
    _copy_tree(source, snapshot, source_before, package=False)
    _copy_tree(package, package_snapshot, package_before, package=True)

    source_after = _scan_tree(source)
    package_after = _scan_tree(package, ignored_directories=frozenset({"__pycache__"}))
    dataset_after = _read_dataset(dataset.path, expected_tasks=expected_tasks)
    if source_after != source_before:
        _fail("external deliverables source changed during snapshot publication")
    if package_after != package_before:
        _fail("checkpoint_e2e package changed during snapshot publication")
    if dataset_after.sha256 != dataset.sha256:
        _fail("dataset changed during snapshot publication")
    snapshot_entries = _scan_tree(snapshot)
    package_snapshot_entries = _scan_tree(package_snapshot)
    if _content_identity(snapshot_entries) != _content_identity(source_before):
        _fail("published deliverables snapshot differs from its source")
    if _content_identity(package_snapshot_entries) != _content_identity(package_before):
        _fail("published judge package differs from its source")
    _validate_finish_markers(snapshot, dataset)

    source_inventory = {
        "schema": INVENTORY_SCHEMA,
        "root": str(source),
        "inventory_sha256": _inventory_sha256(source_before),
        "entries": source_before,
    }
    package_inventory = {
        "schema": PACKAGE_INVENTORY_SCHEMA,
        "root": str(package_snapshot),
        "source_root": str(package),
        "inventory_sha256": _inventory_sha256(package_snapshot_entries),
        "entries": package_snapshot_entries,
    }
    source_inventory_payload = _canonical_bytes(source_inventory)
    package_inventory_payload = _canonical_bytes(package_inventory)
    receipt = {
        "schema": SCHEMA,
        "import_id": identity["import_id"],
        "run_dir": str(run_dir),
        "source": str(source),
        "source_inventory": str(run_dir / INVENTORY_NAME),
        "source_inventory_file_sha256": _sha256_bytes(source_inventory_payload),
        "source_inventory_sha256": source_inventory["inventory_sha256"],
        "source_files": sum(row["type"] == "file" for row in source_before),
        "source_bytes": sum(row.get("bytes", 0) for row in source_before),
        "snapshot": str(snapshot),
        "snapshot_inventory_sha256": _inventory_sha256(snapshot_entries),
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "rows": len(dataset.rows),
            "expected_tasks": expected_tasks,
            "task_ids_sha256": dataset.task_ids_sha256,
        },
        "coverage": coverage,
        "judge_package": str(package_snapshot),
        "judge_package_inventory": str(run_dir / PACKAGE_INVENTORY_NAME),
        "judge_package_inventory_file_sha256": _sha256_bytes(package_inventory_payload),
        "judge_package_inventory_sha256": package_inventory["inventory_sha256"],
        "judge_package_source_inventory_sha256": _inventory_sha256(package_before),
    }
    receipt_payload = _canonical_bytes(receipt)
    receipt_path = run_dir / RECEIPT_NAME
    _publish_immutable(run_dir / INVENTORY_NAME, source_inventory_payload)
    _publish_immutable(run_dir / PACKAGE_INVENTORY_NAME, package_inventory_payload)
    _publish_immutable(receipt_path, receipt_payload)
    sidecar_payload = f"{_sha256_bytes(receipt_payload)}  {receipt_path}\n".encode()
    _publish_immutable(receipt_path.with_suffix(receipt_path.suffix + ".sha256"), sidecar_payload)
    _publish_immutable(run_dir / MARKER_NAME, b"")
    return verify(run_dir, strict_snapshot=True)


def _load_json(path: Path, *, schema: str) -> dict[str, Any]:
    _require_regular(path, mode=FILE_MODE)
    try:
        document = json.loads(_read_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImportError(f"invalid receipt JSON {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != schema:
        _fail(f"receipt schema mismatch: {path}")
    return document


def _validate_original_entries(root: Path, entries: Sequence[dict[str, Any]]) -> None:
    for row in entries:
        path = root / row["path"]
        try:
            metadata = _lstat(path)
        except OSError as exc:
            raise ImportError(f"imported path is unavailable: {path}: {exc}") from exc
        if row["type"] == "directory":
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                _fail(f"imported directory drift: {path}")
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail(f"imported file type drift: {path}")
        size, digest = _hash_open_file(path)
        if size != row["bytes"] or digest != row["sha256"]:
            _fail(f"imported file content drift: {path}")


def verify(run_dir: Path, *, strict_snapshot: bool = False) -> dict[str, Any]:
    run_dir = _absolute_directory(run_dir, label="prepared campaign run")
    _require_regular(run_dir / "settings.env", mode=FILE_MODE)
    _require_regular(run_dir / MARKER_NAME, mode=FILE_MODE)
    if _stat(run_dir / MARKER_NAME).st_size != 0:
        _fail("existing-import completion marker is not empty")
    receipt_path = run_dir / RECEIPT_NAME
    receipt = _load_json(receipt_path, schema=SCHEMA)
    sidecar = _require_regular(receipt_path.with_suffix(receipt_path.suffix + ".sha256"), mode=FILE_MODE)
    expected_sidecar = f"{_sha256_bytes(_read_bytes(receipt_path))}  {receipt_path}\n".encode()
    if _read_bytes(sidecar) != expected_sidecar:
        _fail("existing-import receipt digest sidecar mismatch")
    if receipt.get("run_dir") != str(run_dir):
        _fail("existing-import receipt points at another run")

    source_inventory_path = run_dir / INVENTORY_NAME
    package_inventory_path = run_dir / PACKAGE_INVENTORY_NAME
    source_inventory = _load_json(source_inventory_path, schema=INVENTORY_SCHEMA)
    package_inventory = _load_json(package_inventory_path, schema=PACKAGE_INVENTORY_SCHEMA)
    if _sha256_bytes(_read_bytes(source_inventory_path)) != receipt.get("source_inventory_file_sha256"):
        _fail("source inventory receipt hash mismatch")
    if _sha256_bytes(_read_bytes(package_inventory_path)) != receipt.get(
        "judge_package_inventory_file_sha256"
    ):
        _fail("judge package inventory receipt hash mismatch")
    if _inventory_sha256(source_inventory.get("entries", [])) != receipt.get("source_inventory_sha256"):
        _fail("source inventory identity mismatch")
    if _inventory_sha256(package_inventory.get("entries", [])) != receipt.get(
        "judge_package_inventory_sha256"
    ):
        _fail("judge package inventory identity mismatch")

    dataset_record = receipt.get("dataset")
    if not isinstance(dataset_record, dict):
        _fail("dataset receipt is missing")
    recorded_expected_tasks = dataset_record.get("expected_tasks")
    if not isinstance(recorded_expected_tasks, int) or recorded_expected_tasks <= 0:
        _fail("dataset expected-task contract is invalid")
    dataset = _read_dataset(Path(dataset_record.get("path", "")), expected_tasks=recorded_expected_tasks)
    if (
        dataset.sha256 != dataset_record.get("sha256")
        or dataset.task_ids_sha256 != dataset_record.get("task_ids_sha256")
        or len(dataset.rows) != recorded_expected_tasks
    ):
        _fail("dataset identity drift after import")

    snapshot = _absolute_directory(Path(receipt.get("snapshot", "")), label="deliverables snapshot")
    package = _absolute_directory(Path(receipt.get("judge_package", "")), label="judge package snapshot")
    source_entries = source_inventory.get("entries")
    package_entries = package_inventory.get("entries")
    if not isinstance(source_entries, list) or not isinstance(package_entries, list):
        _fail("inventory entries are malformed")
    _validate_original_entries(snapshot, source_entries)
    _validate_finish_markers(snapshot, dataset)
    if strict_snapshot:
        current = _scan_tree(snapshot)
        if _content_identity(current) != _content_identity(source_entries):
            _fail("strict snapshot verification found derived or missing paths")
    current_package = _scan_tree(package)
    if current_package != package_entries:
        _fail("run-owned judge package drift")
    return {
        "status": "PASS",
        "import_id": receipt["import_id"],
        "run_dir": str(run_dir),
        "dataset_sha256": dataset.sha256,
        "completed": len(dataset.task_ids),
        "source_inventory_sha256": receipt["source_inventory_sha256"],
        "judge_package_inventory_sha256": receipt["judge_package_inventory_sha256"],
        "strict_snapshot": strict_snapshot,
    }


def prepare_input(run_dir: Path, output: Path) -> dict[str, Any]:
    run_dir = _absolute_directory(run_dir, label="prepared campaign run")
    verify(run_dir)
    receipt = _load_json(run_dir / RECEIPT_NAME, schema=SCHEMA)
    dataset_record = receipt["dataset"]
    dataset = _read_dataset(Path(dataset_record["path"]), expected_tasks=int(dataset_record["expected_tasks"]))
    expected_agent = {"type": "responses_api_agents", "name": "gdpval_stirrup_agent"}
    prepared: list[bytes] = []
    for index, source_row in enumerate(dataset.rows):
        row = dict(source_row)
        recorded_agent = row.get("agent_ref")
        if recorded_agent is not None and recorded_agent != expected_agent:
            _fail(f"dataset row {index + 1} has an incompatible agent_ref")
        row["agent_ref"] = expected_agent
        prepared.append(_canonical_bytes(row))
    payload = b"".join(prepared)
    output = output.expanduser().absolute()
    if output == run_dir or run_dir not in output.parents:
        _fail(f"preprocessed output must be below the fresh run: {output}")
    if output.parent.is_symlink():
        _fail(f"preprocessed output parent is a symlink: {output.parent}")
    _publish_immutable(output, payload)
    return {
        "status": "PASS",
        "path": str(output),
        "sha256": _sha256_bytes(payload),
        "rows": len(prepared),
    }


def _envelope_paths(run_dir: Path, suffix: str) -> dict[str, Path]:
    safe = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    if not suffix or len(suffix) > 32 or any(character not in safe for character in suffix):
        _fail(f"unsafe judge suffix: {suffix}")
    return {
        "campaign_manifest": run_dir / "campaign.json",
        "checkpoint_fingerprint": run_dir / "checkpoint_fingerprint.json",
        "campaign_settings": run_dir / "settings.env",
        "import_receipt": run_dir / RECEIPT_NAME,
        "runtime_manifest": run_dir / "judge_runtime_overlay_existing" / "runtime_manifest.json",
        "transport_manifest": run_dir / "judge_transport_views_existing" / "manifest.json",
        "fingerprint_receipt": run_dir / f"fingerprint_{suffix}.json",
        "strict_result": run_dir / f"final_receipt_{suffix}.json",
        "envelope": run_dir / f"final_envelope_{suffix}.json",
    }


def _binding(path: Path) -> dict[str, Any]:
    _require_regular(path)
    payload = _read_bytes(path)
    return {"path": str(path), "bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _expected_envelope(run_dir: Path, suffix: str) -> dict[str, Any]:
    paths = _envelope_paths(run_dir, suffix)
    fingerprint_path = paths["fingerprint_receipt"]
    try:
        fingerprint_document = json.loads(_read_bytes(_require_regular(fingerprint_path)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImportError(f"invalid fingerprint receipt {fingerprint_path}: {exc}") from exc
    fingerprint = fingerprint_document.get("fingerprint") if isinstance(fingerprint_document, dict) else None
    if (
        not isinstance(fingerprint_document, dict)
        or fingerprint_document.get("schema") != "gdpval.multistage-fingerprint-probe.v1"
        or fingerprint_document.get("status") != "PASS"
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        _fail("fingerprint receipt contract mismatch")
    return {
        "schema": ENVELOPE_SCHEMA,
        "run_dir": str(run_dir),
        "judge_suffix": suffix,
        "campaign_manifest": _binding(paths["campaign_manifest"]),
        "checkpoint_fingerprint": _binding(paths["checkpoint_fingerprint"]),
        "campaign_settings": _binding(paths["campaign_settings"]),
        "import_receipt": _binding(paths["import_receipt"]),
        "runtime_manifest": _binding(paths["runtime_manifest"]),
        "transport_manifest": _binding(paths["transport_manifest"]),
        "fingerprint_receipt": {**_binding(fingerprint_path), "fingerprint": fingerprint},
        "strict_result": _binding(paths["strict_result"]),
    }


def publish_envelope(run_dir: Path, suffix: str) -> dict[str, Any]:
    run_dir = _absolute_directory(run_dir, label="prepared campaign run")
    verify(run_dir)
    paths = _envelope_paths(run_dir, suffix)
    payload = _canonical_bytes(_expected_envelope(run_dir, suffix))
    _publish_immutable(paths["envelope"], payload)
    sidecar = paths["envelope"].with_suffix(paths["envelope"].suffix + ".sha256")
    _publish_immutable(sidecar, f"{_sha256_bytes(payload)}  {paths['envelope']}\n".encode())
    return verify_envelope(run_dir, suffix)


def verify_envelope(run_dir: Path, suffix: str) -> dict[str, Any]:
    run_dir = _absolute_directory(run_dir, label="prepared campaign run")
    verify(run_dir)
    paths = _envelope_paths(run_dir, suffix)
    envelope = _require_regular(paths["envelope"], mode=FILE_MODE)
    expected = _canonical_bytes(_expected_envelope(run_dir, suffix))
    if _read_bytes(envelope) != expected:
        _fail("final import envelope drift")
    sidecar = _require_regular(envelope.with_suffix(envelope.suffix + ".sha256"), mode=FILE_MODE)
    expected_sha = f"{_sha256_bytes(expected)}  {envelope}\n".encode()
    if _read_bytes(sidecar) != expected_sha:
        _fail("final import envelope digest sidecar mismatch")
    return {
        "status": "PASS",
        "run_dir": str(run_dir),
        "judge_suffix": suffix,
        "envelope": str(envelope),
        "envelope_sha256": _sha256_bytes(expected),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    identify_parser = commands.add_parser("identify", help="hash and validate an external deliverables tree")
    identify_parser.add_argument("--source", type=Path, required=True)
    identify_parser.add_argument("--dataset", type=Path, required=True)
    identify_parser.add_argument("--expected-tasks", type=int, default=EXPECTED_TASKS)

    package_parser = commands.add_parser("identify-package", help="hash the checkpoint_e2e package")
    package_parser.add_argument("--package", type=Path, required=True)

    prepare_parser = commands.add_parser("prepare", help="publish a fresh immutable import snapshot")
    prepare_parser.add_argument("--run-dir", type=Path, required=True)
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument("--package", type=Path, required=True)
    prepare_parser.add_argument("--expected-tasks", type=int, default=EXPECTED_TASKS)
    prepare_parser.add_argument("--expected-import-id")

    verify_parser = commands.add_parser("verify", help="verify the frozen import and run-owned package")
    verify_parser.add_argument("--run-dir", type=Path, required=True)
    verify_parser.add_argument("--strict-snapshot", action="store_true")

    input_parser = commands.add_parser("prepare-input", help="freeze provider-free benchmark input rows")
    input_parser.add_argument("--run-dir", type=Path, required=True)
    input_parser.add_argument("--output", type=Path, required=True)

    envelope_parser = commands.add_parser("publish-envelope", help="bind all final import-only receipts")
    envelope_parser.add_argument("--run-dir", type=Path, required=True)
    envelope_parser.add_argument("--suffix", required=True)

    verify_envelope_parser = commands.add_parser("verify-envelope", help="verify the final import envelope")
    verify_envelope_parser.add_argument("--run-dir", type=Path, required=True)
    verify_envelope_parser.add_argument("--suffix", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "identify":
            result = identify(args.source, args.dataset, expected_tasks=args.expected_tasks)
        elif args.command == "identify-package":
            result = identify_package(args.package)
        elif args.command == "prepare":
            result = prepare(
                args.run_dir,
                args.source,
                args.dataset,
                args.package,
                expected_tasks=args.expected_tasks,
                expected_import_id=args.expected_import_id,
            )
        elif args.command == "verify":
            result = verify(args.run_dir, strict_snapshot=args.strict_snapshot)
        elif args.command == "prepare-input":
            result = prepare_input(args.run_dir, args.output)
        elif args.command == "publish-envelope":
            result = publish_envelope(args.run_dir, args.suffix)
        elif args.command == "verify-envelope":
            result = verify_envelope(args.run_dir, args.suffix)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (ImportError, OSError, ValueError) as exc:
        print(f"EXISTING_IMPORT_FAIL: {exc}", file=sys.stderr)
        return 64
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
