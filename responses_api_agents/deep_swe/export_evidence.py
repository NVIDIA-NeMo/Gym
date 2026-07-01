# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export completed DeepSWE evidence from the private runtime root.

The agent deliberately keeps live Pier jobs below a UID-private temporary
root.  This module copies only finalized jobs named by completed rollout
rows into a separate, owner-only evidence root after collection finishes.
Rollout rows are never rewritten; the receipt preserves the original paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from responses_api_agents.deep_swe.secure_paths import default_private_path, ensure_private_directory
from responses_api_agents.deep_swe.trajectory import (
    ArtifactLimitError,
    ArtifactLimits,
    artifact_snapshot,
    read_bytes,
)


DEFAULT_SOURCE_ROOT = default_private_path("jobs")
DEFAULT_LIMITS = ArtifactLimits(
    max_files=2048,
    max_file_bytes=64 * 1024 * 1024,
    max_total_bytes=256 * 1024 * 1024,
)
RECEIPT_NAME = "export-receipt.json"
RECEIPT_SCHEMA_VERSION = 1
# A row may contain the configured 64 MiB trajectory twice (raw ATIF plus the
# converted response), a 16 MiB patch, and substantial Pier/JSON metadata.
DEFAULT_ROLLOUT_MAX_LINE_BYTES = 192 * 1024 * 1024
MAX_ROLLOUT_ROWS = 339
MAX_ARTIFACT_ENTRIES_PER_ROW = DEFAULT_LIMITS.max_files
MAX_TASK_STATUS_UTF8_BYTES = 1024
MAX_PATH_URI_UTF8_BYTES = 16 * 1024
MAX_SANDBOX_RUNTIME_CANONICAL_BYTES = 256 * 1024
MAX_ARTIFACT_PATHS_UTF8_BYTES = 256 * 1024
RUNTIME_PROVENANCE_RELATIVE = Path("gym-runtime-provenance.json")
TERMINAL_STATUSES = frozenset({"success", "error", "harness_error"})
GYM_REPOSITORY_URL = "https://github.com/NVIDIA-NeMo/Gym"


class EvidenceExportError(RuntimeError):
    """Raised when an evidence export cannot be proven safe and complete."""


@dataclass(frozen=True, slots=True)
class ExportArtifact:
    path: str
    size: int
    sha256: bytes


@dataclass(frozen=True, slots=True)
class ExportRecord:
    row_number: int
    task_id: str
    task_index: int
    rollout_index: int
    job_dir: Path
    trial_uri: str
    trial_relative: Path
    artifacts: tuple[ExportArtifact, ...]
    sandbox_runtime_sha256: bytes
    gym_source: tuple[str, str, str]

    @property
    def destination_name(self) -> str:
        return self.job_dir.name


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EvidenceExportError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _load_json_object(payload: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError, EvidenceExportError) as error:
        raise EvidenceExportError(f"invalid {description}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceExportError(f"{description} must be a JSON object")
    return value


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def _relative_to_job(path: Path, job_dir: Path, description: str, row_number: int) -> Path:
    normalized = _absolute_lexical(path)
    try:
        relative = normalized.relative_to(job_dir)
    except ValueError as error:
        raise EvidenceExportError(f"row {row_number}: {description} is outside job_dir") from error
    if not relative.parts:
        raise EvidenceExportError(f"row {row_number}: {description} must identify a path inside job_dir")
    return relative


def _parse_file_uri(value: str, row_number: int) -> Path:
    parsed = urlparse(value)
    if parsed.scheme != "file" or parsed.netloc or parsed.params or parsed.query or parsed.fragment:
        raise EvidenceExportError(f"row {row_number}: trial_uri must be a local file:// URI")
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise EvidenceExportError(f"row {row_number}: trial_uri must contain an absolute path")
    return _absolute_lexical(path)


def _require_string(value: Any, description: str, row_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceExportError(f"row {row_number}: {description} must be a nonempty string")
    return value


def _require_bounded_string(
    value: Any,
    description: str,
    row_number: int,
    *,
    max_utf8_bytes: int,
) -> str:
    text = _require_string(value, description, row_number)
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise EvidenceExportError(f"row {row_number}: {description} is not valid UTF-8") from error
    if size > max_utf8_bytes:
        raise EvidenceExportError(f"row {row_number}: {description} exceeds {max_utf8_bytes} UTF-8 bytes")
    return text


def _bounded_canonical_json_sha256(
    value: Any,
    description: str,
    row_number: int,
    *,
    max_utf8_bytes: int,
) -> bytes:
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode("utf-8")
            size += len(encoded)
            if size > max_utf8_bytes:
                raise EvidenceExportError(
                    f"row {row_number}: canonical {description} exceeds {max_utf8_bytes} UTF-8 bytes"
                )
            digest.update(encoded)
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise EvidenceExportError(f"row {row_number}: {description} is not canonical JSON") from error
    return digest.digest()


def _parse_gym_source(value: Any, description: str, row_number: int) -> tuple[str, str, str]:
    if not isinstance(value, dict):
        raise EvidenceExportError(f"row {row_number}: {description} must be an object")
    expected_keys = {"repository_url", "commit", "uv_lock_sha256", "working_tree_clean"}
    if set(value) != expected_keys:
        raise EvidenceExportError(f"row {row_number}: {description} has an unsupported schema")
    repository_url = value.get("repository_url")
    commit = value.get("commit")
    lock_digest = value.get("uv_lock_sha256")
    if repository_url != GYM_REPOSITORY_URL:
        raise EvidenceExportError(f"row {row_number}: {description}.repository_url is not the NeMo Gym origin")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise EvidenceExportError(f"row {row_number}: {description}.commit is not a full Git SHA")
    if (
        not isinstance(lock_digest, str)
        or len(lock_digest) != 64
        or any(character not in "0123456789abcdef" for character in lock_digest)
    ):
        raise EvidenceExportError(f"row {row_number}: {description}.uv_lock_sha256 is invalid")
    if value.get("working_tree_clean") is not True:
        raise EvidenceExportError(f"row {row_number}: {description}.working_tree_clean is not true")
    return repository_url, commit, lock_digest


def _parse_record(row: dict[str, Any], row_number: int, source_root: Path) -> ExportRecord | None:
    metadata = row.get("benchmark_metadata")
    if not isinstance(metadata, dict) or metadata.get("benchmark") != "datacurve-ai/deep-swe":
        return None
    evidence_fields = ("job_dir", "trial_uri", "sandbox_runtime_path", "sandbox_runtime")
    if not any(field in metadata for field in evidence_fields):
        return None
    status = _require_bounded_string(
        row.get("status"), "status", row_number, max_utf8_bytes=MAX_TASK_STATUS_UTF8_BYTES
    )
    if status not in TERMINAL_STATUSES:
        raise EvidenceExportError(f"row {row_number}: status is not a terminal DeepSWE status")

    task_id = _require_bounded_string(
        row.get("task_id"), "task_id", row_number, max_utf8_bytes=MAX_TASK_STATUS_UTF8_BYTES
    )
    task_index = row.get("_ng_task_index")
    rollout_index = row.get("_ng_rollout_index")
    if isinstance(task_index, bool) or not isinstance(task_index, int) or task_index < 0:
        raise EvidenceExportError(f"row {row_number}: _ng_task_index must be a nonnegative integer")
    if isinstance(rollout_index, bool) or not isinstance(rollout_index, int) or rollout_index < 0:
        raise EvidenceExportError(f"row {row_number}: _ng_rollout_index must be a nonnegative integer")

    job_value = _require_bounded_string(
        metadata.get("job_dir"),
        "benchmark_metadata.job_dir",
        row_number,
        max_utf8_bytes=MAX_PATH_URI_UTF8_BYTES,
    )
    job_dir = Path(job_value)
    if not job_dir.is_absolute() or _absolute_lexical(job_dir) != job_dir:
        raise EvidenceExportError(f"row {row_number}: benchmark_metadata.job_dir must be a normalized absolute path")
    if job_dir.parent != source_root or job_dir.name.startswith(".unsafe-"):
        raise EvidenceExportError(f"row {row_number}: job_dir is not a direct safe child of the private source root")

    trial_uri = _require_bounded_string(
        metadata.get("trial_uri"),
        "benchmark_metadata.trial_uri",
        row_number,
        max_utf8_bytes=MAX_PATH_URI_UTF8_BYTES,
    )
    trial_relative = _relative_to_job(_parse_file_uri(trial_uri, row_number), job_dir, "trial_uri", row_number)
    runtime_value = _require_bounded_string(
        metadata.get("sandbox_runtime_path"),
        "benchmark_metadata.sandbox_runtime_path",
        row_number,
        max_utf8_bytes=MAX_PATH_URI_UTF8_BYTES,
    )
    runtime_path = Path(runtime_value)
    if not runtime_path.is_absolute() or _absolute_lexical(runtime_path) != runtime_path:
        raise EvidenceExportError(
            f"row {row_number}: benchmark_metadata.sandbox_runtime_path must be a normalized absolute path"
        )
    runtime_relative = _relative_to_job(runtime_path, job_dir, "sandbox_runtime_path", row_number)
    if runtime_relative != RUNTIME_PROVENANCE_RELATIVE:
        raise EvidenceExportError(f"row {row_number}: sandbox_runtime_path must name the job runtime provenance")
    sandbox_runtime = metadata.get("sandbox_runtime")
    if not isinstance(sandbox_runtime, dict):
        raise EvidenceExportError(f"row {row_number}: benchmark_metadata.sandbox_runtime must be an object")
    sandbox_runtime_sha256 = _bounded_canonical_json_sha256(
        sandbox_runtime,
        "benchmark_metadata.sandbox_runtime",
        row_number,
        max_utf8_bytes=MAX_SANDBOX_RUNTIME_CANONICAL_BYTES,
    )
    gym_source = _parse_gym_source(metadata.get("gym_source"), "benchmark_metadata.gym_source", row_number)
    runtime_gym_source = _parse_gym_source(
        sandbox_runtime.get("gym_source"),
        "benchmark_metadata.sandbox_runtime.gym_source",
        row_number,
    )
    if runtime_gym_source != gym_source:
        raise EvidenceExportError(f"row {row_number}: Gym source provenance copies differ")

    artifacts_value = row.get("artifacts")
    if not isinstance(artifacts_value, list) or not artifacts_value:
        raise EvidenceExportError(f"row {row_number}: artifacts must be a nonempty list")
    if len(artifacts_value) > MAX_ARTIFACT_ENTRIES_PER_ROW:
        raise EvidenceExportError(f"row {row_number}: artifacts exceed {MAX_ARTIFACT_ENTRIES_PER_ROW} entries")
    artifacts: list[ExportArtifact] = []
    seen_paths: set[str] = set()
    artifact_paths_utf8_bytes = 0
    for artifact in artifacts_value:
        if not isinstance(artifact, dict):
            raise EvidenceExportError(f"row {row_number}: every artifact must be an object")
        path_value = _require_bounded_string(
            artifact.get("path"),
            "artifacts[].path",
            row_number,
            max_utf8_bytes=MAX_PATH_URI_UTF8_BYTES,
        )
        artifact_paths_utf8_bytes += len(path_value.encode("utf-8"))
        if artifact_paths_utf8_bytes > MAX_ARTIFACT_PATHS_UTF8_BYTES:
            raise EvidenceExportError(
                f"row {row_number}: artifact paths exceed {MAX_ARTIFACT_PATHS_UTF8_BYTES} UTF-8 bytes"
            )
        relative_path = Path(path_value)
        if relative_path.is_absolute() or ".." in relative_path.parts or path_value in seen_paths:
            raise EvidenceExportError(f"row {row_number}: unsafe or duplicate artifact path {path_value!r}")
        size = artifact.get("bytes")
        digest = artifact.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise EvidenceExportError(f"row {row_number}: artifact {path_value!r} has an invalid size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise EvidenceExportError(f"row {row_number}: artifact {path_value!r} has an invalid SHA-256")
        seen_paths.add(path_value)
        artifacts.append(ExportArtifact(path=path_value, size=size, sha256=bytes.fromhex(digest)))

    return ExportRecord(
        row_number=row_number,
        task_id=task_id,
        task_index=task_index,
        rollout_index=rollout_index,
        job_dir=job_dir,
        trial_uri=trial_uri,
        trial_relative=trial_relative,
        artifacts=tuple(artifacts),
        sandbox_runtime_sha256=sandbox_runtime_sha256,
        gym_source=gym_source,
    )


def _load_rollouts(
    path: Path,
    source_root: Path,
    *,
    max_line_bytes: int,
    allow_incomplete_trailing_line: bool,
    require_stable: bool = True,
) -> tuple[list[ExportRecord], str, int]:
    records: list[ExportRecord] = []
    row_count = 0
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise EvidenceExportError(f"rollout JSONL is unavailable: {path}") from error
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        os.close(descriptor)
        raise EvidenceExportError("rollout JSONL must be a regular single-link file")
    stream = os.fdopen(descriptor, "rb", closefd=False)
    try:
        while raw_line := stream.readline(max_line_bytes + 1):
            candidate_row = row_count + 1
            if candidate_row > MAX_ROLLOUT_ROWS:
                raise EvidenceExportError(f"rollout file exceeds {MAX_ROLLOUT_ROWS} rows")
            if len(raw_line) > max_line_bytes:
                raise EvidenceExportError(f"line {candidate_row}: rollout row exceeds {max_line_bytes} bytes")
            if not raw_line.endswith(b"\n"):
                if allow_incomplete_trailing_line:
                    break
                raise EvidenceExportError(f"line {candidate_row}: rollout JSONL must end each row with a newline")
            row_count = candidate_row
            digest.update(raw_line)
            row_payload = raw_line[:-1]
            if not row_payload.strip():
                raise EvidenceExportError(f"line {row_count}: blank JSONL row")
            row = _load_json_object(row_payload, f"rollout row {row_count}")
            record = _parse_record(row, row_count, source_root)
            if record is not None:
                records.append(record)
        if require_stable:
            after = os.fstat(descriptor)
            try:
                current = path.lstat()
            except OSError as error:
                raise EvidenceExportError("rollout JSONL changed during final export") from error
            if _metadata_signature(after) != _metadata_signature(opened) or _metadata_signature(
                current
            ) != _metadata_signature(opened):
                raise EvidenceExportError("rollout JSONL changed during final export")
    finally:
        stream.close()
        os.close(descriptor)
    if not records:
        raise EvidenceExportError("rollout file contains no finalized DeepSWE evidence rows")
    destinations = [record.destination_name for record in records]
    if len(destinations) != len(set(destinations)):
        raise EvidenceExportError("finalized rollout rows reference a duplicate job directory")
    row_keys = [(record.task_id, record.rollout_index) for record in records]
    if len(row_keys) != len(set(row_keys)):
        raise EvidenceExportError("finalized rollout rows contain duplicate task/repeat keys")
    gym_sources = {record.gym_source for record in records}
    if len(gym_sources) != 1:
        raise EvidenceExportError("finalized rollout rows contain inconsistent Gym source provenance")
    return records, digest.hexdigest(), row_count


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _directory_flags())
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_export_root(path: Path) -> Path:
    """Create only the final owner-only component below trusted ancestors."""

    path = _absolute_lexical(path)
    if path == Path("/"):
        raise EvidenceExportError("export root cannot be the filesystem root")
    descriptor = os.open("/", _directory_flags())
    current = Path("/")
    try:
        for index, component in enumerate(path.parts[1:]):
            final = index == len(path.parts[1:]) - 1
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                if not final:
                    raise EvidenceExportError(f"export root has a missing ancestor: {current / component}") from None
                os.mkdir(component, 0o700, dir_fd=descriptor)
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except OSError as error:
                raise EvidenceExportError(
                    f"export root has a missing or symlinked ancestor: {current / component}"
                ) from error
            os.close(descriptor)
            descriptor = child
            current /= component
            metadata = os.fstat(descriptor)
            if final:
                if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
                    raise EvidenceExportError("export root must be current-user-owned with mode 0700")
            else:
                if metadata.st_uid not in {0, os.geteuid()}:
                    raise EvidenceExportError(f"export root has an untrusted ancestor: {current}")
                writable_by_others = stat.S_IMODE(metadata.st_mode) & 0o022
                if writable_by_others and not metadata.st_mode & stat.S_ISVTX:
                    raise EvidenceExportError(f"export root has an attacker-writable ancestor: {current}")
        metadata = os.fstat(descriptor)
        try:
            current_metadata = path.lstat()
        except OSError as error:
            raise EvidenceExportError("export root changed while it was being validated") from error
        if (metadata.st_dev, metadata.st_ino) != (
            current_metadata.st_dev,
            current_metadata.st_ino,
        ):
            raise EvidenceExportError("export root changed while it was being validated")
        return path
    finally:
        os.close(descriptor)


def _validate_owner_only_tree(root: Path, entry_paths: tuple[str, ...]) -> None:
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_uid != os.geteuid():
        raise EvidenceExportError(f"evidence tree is not a current-user-owned directory: {root}")
    if stat.S_IMODE(root_metadata.st_mode) & 0o077:
        raise EvidenceExportError(f"evidence tree is not owner-only: {root}")
    for relative in entry_paths:
        path = root / relative
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise EvidenceExportError(f"evidence entry is unsafe: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise EvidenceExportError(f"evidence entry is not owner-only: {path}")


def _metadata_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _copy_private_tree(source: Path, destination: Path, limits: ArtifactLimits) -> list[dict[str, Any]]:
    source_descriptor = os.open(source, _directory_flags())
    source_metadata = os.fstat(source_descriptor)
    if source_metadata.st_uid != os.geteuid() or stat.S_IMODE(source_metadata.st_mode) != 0o500:
        os.close(source_descriptor)
        raise EvidenceExportError(f"source job is not sealed mode 0500: {source}")
    destination.mkdir(mode=0o700, exist_ok=False)
    destination_descriptor = os.open(destination, _directory_flags())
    manifest: list[dict[str, Any]] = []
    file_count = 0
    entry_count = 0
    total_bytes = 0

    def copy_directory(source_fd: int, destination_fd: int, prefix: str) -> None:
        nonlocal file_count, entry_count, total_bytes
        entries: list[os.DirEntry[str]] = []
        with os.scandir(source_fd) as iterator:
            for entry in iterator:
                entry_count += 1
                if entry_count > limits.entry_limit:
                    raise ArtifactLimitError(f"DeepSWE export entry count exceeds {limits.entry_limit}")
                entries.append(entry)
        entries.sort(key=lambda entry: entry.name)
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            before = entry.stat(follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) and not stat.S_ISREG(before.st_mode):
                raise EvidenceExportError(f"source evidence entry is not a safe regular file: {relative}")
            if before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) & 0o077:
                raise EvidenceExportError(f"source evidence entry is not owner-only: {relative}")
            if stat.S_ISDIR(before.st_mode):
                if stat.S_IMODE(before.st_mode) != 0o500:
                    raise EvidenceExportError(f"source evidence directory is not sealed mode 0500: {relative}")
                child_source = os.open(entry.name, _directory_flags(), dir_fd=source_fd)
                try:
                    opened = os.fstat(child_source)
                    if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                        raise EvidenceExportError(f"source evidence directory changed before copy: {relative}")
                    os.mkdir(entry.name, 0o700, dir_fd=destination_fd)
                    child_destination = os.open(entry.name, _directory_flags(), dir_fd=destination_fd)
                    try:
                        copy_directory(child_source, child_destination, relative)
                        os.chmod(entry.name, stat.S_IMODE(opened.st_mode) & 0o700, dir_fd=destination_fd)
                        os.fsync(child_destination)
                    finally:
                        os.close(child_destination)
                    if _metadata_signature(os.fstat(child_source)) != _metadata_signature(opened):
                        raise EvidenceExportError(f"source evidence directory changed during copy: {relative}")
                    current = os.stat(entry.name, dir_fd=source_fd, follow_symlinks=False)
                    if _metadata_signature(current) != _metadata_signature(opened):
                        raise EvidenceExportError(f"source evidence directory name changed during copy: {relative}")
                finally:
                    os.close(child_source)
                continue
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise EvidenceExportError(f"source evidence entry is not a safe regular file: {relative}")
            if stat.S_IMODE(before.st_mode) != 0o400:
                raise EvidenceExportError(f"source evidence file is not sealed mode 0400: {relative}")
            file_count += 1
            if file_count > limits.max_files or before.st_size > limits.max_file_bytes:
                raise ArtifactLimitError(f"DeepSWE export file budget exceeded: {relative}")
            if total_bytes + before.st_size > limits.max_total_bytes:
                raise ArtifactLimitError("DeepSWE export total byte budget exceeded")
            source_file = os.open(
                entry.name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=source_fd,
            )
            destination_file = os.open(
                entry.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=destination_fd,
            )
            digest = hashlib.sha256()
            size = 0
            try:
                opened = os.fstat(source_file)
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise EvidenceExportError(f"source evidence file changed before copy: {relative}")
                while chunk := os.read(source_file, min(1024 * 1024, limits.max_file_bytes - size + 1)):
                    size += len(chunk)
                    total_bytes += len(chunk)
                    if size > limits.max_file_bytes or total_bytes > limits.max_total_bytes:
                        raise ArtifactLimitError(f"DeepSWE export byte budget exceeded while copying: {relative}")
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_file, view)
                        if written <= 0:
                            raise EvidenceExportError(f"evidence copy made no write progress: {relative}")
                        view = view[written:]
                os.fsync(destination_file)
                if size != opened.st_size or _metadata_signature(os.fstat(source_file)) != _metadata_signature(opened):
                    raise EvidenceExportError(f"source evidence file changed during copy: {relative}")
                current = os.stat(entry.name, dir_fd=source_fd, follow_symlinks=False)
                if _metadata_signature(current) != _metadata_signature(opened):
                    raise EvidenceExportError(f"source evidence file name changed during copy: {relative}")
            finally:
                os.close(source_file)
                os.close(destination_file)
            os.chmod(entry.name, stat.S_IMODE(opened.st_mode) & 0o600, dir_fd=destination_fd)
            manifest.append({"path": relative, "bytes": size, "sha256": digest.hexdigest()})

        os.fsync(destination_fd)

    try:
        copy_directory(source_descriptor, destination_descriptor, "")
        os.fsync(destination_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(destination_descriptor)
    os.chmod(destination, stat.S_IMODE(source_metadata.st_mode) & 0o700)
    current_source = source.lstat()
    if _metadata_signature(current_source) != _metadata_signature(source_metadata):
        raise EvidenceExportError(f"source job changed during copy: {source}")
    manifest.sort(key=lambda item: item["path"])
    return manifest


def _manifest_digest(manifest: list[dict[str, Any]]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _snapshot(root: Path, limits: ArtifactLimits) -> list[dict[str, Any]]:
    snapshot = artifact_snapshot(root, limits=limits)
    _validate_owner_only_tree(root, snapshot.entry_paths)
    return snapshot.manifest


def _validate_record_evidence(record: ExportRecord, exported_job: Path, manifest: list[dict[str, Any]]) -> None:
    indexed = {item["path"]: item for item in manifest}
    if len(indexed) != len(manifest):
        raise EvidenceExportError(f"row {record.row_number}: exported job contains duplicate paths")
    for claimed in record.artifacts:
        full_path = (record.trial_relative / claimed.path).as_posix()
        retained = indexed.get(full_path)
        if retained is None or retained.get("bytes") != claimed.size or retained.get("sha256") != claimed.sha256.hex():
            raise EvidenceExportError(
                f"row {record.row_number}: exported artifact does not match rollout manifest: {claimed.path!r}"
            )
    runtime_file = exported_job / RUNTIME_PROVENANCE_RELATIVE
    try:
        runtime_payload = read_bytes(runtime_file, max_bytes=1024 * 1024, description="sandbox runtime provenance")
    except (ArtifactLimitError, OSError) as error:
        raise EvidenceExportError(
            f"row {record.row_number}: exported sandbox runtime provenance is unavailable"
        ) from error
    persisted_runtime = _load_json_object(runtime_payload, "exported sandbox runtime provenance")
    persisted_runtime_sha256 = _bounded_canonical_json_sha256(
        persisted_runtime,
        "exported sandbox runtime provenance",
        record.row_number,
        max_utf8_bytes=MAX_SANDBOX_RUNTIME_CANONICAL_BYTES,
    )
    if persisted_runtime_sha256 != record.sandbox_runtime_sha256:
        raise EvidenceExportError(f"row {record.row_number}: exported sandbox runtime provenance differs from rollout")


def _receipt_entry(record: ExportRecord, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_id": record.task_id,
        "task_index": record.task_index,
        "rollout_index": record.rollout_index,
        "destination": record.destination_name,
        "original_job_dir": str(record.job_dir),
        "original_trial_uri": record.trial_uri,
        "original_sandbox_runtime_path": str(record.job_dir / RUNTIME_PROVENANCE_RELATIVE),
        "manifest_sha256": _manifest_digest(manifest),
        "files": len(manifest),
        "bytes": sum(int(item["bytes"]) for item in manifest),
    }


def _entry_matches_record(entry: dict[str, Any], record: ExportRecord) -> bool:
    expected = {
        "task_id": record.task_id,
        "task_index": record.task_index,
        "rollout_index": record.rollout_index,
        "destination": record.destination_name,
        "original_job_dir": str(record.job_dir),
        "original_trial_uri": record.trial_uri,
        "original_sandbox_runtime_path": str(record.job_dir / RUNTIME_PROVENANCE_RELATIVE),
    }
    return all(entry.get(key) == value for key, value in expected.items())


def _load_receipt(path: Path) -> dict[str, dict[str, Any]]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {}
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise EvidenceExportError("existing DeepSWE export receipt is unsafe")
    try:
        payload = read_bytes(
            path,
            max_bytes=16 * 1024 * 1024,
            description="export receipt",
        )
    except ArtifactLimitError as error:
        raise EvidenceExportError("existing DeepSWE export receipt is unsafe") from error
    receipt = _load_json_object(payload, "existing DeepSWE export receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION or not isinstance(receipt.get("jobs"), list):
        raise EvidenceExportError("existing DeepSWE export receipt has an unsupported schema")
    entries: dict[str, dict[str, Any]] = {}
    for value in receipt["jobs"]:
        if not isinstance(value, dict) or not isinstance(value.get("destination"), str):
            raise EvidenceExportError("existing DeepSWE export receipt contains an invalid job entry")
        if value["destination"] in entries:
            raise EvidenceExportError("existing DeepSWE export receipt contains duplicate destinations")
        digest = value.get("manifest_sha256")
        files = value.get("files")
        total_bytes = value.get("bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(files, bool)
            or not isinstance(files, int)
            or files < 1
            or isinstance(total_bytes, bool)
            or not isinstance(total_bytes, int)
            or total_bytes < 0
        ):
            raise EvidenceExportError("existing DeepSWE export receipt contains an invalid manifest summary")
        entries[value["destination"]] = value
    return entries


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    temporary = path.parent / f".{RECEIPT_NAME}.{uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise EvidenceExportError("export receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _remove_partial_tree(path: Path) -> None:
    """Make one exporter-created partial tree removable, then remove it."""

    _snapshot(path, DEFAULT_LIMITS)
    for directory, directory_names, file_names in os.walk(path, topdown=True, followlinks=False):
        os.chmod(directory, 0o700, follow_symlinks=False)
        for name in directory_names:
            child = Path(directory) / name
            if child.is_symlink():
                raise EvidenceExportError(f"partial export unexpectedly contains a symlink: {child}")
        for name in file_names:
            child = Path(directory) / name
            if child.is_symlink():
                raise EvidenceExportError(f"partial export unexpectedly contains a symlink: {child}")
            os.chmod(child, 0o600, follow_symlinks=False)
    shutil.rmtree(path)


def _cleanup_stale_temporary_entries(export_root: Path) -> None:
    for path in export_root.iterdir():
        if path.name.startswith(".partial-"):
            metadata = path.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise EvidenceExportError(f"stale partial export is unsafe: {path.name}")
            _remove_partial_tree(path)
        elif path.name.startswith(f".{RECEIPT_NAME}.") and path.name.endswith(".tmp"):
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise EvidenceExportError(f"stale receipt temporary is unsafe: {path.name}")
            path.unlink()


def export_evidence(
    rollouts: Path,
    exported_artifacts_root: Path,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    limits: ArtifactLimits = DEFAULT_LIMITS,
    rollout_max_line_bytes: int = DEFAULT_ROLLOUT_MAX_LINE_BYTES,
    allow_incomplete_trailing_line: bool = False,
    incremental: bool = False,
) -> dict[str, Any]:
    """Export finalized rollout-referenced jobs and return the stable receipt."""

    source_root = _absolute_lexical(source_root)
    if not source_root.exists():
        raise EvidenceExportError(f"private source root does not exist: {source_root}")
    ensure_private_directory(source_root)
    export_root = _ensure_export_root(exported_artifacts_root)
    if (
        export_root == source_root
        or export_root.is_relative_to(source_root)
        or source_root.is_relative_to(export_root)
    ):
        raise EvidenceExportError("export root and private source root must not overlap")
    _cleanup_stale_temporary_entries(export_root)
    records, rollouts_sha256, row_count = _load_rollouts(
        rollouts,
        source_root,
        max_line_bytes=rollout_max_line_bytes,
        allow_incomplete_trailing_line=allow_incomplete_trailing_line,
        require_stable=not incremental,
    )
    existing_entries = _load_receipt(export_root / RECEIPT_NAME)
    selected_names = {record.destination_name for record in records}
    stale_receipts = set(existing_entries) - selected_names
    if stale_receipts:
        raise EvidenceExportError(
            f"export receipt contains jobs not referenced by current finalized rows: {sorted(stale_receipts)}"
        )

    entries: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (item.task_index, item.rollout_index, item.task_id)):
        destination = export_root / record.destination_name
        existing_entry = existing_entries.get(record.destination_name)
        if destination.exists():
            if existing_entry is not None and not _entry_matches_record(existing_entry, record):
                raise EvidenceExportError(f"existing export receipt does not match row {record.row_number}")
            if incremental and existing_entry is not None:
                metadata = destination.lstat()
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) & 0o077
                ):
                    raise EvidenceExportError(
                        f"existing incremental export is not a private directory: {record.destination_name}"
                    )
                entries.append(dict(existing_entry))
                continue
            destination_manifest = _snapshot(destination, limits)
            _validate_record_evidence(record, destination, destination_manifest)
            if existing_entry is None:
                if not record.job_dir.exists():
                    raise EvidenceExportError(
                        f"row {record.row_number}: cannot recover an unreceipted export after its private source disappeared"
                    )
                source_manifest = _snapshot(record.job_dir, limits)
                if source_manifest != destination_manifest:
                    raise EvidenceExportError(
                        f"row {record.row_number}: unreceipted export differs from its private source"
                    )
            elif (
                existing_entry.get("manifest_sha256") != _manifest_digest(destination_manifest)
                or existing_entry.get("files") != len(destination_manifest)
                or existing_entry.get("bytes") != sum(int(item["bytes"]) for item in destination_manifest)
            ):
                raise EvidenceExportError(
                    f"existing exported job failed receipt validation: {record.destination_name}"
                )
            entries.append(_receipt_entry(record, destination_manifest))
            continue

        if existing_entry is not None:
            raise EvidenceExportError(f"receipt names a missing exported job: {record.destination_name}")
        if not record.job_dir.exists():
            raise EvidenceExportError(f"row {record.row_number}: referenced private job is missing: {record.job_dir}")
        temporary = export_root / f".partial-{record.destination_name}-{uuid4().hex}"
        try:
            copied_manifest = _copy_private_tree(record.job_dir, temporary, limits)
            verified_manifest = _snapshot(temporary, limits)
            if verified_manifest != copied_manifest:
                raise EvidenceExportError(f"row {record.row_number}: copied evidence changed before publication")
            _validate_record_evidence(record, temporary, verified_manifest)
            os.rename(temporary, destination)
            _fsync_directory(export_root)
        except BaseException:
            if temporary.exists():
                _remove_partial_tree(temporary)
            raise
        entries.append(_receipt_entry(record, verified_manifest))

    allowed = selected_names | {RECEIPT_NAME}
    for path in export_root.iterdir():
        if path.name not in allowed:
            raise EvidenceExportError(f"export root contains unreferenced entry: {path.name}")
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "rollouts_path": str(_absolute_lexical(rollouts)),
        "rollouts_sha256": rollouts_sha256,
        "rollout_rows": row_count,
        "exported_jobs": len(entries),
        "jobs": entries,
    }
    _write_receipt(export_root / RECEIPT_NAME, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deep-swe-export-evidence")
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--exported-artifacts-root", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--allow-incomplete-trailing-line", action="store_true")
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = export_evidence(
            args.rollouts,
            args.exported_artifacts_root,
            source_root=args.source_root,
            allow_incomplete_trailing_line=args.allow_incomplete_trailing_line,
            incremental=args.incremental,
        )
    except (ArtifactLimitError, EvidenceExportError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
