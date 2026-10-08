# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest-last storage for one participant's checkpoint records.

Layout under the controller-owned directory::

    <checkpoint_dir>/gym/<kind>/<instance>/records-<sha256>.jsonl
    <checkpoint_dir>/gym/<kind>/<instance>/manifest.json

Each participant writes one records file regardless of how many rollouts it holds, so shared
filesystem metadata operations do not grow with the live set. The records file is written under a
temporary name one record at a time, hashed as it goes, flushed, and renamed after its digest, so a commit
never holds the whole file in memory and a records file never changes once it has its name. The manifest
carrying the digest is written last and only created, never replaced: a directory with a manifest never
changes. Committing the same checkpoint again returns the existing manifest, and committing a different
checkpoint into it fails, even when two writers race.
"""

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from nemo_gym._checkpoint.errors import CheckpointStateError


STATE_SCHEMA_VERSION = 1
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Named after the digest; checkpoints written before that used one fixed name.
_RECORDS_FILE_PATTERN = re.compile(r"^records(-[0-9a-f]{64})?\.jsonl$")


def participant_dir(checkpoint_dir: Path, *, kind: str, instance: str) -> Path:
    for label, value in (("kind", kind), ("instance", instance)):
        if not _NAME_PATTERN.match(value):
            raise CheckpointStateError(f"participant {label} {value!r} is not a safe path component")
    return checkpoint_dir / "gym" / kind / instance


class WriteStop:
    """Lets resume stop a write.
    Once ``stop`` has returned, the write never publishes a manifest.

    The writer checks before each record without the lock, and checks again and publishes under it,
    so ``stop`` waits only for a publication already under way, never for the rest of the write.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stopped = False

    def stop(self) -> None:
        with self._lock:
            self._stopped = True

    def check(self, checkpoint_id: str) -> None:
        if self._stopped:
            raise CheckpointStateError(f"checkpoint {checkpoint_id!r} was resumed before its write finished")

    @contextmanager
    def publishing(self, checkpoint_id: str) -> Iterator[None]:
        with self._lock:
            self.check(checkpoint_id)
            yield


def write_participant_state(
    checkpoint_dir: Path,
    *,
    kind: str,
    instance: str,
    checkpoint_id: str,
    records: Iterable[dict[str, Any]],
    extra: Optional[dict[str, Any]] = None,
    stop: Optional[WriteStop] = None,
) -> dict[str, Any]:
    """Write ``records`` and then the manifest that publishes them, which also keeps the participant's ``extra``
    fields.

    ``stop`` aborts the write before the next record and before publishing,
    so nothing is published once the checkpoint was resumed.
    """
    try:
        return _write(checkpoint_dir, kind, instance, checkpoint_id, records, extra or {}, stop or WriteStop())
    except OSError as error:
        raise CheckpointStateError(f"cannot write checkpoint state under {checkpoint_dir}: {error}") from error


def _write(
    checkpoint_dir: Path,
    kind: str,
    instance: str,
    checkpoint_id: str,
    records: Iterable[dict[str, Any]],
    extra: dict[str, Any],
    stop: WriteStop,
) -> dict[str, Any]:
    directory = participant_dir(checkpoint_dir, kind=kind, instance=instance)
    manifest_path = directory / "manifest.json"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".records.jsonl.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                stop.check(checkpoint_id)
                handle.write(line := _record_line(record))
                digest.update(line)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        records_file = f"records-{digest.hexdigest()}.jsonl"
        manifest = {
            "schema_version": STATE_SCHEMA_VERSION,
            "kind": kind,
            "instance": instance,
            "checkpoint_id": checkpoint_id,
            "records_file": records_file,
            "records_sha256": digest.hexdigest(),
            "record_count": count,
            **extra,
        }
        with stop.publishing(checkpoint_id):
            # Same name, same content: replacing a file another writer of the same records renamed changes nothing.
            os.replace(temporary, directory / records_file)
            created = _create(manifest_path, json.dumps(manifest, sort_keys=True, indent=1).encode())
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    if created:
        _fsync_dir(directory)
        return manifest
    existing = _load_manifest(manifest_path)
    if existing != manifest:
        if existing.get("records_file") != records_file:
            (directory / records_file).unlink(missing_ok=True)
        raise CheckpointStateError(f"{manifest_path} already holds a different commit")
    return existing


def _record_line(record: dict[str, Any]) -> bytes:
    # Keep each record's key order: a restored episode must see its state exactly as exported,
    # and state such as a tool environment is often rendered back to the model by serializing a stored dict.
    try:
        return json.dumps(record).encode() + b"\n"
    except (TypeError, ValueError) as error:
        raise CheckpointStateError(
            f"checkpoint record of episode {record.get('episode_id')!r} is not JSON: {error}"
        ) from error


def read_participant_state(
    checkpoint_dir: Path,
    *,
    kind: str,
    instance: str,
    select: Optional[Callable[[Any], Any]] = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Read and verify a participant's records in one streaming pass.

    ``select`` maps each stored record to what to keep, or ``None`` to drop it,
    so a restore holds only the records it installs.
    The digest and record count always cover the whole file.
    """
    directory = participant_dir(checkpoint_dir, kind=kind, instance=instance)
    try:
        return _read(directory, kind, instance, select)
    except OSError as error:
        raise CheckpointStateError(f"cannot read checkpoint state from {directory}: {error}") from error


def _read(
    directory: Path, kind: str, instance: str, select: Optional[Callable[[Any], Any]]
) -> tuple[dict[str, Any], list[Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise CheckpointStateError(f"missing manifest {manifest_path}")
    manifest = _load_manifest(manifest_path)
    if manifest.get("schema_version") != STATE_SCHEMA_VERSION:
        raise CheckpointStateError(f"{manifest_path} has unsupported schema {manifest.get('schema_version')!r}")
    if (manifest.get("kind"), manifest.get("instance")) != (kind, instance):
        raise CheckpointStateError(f"{manifest_path} belongs to {manifest.get('kind')}/{manifest.get('instance')}")
    if not isinstance(manifest.get("checkpoint_id"), str):
        raise CheckpointStateError(f"{manifest_path} names no checkpoint")
    records_file = manifest.get("records_file")
    if not isinstance(records_file, str) or not _RECORDS_FILE_PATTERN.match(records_file):
        raise CheckpointStateError(f"{manifest_path} names an invalid records file {records_file!r}")
    records_path = directory / records_file
    digest = hashlib.sha256()
    count = 0
    records = []
    with records_path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            count += 1
            try:
                record = json.loads(line)
            except ValueError as error:
                raise CheckpointStateError(f"{records_path} holds a record that is not JSON: {error}") from error
            if select is not None:
                record = select(record)
            if record is not None:
                records.append(record)
    if digest.hexdigest() != manifest.get("records_sha256"):
        raise CheckpointStateError(f"{records_path} does not match its manifest digest")
    if count != manifest.get("record_count"):
        raise CheckpointStateError(f"{manifest_path} expects {manifest.get('record_count')!r} records, found {count}")
    return manifest, records


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except ValueError as error:
        raise CheckpointStateError(f"{path} is not JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise CheckpointStateError(f"{path} is not a manifest")
    return manifest


def _create(path: Path, payload: bytes) -> bool:
    """Write ``path`` atomically unless it already exists; return whether this call created it."""
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link never replaces an existing name, so of two racing writers exactly one publishes.
        os.link(temporary, path)
        return True
    except FileExistsError:
        return False
    finally:
        Path(temporary).unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
