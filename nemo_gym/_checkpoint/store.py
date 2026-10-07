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
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from nemo_gym._checkpoint.errors import CheckpointStateError


STATE_SCHEMA_VERSION = 1
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def participant_dir(checkpoint_dir: Path, *, kind: str, instance: str) -> Path:
    for label, value in (("kind", kind), ("instance", instance)):
        if not _NAME_PATTERN.match(value):
            raise CheckpointStateError(f"participant {label} {value!r} is not a safe path component")
    return checkpoint_dir / "gym" / kind / instance


def write_participant_state(
    checkpoint_dir: Path,
    *,
    kind: str,
    instance: str,
    checkpoint_id: str,
    records: Iterable[dict[str, Any]],
    stop: Optional[threading.Event] = None,
) -> dict[str, Any]:
    """Write ``records`` and then the manifest that publishes them.

    ``stop`` aborts the write before the next record and before the manifest, so nothing is published once the
    checkpoint was resumed.
    """
    directory = participant_dir(checkpoint_dir, kind=kind, instance=instance)
    manifest_path = directory / "manifest.json"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".records.jsonl.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                _check_stop(stop, checkpoint_id)
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
        }
        _check_stop(stop, checkpoint_id)
        # Same name, same content: replacing a file another writer of the same records renamed changes nothing.
        os.replace(temporary, directory / records_file)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    if _create(manifest_path, json.dumps(manifest, sort_keys=True, indent=1).encode()):
        _fsync_dir(directory)
        return manifest
    existing = json.loads(manifest_path.read_text())
    if existing != manifest:
        if existing.get("records_file") != records_file:
            (directory / records_file).unlink(missing_ok=True)
        raise CheckpointStateError(f"{manifest_path} already holds a different commit")
    return existing


def _record_line(record: dict[str, Any]) -> bytes:
    # Keep each record's key order: a restored episode must see its state exactly as exported, and
    # state such as a tool environment is often rendered back to the model by serializing a stored dict.
    try:
        return json.dumps(record).encode() + b"\n"
    except (TypeError, ValueError) as error:
        raise CheckpointStateError(
            f"checkpoint record of episode {record.get('episode_id')!r} is not JSON: {error}"
        ) from error


def _check_stop(stop: Optional[threading.Event], checkpoint_id: str) -> None:
    if stop is not None and stop.is_set():
        raise CheckpointStateError(f"checkpoint {checkpoint_id!r} was resumed before its write finished")


def read_participant_state(checkpoint_dir: Path, *, kind: str, instance: str) -> tuple[dict[str, Any], list[Any]]:
    directory = participant_dir(checkpoint_dir, kind=kind, instance=instance)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise CheckpointStateError(f"missing manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != STATE_SCHEMA_VERSION:
        raise CheckpointStateError(f"{manifest_path} has unsupported schema {manifest.get('schema_version')!r}")
    if (manifest.get("kind"), manifest.get("instance")) != (kind, instance):
        raise CheckpointStateError(f"{manifest_path} belongs to {manifest.get('kind')}/{manifest.get('instance')}")
    records_path = directory / manifest["records_file"]
    digest = hashlib.sha256()
    records = []
    with records_path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            try:
                records.append(json.loads(line))
            except ValueError as error:
                raise CheckpointStateError(f"{records_path} holds a record that is not JSON: {error}") from error
    if digest.hexdigest() != manifest["records_sha256"]:
        raise CheckpointStateError(f"{records_path} does not match its manifest digest")
    if len(records) != manifest["record_count"]:
        raise CheckpointStateError(f"{manifest_path} expects {manifest['record_count']} records, found {len(records)}")
    return manifest, records


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
