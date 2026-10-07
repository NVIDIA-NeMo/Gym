# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest-last storage for one participant's checkpoint records.

Layout under the controller-owned directory::

    <checkpoint_dir>/gym/<kind>/<instance>/records.jsonl
    <checkpoint_dir>/gym/<kind>/<instance>/manifest.json

Each participant writes one records file regardless of how many rollouts it holds, so shared
filesystem metadata operations do not grow with the live set. The records file is written under a
temporary name one record at a time, hashed as it goes, flushed, and renamed, so a commit never holds
the whole file in memory; the manifest carrying its digest is written the same way and last. A
directory with a manifest never changes: committing the same checkpoint again returns the existing
manifest, and committing a different checkpoint into it fails.
"""

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
                # Keep each record's key order: a restored episode must see its state exactly as exported, and
                # state such as a tool environment is often rendered back to the model by serializing a stored dict.
                line = json.dumps(record).encode() + b"\n"
                digest.update(line)
                handle.write(line)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            "schema_version": STATE_SCHEMA_VERSION,
            "kind": kind,
            "instance": instance,
            "checkpoint_id": checkpoint_id,
            "records_file": "records.jsonl",
            "records_sha256": digest.hexdigest(),
            "record_count": count,
        }
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing != manifest:
                raise CheckpointStateError(f"{manifest_path} already holds a different commit")
            Path(temporary).unlink()
            return existing
        os.replace(temporary, directory / "records.jsonl")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    _atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, indent=1).encode())
    _fsync_dir(directory)
    return manifest


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


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
