# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Token-free capture-ledger checkpoint commit and restore.

PR #2872 separates token custody from token storage. The generation worker
stages token arrays in the training framework's TransferQueue. The Gym model
server stores token-free lineage rows that identify those staged entries.
This participant checkpoints only those lineage rows. TransferQueue owns its
own checkpoint and restore.

Three properties make the copy a checkpoint rather than a backup:

- **Tombstone exclusion.** A rollout attempt force-closed at the prepare
  deadline must not restore: its rows describe an execution the restored run
  replaces with a fresh dispatch. Commit skips tombstoned attempts and
  records the tombstones in the manifest so the restored server re-installs
  the fence before serving anything.
- **Manifest-last ordering.** Every bounded lineage archive and its index is
  written and fsynced before the manifest appears (temporary name, fsync,
  rename). A commit that died partway leaves no manifest, and restore refuses
  the directory instead of installing a torn ledger.
- **Digest verification.** The manifest authenticates each archive and the
  index authenticates each lineage member. Restore verifies both layers, so
  silent corruption in transit fails loudly instead of surfacing as wrong
  training data later.
"""

import asyncio
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, runtime_checkable

from fastapi import FastAPI, Header
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym._checkpoint.admission import AdmissionLimiter
from nemo_gym._checkpoint.artifacts import (
    AgentContinuationRoot,
    CheckpointArtifactError,
    CheckpointArtifactReference,
    ExternalStorageReference,
    read_jsonl_artifact,
    write_jsonl_artifact,
)
from nemo_gym._checkpoint.control import (
    CONTROL_URL_PREFIX,
    AdmissionState,
    CheckpointControlRequest,
    CheckpointPhase,
    ControlError,
    ControlFence,
)
from nemo_gym._checkpoint.coordinator import (
    AdmissionCoordinator,
    CoordinatorServiceClient,
    MissingWorkersError,
)
from nemo_gym._checkpoint.model_admission import NotPolicyInstanceError
from nemo_gym._checkpoint.model_control_contracts import (
    GenerationCutLineageRecord,
    GenerationCutReceipt,
    GenerationCutReplacement,
    ModelAbortInflightRequest,
    ModelAdmissionPauseRequest,
    ModelAdmissionResumeRequest,
)
from nemo_gym.rollout_correlation import ROLLOUT_ID_PATTERN, capture_key_for
from nemo_gym.token_id_capture.control_routes import require_control_auth
from nemo_gym.token_id_capture.lineage import (
    generation_cut_lineage_records,
    generation_cut_receipts_from_lineage,
)
from nemo_gym.token_id_capture.protocols import CaptureLedger, GenerationCutCaptureLedger


MODEL_CHECKPOINT_URL_PREFIX = f"{CONTROL_URL_PREFIX}/model-checkpoint"
MODEL_LEDGER_SUBDIR = "model-ledger"
LEDGER_MANIFEST_NAME = "manifest.json"
STORAGE_REFERENCE_INDEX_NAME = "storage-references.jsonl"
LINEAGE_INDEX_NAME = "lineage-index.jsonl"
LEDGER_SCHEMA_VERSION = 3

# FileLineageStore writes one token-free custody file per rollout.
# Lock files and token-store files are not part of this participant.
_LEDGER_SUFFIX = ".lineage.jsonl"
_LEDGER_ARCHIVE_PATTERN = r"^lineage-part-[0-9]{6}\.tar$"
_LEDGER_ARCHIVE_MAX_MEMBERS = 512
_LEDGER_ARCHIVE_MAX_PAYLOAD_BYTES = 64 << 20
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class LedgerMismatchError(ControlError):
    """The checkpoint directory does not match its manifest.

    A missing manifest means the commit tore partway; a digest mismatch
    means a file changed after commit. Either way the ledger must not be
    installed: restored custody would refer to rows that do not exist as
    committed.
    """

    code = "ledger_mismatch"


class LedgerNotCheckpointableError(ControlError):
    """The configured capture backend has no checkpoint lifecycle."""

    code = "ledger_not_checkpointable"


class LedgerNotQuiescentError(ControlError):
    """The model participant still has accepted generation requests."""

    code = "ledger_not_quiescent"


class AttemptIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    attempt_index: int = Field(ge=0)


class _LineageArchiveReference(BaseModel):
    """Digest-bound coordinate for one bounded lineage tar shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=_LEDGER_ARCHIVE_PATTERN)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    members: int = Field(ge=1)
    bytes: int = Field(ge=0)


class _LineageArchiveMember(BaseModel):
    """Location and integrity metadata for one rollout lineage file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capture_key: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    archive: str = Field(pattern=_LEDGER_ARCHIVE_PATTERN)
    member: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    rows: int = Field(ge=0)
    bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_member_name(self) -> "_LineageArchiveMember":
        expected = f"{self.capture_key}{_LEDGER_SUFFIX}"
        if self.member != expected:
            raise ValueError(
                f"lineage archive member does not match its capture key: expected={expected!r}, actual={self.member!r}"
            )
        return self


class CaptureLedgerCommitResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollouts: int = Field(ge=0)
    rows: int = Field(ge=0)
    excluded_tombstoned: int = Field(ge=0)
    excluded_inactive: int = Field(default=0, ge=0)
    generation_cut_records: int = Field(default=0, ge=0)
    manifest_digest: str
    storage_reference_index: CheckpointArtifactReference


class CaptureLedgerRestoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollouts: int = Field(ge=0)
    rows: int = Field(ge=0)
    checkpoint_id: Optional[str] = None
    tombstones: list[AttemptIdentity] = Field(default_factory=list)
    source_attempts: list[AttemptIdentity] = Field(default_factory=list)
    generation_cut_receipts: tuple[GenerationCutReceipt, ...] = ()
    storage_reference_index: CheckpointArtifactReference


@runtime_checkable
class CheckpointableCaptureLedger(CaptureLedger, Protocol):
    """Optional lifecycle implemented by framework-owned capture backends.

    The backend snapshots token-free custody and its private parent-resolution
    state. The framework checkpoints staged token arrays separately.
    Gym supplies a server-specific directory and ``server_name``. Because the
    backend owns its manifest, it must record and validate that identity.
    """

    async def checkpoint_capture_ledger(
        self,
        checkpoint_dir: Path,
        *,
        checkpoint_id: str,
        server_name: str,
        tombstones: tuple[tuple[str, int], ...],
        source_attempts: tuple[tuple[str, int], ...],
        continuation_roots: tuple[AgentContinuationRoot, ...],
        generation_cut_receipts: tuple[GenerationCutReceipt, ...],
    ) -> CaptureLedgerCommitResult: ...

    async def restore_capture_ledger(
        self,
        checkpoint_dir: Path,
        *,
        server_name: str,
    ) -> CaptureLedgerRestoreResult: ...


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_fsynced(source: Path, target: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ledger-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with source.open("rb") as src:
                shutil.copyfileobj(src, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, target)


def _parse_lineage_payload(source_name: str, payload: bytes) -> list[dict[str, Any]]:
    """Parse one lineage payload while preserving its exact archived bytes."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        try:
            record = json.loads(row)
        except json.JSONDecodeError as error:
            raise LedgerMismatchError(f"invalid lineage JSON in {source_name!r} at line {line_number}") from error
        if not isinstance(record, dict):
            raise LedgerMismatchError(f"lineage row in {source_name!r} at line {line_number} is not an object")
        records.append(record)
    return records


def _write_lineage_archive(
    ledger_dir: Path,
    *,
    checkpoint_id: str,
    archive_index: int,
    members: list[tuple[str, AgentContinuationRoot | None, Path]],
) -> tuple[
    _LineageArchiveReference,
    list[_LineageArchiveMember],
    dict[str, ExternalStorageReference],
]:
    """Atomically write and fsync one deterministic lineage tar shard."""
    archive_name = f"lineage-part-{archive_index:06d}.tar"
    target = ledger_dir / archive_name
    member_references: list[_LineageArchiveMember] = []
    external_references: dict[str, ExternalStorageReference] = {}
    with tempfile.NamedTemporaryFile(dir=ledger_dir, prefix=".ledger-archive-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with tarfile.open(fileobj=handle, mode="w") as archive:
                for capture_key, root, source in members:
                    payload = source.read_bytes()
                    records = _parse_lineage_payload(source.name, payload)
                    for reference in _external_references_for_rows(
                        capture_key,
                        records,
                        root.last_committed_model_call_id if root is not None else None,
                        checkpoint_id=checkpoint_id,
                    ):
                        external_references.setdefault(reference.key, reference)

                    member_name = source.name
                    info = tarfile.TarInfo(name=member_name)
                    info.size = len(payload)
                    info.mode = 0o600
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(payload))
                    member_references.append(
                        _LineageArchiveMember(
                            capture_key=capture_key,
                            archive=archive_name,
                            member=member_name,
                            sha256=hashlib.sha256(payload).hexdigest(),
                            rows=len(records),
                            bytes=len(payload),
                        )
                    )
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    archive_size = temporary.stat().st_size
    archive_digest = _file_digest(temporary)
    os.replace(temporary, target)
    return (
        _LineageArchiveReference(
            name=archive_name,
            sha256=archive_digest,
            members=len(member_references),
            bytes=archive_size,
        ),
        member_references,
        external_references,
    )


def _partition_lineage_archives(
    members: list[tuple[str, AgentContinuationRoot | None, Path]],
) -> list[list[tuple[str, AgentContinuationRoot | None, Path]]]:
    """Partition sorted members by both file count and source payload bytes."""
    partitions: list[list[tuple[str, AgentContinuationRoot | None, Path]]] = []
    current: list[tuple[str, AgentContinuationRoot | None, Path]] = []
    current_bytes = 0
    for member in members:
        member_bytes = member[2].stat().st_size
        if current and (
            len(current) >= _LEDGER_ARCHIVE_MAX_MEMBERS
            or current_bytes + member_bytes > _LEDGER_ARCHIVE_MAX_PAYLOAD_BYTES
        ):
            partitions.append(current)
            current = []
            current_bytes = 0
        current.append(member)
        current_bytes += member_bytes
    if current:
        partitions.append(current)
    return partitions


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_continuation_roots_digest(
    continuation_roots: list[AgentContinuationRoot],
) -> str:
    payload = json.dumps(
        [
            root.model_dump(mode="json")
            for root in sorted(
                continuation_roots,
                key=lambda item: (item.capture_key, item.last_committed_model_call_id),
            )
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_generation_cuts_digest(
    receipts: tuple[GenerationCutReceipt, ...],
) -> tuple[str, int]:
    records = [
        record.model_dump(mode="json") for receipt in receipts for record in generation_cut_lineage_records(receipt)
    ]
    records.sort(key=lambda item: (item["capture_key"], item["ticket_id"]))
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), len(records)


def _normalize_continuation_roots(
    continuation_roots: list[AgentContinuationRoot],
) -> dict[str, AgentContinuationRoot]:
    by_capture_key: dict[str, AgentContinuationRoot] = {}
    for root in continuation_roots:
        existing = by_capture_key.get(root.capture_key)
        if existing is not None:
            qualifier = "conflicting" if existing != root else "duplicate"
            raise LedgerMismatchError(f"{qualifier} continuation roots for capture key {root.capture_key!r}")
        by_capture_key[root.capture_key] = root
    return by_capture_key


def _external_references_for_rows(
    capture_key: str,
    records: list[dict[str, Any]],
    boundary_model_call_id: str | None,
    *,
    checkpoint_id: str,
) -> list[ExternalStorageReference]:
    selected_records = []
    if boundary_model_call_id is not None:
        selected_records = [
            record
            for record in records
            if record.get("event") != "generation_cut"
            and record.get("model_call_id") == boundary_model_call_id
            and record.get("failure_reason") is None
        ]
    if boundary_model_call_id is not None and len(selected_records) != 1:
        raise LedgerMismatchError(
            "continuation boundary is missing or ambiguous in model lineage: "
            f"capture_key={capture_key!r}, model_call_id={boundary_model_call_id!r}"
        )

    references: list[ExternalStorageReference] = []
    seen_keys: set[str] = set()
    committed_model_call_ids = {
        record.get("model_call_id")
        for record in records
        if record.get("event") != "generation_cut"
        and record.get("failure_reason") is None
        and isinstance(record.get("staging_key"), str)
    }
    for record in selected_records:
        model_call_id = record.get("model_call_id")
        if not isinstance(model_call_id, str) or not model_call_id:
            raise LedgerMismatchError(f"lineage for {capture_key!r} contains an invalid model_call_id")
        raw_chain = record.get("staging_chain") or []
        if not isinstance(raw_chain, list):
            raise LedgerMismatchError(
                f"lineage for {capture_key!r} model call {model_call_id!r} has an invalid staging_chain"
            )
        raw_keys = [*raw_chain]
        if record.get("staging_key") is not None:
            raw_keys.append(record["staging_key"])
        for key in raw_keys:
            if not isinstance(key, str) or not key:
                raise LedgerMismatchError(
                    f"lineage for {capture_key!r} model call {model_call_id!r} has an invalid staging key"
                )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            references.append(
                ExternalStorageReference(
                    capture_key=capture_key,
                    boundary_model_call_id=boundary_model_call_id,
                    key=key,
                )
            )

    for row in records:
        if row.get("event") != "generation_cut":
            continue
        try:
            cut = GenerationCutLineageRecord.model_validate(row)
        except ValueError as error:
            raise LedgerMismatchError(
                f"lineage for {capture_key!r} contains an invalid generation-cut record"
            ) from error
        if cut.checkpoint_id != checkpoint_id or cut.disposition != "durable_prefix":
            continue
        if cut.model_call_id in committed_model_call_ids:
            continue
        for staging_key in cut.staging_keys:
            if staging_key in seen_keys:
                continue
            seen_keys.add(staging_key)
            references.append(
                ExternalStorageReference(
                    capture_key=capture_key,
                    boundary_model_call_id=cut.model_call_id,
                    kind="generation_prefix_cut",
                    key=staging_key,
                )
            )
    return references


def load_continuation_roots(
    checkpoint_root: Path,
    references: list[CheckpointArtifactReference],
) -> list[AgentContinuationRoot]:
    """Load agent-owned continuation indexes supplied to the model participant."""
    roots: list[AgentContinuationRoot] = []
    for reference in references:
        try:
            roots.extend(read_jsonl_artifact(checkpoint_root, reference, AgentContinuationRoot))
        except CheckpointArtifactError as error:
            raise LedgerMismatchError("agent continuation index is missing or corrupted") from error
    normalized = _normalize_continuation_roots(roots)
    return list(normalized.values())


def _validate_storage_reference_index(
    checkpoint_root: Path,
    manifest: dict[str, Any],
) -> CheckpointArtifactReference:
    raw_reference = manifest.get("storage_reference_index")
    if raw_reference is None:
        raise LedgerMismatchError("ledger manifest is missing its storage-reference index")
    try:
        reference = CheckpointArtifactReference.model_validate(raw_reference)
    except (CheckpointArtifactError, ValueError) as error:
        raise LedgerMismatchError("storage-reference index is missing or corrupted") from error
    _validate_storage_reference_artifact(checkpoint_root, reference)
    return reference


def _validate_storage_reference_artifact(
    checkpoint_root: Path,
    reference: CheckpointArtifactReference,
) -> None:
    try:
        records = read_jsonl_artifact(checkpoint_root, reference, ExternalStorageReference)
    except (CheckpointArtifactError, ValueError) as error:
        raise LedgerMismatchError("storage-reference index is missing or corrupted") from error
    keys = [record.key for record in records]
    if len(keys) != len(set(keys)):
        raise LedgerMismatchError("storage-reference index contains duplicate keys")


def _load_lineage_archive_index(
    checkpoint_root: Path,
    manifest: dict[str, Any],
) -> list[_LineageArchiveMember]:
    raw_reference = manifest.get("lineage_index")
    if raw_reference is None:
        raise LedgerMismatchError("ledger manifest is missing its lineage archive index")
    try:
        reference = CheckpointArtifactReference.model_validate(raw_reference)
        members = read_jsonl_artifact(checkpoint_root, reference, _LineageArchiveMember)
    except (CheckpointArtifactError, ValueError) as error:
        raise LedgerMismatchError("lineage archive index is missing or corrupted") from error
    capture_keys = [member.capture_key for member in members]
    if len(capture_keys) != len(set(capture_keys)):
        raise LedgerMismatchError("lineage archive index contains duplicate capture keys")
    member_names = [(member.archive, member.member) for member in members]
    if len(member_names) != len(set(member_names)):
        raise LedgerMismatchError("lineage archive index contains duplicate members")
    return members


def _load_lineage_archive_references(manifest: dict[str, Any]) -> list[_LineageArchiveReference]:
    raw_references = manifest.get("archives")
    if not isinstance(raw_references, list):
        raise LedgerMismatchError("ledger manifest is missing its lineage archives")
    try:
        references = [_LineageArchiveReference.model_validate(raw) for raw in raw_references]
    except ValueError as error:
        raise LedgerMismatchError("ledger manifest contains an invalid lineage archive") from error
    names = [reference.name for reference in references]
    if len(names) != len(set(names)):
        raise LedgerMismatchError("ledger manifest contains duplicate lineage archives")
    return references


def _validate_lineage_archives(
    checkpoint_root: Path,
    ledger_dir: Path,
    manifest: dict[str, Any],
) -> tuple[list[_LineageArchiveReference], list[_LineageArchiveMember]]:
    """Validate the complete v3 archive set without extracting any files."""
    archive_references, members, members_by_archive = _validate_lineage_archive_inventory(
        checkpoint_root,
        manifest,
    )

    for reference in archive_references:
        path = ledger_dir / reference.name
        if not path.is_file():
            raise LedgerMismatchError(f"lineage archive {reference.name!r} is missing")
        if path.stat().st_size != reference.bytes or _file_digest(path) != reference.sha256:
            raise LedgerMismatchError(f"lineage archive {reference.name!r} is corrupted")
        expected = {member.member: member for member in members_by_archive[reference.name]}
        if reference.members != len(expected):
            raise LedgerMismatchError(f"lineage archive {reference.name!r} member count is corrupted")
        try:
            with tarfile.open(path, mode="r:") as archive:
                infos = archive.getmembers()
                names = [info.name for info in infos]
                if len(names) != len(set(names)) or set(names) != set(expected):
                    raise LedgerMismatchError(f"lineage archive {reference.name!r} has an unexpected member inventory")
                for info in infos:
                    member = expected[info.name]
                    if not info.isfile() or info.size != member.bytes:
                        raise LedgerMismatchError(
                            f"lineage archive member {reference.name!r}/{info.name!r} is invalid"
                        )
                    extracted = archive.extractfile(info)
                    if extracted is None:
                        raise LedgerMismatchError(
                            f"lineage archive member {reference.name!r}/{info.name!r} cannot be read"
                        )
                    payload = extracted.read()
                    if hashlib.sha256(payload).hexdigest() != member.sha256:
                        raise LedgerMismatchError(
                            f"lineage archive member {reference.name!r}/{info.name!r} is corrupted"
                        )
                    records = _parse_lineage_payload(info.name, payload)
                    if len(records) != member.rows:
                        raise LedgerMismatchError(
                            f"lineage archive member {reference.name!r}/{info.name!r} row count is corrupted"
                        )
        except (OSError, tarfile.TarError) as error:
            raise LedgerMismatchError(f"lineage archive {reference.name!r} cannot be read") from error
    return archive_references, members


def _validate_lineage_archive_inventory(
    checkpoint_root: Path,
    manifest: dict[str, Any],
) -> tuple[
    list[_LineageArchiveReference],
    list[_LineageArchiveMember],
    dict[str, list[_LineageArchiveMember]],
]:
    """Validate the authenticated v3 archive inventory without reading tar payloads."""
    archive_references = _load_lineage_archive_references(manifest)
    members = _load_lineage_archive_index(checkpoint_root, manifest)
    members_by_archive: dict[str, list[_LineageArchiveMember]] = {}
    for member in members:
        members_by_archive.setdefault(member.archive, []).append(member)
    archive_names = {reference.name for reference in archive_references}
    referenced_names = set(members_by_archive)
    if archive_names != referenced_names:
        raise LedgerMismatchError(
            "lineage archive inventory does not match its index: "
            f"missing={sorted(referenced_names - archive_names)!r}, "
            f"unreferenced={sorted(archive_names - referenced_names)!r}"
        )
    if int(manifest.get("rollout_count", -1)) != len(members):
        raise LedgerMismatchError("lineage archive rollout count does not match its index")
    total_rows = sum(member.rows for member in members)
    if int(manifest.get("row_count", -1)) != total_rows:
        raise LedgerMismatchError("lineage archive row count does not match its index")
    return archive_references, members, members_by_archive


def _checkpoint_lineage_union(checkpoint_root: Path, checkpoint_id: object) -> dict[str, tuple[str, int, int]]:
    """Return the deduplicated lineage inventory owned by all model participants."""
    model_ledger_root = checkpoint_root / MODEL_LEDGER_SUBDIR
    participant_dirs = sorted(path for path in model_ledger_root.iterdir() if path.is_dir())
    expected: dict[str, tuple[str, int, int]] = {}
    for participant_dir in participant_dirs:
        manifest_path = participant_dir / LEDGER_MANIFEST_NAME
        if not manifest_path.is_file():
            raise LedgerMismatchError(f"model-ledger participant {participant_dir.name!r} is missing its manifest")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise LedgerMismatchError(
                f"model-ledger participant {participant_dir.name!r} has an invalid manifest"
            ) from error
        schema_version = manifest.get("schema_version", 0)
        if not isinstance(schema_version, int) or schema_version > LEDGER_SCHEMA_VERSION:
            raise LedgerMismatchError(
                f"model-ledger participant {participant_dir.name!r} has an unsupported schema version"
            )
        if manifest.get("server_name") != participant_dir.name:
            raise LedgerMismatchError(
                f"model-ledger participant directory {participant_dir.name!r} does not match its manifest"
            )
        if manifest.get("checkpoint_id") != checkpoint_id:
            raise LedgerMismatchError(
                f"model-ledger participant {participant_dir.name!r} belongs to a different checkpoint transaction"
            )
        _validate_storage_reference_index(checkpoint_root, manifest)

        if schema_version >= 3:
            _, members, _ = _validate_lineage_archive_inventory(checkpoint_root, manifest)
            inventory = {member.member: (member.sha256, member.bytes, member.rows) for member in members}
        else:
            inventory = {}
            for metadata in manifest["rollouts"].values():
                for name, digest in metadata["files"].items():
                    inventory[name] = (digest, int(metadata.get("bytes", -1)), int(metadata.get("rows", -1)))

        for name, identity in inventory.items():
            previous = expected.get(name)
            if previous is not None and previous != identity:
                raise LedgerMismatchError(
                    f"model-ledger participants contain conflicting lineage for the same rollout: member={name!r}"
                )
            expected[name] = identity
    return expected


def _write_payload_fsynced(payload: bytes, target: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ledger-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, target)


class CaptureLedgerCheckpointer:
    """Commit and restore one token-capture store directory."""

    def __init__(self, store_root: Path, *, server_name: Optional[str] = None) -> None:
        self.store_root = Path(store_root)
        self.server_name = _validate_server_name(server_name) if server_name is not None else None

    def _ledger_dir(self, checkpoint_dir: Path) -> Path:
        directory = Path(checkpoint_dir) / MODEL_LEDGER_SUBDIR
        return directory / self.server_name if self.server_name is not None else directory

    def commit(
        self,
        checkpoint_dir: Path,
        *,
        checkpoint_id: str,
        tombstones: list[tuple[str, int]],
        source_attempts: Optional[list[tuple[str, int]]] = None,
        continuation_roots: list[AgentContinuationRoot],
        generation_cut_receipts: tuple[GenerationCutReceipt, ...] = (),
    ) -> dict[str, Any]:
        """Archive the ledger into ``checkpoint_dir``; the caller has drained.

        The store must be quiescent (admission paused) when this runs: the
        copy takes no locks because nothing may be writing.
        """
        checkpoint_dir = Path(checkpoint_dir)
        ledger_dir = self._ledger_dir(checkpoint_dir)
        normalized_roots = _normalize_continuation_roots(continuation_roots)
        roots_digest = _canonical_continuation_roots_digest(continuation_roots)
        for receipt in generation_cut_receipts:
            if receipt.checkpoint_id != checkpoint_id:
                raise LedgerMismatchError("generation-cut receipt belongs to a different checkpoint")
            if self.server_name is None or receipt.inventory.server_name != self.server_name:
                raise LedgerMismatchError("generation-cut receipt belongs to a different model server")
        cuts_digest, generation_cut_records = _canonical_generation_cuts_digest(generation_cut_receipts)
        if (ledger_dir / LEDGER_MANIFEST_NAME).exists():
            result = self._validate_committed(
                ledger_dir,
                checkpoint_root=checkpoint_dir,
                checkpoint_id=checkpoint_id,
                server_name=self.server_name,
                tombstones=tombstones,
                source_attempts=source_attempts or [],
                continuation_roots_digest=roots_digest,
                generation_cuts_digest=cuts_digest,
                generation_cut_records=generation_cut_records,
            )
            # A previous attempt may have renamed the manifest and then
            # failed its final directory fsync. Retry that durability barrier.
            _fsync_dir(ledger_dir)
            return result
        fenced = {capture_key_for(rollout_id, attempt_index) for rollout_id, attempt_index in tombstones}
        fenced_roots = sorted(set(normalized_roots) & fenced)
        if fenced_roots:
            raise LedgerMismatchError(
                f"continuation roots refer to retired model attempts: capture_keys={fenced_roots!r}"
            )
        cut_capture_keys = {
            capture_key_for(prefix.rollout_id, prefix.attempt_index)
            for receipt in generation_cut_receipts
            for prefix in receipt.prefixes
        } - fenced
        archived_capture_keys = set(normalized_roots) | cut_capture_keys
        sources = {
            capture_key: self.store_root / f"{capture_key}{_LEDGER_SUFFIX}" for capture_key in archived_capture_keys
        }
        missing_roots = sorted(capture_key for capture_key, source in sources.items() if not source.is_file())
        if missing_roots:
            raise LedgerMismatchError(f"continuation roots have no model lineage: capture_keys={missing_roots!r}")

        ledger_dir.mkdir(parents=True, exist_ok=True)

        excluded = len(tombstones)
        source_capture_keys = {
            capture_key_for(rollout_id, attempt_index) for rollout_id, attempt_index in source_attempts or []
        }
        excluded_inactive = len(source_capture_keys - archived_capture_keys - fenced)
        archive_references: list[_LineageArchiveReference] = []
        lineage_members: list[_LineageArchiveMember] = []
        external_references: dict[str, ExternalStorageReference] = {}
        ordered_sources = [
            (capture_key, normalized_roots.get(capture_key), sources[capture_key])
            for capture_key in sorted(archived_capture_keys)
        ]
        for archive_index, archive_sources in enumerate(_partition_lineage_archives(ordered_sources)):
            archive_reference, archive_members, archive_external_references = _write_lineage_archive(
                ledger_dir,
                checkpoint_id=checkpoint_id,
                archive_index=archive_index,
                members=archive_sources,
            )
            archive_references.append(archive_reference)
            lineage_members.extend(archive_members)
            for reference in archive_external_references.values():
                external_references.setdefault(reference.key, reference)
        lineage_index = write_jsonl_artifact(
            checkpoint_dir,
            ledger_dir.relative_to(checkpoint_dir) / LINEAGE_INDEX_NAME,
            lineage_members,
        )
        storage_reference_index = write_jsonl_artifact(
            checkpoint_dir,
            ledger_dir.relative_to(checkpoint_dir) / STORAGE_REFERENCE_INDEX_NAME,
            (external_references[key] for key in sorted(external_references)),
        )
        _fsync_dir(ledger_dir)

        manifest = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "server_name": self.server_name,
            "archives": [reference.model_dump(mode="json") for reference in archive_references],
            "lineage_index": lineage_index.model_dump(mode="json"),
            "rollout_count": len(lineage_members),
            "row_count": sum(member.rows for member in lineage_members),
            "continuation_roots_sha256": roots_digest,
            "continuation_roots": len(normalized_roots),
            "generation_cuts_sha256": cuts_digest,
            "generation_cut_records": generation_cut_records,
            "excluded_inactive": excluded_inactive,
            "storage_reference_index": storage_reference_index.model_dump(mode="json"),
            "tombstones": [
                {"rollout_id": rollout_id, "attempt_index": attempt} for rollout_id, attempt in sorted(tombstones)
            ],
            "source_attempts": [
                {"rollout_id": rollout_id, "attempt_index": attempt}
                for rollout_id, attempt in sorted(source_attempts or [])
            ],
        }
        payload = json.dumps(manifest, sort_keys=True, indent=1).encode()
        with tempfile.NamedTemporaryFile(dir=ledger_dir, prefix=".manifest-", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, ledger_dir / LEDGER_MANIFEST_NAME)
        _fsync_dir(ledger_dir)

        return {
            "rollouts": len(lineage_members),
            "rows": sum(member.rows for member in lineage_members),
            "excluded_tombstoned": excluded,
            "excluded_inactive": excluded_inactive,
            "generation_cut_records": generation_cut_records,
            "manifest_digest": hashlib.sha256(payload).hexdigest(),
            "storage_reference_index": storage_reference_index.model_dump(mode="json"),
        }

    @staticmethod
    def _validate_committed(
        ledger_dir: Path,
        *,
        checkpoint_root: Path,
        checkpoint_id: str,
        server_name: Optional[str],
        tombstones: list[tuple[str, int]],
        source_attempts: list[tuple[str, int]],
        continuation_roots_digest: str,
        generation_cuts_digest: str,
        generation_cut_records: int,
    ) -> dict[str, Any]:
        manifest_path = ledger_dir / LEDGER_MANIFEST_NAME
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
        if manifest.get("checkpoint_id") != checkpoint_id or manifest.get("server_name") != server_name:
            raise LedgerMismatchError("ledger directory belongs to a different checkpoint transaction or model server")
        expected_tombstones = [
            {"rollout_id": rollout_id, "attempt_index": attempt} for rollout_id, attempt in sorted(tombstones)
        ]
        expected_source_attempts = [
            {"rollout_id": rollout_id, "attempt_index": attempt} for rollout_id, attempt in sorted(source_attempts)
        ]
        if manifest.get("tombstones", []) != expected_tombstones:
            raise LedgerMismatchError("committed ledger abort exclusions changed before commit retry")
        if manifest.get("source_attempts", []) != expected_source_attempts:
            raise LedgerMismatchError("committed ledger source attempts changed before commit retry")
        if manifest.get("continuation_roots_sha256") != continuation_roots_digest:
            raise LedgerMismatchError("committed ledger continuation roots changed before commit retry")
        if manifest.get("generation_cuts_sha256") != generation_cuts_digest:
            raise LedgerMismatchError("committed ledger generation cuts changed before commit retry")
        if int(manifest.get("generation_cut_records", -1)) != generation_cut_records:
            raise LedgerMismatchError("committed ledger generation-cut count changed before commit retry")
        storage_reference_index = _validate_storage_reference_index(checkpoint_root, manifest)
        schema_version = manifest.get("schema_version", 0)
        if not isinstance(schema_version, int) or schema_version > LEDGER_SCHEMA_VERSION:
            raise LedgerMismatchError("committed ledger has an unsupported schema version")
        if schema_version >= 3:
            _, members = _validate_lineage_archives(checkpoint_root, ledger_dir, manifest)
            rollout_count = len(members)
            total_rows = sum(member.rows for member in members)
        else:
            total_rows = 0
            rollouts = manifest.get("rollouts", {})
            for rollout_id, metadata in rollouts.items():
                for name, digest in metadata.get("files", {}).items():
                    path = ledger_dir / name
                    if not path.exists() or _file_digest(path) != digest:
                        raise LedgerMismatchError(f"committed ledger file {name!r} for {rollout_id!r} is corrupted")
                total_rows += int(metadata.get("rows", 0))
            rollout_count = len(rollouts)
        return {
            "rollouts": rollout_count,
            "rows": total_rows,
            "excluded_tombstoned": len(manifest.get("tombstones", [])),
            "excluded_inactive": int(manifest.get("excluded_inactive", 0)),
            "generation_cut_records": int(manifest.get("generation_cut_records", 0)),
            "manifest_digest": hashlib.sha256(payload).hexdigest(),
            "storage_reference_index": storage_reference_index.model_dump(mode="json"),
        }

    def restore(self, checkpoint_dir: Path) -> dict[str, Any]:
        """Install a committed ledger into this store root and verify it."""
        checkpoint_dir = Path(checkpoint_dir)
        ledger_dir = self._ledger_dir(checkpoint_dir)
        manifest_path = ledger_dir / LEDGER_MANIFEST_NAME
        if not manifest_path.exists():
            raise LedgerMismatchError(
                f"no ledger manifest at {manifest_path}; the commit tore partway or never ran, "
                f"so this directory must not be installed"
            )
        manifest = json.loads(manifest_path.read_text())
        schema_version = manifest.get("schema_version", 0)
        if not isinstance(schema_version, int) or schema_version > LEDGER_SCHEMA_VERSION:
            raise LedgerMismatchError(
                f"ledger manifest schema_version {manifest.get('schema_version')} is newer than this "
                f"reader ({LEDGER_SCHEMA_VERSION})"
            )
        if manifest.get("server_name") != self.server_name:
            raise LedgerMismatchError("ledger checkpoint belongs to a different model server")
        storage_reference_index = _validate_storage_reference_index(checkpoint_dir, manifest)

        if schema_version >= 3:
            archive_references, archive_members = _validate_lineage_archives(checkpoint_dir, ledger_dir, manifest)
        else:
            archive_references = []
            archive_members = []
        if self.server_name is None:
            expected = {member.member: (member.sha256, member.bytes, member.rows) for member in archive_members}
            if schema_version < 3:
                expected = {
                    name: (digest, int(metadata.get("bytes", -1)), int(metadata.get("rows", -1)))
                    for metadata in manifest["rollouts"].values()
                    for name, digest in metadata["files"].items()
                }
        else:
            expected = _checkpoint_lineage_union(checkpoint_dir, manifest.get("checkpoint_id"))
        existing = {path.name: path for path in self.store_root.glob(f"*{_LEDGER_SUFFIX}")}
        unexpected = set(existing) - set(expected)
        if unexpected:
            raise LedgerMismatchError(
                "restore requires a checkpoint-owned capture-ledger namespace; "
                f"found files absent from the checkpoint union: {sorted(unexpected)}"
            )
        corrupted = sorted(name for name, path in existing.items() if _file_digest(path) != expected[name][0])
        if corrupted:
            raise LedgerMismatchError(
                f"live capture-ledger files do not match the checkpoint union: members={corrupted}"
            )

        # Validate the complete source before changing the live namespace.
        validated: list[tuple[Path, str]] = []
        if schema_version < 3:
            for rollout_id, meta in manifest["rollouts"].items():
                for name, digest in meta["files"].items():
                    source = ledger_dir / name
                    if not source.exists() or _file_digest(source) != digest:
                        raise LedgerMismatchError(
                            f"ledger file {name} for rollout {rollout_id!r} is missing or does not match "
                            f"its committed digest; refusing to install a corrupted ledger"
                        )
                    validated.append((source, name))
        self.store_root.mkdir(parents=True, exist_ok=True)
        if schema_version >= 3:
            members_by_archive: dict[str, list[_LineageArchiveMember]] = {}
            for member in archive_members:
                members_by_archive.setdefault(member.archive, []).append(member)
            for reference in archive_references:
                with tarfile.open(ledger_dir / reference.name, mode="r:") as archive:
                    for member in members_by_archive[reference.name]:
                        extracted = archive.extractfile(member.member)
                        if extracted is None:  # Already validated; guard against an in-place source mutation.
                            raise LedgerMismatchError(
                                f"lineage archive member {reference.name!r}/{member.member!r} disappeared"
                            )
                        _write_payload_fsynced(extracted.read(), self.store_root / member.member)
            rollout_count = len(archive_members)
            total_rows = sum(member.rows for member in archive_members)
        else:
            for source, name in validated:
                _copy_fsynced(source, self.store_root / name)
            rollout_count = len(manifest["rollouts"])
            total_rows = sum(int(meta.get("rows", 0)) for meta in manifest["rollouts"].values())
        _fsync_dir(self.store_root)

        restored_cut_receipts: tuple[GenerationCutReceipt, ...] = ()
        if self.server_name is not None and isinstance(manifest.get("checkpoint_id"), str):
            rows_by_capture_key = {
                member.capture_key: _parse_lineage_payload(
                    member.member,
                    (self.store_root / member.member).read_bytes(),
                )
                for member in archive_members
            }
            restored_cut_receipts = generation_cut_receipts_from_lineage(
                rows_by_capture_key,
                checkpoint_id=manifest["checkpoint_id"],
                server_name=self.server_name,
            )

        result: dict[str, Any] = {
            "rollouts": rollout_count,
            "rows": total_rows,
            "checkpoint_id": manifest.get("checkpoint_id"),
            "tombstones": list(manifest.get("tombstones", ())),
            "source_attempts": list(manifest.get("source_attempts", ())),
            "generation_cut_receipts": [receipt.model_dump(mode="json") for receipt in restored_cut_receipts],
        }
        result["storage_reference_index"] = storage_reference_index.model_dump(mode="json")
        return result


class ModelCheckpointCommitRequest(CheckpointControlRequest):
    checkpoint_dir: str
    continuation_indexes: list[CheckpointArtifactReference]


class ModelCheckpointRestoreRequest(CheckpointControlRequest):
    checkpoint_dir: str
    generation_cut_receipts: tuple[GenerationCutReceipt, ...] = ()
    generation_cut_exclusions: tuple[GenerationCutReplacement, ...] = ()


def _validate_server_name(server_name: str) -> str:
    if ROLLOUT_ID_PATTERN.fullmatch(server_name) is None:
        raise ValueError(
            "model server name must contain only letters, digits, dots, dashes, or underscores "
            "and start with a letter or digit"
        )
    return server_name


async def _commit_model_ledger(
    checkpoint_dir: Path,
    *,
    checkpoint_id: str,
    server_name: str,
    ledger: Optional[CaptureLedger],
    file_ledger_root: Optional[Path],
    tombstones: set[tuple[str, int]],
    source_attempts: set[tuple[str, int]],
    continuation_roots: list[AgentContinuationRoot],
    generation_cut_receipts: tuple[GenerationCutReceipt, ...],
) -> dict[str, Any]:
    if generation_cut_receipts:
        if not isinstance(ledger, GenerationCutCaptureLedger):
            raise LedgerNotCheckpointableError(
                "generation-prefix cuts require a capture ledger that can record cut coordinates"
            )
        for receipt in generation_cut_receipts:
            await ledger.record_generation_cut(receipt)
    expected_cut_records = sum(len(receipt.prefixes) for receipt in generation_cut_receipts)
    if isinstance(ledger, CheckpointableCaptureLedger):
        participant_dir = checkpoint_dir / MODEL_LEDGER_SUBDIR / server_name
        commit_result = await ledger.checkpoint_capture_ledger(
            participant_dir,
            checkpoint_id=checkpoint_id,
            server_name=server_name,
            tombstones=tuple(tombstones),
            source_attempts=tuple(source_attempts),
            continuation_roots=tuple(continuation_roots),
            generation_cut_receipts=generation_cut_receipts,
        )
        validated = CaptureLedgerCommitResult.model_validate(commit_result)
        if validated.generation_cut_records != expected_cut_records:
            raise LedgerMismatchError("capture-ledger checkpoint did not commit the complete generation-cut inventory")
        _validate_storage_reference_artifact(
            checkpoint_dir,
            validated.storage_reference_index,
        )
        return validated.model_dump(mode="json")

    if file_ledger_root is None:
        raise LedgerNotCheckpointableError(
            "the configured CaptureLedger must implement CheckpointableCaptureLedger; "
            "Gym cannot infer how to snapshot a framework-owned backend"
        )
    checkpointer = CaptureLedgerCheckpointer(file_ledger_root, server_name=server_name)
    return await _run_sync(
        lambda: checkpointer.commit(
            checkpoint_dir,
            checkpoint_id=checkpoint_id,
            tombstones=sorted(tombstones),
            source_attempts=sorted(source_attempts),
            continuation_roots=continuation_roots,
            generation_cut_receipts=generation_cut_receipts,
        )
    )


async def _restore_model_ledger(
    checkpoint_dir: Path,
    *,
    server_name: str,
    ledger: Optional[CaptureLedger],
    file_ledger_root: Optional[Path],
) -> dict[str, Any]:
    if isinstance(ledger, CheckpointableCaptureLedger):
        participant_dir = checkpoint_dir / MODEL_LEDGER_SUBDIR / server_name
        restore_result = await ledger.restore_capture_ledger(
            participant_dir,
            server_name=server_name,
        )
        validated = CaptureLedgerRestoreResult.model_validate(restore_result)
        _validate_storage_reference_artifact(
            checkpoint_dir,
            validated.storage_reference_index,
        )
        return validated.model_dump(mode="json")

    if file_ledger_root is None:
        raise LedgerNotCheckpointableError(
            "the configured CaptureLedger must implement CheckpointableCaptureLedger; "
            "Gym cannot infer how to restore a framework-owned backend"
        )
    checkpointer = CaptureLedgerCheckpointer(file_ledger_root, server_name=server_name)
    return await _run_sync(lambda: checkpointer.restore(checkpoint_dir))


class PolicyModelCheckpointCoordinatorService:
    """Own policy-model checkpoint state shared by every Uvicorn worker."""

    def __init__(
        self,
        coordinator: AdmissionCoordinator,
        *,
        ledger_provider: Callable[[], Optional[CaptureLedger]],
        file_ledger_root_provider: Callable[[], Optional[Path]],
        instance_role: Literal["policy", "auxiliary"],
        server_name: str,
        supports_generation_cuts: bool,
    ) -> None:
        self.coordinator = coordinator
        self.ledger_provider = ledger_provider
        self.file_ledger_root_provider = file_ledger_root_provider
        self.instance_role = instance_role
        self.server_name = _validate_server_name(server_name)
        self.supports_generation_cuts = supports_generation_cuts
        self.fence = ControlFence()

    async def __call__(
        self,
        worker_id: str,
        operation: str,
        payload: dict[str, Any],
    ) -> Any:
        self._require_policy()
        if operation == "model_admission_pause":
            return await self._pause(ModelAdmissionPauseRequest.model_validate(payload))
        if operation == "model_admission_status":
            return await self._status(payload)
        if operation == "model_admission_resume":
            return await self._resume(ModelAdmissionResumeRequest.model_validate(payload))
        if operation == "model_admission_abort_inflight":
            return await self._abort(ModelAbortInflightRequest.model_validate(payload))
        if operation == "model_checkpoint_commit":
            return await self._commit(ModelCheckpointCommitRequest.model_validate(payload))
        if operation == "model_checkpoint_restore":
            return await self._restore(ModelCheckpointRestoreRequest.model_validate(payload))
        if operation == "claim_generation_cut":
            key = (str(payload["rollout_id"]), int(payload["attempt_index"]))
            return self.coordinator.restored_cuts.claim(
                key,
                worker_id=worker_id,
                model_call_id=str(payload["model_call_id"]),
            )
        if operation == "abandon_generation_cut_claim":
            key = (str(payload["rollout_id"]), int(payload["attempt_index"]))
            return self.coordinator.restored_cuts.release_if_owned(
                key,
                worker_id=worker_id,
                model_call_id=str(payload["model_call_id"]),
            )
        if operation == "consume_generation_cut":
            key = (str(payload["rollout_id"]), int(payload["attempt_index"]))
            self.coordinator.restored_cuts.consume(
                key,
                worker_id=worker_id,
                model_call_id=str(payload["model_call_id"]),
            )
            if not self.coordinator.restored_cuts.has_unconsumed():
                await self.coordinator.publish_restored_cut_state()
            return None
        if operation == "release_generation_cut":
            key = (str(payload["rollout_id"]), int(payload["attempt_index"]))
            self.coordinator.restored_cuts.release(
                key,
                worker_id=worker_id,
                model_call_id=str(payload["model_call_id"]),
            )
            return None
        raise ValueError(f"unknown policy checkpoint coordinator operation {operation!r}")

    def _require_policy(self) -> None:
        if self.instance_role != "policy":
            raise NotPolicyInstanceError(
                "this model-server instance is auxiliary; only policy instances coordinate checkpoints"
            )

    def _all_workers_acked(self, status: dict[str, Any]) -> bool:
        workers = status["workers"]
        return status["missing_workers"] == 0 and workers["acknowledged"] == workers["live"]

    async def _await_worker_acks(self, timeout_s: float) -> dict[str, Any]:
        status = await self.coordinator.wait_until(
            self._all_workers_acked,
            timeout_s=max(timeout_s, 0.001),
        )
        if not self._all_workers_acked(status):
            workers = status["workers"]
            raise MissingWorkersError(
                f"{status['missing_workers']} of {self.coordinator.expected_workers} workers missing and "
                f"{workers['acknowledged']}/{workers['live']} live workers acknowledged"
            )
        return status

    def _status_payload(self, status: dict[str, Any]) -> dict[str, Any]:
        result = {
            "state": status["state"],
            "workers": {
                "acknowledged": status["workers"]["acknowledged"],
                "expected": self.coordinator.expected_workers,
            },
            "missing_workers": status["missing_workers"],
            "inflight_total": status["inflight_total"],
            "response_inflight_total": status["response_inflight_total"],
            "generation_pending_total": status["generation_pending_total"],
            "waiters_total": status["waiters_total"],
        }
        if status["state"] == AdmissionState.PAUSED.value:
            result["generation_cut_proof"] = self.coordinator.generation_cut_proof().model_dump(mode="json")
        return result

    async def _pause(self, body: ModelAdmissionPauseRequest) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            await self.coordinator.close_admission(
                body.checkpoint_id,
                cut_timeout_s=body.remaining(),
            )
            try:
                status = await self._await_worker_acks(body.remaining())
            except BaseException:
                await self.coordinator.resume_admission()
                raise
            return self._status_payload(status)

        result = await self.fence.run_operation(
            body.checkpoint_id,
            "model-admission/pause",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.PREPARING,
            phase_after=CheckpointPhase.PREPARING,
            run=run,
            deadline=body,
        )
        if result["state"] == AdmissionState.PAUSED.value:
            self.fence.mark_prepared(body.checkpoint_id)
        return result

    async def _status(self, payload: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id = str(payload["checkpoint_id"])
        self.fence.require_phase(
            checkpoint_id,
            frozenset(
                {
                    CheckpointPhase.PREPARING,
                    CheckpointPhase.PREPARED,
                    CheckpointPhase.COMMITTING,
                    CheckpointPhase.COMMITTED_PAUSED,
                    CheckpointPhase.RESTORING,
                    CheckpointPhase.RESTORE_FAILED_PAUSED,
                    CheckpointPhase.RESTORED_PAUSED,
                }
            ),
        )
        wait_state = payload.get("wait_state")
        timeout_s = float(payload.get("timeout_s", 0.0))
        if wait_state == "paused" and timeout_s > 0:
            status = await self.coordinator.wait_until(
                lambda value: value["state"] == AdmissionState.PAUSED.value,
                timeout_s=timeout_s,
            )
        else:
            status = self.coordinator.status()
        if status["state"] == AdmissionState.PAUSED.value and self.fence.phase == CheckpointPhase.PREPARING:
            self.fence.mark_prepared(checkpoint_id)
        return {"checkpoint_id": checkpoint_id, **self._status_payload(status)}

    async def _resume(self, body: ModelAdmissionResumeRequest) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            await self.coordinator.resume_admission()
            status = await self._await_worker_acks(10.0)
            return {
                "state": status["state"],
                "workers": {
                    "acknowledged": status["workers"]["acknowledged"],
                    "expected": self.coordinator.expected_workers,
                },
                "released_waiters": 0,
            }

        return await self.fence.run_operation(
            body.checkpoint_id,
            "model-admission/resume",
            allowed_phases=frozenset(
                {
                    CheckpointPhase.IDLE,
                    CheckpointPhase.PREPARING,
                    CheckpointPhase.PREPARED,
                    CheckpointPhase.COMMITTED_PAUSED,
                    CheckpointPhase.RESTORE_FAILED_PAUSED,
                    CheckpointPhase.RESTORED_PAUSED,
                }
            ),
            phase_during=self.fence.phase,
            phase_after=CheckpointPhase.IDLE,
            run=run,
            retire_outcome="resumed",
        )

    async def _abort(self, body: ModelAbortInflightRequest) -> dict[str, Any]:
        entry_phase = self.fence.phase

        async def run() -> dict[str, Any]:
            await self.coordinator.add_tombstone(body.rollout_id, body.attempt_index)
            status = await self._await_worker_acks(10.0)
            return {
                "state": status["state"],
                "aborted_inflight": 0,
                "inflight_total": status["inflight_total"],
            }

        return await self.fence.run_operation(
            body.checkpoint_id,
            f"model-admission/abort_inflight:{body.rollout_id}:{body.attempt_index}",
            allowed_phases=frozenset({CheckpointPhase.PREPARING, CheckpointPhase.PREPARED}),
            phase_during=entry_phase,
            phase_after=entry_phase,
            run=run,
        )

    def _generation_cut_receipts(self) -> tuple[GenerationCutReceipt, ...]:
        proof = self.coordinator.generation_cut_proof()
        return tuple(
            worker.generation_cut_receipt for worker in proof.workers if worker.generation_cut_receipt is not None
        )

    async def _commit(self, body: ModelCheckpointCommitRequest) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            status = self.coordinator.status()
            if status["state"] != AdmissionState.PAUSED.value:
                raise LedgerNotQuiescentError("capture-ledger commit requires every policy worker to be paused")
            continuation_roots = await asyncio.to_thread(
                load_continuation_roots,
                Path(body.checkpoint_dir),
                body.continuation_indexes,
            )
            return await _commit_model_ledger(
                Path(body.checkpoint_dir),
                checkpoint_id=body.checkpoint_id,
                server_name=self.server_name,
                ledger=self.ledger_provider(),
                file_ledger_root=self.file_ledger_root_provider(),
                tombstones=self.coordinator.checkpoint_exclusions(),
                source_attempts=self.coordinator.seen_attempts(),
                continuation_roots=continuation_roots,
                generation_cut_receipts=self._generation_cut_receipts(),
            )

        return await self.fence.run_operation(
            body.checkpoint_id,
            "model-checkpoint/commit",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.COMMITTING,
            phase_after=CheckpointPhase.COMMITTED_PAUSED,
            run=run,
        )

    async def _restore(self, body: ModelCheckpointRestoreRequest) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            await self.coordinator.close_admission(
                body.checkpoint_id,
                cut_timeout_s=body.remaining(),
            )
            await self._await_worker_acks(body.remaining())
            result = await _restore_model_ledger(
                Path(body.checkpoint_dir),
                server_name=self.server_name,
                ledger=self.ledger_provider(),
                file_ledger_root=self.file_ledger_root_provider(),
            )
            lineage_receipts = tuple(
                GenerationCutReceipt.model_validate(receipt) for receipt in result.pop("generation_cut_receipts", ())
            )
            generation_cut_receipts = _validate_restored_generation_cut_receipts(
                lineage_receipts=lineage_receipts,
                requested_receipts=body.generation_cut_receipts,
                server_name=self.server_name,
            )
            if generation_cut_receipts and not self.supports_generation_cuts:
                raise LedgerNotCheckpointableError(
                    "checkpoint contains generation cuts but this model server has no restore backend"
                )
            exclusions = {(item.rollout_id, item.attempt_index) for item in body.generation_cut_exclusions}
            restored_entries: dict[tuple[str, int], dict[str, Any]] = {}
            for receipt in generation_cut_receipts:
                for prefix in receipt.prefixes:
                    replacement = (prefix.rollout_id, prefix.attempt_index + 1)
                    if (
                        prefix.disposition == "durable_prefix"
                        and prefix.cut_kind == "active_prefix"
                        and replacement not in exclusions
                    ):
                        value = prefix.model_dump(mode="json")
                        previous = restored_entries.get(replacement)
                        if previous is not None and previous != value:
                            raise LedgerMismatchError(
                                "multiple durable generation cuts target the same "
                                f"replacement attempt: {replacement!r}"
                            )
                        restored_entries[replacement] = value
            self.coordinator.restored_cuts.install(restored_entries)
            restored_tombstones = {
                (str(item["rollout_id"]), int(item["attempt_index"]))
                for item in (*result["tombstones"], *result.get("source_attempts", []))
            }
            await self.coordinator.install_tombstones(restored_tombstones)
            await self._await_worker_acks(10.0)
            result["generation_cuts_restored"] = len(restored_entries)
            return result

        return await self.fence.run_operation(
            body.checkpoint_id,
            "model-checkpoint/restore",
            allowed_phases=frozenset({CheckpointPhase.IDLE, CheckpointPhase.RESTORE_FAILED_PAUSED}),
            phase_during=CheckpointPhase.RESTORING,
            phase_after=CheckpointPhase.RESTORED_PAUSED,
            run=run,
            phase_on_failure=CheckpointPhase.RESTORE_FAILED_PAUSED,
        )


def _validate_restored_generation_cut_receipts(
    *,
    lineage_receipts: tuple[GenerationCutReceipt, ...],
    requested_receipts: tuple[GenerationCutReceipt, ...],
    server_name: str,
) -> tuple[GenerationCutReceipt, ...]:
    if lineage_receipts and requested_receipts:
        lineage_prefixes = sorted(
            json.dumps(prefix.model_dump(mode="json"), sort_keys=True)
            for receipt in lineage_receipts
            for prefix in receipt.prefixes
        )
        request_prefixes = sorted(
            json.dumps(prefix.model_dump(mode="json"), sort_keys=True)
            for receipt in requested_receipts
            for prefix in receipt.prefixes
        )
        if lineage_prefixes != request_prefixes:
            raise LedgerMismatchError("request-carried generation cuts disagree with the restored model lineage")
    receipts = lineage_receipts or requested_receipts
    for receipt in receipts:
        if receipt.inventory.server_name != server_name:
            raise LedgerMismatchError(
                "generation-cut receipt belongs to a different model server: "
                f"expected={server_name!r}, actual={receipt.inventory.server_name!r}"
            )
    return receipts


def install_model_checkpoint(
    app: FastAPI,
    *,
    fence: ControlFence,
    limiter: AdmissionLimiter,
    ledger_provider: Callable[[], Optional[CaptureLedger]],
    file_ledger_root_provider: Callable[[], Optional[Path]],
    instance_role: Literal["policy", "auxiliary"],
    server_name: str,
    auth_token: str,
    coordinator_client: Optional[CoordinatorServiceClient] = None,
) -> None:
    """Register ``/ng-control/v1/model-checkpoint`` on a model-server app.

    Commit requires the prepared (drained) phase; restore runs on a freshly
    started server and leaves it paused, so nothing serves until the
    coordinator has restored every component and explicitly resumes.
    """

    server_name = _validate_server_name(server_name)

    def _require_policy() -> None:
        if instance_role != "policy":
            raise NotPolicyInstanceError(
                "this model-server instance is auxiliary (judge or simulator traffic); "
                "it produces no training tokens and has no capture ledger to checkpoint"
            )

    def _require_quiescent() -> None:
        counts = limiter.counts()
        if counts["state"] != "paused" or counts["generation_pending_total"] != 0:
            raise LedgerNotQuiescentError(
                "capture-ledger commit requires paused admission and zero checkpoint-pending generation requests"
            )

    async def _commit_ledger(
        checkpoint_dir: Path,
        *,
        checkpoint_id: str,
        continuation_roots: list[AgentContinuationRoot],
        generation_cut_receipts: tuple[GenerationCutReceipt, ...],
    ) -> dict[str, Any]:
        ledger = ledger_provider()
        if generation_cut_receipts:
            if not isinstance(ledger, GenerationCutCaptureLedger):
                raise LedgerNotCheckpointableError(
                    "generation-prefix cuts require a capture ledger that can record cut coordinates"
                )
            for receipt in generation_cut_receipts:
                await ledger.record_generation_cut(receipt)
        expected_cut_records = sum(len(receipt.prefixes) for receipt in generation_cut_receipts)
        if isinstance(ledger, CheckpointableCaptureLedger):
            participant_dir = checkpoint_dir / MODEL_LEDGER_SUBDIR / server_name
            commit_result = await ledger.checkpoint_capture_ledger(
                participant_dir,
                checkpoint_id=checkpoint_id,
                server_name=server_name,
                tombstones=tuple(limiter.checkpoint_exclusions()),
                source_attempts=tuple(limiter.seen_attempts()),
                continuation_roots=tuple(continuation_roots),
                generation_cut_receipts=generation_cut_receipts,
            )
            validated = CaptureLedgerCommitResult.model_validate(commit_result)
            if validated.generation_cut_records != expected_cut_records:
                raise LedgerMismatchError(
                    "capture-ledger checkpoint did not commit the complete generation-cut inventory"
                )
            _validate_storage_reference_artifact(
                checkpoint_dir,
                validated.storage_reference_index,
            )
            return validated.model_dump(mode="json")

        file_root = file_ledger_root_provider()
        if file_root is None:
            raise LedgerNotCheckpointableError(
                "the configured CaptureLedger must implement CheckpointableCaptureLedger; "
                "Gym cannot infer how to snapshot a framework-owned backend"
            )
        checkpointer = CaptureLedgerCheckpointer(file_root, server_name=server_name)
        return await _run_sync(
            lambda: checkpointer.commit(
                checkpoint_dir,
                checkpoint_id=checkpoint_id,
                tombstones=limiter.checkpoint_exclusions(),
                source_attempts=limiter.seen_attempts(),
                continuation_roots=continuation_roots,
                generation_cut_receipts=generation_cut_receipts,
            )
        )

    async def _restore_ledger(checkpoint_dir: Path) -> dict[str, Any]:
        ledger = ledger_provider()
        if isinstance(ledger, CheckpointableCaptureLedger):
            participant_dir = checkpoint_dir / MODEL_LEDGER_SUBDIR / server_name
            restore_result = await ledger.restore_capture_ledger(participant_dir, server_name=server_name)
            validated = CaptureLedgerRestoreResult.model_validate(restore_result)
            _validate_storage_reference_artifact(
                checkpoint_dir,
                validated.storage_reference_index,
            )
            return validated.model_dump(mode="json")

        file_root = file_ledger_root_provider()
        if file_root is None:
            raise LedgerNotCheckpointableError(
                "the configured CaptureLedger must implement CheckpointableCaptureLedger; "
                "Gym cannot infer how to restore a framework-owned backend"
            )
        checkpointer = CaptureLedgerCheckpointer(file_root, server_name=server_name)
        return await _run_sync(lambda: checkpointer.restore(checkpoint_dir))

    @app.post(f"{MODEL_CHECKPOINT_URL_PREFIX}/commit")
    async def model_checkpoint_commit(
        body: ModelCheckpointCommitRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        _require_policy()
        if coordinator_client is not None:
            return await coordinator_client.request(
                "model_checkpoint_commit",
                body.model_dump(mode="json"),
                timeout_s=max(body.remaining(), 0.001),
            )

        async def run() -> dict[str, Any]:
            _require_quiescent()
            continuation_roots = await asyncio.to_thread(
                load_continuation_roots,
                Path(body.checkpoint_dir),
                body.continuation_indexes,
            )
            return await _commit_ledger(
                Path(body.checkpoint_dir),
                checkpoint_id=body.checkpoint_id,
                continuation_roots=continuation_roots,
                generation_cut_receipts=(
                    (limiter.generation_cut_receipt,) if limiter.generation_cut_receipt is not None else ()
                ),
            )

        return await fence.run_operation(
            body.checkpoint_id,
            "model-checkpoint/commit",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.COMMITTING,
            phase_after=CheckpointPhase.COMMITTED_PAUSED,
            run=run,
        )

    @app.post(f"{MODEL_CHECKPOINT_URL_PREFIX}/restore")
    async def model_checkpoint_restore(
        body: ModelCheckpointRestoreRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        _require_policy()
        if coordinator_client is not None:
            return await coordinator_client.request(
                "model_checkpoint_restore",
                body.model_dump(mode="json"),
                timeout_s=max(body.remaining(), 0.001),
            )

        async def run() -> dict[str, Any]:
            # The restored server boots into the paused state: nothing may be
            # admitted until every component is restored and the coordinator
            # explicitly resumes.
            limiter.close(body.checkpoint_id)
            result = await _restore_ledger(Path(body.checkpoint_dir))
            lineage_receipts = tuple(
                GenerationCutReceipt.model_validate(receipt) for receipt in result.pop("generation_cut_receipts", ())
            )
            if lineage_receipts and body.generation_cut_receipts:
                lineage_prefixes = sorted(
                    json.dumps(prefix.model_dump(mode="json"), sort_keys=True)
                    for receipt in lineage_receipts
                    for prefix in receipt.prefixes
                )
                request_prefixes = sorted(
                    json.dumps(prefix.model_dump(mode="json"), sort_keys=True)
                    for receipt in body.generation_cut_receipts
                    for prefix in receipt.prefixes
                )
                if lineage_prefixes != request_prefixes:
                    raise LedgerMismatchError(
                        "request-carried generation cuts disagree with the restored model lineage"
                    )
            generation_cut_receipts = lineage_receipts or body.generation_cut_receipts
            backend = limiter.generation_cut_backend
            if generation_cut_receipts and backend is None:
                raise LedgerNotCheckpointableError(
                    "checkpoint contains generation cuts but this model server has no restore backend"
                )
            restored_cuts = 0
            exclusions = frozenset((item.rollout_id, item.attempt_index) for item in body.generation_cut_exclusions)
            for receipt in generation_cut_receipts:
                if receipt.inventory.server_name != server_name:
                    raise LedgerMismatchError(
                        "generation-cut receipt belongs to a different model server: "
                        f"expected={server_name!r}, actual={receipt.inventory.server_name!r}"
                    )
                assert backend is not None
                restored = await backend.restore_generation_cut(
                    receipt,
                    excluded_replacements=exclusions,
                )
                if restored != receipt:
                    raise LedgerMismatchError("generation-cut restore did not acknowledge the persisted receipt")
                restored_cuts += sum(
                    prefix.disposition == "durable_prefix"
                    and prefix.cut_kind == "active_prefix"
                    and (prefix.rollout_id, prefix.attempt_index + 1) not in exclusions
                    for prefix in receipt.prefixes
                )
            for tombstone in result["tombstones"]:
                limiter.install_tombstone(tombstone["rollout_id"], tombstone["attempt_index"])
            for source_attempt in result.get("source_attempts", []):
                limiter.install_tombstone(source_attempt["rollout_id"], source_attempt["attempt_index"])
            result["generation_cuts_restored"] = restored_cuts
            return result

        return await fence.run_operation(
            body.checkpoint_id,
            "model-checkpoint/restore",
            allowed_phases=frozenset({CheckpointPhase.IDLE, CheckpointPhase.RESTORE_FAILED_PAUSED}),
            phase_during=CheckpointPhase.RESTORING,
            phase_after=CheckpointPhase.RESTORED_PAUSED,
            run=run,
            phase_on_failure=CheckpointPhase.RESTORE_FAILED_PAUSED,
        )


async def _run_sync(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    return await asyncio.to_thread(operation)
