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
- **Manifest-last ordering.** Every ledger file is written and fsynced
  before the manifest appears (temporary name, fsync, rename). A commit that
  died partway leaves no manifest, and restore refuses the directory instead
  of installing a torn ledger.
- **Digest verification.** The manifest records each rollout file's SHA-256.
  Restore verifies every installed file against it, so silent corruption in
  transit fails loudly at restore instead of surfacing as wrong training
  data later.
"""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, runtime_checkable

from fastapi import FastAPI, Header
from pydantic import BaseModel, ConfigDict, Field

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
    CheckpointControlRequest,
    CheckpointPhase,
    ControlError,
    ControlFence,
)
from nemo_gym._checkpoint.model_admission import NotPolicyInstanceError
from nemo_gym.rollout_correlation import ROLLOUT_ID_PATTERN, capture_key_for
from nemo_gym.token_id_capture.control_routes import require_control_auth
from nemo_gym.token_id_capture.protocols import CaptureLedger


MODEL_CHECKPOINT_URL_PREFIX = f"{CONTROL_URL_PREFIX}/model-checkpoint"
MODEL_LEDGER_SUBDIR = "model-ledger"
LEDGER_MANIFEST_NAME = "manifest.json"
STORAGE_REFERENCE_INDEX_NAME = "storage-references.jsonl"
LEDGER_SCHEMA_VERSION = 2

# FileLineageStore writes one token-free custody file per rollout.
# Lock files and token-store files are not part of this participant.
_LEDGER_SUFFIX = ".lineage.jsonl"


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


class CaptureLedgerCommitResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollouts: int = Field(ge=0)
    rows: int = Field(ge=0)
    excluded_tombstoned: int = Field(ge=0)
    excluded_inactive: int = Field(default=0, ge=0)
    manifest_digest: str
    storage_reference_index: CheckpointArtifactReference


class CaptureLedgerRestoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollouts: int = Field(ge=0)
    rows: int = Field(ge=0)
    checkpoint_id: Optional[str] = None
    tombstones: list[AttemptIdentity] = Field(default_factory=list)
    source_attempts: list[AttemptIdentity] = Field(default_factory=list)
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


def _copy_lineage_fsynced(source: Path, target: Path) -> tuple[str, int, list[dict[str, Any]]]:
    """Copy, hash, count, and parse one quiescent lineage file in one source pass."""
    digest = hashlib.sha256()
    byte_count = 0
    records: list[dict[str, Any]] = []
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ledger-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with source.open("rb") as src:
                for line_number, line in enumerate(src, start=1):
                    handle.write(line)
                    digest.update(line)
                    byte_count += len(line)
                    payload = line.strip()
                    if not payload:
                        continue
                    try:
                        record = json.loads(payload)
                    except json.JSONDecodeError as error:
                        raise LedgerMismatchError(
                            f"invalid lineage JSON in {source.name!r} at line {line_number}"
                        ) from error
                    if not isinstance(record, dict):
                        raise LedgerMismatchError(
                            f"lineage row in {source.name!r} at line {line_number} is not an object"
                        )
                    records.append(record)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, target)
    return digest.hexdigest(), byte_count, records


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
    boundary_model_call_id: str,
) -> list[ExternalStorageReference]:
    selected_records = [
        record
        for record in records
        if record.get("model_call_id") == boundary_model_call_id and record.get("failure_reason") is None
    ]
    if len(selected_records) != 1:
        raise LedgerMismatchError(
            "continuation boundary is missing or ambiguous in model lineage: "
            f"capture_key={capture_key!r}, model_call_id={boundary_model_call_id!r}"
        )

    references: list[ExternalStorageReference] = []
    seen_keys: set[str] = set()
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
    ) -> dict[str, Any]:
        """Copy the ledger into ``checkpoint_dir``; the caller has already drained.

        The store must be quiescent (admission paused) when this runs: the
        copy takes no locks because nothing may be writing.
        """
        checkpoint_dir = Path(checkpoint_dir)
        ledger_dir = self._ledger_dir(checkpoint_dir)
        normalized_roots = _normalize_continuation_roots(continuation_roots)
        roots_digest = _canonical_continuation_roots_digest(continuation_roots)
        if (ledger_dir / LEDGER_MANIFEST_NAME).exists():
            result = self._validate_committed(
                ledger_dir,
                checkpoint_root=checkpoint_dir,
                checkpoint_id=checkpoint_id,
                server_name=self.server_name,
                tombstones=tombstones,
                source_attempts=source_attempts or [],
                continuation_roots_digest=roots_digest,
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
        sources = {capture_key: self.store_root / f"{capture_key}{_LEDGER_SUFFIX}" for capture_key in normalized_roots}
        missing_roots = sorted(capture_key for capture_key, source in sources.items() if not source.is_file())
        if missing_roots:
            raise LedgerMismatchError(f"continuation roots have no model lineage: capture_keys={missing_roots!r}")

        ledger_dir.mkdir(parents=True, exist_ok=True)

        rollouts: dict[str, dict[str, Any]] = {}
        excluded = len(tombstones)
        source_capture_keys = {
            capture_key_for(rollout_id, attempt_index) for rollout_id, attempt_index in source_attempts or []
        }
        excluded_inactive = len(source_capture_keys - set(normalized_roots) - fenced)
        total_rows = 0
        external_references: dict[str, ExternalStorageReference] = {}
        for capture_key, root in sorted(normalized_roots.items()):
            files: dict[str, str] = {}
            source = sources[capture_key]
            target = ledger_dir / source.name
            file_digest, byte_count, records = _copy_lineage_fsynced(source, target)
            files[source.name] = file_digest
            references = _external_references_for_rows(
                capture_key,
                records,
                root.last_committed_model_call_id,
            )
            for reference in references:
                external_references.setdefault(reference.key, reference)
            rollouts[capture_key] = {
                "files": files,
                "rows": len(records),
                "bytes": byte_count,
            }
            total_rows += len(records)
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
            "rollouts": rollouts,
            "continuation_roots_sha256": roots_digest,
            "continuation_roots": len(normalized_roots),
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
            "rollouts": len(rollouts),
            "rows": total_rows,
            "excluded_tombstoned": excluded,
            "excluded_inactive": excluded_inactive,
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
        storage_reference_index = _validate_storage_reference_index(checkpoint_root, manifest)
        total_rows = 0
        for rollout_id, metadata in manifest.get("rollouts", {}).items():
            for name, digest in metadata.get("files", {}).items():
                path = ledger_dir / name
                if not path.exists() or _file_digest(path) != digest:
                    raise LedgerMismatchError(f"committed ledger file {name!r} for {rollout_id!r} is corrupted")
            total_rows += int(metadata.get("rows", 0))
        return {
            "rollouts": len(manifest.get("rollouts", {})),
            "rows": total_rows,
            "excluded_tombstoned": len(manifest.get("tombstones", [])),
            "excluded_inactive": int(manifest.get("excluded_inactive", 0)),
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
        if manifest.get("schema_version", 0) > LEDGER_SCHEMA_VERSION:
            raise LedgerMismatchError(
                f"ledger manifest schema_version {manifest.get('schema_version')} is newer than this "
                f"reader ({LEDGER_SCHEMA_VERSION})"
            )
        if manifest.get("server_name") != self.server_name:
            raise LedgerMismatchError("ledger checkpoint belongs to a different model server")
        storage_reference_index = _validate_storage_reference_index(checkpoint_dir, manifest)

        expected_names = {name for metadata in manifest["rollouts"].values() for name in metadata["files"]}
        existing_names = {path.name for path in self.store_root.glob(f"*{_LEDGER_SUFFIX}")}
        unexpected = existing_names - expected_names
        if unexpected:
            raise LedgerMismatchError(
                "restore requires a fresh capture-ledger namespace; "
                f"found files absent from the checkpoint: {sorted(unexpected)}"
            )

        # Validate the complete source before changing the live namespace.
        validated: list[tuple[Path, str]] = []
        total_rows = 0
        for rollout_id, meta in manifest["rollouts"].items():
            for name, digest in meta["files"].items():
                source = ledger_dir / name
                if not source.exists() or _file_digest(source) != digest:
                    raise LedgerMismatchError(
                        f"ledger file {name} for rollout {rollout_id!r} is missing or does not match "
                        f"its committed digest; refusing to install a corrupted ledger"
                    )
                validated.append((source, name))
            total_rows += int(meta.get("rows", 0))

        self.store_root.mkdir(parents=True, exist_ok=True)
        for source, name in validated:
            _copy_fsynced(source, self.store_root / name)
        _fsync_dir(self.store_root)

        result: dict[str, Any] = {
            "rollouts": len(manifest["rollouts"]),
            "rows": total_rows,
            "checkpoint_id": manifest.get("checkpoint_id"),
            "tombstones": list(manifest.get("tombstones", ())),
            "source_attempts": list(manifest.get("source_attempts", ())),
        }
        result["storage_reference_index"] = storage_reference_index.model_dump(mode="json")
        return result


class ModelCheckpointCommitRequest(CheckpointControlRequest):
    checkpoint_dir: str
    continuation_indexes: list[CheckpointArtifactReference]


class ModelCheckpointRestoreRequest(CheckpointControlRequest):
    checkpoint_dir: str


def _validate_server_name(server_name: str) -> str:
    if ROLLOUT_ID_PATTERN.fullmatch(server_name) is None:
        raise ValueError(
            "model server name must contain only letters, digits, dots, dashes, or underscores "
            "and start with a letter or digit"
        )
    return server_name


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
        if counts["state"] != "paused" or counts["inflight_total"] != 0:
            raise LedgerNotQuiescentError(
                "capture-ledger commit requires paused admission and zero in-flight generation requests"
            )

    async def _commit_ledger(
        checkpoint_dir: Path,
        *,
        checkpoint_id: str,
        continuation_roots: list[AgentContinuationRoot],
    ) -> dict[str, Any]:
        ledger = ledger_provider()
        if isinstance(ledger, CheckpointableCaptureLedger):
            participant_dir = checkpoint_dir / MODEL_LEDGER_SUBDIR / server_name
            commit_result = await ledger.checkpoint_capture_ledger(
                participant_dir,
                checkpoint_id=checkpoint_id,
                server_name=server_name,
                tombstones=tuple(limiter.checkpoint_exclusions()),
                source_attempts=tuple(limiter.seen_attempts()),
                continuation_roots=tuple(continuation_roots),
            )
            validated = CaptureLedgerCommitResult.model_validate(commit_result)
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

        async def run() -> dict[str, Any]:
            # The restored server boots into the paused state: nothing may be
            # admitted until every component is restored and the coordinator
            # explicitly resumes.
            limiter.close()
            result = await _restore_ledger(Path(body.checkpoint_dir))
            for tombstone in result["tombstones"]:
                limiter.install_tombstone(tombstone["rollout_id"], tombstone["attempt_index"])
            for source_attempt in result.get("source_attempts", []):
                limiter.install_tombstone(source_attempt["rollout_id"], source_attempt["attempt_index"])
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
