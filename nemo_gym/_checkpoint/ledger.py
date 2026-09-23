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
from nemo_gym._checkpoint.control import (
    CONTROL_URL_PREFIX,
    CheckpointControlRequest,
    CheckpointPhase,
    ControlError,
    ControlFence,
)
from nemo_gym._checkpoint.model_admission import NotPolicyInstanceError
from nemo_gym._checkpoint.model_control_contracts import (
    GenerationCutCoordinatorProof,
    GenerationCutReceipt,
)
from nemo_gym.rollout_correlation import ROLLOUT_ID_PATTERN, capture_key_for
from nemo_gym.token_id_capture.control_routes import require_control_auth
from nemo_gym.token_id_capture.protocols import CaptureLedger


MODEL_CHECKPOINT_URL_PREFIX = f"{CONTROL_URL_PREFIX}/model-checkpoint"
MODEL_LEDGER_SUBDIR = "model-ledger"
LEDGER_MANIFEST_NAME = "manifest.json"
GENERATION_CUT_COORDINATOR_PROOF_NAME = "generation-cut-workers.json"
LEGACY_GENERATION_CUT_ACK_NAME = "generation-cut.json"
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
    manifest_digest: str
    generation_cut_receipt: GenerationCutReceipt | None = None
    generation_cut_proof: GenerationCutCoordinatorProof | None = None


class CaptureLedgerRestoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollouts: int = Field(ge=0)
    rows: int = Field(ge=0)
    checkpoint_id: Optional[str] = None
    tombstones: list[AttemptIdentity] = Field(default_factory=list)
    source_attempts: list[AttemptIdentity] = Field(default_factory=list)
    generation_cut_receipt: GenerationCutReceipt | None = None
    generation_cut_proof: GenerationCutCoordinatorProof | None = None


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


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_ledger_schema(
    manifest: dict[str, Any],
    ledger_dir: Path,
    *,
    commit_retry: bool = False,
) -> None:
    version = manifest.get("schema_version")
    if not isinstance(version, int) or version < 1:
        raise LedgerMismatchError(f"ledger manifest has invalid schema_version {version!r}")
    if version > LEDGER_SCHEMA_VERSION:
        raise LedgerMismatchError(
            f"ledger manifest schema_version {version} is newer than this reader ({LEDGER_SCHEMA_VERSION})"
        )
    legacy_cut = "generation_cut_ack" in manifest or (ledger_dir / LEGACY_GENERATION_CUT_ACK_NAME).exists()
    if legacy_cut:
        raise LedgerMismatchError(
            "generation-cut checkpoint schema v1 cannot be migrated safely because its ack and sidecar "
            "did not prove a final durable backend snapshot; recreate the checkpoint with schema_version 2"
        )
    if commit_retry and version != LEDGER_SCHEMA_VERSION:
        raise LedgerMismatchError(
            f"cannot retry a schema_version {version} ledger commit with schema_version {LEDGER_SCHEMA_VERSION}; "
            "restore the old checkpoint first and create a new checkpoint"
        )


class CaptureLedgerCheckpointer:
    """Commit and restore one token-capture store directory."""

    def __init__(self, store_root: Path, *, server_name: Optional[str] = None) -> None:
        self.store_root = Path(store_root)
        self.server_name = _validate_server_name(server_name) if server_name is not None else None

    def _ledger_dir(self, checkpoint_dir: Path) -> Path:
        directory = Path(checkpoint_dir) / MODEL_LEDGER_SUBDIR
        return directory / self.server_name if self.server_name is not None else directory

    def _rollout_ids(self) -> list[str]:
        return sorted(path.name[: -len(_LEDGER_SUFFIX)] for path in self.store_root.glob(f"*{_LEDGER_SUFFIX}"))

    def commit(
        self,
        checkpoint_dir: Path,
        *,
        checkpoint_id: str,
        tombstones: list[tuple[str, int]],
        source_attempts: Optional[list[tuple[str, int]]] = None,
        generation_cut_receipt: GenerationCutReceipt | None = None,
        generation_cut_proof: GenerationCutCoordinatorProof | None = None,
    ) -> dict[str, Any]:
        """Copy the ledger into ``checkpoint_dir``; the caller has already drained.

        The store must be quiescent (admission paused) when this runs: the
        copy takes no locks because nothing may be writing.
        """
        if generation_cut_receipt is not None and generation_cut_proof is not None:
            raise ValueError("single-worker cut receipt and coordinator cut proof are mutually exclusive")
        if generation_cut_proof is not None:
            generation_cut_proof = GenerationCutCoordinatorProof.model_validate(
                generation_cut_proof.model_dump(mode="json")
            )
            if generation_cut_proof.checkpoint_id != checkpoint_id:
                raise ValueError("coordinator generation-cut proof belongs to a different checkpoint")
        ledger_dir = self._ledger_dir(checkpoint_dir)
        if (ledger_dir / LEDGER_MANIFEST_NAME).exists():
            result = self._validate_committed(
                ledger_dir,
                checkpoint_id=checkpoint_id,
                server_name=self.server_name,
                tombstones=tombstones,
                source_attempts=source_attempts or [],
                generation_cut_receipt=generation_cut_receipt,
                generation_cut_proof=generation_cut_proof,
            )
            # A previous attempt may have renamed the manifest and then
            # failed its final directory fsync. Retry that durability barrier.
            _fsync_dir(ledger_dir)
            return result
        ledger_dir.mkdir(parents=True, exist_ok=True)
        fenced = {capture_key_for(rollout_id, attempt_index) for rollout_id, attempt_index in tombstones}

        rollouts: dict[str, dict[str, Any]] = {}
        excluded = 0
        total_rows = 0
        for rollout_id in self._rollout_ids():
            if rollout_id in fenced:
                excluded += 1
                continue
            files: dict[str, str] = {}
            rows = 0
            source = self.store_root / f"{rollout_id}{_LEDGER_SUFFIX}"
            target = ledger_dir / source.name
            _copy_fsynced(source, target)
            files[source.name] = _file_digest(target)
            rows = sum(1 for line in target.read_bytes().splitlines() if line.strip())
            rollouts[rollout_id] = {"files": files, "rows": rows}
            total_rows += rows
        _fsync_dir(ledger_dir)

        manifest = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "server_name": self.server_name,
            "rollouts": rollouts,
            "tombstones": [
                {"rollout_id": rollout_id, "attempt_index": attempt} for rollout_id, attempt in sorted(tombstones)
            ],
            "source_attempts": [
                {"rollout_id": rollout_id, "attempt_index": attempt}
                for rollout_id, attempt in sorted(source_attempts or [])
            ],
        }
        if generation_cut_receipt is not None:
            manifest["generation_cut_receipt"] = generation_cut_receipt.model_dump(mode="json")
        if generation_cut_proof is not None:
            manifest["generation_cut_proof"] = generation_cut_proof.model_dump(mode="json")
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

        result = {
            "rollouts": len(rollouts),
            "rows": total_rows,
            "excluded_tombstoned": excluded,
            "manifest_digest": hashlib.sha256(payload).hexdigest(),
        }
        if generation_cut_receipt is not None:
            result["generation_cut_receipt"] = generation_cut_receipt.model_dump(mode="json")
        if generation_cut_proof is not None:
            result["generation_cut_proof"] = generation_cut_proof.model_dump(mode="json")
        return result

    @staticmethod
    def _validate_committed(
        ledger_dir: Path,
        *,
        checkpoint_id: str,
        server_name: Optional[str],
        tombstones: list[tuple[str, int]],
        source_attempts: list[tuple[str, int]],
        generation_cut_receipt: GenerationCutReceipt | None,
        generation_cut_proof: GenerationCutCoordinatorProof | None,
    ) -> dict[str, Any]:
        manifest_path = ledger_dir / LEDGER_MANIFEST_NAME
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
        _validate_ledger_schema(manifest, ledger_dir, commit_retry=True)
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
        expected_cut = generation_cut_receipt.model_dump(mode="json") if generation_cut_receipt is not None else None
        if manifest.get("generation_cut_receipt") != expected_cut:
            raise LedgerMismatchError("committed ledger generation cut changed before commit retry")
        expected_proof = generation_cut_proof.model_dump(mode="json") if generation_cut_proof is not None else None
        if manifest.get("generation_cut_proof") != expected_proof:
            raise LedgerMismatchError("committed ledger worker generation-cut proof changed before commit retry")
        total_rows = 0
        for rollout_id, metadata in manifest.get("rollouts", {}).items():
            for name, digest in metadata.get("files", {}).items():
                path = ledger_dir / name
                if not path.exists() or _file_digest(path) != digest:
                    raise LedgerMismatchError(f"committed ledger file {name!r} for {rollout_id!r} is corrupted")
            total_rows += int(metadata.get("rows", 0))
        result = {
            "rollouts": len(manifest.get("rollouts", {})),
            "rows": total_rows,
            "excluded_tombstoned": len(manifest.get("tombstones", [])),
            "manifest_digest": hashlib.sha256(payload).hexdigest(),
        }
        if manifest.get("generation_cut_receipt") is not None:
            result["generation_cut_receipt"] = manifest["generation_cut_receipt"]
        if manifest.get("generation_cut_proof") is not None:
            result["generation_cut_proof"] = manifest["generation_cut_proof"]
        return result

    def restore(self, checkpoint_dir: Path) -> dict[str, Any]:
        """Install a committed ledger into this store root and verify it."""
        ledger_dir = self._ledger_dir(checkpoint_dir)
        manifest_path = ledger_dir / LEDGER_MANIFEST_NAME
        if not manifest_path.exists():
            raise LedgerMismatchError(
                f"no ledger manifest at {manifest_path}; the commit tore partway or never ran, "
                f"so this directory must not be installed"
            )
        manifest = json.loads(manifest_path.read_text())
        _validate_ledger_schema(manifest, ledger_dir)
        if manifest.get("server_name") != self.server_name:
            raise LedgerMismatchError("ledger checkpoint belongs to a different model server")

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

        result = {
            "rollouts": len(manifest["rollouts"]),
            "rows": total_rows,
            "checkpoint_id": manifest.get("checkpoint_id"),
            "tombstones": list(manifest.get("tombstones", ())),
            "source_attempts": list(manifest.get("source_attempts", ())),
        }
        if manifest.get("generation_cut_receipt") is not None:
            receipt = GenerationCutReceipt.model_validate(manifest["generation_cut_receipt"])
            result["generation_cut_receipt"] = receipt.model_dump(mode="json")
        if manifest.get("generation_cut_proof") is not None:
            proof = GenerationCutCoordinatorProof.model_validate(manifest["generation_cut_proof"])
            result["generation_cut_proof"] = proof.model_dump(mode="json")
        return result


class ModelCheckpointCommitRequest(CheckpointControlRequest):
    checkpoint_dir: str


class ModelCheckpointRestoreRequest(CheckpointControlRequest):
    checkpoint_dir: str


def _validate_server_name(server_name: str) -> str:
    if ROLLOUT_ID_PATTERN.fullmatch(server_name) is None:
        raise ValueError(
            "model server name must contain only letters, digits, dots, dashes, or underscores "
            "and start with a letter or digit"
        )
    return server_name


async def _restore_generation_cut(
    backend: Any,
    expected: GenerationCutReceipt | None,
) -> None:
    if expected is None:
        return
    if backend is None:
        raise LedgerMismatchError("checkpoint contains a generation cut but no cut backend is configured")
    restored = GenerationCutReceipt.model_validate(await backend.restore_generation_cut(expected))
    if restored != expected:
        raise LedgerMismatchError("restored generation cut does not match the model-ledger manifest")


def _store_generation_cut_proof(directory: Path, proof: GenerationCutCoordinatorProof) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / GENERATION_CUT_COORDINATOR_PROOF_NAME
    payload = json.dumps(proof.model_dump(mode="json"), sort_keys=True, indent=1).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise LedgerMismatchError("coordinator generation-cut proof changed before checkpoint retry")
        return
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".generation-cut-workers-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    _fsync_dir(directory)


def _load_generation_cut_proof(directory: Path) -> GenerationCutCoordinatorProof | None:
    path = directory / GENERATION_CUT_COORDINATOR_PROOF_NAME
    if not path.exists():
        return None
    return GenerationCutCoordinatorProof.model_validate_json(path.read_bytes())


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
    generation_cut_proof_provider: Callable[[], GenerationCutCoordinatorProof] | None = None,
    expected_workers: int = 1,
) -> None:
    """Register ``/ng-control/v1/model-checkpoint`` on a model-server app.

    Commit requires the prepared (drained) phase; restore runs on a freshly
    started server and leaves it paused, so nothing serves until the
    coordinator has restored every component and explicitly resumes.
    """

    server_name = _validate_server_name(server_name)
    if expected_workers < 1:
        raise ValueError("expected_workers must be positive")

    def _require_policy() -> None:
        if instance_role != "policy":
            raise NotPolicyInstanceError(
                "this model-server instance is auxiliary (judge or simulator traffic); "
                "it produces no training tokens and has no capture ledger to checkpoint"
            )

    def _require_quiescent(checkpoint_id: str) -> GenerationCutCoordinatorProof | None:
        if expected_workers > 1 and generation_cut_proof_provider is None:
            raise LedgerNotQuiescentError(
                "multi-worker ledger commit requires the coordinator's complete generation-cut proof"
            )
        if generation_cut_proof_provider is not None:
            try:
                supplied_proof = generation_cut_proof_provider()
                proof = GenerationCutCoordinatorProof.model_validate(supplied_proof.model_dump(mode="json"))
            except (AttributeError, TypeError, ValueError) as error:
                raise LedgerNotQuiescentError(f"coordinator generation-cut proof is incomplete: {error}") from error
            if proof.checkpoint_id != checkpoint_id:
                raise LedgerNotQuiescentError("coordinator generation-cut proof belongs to a different checkpoint")
            if proof.expected_workers != expected_workers:
                raise LedgerNotQuiescentError(
                    "coordinator generation-cut proof does not match configured worker membership"
                )
            return proof
        counts = limiter.counts()
        if counts["state"] != "paused" or not limiter.is_prepare_safe():
            raise LedgerNotQuiescentError(
                "capture-ledger commit requires paused admission and a generation-safe frozen cut"
            )
        return None

    async def _with_ledger(
        checkpoint_dir: Path,
        *,
        checkpoint_id: str,
        operation: Literal["commit", "restore"],
        generation_cut_proof: GenerationCutCoordinatorProof | None = None,
    ) -> dict[str, Any]:
        ledger = ledger_provider()
        cut_backend = limiter.generation_cut_backend
        cut_receipt = None if generation_cut_proof is not None else limiter.generation_cut_receipt
        participant_dir = checkpoint_dir / MODEL_LEDGER_SUBDIR / server_name
        if operation == "commit" and generation_cut_proof is not None:
            await _run_sync(lambda: _store_generation_cut_proof(participant_dir, generation_cut_proof))
        if isinstance(ledger, CheckpointableCaptureLedger):
            if operation == "commit":
                result = await ledger.checkpoint_capture_ledger(
                    participant_dir,
                    checkpoint_id=checkpoint_id,
                    server_name=server_name,
                    tombstones=tuple(limiter.checkpoint_exclusions()),
                    source_attempts=tuple(limiter.seen_attempts()),
                )
                validated = CaptureLedgerCommitResult.model_validate(result)
                if (
                    validated.generation_cut_proof is not None
                    and validated.generation_cut_proof != generation_cut_proof
                ):
                    raise LedgerMismatchError("capture ledger and coordinator generation-cut proof disagree")
                validated.generation_cut_proof = generation_cut_proof
                if validated.generation_cut_receipt is not None and validated.generation_cut_receipt != cut_receipt:
                    raise LedgerMismatchError("capture ledger and generation-cut receipt disagree")
                validated.generation_cut_receipt = cut_receipt
                return validated.model_dump(mode="json")
            result = await ledger.restore_capture_ledger(participant_dir, server_name=server_name)
            validated = CaptureLedgerRestoreResult.model_validate(result)
            sidecar_proof = await _run_sync(lambda: _load_generation_cut_proof(participant_dir))
            if validated.generation_cut_proof is not None and sidecar_proof not in (
                None,
                validated.generation_cut_proof,
            ):
                raise LedgerMismatchError("capture ledger and coordinator generation-cut proof disagree")
            validated.generation_cut_proof = validated.generation_cut_proof or sidecar_proof
            await _restore_generation_cut(
                cut_backend,
                validated.generation_cut_receipt,
            )
            return validated.model_dump(mode="json")

        file_root = file_ledger_root_provider()
        if file_root is None:
            raise LedgerNotCheckpointableError(
                "the configured CaptureLedger must implement CheckpointableCaptureLedger; "
                "Gym cannot infer how to snapshot a framework-owned backend"
            )
        checkpointer = CaptureLedgerCheckpointer(file_root, server_name=server_name)
        if operation == "commit":
            result = await _run_sync(
                lambda: checkpointer.commit(
                    checkpoint_dir,
                    checkpoint_id=checkpoint_id,
                    tombstones=limiter.checkpoint_exclusions(),
                    source_attempts=limiter.seen_attempts(),
                    generation_cut_receipt=cut_receipt,
                    generation_cut_proof=generation_cut_proof,
                )
            )
            return result
        result = await _run_sync(lambda: checkpointer.restore(checkpoint_dir))
        sidecar_proof = await _run_sync(lambda: _load_generation_cut_proof(participant_dir))
        manifest_proof = (
            GenerationCutCoordinatorProof.model_validate(result["generation_cut_proof"])
            if result.get("generation_cut_proof") is not None
            else None
        )
        if sidecar_proof is not None and manifest_proof not in (None, sidecar_proof):
            raise LedgerMismatchError("model ledger and coordinator generation-cut proof disagree")
        if sidecar_proof is not None:
            result["generation_cut_proof"] = sidecar_proof.model_dump(mode="json")
        expected_receipt = (
            GenerationCutReceipt.model_validate(result["generation_cut_receipt"])
            if result.get("generation_cut_receipt") is not None
            else None
        )
        await _restore_generation_cut(
            cut_backend,
            expected_receipt,
        )
        return result

    @app.post(f"{MODEL_CHECKPOINT_URL_PREFIX}/commit")
    async def model_checkpoint_commit(
        body: ModelCheckpointCommitRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        _require_policy()

        async def run() -> dict[str, Any]:
            generation_cut_proof = _require_quiescent(body.checkpoint_id)
            return await _with_ledger(
                Path(body.checkpoint_dir),
                checkpoint_id=body.checkpoint_id,
                operation="commit",
                generation_cut_proof=generation_cut_proof,
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
            result = await _with_ledger(
                Path(body.checkpoint_dir),
                checkpoint_id=body.checkpoint_id,
                operation="restore",
            )
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
