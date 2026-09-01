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

from nemo_gym.checkpoint.admission import AdmissionLimiter
from nemo_gym.checkpoint.control import (
    CONTROL_URL_PREFIX,
    CheckpointControlRequest,
    CheckpointPhase,
    ControlError,
    ControlFence,
)
from nemo_gym.checkpoint.model_admission import NotPolicyInstanceError
from nemo_gym.rollout_correlation import capture_key_for
from nemo_gym.token_id_capture.control_routes import require_control_auth
from nemo_gym.token_id_capture.protocols import CaptureLedger


MODEL_CHECKPOINT_URL_PREFIX = f"{CONTROL_URL_PREFIX}/model-checkpoint"
MODEL_LEDGER_SUBDIR = "model-ledger"
LEDGER_MANIFEST_NAME = "manifest.json"
LEDGER_SCHEMA_VERSION = 1

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


@runtime_checkable
class CheckpointableCaptureLedger(CaptureLedger, Protocol):
    """Optional lifecycle implemented by framework-owned capture backends.

    The backend snapshots token-free custody and its private parent-resolution
    state. The framework checkpoints staged token arrays separately.
    """

    async def checkpoint_capture_ledger(
        self,
        checkpoint_dir: Path,
        *,
        checkpoint_id: str,
        tombstones: tuple[tuple[str, int], ...],
        source_attempts: tuple[tuple[str, int], ...],
    ) -> dict[str, Any]: ...

    async def restore_capture_ledger(self, checkpoint_dir: Path) -> dict[str, Any]: ...


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


class CaptureLedgerCheckpointer:
    """Commit and restore one token-capture store directory."""

    def __init__(self, store_root: Path) -> None:
        self.store_root = Path(store_root)

    def _rollout_ids(self) -> list[str]:
        return sorted(path.name[: -len(_LEDGER_SUFFIX)] for path in self.store_root.glob(f"*{_LEDGER_SUFFIX}"))

    def commit(
        self,
        checkpoint_dir: Path,
        *,
        checkpoint_id: str,
        tombstones: list[tuple[str, int]],
        source_attempts: Optional[list[tuple[str, int]]] = None,
    ) -> dict[str, Any]:
        """Copy the ledger into ``checkpoint_dir``; the caller has already drained.

        The store must be quiescent (admission paused) when this runs: the
        copy takes no locks because nothing may be writing.
        """
        ledger_dir = Path(checkpoint_dir) / MODEL_LEDGER_SUBDIR
        if (ledger_dir / LEDGER_MANIFEST_NAME).exists():
            return self._validate_committed(ledger_dir, checkpoint_id=checkpoint_id)
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
            "rollouts": rollouts,
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
            "manifest_digest": hashlib.sha256(payload).hexdigest(),
        }

    @staticmethod
    def _validate_committed(ledger_dir: Path, *, checkpoint_id: str) -> dict[str, Any]:
        manifest_path = ledger_dir / LEDGER_MANIFEST_NAME
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
        if manifest.get("checkpoint_id") != checkpoint_id:
            raise LedgerMismatchError(
                f"ledger directory belongs to {manifest.get('checkpoint_id')!r}, not {checkpoint_id!r}"
            )
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
            "manifest_digest": hashlib.sha256(payload).hexdigest(),
        }

    def restore(self, checkpoint_dir: Path) -> dict[str, Any]:
        """Install a committed ledger into this store root and verify it."""
        ledger_dir = Path(checkpoint_dir) / MODEL_LEDGER_SUBDIR
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

        return {
            "rollouts": len(manifest["rollouts"]),
            "rows": total_rows,
            "checkpoint_id": manifest.get("checkpoint_id"),
            "tombstones": list(manifest.get("tombstones", ())),
            "source_attempts": list(manifest.get("source_attempts", ())),
        }


class ModelCheckpointCommitRequest(CheckpointControlRequest):
    checkpoint_dir: str


class ModelCheckpointRestoreRequest(CheckpointControlRequest):
    checkpoint_dir: str


def install_model_checkpoint(
    app: FastAPI,
    *,
    fence: ControlFence,
    limiter: AdmissionLimiter,
    ledger_provider: Callable[[], Optional[CaptureLedger]],
    file_ledger_root_provider: Callable[[], Optional[Path]],
    instance_role: Literal["policy", "auxiliary"],
    auth_token: str,
) -> None:
    """Register ``/ng-control/v1/model-checkpoint`` on a model-server app.

    Commit requires the prepared (drained) phase; restore runs on a freshly
    started server and leaves it paused, so nothing serves until the
    coordinator has restored every component and explicitly resumes.
    """

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

    async def _with_ledger(
        checkpoint_dir: Path,
        *,
        checkpoint_id: str,
        operation: Literal["commit", "restore"],
    ) -> dict[str, Any]:
        ledger = ledger_provider()
        if isinstance(ledger, CheckpointableCaptureLedger):
            if operation == "commit":
                return await ledger.checkpoint_capture_ledger(
                    checkpoint_dir,
                    checkpoint_id=checkpoint_id,
                    tombstones=tuple(limiter.tombstones()),
                    source_attempts=tuple(limiter.seen_attempts()),
                )
            return await ledger.restore_capture_ledger(checkpoint_dir)

        file_root = file_ledger_root_provider()
        if file_root is None:
            raise LedgerNotCheckpointableError(
                "the configured CaptureLedger must implement CheckpointableCaptureLedger; "
                "Gym cannot infer how to snapshot a framework-owned backend"
            )
        checkpointer = CaptureLedgerCheckpointer(file_root)
        if operation == "commit":
            return await _run_sync(
                lambda: checkpointer.commit(
                    checkpoint_dir,
                    checkpoint_id=checkpoint_id,
                    tombstones=limiter.tombstones(),
                    source_attempts=limiter.seen_attempts(),
                )
            )
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
            return await _with_ledger(
                Path(body.checkpoint_dir),
                checkpoint_id=body.checkpoint_id,
                operation="commit",
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
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.RESTORING,
            phase_after=CheckpointPhase.RESTORED_PAUSED,
            run=run,
        )


async def _run_sync(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    return await asyncio.to_thread(operation)
