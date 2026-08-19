# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic cross-process state storage for the rollout capture gate."""

from __future__ import annotations

import fcntl
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    CaptureAdmission,
    CommitCoords,
    RolloutReceipt,
)


def _default_metrics() -> dict[str, int]:
    return {
        "registered": 0,
        "register_retries": 0,
        "admitted": 0,
        "token_in": 0,
        "text": 0,
        "staged": 0,
        "capture_failed": 0,
        "sealed": 0,
        "seal_retries": 0,
        "failed": 0,
        "fail_retries": 0,
        "expired": 0,
        "capability_rejected": 0,
        "unattributed_calls": 0,
    }


class _StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RolloutRegistration(_StateModel):
    """Controller-only registration response carrying the data capability."""

    rollout_id: str
    owner_id: str
    operation_id: str
    data_capability: str
    capability_id: str
    expires_at: float


class CleanupManifest(_StateModel):
    """Staging keys whose rollout reached a terminal non-training outcome."""

    rollout_id: str
    reason: str
    staging_keys: list[str] = Field(default_factory=list)


class GateCallState(_StateModel):
    admission: CaptureAdmission
    request_items: list[dict]
    logical_request_id: str | None
    capability_id: str
    status: Literal["admitted", "committing", "staged", "failed"] = "admitted"
    manifest_record: CallRecord | None = None
    cumulative_token_ids: list[int] = Field(default_factory=list)
    pending_coords: CommitCoords | None = None
    # Retain the durable backend key even when lineage publication fails so
    # fail/expiry cleanup can return every staged object to its owner.
    cleanup_staging_key: str | None = None
    failure_reason: str | None = None


class GateRolloutState(_StateModel):
    registration: RolloutRegistration
    capability_digest: str
    calls: dict[str, GateCallState] = Field(default_factory=dict)
    logical_requests: dict[str, str] = Field(default_factory=dict)
    capture_poisoned: bool = False
    failure_reason: str | None = None


class GateTombstone(_StateModel):
    owner_id: str
    operation_id: str
    expires_at: float
    receipt: RolloutReceipt | None = None
    cleanup: CleanupManifest | None = None
    request_fingerprint: str = ""
    outcome: Literal["sealed", "failed", "expired"]


class SharedGateState(_StateModel):
    """Complete atomic gate state shared by every serving worker."""

    state_version: Literal[1] = 1
    rollouts: dict[str, GateRolloutState] = Field(default_factory=dict)
    tombstones: dict[str, GateTombstone] = Field(default_factory=dict)
    cleanup_manifests: list[CleanupManifest] = Field(default_factory=list)
    metrics: dict[str, int] = Field(default_factory=_default_metrics)


class GateStateStore(Protocol):
    """Provide an exclusive transaction over the complete gate state."""

    def transaction(self) -> AbstractContextManager[SharedGateState]: ...


class InMemoryGateStateStore:
    """Thread-safe state store for unit tests and embedded single-process use."""

    def __init__(self) -> None:
        self._state = SharedGateState()
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[SharedGateState]:
        with self._lock:
            working = self._state.model_copy(deep=True)
            yield working
            self._state = working


class FileGateStateStore:
    """Process-shared JSON state guarded by a POSIX advisory file lock."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if not self._path.is_absolute():
            raise ValueError("gate state_store_path must be absolute")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._path.with_name(f"{self._path.name}.lock")
        lock_fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(lock_fd)
        os.chmod(self._lock_path, 0o600)

    def _read_locked(self) -> SharedGateState:
        if not self._path.exists():
            return SharedGateState()
        try:
            payload = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return SharedGateState()
        if not payload:
            return SharedGateState()
        return SharedGateState.model_validate_json(payload)

    def _write_locked(self, state: SharedGateState) -> None:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            dir=self._path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                stream.write(state.model_dump_json())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self._path)
            os.chmod(self._path, 0o600)
            directory_fd = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    @contextmanager
    def transaction(self) -> Iterator[SharedGateState]:
        with self._lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_locked()
                yield state
                self._write_locked(state)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
