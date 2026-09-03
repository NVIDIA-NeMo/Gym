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
"""Bulk checkpoint participant for stateful resources servers."""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

from fastapi import FastAPI, Header, Query
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym._checkpoint.control import (
    CheckpointControlRequest,
    CheckpointPhase,
    ControlError,
    ControlFence,
    Deadline,
)
from nemo_gym.rollout_correlation import (
    ATTEMPT_INDEX_HEADER,
    ROLLOUT_ID_HEADER,
    ROLLOUT_ID_PATTERN,
    current_execution_identity,
)
from nemo_gym.token_id_capture.control_routes import require_control_auth


RESOURCES_CHECKPOINT_URL_PREFIX = "/ng-control/v1/resources-checkpoint"
RESOURCES_STATE_SUBDIR = "resources"
RESOURCES_MANIFEST_NAME = "manifest.json"
RESOURCES_CHECKPOINT_SCHEMA_VERSION = 1
RESOURCE_STATE_REVISION_HEADER = "x-nemo-gym-resource-state-revision"

ResourcesRouteKind = Literal["read", "start", "mutation", "terminal"]


class ResourcesCheckpointError(ControlError):
    code = "resources_checkpoint_error"


class ResourcesAdmissionClosedError(ControlError):
    code = "resources_admission_closed"


class ResourcesStaleAttemptError(ControlError):
    code = "stale_attempt"


class ResourcesSessionUnboundError(ControlError):
    code = "resources_session_unbound"


class ResourcesUnsafeRestoreError(ControlError):
    code = "unsafe_resources_restore"


class ResourceSnapshot(BaseModel):
    """One resources server's state after a completed tool mutation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[RESOURCES_CHECKPOINT_SCHEMA_VERSION] = RESOURCES_CHECKPOINT_SCHEMA_VERSION
    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)
    state_revision: int = Field(ge=0)
    state: dict[str, Any]
    created_at: float = Field(default_factory=time.time)


class ResourcesPrepareRequest(CheckpointControlRequest):
    pass


class ResourcesCommitRequest(CheckpointControlRequest):
    checkpoint_dir: str


class ResourcesRestoreRequest(CheckpointControlRequest):
    checkpoint_dir: str


class ResourcesResumeRequest(CheckpointControlRequest):
    pass


class ResourcesRetireRequest(CheckpointControlRequest):
    """Identify one execution to retire from an active checkpoint."""

    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)


class _LockEntry:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users = 0


class ResourcesCheckpointParticipant:
    """Serialize mutations per execution and export all known live sessions."""

    def __init__(
        self,
        *,
        export_state: Callable[[str, int], Awaitable[dict[str, Any]]],
        restore_states: Callable[[list[ResourceSnapshot]], Awaitable[None]],
        retire_state: Optional[Callable[[str, int], Awaitable[None]]] = None,
        restore_expected: bool = False,
    ) -> None:
        self._export_state = export_state
        self._restore_states = restore_states
        self._retire_state = retire_state
        self._locks: dict[tuple[str, int], _LockEntry] = {}
        self._revisions: dict[tuple[str, int], int] = {}
        self._prepared: list[ResourceSnapshot] = []
        self._tombstones: set[tuple[str, int]] = set()
        self._terminal_after_request: set[tuple[str, int]] = set()
        self._restore_expected = restore_expected
        self._served_stateful_traffic = False
        self._untracked_stateful_traffic = False
        self._accepting = not restore_expected
        self._prepare_lock = asyncio.Lock()

    @asynccontextmanager
    async def mutation_lock(
        self,
        rollout_id: str,
        attempt_index: int,
        *,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[None]:
        """Hold one session lock and prune it after the final user releases it."""
        key = (rollout_id, attempt_index)
        entry = self._locks.setdefault(key, _LockEntry())
        entry.users += 1
        acquired = False
        try:
            if timeout is None:
                await entry.lock.acquire()
            else:
                await asyncio.wait_for(entry.lock.acquire(), timeout=timeout)
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            entry.users -= 1
            self._prune_lock(key, entry)

    def _prune_lock(self, key: tuple[str, int], entry: Optional[_LockEntry] = None) -> None:
        entry = entry or self._locks.get(key)
        if (
            entry is not None
            and entry.users == 0
            and not entry.lock.locked()
            and key not in self._revisions
            and self._locks.get(key) is entry
        ):
            self._locks.pop(key, None)

    def bind(self, rollout_id: str, attempt_index: int) -> None:
        """Bind a successfully created logical session to its execution."""
        self._served_stateful_traffic = True
        self._revisions.setdefault((rollout_id, attempt_index), 0)

    def register(self, rollout_id: str, attempt_index: int) -> None:
        """Compatibility alias for tests and adapters that predate explicit binding."""
        self.bind(rollout_id, attempt_index)

    def is_bound(self, rollout_id: str, attempt_index: int) -> bool:
        return (rollout_id, attempt_index) in self._revisions

    def revision_for(self, rollout_id: str, attempt_index: int) -> Optional[int]:
        return self._revisions.get((rollout_id, attempt_index))

    def record_untracked_stateful_traffic(self) -> None:
        """Remember legacy traffic that cannot be included in a safe checkpoint."""
        self._served_stateful_traffic = True
        self._untracked_stateful_traffic = True

    def is_tombstoned(self, rollout_id: str, attempt_index: int) -> bool:
        return (rollout_id, attempt_index) in self._tombstones

    def mark_terminal_after_request(self, rollout_id: str, attempt_index: int) -> None:
        self._terminal_after_request.add((rollout_id, attempt_index))

    def terminal_after_request(self, rollout_id: str, attempt_index: int) -> bool:
        return (rollout_id, attempt_index) in self._terminal_after_request

    def retire(self, rollout_id: str, attempt_index: int) -> None:
        key = (rollout_id, attempt_index)
        self._revisions.pop(key, None)
        self._terminal_after_request.discard(key)
        self._prune_lock(key)

    async def retire_execution(
        self,
        rollout_id: str,
        attempt_index: int,
        *,
        deadline_ts: Optional[float] = None,
    ) -> dict[str, Any]:
        """Fence and remove one sacrificed execution before checkpoint prepare."""
        key = (rollout_id, attempt_index)
        self._tombstones.add(key)
        timeout = None if deadline_ts is None else max(0.0, deadline_ts - time.time())
        try:
            async with self.mutation_lock(rollout_id, attempt_index, timeout=timeout):
                if self._retire_state is not None:
                    await self._retire_state(rollout_id, attempt_index)
                self.retire(rollout_id, attempt_index)
                self._prepared = [
                    snapshot for snapshot in self._prepared if (snapshot.rollout_id, snapshot.attempt_index) != key
                ]
        except asyncio.TimeoutError as error:
            raise ResourcesCheckpointError(
                f"timed out retiring resources state for rollout {rollout_id!r} attempt {attempt_index}"
            ) from error
        return {"retired": True, "rollout_id": rollout_id, "attempt_index": attempt_index}

    def record_mutation(self, rollout_id: str, attempt_index: int) -> int:
        key = (rollout_id, attempt_index)
        revision = self._revisions.get(key, 0) + 1
        if key in self._terminal_after_request:
            self._terminal_after_request.remove(key)
            self._revisions.pop(key, None)
        else:
            self._revisions[key] = revision
        return revision

    @property
    def accepting(self) -> bool:
        return self._accepting

    async def prepare(self, deadline_ts: float) -> dict[str, Any]:
        async with self._prepare_lock:
            self._accepting = False
            if self._untracked_stateful_traffic:
                raise ResourcesCheckpointError(
                    "stateful traffic without execution identity was served; refusing to omit live state"
                )
            snapshots: list[ResourceSnapshot] = []
            for rollout_id, attempt_index in sorted(self._revisions):
                remaining = deadline_ts - time.time()
                if remaining <= 0:
                    raise ResourcesCheckpointError("deadline expired before all resources sessions were exported")
                try:
                    async with self.mutation_lock(rollout_id, attempt_index, timeout=remaining):
                        key = (rollout_id, attempt_index)
                        revision = self._revisions.get(key)
                        # A terminal request may retire the session while prepare waits for its lock.
                        # A retired session is no longer part of the cut.
                        if revision is None:
                            continue
                        state = await self._export_state(rollout_id, attempt_index)
                except asyncio.TimeoutError as error:
                    raise ResourcesCheckpointError(
                        f"timed out draining resources state for rollout {rollout_id!r} attempt {attempt_index}"
                    ) from error
                snapshots.append(
                    ResourceSnapshot(
                        rollout_id=rollout_id,
                        attempt_index=attempt_index,
                        state_revision=revision,
                        state=state,
                    )
                )
            self._prepared = snapshots
            return {"sessions": len(snapshots), "state": "prepared"}

    async def restore(self, snapshots: list[ResourceSnapshot]) -> None:
        self._accepting = False
        if self._served_stateful_traffic or self._revisions:
            raise ResourcesUnsafeRestoreError(
                "resources restore requires a fresh process that has not served stateful traffic"
            )
        replacements = [
            snapshot.model_copy(update={"attempt_index": snapshot.attempt_index + 1}) for snapshot in snapshots
        ]
        # The environment validates and activates the complete replacement set
        # as one operation. A per-session restore loop could expose a mixed cut.
        await self._restore_states(replacements)
        self._tombstones.update((snapshot.rollout_id, snapshot.attempt_index) for snapshot in snapshots)
        for snapshot in replacements:
            key = (snapshot.rollout_id, snapshot.attempt_index)
            self._revisions[key] = snapshot.state_revision
        self._restore_expected = False

    def resume(self) -> dict[str, Any]:
        if self._restore_expected:
            raise ResourcesCheckpointError("resources process was started restore-expected and has not restored state")
        self._accepting = True
        self._prepared = []
        return {"state": "accepting"}

    def prepared_snapshots(self) -> list[ResourceSnapshot]:
        if self._accepting:
            raise ResourcesCheckpointError("resources state must be prepared before commit")
        return list(self._prepared)

    def status(self) -> dict[str, Any]:
        admission_state = "accepting" if self._accepting else "paused"
        return {
            "state": admission_state,
            "admission_state": admission_state,
            "sessions": len(self._revisions),
            "per_session": [
                {
                    "rollout_id": rollout_id,
                    "attempt_index": attempt_index,
                    "revision": revision,
                    "locked": bool((entry := self._locks.get((rollout_id, attempt_index))) and entry.lock.locked()),
                }
                for (rollout_id, attempt_index), revision in sorted(self._revisions.items())
            ],
            "lock_entries": len(self._locks),
            "locked_sessions": sum(entry.lock.locked() for entry in self._locks.values()),
            "tombstones": len(self._tombstones),
        }


class ResourcesSessionMiddleware:
    """Fence and serialize state mutations for one rollout attempt."""

    def __init__(
        self,
        app: Any,
        participant: ResourcesCheckpointParticipant,
        route_kind: Optional[Callable[[str, str], Optional[ResourcesRouteKind]]] = None,
    ) -> None:
        self._app = app
        self._participant = participant
        self._route_kind = route_kind or self._default_route_kind

    @staticmethod
    def _default_route_kind(path: str, method: str) -> Optional[ResourcesRouteKind]:
        if method != "POST":
            return None
        return "terminal" if path == "/verify" else "mutation"

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        kind = self._route_kind(path, scope.get("method", "GET"))
        if kind is None:
            await self._app(scope, receive, send)
            return
        try:
            identity = self._identity(scope)
        except ValueError:
            await self._reject_identity(send)
            return
        if identity is None:
            if not self._participant.accepting:
                await self._reject(send)
                return

            async def send_untracked(message: dict[str, Any]) -> None:
                if (
                    kind in {"start", "mutation"}
                    and message.get("type") == "http.response.start"
                    and int(message.get("status", 500)) < 400
                ):
                    self._participant.record_untracked_stateful_traffic()
                await send(message)

            await self._app(scope, receive, send_untracked)
            return
        rollout_id, attempt_index = identity
        if self._participant.is_tombstoned(rollout_id, attempt_index):
            await self._reject_stale(send, rollout_id, attempt_index)
            return
        if not self._participant.accepting:
            await self._reject(send)
            return
        if kind != "start" and not self._participant.is_bound(rollout_id, attempt_index):
            await self._reject_unbound(send, rollout_id, attempt_index)
            return
        async with self._participant.mutation_lock(rollout_id, attempt_index):
            # Re-check after waiting: prepare may have closed admission while
            # this request waited behind an earlier mutation.
            if not self._participant.accepting:
                await self._reject(send)
                return
            if self._participant.is_tombstoned(rollout_id, attempt_index):
                await self._reject_stale(send, rollout_id, attempt_index)
                return
            if kind != "start" and not self._participant.is_bound(rollout_id, attempt_index):
                await self._reject_unbound(send, rollout_id, attempt_index)
                return
            successful_terminal_response = False

            async def send_with_revision(message: dict[str, Any]) -> None:
                nonlocal successful_terminal_response
                if message.get("type") == "http.response.start" and int(message.get("status", 500)) < 400:
                    if kind == "start":
                        self._participant.bind(rollout_id, attempt_index)
                    if kind == "terminal":
                        successful_terminal_response = True
                        self._participant.mark_terminal_after_request(rollout_id, attempt_index)
                    if kind != "read":
                        revision = self._participant.record_mutation(rollout_id, attempt_index)
                        headers = list(message.get("headers") or ())
                        headers.append(
                            (
                                RESOURCE_STATE_REVISION_HEADER.encode("ascii"),
                                str(revision).encode("ascii"),
                            )
                        )
                        message = {**message, "headers": headers}
                await send(message)

            try:
                await self._app(scope, receive, send_with_revision)
            finally:
                if kind == "terminal" and (
                    successful_terminal_response or self._participant.terminal_after_request(rollout_id, attempt_index)
                ):
                    self._participant.retire(rollout_id, attempt_index)

    @staticmethod
    def _identity(scope: dict[str, Any]) -> Optional[tuple[str, int]]:
        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        raw_rollout = headers.get(ROLLOUT_ID_HEADER.encode())
        raw_attempt = headers.get(ATTEMPT_INDEX_HEADER.encode())
        if raw_rollout is None and raw_attempt is None:
            rollout_id, attempt_index = current_execution_identity()
            if rollout_id is None and attempt_index is None:
                return None
            if rollout_id is None or attempt_index is None:
                raise ValueError("partial execution identity")
            return rollout_id, attempt_index
        if raw_rollout is None or raw_attempt is None:
            raise ValueError("partial execution identity")
        try:
            rollout_id = raw_rollout.decode("ascii")
            attempt_index = int(raw_attempt.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            raise ValueError("malformed execution identity") from None
        if not ROLLOUT_ID_PATTERN.fullmatch(rollout_id) or attempt_index < 0:
            raise ValueError("malformed execution identity")
        return rollout_id, attempt_index

    @staticmethod
    async def _reject(send: Any) -> None:
        payload = json.dumps(
            {
                "error": {
                    "code": ResourcesAdmissionClosedError.code,
                    "message": "resources mutation admission is closed for checkpoint preparation",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    @staticmethod
    async def _reject_identity(send: Any) -> None:
        payload = b'{"error":{"code":"execution_identity_mismatch","message":"invalid execution identity headers"}}'
        await send(
            {
                "type": "http.response.start",
                "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    @staticmethod
    async def _reject_unbound(send: Any, rollout_id: str, attempt_index: int) -> None:
        payload = json.dumps(
            {
                "error": {
                    "code": ResourcesSessionUnboundError.code,
                    "message": f"rollout {rollout_id!r} attempt {attempt_index} has no successful seed/reset binding",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    @staticmethod
    async def _reject_stale(send: Any, rollout_id: str, attempt_index: int) -> None:
        payload = json.dumps(
            {
                "error": {
                    "code": ResourcesStaleAttemptError.code,
                    "message": f"rollout {rollout_id!r} attempt {attempt_index} was retired by restore",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def commit_resources_state(
    participant: ResourcesCheckpointParticipant,
    checkpoint_dir: Path,
    *,
    checkpoint_id: str,
    server_name: str,
) -> dict[str, Any]:
    directory = Path(checkpoint_dir) / RESOURCES_STATE_SUBDIR / server_name
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / RESOURCES_MANIFEST_NAME
    if manifest_path.exists():
        return _validate_resources_manifest(directory, checkpoint_id=checkpoint_id, server_name=server_name)
    files: dict[str, str] = {}
    for snapshot in participant.prepared_snapshots():
        name = f"{snapshot.rollout_id}.a{snapshot.attempt_index}.json"
        path = directory / name
        _write_atomic(path, snapshot.model_dump_json(indent=2).encode())
        files[name] = _digest(path)
    _fsync_dir(directory)
    manifest = {
        "schema_version": RESOURCES_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "server_name": server_name,
        "files": files,
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2).encode()
    _write_atomic(manifest_path, payload)
    _fsync_dir(directory)
    return {"sessions": len(files), "manifest_digest": hashlib.sha256(payload).hexdigest()}


def _validate_resources_manifest(directory: Path, *, checkpoint_id: str, server_name: str) -> dict[str, Any]:
    manifest_path = directory / RESOURCES_MANIFEST_NAME
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("checkpoint_id") != checkpoint_id or manifest.get("server_name") != server_name:
        raise ResourcesCheckpointError("resources checkpoint manifest belongs to a different transaction or server")
    for name, digest in manifest.get("files", {}).items():
        path = directory / name
        if not path.exists() or _digest(path) != digest:
            raise ResourcesCheckpointError(f"resources checkpoint state {name!r} is missing or corrupted")
    return {
        "sessions": len(manifest.get("files", {})),
        "manifest_digest": hashlib.sha256(payload).hexdigest(),
    }


def load_resources_state(checkpoint_dir: Path, *, server_name: str) -> tuple[str, list[ResourceSnapshot]]:
    directory = Path(checkpoint_dir) / RESOURCES_STATE_SUBDIR / server_name
    manifest_path = directory / RESOURCES_MANIFEST_NAME
    if not manifest_path.exists():
        raise ResourcesCheckpointError(f"resources checkpoint has no committed manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    snapshots: list[ResourceSnapshot] = []
    for name, digest in manifest["files"].items():
        path = directory / name
        if not path.exists() or _digest(path) != digest:
            raise ResourcesCheckpointError(f"resources checkpoint state {name!r} is missing or corrupted")
        snapshots.append(ResourceSnapshot.model_validate_json(path.read_bytes()))
    return manifest["checkpoint_id"], snapshots


def install_resources_checkpoint(
    app: FastAPI,
    *,
    participant: ResourcesCheckpointParticipant,
    fence: ControlFence,
    auth_token: str,
    server_name: str,
    route_kind: Callable[[str, str], Optional[ResourcesRouteKind]],
) -> None:
    """Install bulk prepare, commit, restore, and resume routes."""

    @app.post(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/prepare")
    async def prepare(
        body: ResourcesPrepareRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            return await participant.prepare(body.deadline_ts)

        result = await fence.run_operation(
            body.checkpoint_id,
            "resources-checkpoint/prepare",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.PREPARING,
            phase_after=CheckpointPhase.PREPARED,
            run=run,
            deadline=body,
        )
        return result

    @app.post(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/commit")
    async def commit(
        body: ResourcesCommitRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            return await asyncio.to_thread(
                commit_resources_state,
                participant,
                Path(body.checkpoint_dir),
                checkpoint_id=body.checkpoint_id,
                server_name=server_name,
            )

        return await fence.run_operation(
            body.checkpoint_id,
            "resources-checkpoint/commit",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.COMMITTING,
            phase_after=CheckpointPhase.COMMITTED_PAUSED,
            run=run,
        )

    @app.post(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/restore")
    async def restore(
        body: ResourcesRestoreRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            source_checkpoint_id, snapshots = await asyncio.to_thread(
                load_resources_state,
                Path(body.checkpoint_dir),
                server_name=server_name,
            )
            await participant.restore(snapshots)
            return {"sessions": len(snapshots), "source_checkpoint_id": source_checkpoint_id}

        return await fence.run_operation(
            body.checkpoint_id,
            "resources-checkpoint/restore",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.RESTORING,
            phase_after=CheckpointPhase.RESTORED_PAUSED,
            run=run,
        )

    @app.post(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/resume")
    async def resume(
        body: ResourcesResumeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            return participant.resume()

        return await fence.run_operation(
            body.checkpoint_id,
            "resources-checkpoint/resume",
            allowed_phases=frozenset(
                {
                    CheckpointPhase.IDLE,
                    CheckpointPhase.PREPARED,
                    CheckpointPhase.COMMITTED_PAUSED,
                    CheckpointPhase.RESTORED_PAUSED,
                }
            ),
            phase_during=fence.phase,
            phase_after=CheckpointPhase.IDLE,
            run=run,
            retire_outcome="resumed",
        )

    @app.post(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/retire")
    async def retire(
        body: ResourcesRetireRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        fence.require_phase(
            body.checkpoint_id,
            frozenset({CheckpointPhase.PREPARING, CheckpointPhase.PREPARED}),
        )
        return await participant.retire_execution(
            body.rollout_id,
            body.attempt_index,
            deadline_ts=body.deadline_ts,
        )

    @app.get(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/status")
    async def status(
        checkpoint_id: str = Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
        deadline_ts: float = Query(),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        fence.require_phase(
            checkpoint_id,
            frozenset(
                {
                    CheckpointPhase.PREPARING,
                    CheckpointPhase.PREPARED,
                    CheckpointPhase.COMMITTING,
                    CheckpointPhase.COMMITTED_PAUSED,
                    CheckpointPhase.RESTORING,
                    CheckpointPhase.RESTORED_PAUSED,
                }
            ),
        )
        Deadline(deadline_ts=deadline_ts)
        return {"checkpoint_id": checkpoint_id, **participant.status()}

    app.add_middleware(ResourcesSessionMiddleware, participant=participant, route_kind=route_kind)
