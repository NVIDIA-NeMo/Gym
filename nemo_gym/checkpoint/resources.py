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
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, Header
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.checkpoint.control import CheckpointControlRequest, CheckpointPhase, ControlError, ControlFence
from nemo_gym.rollout_correlation import ATTEMPT_INDEX_HEADER, ROLLOUT_ID_HEADER, ROLLOUT_ID_PATTERN
from nemo_gym.token_id_capture.control_routes import require_control_auth


RESOURCES_CHECKPOINT_URL_PREFIX = "/ng-control/v1/resources-checkpoint"
RESOURCES_STATE_SUBDIR = "resources"
RESOURCES_MANIFEST_NAME = "manifest.json"
RESOURCES_CHECKPOINT_SCHEMA_VERSION = 1
RESOURCE_STATE_REVISION_HEADER = "x-nemo-gym-resource-state-revision"

_NONMUTATING_PATHS = frozenset({"/verify", "/aggregate_metrics", "/reverify_mode"})


class ResourcesCheckpointError(ControlError):
    code = "resources_checkpoint_error"


class ResourcesAdmissionClosedError(ControlError):
    code = "resources_admission_closed"


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


class ResourcesCheckpointParticipant:
    """Serialize mutations per execution and export all known live sessions."""

    def __init__(
        self,
        *,
        export_state: Callable[[str, int], Awaitable[dict[str, Any]]],
        restore_states: Callable[[list[ResourceSnapshot]], Awaitable[None]],
    ) -> None:
        self._export_state = export_state
        self._restore_states = restore_states
        self._locks: dict[tuple[str, int], asyncio.Lock] = {}
        self._revisions: dict[tuple[str, int], int] = {}
        self._prepared: list[ResourceSnapshot] = []
        self._accepting = True
        self._prepare_lock = asyncio.Lock()

    def lock_for(self, rollout_id: str, attempt_index: int) -> asyncio.Lock:
        return self._locks.setdefault((rollout_id, attempt_index), asyncio.Lock())

    def register(self, rollout_id: str, attempt_index: int) -> None:
        self._revisions.setdefault((rollout_id, attempt_index), 0)

    def record_mutation(self, rollout_id: str, attempt_index: int) -> int:
        key = (rollout_id, attempt_index)
        revision = self._revisions.get(key, 0) + 1
        self._revisions[key] = revision
        return revision

    @property
    def accepting(self) -> bool:
        return self._accepting

    async def prepare(self, deadline_ts: float) -> dict[str, Any]:
        async with self._prepare_lock:
            self._accepting = False
            snapshots: list[ResourceSnapshot] = []
            for (rollout_id, attempt_index), revision in sorted(self._revisions.items()):
                remaining = deadline_ts - time.time()
                if remaining <= 0:
                    raise ResourcesCheckpointError("deadline expired before all resources sessions were exported")
                lock = self.lock_for(rollout_id, attempt_index)
                try:
                    await asyncio.wait_for(lock.acquire(), timeout=remaining)
                except asyncio.TimeoutError as error:
                    raise ResourcesCheckpointError(
                        f"timed out draining resources state for rollout {rollout_id!r} attempt {attempt_index}"
                    ) from error
                try:
                    state = await self._export_state(rollout_id, attempt_index)
                finally:
                    lock.release()
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
        replacements = [
            snapshot.model_copy(update={"attempt_index": snapshot.attempt_index + 1}) for snapshot in snapshots
        ]
        # The environment validates and activates the complete replacement set
        # as one operation. A per-session restore loop could expose a mixed cut.
        await self._restore_states(replacements)
        for snapshot in replacements:
            key = (snapshot.rollout_id, snapshot.attempt_index)
            self._revisions[key] = snapshot.state_revision
            self._locks.setdefault(key, asyncio.Lock())

    def resume(self) -> dict[str, Any]:
        self._accepting = True
        self._prepared = []
        return {"state": "accepting"}

    def prepared_snapshots(self) -> list[ResourceSnapshot]:
        if self._accepting:
            raise ResourcesCheckpointError("resources state must be prepared before commit")
        return list(self._prepared)

    def status(self) -> dict[str, Any]:
        return {
            "state": "accepting" if self._accepting else "paused",
            "sessions": len(self._revisions),
            "locked_sessions": sum(lock.locked() for lock in self._locks.values()),
        }


class ResourcesSessionMiddleware:
    """Fence and serialize state mutations for one rollout attempt."""

    def __init__(self, app: Any, participant: ResourcesCheckpointParticipant) -> None:
        self._app = app
        self._participant = participant

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not self._is_mutation(scope.get("path", "")):
            await self._app(scope, receive, send)
            return
        try:
            identity = self._identity(scope)
        except ValueError:
            await self._reject_identity(send)
            return
        if identity is None:
            await self._reject_identity(send)
            return
        rollout_id, attempt_index = identity
        if not self._participant.accepting:
            await self._reject(send)
            return
        lock = self._participant.lock_for(rollout_id, attempt_index)
        async with lock:
            # Re-check after waiting: prepare may have closed admission while
            # this request waited behind an earlier mutation.
            if not self._participant.accepting:
                await self._reject(send)
                return
            self._participant.register(rollout_id, attempt_index)
            revision: Optional[int] = None

            async def send_with_revision(message: dict[str, Any]) -> None:
                nonlocal revision
                if message.get("type") == "http.response.start" and int(message.get("status", 500)) < 400:
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

            await self._app(scope, receive, send_with_revision)

    @staticmethod
    def _is_mutation(path: str) -> bool:
        return not path.startswith("/ng-control/") and path not in _NONMUTATING_PATHS

    @staticmethod
    def _identity(scope: dict[str, Any]) -> Optional[tuple[str, int]]:
        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        raw_rollout = headers.get(ROLLOUT_ID_HEADER.encode())
        raw_attempt = headers.get(ATTEMPT_INDEX_HEADER.encode())
        if raw_rollout is None and raw_attempt is None:
            return None
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
        raise ResourcesCheckpointError(f"resources checkpoint already committed at {directory}")
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

    app.add_middleware(ResourcesSessionMiddleware, participant=participant)
