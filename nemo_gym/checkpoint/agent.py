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
"""Shared checkpoint participant for whitebox agent servers."""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, Header
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.checkpoint.control import CheckpointControlRequest, CheckpointPhase, ControlError, ControlFence
from nemo_gym.rollout_correlation import ROLLOUT_ID_PATTERN
from nemo_gym.token_id_capture.control_routes import require_control_auth


AGENT_CHECKPOINT_URL_PREFIX = "/ng-control/v1/agent-checkpoint"
AGENT_STATE_SUBDIR = "agent"
AGENT_MANIFEST_NAME = "manifest.json"
AGENT_CHECKPOINT_SCHEMA_VERSION = 1


class AgentExecutionState(str, Enum):
    RUNNING = "running"
    PARK_REQUESTED = "park_requested"
    PARKED = "parked"
    COMPLETED = "completed"
    RETIRED = "retired"


class AgentCheckpointError(ControlError):
    code = "agent_checkpoint_error"


class DuplicateExecutionError(ControlError):
    code = "duplicate_execution"


class AgentAdmissionClosedError(ControlError):
    code = "agent_admission_closed"


class AgentBoundaryRecord(BaseModel):
    """Continuation state after one complete whitebox agent turn."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[AGENT_CHECKPOINT_SCHEMA_VERSION] = AGENT_CHECKPOINT_SCHEMA_VERSION
    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)
    boundary_index: int = Field(ge=0)
    output_items: list[dict[str, Any]]
    usage: Optional[dict[str, Any]] = None
    last_committed_model_call_id: Optional[str] = None
    resource_state_revisions: dict[str, int] = Field(default_factory=dict)
    agent_state: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class AgentExecution:
    """One active `/run` invocation and its latest committed boundary."""

    def __init__(self, rollout_id: str, attempt_index: int, task: Optional[asyncio.Task]) -> None:
        self.rollout_id = rollout_id
        self.attempt_index = attempt_index
        self.task = task
        self.state = AgentExecutionState.RUNNING
        self.boundary: Optional[AgentBoundaryRecord] = None
        self.resume_event = asyncio.Event()
        self.resume_event.set()


class AgentPrepareRequest(CheckpointControlRequest):
    pass


class AgentCommitRequest(CheckpointControlRequest):
    checkpoint_dir: str


class AgentRestoreRequest(CheckpointControlRequest):
    checkpoint_dir: str


class AgentResumeRequest(CheckpointControlRequest):
    pass


class AgentRetireRequest(CheckpointControlRequest):
    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)


class AgentCheckpointParticipant:
    """Own active whitebox executions and park them at committed boundaries."""

    def __init__(self) -> None:
        self._executions: dict[tuple[str, int], AgentExecution] = {}
        self._restored: dict[tuple[str, int], AgentBoundaryRecord] = {}
        self._accepting = True
        self._changed = asyncio.Condition()

    async def begin(
        self,
        rollout_id: str,
        attempt_index: int,
        *,
        task: Optional[asyncio.Task],
    ) -> AgentExecution:
        if not self._accepting:
            raise AgentAdmissionClosedError("agent admission is closed for checkpoint preparation")
        key = (rollout_id, attempt_index)
        existing = self._executions.get(key)
        if existing is not None and existing.state not in {
            AgentExecutionState.COMPLETED,
            AgentExecutionState.RETIRED,
        }:
            raise DuplicateExecutionError(f"rollout {rollout_id!r} attempt {attempt_index} already has an active /run")
        execution = AgentExecution(rollout_id, attempt_index, task)
        self._executions[key] = execution
        await self._notify()
        return execution

    async def finish(self, execution: AgentExecution) -> None:
        if execution.state != AgentExecutionState.RETIRED:
            execution.state = AgentExecutionState.COMPLETED
        await self._notify()

    def continuation(self, rollout_id: str, attempt_index: int) -> Optional[AgentBoundaryRecord]:
        return self._restored.get((rollout_id, attempt_index))

    async def commit_boundary(self, record: AgentBoundaryRecord) -> None:
        key = (record.rollout_id, record.attempt_index)
        execution = self._executions.get(key)
        if execution is None:
            raise AgentCheckpointError(
                f"no active /run for rollout {record.rollout_id!r} attempt {record.attempt_index}"
            )
        previous = execution.boundary
        if previous is not None and record.boundary_index <= previous.boundary_index:
            if record == previous:
                return
            raise AgentCheckpointError(
                f"boundary indices must increase for rollout {record.rollout_id!r} attempt {record.attempt_index}"
            )
        execution.boundary = record
        if execution.state == AgentExecutionState.PARK_REQUESTED:
            execution.state = AgentExecutionState.PARKED
            execution.resume_event.clear()
            await self._notify()
            await execution.resume_event.wait()
            if execution.state != AgentExecutionState.RETIRED:
                execution.state = AgentExecutionState.RUNNING
        await self._notify()

    async def prepare(self, deadline_ts: float) -> dict[str, Any]:
        self._accepting = False
        for execution in self._executions.values():
            if execution.state == AgentExecutionState.RUNNING:
                execution.state = AgentExecutionState.PARK_REQUESTED
        await self._notify()
        return await self._wait_prepared(deadline_ts)

    async def _wait_prepared(self, deadline_ts: float) -> dict[str, Any]:
        async with self._changed:
            while True:
                report = self.status()
                if report["running"] == 0:
                    return report
                remaining = deadline_ts - time.time()
                if remaining <= 0:
                    return report
                try:
                    await asyncio.wait_for(self._changed.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return self.status()

    async def resume(self) -> dict[str, Any]:
        self._accepting = True
        released = 0
        for execution in self._executions.values():
            if execution.state == AgentExecutionState.PARKED:
                execution.resume_event.set()
                released += 1
        await self._notify()
        return {"state": "accepting", "released": released}

    async def retire(self, rollout_id: str, attempt_index: int) -> dict[str, Any]:
        execution = self._executions.get((rollout_id, attempt_index))
        if execution is None:
            return {"retired": False}
        execution.state = AgentExecutionState.RETIRED
        execution.resume_event.set()
        if execution.task is not None:
            execution.task.cancel()
        await self._notify()
        return {"retired": True}

    def status(self) -> dict[str, Any]:
        active = [
            execution
            for execution in self._executions.values()
            if execution.state not in {AgentExecutionState.COMPLETED, AgentExecutionState.RETIRED}
        ]
        return {
            "state": "accepting" if self._accepting else "preparing",
            "running": sum(
                execution.state in {AgentExecutionState.RUNNING, AgentExecutionState.PARK_REQUESTED}
                for execution in active
            ),
            "parked": sum(execution.state == AgentExecutionState.PARKED for execution in active),
            "active": len(active),
            "executions": [
                {
                    "rollout_id": execution.rollout_id,
                    "attempt_index": execution.attempt_index,
                    "state": execution.state.value,
                    "boundary_index": execution.boundary.boundary_index if execution.boundary is not None else None,
                }
                for execution in active
            ],
        }

    def records_for_commit(self) -> list[AgentBoundaryRecord]:
        report = self.status()
        if report["running"]:
            raise AgentCheckpointError("cannot commit while an agent /run is between committed boundaries")
        return [
            execution.boundary
            for execution in self._executions.values()
            if execution.state == AgentExecutionState.PARKED and execution.boundary is not None
        ]

    def install_restored(self, records: list[AgentBoundaryRecord]) -> None:
        for record in records:
            self._restored[(record.rollout_id, record.attempt_index + 1)] = record
        self._accepting = False

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def commit_agent_state(
    participant: AgentCheckpointParticipant,
    checkpoint_dir: Path,
    *,
    checkpoint_id: str,
) -> dict[str, Any]:
    directory = Path(checkpoint_dir) / AGENT_STATE_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / AGENT_MANIFEST_NAME
    if manifest_path.exists():
        raise AgentCheckpointError(f"agent checkpoint already committed at {directory}")

    files: dict[str, str] = {}
    for record in participant.records_for_commit():
        name = f"{record.rollout_id}.a{record.attempt_index}.json"
        target = directory / name
        payload = record.model_dump_json(indent=2).encode()
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".agent-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        files[name] = _digest(target)
    _fsync_dir(directory)

    manifest = {
        "schema_version": AGENT_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "files": files,
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2).encode()
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".manifest-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)
    _fsync_dir(directory)
    return {"records": len(files), "manifest_digest": hashlib.sha256(payload).hexdigest()}


def restore_agent_state(participant: AgentCheckpointParticipant, checkpoint_dir: Path) -> dict[str, Any]:
    directory = Path(checkpoint_dir) / AGENT_STATE_SUBDIR
    manifest_path = directory / AGENT_MANIFEST_NAME
    if not manifest_path.exists():
        raise AgentCheckpointError(f"agent checkpoint has no committed manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    records: list[AgentBoundaryRecord] = []
    for name, digest in manifest["files"].items():
        path = directory / name
        if not path.exists() or _digest(path) != digest:
            raise AgentCheckpointError(f"agent checkpoint record {name!r} is missing or corrupted")
        records.append(AgentBoundaryRecord.model_validate_json(path.read_bytes()))
    participant.install_restored(records)
    return {"records": len(records), "source_checkpoint_id": manifest["checkpoint_id"]}


def install_agent_checkpoint(
    app: FastAPI,
    *,
    participant: AgentCheckpointParticipant,
    fence: ControlFence,
    auth_token: str,
) -> None:
    """Install bulk prepare, commit, restore, resume, and retire routes."""

    @app.post(f"{AGENT_CHECKPOINT_URL_PREFIX}/prepare")
    async def prepare(
        body: AgentPrepareRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            return await participant.prepare(body.deadline_ts)

        result = await fence.run_operation(
            body.checkpoint_id,
            "agent-checkpoint/prepare",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.PREPARING,
            phase_after=CheckpointPhase.PREPARING,
            run=run,
            deadline=body,
        )
        if result["running"] == 0:
            fence.mark_prepared(body.checkpoint_id)
        return result

    @app.post(f"{AGENT_CHECKPOINT_URL_PREFIX}/commit")
    async def commit(
        body: AgentCommitRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            return await asyncio.to_thread(
                commit_agent_state,
                participant,
                Path(body.checkpoint_dir),
                checkpoint_id=body.checkpoint_id,
            )

        return await fence.run_operation(
            body.checkpoint_id,
            "agent-checkpoint/commit",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.COMMITTING,
            phase_after=CheckpointPhase.COMMITTED_PAUSED,
            run=run,
        )

    @app.post(f"{AGENT_CHECKPOINT_URL_PREFIX}/restore")
    async def restore(
        body: AgentRestoreRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            return await asyncio.to_thread(restore_agent_state, participant, Path(body.checkpoint_dir))

        return await fence.run_operation(
            body.checkpoint_id,
            "agent-checkpoint/restore",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.RESTORING,
            phase_after=CheckpointPhase.RESTORED_PAUSED,
            run=run,
        )

    @app.post(f"{AGENT_CHECKPOINT_URL_PREFIX}/resume")
    async def resume(
        body: AgentResumeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            return await participant.resume()

        return await fence.run_operation(
            body.checkpoint_id,
            "agent-checkpoint/resume",
            allowed_phases=frozenset(
                {
                    CheckpointPhase.PREPARING,
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

    @app.post(f"{AGENT_CHECKPOINT_URL_PREFIX}/retire")
    async def retire(
        body: AgentRetireRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        fence.require_phase(
            body.checkpoint_id,
            frozenset({CheckpointPhase.PREPARING, CheckpointPhase.PREPARED}),
        )
        return await participant.retire(body.rollout_id, body.attempt_index)
