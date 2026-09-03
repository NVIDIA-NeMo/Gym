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
from contextvars import ContextVar, Token
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, Header, Query
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym._checkpoint.control import CheckpointControlRequest, CheckpointPhase, ControlError, ControlFence
from nemo_gym.rollout_correlation import ROLLOUT_ID_PATTERN
from nemo_gym.token_id_capture.control_routes import require_control_auth


AGENT_CHECKPOINT_URL_PREFIX = "/ng-control/v1/agent-checkpoint"
AGENT_STATE_SUBDIR = "agent"
AGENT_MANIFEST_NAME = "manifest.json"
AGENT_CHECKPOINT_SCHEMA_VERSION = 1
AGENT_EXECUTION_GENERATION_HEADER = "x-nemo-gym-agent-execution-generation"

_CURRENT_AGENT_EXECUTION: ContextVar[Optional["AgentExecution"]] = ContextVar(
    "nemo_gym_current_agent_execution",
    default=None,
)


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


class AgentStaleAttemptError(ControlError):
    code = "stale_attempt"


class AgentPrepareIncompleteError(ControlError):
    code = "agent_prepare_incomplete"


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

    def __init__(
        self,
        rollout_id: str,
        attempt_index: int,
        generation: int,
        task: Optional[asyncio.Task],
        continuation: Optional[AgentBoundaryRecord],
    ) -> None:
        self.rollout_id = rollout_id
        self.attempt_index = attempt_index
        self.generation = generation
        self.outer_task = task
        self.parked_task: Optional[asyncio.Task] = None
        self.state = AgentExecutionState.RUNNING
        self.boundary: Optional[AgentBoundaryRecord] = None
        self.continuation = continuation
        self.terminal_result: Any = None
        self.started_at = time.time()
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
    """Own active whitebox executions and park them at committed boundaries.

    Successful terminal results remain replayable until a shared durable
    acknowledgement layer releases them. Prepare reports those results as
    ``completed_unacknowledged`` and must not be treated as publishable while
    that count is nonzero.
    """

    def __init__(self, instance_name: Optional[str] = None) -> None:
        self.instance_name = _validate_instance_name(instance_name)
        self._executions: dict[tuple[str, int], AgentExecution] = {}
        self._generations: dict[tuple[str, int], int] = {}
        self._restored: dict[tuple[str, int], AgentBoundaryRecord] = {}
        self._tombstones: set[tuple[str, int]] = set()
        self._accepting = True
        self._changed = asyncio.Condition()

    async def begin(
        self,
        rollout_id: str,
        attempt_index: int,
        *,
        task: Optional[asyncio.Task],
    ) -> AgentExecution:
        key = (rollout_id, attempt_index)
        if key in self._tombstones:
            raise AgentStaleAttemptError(f"rollout {rollout_id!r} attempt {attempt_index} was retired by restore")
        existing = self._executions.get(key)
        if existing is not None:
            if existing.state == AgentExecutionState.COMPLETED and existing.terminal_result is not None:
                return existing
            raise DuplicateExecutionError(f"rollout {rollout_id!r} attempt {attempt_index} already has an active /run")
        if not self._accepting:
            raise AgentAdmissionClosedError("agent admission is closed for checkpoint preparation")
        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation
        execution = AgentExecution(
            rollout_id,
            attempt_index,
            generation,
            task,
            self._restored.pop(key, None),
        )
        self._executions[key] = execution
        await self._notify()
        return execution

    def bind(self, execution: AgentExecution) -> Token:
        return _CURRENT_AGENT_EXECUTION.set(execution)

    def unbind(self, token: Token) -> None:
        _CURRENT_AGENT_EXECUTION.reset(token)

    def current_execution(self) -> Optional[AgentExecution]:
        execution = _CURRENT_AGENT_EXECUTION.get()
        if execution is None or not self._owns(execution):
            return None
        return execution

    def resolve(
        self,
        rollout_id: str,
        attempt_index: int,
        *,
        generation: Optional[int] = None,
    ) -> Optional[AgentExecution]:
        execution = self._executions.get((rollout_id, attempt_index))
        if execution is None or (generation is not None and execution.generation != generation):
            return None
        return execution

    async def finish(
        self,
        execution: AgentExecution,
        *,
        outcome: Literal["completed", "failed", "cancelled"],
        result: Any = None,
    ) -> None:
        if not self._owns(execution):
            return
        if outcome == "cancelled" and execution.state == AgentExecutionState.PARKED and execution.boundary is not None:
            execution.outer_task = None
        elif execution.state != AgentExecutionState.RETIRED:
            if outcome == "completed":
                execution.state = AgentExecutionState.COMPLETED
                execution.terminal_result = result
                execution.continuation = None
                execution.outer_task = None
            else:
                execution.state = AgentExecutionState.RETIRED
                key = (execution.rollout_id, execution.attempt_index)
                self._tombstones.add(key)
                execution.resume_event.set()
                if execution.parked_task is not None and execution.parked_task is not asyncio.current_task():
                    execution.parked_task.cancel()
                execution.boundary = None
                execution.continuation = None
                execution.outer_task = None
                execution.parked_task = None
                self._executions.pop(key, None)
        await self._notify()

    def continuation(self, execution: AgentExecution) -> Optional[AgentBoundaryRecord]:
        if not self._owns(execution):
            raise AgentStaleAttemptError("agent execution was replaced before its continuation was consumed")
        return execution.continuation

    async def commit_boundary(self, execution: AgentExecution, record: AgentBoundaryRecord) -> None:
        self._require_owner(execution)
        if (record.rollout_id, record.attempt_index) != (execution.rollout_id, execution.attempt_index):
            raise AgentCheckpointError("boundary identity does not match its agent execution")
        previous = execution.boundary
        if previous is not None and record.boundary_index <= previous.boundary_index:
            if record == previous:
                return
            raise AgentCheckpointError(
                f"boundary indices must increase for rollout {record.rollout_id!r} attempt {record.attempt_index}"
            )
        execution.boundary = record
        if execution.state == AgentExecutionState.PARK_REQUESTED:
            await self.park(execution)
        await self._notify()

    async def park(self, execution: AgentExecution) -> None:
        self._require_owner(execution)
        if execution.state == AgentExecutionState.RETIRED:
            raise AgentStaleAttemptError("agent execution was retired")
        execution.state = AgentExecutionState.PARKED
        execution.parked_task = asyncio.current_task()
        execution.resume_event.clear()
        await self._notify()
        try:
            await execution.resume_event.wait()
        finally:
            if execution.parked_task is asyncio.current_task():
                execution.parked_task = None
        self._require_owner(execution)
        if execution.state == AgentExecutionState.RETIRED:
            raise AgentStaleAttemptError("agent execution was retired while parked")
        execution.state = AgentExecutionState.RUNNING
        await self._notify()

    async def prepare(self, deadline_ts: float) -> dict[str, Any]:
        """Park running work and expose every condition blocking publication."""
        self._accepting = False
        requested: list[AgentExecution] = []
        for execution in self._executions.values():
            if execution.state == AgentExecutionState.RUNNING:
                execution.state = AgentExecutionState.PARK_REQUESTED
                requested.append(execution)
        await self._notify()
        completed = False
        try:
            report = await self._wait_prepared(deadline_ts)
            completed = report["ready_to_commit"]
            return report
        finally:
            if not completed:
                for execution in requested:
                    if self._owns(execution) and execution.state == AgentExecutionState.PARK_REQUESTED:
                        execution.state = AgentExecutionState.RUNNING
                await self._notify()

    async def _wait_prepared(self, deadline_ts: float) -> dict[str, Any]:
        async with self._changed:
            while True:
                report = self.status()
                if report["ready_to_commit"] or report["running"] == 0:
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
            if execution.state == AgentExecutionState.PARK_REQUESTED:
                execution.state = AgentExecutionState.RUNNING
            elif execution.state == AgentExecutionState.PARKED:
                if execution.outer_task is None:
                    execution.state = AgentExecutionState.RETIRED
                    self._tombstones.add((execution.rollout_id, execution.attempt_index))
                    execution.resume_event.set()
                    if execution.parked_task is not None:
                        execution.parked_task.cancel()
                else:
                    execution.resume_event.set()
                    released += 1
        await self._notify()
        return {"state": "accepting", "released": released}

    async def retire(self, rollout_id: str, attempt_index: int) -> dict[str, Any]:
        key = (rollout_id, attempt_index)
        execution = self._executions.get(key)
        if execution is not None and execution.state == AgentExecutionState.COMPLETED:
            return {
                "retired": False,
                "tombstoned": False,
                "completed_unacknowledged": True,
            }
        self._tombstones.add(key)
        self._restored.pop(key, None)
        execution = self._executions.pop(key, None)
        if execution is None:
            await self._notify()
            return {"retired": False, "tombstoned": True}
        execution.state = AgentExecutionState.RETIRED
        execution.resume_event.set()
        tasks = {execution.outer_task, execution.parked_task}
        current = asyncio.current_task()
        for task in tasks:
            if task is not None and task is not current:
                task.cancel()
        await self._notify()
        return {"retired": True, "tombstoned": True}

    def status(self) -> dict[str, Any]:
        all_executions = list(self._executions.values())
        active = [
            execution
            for execution in all_executions
            if execution.state not in {AgentExecutionState.COMPLETED, AgentExecutionState.RETIRED}
        ]
        parked_with_boundary = [
            execution
            for execution in active
            if execution.state == AgentExecutionState.PARKED and execution.boundary is not None
        ]
        parked_without_boundary = [
            execution
            for execution in active
            if execution.state == AgentExecutionState.PARKED and execution.boundary is None
        ]
        completed_unacknowledged = [
            execution for execution in all_executions if execution.state == AgentExecutionState.COMPLETED
        ]
        blocking_attempts = [
            execution
            for execution in active
            if execution.state in {AgentExecutionState.RUNNING, AgentExecutionState.PARK_REQUESTED}
        ]
        return {
            "state": "accepting" if self._accepting else "preparing",
            "ready_to_commit": not blocking_attempts and not parked_without_boundary and not completed_unacknowledged,
            "running": len(blocking_attempts),
            "parked": len(parked_with_boundary) + len(parked_without_boundary),
            "parked_with_boundary": len(parked_with_boundary),
            "parked_without_boundary": len(parked_without_boundary),
            "completed_unacknowledged": len(completed_unacknowledged),
            "active": len(active),
            "blocking_attempts": [self._execution_status(execution) for execution in blocking_attempts],
            "completed_unacknowledged_attempts": [
                self._execution_status(execution) for execution in completed_unacknowledged
            ],
            "executions": [self._execution_status(execution) for execution in all_executions],
        }

    def records_for_commit(self) -> list[AgentBoundaryRecord]:
        report = self.status()
        if not report["ready_to_commit"]:
            raise AgentCheckpointError("cannot commit while agent executions still block durable publication")
        return [
            execution.boundary
            for execution in self._executions.values()
            if execution.state == AgentExecutionState.PARKED and execution.boundary is not None
        ]

    def install_restored(self, records: list[AgentBoundaryRecord]) -> None:
        for record in records:
            self._tombstones.add((record.rollout_id, record.attempt_index))
            self._restored[(record.rollout_id, record.attempt_index + 1)] = record
        self._accepting = False

    def _owns(self, execution: AgentExecution) -> bool:
        return self._executions.get((execution.rollout_id, execution.attempt_index)) is execution

    def _require_owner(self, execution: AgentExecution) -> None:
        if not self._owns(execution):
            raise AgentStaleAttemptError(
                f"rollout {execution.rollout_id!r} attempt {execution.attempt_index} execution "
                f"generation {execution.generation} is no longer current"
            )

    @staticmethod
    def _execution_status(execution: AgentExecution) -> dict[str, Any]:
        parked_boundary_state = None
        if execution.state == AgentExecutionState.PARKED:
            parked_boundary_state = (
                "parked_with_boundary" if execution.boundary is not None else "parked_without_boundary"
            )
        return {
            "rollout_id": execution.rollout_id,
            "attempt_index": execution.attempt_index,
            "generation": execution.generation,
            "state": execution.state.value,
            "parked_boundary_state": parked_boundary_state,
            "boundary_index": execution.boundary.boundary_index if execution.boundary is not None else None,
            "age_seconds": round(time.time() - execution.started_at, 3),
        }

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_instance_name(instance_name: Optional[str]) -> Optional[str]:
    if instance_name is None:
        return None
    if not instance_name or len(instance_name.encode("utf-8")) > 512:
        raise ValueError("agent checkpoint instance name must contain 1 to 512 UTF-8 bytes")
    if any(ord(character) < 32 or ord(character) == 127 for character in instance_name):
        raise ValueError("agent checkpoint instance name must not contain control characters")
    return instance_name


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
    directory = _agent_checkpoint_directory(checkpoint_dir, participant.instance_name)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / AGENT_MANIFEST_NAME
    if manifest_path.exists():
        return _validate_agent_manifest(
            directory,
            checkpoint_id=checkpoint_id,
            instance_name=participant.instance_name,
        )

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
        "instance_name": participant.instance_name,
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


def _validate_agent_manifest(
    directory: Path,
    *,
    checkpoint_id: str,
    instance_name: Optional[str] = None,
) -> dict[str, Any]:
    manifest_path = directory / AGENT_MANIFEST_NAME
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("checkpoint_id") != checkpoint_id:
        raise AgentCheckpointError(
            f"agent checkpoint directory belongs to {manifest.get('checkpoint_id')!r}, not {checkpoint_id!r}"
        )
    if manifest.get("instance_name") != instance_name:
        raise AgentCheckpointError(
            f"agent checkpoint belongs to instance {manifest.get('instance_name')!r}, not {instance_name!r}"
        )
    for name, digest in manifest.get("files", {}).items():
        path = directory / name
        if not path.exists() or _digest(path) != digest:
            raise AgentCheckpointError(f"agent checkpoint record {name!r} is missing or corrupted")
    return {
        "records": len(manifest.get("files", {})),
        "manifest_digest": hashlib.sha256(payload).hexdigest(),
    }


def restore_agent_state(participant: AgentCheckpointParticipant, checkpoint_dir: Path) -> dict[str, Any]:
    directory = _agent_checkpoint_directory(checkpoint_dir, participant.instance_name)
    manifest_path = directory / AGENT_MANIFEST_NAME
    if not manifest_path.exists():
        raise AgentCheckpointError(f"agent checkpoint has no committed manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("instance_name") != participant.instance_name:
        raise AgentCheckpointError(
            f"agent checkpoint belongs to instance {manifest.get('instance_name')!r}, "
            f"not {participant.instance_name!r}"
        )
    records: list[AgentBoundaryRecord] = []
    for name, digest in manifest["files"].items():
        path = directory / name
        if not path.exists() or _digest(path) != digest:
            raise AgentCheckpointError(f"agent checkpoint record {name!r} is missing or corrupted")
        records.append(AgentBoundaryRecord.model_validate_json(path.read_bytes()))
    participant.install_restored(records)
    return {"records": len(records), "source_checkpoint_id": manifest["checkpoint_id"]}


def _agent_checkpoint_directory(checkpoint_dir: Path, instance_name: Optional[str]) -> Path:
    directory = Path(checkpoint_dir) / AGENT_STATE_SUBDIR
    if instance_name is not None:
        validated_name = _validate_instance_name(instance_name)
        assert validated_name is not None
        directory /= f"instance-{hashlib.sha256(validated_name.encode('utf-8')).hexdigest()}"
    return directory


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
            result = await participant.prepare(body.deadline_ts)
            if not result["ready_to_commit"]:
                raise AgentPrepareIncompleteError(
                    "agent prepare is incomplete: "
                    f"running={result['running']}, "
                    f"parked_without_boundary={result['parked_without_boundary']}, "
                    f"completed_unacknowledged={result['completed_unacknowledged']}"
                )
            return result

        result = await fence.run_operation(
            body.checkpoint_id,
            "agent-checkpoint/prepare",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.PREPARING,
            phase_after=CheckpointPhase.PREPARING,
            run=run,
            deadline=body,
        )
        if result["ready_to_commit"]:
            fence.mark_prepared(body.checkpoint_id)
        return result

    @app.get(f"{AGENT_CHECKPOINT_URL_PREFIX}/status")
    async def status(
        checkpoint_id: str = Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        fence.require_phase(
            checkpoint_id,
            frozenset(CheckpointPhase),
        )
        return {"checkpoint_id": checkpoint_id, **participant.status()}

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
                    CheckpointPhase.IDLE,
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
            frozenset({CheckpointPhase.IDLE, CheckpointPhase.PREPARING, CheckpointPhase.PREPARED}),
        )
        return await participant.retire(body.rollout_id, body.attempt_index)
