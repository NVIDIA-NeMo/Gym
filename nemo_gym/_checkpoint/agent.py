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
import io
import json
import os
import tarfile
import tempfile
import time
from contextvars import ContextVar, Token
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Sequence

from fastapi import FastAPI, Header, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym._checkpoint.artifacts import (
    AgentContinuationRoot,
    CheckpointArtifactError,
    CheckpointArtifactReference,
    read_jsonl_artifact,
    write_jsonl_artifact,
)
from nemo_gym._checkpoint.control import CheckpointControlRequest, CheckpointPhase, ControlError, ControlFence
from nemo_gym._checkpoint.coordinator import (
    AdmissionCoordinator,
    ContinuationAlreadyOwnedError,
    ContinuationRegistry,
    ContinuationRegistryClient,
    ContinuationRetiredError,
    CoordinatorServiceClient,
)
from nemo_gym.rollout_correlation import ROLLOUT_ID_PATTERN, capture_key_for
from nemo_gym.token_id_capture.control_routes import require_control_auth


AGENT_CHECKPOINT_URL_PREFIX = "/ng-control/v1/agent-checkpoint"
AGENT_STATE_SUBDIR = "agent"
AGENT_MANIFEST_NAME = "manifest.json"
AGENT_CONTINUATION_INDEX_NAME = "continuations.jsonl"
AGENT_RECORD_INDEX_NAME = "agent-index.jsonl"
AGENT_CHECKPOINT_SCHEMA_VERSION = 1
AGENT_STATE_MANIFEST_SCHEMA_VERSION = 2
AGENT_BOUNDARY_SCHEMA_VERSION = 2
AGENT_EXECUTION_GENERATION_HEADER = "x-nemo-gym-agent-execution-generation"
COMPLETED_RESULT_ACKNOWLEDGEMENT_FEATURE = "completed_result_acknowledgement"
DISCARD_RESTORED_CONTINUATION_FEATURE = "discard_restored_continuation_v1"
_AGENT_ARCHIVE_PATTERN = r"^agent-part-[0-9]{6}\.tar$"
_AGENT_ARCHIVE_MAX_MEMBERS = 512
_AGENT_ARCHIVE_MAX_PAYLOAD_BYTES = 64 << 20
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

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


class _AgentArchiveReference(BaseModel):
    """Digest-bound coordinate for one bounded agent-state tar shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=_AGENT_ARCHIVE_PATTERN)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    members: int = Field(ge=1)
    bytes: int = Field(ge=0)


class _AgentArchiveMember(BaseModel):
    """Location and integrity metadata for one agent boundary record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)
    archive: str = Field(pattern=_AGENT_ARCHIVE_PATTERN)
    member: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_member_name(self) -> "_AgentArchiveMember":
        expected = _agent_record_name(self.rollout_id, self.attempt_index)
        if self.member != expected:
            raise ValueError(
                f"agent archive member does not match its rollout identity: expected={expected!r}, actual={self.member!r}"
            )
        return self


class DuplicateExecutionError(ControlError):
    code = "duplicate_execution"


class AgentAdmissionClosedError(ControlError):
    code = "agent_admission_closed"


class AgentStaleAttemptError(ControlError):
    code = "stale_attempt"


class AgentPrepareIncompleteError(ControlError):
    code = "agent_prepare_incomplete"


class AgentCompletedExecutionAcknowledgementError(ControlError):
    code = "completed_execution_acknowledgement_error"


class AgentExecutionIdentity(BaseModel):
    """Stable identity of one physical agent execution."""

    model_config = ConfigDict(extra="forbid")

    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)


class AgentCompletionReceipt(AgentExecutionIdentity):
    """Exact retained result that a durable caller may acknowledge."""

    execution_generation: int = Field(ge=1)
    result_identity: str = Field(min_length=1, max_length=512)
    result_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class AgentAcknowledgeRequest(AgentCompletionReceipt):
    """Compatibility name for acknowledging one exact completion receipt."""


class AgentCompletedExecutionAcknowledgementRequest(BaseModel):
    """Batch of completed results now owned durably by the caller."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[AGENT_CHECKPOINT_SCHEMA_VERSION] = AGENT_CHECKPOINT_SCHEMA_VERSION
    executions: list[AgentCompletionReceipt] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_executions(self) -> "AgentCompletedExecutionAcknowledgementRequest":
        keys = [(execution.rollout_id, execution.attempt_index) for execution in self.executions]
        if len(keys) != len(set(keys)):
            raise ValueError("completed execution acknowledgements must be unique")
        return self


class AgentCompletedExecutionAcknowledgementResponse(BaseModel):
    """Every requested execution whose cached terminal result is released."""

    model_config = ConfigDict(extra="forbid")

    acknowledged: list[AgentCompletionReceipt]


class AgentBoundaryKind(str, Enum):
    """The durable phase represented by an agent boundary."""

    PENDING_MODEL = "pending_model"
    TURN_COMPLETE = "turn_complete"


class PendingModelPayload(BaseModel):
    """A completed model generation whose actions may still be pending."""

    model_config = ConfigDict(extra="forbid")

    model_call_id: str = Field(min_length=1)
    response: dict[str, Any]
    model_server_cookies: dict[str, str] = Field(default_factory=dict)
    usage: Optional[dict[str, Any]] = None
    pending_action_cursor: int = Field(ge=0)
    resource_request_id: str = Field(min_length=1)


class AgentBoundaryRecord(BaseModel):
    """Continuation state at a generation-safe whitebox agent boundary."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, AGENT_BOUNDARY_SCHEMA_VERSION] = AGENT_BOUNDARY_SCHEMA_VERSION
    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)
    boundary_index: int = Field(ge=0)
    turn_index: int = Field(default=0, ge=0)
    boundary_kind: AgentBoundaryKind = AgentBoundaryKind.TURN_COMPLETE
    pending_model: Optional[PendingModelPayload] = None
    output_items: list[dict[str, Any]]
    usage: Optional[dict[str, Any]] = None
    last_committed_model_call_id: Optional[str] = None
    resource_state_revisions: dict[str, int] = Field(default_factory=dict)
    agent_state: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def validate_boundary_phase(self) -> "AgentBoundaryRecord":
        if "turn_index" not in self.model_fields_set:
            self.turn_index = self.boundary_index
        if self.boundary_kind == AgentBoundaryKind.PENDING_MODEL and self.pending_model is None:
            raise ValueError("pending_model boundaries require a pending_model payload")
        if self.boundary_kind == AgentBoundaryKind.TURN_COMPLETE and self.pending_model is not None:
            raise ValueError("turn_complete boundaries cannot carry a pending_model payload")
        return self


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
        self.result_identity: Optional[str] = None
        self.result_digest: Optional[str] = None
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


class AgentDiscardRestoredContinuationRequest(CheckpointControlRequest):
    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)


class AgentCheckpointParticipant:
    """Own active whitebox executions and park them at committed boundaries.

    Successful terminal results remain replayable until a shared durable
    acknowledgement layer releases them. Prepare reports those results as
    ``completed_unacknowledged`` and must not be treated as publishable while
    that count is nonzero.
    """

    def __init__(
        self,
        instance_name: Optional[str] = None,
        *,
        continuation_registry: Optional[
            ContinuationRegistry[tuple[str, int], AgentBoundaryRecord] | ContinuationRegistryClient
        ] = None,
        owner_id: Optional[str] = None,
    ) -> None:
        self.instance_name = _validate_instance_name(instance_name)
        self._executions: dict[tuple[str, int], AgentExecution] = {}
        self._generations: dict[tuple[str, int], int] = {}
        self._continuation_registry = continuation_registry or ContinuationRegistry()
        self._continuation_owner_id = owner_id or f"{os.getpid()}:{id(self):x}"
        # These exact process-lifetime fences make delayed retries deterministic.
        # They cannot be bounded safely until the wire protocol supplies a
        # coordinated epoch/high-watermark after which old identities cannot recur.
        self._tombstones: set[tuple[str, int]] = set()
        self._acknowledged: dict[tuple[str, int], tuple[int, str, str]] = {}
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
        if key in self._acknowledged:
            raise AgentStaleAttemptError(
                f"rollout {rollout_id!r} attempt {attempt_index} completed result was acknowledged"
            )
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
        try:
            continuation = await self._claim_continuation(key)
        except ContinuationRetiredError as error:
            raise AgentStaleAttemptError(
                f"rollout {rollout_id!r} attempt {attempt_index} was retired by restore"
            ) from error
        except ContinuationAlreadyOwnedError as error:
            raise DuplicateExecutionError(
                f"rollout {rollout_id!r} attempt {attempt_index} already has an active /run"
            ) from error
        execution = AgentExecution(
            rollout_id,
            attempt_index,
            generation,
            task,
            continuation,
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

    def completion_receipt(
        self,
        rollout_id: str,
        attempt_index: int,
    ) -> AgentCompletionReceipt:
        """Return the exact receipt for one retained terminal result."""
        execution = self.resolve(rollout_id, attempt_index)
        if (
            execution is None
            or execution.state != AgentExecutionState.COMPLETED
            or execution.terminal_result is None
            or execution.result_identity is None
            or execution.result_digest is None
        ):
            raise AgentCompletedExecutionAcknowledgementError(
                f"rollout {rollout_id!r} attempt {attempt_index} has no completed result receipt"
            )
        return AgentCompletionReceipt(
            rollout_id=execution.rollout_id,
            attempt_index=execution.attempt_index,
            execution_generation=execution.generation,
            result_identity=execution.result_identity,
            result_digest=execution.result_digest,
        )

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
                execution.result_identity, execution.result_digest = _result_receipt(result)
                execution.continuation = None
                execution.outer_task = None
                await self._mark_continuation_completed((execution.rollout_id, execution.attempt_index))
            else:
                execution.state = AgentExecutionState.RETIRED
                key = (execution.rollout_id, execution.attempt_index)
                await self._remember_tombstone(key, require_owner=True)
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
        for execution in list(self._executions.values()):
            if execution.state == AgentExecutionState.PARK_REQUESTED:
                execution.state = AgentExecutionState.RUNNING
            elif execution.state == AgentExecutionState.PARKED:
                if execution.outer_task is None:
                    execution.state = AgentExecutionState.RETIRED
                    key = (execution.rollout_id, execution.attempt_index)
                    await self._remember_tombstone(key, require_owner=True)
                    execution.resume_event.set()
                    if execution.parked_task is not None:
                        execution.parked_task.cancel()
                    self._executions.pop(key, None)
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
        await self._remember_tombstone(key, require_owner=execution is not None)
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

    async def discard_restored_continuation(
        self,
        rollout_id: str,
        attempt_index: int,
    ) -> dict[str, Any]:
        """Drop saved turn state while keeping its replacement attempt admissible."""
        key = (rollout_id, attempt_index)
        if key in self._executions:
            raise DuplicateExecutionError(f"rollout {rollout_id!r} attempt {attempt_index} is already active")
        try:
            discarded = await self._discard_continuation(key)
        except ContinuationAlreadyOwnedError as error:
            raise DuplicateExecutionError(
                f"rollout {rollout_id!r} attempt {attempt_index} is already active"
            ) from error
        await self._notify()
        return {"discarded": discarded}

    async def acknowledge_completed(
        self,
        receipts: list[AgentCompletionReceipt],
    ) -> list[AgentCompletionReceipt]:
        """Atomically release terminal results after the caller owns them durably."""
        keys = [(receipt.rollout_id, receipt.attempt_index) for receipt in receipts]
        if len(keys) != len(set(keys)):
            raise AgentCompletedExecutionAcknowledgementError("completed execution acknowledgements must be unique")

        # Validate the complete batch before releasing any result. A malformed
        # batch must not leave only some executions acknowledged.
        for key, receipt in zip(keys, receipts):
            expected = (
                receipt.execution_generation,
                receipt.result_identity,
                receipt.result_digest,
            )
            acknowledged = self._acknowledged.get(key)
            if acknowledged is not None:
                if acknowledged != expected:
                    raise AgentCompletedExecutionAcknowledgementError(
                        f"acknowledgement does not match rollout {key[0]!r} attempt {key[1]}'s completed receipt"
                    )
                continue
            execution = self._executions.get(key)
            if execution is None:
                raise AgentCompletedExecutionAcknowledgementError(
                    f"rollout {key[0]!r} attempt {key[1]} has no completed result to acknowledge"
                )
            if execution.state != AgentExecutionState.COMPLETED or execution.terminal_result is None:
                raise AgentCompletedExecutionAcknowledgementError(
                    f"rollout {key[0]!r} attempt {key[1]} is not completed with a retained result"
                )
            actual = (
                execution.generation,
                execution.result_identity,
                execution.result_digest,
            )
            if actual != expected:
                raise AgentCompletedExecutionAcknowledgementError(
                    f"acknowledgement receipt mismatch for rollout {key[0]!r} attempt {key[1]}"
                )

        newly_acknowledged = [key for key in keys if key not in self._acknowledged]
        await self._retire_completed_continuations(newly_acknowledged)
        for key, receipt in zip(keys, receipts):
            if key in self._acknowledged:
                continue
            execution = self._executions.pop(key)
            execution.state = AgentExecutionState.RETIRED
            execution.terminal_result = None
            self._acknowledged[key] = (
                receipt.execution_generation,
                receipt.result_identity,
                receipt.result_digest,
            )
            self._remember_local_tombstone(key)
        await self._notify()
        return receipts

    async def acknowledge(self, receipt: AgentAcknowledgeRequest) -> dict[str, Any]:
        """Release one exact terminal result, preserving idempotent retry semantics."""
        key = (receipt.rollout_id, receipt.attempt_index)
        idempotent = key in self._acknowledged
        await self.acknowledge_completed([receipt])
        return {"acknowledged": not idempotent, "idempotent": idempotent}

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
            "acknowledged_completed": len(self._acknowledged),
            "active": len(active),
            "blocking_attempts": [self._execution_status(execution) for execution in blocking_attempts],
            "completed_unacknowledged_attempts": [
                self._execution_status(execution) for execution in completed_unacknowledged
            ],
            "selected_boundaries": [
                {
                    "rollout_id": execution.rollout_id,
                    "attempt_index": execution.attempt_index,
                    "boundary_index": execution.boundary.boundary_index,
                    "turn_index": execution.boundary.turn_index,
                    "boundary_kind": execution.boundary.boundary_kind.value,
                    "resource_state_revisions": execution.boundary.resource_state_revisions,
                }
                for execution in parked_with_boundary
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
        """Install records into an in-process registry for single-worker use."""
        if isinstance(self._continuation_registry, ContinuationRegistryClient):
            raise RuntimeError("coordinator-backed participants require install_restored_async")
        self._continuation_registry.install(
            {(record.rollout_id, record.attempt_index + 1): record for record in records},
            retired={(record.rollout_id, record.attempt_index) for record in records},
        )
        for record in records:
            self._remember_local_tombstone((record.rollout_id, record.attempt_index))
        self._accepting = False

    async def install_restored_async(self, records: list[AgentBoundaryRecord]) -> None:
        """Install records through either the local or coordinator registry."""
        entries = {(record.rollout_id, record.attempt_index + 1): record for record in records}
        retired = {(record.rollout_id, record.attempt_index) for record in records}
        if isinstance(self._continuation_registry, ContinuationRegistryClient):
            await self._continuation_registry.install(
                {key: record.model_dump(mode="json") for key, record in entries.items()},
                retired=retired,
            )
        else:
            self._continuation_registry.install(entries, retired=retired)
        for key in retired:
            self._remember_local_tombstone(key)
        self._accepting = False

    async def _remember_tombstone(self, key: tuple[str, int], *, require_owner: bool) -> None:
        self._remember_local_tombstone(key)
        if isinstance(self._continuation_registry, ContinuationRegistryClient):
            await self._continuation_registry.retire(key, require_owner=require_owner)
        else:
            self._continuation_registry.retire(
                key,
                owner_id=self._continuation_owner_id if require_owner else None,
            )

    def _remember_local_tombstone(self, key: tuple[str, int]) -> None:
        self._tombstones.add(key)
        self._generations.pop(key, None)

    async def _claim_continuation(self, key: tuple[str, int]) -> Optional[AgentBoundaryRecord]:
        if isinstance(self._continuation_registry, ContinuationRegistryClient):
            value = await self._continuation_registry.claim_or_register(key)
            if value is None:
                return None
            try:
                return AgentBoundaryRecord.model_validate(value)
            except ValueError as error:
                raise AgentCheckpointError("coordinator returned an invalid agent continuation") from error
        return self._continuation_registry.claim_or_register(key, owner_id=self._continuation_owner_id)

    async def _mark_continuation_completed(self, key: tuple[str, int]) -> None:
        if isinstance(self._continuation_registry, ContinuationRegistryClient):
            await self._continuation_registry.mark_completed(key)
        else:
            self._continuation_registry.mark_completed(key, owner_id=self._continuation_owner_id)

    async def _discard_continuation(self, key: tuple[str, int]) -> bool:
        if isinstance(self._continuation_registry, ContinuationRegistryClient):
            return await self._continuation_registry.discard_available(key)
        return self._continuation_registry.discard_available(key)

    async def _retire_completed_continuations(self, keys: list[tuple[str, int]]) -> None:
        if not keys:
            return
        if isinstance(self._continuation_registry, ContinuationRegistryClient):
            await self._continuation_registry.retire_many(keys)
        else:
            self._continuation_registry.retire_many(keys, owner_id=self._continuation_owner_id)

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
            "turn_index": execution.boundary.turn_index if execution.boundary is not None else None,
            "boundary_kind": execution.boundary.boundary_kind.value if execution.boundary is not None else None,
            "resource_state_revisions": (
                execution.boundary.resource_state_revisions if execution.boundary is not None else {}
            ),
            **(
                {
                    "completion_receipt": {
                        "rollout_id": execution.rollout_id,
                        "attempt_index": execution.attempt_index,
                        "execution_generation": execution.generation,
                        "result_identity": execution.result_identity,
                        "result_digest": execution.result_digest,
                    }
                }
                if execution.state == AgentExecutionState.COMPLETED
                else {}
            ),
            "age_seconds": round(time.time() - execution.started_at, 3),
        }

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()


def _result_receipt(result: Any) -> tuple[str, str]:
    if isinstance(result, BaseModel):
        value = result.model_dump(mode="json")
    else:
        value = result
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    result_identity = None
    if isinstance(value, dict):
        candidate = value.get("id")
        if candidate is None and isinstance(value.get("response"), dict):
            candidate = value["response"].get("id")
        if candidate is not None:
            result_identity = str(candidate)
    return result_identity or digest, digest


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _agent_record_name(rollout_id: str, attempt_index: int) -> str:
    return f"{rollout_id}.a{attempt_index}.json"


def _partition_agent_archives(
    records: Sequence[AgentBoundaryRecord],
) -> Iterator[list[tuple[AgentBoundaryRecord, bytes]]]:
    """Serialize and yield one bounded shard at a time."""
    current: list[tuple[AgentBoundaryRecord, bytes]] = []
    current_bytes = 0
    for record in records:
        payload = record.model_dump_json(indent=2).encode()
        member_bytes = len(payload)
        if current and (
            len(current) >= _AGENT_ARCHIVE_MAX_MEMBERS
            or current_bytes + member_bytes > _AGENT_ARCHIVE_MAX_PAYLOAD_BYTES
        ):
            yield current
            current = []
            current_bytes = 0
        current.append((record, payload))
        current_bytes += member_bytes
    if current:
        yield current


def _write_agent_archive(
    directory: Path,
    *,
    archive_index: int,
    members: list[tuple[AgentBoundaryRecord, bytes]],
) -> tuple[_AgentArchiveReference, list[_AgentArchiveMember]]:
    """Atomically write and fsync one deterministic agent-state tar shard."""
    archive_name = f"agent-part-{archive_index:06d}.tar"
    target = directory / archive_name
    member_references: list[_AgentArchiveMember] = []
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".agent-archive-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with tarfile.open(fileobj=handle, mode="w") as archive:
                for record, payload in members:
                    member_name = _agent_record_name(record.rollout_id, record.attempt_index)
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
                        _AgentArchiveMember(
                            rollout_id=record.rollout_id,
                            attempt_index=record.attempt_index,
                            archive=archive_name,
                            member=member_name,
                            sha256=hashlib.sha256(payload).hexdigest(),
                            bytes=len(payload),
                        )
                    )
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    archive_size = temporary.stat().st_size
    archive_digest = _digest(temporary)
    os.replace(temporary, target)
    return (
        _AgentArchiveReference(
            name=archive_name,
            sha256=archive_digest,
            members=len(member_references),
            bytes=archive_size,
        ),
        member_references,
    )


def _load_agent_archive_records(
    directory: Path,
    *,
    checkpoint_root: Path,
    manifest: dict[str, Any],
) -> list[AgentBoundaryRecord]:
    """Validate every archive and deserialize its agent boundary records."""
    if manifest.get("schema_version") != AGENT_STATE_MANIFEST_SCHEMA_VERSION:
        raise AgentCheckpointError(
            "unsupported agent checkpoint manifest schema: "
            f"expected={AGENT_STATE_MANIFEST_SCHEMA_VERSION}, actual={manifest.get('schema_version')!r}"
        )
    try:
        archives = [_AgentArchiveReference.model_validate(item) for item in manifest["archives"]]
        record_index = CheckpointArtifactReference.model_validate(manifest["record_index"])
        members = read_jsonl_artifact(checkpoint_root, record_index, _AgentArchiveMember)
    except (KeyError, TypeError, ValueError, CheckpointArtifactError) as error:
        raise AgentCheckpointError("agent checkpoint archive metadata is missing or corrupted") from error

    archive_names = [archive.name for archive in archives]
    if len(set(archive_names)) != len(archive_names):
        raise AgentCheckpointError("agent checkpoint manifest contains duplicate archives")
    identities = [(member.rollout_id, member.attempt_index) for member in members]
    archive_members = [(member.archive, member.member) for member in members]
    if len(set(identities)) != len(identities) or len(set(archive_members)) != len(archive_members):
        raise AgentCheckpointError("agent checkpoint record index contains duplicate records")
    if manifest.get("records") != len(members) or record_index.records != len(members):
        raise AgentCheckpointError("agent checkpoint record count does not match its index")

    members_by_archive: dict[str, list[_AgentArchiveMember]] = {}
    for member in members:
        members_by_archive.setdefault(member.archive, []).append(member)
    if set(archive_names) != set(members_by_archive):
        raise AgentCheckpointError("agent checkpoint archive inventory does not match its index")

    records_by_identity: dict[tuple[str, int], AgentBoundaryRecord] = {}
    for archive_reference in archives:
        path = directory / archive_reference.name
        if not path.is_file():
            raise AgentCheckpointError(f"agent checkpoint archive {archive_reference.name!r} is missing")
        if path.stat().st_size != archive_reference.bytes or _digest(path) != archive_reference.sha256:
            raise AgentCheckpointError(f"agent checkpoint archive {archive_reference.name!r} is corrupted")
        expected = members_by_archive[archive_reference.name]
        if archive_reference.members != len(expected):
            raise AgentCheckpointError(
                f"agent checkpoint archive {archive_reference.name!r} member count is corrupted"
            )
        try:
            with tarfile.open(path, mode="r:") as archive:
                infos = archive.getmembers()
                if [info.name for info in infos] != [member.member for member in expected]:
                    raise AgentCheckpointError(
                        f"agent checkpoint archive {archive_reference.name!r} has an unexpected member inventory"
                    )
                for info, member in zip(infos, expected, strict=True):
                    if not info.isfile():
                        raise AgentCheckpointError(
                            f"agent checkpoint archive member {archive_reference.name!r}/{info.name!r} is invalid"
                        )
                    extracted = archive.extractfile(info)
                    if extracted is None:
                        raise AgentCheckpointError(
                            f"agent checkpoint archive member {archive_reference.name!r}/{info.name!r} cannot be read"
                        )
                    payload = extracted.read()
                    if len(payload) != member.bytes or hashlib.sha256(payload).hexdigest() != member.sha256:
                        raise AgentCheckpointError(
                            f"agent checkpoint archive member {archive_reference.name!r}/{info.name!r} is corrupted"
                        )
                    try:
                        record = AgentBoundaryRecord.model_validate_json(payload)
                    except ValueError as error:
                        raise AgentCheckpointError(
                            f"agent checkpoint archive member {archive_reference.name!r}/{info.name!r} is invalid"
                        ) from error
                    identity = (record.rollout_id, record.attempt_index)
                    if identity != (member.rollout_id, member.attempt_index):
                        raise AgentCheckpointError(
                            f"agent checkpoint archive member {archive_reference.name!r}/{info.name!r} has the wrong identity"
                        )
                    records_by_identity[identity] = record
        except (OSError, tarfile.TarError) as error:
            raise AgentCheckpointError(
                f"agent checkpoint archive {archive_reference.name!r} cannot be read"
            ) from error
    return [records_by_identity[identity] for identity in identities]


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
    """Synchronously snapshot and commit one agent participant."""
    return _commit_agent_records(
        tuple(participant.records_for_commit()),
        checkpoint_dir,
        checkpoint_id=checkpoint_id,
        instance_name=participant.instance_name,
    )


def _commit_agent_records(
    records: Sequence[AgentBoundaryRecord],
    checkpoint_dir: Path,
    *,
    checkpoint_id: str,
    instance_name: Optional[str] = None,
) -> dict[str, Any]:
    directory = _agent_checkpoint_directory(checkpoint_dir, instance_name)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / AGENT_MANIFEST_NAME
    if manifest_path.exists():
        return _validate_agent_manifest(
            directory,
            checkpoint_root=checkpoint_dir,
            checkpoint_id=checkpoint_id,
            instance_name=instance_name,
        )

    ordered_records = sorted(records, key=lambda record: (record.rollout_id, record.attempt_index))
    identities = [(record.rollout_id, record.attempt_index) for record in ordered_records]
    if len(set(identities)) != len(identities):
        raise AgentCheckpointError("agent checkpoint contains duplicate rollout attempts")
    archive_references: list[_AgentArchiveReference] = []
    archive_members: list[_AgentArchiveMember] = []
    for archive_index, archive_records in enumerate(_partition_agent_archives(ordered_records)):
        archive_reference, members = _write_agent_archive(
            directory,
            archive_index=archive_index,
            members=archive_records,
        )
        archive_references.append(archive_reference)
        archive_members.extend(members)
    record_index = write_jsonl_artifact(
        checkpoint_dir,
        directory.relative_to(checkpoint_dir) / AGENT_RECORD_INDEX_NAME,
        archive_members,
    )
    continuation_roots = sorted(
        (
            AgentContinuationRoot(
                rollout_id=record.rollout_id,
                attempt_index=record.attempt_index,
                capture_key=capture_key_for(record.rollout_id, record.attempt_index),
                last_committed_model_call_id=record.last_committed_model_call_id,
                resource_state_revisions=dict(record.resource_state_revisions),
            )
            for record in ordered_records
            if record.last_committed_model_call_id is not None
        ),
        key=lambda root: (root.capture_key, root.last_committed_model_call_id),
    )
    continuation_index = write_jsonl_artifact(
        checkpoint_dir,
        directory.relative_to(checkpoint_dir) / AGENT_CONTINUATION_INDEX_NAME,
        continuation_roots,
    )
    _fsync_dir(directory)

    manifest = {
        "schema_version": AGENT_STATE_MANIFEST_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "instance_name": instance_name,
        "archives": [reference.model_dump(mode="json") for reference in archive_references],
        "record_index": record_index.model_dump(mode="json"),
        "records": len(archive_members),
        "continuation_index": continuation_index.model_dump(mode="json"),
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2).encode()
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".manifest-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)
    _fsync_dir(directory)
    return {
        "records": len(archive_members),
        "manifest_digest": hashlib.sha256(payload).hexdigest(),
        "continuation_index": continuation_index.model_dump(mode="json"),
    }


def _validate_agent_manifest(
    directory: Path,
    *,
    checkpoint_root: Path,
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
    records = _load_agent_archive_records(directory, checkpoint_root=checkpoint_root, manifest=manifest)
    continuation_index = _validate_continuation_index(checkpoint_root, manifest, records)
    result: dict[str, Any] = {
        "records": len(records),
        "manifest_digest": hashlib.sha256(payload).hexdigest(),
    }
    result["continuation_index"] = continuation_index.model_dump(mode="json")
    return result


def load_agent_checkpoint_records(checkpoint_root: Path, manifest_path: Path) -> list[AgentBoundaryRecord]:
    """Validate and load records from one committed agent participant manifest."""
    manifest_path = Path(manifest_path)
    if manifest_path.name != AGENT_MANIFEST_NAME or not manifest_path.is_file():
        raise AgentCheckpointError(f"agent checkpoint manifest is missing at {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise AgentCheckpointError(f"agent checkpoint manifest is corrupted at {manifest_path}") from error
    return _load_agent_archive_records(
        manifest_path.parent,
        checkpoint_root=Path(checkpoint_root),
        manifest=manifest,
    )


def restore_agent_state(participant: AgentCheckpointParticipant, checkpoint_dir: Path) -> dict[str, Any]:
    records, source_checkpoint_id, continuation_index = _load_agent_state_for_restore(
        checkpoint_dir,
        participant.instance_name,
    )
    participant.install_restored(records)
    return {
        "records": len(records),
        "source_checkpoint_id": source_checkpoint_id,
        "continuation_index": continuation_index.model_dump(mode="json"),
    }


def _load_agent_state_for_restore(
    checkpoint_dir: Path,
    instance_name: Optional[str],
) -> tuple[list[AgentBoundaryRecord], str, CheckpointArtifactReference]:
    directory = _agent_checkpoint_directory(checkpoint_dir, instance_name)
    manifest_path = directory / AGENT_MANIFEST_NAME
    if not manifest_path.exists():
        raise AgentCheckpointError(f"agent checkpoint has no committed manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("instance_name") != instance_name:
        raise AgentCheckpointError(
            f"agent checkpoint belongs to instance {manifest.get('instance_name')!r}, not {instance_name!r}"
        )
    records = _load_agent_archive_records(directory, checkpoint_root=checkpoint_dir, manifest=manifest)
    continuation_index = _validate_continuation_index(checkpoint_dir, manifest, records)
    return records, str(manifest["checkpoint_id"]), continuation_index


def _validate_continuation_index(
    checkpoint_root: Path,
    manifest: dict[str, Any],
    records: Sequence[AgentBoundaryRecord],
) -> CheckpointArtifactReference:
    raw_reference = manifest.get("continuation_index")
    if raw_reference is None:
        raise AgentCheckpointError("agent checkpoint manifest is missing its continuation index")
    try:
        reference = CheckpointArtifactReference.model_validate(raw_reference)
        roots = read_jsonl_artifact(checkpoint_root, reference, AgentContinuationRoot)
    except (CheckpointArtifactError, ValueError) as error:
        raise AgentCheckpointError("agent continuation index is missing or corrupted") from error
    expected = {
        (
            record.rollout_id,
            record.attempt_index,
            capture_key_for(record.rollout_id, record.attempt_index),
            record.last_committed_model_call_id,
        ): record.resource_state_revisions
        for record in records
        if record.last_committed_model_call_id is not None
    }
    actual = [
        (
            root.rollout_id,
            root.attempt_index,
            root.capture_key,
            root.last_committed_model_call_id,
        )
        for root in roots
    ]
    if len(set(actual)) != len(roots) or set(actual) != set(expected):
        raise AgentCheckpointError("agent continuation index does not match committed boundary records")
    for root, identity in zip(roots, actual, strict=True):
        if root.resource_state_revisions is not None and root.resource_state_revisions != expected[identity]:
            raise AgentCheckpointError(
                "agent continuation index resource revisions do not match committed boundary records"
            )
    return reference


def _agent_checkpoint_directory(checkpoint_dir: Path, instance_name: Optional[str]) -> Path:
    directory = Path(checkpoint_dir) / AGENT_STATE_SUBDIR
    if instance_name is not None:
        validated_name = _validate_instance_name(instance_name)
        assert validated_name is not None
        directory /= f"instance-{hashlib.sha256(validated_name.encode('utf-8')).hexdigest()}"
    return directory


class AgentCheckpointWorkerHandler:
    """Execute coordinator commands against one worker-local participant."""

    def __init__(self, participant: AgentCheckpointParticipant) -> None:
        self.participant = participant

    async def __call__(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "prepare":
            return await self.participant.prepare(float(payload["deadline_ts"]))
        if operation == "status":
            return self.participant.status()
        if operation == "records_for_commit":
            return [record.model_dump(mode="json") for record in self.participant.records_for_commit()]
        if operation == "install_restored":
            records = [AgentBoundaryRecord.model_validate(item) for item in payload.get("records", ())]
            await self.participant.install_restored_async(records)
            return {"records": len(records)}
        if operation == "resume":
            return await self.participant.resume()
        if operation == "retire_owned":
            rollout_id = str(payload["rollout_id"])
            attempt_index = int(payload["attempt_index"])
            if self.participant.resolve(rollout_id, attempt_index) is None:
                return {"retired": False, "tombstoned": False}
            return await self.participant.retire(rollout_id, attempt_index)
        if operation == "completion_receipt":
            return self.participant.completion_receipt(
                str(payload["rollout_id"]),
                int(payload["attempt_index"]),
            ).model_dump(mode="json")
        if operation == "acknowledge_completed":
            receipts = [AgentCompletionReceipt.model_validate(item) for item in payload.get("executions", ())]
            acknowledged = await self.participant.acknowledge_completed(receipts)
            return [receipt.model_dump(mode="json") for receipt in acknowledged]
        raise ValueError(f"unknown agent checkpoint worker operation {operation!r}")


class AgentCheckpointCoordinatorService:
    """Own one logical agent checkpoint participant across Uvicorn workers."""

    def __init__(self, coordinator: AdmissionCoordinator, *, instance_name: Optional[str]) -> None:
        self.coordinator = coordinator
        self.instance_name = _validate_instance_name(instance_name)
        self.fence = ControlFence()
        self._acknowledged: dict[tuple[str, int], AgentCompletionReceipt] = {}

    async def __call__(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "prepare":
            return await self._prepare(AgentPrepareRequest.model_validate(payload))
        if operation == "status":
            checkpoint_id = str(payload.get("checkpoint_id", ""))
            self.fence.require_phase(checkpoint_id, frozenset(CheckpointPhase))
            return {"checkpoint_id": checkpoint_id, **await self._status()}
        if operation == "commit":
            return await self._commit(AgentCommitRequest.model_validate(payload))
        if operation == "restore":
            return await self._restore(AgentRestoreRequest.model_validate(payload))
        if operation == "resume":
            return await self._resume(AgentResumeRequest.model_validate(payload))
        if operation == "retire":
            return await self._retire(AgentRetireRequest.model_validate(payload))
        if operation == "discard_restored_continuation":
            return self._discard(AgentDiscardRestoredContinuationRequest.model_validate(payload))
        if operation == "completion_receipt":
            return await self._completion_receipt(payload)
        if operation == "acknowledge_completed":
            request = AgentCompletedExecutionAcknowledgementRequest.model_validate(payload)
            return await self._acknowledge_completed(request)
        if operation == "acknowledge":
            receipt = AgentAcknowledgeRequest.model_validate(payload)
            idempotent = (receipt.rollout_id, receipt.attempt_index) in self._acknowledged
            await self._acknowledge_completed(AgentCompletedExecutionAcknowledgementRequest(executions=[receipt]))
            return {"acknowledged": not idempotent, "idempotent": idempotent}
        raise ValueError(f"unknown agent checkpoint service operation {operation!r}")

    async def _prepare(self, body: AgentPrepareRequest) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            reports = await self.coordinator.run_worker_command(
                "prepare",
                {"deadline_ts": body.deadline_ts},
                timeout_s=max(body.remaining(), 0.001),
            )
            result = self._aggregate_status(reports)
            if not result["ready_to_commit"]:
                raise AgentPrepareIncompleteError(
                    "agent prepare is incomplete: "
                    f"running={result['running']}, "
                    f"parked_without_boundary={result['parked_without_boundary']}, "
                    f"completed_unacknowledged={result['completed_unacknowledged']}"
                )
            return result

        result = await self.fence.run_operation(
            body.checkpoint_id,
            "agent-checkpoint/prepare",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.PREPARING,
            phase_after=CheckpointPhase.PREPARING,
            run=run,
            deadline=body,
        )
        self.fence.mark_prepared(body.checkpoint_id)
        return result

    async def _status(self) -> dict[str, Any]:
        reports = await self.coordinator.run_worker_command("status", {}, timeout_s=10.0)
        return self._aggregate_status(reports)

    async def _commit(self, body: AgentCommitRequest) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            worker_records = await self.coordinator.run_worker_command(
                "records_for_commit",
                {},
                timeout_s=max(body.remaining(), 0.001),
            )
            records = [AgentBoundaryRecord.model_validate(item) for items in worker_records.values() for item in items]
            return await asyncio.to_thread(
                _commit_agent_records,
                records,
                Path(body.checkpoint_dir),
                checkpoint_id=body.checkpoint_id,
                instance_name=self.instance_name,
            )

        return await self.fence.run_operation(
            body.checkpoint_id,
            "agent-checkpoint/commit",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.COMMITTING,
            phase_after=CheckpointPhase.COMMITTED_PAUSED,
            run=run,
            deadline=body,
        )

    async def _restore(self, body: AgentRestoreRequest) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            records, source_checkpoint_id, continuation_index = await asyncio.to_thread(
                _load_agent_state_for_restore,
                Path(body.checkpoint_dir),
                self.instance_name,
            )
            await self.coordinator.run_worker_command(
                "install_restored",
                {"records": [record.model_dump(mode="json") for record in records]},
                timeout_s=max(body.remaining(), 0.001),
            )
            return {
                "records": len(records),
                "source_checkpoint_id": source_checkpoint_id,
                "continuation_index": continuation_index.model_dump(mode="json"),
            }

        return await self.fence.run_operation(
            body.checkpoint_id,
            "agent-checkpoint/restore",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.RESTORING,
            phase_after=CheckpointPhase.RESTORED_PAUSED,
            run=run,
            deadline=body,
        )

    async def _resume(self, body: AgentResumeRequest) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            reports = await self.coordinator.run_worker_command(
                "resume",
                {},
                timeout_s=max(body.remaining(), 0.001),
            )
            return {
                "state": "accepting",
                "released": sum(int(report["released"]) for report in reports.values()),
            }

        return await self.fence.run_operation(
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
            phase_during=self.fence.phase,
            phase_after=CheckpointPhase.IDLE,
            run=run,
            retire_outcome="resumed",
        )

    async def _retire(self, body: AgentRetireRequest) -> dict[str, Any]:
        self.fence.require_phase(
            body.checkpoint_id,
            frozenset({CheckpointPhase.IDLE, CheckpointPhase.PREPARING, CheckpointPhase.PREPARED}),
        )
        key = (body.rollout_id, body.attempt_index)
        owner_id = self.coordinator.continuation_registry.owner_id(key)
        if owner_id is None:
            self.coordinator.continuation_registry.retire(key)
            return {"retired": False, "tombstoned": True}
        results = await self.coordinator.run_worker_command(
            "retire_owned",
            {"rollout_id": body.rollout_id, "attempt_index": body.attempt_index},
            timeout_s=max(body.remaining(), 0.001),
            worker_ids=(owner_id,),
        )
        return results[owner_id]

    def _discard(self, body: AgentDiscardRestoredContinuationRequest) -> dict[str, Any]:
        self.fence.require_phase(body.checkpoint_id, frozenset({CheckpointPhase.RESTORED_PAUSED}))
        return {
            "discarded": self.coordinator.continuation_registry.discard_available(
                (body.rollout_id, body.attempt_index)
            )
        }

    async def _completion_receipt(self, payload: dict[str, Any]) -> dict[str, Any]:
        rollout_id = str(payload["rollout_id"])
        attempt_index = int(payload["attempt_index"])
        owner_id = self.coordinator.continuation_registry.owner_id((rollout_id, attempt_index))
        if owner_id is None:
            raise AgentCompletedExecutionAcknowledgementError(
                f"rollout {rollout_id!r} attempt {attempt_index} has no completed result receipt"
            )
        result = await self.coordinator.run_worker_command(
            "completion_receipt",
            {"rollout_id": rollout_id, "attempt_index": attempt_index},
            timeout_s=10.0,
            worker_ids=(owner_id,),
        )
        return result[owner_id]

    async def _acknowledge_completed(
        self,
        body: AgentCompletedExecutionAcknowledgementRequest,
    ) -> dict[str, Any]:
        by_owner: dict[str, list[AgentCompletionReceipt]] = {}
        pending: list[tuple[str, AgentCompletionReceipt]] = []
        for receipt in body.executions:
            key = (receipt.rollout_id, receipt.attempt_index)
            acknowledged = self._acknowledged.get(key)
            if acknowledged is not None:
                if acknowledged != receipt:
                    raise AgentCompletedExecutionAcknowledgementError(
                        f"acknowledgement does not match rollout {receipt.rollout_id!r} "
                        f"attempt {receipt.attempt_index}'s completed receipt"
                    )
                continue
            owner_id = self.coordinator.continuation_registry.owner_id(key)
            if owner_id is None:
                raise AgentCompletedExecutionAcknowledgementError(
                    f"rollout {receipt.rollout_id!r} attempt {receipt.attempt_index} "
                    "has no completed result to acknowledge"
                )
            by_owner.setdefault(owner_id, []).append(receipt)
            pending.append((owner_id, receipt))

        async def validate_receipt(owner_id: str, receipt: AgentCompletionReceipt) -> None:
            results = await self.coordinator.run_worker_command(
                "completion_receipt",
                {
                    "rollout_id": receipt.rollout_id,
                    "attempt_index": receipt.attempt_index,
                },
                timeout_s=10.0,
                worker_ids=(owner_id,),
            )
            actual = AgentCompletionReceipt.model_validate(results[owner_id])
            if actual != receipt:
                raise AgentCompletedExecutionAcknowledgementError(
                    f"acknowledgement receipt mismatch for rollout {receipt.rollout_id!r} "
                    f"attempt {receipt.attempt_index}"
                )

        # Preserve the single-worker all-or-nothing contract: validate the
        # complete cross-worker batch before asking any owner to release data.
        await asyncio.gather(*(validate_receipt(owner_id, receipt) for owner_id, receipt in pending))

        async def acknowledge_owner(owner_id: str, receipts: list[AgentCompletionReceipt]) -> None:
            await self.coordinator.run_worker_command(
                "acknowledge_completed",
                {"executions": [receipt.model_dump(mode="json") for receipt in receipts]},
                timeout_s=10.0,
                worker_ids=(owner_id,),
            )

        await asyncio.gather(*(acknowledge_owner(owner_id, receipts) for owner_id, receipts in by_owner.items()))
        for receipt in body.executions:
            self._acknowledged[(receipt.rollout_id, receipt.attempt_index)] = receipt
        return AgentCompletedExecutionAcknowledgementResponse(acknowledged=body.executions).model_dump(mode="json")

    @staticmethod
    def _aggregate_status(reports: dict[str, Any]) -> dict[str, Any]:
        values = list(reports.values())
        count_fields = (
            "running",
            "parked",
            "parked_with_boundary",
            "parked_without_boundary",
            "completed_unacknowledged",
            "acknowledged_completed",
            "active",
        )
        list_fields = (
            "blocking_attempts",
            "completed_unacknowledged_attempts",
            "selected_boundaries",
            "executions",
        )
        return {
            "state": "accepting" if all(value["state"] == "accepting" for value in values) else "preparing",
            "ready_to_commit": all(bool(value["ready_to_commit"]) for value in values),
            **{field: sum(int(value[field]) for value in values) for field in count_fields},
            **{field: [item for value in values for item in value[field]] for field in list_fields},
        }


def install_agent_checkpoint(
    app: FastAPI,
    *,
    participant: AgentCheckpointParticipant,
    fence: ControlFence,
    auth_token: str,
    coordinator_client: Optional[CoordinatorServiceClient] = None,
) -> None:
    """Install acknowledgement, prepare, commit, restore, resume, and retire routes."""

    @app.post(f"{AGENT_CHECKPOINT_URL_PREFIX}/acknowledge")
    async def acknowledge(
        body: AgentAcknowledgeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        if coordinator_client is not None:
            return await coordinator_client.request(
                "acknowledge",
                body.model_dump(mode="json"),
                timeout_s=_coordinator_request_timeout(body),
            )
        return await participant.acknowledge(body)

    @app.post(f"{AGENT_CHECKPOINT_URL_PREFIX}/acknowledge-completed")
    async def acknowledge_completed(
        body: AgentCompletedExecutionAcknowledgementRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        if coordinator_client is not None:
            return await coordinator_client.request(
                "acknowledge_completed",
                body.model_dump(mode="json"),
                timeout_s=10.0,
            )
        acknowledged = await participant.acknowledge_completed(body.executions)
        return AgentCompletedExecutionAcknowledgementResponse(acknowledged=acknowledged).model_dump()

    @app.get(f"{AGENT_CHECKPOINT_URL_PREFIX}/completion-receipt")
    async def completion_receipt(
        rollout_id: str = Query(pattern=ROLLOUT_ID_PATTERN.pattern),
        attempt_index: int = Query(ge=0),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        if coordinator_client is not None:
            return await coordinator_client.request(
                "completion_receipt",
                {"rollout_id": rollout_id, "attempt_index": attempt_index},
                timeout_s=10.0,
            )
        return participant.completion_receipt(rollout_id, attempt_index).model_dump(mode="json")

    @app.post(f"{AGENT_CHECKPOINT_URL_PREFIX}/prepare")
    async def prepare(
        body: AgentPrepareRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        if coordinator_client is not None:
            return await coordinator_client.request(
                "prepare",
                body.model_dump(mode="json"),
                timeout_s=_coordinator_request_timeout(body),
            )

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
        if coordinator_client is not None:
            return await coordinator_client.request(
                "status",
                {"checkpoint_id": checkpoint_id},
                timeout_s=10.0,
            )
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
        if coordinator_client is not None:
            return await coordinator_client.request(
                "commit",
                body.model_dump(mode="json"),
                timeout_s=_coordinator_request_timeout(body),
            )

        async def run() -> dict[str, Any]:
            # The participant belongs to this event loop. Materialize its state
            # here so the worker thread performs file I/O only.
            records = tuple(participant.records_for_commit())
            return await asyncio.to_thread(
                _commit_agent_records,
                records,
                Path(body.checkpoint_dir),
                checkpoint_id=body.checkpoint_id,
                instance_name=participant.instance_name,
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
        if coordinator_client is not None:
            return await coordinator_client.request(
                "restore",
                body.model_dump(mode="json"),
                timeout_s=_coordinator_request_timeout(body),
            )

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
        if coordinator_client is not None:
            return await coordinator_client.request(
                "resume",
                body.model_dump(mode="json"),
                timeout_s=_coordinator_request_timeout(body),
            )

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
        if coordinator_client is not None:
            return await coordinator_client.request(
                "retire",
                body.model_dump(mode="json"),
                timeout_s=_coordinator_request_timeout(body),
            )
        fence.require_phase(
            body.checkpoint_id,
            frozenset({CheckpointPhase.IDLE, CheckpointPhase.PREPARING, CheckpointPhase.PREPARED}),
        )
        return await participant.retire(body.rollout_id, body.attempt_index)

    @app.post(f"{AGENT_CHECKPOINT_URL_PREFIX}/discard-restored-continuation")
    async def discard_restored_continuation(
        body: AgentDiscardRestoredContinuationRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        if coordinator_client is not None:
            return await coordinator_client.request(
                "discard_restored_continuation",
                body.model_dump(mode="json"),
                timeout_s=_coordinator_request_timeout(body),
            )
        fence.require_phase(
            body.checkpoint_id,
            frozenset({CheckpointPhase.RESTORED_PAUSED}),
        )
        return await participant.discard_restored_continuation(
            body.rollout_id,
            body.attempt_index,
        )


def _coordinator_request_timeout(body: CheckpointControlRequest) -> float:
    """Leave a small response budget after the coordinator's own deadline."""
    return max(body.remaining(), 0.001) + 1.0
