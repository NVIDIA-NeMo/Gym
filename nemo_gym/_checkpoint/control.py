# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint control plane shared by every participating server.

A participant is the part of one server process that owns checkpointable state. Every participant
exposes the same bearer-protected routes under ``/ng-control/v1/checkpoint`` and moves through one
phase machine::

    idle --prepare--> preparing --(ready)--> prepared --commit--> committed
    idle --restore--> restored
    preparing | prepared | committed | restored --resume--> idle

The control plane here owns everything that is the same for every participant:

- checkpoint-ID fencing, phases, and idempotent replay of completed operations;
- retire: refusing an attempt's requests, stopping its work and freeing its state before replying. The
  refusal outlives the retire, because a request a caller sent before its own retire can arrive after
  this server's retire finished; it lasts until the controller calls ``forget`` for the rollout;
- the readiness wait, deadlines, and manifest-last storage, with one write per checkpoint that resume stops;
- a lease: if the controller that started a checkpoint stops calling, the participant resumes on its
  own when the lease expires, which is the same as an abort.

``resume`` also aborts: it reopens admission and lets parked work continue whether or not the checkpoint
was published. Resuming remembers the checkpoint ID, so a delayed call from a stale controller cannot
reopen or overwrite anything.

Transitions are serialized by one lock. Prepare waits for readiness outside the lock, so a controller
can resume while a prepare is waiting. Retire is refused until it does. A prepare that misses its deadline
returns its blockers and leaves admission closed. A prepare that raises reopens admission, so phase
and admission never disagree.
"""

import asyncio
import hmac
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Optional

import orjson
from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PlainValidator, ValidationError, model_validator

from nemo_gym._checkpoint.errors import (
    CheckpointConflictError,
    CheckpointStateError,
    ControlError,
    DeadlineExceededError,
    InvalidPhaseError,
    RestartInScopeError,
    StaleAttemptError,
    StaleCheckpointError,
    UnauthorizedError,
)
from nemo_gym._checkpoint.store import WriteStop, participant_dir, read_participant_state, write_participant_state
from nemo_gym._checkpoint.telemetry import OperationSpan, checkpoint_span
from nemo_gym._checkpoint.telemetry import operation as checkpoint_operation
from nemo_gym.episode_types import EpisodeId
from nemo_gym.telemetry.gym_metrics import record_checkpoint_event, record_checkpoint_volume


LOGGER = logging.getLogger(__name__)

CHECKPOINT_ROUTE_PREFIX = "/ng-control/v1/checkpoint"
CHECKPOINT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
# A stale controller retries within moments; this many resumed checkpoint IDs are plenty to refuse it.
_RESUMED_IDS_KEPT = 64
# How long an expired lease waits for restored state still being deleted before it tries again.
_LEASE_REOPEN_WAIT_SECONDS = 30.0


class CheckpointPhase(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTED = "committed"
    RESTORED = "restored"


class CheckpointRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str = Field(min_length=1, max_length=128, pattern=CHECKPOINT_ID_PATTERN)
    deadline_ts: FiniteFloat = Field(description="Absolute deadline in seconds since the Unix epoch.")

    def remaining(self) -> float:
        remaining = self.deadline_ts - time.time()
        if remaining <= 0:
            raise DeadlineExceededError(f"checkpoint {self.checkpoint_id!r} deadline passed")
        return remaining


class CommitRequest(CheckpointRequest):
    checkpoint_dir: str = Field(min_length=1, description="Controller-owned directory visible to this server.")
    episode_ids: Optional[list[EpisodeId]] = Field(
        default=None,
        description="The episodes the controller continues from this checkpoint. Participants that keep "
        "state for every episode they ever served, such as the model ledger, export only these.",
    )


class RestoreRequest(CheckpointRequest):
    checkpoint_dir: str = Field(min_length=1, description="Controller-owned directory visible to this server.")
    episode_ids: list[EpisodeId] = Field(
        description="The checkpointed attempts the controller will continue. Records of other episodes are "
        "not installed, so state that no replacement attempt will claim is never restored."
    )


class RetireRequest(CheckpointRequest):
    episode_ids: list[EpisodeId] = Field(min_length=1)


class ForgetRequest(CheckpointRequest):
    rollout_ids: list[str] = Field(
        min_length=1,
        description="Rollouts that are finished for good: nothing of any of their attempts can still run.",
    )


def _check_json(value: Any) -> Any:
    """Accept only what JSON carries back unchanged, as ``JsonValue`` would, usually without walking the value.

    ``orjson`` is the fast check.
    It also rejects integers beyond 64 bits and values nested deeper than it supports, which JSON carries fine,
    so a rejection is checked again in Python.
    ``orjson`` accepts some values the writer's ``json`` cannot write, such as ``datetime``;
    for those the writer fails the commit with a ``CheckpointStateError`` naming the episode.
    """
    try:
        orjson.dumps(value)
    except TypeError as error:
        try:
            plain = _plain_json(value)
        except RecursionError:
            plain = False
        if not plain:
            raise ValueError(f"checkpoint payload is not JSON: {error}") from error
    return value


def _plain_json(value: Any) -> bool:
    """Whether ``value`` is built only of JSON types, with string keys, so it reads back with the same keys."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_plain_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _plain_json(item) for key, item in value.items())
    return False


#: Opaque, server-owned state inside a record (a session's environment, an agent's boundary). It is checked once
#: for JSON-ness and otherwise passed through as is: the server's own restore hook validates its meaning, and a
#: deep pydantic validation and dump of large states cost more than the rest of a commit or restore.
JsonPayload = Annotated[Any, PlainValidator(_check_json)]


class CheckpointRecord(BaseModel):
    """Base of every participant record: each record belongs to one episode attempt."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId

    def to_json_record(self) -> dict[str, Any]:
        """The record as written to the store. ``JsonPayload`` fields go out as they are, without a copy."""
        payload_fields = {
            name for name, field in type(self).model_fields.items() if _check_json in _validators(field.metadata)
        }
        record = self.model_dump(mode="json", exclude=payload_fields)
        record.update({name: getattr(self, name) for name in payload_fields})
        return record


def _validators(metadata: list[Any]) -> list[Any]:
    return [getattr(item, "func", None) for item in metadata]


MAX_REPORTED_BLOCKERS = 100


class PrepareReport(BaseModel):
    """Whether a participant's frozen state can be committed, and what blocks it otherwise.

    At most ``MAX_REPORTED_BLOCKERS`` blocker keys are listed so a prepare
    that misses its deadline with thousands of stragglers still returns a small reply;
    ``blocker_count`` is always the full count.
    """

    ready: bool
    blockers: list[str] = Field(default_factory=list, description="Capture keys still preventing a safe cut.")
    blocker_count: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    restarts: list[str] = Field(
        default_factory=list,
        description="Capture keys of in-flight episodes this participant cannot capture. They do not block "
        "prepare; the controller leaves them out of the commit scope and starts them over after a crash. "
        "Never capped: the controller must see every one.",
    )

    @model_validator(mode="before")
    @classmethod
    def cap_blockers(cls, data: Any) -> Any:
        if isinstance(data, dict) and "blocker_count" not in data:
            blockers = list(data.get("blockers") or [])
            data = {**data, "blockers": blockers[:MAX_REPORTED_BLOCKERS], "blocker_count": len(blockers)}
        return data


class RetiredAttempts:
    """Refuse requests from retired attempts until the controller forgets their rollout.

    Retire stops callers before callees,
    but a request a caller sent just before its own retire travels on a different connection
    than the callee's retire, and can arrive after the callee's retire finished.
    So a retire raises its rollout's mark before it stops anything, never lowers it,
    and keeps it whether or not the retire succeeds.
    The mark goes away only when the controller calls ``forget`` for the rollout,
    once nothing of any of its attempts can still run: one entry per rollout retired and not yet forgotten.
    """

    def __init__(self) -> None:
        self._marks: dict[str, int] = {}

    def check(self, episode_id: EpisodeId) -> None:
        mark = self._marks.get(episode_id.rollout_id)
        if mark is not None and episode_id.attempt <= mark:
            raise StaleAttemptError(f"rollout {episode_id.rollout_id!r} attempt {episode_id.attempt} was retired")

    def mark(self, episode_ids: Iterable[EpisodeId]) -> None:
        """Refuse ``episode_ids`` and their earlier attempts from now on."""
        for episode_id in episode_ids:
            self._marks[episode_id.rollout_id] = max(self._marks.get(episode_id.rollout_id, -1), episode_id.attempt)

    def forget(self, rollout_ids: Iterable[str]) -> None:
        for rollout_id in rollout_ids:
            self._marks.pop(rollout_id, None)

    def marks(self) -> dict[str, int]:
        """Each retired rollout's highest retired attempt, for a worker that joins after the retires."""
        return dict(self._marks)

    def update(self, marks: dict[str, int]) -> None:
        for rollout_id, attempt in marks.items():
            self.mark([EpisodeId(rollout_id=rollout_id, attempt=attempt)])

    def __len__(self) -> int:
        return len(self._marks)


class CheckpointParticipant(ABC):
    """State owner inside one server process.

    A participant implements admission, readiness, export, and restore for the state it owns,
    and checks ``retired`` on its data plane.
    The ``ParticipantControlPlane`` does checkpoint-ID fencing, phases, deadlines, and storage.
    """

    kind: ClassVar[str]
    record_model: ClassVar[type[CheckpointRecord]]

    def __init__(self) -> None:
        self.retired = RetiredAttempts()
        self._changed = asyncio.Condition()

    async def notify(self) -> None:
        """Wake a prepare that waits for this participant to become ready."""
        async with self._changed:
            self._changed.notify_all()

    async def wait_changed(self, timeout: float) -> None:
        async with self._changed:
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=timeout)
            except TimeoutError:
                pass

    @abstractmethod
    async def close_admission(self, request: CheckpointRequest) -> None:
        """Stop admitting new work and ask running work to park, within ``request``'s deadline."""

    @abstractmethod
    async def open_admission(self) -> None:
        """Admit work again: parked work continues and held responses are delivered."""

    @abstractmethod
    def readiness(self) -> PrepareReport:
        """Report whether every live execution is at a committable boundary."""

    def ready(self) -> bool:
        """Whether ``readiness()`` would report ready.
        A prepare asks after every change, so override it when listing the blockers costs more than counting them."""
        return self.readiness().ready

    @abstractmethod
    async def retire(self, episode_id: EpisodeId) -> None:
        """Stop the running work of ``episode_id`` and earlier attempts, wait until it has stopped,
        then free their state, live or restored.
        Return only once nothing of them remains.

        If this raises or is cancelled, whatever is still running must stay tracked, so a retry stops it again.
        """

    @abstractmethod
    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        """Return the frozen records of a prepared participant, within the commit's episode scope.

        Restored state the scope leaves out must not be exported: the commit retires it after the write,
        and a retried commit must write the same records.
        """

    @abstractmethod
    def restore_records(self, records: list[CheckpointRecord]) -> None:
        """Validate every record, then install all of them under their next attempt.

        Implementations must not change live state unless every record is valid.
        """

    async def mark_retired(self, episode_ids: list[EpisodeId]) -> None:
        """Refuse these attempts from now on, before a retire stops them. Override to tell other processes as well."""
        self.retired.mark(episode_ids)

    def check_forget(self, rollout_ids: list[str]) -> None:
        """Raise if forgetting ``rollout_ids`` now would make retired state live again; ``forget`` checks first.

        A server with several workers asks every worker before any of them forgets.
        """

    async def forget(self, rollout_ids: list[str]) -> None:
        """Stop refusing these rollouts' retired attempts. Override to tell other processes as well."""
        self.retired.forget(rollout_ids)

    async def restored_pending(self) -> list[EpisodeId]:
        """Restored episodes whose replacement has not started here yet, as the attempts that continue them.

        A commit deletes the ones its scope leaves out: the controller no longer continues them,
        and nothing else would free their restored state.
        """
        return []

    async def delete_restored(self, episode_id: EpisodeId) -> None:
        """Free the restored state of ``episode_id``,
        a replacement attempt that never started here and that the controller no longer continues.

        Unlike a retire, it must leave nothing that refuses the attempt later, such as a capture-ledger fence:
        the controller may still start the rollout over as that attempt.
        The default retires it,
        which is enough for a participant whose retire refuses nothing beyond the retire itself.
        """
        await self.retire(episode_id)

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        """Export for a commit. Override to keep slow I/O off the event loop; the default is synchronous."""
        return self.export_records(episode_ids)

    async def install(self, records: list[CheckpointRecord], scope: list[EpisodeId]) -> None:
        """Install for a restore.
        ``scope`` is every episode the restore continues, including ones this participant has no record of,
        so a participant that keeps files per episode can clear theirs too.
        Override to keep slow I/O off the event loop; the default is synchronous."""
        self.restore_records(records)

    def commit_reply(self, records: list[CheckpointRecord]) -> dict[str, Any]:
        """Participant-specific fields the controller needs in the commit reply."""
        return {}

    def status_extra(self) -> dict[str, Any]:
        return {}


def next_attempt(episode_id: EpisodeId) -> EpisodeId:
    return episode_id.model_copy(update={"attempt": episode_id.attempt + 1})


@dataclass
class _Write:
    """A checkpoint's one export, the task storing it, and the commit's cleanup of unscoped restored state.

    Each runs as a task that outlives the commit call that started it: a retried commit awaits the same export,
    the same write (or stores the same records again if storing failed), and the same cleanup,
    and resume waits for an export or cleanup still running before it reopens admission.
    """

    checkpoint_id: str
    checkpoint_dir: str
    episode_ids: Optional[list[EpisodeId]]
    records: Optional[list[CheckpointRecord]] = None
    export: Optional["asyncio.Task[list[CheckpointRecord]]"] = None
    task: Optional["asyncio.Task[dict[str, Any]]"] = None
    cleanup: Optional["asyncio.Task[None]"] = None
    # Stopped by resume: the writer stops before its next record and never publishes the manifest.
    stop: WriteStop = field(default_factory=WriteStop)


class ParticipantControlPlane:
    """Checkpoint-ID fencing, phase machine, lease, and storage that drive one participant."""

    def __init__(self, participant: CheckpointParticipant, *, instance_name: str, lease_grace_seconds: float) -> None:
        self.participant = participant
        self.instance_name = instance_name
        self.lease_grace_seconds = lease_grace_seconds
        self.phase = CheckpointPhase.IDLE
        self.checkpoint_id: Optional[str] = None
        self._lease_expires_at: Optional[float] = None
        self._lease_task: Optional[asyncio.Task] = None
        # Recently resumed checkpoint IDs, so a delayed request of a stale controller cannot reopen one.
        self._resumed_ids: OrderedDict[str, None] = OrderedDict()
        # Each completed operation's result, with the arguments it ran with: a retry with other arguments conflicts.
        self._results: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
        self._write: Optional[_Write] = None
        # The last restore's install.
        # It can outlive the restore's deadline, as a thread does,
        # and a retire must not free restored state before the install that brings it finishes.
        self._install: Optional[asyncio.Task] = None
        # Reopens admission once an install that outlived its failed restore has finished.
        self._reopening: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "kind": self.participant.kind,
            "instance": self.instance_name,
            "phase": self.phase.value,
            "checkpoint_id": self.checkpoint_id,
            "lease_expires_at": self._lease_expires_at,
            # Rollouts retired and not yet forgotten: each refuses its retired attempts until ``forget``.
            "retired_rollouts": len(self.participant.retired),
            "report": self.participant.readiness().model_dump(),
            **self.participant.status_extra(),
        }

    def _recorded(self, checkpoint_id: str, operation: str, arguments: Any = None) -> Optional[dict[str, Any]]:
        """The result of ``operation`` already completed for this checkpoint, if any;
        a conflict if it ran with other arguments."""
        recorded = self._results.get((checkpoint_id, operation))
        if recorded is None:
            return None
        if recorded[0] != arguments:
            raise CheckpointConflictError(
                f"checkpoint {checkpoint_id!r} already completed {operation} with another directory or scope"
            )
        return recorded[1]

    def _admit(self, checkpoint_id: str, allowed: set[CheckpointPhase]) -> None:
        if checkpoint_id in self._resumed_ids:
            raise StaleCheckpointError(f"checkpoint {checkpoint_id!r} was already resumed")
        if self.checkpoint_id is not None and self.checkpoint_id != checkpoint_id:
            raise CheckpointConflictError(f"checkpoint {self.checkpoint_id!r} is active in phase {self.phase.value}")
        if self.phase not in allowed:
            raise InvalidPhaseError(f"not valid in phase {self.phase.value}")

    # -- lease ----------------------------------------------------------------------------------

    def _renew(self, request: CheckpointRequest) -> None:
        expires = request.deadline_ts + self.lease_grace_seconds
        self._lease_expires_at = max(self._lease_expires_at or 0.0, expires)
        if self._lease_task is None or self._lease_task.done():
            self._lease_task = asyncio.create_task(self._watch_lease(request.checkpoint_id))

    async def _watch_lease(self, checkpoint_id: str) -> None:
        while self._lease_expires_at is not None and self.checkpoint_id in (None, checkpoint_id):
            remaining = self._lease_expires_at - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)
                continue
            async with self._lock:
                if self.checkpoint_id != checkpoint_id or time.time() < (self._lease_expires_at or 0.0):
                    continue
                LOGGER.warning(
                    "checkpoint %s lease expired in phase %s; resuming %s",
                    checkpoint_id,
                    self.phase.value,
                    self.instance_name,
                )
                record_checkpoint_event(
                    "lease_expired", participant_kind=self.participant.kind, phase=self.phase.value
                )
                try:
                    await self._reopen(resume_id=checkpoint_id, deadline_ts=time.time() + _LEASE_REOPEN_WAIT_SECONDS)
                except DeadlineExceededError:
                    # Restored state is still being deleted, and a replacement may start as that attempt:
                    # stay closed and try again, letting a controller's resume in meanwhile.
                    LOGGER.warning("checkpoint %s still deleting restored state; resuming again", checkpoint_id)
                    reopened = False
                else:
                    reopened = True
            if reopened:
                return
            await asyncio.sleep(_LEASE_REOPEN_WAIT_SECONDS)

    async def renew(self, request: CheckpointRequest) -> dict[str, Any]:
        async with self._lock:
            self._admit(request.checkpoint_id, set(CheckpointPhase) - {CheckpointPhase.IDLE})
            self._renew(request)
            return {"lease_expires_at": self._lease_expires_at}

    # -- phases ---------------------------------------------------------------------------------

    async def prepare(self, request: CheckpointRequest) -> dict[str, Any]:
        with self._operation("prepare", request.checkpoint_id) as span:
            result = await self._prepare(request)
            span.set(phase=result["phase"], blockers=len(result["report"]["blockers"]))
            return result

    async def _prepare(self, request: CheckpointRequest) -> dict[str, Any]:
        async with self._lock:
            recorded = self._recorded(request.checkpoint_id, "prepare")
            if recorded is not None:
                return recorded
            self._admit(request.checkpoint_id, {CheckpointPhase.IDLE, CheckpointPhase.PREPARING})
            if self.phase == CheckpointPhase.IDLE:
                self.checkpoint_id = request.checkpoint_id
                self.phase = CheckpointPhase.PREPARING
                self._renew(request)
                try:
                    await self.participant.close_admission(request)
                except BaseException:
                    await self._reopen()
                    raise
            else:
                self._renew(request)

        try:
            with checkpoint_span("gym.checkpoint.wait_ready"):
                await self._wait_ready(request)
        except BaseException:
            async with self._lock:
                if self.checkpoint_id == request.checkpoint_id and self.phase == CheckpointPhase.PREPARING:
                    await self._reopen()
            raise

        async with self._lock:
            if self.checkpoint_id != request.checkpoint_id or self.phase != CheckpointPhase.PREPARING:
                # A resume, lease expiry, or concurrent prepare finished this checkpoint while we waited.
                return {"phase": self.phase.value, "report": self.participant.readiness().model_dump()}
            report = self.participant.readiness()
            if report.ready:
                self.phase = CheckpointPhase.PREPARED
            elif time.time() >= request.deadline_ts:
                record_checkpoint_event("prepare_not_ready", participant_kind=self.participant.kind)
            result = {"phase": self.phase.value, "report": report.model_dump()}
            if report.ready:
                self._results[(request.checkpoint_id, "prepare")] = (None, result)
            return result

    async def _wait_ready(self, request: CheckpointRequest) -> None:
        # Only whether the participant is ready: the blockers are listed once, for the reply.
        while True:
            remaining = request.deadline_ts - time.time()
            if self.participant.ready() or remaining <= 0 or self.checkpoint_id != request.checkpoint_id:
                return
            await self.participant.wait_changed(min(remaining, 1.0))

    async def retire(self, request: RetireRequest) -> dict[str, Any]:
        """Refuse these attempts, stop their running work and free their state, then reply.

        Allowed when idle, and after a restore, where nothing is running yet.
        While a checkpoint is open it is refused: stopping an episode waits for its final cleanup,
        whose calls wait for resume.
        A controller with a straggler resumes first.
        The attempts stay refused, even if this retire fails, until ``forget``.
        """
        with self._operation("retire", request.checkpoint_id) as span:
            span.set(episodes=len(request.episode_ids))
            return await self._retire(request)

    async def _retire(self, request: RetireRequest) -> dict[str, Any]:
        async with self._lock:
            self._admit_outside_checkpoint(request)
            await _within(request, self.participant.mark_retired(request.episode_ids))
            await _within(request, self._stop(request.episode_ids))
        await self.participant.notify()
        return {"retired": [episode_id.capture_key for episode_id in request.episode_ids]}

    async def forget(self, request: ForgetRequest) -> dict[str, Any]:
        """Stop refusing these rollouts' retired attempts: the controller is done with them for good.

        Callers before callees, like retire, and only once nothing of any attempt of these rollouts can still run.
        """
        with self._operation("forget", request.checkpoint_id) as span:
            span.set(rollouts=len(request.rollout_ids))
            async with self._lock:
                self._admit_outside_checkpoint(request)
                await _within(request, self.participant.forget(request.rollout_ids))
            return {"forgotten": sorted(request.rollout_ids)}

    def _admit_outside_checkpoint(self, request: CheckpointRequest) -> None:
        if self.phase not in (CheckpointPhase.IDLE, CheckpointPhase.RESTORED):
            raise InvalidPhaseError(f"not valid in phase {self.phase.value}; resume first")
        if self.checkpoint_id is not None and self.checkpoint_id != request.checkpoint_id:
            raise CheckpointConflictError(f"checkpoint {self.checkpoint_id!r} is active")
        if self.phase != CheckpointPhase.IDLE:
            self._renew(request)

    async def _stop(self, episode_ids: list[EpisodeId]) -> None:
        await self._installed()
        for episode_id in episode_ids:
            await self.participant.retire(episode_id)

    def _reopen_after_failed_restore(self, install: asyncio.Task) -> None:
        self._reopening = asyncio.ensure_future(self._open_after_failed_restore(install))

    async def _open_after_failed_restore(self, install: Optional[asyncio.Task] = None) -> None:
        if install is not None:
            async with self._lock:
                # A retried restore that succeeded meanwhile, or the next one under way, owns admission now.
                if (
                    self.phase != CheckpointPhase.IDLE
                    or self.checkpoint_id is not None
                    or self._install is not install
                ):
                    return
                await self._open_after_failed_restore()
            return
        try:
            await self.participant.open_admission()
        except Exception:
            LOGGER.exception("reopening admission after a failed restore failed")

    async def _installed(self) -> None:
        """Wait for an install still running after its restore's deadline; its outcome was already reported."""
        if self._install is not None and not self._install.done():
            await asyncio.wait([self._install])

    async def _delete_unscoped(self, episode_ids: Optional[list[EpisodeId]]) -> None:
        """Delete restored state the commit's scope leaves out: the scope is everything the controller continues.

        Their attempts are not refused: a replacement that never started here cannot send a late request,
        and the controller may still start the rollout over as that attempt.
        """
        if episode_ids is None:
            return
        scope = set(episode_ids)
        for episode_id in await self.participant.restored_pending():
            if episode_id not in scope:
                await self.participant.delete_restored(episode_id)

    async def commit(self, request: CommitRequest) -> dict[str, Any]:
        with self._operation("commit", request.checkpoint_id) as span:
            result = await self._commit(request)
            span.set(records=result["manifest"]["record_count"])
            return result

    async def _commit(self, request: CommitRequest) -> dict[str, Any]:
        async with self._lock:
            arguments = (request.checkpoint_dir, _scope_key(request.episode_ids))
            recorded = self._recorded(request.checkpoint_id, "commit", arguments)
            if recorded is not None:
                return recorded
            self._admit(request.checkpoint_id, {CheckpointPhase.PREPARED})
            self._renew(request)
            # Readiness can regress after prepare if participants were prepared out of order,
            # for example an agent whose held model response was delivered.
            # Never commit such a cut.
            report = self.participant.readiness()
            if not report.ready:
                raise InvalidPhaseError(f"participant is no longer ready to commit: blockers={report.blockers}")
            if request.episode_ids is not None:
                # Fail closed: this participant holds none of a restart's state,
                # so continuing it would resume a rollout whose other parts start over.
                restarted = sorted(
                    set(report.restarts) & {episode_id.capture_key for episode_id in request.episode_ids}
                )
                if restarted:
                    raise RestartInScopeError(f"the commit scope names episodes that restart: {restarted}")
            write = await self._start_write(request)
            # The thread cannot be cancelled: a write that outlives the deadline fails this call and keeps going,
            # and a retry awaits the same write.
            with checkpoint_span("gym.checkpoint.write") as span:
                manifest = await _within(request, asyncio.shield(write.task))
                size = _records_size(
                    Path(request.checkpoint_dir), self.participant.kind, self.instance_name, manifest["records_file"]
                )
                span.set(records=manifest["record_count"], bytes=size)
            record_checkpoint_volume(
                operation="commit", kind=self.participant.kind, records=manifest["record_count"], size_bytes=size
            )
            records = write.records
            # Restored state outside the scope was not exported,
            # so a retry after a failure here returns the same manifest.
            # The cleanup also outlives a deadline, and resume waits for it:
            # a replacement may start as the attempt it deletes.
            if write.cleanup is None or _failed(write.cleanup):
                write.cleanup = _background(self._delete_unscoped(request.episode_ids))
            await _within(request, asyncio.shield(write.cleanup))
            self.phase = CheckpointPhase.COMMITTED
            result = {
                "phase": self.phase.value,
                "manifest": manifest,
                "episode_ids": sorted({record.episode_id.capture_key for record in records}),
                **self.participant.commit_reply(records),
            }
            self._results[(request.checkpoint_id, "commit")] = (arguments, result)
            return result

    async def _start_write(self, request: CommitRequest) -> _Write:
        """The checkpoint's write: the running or finished one, or the same records stored again if storing failed."""
        write = self._write
        if write is None or write.checkpoint_id != request.checkpoint_id:
            write = self._write = _Write(request.checkpoint_id, request.checkpoint_dir, request.episode_ids)
        elif (write.checkpoint_dir, _scope_key(write.episode_ids)) != (
            request.checkpoint_dir,
            _scope_key(request.episode_ids),
        ):
            raise CheckpointConflictError(
                f"checkpoint {request.checkpoint_id!r} is already being written to another directory or scope"
            )
        if write.records is None:
            # An export that outlives the deadline keeps running, and a retry awaits it:
            # exporting again could take a later cut than the other participants already committed.
            if write.export is None or _failed(write.export):
                write.export = _background(self.participant.export(request.episode_ids))
            with checkpoint_span("gym.checkpoint.export") as span:
                write.records = await _within(request, asyncio.shield(write.export))
                span.set(records=len(write.records))
        if write.task is None or _failed(write.task):
            write.task = asyncio.create_task(
                asyncio.to_thread(
                    write_participant_state,
                    Path(request.checkpoint_dir),
                    kind=self.participant.kind,
                    instance=self.instance_name,
                    checkpoint_id=request.checkpoint_id,
                    # Converted in the writer thread:
                    # exported state is not changed in place until resume, which stops the writer.
                    records=(record.to_json_record() for record in write.records),
                    stop=write.stop,
                )
            )
            _retrieve_exception(write.task)
        return write

    async def restore(self, request: RestoreRequest) -> dict[str, Any]:
        with self._operation("restore", request.checkpoint_id) as span:
            result = await self._restore(request)
            span.set(records=len(result["restored"]))
            return result

    async def _restore(self, request: RestoreRequest) -> dict[str, Any]:
        async with self._lock:
            arguments = (request.checkpoint_dir, _scope_key(request.episode_ids))
            recorded = self._recorded(request.checkpoint_id, "restore", arguments)
            if recorded is not None:
                return recorded
            self._admit(request.checkpoint_id, {CheckpointPhase.IDLE})
            # One streaming pass in a thread keeps and validates only the records in scope:
            # the scope can be much smaller than the checkpoint.
            scope = {episode_id.capture_key for episode_id in request.episode_ids}
            model = self.participant.record_model
            read = asyncio.to_thread(
                read_participant_state,
                Path(request.checkpoint_dir),
                kind=self.participant.kind,
                instance=self.instance_name,
                select=lambda record: _validate_record(model, record) if _record_key(record) in scope else None,
            )
            with checkpoint_span("gym.checkpoint.read") as span:
                manifest, records = await _within(request, read)
                size = _records_size(
                    Path(request.checkpoint_dir), self.participant.kind, self.instance_name, manifest["records_file"]
                )
                span.set(records=len(records), bytes=size)
            record_checkpoint_volume(
                operation="restore", kind=self.participant.kind, records=len(records), size_bytes=size
            )
            await _within(request, self._installed())
            try:
                await self.participant.close_admission(request)
                self._install = asyncio.ensure_future(self.participant.install(records, request.episode_ids))
                # A restore that fails after this retires what the install brings: the retire waits for it.
                self._install.add_done_callback(lambda done: done.cancelled() or done.exception())
                with checkpoint_span("gym.checkpoint.install") as span:
                    span.set(records=len(records))
                    await _within(request, asyncio.shield(self._install))
            except BaseException:
                # Closing can fail part way, for example on one of several workers;
                # nothing else would reopen the rest, because the phase is still idle.
                # An install that outlived the deadline is still changing state,
                # so admission reopens only once it has finished.
                if self._install is not None and not self._install.done():
                    install = self._install
                    install.add_done_callback(lambda _: self._reopen_after_failed_restore(install))
                else:
                    await self._open_after_failed_restore()
                raise
            self.checkpoint_id = request.checkpoint_id
            self.phase = CheckpointPhase.RESTORED
            self._renew(request)
            result = {
                "phase": self.phase.value,
                "source_checkpoint_id": manifest["checkpoint_id"],
                "restored": sorted(next_attempt(record.episode_id).capture_key for record in records),
            }
            self._results[(request.checkpoint_id, "restore")] = (arguments, result)
            return result

    async def resume(self, request: CheckpointRequest) -> dict[str, Any]:
        with self._operation("resume", request.checkpoint_id):
            return await self._resume(request)

    async def _resume(self, request: CheckpointRequest) -> dict[str, Any]:
        async with self._lock:
            if request.checkpoint_id in self._resumed_ids:
                return {"phase": CheckpointPhase.IDLE.value, "idempotent": True}
            if self.phase == CheckpointPhase.IDLE and self.checkpoint_id is None:
                # A prepare that stopped at an earlier stage never reached this participant.
                # Fence the ID anyway so a delayed prepare for the aborted checkpoint cannot close admission.
                self._remember_resumed(request.checkpoint_id)
                return {"phase": CheckpointPhase.IDLE.value, "idempotent": True}
            self._admit(request.checkpoint_id, set(CheckpointPhase) - {CheckpointPhase.IDLE})
            await self._reopen(resume_id=request.checkpoint_id, deadline_ts=request.deadline_ts)
            return {"phase": self.phase.value, "idempotent": False}

    def _remember_resumed(self, checkpoint_id: str) -> None:
        self._resumed_ids[checkpoint_id] = None
        while len(self._resumed_ids) > _RESUMED_IDS_KEPT:
            self._resumed_ids.popitem(last=False)

    def _operation(self, name: str, checkpoint_id: str) -> AbstractContextManager[OperationSpan]:
        return checkpoint_operation(
            name, kind=self.participant.kind, instance=self.instance_name, checkpoint_id=checkpoint_id
        )

    async def _reopen(self, *, resume_id: Optional[str] = None, deadline_ts: Optional[float] = None) -> None:
        """Reopen admission.

        Raises ``DeadlineExceededError``, changing nothing a retry depends on,
        if restored state the last commit is deleting is still being deleted at ``deadline_ts``.
        """
        if self._write is not None:
            # Waits only for a manifest publication already under way, off the event loop: once it returns,
            # the write never publishes.
            await asyncio.to_thread(self._write.stop.stop)
            # Nothing will use an export still running.
            if self._write.export is not None and not self._write.export.done():
                self._write.export.cancel()
            # A cleanup still deleting restored state must finish first: a replacement may start as that attempt.
            cleanup = self._write.cleanup
            if cleanup is not None and not cleanup.done():
                remaining = None if deadline_ts is None else max(0.0, deadline_ts - time.time())
                await asyncio.wait([cleanup], timeout=remaining)
                if not cleanup.done():
                    raise DeadlineExceededError("restored state is still being deleted; resume again")
            self._write = None
        await self.participant.open_admission()
        if resume_id is not None:
            self._remember_resumed(resume_id)
            self._results = {key: value for key, value in self._results.items() if key[0] != resume_id}
        self.phase = CheckpointPhase.IDLE
        self.checkpoint_id = None
        self._lease_expires_at = None
        if self._lease_task is not None and self._lease_task is not asyncio.current_task():
            self._lease_task.cancel()
        self._lease_task = None
        await self.participant.notify()


def _scope_key(episode_ids: Optional[list[EpisodeId]]) -> Optional[tuple[str, ...]]:
    return None if episode_ids is None else tuple(sorted(episode_id.capture_key for episode_id in episode_ids))


def _background(operation: Any) -> asyncio.Task:
    """Run ``operation`` as a task that a deadline on whoever awaits it does not cancel."""
    task = asyncio.ensure_future(operation)
    _retrieve_exception(task)
    return task


def _retrieve_exception(task: asyncio.Task) -> None:
    # A task nobody awaits any more, after a deadline or a resume, must not log an unretrieved exception.
    task.add_done_callback(lambda done: done.cancelled() or done.exception())


def _failed(task: asyncio.Task) -> bool:
    """Whether ``task`` ended without a result, so a retry must run it again."""
    return task.done() and (task.cancelled() or task.exception() is not None)


def _record_key(record: Any) -> Optional[str]:
    """The capture key of a stored record, without validating the rest of it."""
    try:
        return EpisodeId.model_validate(record["episode_id"]).capture_key
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointStateError(f"checkpoint record has no valid episode_id: {error}") from error


def _validate_record(model: type[CheckpointRecord], record: Any) -> CheckpointRecord:
    try:
        return model.model_validate(record)
    except ValidationError as error:
        raise CheckpointStateError(
            f"checkpoint record of episode {_record_key(record)!r} is invalid: {error}"
        ) from error


async def _within(request: CheckpointRequest, operation: Any) -> Any:
    try:
        async with asyncio.timeout(request.remaining()):
            return await operation
    except TimeoutError as error:
        raise DeadlineExceededError(f"checkpoint {request.checkpoint_id!r} deadline passed during I/O") from error


def _require_bearer(authorization: Optional[str], token: str) -> None:
    if authorization is None or not hmac.compare_digest(authorization, f"Bearer {token}"):
        raise UnauthorizedError("missing or invalid checkpoint control bearer token")


# Validated request body of each mutating control operation.
_OPERATION_REQUESTS: dict[str, type[CheckpointRequest]] = {
    "prepare": CheckpointRequest,
    "renew": CheckpointRequest,
    "retire": RetireRequest,
    "forget": ForgetRequest,
    "commit": CommitRequest,
    "restore": RestoreRequest,
    "resume": CheckpointRequest,
}

# Runs one control operation ("status" or a key of ``_OPERATION_REQUESTS``) and returns its reply.
ControlDispatch = Callable[[str, Optional[dict[str, Any]]], Awaitable[dict[str, Any]]]


async def dispatch_control(
    controller: "ParticipantControlPlane", operation: str, body: Optional[dict[str, Any]]
) -> dict[str, Any]:
    if operation == "status":
        return controller.status()
    request = _OPERATION_REQUESTS[operation].model_validate(body)
    return await getattr(controller, operation)(request)


def install_control_routes(app: FastAPI, dispatch: ControlDispatch, *, auth_token: str) -> None:
    """Register the bearer-protected ``/ng-control/v1/checkpoint`` routes, served by ``dispatch``.

    A participant in this process dispatches to its controller;
    a server with several workers dispatches to the one controller that coordinates them.
    """
    if not auth_token:
        raise ValueError("checkpoint control routes require a non-empty bearer token")
    router = APIRouter(prefix=CHECKPOINT_ROUTE_PREFIX)

    @router.get("/status")
    async def status(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
        _require_bearer(authorization, auth_token)
        return await dispatch("status", None)

    def route(operation: str, request_model: type[CheckpointRequest]) -> None:
        async def handle(body: request_model, authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:  # type: ignore[valid-type]
            _require_bearer(authorization, auth_token)
            return await dispatch(operation, body.model_dump(mode="json"))

        router.add_api_route(f"/{operation}", handle, methods=["POST"], name=operation)

    for operation, request_model in _OPERATION_REQUESTS.items():
        route(operation, request_model)
    app.include_router(router)
    install_control_error_handler(app)


def install_participant(
    app: FastAPI,
    participant: CheckpointParticipant,
    *,
    auth_token: str,
    instance_name: str,
    lease_grace_seconds: float,
) -> ParticipantControlPlane:
    """Attach ``participant`` to ``app`` and register its bearer-protected control routes."""
    if not auth_token:
        raise ValueError("checkpoint control routes require a non-empty bearer token")
    controller = ParticipantControlPlane(
        participant, instance_name=instance_name, lease_grace_seconds=lease_grace_seconds
    )

    async def dispatch(operation: str, body: Optional[dict[str, Any]]) -> dict[str, Any]:
        return await dispatch_control(controller, operation, body)

    install_control_routes(app, dispatch, auth_token=auth_token)
    return controller


def install_control_error_handler(app: FastAPI) -> None:
    async def handle(request: Request, error: ControlError) -> JSONResponse:
        return error.response()

    app.add_exception_handler(ControlError, handle)


def _records_size(checkpoint_dir: Path, kind: str, instance: str, records_file: str) -> int:
    path = participant_dir(checkpoint_dir, kind=kind, instance=instance) / records_file
    try:
        return path.stat().st_size
    except OSError:
        return 0
