# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint control plane shared by every participating server.

A participant is the part of one server process that owns checkpointable state. Every participant
exposes the same bearer-protected routes under ``/ng-control/v1/checkpoint`` and moves through one
phase machine::

    idle --prepare--> preparing --(ready)--> prepared --commit--> committed
    idle --restore--> restored
    preparing | prepared | committed | restored --resume--> idle

The controller here owns everything that is the same for every participant:

- checkpoint-ID fencing, phases, and idempotent replay of completed operations;
- attempt fencing: retiring or restoring an attempt fences it, so a request from a replaced attempt
  gets 409 ``stale_attempt`` from the participant's data plane;
- the readiness wait, deadlines, and manifest-last storage;
- a lease: if the controller that started a checkpoint stops calling, the participant resumes on its
  own when the lease expires, which is the same as an abort.

``resume`` also aborts: it reopens admission and releases parked work whether or not the checkpoint
was published. Resuming retires the checkpoint ID, so a delayed call from a stale controller cannot
reopen or overwrite anything.

Transitions are serialized by one lock. Prepare waits for readiness outside the lock, so a controller
can retire a straggler or resume while a prepare is waiting. A prepare that misses its deadline
returns its blockers and leaves admission closed. A prepare that raises reopens admission, so phase
and admission never disagree.
"""

import asyncio
import hmac
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Optional

import orjson
from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PlainValidator, model_validator

from nemo_gym._checkpoint.errors import (
    CheckpointConflictError,
    ControlError,
    DeadlineExceededError,
    InvalidPhaseError,
    StaleAttemptError,
    StaleCheckpointError,
    UnauthorizedError,
)
from nemo_gym._checkpoint.store import read_participant_state, write_participant_state
from nemo_gym.episode_types import EpisodeId


LOGGER = logging.getLogger(__name__)

CHECKPOINT_ROUTE_PREFIX = "/ng-control/v1/checkpoint"
CHECKPOINT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


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


def _check_json(value: Any) -> Any:
    """Accept only what JSON can carry, as ``JsonValue`` would, without walking the value in Python.

    ``orjson`` rejects non-string keys and non-JSON types natively. Its output is discarded: records are written
    with the standard ``json`` module, which round-trips NaN and infinities exactly.
    """
    try:
        orjson.dumps(value)
    except TypeError as error:
        raise ValueError(f"checkpoint payload is not JSON: {error}") from error
    return value


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

    At most ``MAX_REPORTED_BLOCKERS`` blocker keys are listed so a prepare that misses its deadline with
    thousands of stragglers still returns a small reply; ``blocker_count`` is always the full count.
    """

    ready: bool
    blockers: list[str] = Field(default_factory=list, description="Capture keys still preventing a safe cut.")
    blocker_count: int = 0
    counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def cap_blockers(cls, data: Any) -> Any:
        if isinstance(data, dict) and "blocker_count" not in data:
            blockers = list(data.get("blockers") or [])
            data = {**data, "blockers": blockers[:MAX_REPORTED_BLOCKERS], "blocker_count": len(blockers)}
        return data


class AttemptFence:
    """Reject requests from attempts that a retire or restore replaced.

    Only the lowest live attempt per rollout is kept, so fencing survives any number of restores and
    stays one integer per rollout that a checkpoint operation touched.
    """

    def __init__(self) -> None:
        self._min_attempt: dict[str, int] = {}

    def check(self, episode_id: EpisodeId) -> None:
        minimum = self._min_attempt.get(episode_id.rollout_id, 0)
        if episode_id.attempt < minimum:
            raise StaleAttemptError(
                f"rollout {episode_id.rollout_id!r} attempt {episode_id.attempt} was replaced by attempt {minimum}"
            )

    def retire(self, episode_id: EpisodeId) -> None:
        """Fence ``episode_id`` and every earlier attempt of its rollout."""
        rollout_id = episode_id.rollout_id
        self._min_attempt[rollout_id] = max(self._min_attempt.get(rollout_id, 0), episode_id.attempt + 1)

    def minimums(self) -> dict[str, int]:
        """The lowest live attempt of every fenced rollout, for copying the fence to another process."""
        return dict(self._min_attempt)

    def raise_to(self, minimums: dict[str, int]) -> None:
        """Apply another fence's minimums; a fence only ever rises."""
        for rollout_id, minimum in minimums.items():
            self._min_attempt[rollout_id] = max(self._min_attempt.get(rollout_id, 0), minimum)


class CheckpointParticipant(ABC):
    """State owner inside one server process.

    A participant implements admission, readiness, export, and restore for the state it owns, and
    checks ``attempts`` on its data plane. The ``ParticipantController`` does all fencing, phases,
    deadlines, and storage.
    """

    kind: ClassVar[str]
    record_model: ClassVar[type[CheckpointRecord]]

    def __init__(self) -> None:
        self.attempts = AttemptFence()
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
        """Admit work again and release everything parked or held."""

    @abstractmethod
    def readiness(self) -> PrepareReport:
        """Report whether every live execution is at a committable boundary."""

    @abstractmethod
    async def retire(self, episode_id: EpisodeId) -> None:
        """Discard the state of ``episode_id`` and earlier attempts, live or restored."""

    @abstractmethod
    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        """Return the frozen records of a prepared participant, within the commit's episode scope."""

    @abstractmethod
    def restore_records(self, records: list[CheckpointRecord]) -> None:
        """Validate every record, then install all of them under their next attempt.

        Implementations must not change live state unless every record is valid.
        """

    async def restored_pending(self) -> list[EpisodeId]:
        """Restored episodes whose replacement has not started here yet, as the attempts that continue them.

        A commit retires the ones its scope leaves out: the controller no longer continues them, and nothing
        else would release their restored state.
        """
        return []

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        """Export for a commit. Override to keep slow I/O off the event loop; the default is synchronous."""
        return self.export_records(episode_ids)

    async def install(self, records: list[CheckpointRecord]) -> None:
        """Install for a restore. Override to keep slow I/O off the event loop; the default is synchronous."""
        self.restore_records(records)

    def commit_reply(self, records: list[CheckpointRecord]) -> dict[str, Any]:
        """Participant-specific fields the controller needs in the commit reply."""
        return {}

    def status_extra(self) -> dict[str, Any]:
        return {}


def next_attempt(episode_id: EpisodeId) -> EpisodeId:
    return episode_id.model_copy(update={"attempt": episode_id.attempt + 1})


class ParticipantController:
    """Fence, phase machine, lease, and storage that drive one participant."""

    def __init__(self, participant: CheckpointParticipant, *, instance_name: str, lease_grace_seconds: float) -> None:
        self.participant = participant
        self.instance_name = instance_name
        self.lease_grace_seconds = lease_grace_seconds
        self.phase = CheckpointPhase.IDLE
        self.checkpoint_id: Optional[str] = None
        self._lease_expires_at: Optional[float] = None
        self._lease_task: Optional[asyncio.Task] = None
        self._retired_ids: set[str] = set()
        self._results: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "kind": self.participant.kind,
            "instance": self.instance_name,
            "phase": self.phase.value,
            "checkpoint_id": self.checkpoint_id,
            "lease_expires_at": self._lease_expires_at,
            "report": self.participant.readiness().model_dump(),
            **self.participant.status_extra(),
        }

    def _admit(self, checkpoint_id: str, allowed: set[CheckpointPhase]) -> None:
        if checkpoint_id in self._retired_ids:
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
                await self._reopen(retire_id=checkpoint_id)
            return

    async def renew(self, request: CheckpointRequest) -> dict[str, Any]:
        async with self._lock:
            self._admit(request.checkpoint_id, set(CheckpointPhase) - {CheckpointPhase.IDLE})
            self._renew(request)
            return {"lease_expires_at": self._lease_expires_at}

    # -- phases ---------------------------------------------------------------------------------

    async def prepare(self, request: CheckpointRequest) -> dict[str, Any]:
        async with self._lock:
            recorded = self._results.get((request.checkpoint_id, "prepare"))
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
            report = await self._wait_ready(request)
        except BaseException:
            async with self._lock:
                if self.checkpoint_id == request.checkpoint_id and self.phase == CheckpointPhase.PREPARING:
                    await self._reopen()
            raise

        async with self._lock:
            if self.checkpoint_id != request.checkpoint_id or self.phase != CheckpointPhase.PREPARING:
                # A resume, lease expiry, or concurrent prepare finished this checkpoint while we waited.
                return {"phase": self.phase.value, "report": report.model_dump()}
            report = self.participant.readiness()
            if report.ready:
                self.phase = CheckpointPhase.PREPARED
            result = {"phase": self.phase.value, "report": report.model_dump()}
            if report.ready:
                self._results[(request.checkpoint_id, "prepare")] = result
            return result

    async def _wait_ready(self, request: CheckpointRequest) -> PrepareReport:
        while True:
            report = self.participant.readiness()
            remaining = request.deadline_ts - time.time()
            if report.ready or remaining <= 0 or self.checkpoint_id != request.checkpoint_id:
                return report
            await self.participant.wait_changed(min(remaining, 1.0))

    async def retire(self, request: RetireRequest) -> dict[str, Any]:
        """Discard attempts during a checkpoint, after a restore, or when idle.

        Outside a checkpoint the controller uses this to drop restored state it will not continue.
        """
        async with self._lock:
            if self.checkpoint_id is not None and self.checkpoint_id != request.checkpoint_id:
                raise CheckpointConflictError(f"checkpoint {self.checkpoint_id!r} is active")
            if self.phase == CheckpointPhase.COMMITTED:
                raise InvalidPhaseError("cannot retire after commit; resume first")
            if self.phase != CheckpointPhase.IDLE:
                self._renew(request)
            for episode_id in request.episode_ids:
                self.participant.attempts.retire(episode_id)
                await self.participant.retire(episode_id)
        await self.participant.notify()
        return {"retired": [episode_id.capture_key for episode_id in request.episode_ids]}

    async def _release_unscoped(self, episode_ids: Optional[list[EpisodeId]]) -> None:
        """Retire restored episodes the commit's scope leaves out: the scope is everything the controller continues."""
        if episode_ids is None:
            return
        scope = set(episode_ids)
        for episode_id in await self.participant.restored_pending():
            if episode_id not in scope:
                self.participant.attempts.retire(episode_id)
                await self.participant.retire(episode_id)

    async def commit(self, request: CommitRequest) -> dict[str, Any]:
        async with self._lock:
            recorded = self._results.get((request.checkpoint_id, "commit"))
            if recorded is not None:
                return recorded
            self._admit(request.checkpoint_id, {CheckpointPhase.PREPARED})
            self._renew(request)
            # Readiness can regress after prepare if participants were prepared out of order, for
            # example an agent whose held model response was delivered. Never commit such a cut.
            report = self.participant.readiness()
            if not report.ready:
                raise InvalidPhaseError(f"participant is no longer ready to commit: blockers={report.blockers}")
            records = await _within(request, self.participant.export(request.episode_ids))
            write = asyncio.to_thread(
                write_participant_state,
                Path(request.checkpoint_dir),
                kind=self.participant.kind,
                instance=self.instance_name,
                checkpoint_id=request.checkpoint_id,
                records=[record.to_json_record() for record in records],
            )
            # A write that outlives the deadline fails this call; a retry returns the same manifest.
            manifest = await _within(request, write)
            await self._release_unscoped(request.episode_ids)
            self.phase = CheckpointPhase.COMMITTED
            result = {
                "phase": self.phase.value,
                "manifest": manifest,
                "episode_ids": sorted({record.episode_id.capture_key for record in records}),
                **self.participant.commit_reply(records),
            }
            self._results[(request.checkpoint_id, "commit")] = result
            return result

    async def restore(self, request: RestoreRequest) -> dict[str, Any]:
        async with self._lock:
            recorded = self._results.get((request.checkpoint_id, "restore"))
            if recorded is not None:
                return recorded
            self._admit(request.checkpoint_id, {CheckpointPhase.IDLE})
            read = asyncio.to_thread(
                read_participant_state,
                Path(request.checkpoint_dir),
                kind=self.participant.kind,
                instance=self.instance_name,
            )
            manifest, raw_records = await _within(request, read)
            scope = set(request.episode_ids)
            records = [self.participant.record_model.model_validate(record) for record in raw_records]
            records = [record for record in records if record.episode_id in scope]
            await self.participant.close_admission(request)
            try:
                await _within(request, self.participant.install(records))
            except BaseException:
                await self.participant.open_admission()
                raise
            for record in records:
                self.participant.attempts.retire(record.episode_id)
            self.checkpoint_id = request.checkpoint_id
            self.phase = CheckpointPhase.RESTORED
            self._renew(request)
            result = {
                "phase": self.phase.value,
                "source_checkpoint_id": manifest["checkpoint_id"],
                "restored": sorted(next_attempt(record.episode_id).capture_key for record in records),
            }
            self._results[(request.checkpoint_id, "restore")] = result
            return result

    async def resume(self, request: CheckpointRequest) -> dict[str, Any]:
        async with self._lock:
            if request.checkpoint_id in self._retired_ids:
                return {"phase": CheckpointPhase.IDLE.value, "idempotent": True}
            if self.phase == CheckpointPhase.IDLE and self.checkpoint_id is None:
                # A prepare that stopped at an earlier stage never reached this participant. Fence the
                # ID anyway so a delayed prepare for the aborted checkpoint cannot close admission.
                self._retired_ids.add(request.checkpoint_id)
                return {"phase": CheckpointPhase.IDLE.value, "idempotent": True}
            self._admit(request.checkpoint_id, set(CheckpointPhase) - {CheckpointPhase.IDLE})
            await self._reopen(retire_id=request.checkpoint_id)
            return {"phase": self.phase.value, "idempotent": False}

    async def _reopen(self, *, retire_id: Optional[str] = None) -> None:
        await self.participant.open_admission()
        if retire_id is not None:
            self._retired_ids.add(retire_id)
            self._results = {key: value for key, value in self._results.items() if key[0] != retire_id}
        self.phase = CheckpointPhase.IDLE
        self.checkpoint_id = None
        self._lease_expires_at = None
        if self._lease_task is not None and self._lease_task is not asyncio.current_task():
            self._lease_task.cancel()
        self._lease_task = None
        await self.participant.notify()


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
    "commit": CommitRequest,
    "restore": RestoreRequest,
    "resume": CheckpointRequest,
}

# Runs one control operation ("status" or a key of ``_OPERATION_REQUESTS``) and returns its reply.
ControlDispatch = Callable[[str, Optional[dict[str, Any]]], Awaitable[dict[str, Any]]]


async def dispatch_control(
    controller: "ParticipantController", operation: str, body: Optional[dict[str, Any]]
) -> dict[str, Any]:
    if operation == "status":
        return controller.status()
    request = _OPERATION_REQUESTS[operation].model_validate(body)
    return await getattr(controller, operation)(request)


def install_control_routes(app: FastAPI, dispatch: ControlDispatch, *, auth_token: str) -> None:
    """Register the bearer-protected ``/ng-control/v1/checkpoint`` routes, served by ``dispatch``.

    A participant in this process dispatches to its controller; a server with several workers
    dispatches to the one controller that coordinates them.
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
) -> ParticipantController:
    """Attach ``participant`` to ``app`` and register its bearer-protected control routes."""
    if not auth_token:
        raise ValueError("checkpoint control routes require a non-empty bearer token")
    controller = ParticipantController(
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
