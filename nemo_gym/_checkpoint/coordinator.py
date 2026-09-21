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
"""Multi-worker admission coordination.

A uvicorn worker pool breaks the single-process admission story in a
specific way: each worker holds its own ``AdmissionLimiter``, so a control
request handled by one arbitrary data-plane worker closes one worker's
admission and reports one worker's in-flight count as if it were the
service's. The coordinator fixes this by owning the service-level truth:

- A companion coordinator (started before the worker pool) listens on a
  Unix-domain socket. Every worker connects at startup, registers, and holds
  the connection open.
- The coordinator pushes checkpoint-state changes (close, resume, tombstone)
  down every connection; each worker applies them to its in-process limiter
  and acknowledges with the state sequence number it installed.
- Workers report their local in-flight count whenever it changes. The
  coordinator reports ``paused`` only when every live worker has
  acknowledged the closed state AND the summed in-flight count is zero.
- Close freezes the exact connected worker IDs. Missing or excess workers
  reject close, and registrations remain closed until resume so replacement
  processes cannot substitute for frozen membership.

The message protocol uses length-prefixed, size-bounded JSON frames:
``register``, ``ack``, ``counters`` upstream; ``state`` downstream. The
transport is a Unix-domain socket because the coordinator and its workers
are one service on one host; nothing here crosses machines.
"""

import asyncio
import json
import logging
import struct
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Optional

from fastapi import FastAPI, Header, Query

from nemo_gym._checkpoint.admission import AdmissionLimiter
from nemo_gym._checkpoint.control import (
    AdmissionState,
    CheckpointPhase,
    ControlCapabilities,
    ControlError,
    ControlFence,
    Deadline,
    install_control_plane,
)
from nemo_gym._checkpoint.model_control_contracts import (
    MODEL_ADMISSION_URL_PREFIX,
    GenerationCutCoordinatorProof,
    GenerationCutWorkerProof,
    ModelAbortInflightRequest,
    ModelAdmissionPauseRequest,
    ModelAdmissionResumeRequest,
)
from nemo_gym.token_id_capture.control_routes import require_control_auth


class MissingWorkersError(ControlError):
    """Expected workers are not connected to the coordinator.

    A missing worker may hold in-flight requests the coordinator cannot see,
    so it is an error to proceed — never an implicit zero.
    """

    code = "missing_workers"


class WorkerRegistrationError(ControlError):
    """A worker attempted to join while checkpoint membership was frozen."""

    code = "worker_registration_rejected"


class CoordinatorServiceError(ControlError):
    """A coordinator-owned service operation failed."""

    def __init__(
        self,
        detail: str,
        *,
        code: str = "coordinator_service_error",
        status_code: int = 409,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.status_code = status_code


class RestoredCutAlreadyOwnedError(ControlError):
    """A second model request attempted to consume a leased restored cut."""

    code = "restored_cut_already_owned"


class RestoredCutConsumedError(ControlError):
    """A duplicate request attempted to reuse an already consumed cut."""

    code = "restored_cut_consumed"


CHECKPOINT_COORDINATOR_SOCKET_ENV = "NG_CHECKPOINT_COORDINATOR_SOCKET"

_LOG = logging.getLogger(__name__)
_FRAME_HEADER = struct.Struct("!I")
# A scale checkpoint may report thousands of lineage identities and frozen
# generation tickets in one worker acknowledgement. Keep ample headroom above
# those expected multi-megabyte messages while rejecting corrupt lengths before
# allocating or buffering an unbounded payload.
_MAX_COORDINATOR_PAYLOAD_BYTES = 32 * 1024 * 1024


class _CoordinatorProtocolError(ValueError):
    """A coordinator socket frame is malformed or exceeds its size bound."""


_LEASE_MUTATING_SERVICE_OPERATIONS = frozenset(
    {
        "abandon_generation_cut_claim",
        "claim_generation_cut",
        "consume_generation_cut",
        "release_generation_cut",
    }
)


def _identity_set(values: Iterable[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(value["rollout_id"]), int(value["attempt_index"])) for value in values}


class _RestoredCutEntry:
    """Coordinator-local ownership state for one restored generation cut."""

    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value
        self.consumed = False
        self.owner_worker_id: str | None = None
        self.owner_model_call_id: str | None = None


class RestoredCutRegistry:
    """Atomically lease restored cuts to one policy-server worker."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int], _RestoredCutEntry] = {}

    def install(
        self,
        entries: dict[tuple[str, int], dict[str, Any]],
    ) -> None:
        """Replace the registry with one restored checkpoint's cut inventory."""
        replacement: dict[tuple[str, int], _RestoredCutEntry] = {}
        for key, value in entries.items():
            existing = replacement.get(key)
            if existing is not None and existing.value != value:
                raise ValueError(f"multiple restored cuts target replacement attempt {key!r}")
            replacement[key] = _RestoredCutEntry(value)
        self._entries = replacement

    def claim(
        self,
        key: tuple[str, int],
        *,
        worker_id: str,
        model_call_id: str,
    ) -> dict[str, Any] | None:
        """Lease one available cut; return ``None`` when no cut targets the request."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.consumed:
            raise RestoredCutConsumedError(f"restored cut {key!r} was already consumed by its replacement request")
        if entry.owner_worker_id is None:
            entry.owner_worker_id = worker_id
            entry.owner_model_call_id = model_call_id
            return entry.value
        if entry.owner_worker_id == worker_id and entry.owner_model_call_id == model_call_id:
            return entry.value
        raise RestoredCutAlreadyOwnedError(f"restored cut {key!r} is owned by another policy request")

    def consume(
        self,
        key: tuple[str, int],
        *,
        worker_id: str,
        model_call_id: str,
    ) -> None:
        """Retire one successfully consumed or deliberately declined cut."""
        entry = self._require_owner(key, worker_id=worker_id, model_call_id=model_call_id)
        entry.consumed = True

    def release(
        self,
        key: tuple[str, int],
        *,
        worker_id: str,
        model_call_id: str,
    ) -> None:
        """Make a failed request's restored cut available to another worker."""
        entry = self._require_owner(key, worker_id=worker_id, model_call_id=model_call_id)
        entry.owner_worker_id = None
        entry.owner_model_call_id = None

    def release_if_owned(
        self,
        key: tuple[str, int],
        *,
        worker_id: str,
        model_call_id: str,
    ) -> bool:
        """Release an abandoned claim without disturbing another owner."""
        entry = self._entries.get(key)
        if (
            entry is None
            or entry.consumed
            or entry.owner_worker_id != worker_id
            or entry.owner_model_call_id != model_call_id
        ):
            return False
        entry.owner_worker_id = None
        entry.owner_model_call_id = None
        return True

    def release_worker(self, worker_id: str) -> int:
        """Release every unfinished lease owned by a disconnected worker."""
        released = 0
        for entry in self._entries.values():
            if entry.consumed or entry.owner_worker_id != worker_id:
                continue
            entry.owner_worker_id = None
            entry.owner_model_call_id = None
            released += 1
        return released

    def status(self) -> dict[str, int]:
        consumed = sum(entry.consumed for entry in self._entries.values())
        leased = sum(not entry.consumed and entry.owner_worker_id is not None for entry in self._entries.values())
        return {
            "entries": len(self._entries),
            "available": len(self._entries) - leased - consumed,
            "leased": leased,
            "consumed": consumed,
        }

    def has_unconsumed(self) -> bool:
        """Return whether any restored cut may still be claimed or completed."""
        return any(not entry.consumed for entry in self._entries.values())

    def _require_owner(
        self,
        key: tuple[str, int],
        *,
        worker_id: str,
        model_call_id: str,
    ) -> _RestoredCutEntry:
        entry = self._entries.get(key)
        if entry is None or entry.owner_worker_id != worker_id or entry.owner_model_call_id != model_call_id:
            raise RestoredCutAlreadyOwnedError(f"restored cut {key!r} is not owned by this policy request")
        return entry


class WorkerRecord:
    __slots__ = (
        "worker_id",
        "pid",
        "acked_seq",
        "inflight",
        "generation_pending",
        "seen_attempts",
        "checkpoint_exclusions",
        "cut_proof",
        "proof_error",
        "writer",
        "write_lock",
        "lease_lock",
        "lease_tasks",
        "connected",
    )

    def __init__(self, worker_id: str, pid: int, writer: asyncio.StreamWriter) -> None:
        self.worker_id = worker_id
        self.pid = pid
        self.acked_seq = 0
        self.inflight = 0
        self.generation_pending = 0
        self.seen_attempts: set[tuple[str, int]] = set()
        self.checkpoint_exclusions: set[tuple[str, int]] = set()
        self.cut_proof: GenerationCutWorkerProof | None = None
        self.proof_error: str | None = None
        self.writer = writer
        self.write_lock = asyncio.Lock()
        self.lease_lock = asyncio.Lock()
        self.lease_tasks: set[asyncio.Task[Any]] = set()
        self.connected = True


class AdmissionCoordinator:
    """Service-level admission truth for one multi-worker server instance.

    ``expected_workers`` comes from configuration, not discovery: the
    coordinator must know how many workers should exist to tell "all workers
    drained" apart from "the missing worker never reported".
    """

    def __init__(
        self,
        socket_path: Path,
        expected_workers: int,
        *,
        restored_cuts: RestoredCutRegistry | None = None,
        service_handler: Callable[[str, str, dict[str, Any]], Awaitable[Any]] | None = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.expected_workers = expected_workers
        self.restored_cuts = restored_cuts or RestoredCutRegistry()
        self.service_handler = service_handler
        self._workers: dict[str, WorkerRecord] = {}
        self._state = AdmissionState.ACCEPTING
        self._checkpoint_id: Optional[str] = None
        self._cut_timeout_s: float | None = None
        self._frozen_worker_ids: tuple[str, ...] = ()
        self._seq = 0
        self._tombstones: list[dict[str, Any]] = []
        self._checkpoint_exclusions: set[tuple[str, int]] = set()
        self._server: Optional[asyncio.base_events.Server] = None
        self._changed = asyncio.Condition()
        self._request_tasks: set[asyncio.Task[Any]] = set()

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._server = await asyncio.start_unix_server(self._serve_worker, path=str(self.socket_path))

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for record in self._workers.values():
            if record.connected:
                record.writer.close()
        for task in self._request_tasks:
            task.cancel()
        if self._request_tasks:
            await asyncio.gather(*self._request_tasks, return_exceptions=True)
        self._request_tasks.clear()
        if self.socket_path.exists():
            self.socket_path.unlink()

    # -- worker connections --------------------------------------------------

    async def _serve_worker(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        record: Optional[WorkerRecord] = None
        try:
            async for message in _read_messages(reader):
                kind = message.get("type")
                if kind == "register":
                    worker_id = str(message["worker_id"])
                    existing = self._workers.get(worker_id)
                    if existing is not None and existing.connected:
                        await _write_message(
                            writer,
                            {
                                "type": "registration_rejected",
                                "checkpoint_id": self._checkpoint_id,
                                "reason": f"connected worker ID {worker_id!r} is already registered",
                            },
                        )
                        break
                    if self._state != AdmissionState.ACCEPTING:
                        await _write_message(
                            writer,
                            {
                                "type": "registration_rejected",
                                "checkpoint_id": self._checkpoint_id,
                                "reason": "worker registration is closed for the active checkpoint cut",
                            },
                        )
                        break
                    record = WorkerRecord(worker_id, int(message.get("pid", 0)), writer)
                    self._workers[record.worker_id] = record
                    # A late-joining worker immediately receives the current
                    # state so it can never serve traffic against a stale one.
                    await self._write_to_worker(record, self._state_message())
                    await self._notify()
                elif record is None:
                    continue
                elif kind == "ack":
                    message_seq = int(message["seq"])
                    if message_seq < self._seq:
                        continue
                    record.inflight = int(message.get("inflight", record.inflight))
                    record.generation_pending = int(message.get("generation_pending", record.inflight))
                    if "seen_attempts" in message:
                        record.seen_attempts = _identity_set(message["seen_attempts"])
                    if "checkpoint_exclusions" in message:
                        record.checkpoint_exclusions = _identity_set(message["checkpoint_exclusions"])
                    if self._state != AdmissionState.ACCEPTING:
                        if not self._accept_cut_proof(record, message, message_seq):
                            await self._notify()
                            continue
                    else:
                        record.cut_proof = None
                        record.proof_error = None
                    record.acked_seq = message_seq
                    await self._notify()
                elif kind == "counters":
                    message_seq = int(message.get("seq", -1))
                    if self._state != AdmissionState.ACCEPTING and message_seq < self._seq:
                        continue
                    record.inflight = int(message["inflight"])
                    record.generation_pending = int(message.get("generation_pending", record.inflight))
                    if self._state != AdmissionState.ACCEPTING:
                        self._accept_cut_proof(record, message, message_seq)
                    await self._notify()
                elif kind == "service_request":
                    operation = str(message.get("operation", ""))
                    lease_mutating = operation in _LEASE_MUTATING_SERVICE_OPERATIONS
                    task = asyncio.create_task(
                        self._handle_service_request(
                            record,
                            message,
                            lease_mutating=lease_mutating,
                        )
                    )
                    self._request_tasks.add(task)
                    task.add_done_callback(self._request_tasks.discard)
                    if lease_mutating:
                        record.lease_tasks.add(task)
                        task.add_done_callback(record.lease_tasks.discard)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except _CoordinatorProtocolError as error:
            _LOG.warning("closing checkpoint coordinator worker connection after protocol error: %s", error)
        finally:
            if record is not None:
                record.connected = False
                # A worker can disconnect after sending a lease mutation but
                # before receiving its response. Settle those operations in
                # receive order before reclaiming unfinished leases. This
                # preserves a durable consume while ensuring a late claim can
                # never leave a lease owned by a dead worker.
                if record.lease_tasks:
                    await asyncio.gather(*tuple(record.lease_tasks), return_exceptions=True)
                self.restored_cuts.release_worker(record.worker_id)
                await self._notify()
            writer.close()

    async def _handle_service_request(
        self,
        record: WorkerRecord,
        message: dict[str, Any],
        *,
        lease_mutating: bool,
    ) -> None:
        request_id = str(message.get("request_id", ""))
        operation = str(message.get("operation", ""))
        payload = message.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        try:
            if not request_id:
                raise ValueError("service request_id must be non-empty")
            if self.service_handler is None:
                raise CoordinatorServiceError("checkpoint coordinator has no service handler")
            if lease_mutating:
                async with record.lease_lock:
                    result = await self.service_handler(record.worker_id, operation, payload)
            else:
                result = await self.service_handler(record.worker_id, operation, payload)
        except ControlError as error:
            response = {
                "type": "service_error",
                "request_id": request_id,
                "error_code": error.code,
                "detail": error.detail,
                "status_code": error.status_code,
            }
        except ValueError as error:
            response = {
                "type": "service_error",
                "request_id": request_id,
                "error_code": "invalid_service_request",
                "detail": str(error),
                "status_code": 409,
            }
        except Exception as error:
            response = {
                "type": "service_error",
                "request_id": request_id,
                "error_code": "coordinator_service_error",
                "detail": f"{type(error).__name__}: {error}",
                "status_code": 500,
            }
        else:
            response = {
                "type": "service_result",
                "request_id": request_id,
                "result": result,
            }
        if not record.connected:
            return
        try:
            await self._write_to_worker(record, response)
        except (BrokenPipeError, ConnectionResetError):
            record.connected = False

    async def _write_to_worker(self, record: WorkerRecord, message: dict[str, Any]) -> None:
        async with record.write_lock:
            await _write_message(record.writer, message)

    def _accept_cut_proof(self, record: WorkerRecord, message: dict[str, Any], message_seq: int) -> bool:
        try:
            proof = GenerationCutWorkerProof.model_validate(message.get("generation_cut_proof"))
            if (
                message_seq != self._seq
                or proof.coordinator_sequence != self._seq
                or proof.worker_id != record.worker_id
                or proof.checkpoint_id != self._checkpoint_id
            ):
                raise ValueError("worker generation-cut proof identity does not match coordinator state")
        except (TypeError, ValueError) as error:
            record.cut_proof = None
            record.proof_error = str(error)
            return False
        record.cut_proof = proof
        record.proof_error = None
        return True

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    # -- state distribution --------------------------------------------------

    def _state_message(self) -> dict[str, Any]:
        return {
            "type": "state",
            "seq": self._seq,
            "state": self._state.value,
            "checkpoint_id": self._checkpoint_id,
            "cut_timeout_s": self._cut_timeout_s,
            "frozen_worker_ids": self._frozen_worker_ids,
            "tombstones": self._tombstones,
            "restored_cuts_available": self.restored_cuts.has_unconsumed(),
        }

    async def _broadcast(self) -> None:
        self._seq += 1
        message = self._state_message()
        for record in self._workers.values():
            if record.connected:
                try:
                    await self._write_to_worker(record, message)
                except (BrokenPipeError, ConnectionResetError):
                    record.connected = False

    async def close_admission(
        self,
        checkpoint_id: str,
        *,
        cut_timeout_s: float | None = None,
    ) -> None:
        connected = tuple(sorted(record.worker_id for record in self._workers.values() if record.connected))
        if len(connected) != self.expected_workers:
            raise MissingWorkersError(
                f"cannot freeze checkpoint worker membership: expected {self.expected_workers}, "
                f"found {len(connected)} connected workers"
            )
        self._state = AdmissionState.DRAINING
        self._checkpoint_id = checkpoint_id
        self._cut_timeout_s = cut_timeout_s
        self._frozen_worker_ids = connected
        self._checkpoint_exclusions.clear()
        for record in self._workers.values():
            record.cut_proof = None
            record.proof_error = None
        await self._broadcast()

    async def resume_admission(self) -> None:
        self._state = AdmissionState.ACCEPTING
        self._checkpoint_id = None
        self._cut_timeout_s = None
        self._frozen_worker_ids = ()
        self._checkpoint_exclusions.clear()
        for record in self._workers.values():
            record.cut_proof = None
            record.proof_error = None
        await self._broadcast()

    async def add_tombstone(self, rollout_id: str, attempt_index: int) -> None:
        self._tombstones.append({"rollout_id": rollout_id, "attempt_index": attempt_index})
        self._checkpoint_exclusions.add((rollout_id, attempt_index))
        await self._broadcast()

    async def install_tombstones(self, identities: Iterable[tuple[str, int]]) -> None:
        """Install restored attempt fences and publish them in one state update."""
        known = {(item["rollout_id"], int(item["attempt_index"])) for item in self._tombstones}
        for rollout_id, attempt_index in identities:
            if (rollout_id, attempt_index) in known:
                continue
            self._tombstones.append({"rollout_id": rollout_id, "attempt_index": attempt_index})
            known.add((rollout_id, attempt_index))
        await self._broadcast()

    async def publish_restored_cut_state(self) -> None:
        """Publish a restored-cut availability transition to every worker."""
        if self._state == AdmissionState.ACCEPTING:
            await self._broadcast()

    def seen_attempts(self) -> set[tuple[str, int]]:
        result: set[tuple[str, int]] = set()
        for record in self._workers.values():
            if record.connected:
                result.update(record.seen_attempts)
        return result

    def checkpoint_exclusions(self) -> set[tuple[str, int]]:
        result = set(self._checkpoint_exclusions)
        for record in self._workers.values():
            if record.connected:
                result.update(record.checkpoint_exclusions)
        return result

    # -- aggregation ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        live = [record for record in self._workers.values() if record.connected]
        missing = self.expected_workers - len(live)
        acknowledged = sum(1 for record in live if record.acked_seq >= self._seq)
        inflight_total = sum(record.inflight for record in live)
        generation_pending_total = sum(record.generation_pending for record in live)
        all_acked = missing == 0 and acknowledged == len(live)
        all_proofs_complete = all(
            record.cut_proof is not None
            and record.cut_proof.coordinator_sequence == self._seq
            and record.cut_proof.generation_pending == 0
            for record in live
        )
        drained = all_acked and generation_pending_total == 0 and all_proofs_complete
        if self._state == AdmissionState.DRAINING and drained:
            state = AdmissionState.PAUSED.value
        else:
            state = self._state.value
        return {
            "state": state,
            "workers": {"acknowledged": acknowledged, "expected": self.expected_workers, "live": len(live)},
            # A missing worker is an error, never an implicit zero: it may
            # hold in-flight requests the coordinator cannot see.
            "missing_workers": missing,
            "inflight_total": inflight_total,
            "response_inflight_total": inflight_total,
            "generation_pending_total": generation_pending_total,
            "waiters_total": 0,
            "per_worker": {
                record.worker_id: {
                    "acked_seq": record.acked_seq,
                    "inflight": record.inflight,
                    "generation_pending": record.generation_pending,
                    "generation_cut_proof": (
                        record.cut_proof.model_dump(mode="json") if record.cut_proof is not None else None
                    ),
                    "proof_error": record.proof_error,
                    "connected": record.connected,
                }
                for record in self._workers.values()
            },
        }

    def generation_cut_proof(self) -> GenerationCutCoordinatorProof:
        """Return the complete proof for the current frozen worker sequence."""
        if self._checkpoint_id is None:
            raise ValueError("coordinator has no active checkpoint")
        live = [record for record in self._workers.values() if record.connected]
        if len(live) != self.expected_workers or any(
            record.acked_seq < self._seq or record.cut_proof is None for record in live
        ):
            raise ValueError("coordinator generation-cut proof omits frozen worker membership")
        return GenerationCutCoordinatorProof.build(
            checkpoint_id=self._checkpoint_id,
            coordinator_sequence=self._seq,
            expected_workers=self.expected_workers,
            frozen_worker_ids=self._frozen_worker_ids,
            workers=[record.cut_proof for record in live if record.cut_proof is not None],
        )

    async def wait_until(self, predicate: Callable[[dict[str, Any]], bool], timeout_s: float) -> dict[str, Any]:
        """Wait for the aggregated status to satisfy ``predicate``; return the last status."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        async with self._changed:
            while True:
                status = self.status()
                if predicate(status):
                    return status
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return status
                try:
                    await asyncio.wait_for(self._changed.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return self.status()


class AdmissionCoordinatorRunner:
    """Run one coordinator event loop beside a multi-worker Uvicorn parent."""

    def __init__(self, coordinator: AdmissionCoordinator) -> None:
        self.coordinator = coordinator
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._error: BaseException | None = None

    def start(self, *, timeout_s: float = 10.0) -> None:
        if self._thread is not None:
            raise RuntimeError("checkpoint coordinator runner is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="nemo-gym-checkpoint-coordinator",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise RuntimeError("checkpoint coordinator did not start before its deadline")
        if self._error is not None:
            raise RuntimeError("checkpoint coordinator failed to start") from self._error

    def stop(self, *, timeout_s: float = 10.0) -> None:
        thread = self._thread
        loop = self._loop
        stop_event = self._stop_event
        if thread is None:
            return
        if loop is not None and stop_event is not None:
            loop.call_soon_threadsafe(stop_event.set)
        thread.join(timeout_s)
        if thread.is_alive():
            raise RuntimeError("checkpoint coordinator did not stop before its deadline")
        self._thread = None
        if self._error is not None:
            raise RuntimeError("checkpoint coordinator failed") from self._error

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        async def serve() -> None:
            try:
                await self.coordinator.start()
                self._stop_event = asyncio.Event()
            except BaseException as error:
                self._error = error
                self._ready.set()
                return
            self._ready.set()
            try:
                await self._stop_event.wait()
            finally:
                await self.coordinator.stop()

        try:
            loop.run_until_complete(serve())
        except BaseException as error:
            self._error = error
            self._ready.set()
        finally:
            loop.close()
            self._loop = None
            self._stop_event = None


class WorkerAdmissionAgent:
    """The per-worker side of the coordination protocol.

    Runs inside each uvicorn worker process next to that worker's
    ``AdmissionLimiter``. Applies coordinator state pushes to the limiter,
    acknowledges each one with the sequence number it installed, and reports
    the local in-flight count whenever it changes.
    """

    def __init__(
        self,
        socket_path: Path,
        worker_id: str,
        limiter: AdmissionLimiter,
        *,
        pid: int = 0,
        server_name: str = "policy",
        cut_timeout_s: float | None = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.worker_id = worker_id
        self.limiter = limiter
        self.pid = pid
        self.server_name = server_name
        self.cut_timeout_s = cut_timeout_s
        self._writer: Optional[asyncio.StreamWriter] = None
        self._listener: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()
        self._service_requests: dict[str, asyncio.Future[Any]] = {}
        self._next_service_request_id = 0
        self._coordinator_sequence = 0
        self._checkpoint_id: str | None = None
        self._restored_cuts_available = False
        self._reported_seen_attempts_checkpoint_id: str | None = None
        self._counter_report_pending = False
        self._counter_report_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        reader, writer = await asyncio.open_unix_connection(path=str(self.socket_path))
        self._writer = writer
        self.limiter.add_listener(self._on_limiter_change)
        await self._write({"type": "register", "worker_id": self.worker_id, "pid": self.pid})
        try:
            first_message = await _read_message(reader)
        except _CoordinatorProtocolError as error:
            self.limiter.remove_listener(self._on_limiter_change)
            self._writer = None
            writer.close()
            raise WorkerRegistrationError(f"invalid checkpoint coordinator registration response: {error}") from error
        if first_message is None:
            self.limiter.remove_listener(self._on_limiter_change)
            self._writer = None
            writer.close()
            raise WorkerRegistrationError("coordinator closed the worker registration connection")
        if first_message.get("type") == "registration_rejected":
            self._checkpoint_id = first_message.get("checkpoint_id")
            self.limiter.close(self._checkpoint_id)
            self.limiter.remove_listener(self._on_limiter_change)
            self._writer = None
            writer.close()
            raise WorkerRegistrationError(str(first_message.get("reason", "worker registration rejected")))
        await self._apply_state_message(first_message)
        self._listener = asyncio.create_task(self._listen(reader))

    async def stop(self) -> None:
        self.limiter.remove_listener(self._on_limiter_change)
        self._counter_report_pending = False
        if self._counter_report_task is not None:
            self._counter_report_task.cancel()
            try:
                await self._counter_report_task
            except asyncio.CancelledError:
                pass
            self._counter_report_task = None
        if self._listener is not None:
            self._listener.cancel()
            try:
                await self._listener
            except asyncio.CancelledError:
                pass
            self._listener = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        for future in self._service_requests.values():
            if not future.done():
                future.set_exception(ConnectionError("checkpoint coordinator connection closed"))
        self._service_requests.clear()

    async def _listen(self, reader: asyncio.StreamReader) -> None:
        try:
            async for message in _read_messages(reader):
                if message.get("type") in {"service_result", "service_error"}:
                    self._complete_service_request(message)
                else:
                    await self._apply_state_message(message)
        except _CoordinatorProtocolError as error:
            _LOG.warning("checkpoint coordinator worker received an invalid frame: %s", error)
            if self._writer is not None:
                self._writer.close()
        finally:
            for future in tuple(self._service_requests.values()):
                if not future.done():
                    future.set_exception(ConnectionError("checkpoint coordinator connection closed"))

    async def _apply_state_message(self, message: dict[str, Any]) -> None:
        if message.get("type") != "state":
            return
        state = AdmissionState(message["state"])
        self._coordinator_sequence = int(message["seq"])
        self._checkpoint_id = message.get("checkpoint_id")
        self._restored_cuts_available = bool(message.get("restored_cuts_available", False))
        if state == AdmissionState.ACCEPTING:
            self.limiter.resume()
            self._reported_seen_attempts_checkpoint_id = None
        else:
            checkpoint_id = message.get("checkpoint_id")
            self.limiter.close(checkpoint_id)
        for tombstone in message.get("tombstones", ()):
            self.limiter.abort_inflight(tombstone["rollout_id"], tombstone["attempt_index"])
        if state != AdmissionState.ACCEPTING and checkpoint_id is not None:
            requested_cut_timeout_s = message.get("cut_timeout_s")
            cut_timeout_s = 10.0 if requested_cut_timeout_s is None else max(float(requested_cut_timeout_s), 0.0)
            if self.cut_timeout_s is not None:
                cut_timeout_s = min(cut_timeout_s, self.cut_timeout_s)
            await self.limiter.prepare_generation_cut(
                checkpoint_id,
                server_name=self.server_name,
                timeout_s=cut_timeout_s,
            )
        assert self._writer is not None
        payload = {
            "type": "ack",
            "seq": message["seq"],
            "inflight": self.limiter.counts()["inflight_total"],
            "generation_pending": self.limiter.counts()["generation_pending_total"],
            **self._generation_cut_proof_payload(),
        }
        report_seen_attempts = (
            state != AdmissionState.ACCEPTING
            and checkpoint_id is not None
            and self._reported_seen_attempts_checkpoint_id != checkpoint_id
        )
        if report_seen_attempts:
            payload.update(self._lineage_identity_payload())
        await self._write(payload)
        if report_seen_attempts:
            self._reported_seen_attempts_checkpoint_id = checkpoint_id

    def _on_limiter_change(self) -> None:
        writer = self._writer
        if writer is None or writer.is_closing():
            return
        self._counter_report_pending = True
        if self._counter_report_task is None:
            self._counter_report_task = asyncio.get_running_loop().create_task(self._flush_counter_reports())

    async def _flush_counter_reports(self) -> None:
        """Publish the latest counters with at most one report task per worker."""
        try:
            while self._counter_report_pending:
                self._counter_report_pending = False
                counts = self.limiter.counts()
                payload = {
                    "type": "counters",
                    "seq": self._coordinator_sequence,
                    "inflight": counts["inflight_total"],
                    "generation_pending": counts["generation_pending_total"],
                    **self._generation_cut_proof_payload(),
                }
                await self._write(payload)
        except (BrokenPipeError, ConnectionError):
            # The listener owns connection-loss handling. Counter changes are
            # snapshots, so reconnect/state acknowledgement supersedes them.
            self._counter_report_pending = False
        finally:
            self._counter_report_task = None
            writer = self._writer
            if self._counter_report_pending and writer is not None and not writer.is_closing():
                self._counter_report_task = asyncio.get_running_loop().create_task(self._flush_counter_reports())

    def _generation_cut_proof_payload(self) -> dict[str, Any]:
        if self._checkpoint_id is None or self._coordinator_sequence <= 0:
            return {}
        proof = self.limiter.generation_cut_worker_proof(
            self._checkpoint_id,
            coordinator_sequence=self._coordinator_sequence,
            worker_id=self.worker_id,
        )
        return {"generation_cut_proof": proof.model_dump(mode="json")}

    def _lineage_identity_payload(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "seen_attempts": [
                {"rollout_id": rollout_id, "attempt_index": attempt_index}
                for rollout_id, attempt_index in sorted(self.limiter.seen_attempts())
            ],
            "checkpoint_exclusions": [
                {"rollout_id": rollout_id, "attempt_index": attempt_index}
                for rollout_id, attempt_index in sorted(self.limiter.checkpoint_exclusions())
            ],
        }

    def service_client(self) -> "CoordinatorServiceClient":
        """Return the async client for coordinator-owned checkpoint operations."""
        return CoordinatorServiceClient(self)

    def has_restored_cuts(self) -> bool:
        """Return whether the coordinator advertises an unconsumed restored cut."""
        return self._restored_cuts_available

    async def _service_request(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
        wait_without_timeout: bool = False,
    ) -> Any:
        if self._writer is None or self._writer.is_closing():
            raise ConnectionError("checkpoint coordinator connection is not active")
        self._next_service_request_id += 1
        request_id = f"{self.worker_id}:service:{self._next_service_request_id}"
        future = asyncio.get_running_loop().create_future()
        self._service_requests[request_id] = future
        try:
            try:
                await self._write(
                    {
                        "type": "service_request",
                        "request_id": request_id,
                        "operation": operation,
                        "payload": payload,
                    }
                )
                if wait_without_timeout:
                    return await future
                return await asyncio.wait_for(
                    future,
                    timeout=timeout_s if timeout_s is not None else (self.cut_timeout_s or 10.0),
                )
            except (asyncio.CancelledError, TimeoutError):
                if operation == "claim_generation_cut":
                    try:
                        await self._service_request(
                            "abandon_generation_cut_claim",
                            payload,
                            wait_without_timeout=True,
                        )
                    except ConnectionError:
                        # A disconnected worker has all of its unfinished
                        # leases reclaimed by the coordinator connection's
                        # teardown path.
                        pass
                raise
        finally:
            self._service_requests.pop(request_id, None)

    def _complete_service_request(self, message: dict[str, Any]) -> None:
        request_id = str(message.get("request_id", ""))
        future = self._service_requests.get(request_id)
        if future is None or future.done():
            return
        if message.get("type") == "service_error":
            future.set_exception(
                CoordinatorServiceError(
                    str(message.get("detail", "checkpoint coordinator service request failed")),
                    code=str(message.get("error_code", "coordinator_service_error")),
                    status_code=int(message.get("status_code", 409)),
                )
            )
        else:
            future.set_result(message.get("result"))

    async def _write(self, message: dict[str, Any]) -> None:
        writer = self._writer
        if writer is None:
            raise ConnectionError("checkpoint coordinator connection is not active")
        async with self._write_lock:
            await _write_message(writer, message)


class CoordinatorServiceClient:
    """Async client for one coordinator-owned checkpoint service."""

    def __init__(self, worker: WorkerAdmissionAgent) -> None:
        self._worker = worker

    @property
    def has_restored_cuts(self) -> bool:
        """Return whether a claim can possibly succeed without coordinator I/O."""
        return self._worker.has_restored_cuts()

    async def request(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        return await self._worker._service_request(operation, payload, timeout_s=timeout_s)


def build_coordinator_control_app(
    coordinator: AdmissionCoordinator,
    *,
    capabilities: ControlCapabilities,
    auth_token: str,
    fence: Optional[ControlFence] = None,
    ack_timeout_s: float = 10.0,
) -> FastAPI:
    """Build the control app the coordinator process serves.

    This app owns the instance's control URL: the same
    ``/ng-control/v1/model-admission`` contract as the single-worker server,
    but every answer is aggregated over the worker pool. Pause returns only
    after every live worker acknowledged the closed admission state, and it
    fails with ``missing_workers`` when the pool is incomplete rather than
    reporting a partial pool as drained.
    """
    fence = fence or ControlFence()
    app = FastAPI()
    install_control_plane(app, capabilities=capabilities, fence=fence)

    def _all_live_acked(status: dict[str, Any]) -> bool:
        return status["missing_workers"] == 0 and status["workers"]["acknowledged"] == status["workers"]["live"]

    async def _await_worker_acks(deadline_s: float) -> dict[str, Any]:
        status = await coordinator.wait_until(_all_live_acked, timeout_s=deadline_s)
        if not _all_live_acked(status):
            raise MissingWorkersError(
                f"{status['missing_workers']} of {coordinator.expected_workers} workers missing and "
                f"{status['workers']['acknowledged']}/{status['workers']['live']} live workers acknowledged; "
                f"a missing worker may hold in-flight requests and cannot be counted as drained"
            )
        return status

    @app.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause")
    async def coordinator_pause(
        body: ModelAdmissionPauseRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        deadline = Deadline(deadline_ts=body.deadline_ts)

        async def run() -> dict[str, Any]:
            await coordinator.close_admission(
                body.checkpoint_id,
                cut_timeout_s=deadline.remaining(),
            )
            try:
                status = await _await_worker_acks(min(ack_timeout_s, max(deadline.remaining(), 0.001)))
            except BaseException:
                await coordinator.resume_admission()
                raise
            return {
                "state": status["state"],
                "workers": {
                    "acknowledged": status["workers"]["acknowledged"],
                    "expected": coordinator.expected_workers,
                },
                "inflight_total": status["inflight_total"],
                "response_inflight_total": status["response_inflight_total"],
                "generation_pending_total": status["generation_pending_total"],
                "generation_cut_proof": (
                    coordinator.generation_cut_proof().model_dump(mode="json")
                    if status["state"] == AdmissionState.PAUSED.value
                    else None
                ),
                "waiters_total": status["waiters_total"],
            }

        result = await fence.run_operation(
            body.checkpoint_id,
            "model-admission/pause",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.PREPARING,
            phase_after=CheckpointPhase.PREPARING,
            run=run,
            deadline=deadline,
        )
        if result["state"] == AdmissionState.PAUSED.value:
            fence.mark_prepared(body.checkpoint_id)
        return result

    @app.get(f"{MODEL_ADMISSION_URL_PREFIX}/status")
    async def coordinator_status(
        checkpoint_id: str = Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
        wait_state: Optional[str] = None,
        timeout_s: float = 0.0,
        authorization: Optional[str] = Header(default=None),
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
                    CheckpointPhase.RESTORE_FAILED_PAUSED,
                    CheckpointPhase.RESTORED_PAUSED,
                }
            ),
        )
        if wait_state == "paused" and timeout_s > 0:
            status = await coordinator.wait_until(lambda s: s["state"] == "paused", timeout_s=timeout_s)
        else:
            status = coordinator.status()
        if status["state"] == AdmissionState.PAUSED.value and fence.phase == CheckpointPhase.PREPARING:
            fence.mark_prepared(checkpoint_id)
        if status["state"] == AdmissionState.PAUSED.value:
            status["generation_cut_proof"] = coordinator.generation_cut_proof().model_dump(mode="json")
        return status

    @app.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume")
    async def coordinator_resume(
        body: ModelAdmissionResumeRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            await coordinator.resume_admission()
            status = await _await_worker_acks(ack_timeout_s)
            return {
                "state": status["state"],
                "workers": {
                    "acknowledged": status["workers"]["acknowledged"],
                    "expected": coordinator.expected_workers,
                },
                "released_waiters": 0,
            }

        return await fence.run_operation(
            body.checkpoint_id,
            "model-admission/resume",
            allowed_phases=frozenset(
                {
                    CheckpointPhase.PREPARING,
                    CheckpointPhase.PREPARED,
                    CheckpointPhase.COMMITTED_PAUSED,
                    CheckpointPhase.RESTORE_FAILED_PAUSED,
                    CheckpointPhase.RESTORED_PAUSED,
                }
            ),
            phase_during=fence.phase,
            phase_after=CheckpointPhase.IDLE,
            run=run,
            retire_outcome="resumed",
        )

    @app.post(f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight")
    async def coordinator_abort_inflight(
        body: ModelAbortInflightRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            await coordinator.add_tombstone(body.rollout_id, body.attempt_index)
            status = await _await_worker_acks(ack_timeout_s)
            return {"state": status["state"], "inflight_total": status["inflight_total"]}

        return await fence.run_operation(
            body.checkpoint_id,
            f"model-admission/abort_inflight:{body.rollout_id}:{body.attempt_index}",
            allowed_phases=frozenset({CheckpointPhase.PREPARING, CheckpointPhase.PREPARED}),
            phase_during=fence.phase,
            phase_after=fence.phase,
            run=run,
        )

    return app


async def _write_message(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode()
    payload_size = len(payload)
    if payload_size == 0 or payload_size > _MAX_COORDINATOR_PAYLOAD_BYTES:
        raise _CoordinatorProtocolError(
            "checkpoint coordinator payload size "
            f"{payload_size} is outside the allowed range 1..{_MAX_COORDINATOR_PAYLOAD_BYTES}"
        )
    writer.write(_FRAME_HEADER.pack(payload_size) + payload)
    await writer.drain()


async def _read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    try:
        header = await reader.readexactly(_FRAME_HEADER.size)
    except asyncio.IncompleteReadError as error:
        if not error.partial:
            return None
        raise _CoordinatorProtocolError(
            f"truncated checkpoint coordinator frame header: received {len(error.partial)} of {_FRAME_HEADER.size} bytes"
        ) from error

    (payload_size,) = _FRAME_HEADER.unpack(header)
    if payload_size == 0 or payload_size > _MAX_COORDINATOR_PAYLOAD_BYTES:
        raise _CoordinatorProtocolError(
            "checkpoint coordinator payload size "
            f"{payload_size} is outside the allowed range 1..{_MAX_COORDINATOR_PAYLOAD_BYTES}"
        )
    try:
        payload = await reader.readexactly(payload_size)
    except asyncio.IncompleteReadError as error:
        raise _CoordinatorProtocolError(
            f"truncated checkpoint coordinator payload: received {len(error.partial)} of {payload_size} bytes"
        ) from error
    try:
        message = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _CoordinatorProtocolError("checkpoint coordinator payload is not valid UTF-8 JSON") from error
    if not isinstance(message, dict):
        raise _CoordinatorProtocolError("checkpoint coordinator payload must be a JSON object")
    return message


async def _read_messages(reader: asyncio.StreamReader) -> AsyncIterator[dict[str, Any]]:
    while True:
        message = await _read_message(reader)
        if message is None:
            return
        yield message
