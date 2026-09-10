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

The message protocol is newline-delimited JSON, chosen for debuggability:
``register``, ``ack``, ``counters`` upstream; ``state`` downstream. The
transport is a Unix-domain socket because the coordinator and its workers
are one service on one host; nothing here crosses machines.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Optional

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


class WorkerRecord:
    __slots__ = (
        "worker_id",
        "pid",
        "acked_seq",
        "inflight",
        "generation_pending",
        "cut_proof",
        "proof_error",
        "writer",
        "connected",
    )

    def __init__(self, worker_id: str, pid: int, writer: asyncio.StreamWriter) -> None:
        self.worker_id = worker_id
        self.pid = pid
        self.acked_seq = 0
        self.inflight = 0
        self.generation_pending = 0
        self.cut_proof: GenerationCutWorkerProof | None = None
        self.proof_error: str | None = None
        self.writer = writer
        self.connected = True


class AdmissionCoordinator:
    """Service-level admission truth for one multi-worker server instance.

    ``expected_workers`` comes from configuration, not discovery: the
    coordinator must know how many workers should exist to tell "all workers
    drained" apart from "the missing worker never reported".
    """

    def __init__(self, socket_path: Path, expected_workers: int) -> None:
        self.socket_path = Path(socket_path)
        self.expected_workers = expected_workers
        self._workers: dict[str, WorkerRecord] = {}
        self._state = AdmissionState.ACCEPTING
        self._checkpoint_id: Optional[str] = None
        self._frozen_worker_ids: tuple[str, ...] = ()
        self._seq = 0
        self._tombstones: list[dict[str, Any]] = []
        self._server: Optional[asyncio.base_events.Server] = None
        self._changed = asyncio.Condition()

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
                    await _write_message(writer, self._state_message())
                    await self._notify()
                elif record is None:
                    continue
                elif kind == "ack":
                    message_seq = int(message["seq"])
                    if message_seq < self._seq:
                        continue
                    record.inflight = int(message.get("inflight", record.inflight))
                    record.generation_pending = int(message.get("generation_pending", record.inflight))
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
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            if record is not None:
                record.connected = False
                await self._notify()
            writer.close()

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
            "frozen_worker_ids": self._frozen_worker_ids,
            "tombstones": self._tombstones,
        }

    async def _broadcast(self) -> None:
        self._seq += 1
        message = self._state_message()
        for record in self._workers.values():
            if record.connected:
                try:
                    await _write_message(record.writer, message)
                except ConnectionResetError:
                    record.connected = False

    async def close_admission(self, checkpoint_id: str) -> None:
        connected = tuple(sorted(record.worker_id for record in self._workers.values() if record.connected))
        if len(connected) != self.expected_workers:
            raise MissingWorkersError(
                f"cannot freeze checkpoint worker membership: expected {self.expected_workers}, "
                f"found {len(connected)} connected workers"
            )
        self._state = AdmissionState.DRAINING
        self._checkpoint_id = checkpoint_id
        self._frozen_worker_ids = connected
        for record in self._workers.values():
            record.cut_proof = None
            record.proof_error = None
        await self._broadcast()

    async def resume_admission(self) -> None:
        self._state = AdmissionState.ACCEPTING
        self._checkpoint_id = None
        self._frozen_worker_ids = ()
        for record in self._workers.values():
            record.cut_proof = None
            record.proof_error = None
        await self._broadcast()

    async def add_tombstone(self, rollout_id: str, attempt_index: int) -> None:
        self._tombstones.append({"rollout_id": rollout_id, "attempt_index": attempt_index})
        await self._broadcast()

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
        cut_timeout_s: float = 10.0,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.worker_id = worker_id
        self.limiter = limiter
        self.pid = pid
        self.server_name = server_name
        self.cut_timeout_s = cut_timeout_s
        self._writer: Optional[asyncio.StreamWriter] = None
        self._listener: Optional[asyncio.Task] = None
        self._coordinator_sequence = 0
        self._checkpoint_id: str | None = None

    async def start(self) -> None:
        reader, writer = await asyncio.open_unix_connection(path=str(self.socket_path))
        self._writer = writer
        self.limiter.add_listener(self._on_limiter_change)
        await _write_message(writer, {"type": "register", "worker_id": self.worker_id, "pid": self.pid})
        line = await reader.readline()
        if not line:
            self.limiter.remove_listener(self._on_limiter_change)
            self._writer = None
            writer.close()
            raise WorkerRegistrationError("coordinator closed the worker registration connection")
        first_message = json.loads(line)
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

    async def _listen(self, reader: asyncio.StreamReader) -> None:
        async for message in _read_messages(reader):
            await self._apply_state_message(message)

    async def _apply_state_message(self, message: dict[str, Any]) -> None:
        if message.get("type") != "state":
            return
        state = AdmissionState(message["state"])
        self._coordinator_sequence = int(message["seq"])
        self._checkpoint_id = message.get("checkpoint_id")
        if state == AdmissionState.ACCEPTING:
            self.limiter.resume()
        else:
            checkpoint_id = message.get("checkpoint_id")
            self.limiter.close(checkpoint_id)
        for tombstone in message.get("tombstones", ()):
            self.limiter.abort_inflight(tombstone["rollout_id"], tombstone["attempt_index"])
        if state != AdmissionState.ACCEPTING and checkpoint_id is not None:
            await self.limiter.prepare_generation_cut(
                checkpoint_id,
                server_name=self.server_name,
                timeout_s=self.cut_timeout_s,
            )
        assert self._writer is not None
        await _write_message(
            self._writer,
            {
                "type": "ack",
                "seq": message["seq"],
                "inflight": self.limiter.counts()["inflight_total"],
                "generation_pending": self.limiter.counts()["generation_pending_total"],
                **self._generation_cut_proof_payload(),
            },
        )

    def _on_limiter_change(self) -> None:
        writer = self._writer
        if writer is None or writer.is_closing():
            return
        counts = self.limiter.counts()
        payload = {
            "type": "counters",
            "seq": self._coordinator_sequence,
            "inflight": counts["inflight_total"],
            "generation_pending": counts["generation_pending_total"],
            **self._generation_cut_proof_payload(),
        }
        # Fire-and-forget: counter reports are monotone-refreshed, so a lost
        # one is corrected by the next change or the next ack.
        asyncio.get_running_loop().create_task(_write_message(writer, payload))

    def _generation_cut_proof_payload(self) -> dict[str, Any]:
        if self._checkpoint_id is None or self._coordinator_sequence <= 0:
            return {}
        proof = self.limiter.generation_cut_worker_proof(
            self._checkpoint_id,
            coordinator_sequence=self._coordinator_sequence,
            worker_id=self.worker_id,
        )
        return {"generation_cut_proof": proof.model_dump(mode="json")}


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
            await coordinator.close_admission(body.checkpoint_id)
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
    writer.write(json.dumps(message).encode() + b"\n")
    await writer.drain()


async def _read_messages(reader: asyncio.StreamReader):
    while True:
        line = await reader.readline()
        if not line:
            return
        line = line.strip()
        if line:
            yield json.loads(line)
