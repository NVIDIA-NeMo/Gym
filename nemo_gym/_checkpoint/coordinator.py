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
- A worker that is expected but not connected (crashed, still starting) is
  an error in the status report, never an implicit zero: a missing worker
  may hold in-flight requests the coordinator cannot see.

The message protocol is newline-delimited JSON, chosen for debuggability:
``register``, ``ack``, ``counters`` upstream; ``state`` downstream. The
transport is a Unix-domain socket because the coordinator and its workers
are one service on one host; nothing here crosses machines.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI

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
    ModelAbortInflightRequest,
    ModelAdmissionPauseRequest,
    ModelAdmissionResumeRequest,
)


class MissingWorkersError(ControlError):
    """Expected workers are not connected to the coordinator.

    A missing worker may hold in-flight requests the coordinator cannot see,
    so it is an error to proceed — never an implicit zero.
    """

    code = "missing_workers"


class WorkerRecord:
    __slots__ = ("worker_id", "pid", "acked_seq", "inflight", "writer", "connected")

    def __init__(self, worker_id: str, pid: int, writer: asyncio.StreamWriter) -> None:
        self.worker_id = worker_id
        self.pid = pid
        self.acked_seq = 0
        self.inflight = 0
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
                    record = WorkerRecord(str(message["worker_id"]), int(message.get("pid", 0)), writer)
                    self._workers[record.worker_id] = record
                    # A late-joining worker immediately receives the current
                    # state so it can never serve traffic against a stale one.
                    await _write_message(writer, self._state_message())
                    await self._notify()
                elif record is None:
                    continue
                elif kind == "ack":
                    record.acked_seq = int(message["seq"])
                    record.inflight = int(message.get("inflight", record.inflight))
                    await self._notify()
                elif kind == "counters":
                    record.inflight = int(message["inflight"])
                    await self._notify()
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            if record is not None:
                record.connected = False
                await self._notify()
            writer.close()

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
        self._state = AdmissionState.DRAINING
        self._checkpoint_id = checkpoint_id
        await self._broadcast()

    async def resume_admission(self) -> None:
        self._state = AdmissionState.ACCEPTING
        self._checkpoint_id = None
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
        all_acked = missing == 0 and acknowledged == len(live)
        drained = all_acked and inflight_total == 0
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
            "waiters_total": 0,
            "per_worker": {
                record.worker_id: {
                    "acked_seq": record.acked_seq,
                    "inflight": record.inflight,
                    "connected": record.connected,
                }
                for record in self._workers.values()
            },
        }

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

    def __init__(self, socket_path: Path, worker_id: str, limiter: AdmissionLimiter, *, pid: int = 0) -> None:
        self.socket_path = Path(socket_path)
        self.worker_id = worker_id
        self.limiter = limiter
        self.pid = pid
        self._writer: Optional[asyncio.StreamWriter] = None
        self._listener: Optional[asyncio.Task] = None

    async def start(self) -> None:
        reader, writer = await asyncio.open_unix_connection(path=str(self.socket_path))
        self._writer = writer
        await _write_message(writer, {"type": "register", "worker_id": self.worker_id, "pid": self.pid})
        self.limiter.add_listener(self._on_limiter_change)
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
            if message.get("type") != "state":
                continue
            state = AdmissionState(message["state"])
            if state == AdmissionState.ACCEPTING:
                self.limiter.resume()
            else:
                self.limiter.close()
            for tombstone in message.get("tombstones", ()):
                self.limiter.abort_inflight(tombstone["rollout_id"], tombstone["attempt_index"])
            assert self._writer is not None
            await _write_message(
                self._writer,
                {
                    "type": "ack",
                    "seq": message["seq"],
                    "inflight": self.limiter.counts()["inflight_total"],
                },
            )

    def _on_limiter_change(self) -> None:
        writer = self._writer
        if writer is None or writer.is_closing():
            return
        payload = {"type": "counters", "inflight": self.limiter.counts()["inflight_total"]}
        # Fire-and-forget: counter reports are monotone-refreshed, so a lost
        # one is corrected by the next change or the next ack.
        asyncio.get_running_loop().create_task(_write_message(writer, payload))


def build_coordinator_control_app(
    coordinator: AdmissionCoordinator,
    *,
    capabilities: ControlCapabilities,
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
    async def coordinator_pause(body: ModelAdmissionPauseRequest) -> dict[str, Any]:
        deadline = Deadline(deadline_ts=body.deadline_ts)

        async def run() -> dict[str, Any]:
            await coordinator.close_admission(body.checkpoint_id)
            status = await _await_worker_acks(min(ack_timeout_s, max(deadline.remaining(), 0.001)))
            return {
                "state": status["state"],
                "workers": {
                    "acknowledged": status["workers"]["acknowledged"],
                    "expected": coordinator.expected_workers,
                },
                "inflight_total": status["inflight_total"],
                "waiters_total": status["waiters_total"],
            }

        return await fence.run_operation(
            body.checkpoint_id,
            "model-admission/pause",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.PREPARING,
            phase_after=CheckpointPhase.PREPARED,
            run=run,
            deadline=deadline,
        )

    @app.get(f"{MODEL_ADMISSION_URL_PREFIX}/status")
    async def coordinator_status(wait_state: Optional[str] = None, timeout_s: float = 0.0) -> dict[str, Any]:
        if wait_state == "paused" and timeout_s > 0:
            status = await coordinator.wait_until(lambda s: s["state"] == "paused", timeout_s=timeout_s)
        else:
            status = coordinator.status()
        return status

    @app.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume")
    async def coordinator_resume(body: ModelAdmissionResumeRequest) -> dict[str, Any]:
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
                {CheckpointPhase.PREPARED, CheckpointPhase.COMMITTED_PAUSED, CheckpointPhase.RESTORED_PAUSED}
            ),
            phase_during=CheckpointPhase.PREPARED,
            phase_after=CheckpointPhase.IDLE,
            run=run,
            retire_outcome="resumed",
        )

    @app.post(f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight")
    async def coordinator_abort_inflight(body: ModelAbortInflightRequest) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            await coordinator.add_tombstone(body.rollout_id, body.attempt_index)
            status = await _await_worker_acks(ack_timeout_s)
            return {"state": status["state"], "inflight_total": status["inflight_total"]}

        return await fence.run_operation(
            body.checkpoint_id,
            f"model-admission/abort_inflight:{body.rollout_id}:{body.attempt_index}",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.PREPARED,
            phase_after=CheckpointPhase.PREPARED,
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
