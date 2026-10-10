# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A policy model server with several uvicorn workers is still one checkpoint participant.

uvicorn's workers share one port, so a control call reaches an arbitrary worker.
One coordinator in the main process, which serves no requests, owns the participant:
the control plane (phases, checkpoint-ID fencing, lease, storage), the restored generation cuts,
and the ledger export and import through the process-shared capture ledger.
Each worker runs a ``PolicyGate`` for its own calls and:

- forwards every checkpoint control call to the coordinator;
- closes, reopens, and retires when the coordinator tells it to,
  and reports what it holds while a checkpoint is open (which responses are streaming,
  which calls are undelivered, and their cuts);
- claims a restored cut from the coordinator before running a re-issued call that may continue one.

Workers talk to the coordinator over a Unix socket: length-prefixed JSON frames,
with request and reply messages in both directions on one connection per worker.

Failures close the checkpoint rather than weaken it.
A worker that has not reported for the current checkpoint, or that disconnected while it was open,
blocks prepare and commit until the controller resumes.
A worker that starts, or restarts, while a checkpoint is open closes immediately.
"""

import asyncio
import json
import logging
import os
import signal
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    ParticipantControlPlane,
    PrepareReport,
    RetiredAttempts,
    dispatch_control,
    next_attempt,
)
from nemo_gym._checkpoint.errors import ControlError
from nemo_gym._checkpoint.model import (
    CheckpointableLedger,
    CutRequester,
    GateReport,
    GateSnapshot,
    GenerationCutRecord,
    ModelRecord,
    PolicyGate,
    _Ticket,
    covers,
    export_model_records,
    import_model_records,
    merge_reports,
    retained_staging_keys,
    retire_ledgers,
    staging_keys_by_episode,
    with_scope,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.runtime_dir import server_runtime_dir


LOGGER = logging.getLogger(__name__)

# Set by the main process for the workers it spawns.
COORDINATOR_SOCKET_ENV = "NEMO_GYM_POLICY_CHECKPOINT_SOCKET"

_MAX_FRAME_BYTES = 64 * 1024 * 1024
# How long past a control call's own deadline a worker waits for the coordinator's reply.
_REPLY_GRACE_SECONDS = 10.0
# Timeout for messages that do no slow work (registration, reports, claims, reopening).
_MESSAGE_TIMEOUT_SECONDS = 30.0


class CoordinatorUnavailableError(ControlError):
    status_code = 503
    code = "checkpoint_coordinator_unavailable"


class _RemoteControlError(ControlError):
    """A control error raised in the other process, re-raised here with the same status and code."""

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code


def coordinator_socket_path() -> str:
    return os.path.join(server_runtime_dir(), f"policy-{uuid.uuid4().hex[:8]}.sock")


async def _write_frame(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    # The same encoder as the checkpoint writer:
    # orjson would refuse integers beyond 64 bits and silently turn NaN and infinities,
    # such as a -inf logprob, into null.
    data = json.dumps(message, separators=(",", ":")).encode()
    if len(data) > _MAX_FRAME_BYTES:
        raise ValueError(f"checkpoint message of {len(data)} bytes exceeds the frame limit")
    writer.write(len(data).to_bytes(4, "big") + data)
    await writer.drain()


async def _read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    size = int.from_bytes(await reader.readexactly(4), "big")
    if size > _MAX_FRAME_BYTES:
        raise ValueError(f"checkpoint message of {size} bytes exceeds the frame limit")
    return json.loads(await reader.readexactly(size))


# Handles one incoming message kind and body; returns the reply body.
_Handler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class _Channel:
    """Request and reply messages in both directions over one connection."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, handler: _Handler) -> None:
        self._reader = reader
        self._writer = writer
        self._handler = handler
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    async def call(self, kind: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self._next_id += 1
        message_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._pending[message_id] = future
        try:
            # The send is bounded too: a stuck drain() must not hold a caller past its timeout.
            async with asyncio.timeout(timeout):
                await self._send({"id": message_id, "kind": kind, "body": body})
                return await future
        except (TimeoutError, ConnectionError, RuntimeError) as error:
            # Typed, so callers that fall back on a lost coordinator, such as a cut claim, do so here too.
            raise CoordinatorUnavailableError(f"checkpoint message {kind!r} failed: {error!r}") from error
        finally:
            self._pending.pop(message_id, None)

    async def run(self) -> None:
        """Read messages until the connection closes."""
        try:
            while True:
                message = await _read_frame(self._reader)
                if "reply_to" in message:
                    future = self._pending.get(message["reply_to"])
                    if future is not None and not future.done():
                        if message["ok"]:
                            future.set_result(message["body"])
                        else:
                            error = message["error"]
                            future.set_exception(_RemoteControlError(error["status"], error["code"], error["detail"]))
                else:
                    task = asyncio.create_task(self._serve(message))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
        except (asyncio.IncompleteReadError, ConnectionError, ValueError):
            pass
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(CoordinatorUnavailableError("checkpoint coordination connection closed"))
            self._writer.close()

    def close(self) -> None:
        self._writer.close()

    async def _send(self, message: dict[str, Any]) -> None:
        async with self._write_lock:
            await _write_frame(self._writer, message)

    async def _serve(self, message: dict[str, Any]) -> None:
        try:
            reply = {
                "reply_to": message["id"],
                "ok": True,
                "body": await self._handler(message["kind"], message["body"]),
            }
        except ControlError as error:
            reply = {
                "reply_to": message["id"],
                "ok": False,
                "error": {"status": error.status_code, "code": error.code, "detail": error.detail},
            }
        except Exception as error:
            # A process boundary: report the failure to the caller instead of dropping the reply.
            LOGGER.exception("checkpoint message %r failed", message.get("kind"))
            reply = {
                "reply_to": message["id"],
                "ok": False,
                "error": {"status": 500, "code": "checkpoint_error", "detail": str(error)},
            }
        try:
            try:
                await self._send(reply)
            except (TypeError, ValueError) as error:
                # A reply that cannot be framed must still answer the caller, or it would wait until its deadline.
                LOGGER.exception("checkpoint reply to %r cannot be sent", message.get("kind"))
                await self._send(
                    {
                        "reply_to": message["id"],
                        "ok": False,
                        "error": {"status": 422, "code": "invalid_checkpoint_state", "detail": str(error)},
                    }
                )
        except (ConnectionError, RuntimeError):
            pass


# -- coordinator (main process) --------------------------------------------------------------------


class _Worker:
    def __init__(self, channel: _Channel) -> None:
        self.channel = channel
        self.report: Optional[GateReport] = None
        # Reports can arrive out of order across the reply and push paths; keep the newest.
        self.report_seq = -1

    def keep(self, seq: int, report: GateReport) -> None:
        if seq > self.report_seq:
            self.report_seq = seq
            self.report = report


class CoordinatedPolicyParticipant(CheckpointParticipant):
    """The participant of a policy model server whose calls are spread over several worker processes."""

    kind = "model"
    record_model = ModelRecord

    def __init__(self, ledger: Optional[CheckpointableLedger], *, expected_workers: int) -> None:
        super().__init__()
        self.ledger = ledger
        self.expected_workers = expected_workers
        self.workers: dict[int, _Worker] = {}
        self.accepting = True
        # Increments with each close, so a report from before it is never mistaken for a current one.
        self.generation = 0
        self.request: Optional[CheckpointRequest] = None
        # A worker disconnected while admission was closed: its undelivered calls are unknown.
        self.lost_worker = False
        # A worker left since this server started: what it served, and which restored attempts it started,
        # are unknown, so restore is refused and no restored attempt counts as pending any more.
        self.worker_left = False
        self.restored_cuts: dict[str, GenerationCutRecord] = {}
        # Restored attempts, by capture key, that no worker had started as of the last check.
        self.restored_targets: set[str] = set()
        # Whether the next reopen gives the workers the targets of an install.
        # Only an install adds targets; a later reopen must not give back one a worker has started since.
        self.targets_installed = False

    # -- workers --------------------------------------------------------------------------------

    def join(self, worker_index: int, channel: _Channel) -> dict[str, Any]:
        """Register a worker and return the state it must adopt."""
        self.workers[worker_index] = _Worker(channel)
        return {
            "accepting": self.accepting,
            "generation": self.generation,
            "request": self.request.model_dump(mode="json") if self.request is not None else None,
            "restored_keys": sorted(self.restored_cuts),
            "restored_targets": sorted(self.restored_targets),
            # A worker that starts after a retire must refuse its late requests too.
            "retired": self.retired.marks(),
        }

    async def leave(self, worker_index: int) -> None:
        if self.workers.pop(worker_index, None) is not None:
            self.worker_left = True
            self.restored_targets.clear()
            if not self.accepting:
                self.lost_worker = True
        await self.notify()

    async def receive_report(self, worker_index: int, generation: int, seq: int, report: GateReport) -> None:
        worker = self.workers.get(worker_index)
        if worker is not None and generation == self.generation and not self.accepting:
            worker.keep(seq, report)
            await self.notify()

    def claim_cut(self, capture_key: str) -> Optional[GenerationCutRecord]:
        return self.restored_cuts.pop(capture_key, None)

    def return_cut(self, capture_key: str, record: GenerationCutRecord) -> None:
        try:
            self.retired.check(EpisodeId.from_capture_key(capture_key))
        except ControlError:
            return  # The attempt was retired while the call held the cut.
        self.restored_cuts.setdefault(capture_key, record)

    async def _broadcast(self, kind: str, body: dict[str, Any], *, timeout: float) -> dict[int, dict[str, Any]]:
        workers = dict(self.workers)
        results = await asyncio.gather(
            *(worker.channel.call(kind, body, timeout=timeout) for worker in workers.values()), return_exceptions=True
        )
        replies = {}
        for worker_index, result in zip(workers, results):
            if isinstance(result, BaseException):
                raise ControlError(f"policy worker {worker_index} did not complete {kind}: {result}") from result
            replies[worker_index] = result
        return replies

    # -- participant ----------------------------------------------------------------------------

    async def close_admission(self, request: CheckpointRequest) -> None:
        self.accepting = False
        self.generation += 1
        # Restore closes admission with a RestoreRequest; workers need only the checkpoint and deadline.
        request = CheckpointRequest(checkpoint_id=request.checkpoint_id, deadline_ts=request.deadline_ts)
        self.request = request
        for worker in self.workers.values():
            worker.report = None
        replies = await self._broadcast(
            "close",
            {"request": request.model_dump(mode="json"), "generation": self.generation},
            timeout=max(0.0, request.deadline_ts - time.time()) + _REPLY_GRACE_SECONDS,
        )
        for worker_index, reply in replies.items():
            if worker_index in self.workers:
                self.workers[worker_index].keep(reply["seq"], GateReport.model_validate(reply["report"]))

    async def open_admission(self) -> None:
        self.accepting = True
        self.request = None
        self.lost_worker = False
        for worker in self.workers.values():
            worker.report = None
        try:
            targets = sorted(self.restored_targets) if self.targets_installed else None
            self.targets_installed = False
            await self._broadcast(
                "open",
                {"restored_keys": sorted(self.restored_cuts), "restored_targets": targets},
                timeout=_MESSAGE_TIMEOUT_SECONDS,
            )
        except ControlError:
            # A worker that cannot be reached is gone; one that restarts adopts the open state on joining.
            LOGGER.warning("a policy worker did not acknowledge reopening", exc_info=True)

    def readiness(self) -> PrepareReport:
        reports = [worker.report for worker in self.workers.values() if worker.report is not None]
        merged = merge_reports(reports)
        blockers = list(merged.blockers)
        if self.lost_worker:
            blockers.insert(0, "policy-worker-lost")
        if len(reports) < self.expected_workers:
            blockers.insert(0, f"policy-workers-unreported:{self.expected_workers - len(reports)}")
        return PrepareReport(
            ready=not blockers,
            blockers=blockers,
            counts={**merged.counts, "workers": len(self.workers)},
        )

    async def retire(self, episode_id: EpisodeId) -> None:
        for capture_key in [key for key in self.restored_cuts if covers(episode_id, key)]:
            del self.restored_cuts[capture_key]
        self.restored_targets = {key for key in self.restored_targets if not covers(episode_id, key)}
        await self._broadcast(
            "retire", {"episode_id": episode_id.model_dump(mode="json")}, timeout=_MESSAGE_TIMEOUT_SECONDS
        )
        # After every worker cancelled the attempts' calls, so no late row recreates a ledger.
        await retire_ledgers(self.ledger, [episode_id])

    async def mark_retired(self, episode_ids: list[EpisodeId]) -> None:
        self.retired.mark(episode_ids)
        await self._broadcast(
            "mark", {"episode_ids": [e.model_dump(mode="json") for e in episode_ids]}, timeout=_MESSAGE_TIMEOUT_SECONDS
        )

    async def forget(self, rollout_ids: list[str]) -> None:
        self.retired.forget(rollout_ids)
        await self._broadcast("forget", {"rollout_ids": rollout_ids}, timeout=_MESSAGE_TIMEOUT_SECONDS)

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        raise NotImplementedError("the coordinated policy participant exports asynchronously; use export()")

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        if self.ledger is None:
            return []
        # Workers are closed, so what they hold is stable: fetch it once, then read the ledgers off the loop.
        replies = await self._broadcast("snapshot", {}, timeout=_MESSAGE_TIMEOUT_SECONDS)
        snapshots = [GateSnapshot.model_validate(reply) for reply in replies.values()]
        return await asyncio.to_thread(
            export_model_records, self.ledger, episode_ids, snapshots, dict(self.restored_cuts)
        )

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        raise NotImplementedError("the policy participant restores through install(), which needs the restore scope")

    async def install(self, records: list[CheckpointRecord], scope: list[EpisodeId]) -> None:
        self._check_restorable()
        imported = await asyncio.to_thread(import_model_records, self.ledger, with_scope(records, scope, self.ledger))
        self.restored_cuts.update(imported)
        self.restored_targets = {next_attempt(episode_id).capture_key for episode_id in scope}
        self.targets_installed = True
        # The restored attempts continue as the next attempt, so their own ledgers are no longer used.
        await retire_ledgers(self.ledger, scope)

    async def restored_pending(self) -> list[EpisodeId]:
        if self.restored_targets:
            # An attempt any worker started is no longer pending.
            replies = await self._broadcast("restored_pending", {}, timeout=_MESSAGE_TIMEOUT_SECONDS)
            for reply in replies.values():
                self.restored_targets &= set(reply["targets"])
        return [EpisodeId.from_capture_key(key) for key in sorted(self.restored_targets)]

    async def delete_restored(self, episode_id: EpisodeId) -> None:
        """Delete a restored attempt no worker started, without the fence a retire leaves:
        the controller may still start the rollout over as this attempt."""
        key = episode_id.capture_key
        self.restored_targets.discard(key)
        self.restored_cuts.pop(key, None)
        if self.ledger is not None:
            await self.ledger.delete([key])

    def _check_restorable(self) -> None:
        if self.worker_left:
            raise ControlError("model restore requires freshly started policy workers; a worker has left since start")
        if any(worker.report is None or worker.report.served for worker in self.workers.values()):
            raise ControlError("model restore requires freshly started policy workers; a worker has served calls")

    def commit_reply(self, records: list[CheckpointRecord]) -> dict[str, Any]:
        return {
            "staging_keys": retained_staging_keys(records),
            "staging_keys_by_episode": staging_keys_by_episode(records),
        }

    def status_extra(self) -> dict[str, Any]:
        return {
            "workers": len(self.workers),
            "expected_workers": self.expected_workers,
            "restored_generation_cuts": sorted(self.restored_cuts),
        }


class PolicyCoordinator:
    """Serves the coordinated participant to the workers of one policy model server."""

    def __init__(
        self,
        ledger: Optional[CheckpointableLedger],
        *,
        expected_workers: int,
        instance_name: str,
        lease_grace_seconds: float,
        socket_path: str,
    ) -> None:
        self.participant = CoordinatedPolicyParticipant(ledger, expected_workers=expected_workers)
        self.controller = ParticipantControlPlane(
            self.participant, instance_name=instance_name, lease_grace_seconds=lease_grace_seconds
        )
        self.socket_path = socket_path
        self._next_worker_id = 0
        # Held for the life of the process: a collected server closes the workers' connections.
        self._server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def serve(self) -> asyncio.AbstractServer:
        self._server = await asyncio.start_unix_server(self._connected, path=self.socket_path)
        os.chmod(self.socket_path, 0o600)
        return self._server

    def start_in_background(self) -> None:
        """Serve on a daemon thread with its own event loop, alongside uvicorn's process supervisor."""
        ready = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            loop = self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.serve())
            except BaseException as error:
                failure.append(error)
                ready.set()
                return
            ready.set()
            loop.run_forever()

        threading.Thread(target=run, name="policy-checkpoint-coordinator", daemon=True).start()
        if not ready.wait(timeout=_MESSAGE_TIMEOUT_SECONDS):
            raise RuntimeError("the policy checkpoint coordinator did not start")
        if failure:
            raise RuntimeError("the policy checkpoint coordinator could not start") from failure[0]

    async def _connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._next_worker_id += 1
        worker_index = self._next_worker_id
        channel: Optional[_Channel] = None

        async def handle(kind: str, body: dict[str, Any]) -> dict[str, Any]:
            if kind == "register":
                return self.participant.join(worker_index, channel)
            if kind == "report":
                report = GateReport.model_validate(body["report"])
                await self.participant.receive_report(worker_index, body["generation"], body["seq"], report)
                return {}
            if kind == "control":
                return await dispatch_control(self.controller, body["operation"], body.get("body"))
            if kind == "claim_cut":
                record = self.participant.claim_cut(body["capture_key"])
                return {"record": record.model_dump(mode="json") if record is not None else None}
            if kind == "return_cut":
                self.participant.return_cut(body["capture_key"], GenerationCutRecord.model_validate(body["record"]))
                return {}
            raise ControlError(f"unknown checkpoint message {kind!r}")

        channel = _Channel(reader, writer, handle)
        try:
            await channel.run()
        finally:
            await self.participant.leave(worker_index)


# -- worker ---------------------------------------------------------------------------------------


def _terminate_this_worker() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


class CoordinatedRestoredCuts:
    """Restored cuts held by the coordinator; a worker claims one before running the call it may continue."""

    def __init__(self, link: "PolicyWorkerLink") -> None:
        self.link = link
        # Capture keys the coordinator holds a restored cut for, as of the last reopen.
        self.keys: set[str] = set()

    async def prefetch(self, ticket: _Ticket) -> None:
        if ticket.capture_key not in self.keys:
            return
        self.keys.discard(ticket.capture_key)
        try:
            reply = await self.link.call("claim_cut", {"capture_key": ticket.capture_key})
        except ControlError:
            LOGGER.warning("could not claim the restored cut of %s; the call regenerates", ticket.capture_key)
            return
        if reply["record"] is not None:
            ticket.claimed_cut = GenerationCutRecord.model_validate(reply["record"])

    def take(self, ticket: Optional[_Ticket], capture_key: str, request_digest: str) -> Optional[GenerationCutRecord]:
        if ticket is None or ticket.claimed_cut is None or ticket.claimed_cut.request_digest != request_digest:
            return None
        ticket.claimed_cut_used = True
        return ticket.claimed_cut

    async def settle(self, ticket: _Ticket) -> None:
        record, ticket.claimed_cut = ticket.claimed_cut, None
        if record is None or ticket.claimed_cut_used:
            return
        # The call was not the one the cut belongs to: give it back for the call that is.
        self.keys.add(ticket.capture_key)
        try:
            await self.link.call(
                "return_cut", {"capture_key": ticket.capture_key, "record": record.model_dump(mode="json")}
            )
        except ControlError:
            LOGGER.warning("could not release the restored cut of %s", ticket.capture_key)


class PolicyWorkerLink:
    """One worker's gate, driven by the coordinator in the main process."""

    def __init__(
        self,
        *,
        socket_path: str,
        server_name: str,
        cut_requester: Optional[CutRequester],
        on_coordinator_lost: Optional[Callable[[], None]] = None,
    ) -> None:
        self.socket_path = socket_path
        self.on_coordinator_lost = on_coordinator_lost or _terminate_this_worker
        self._disconnecting = False
        self.retired = RetiredAttempts()
        self.restored_cuts = CoordinatedRestoredCuts(self)
        self.gate = PolicyGate(
            server_name=server_name,
            cut_requester=cut_requester,
            retired=self.retired,
            restored_cuts=self.restored_cuts,
            on_change=self._changed,
        )
        self.generation = 0
        self._report_seq = 0
        self._channel: Optional[_Channel] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._report_task: Optional[asyncio.Task] = None
        self._report_pending = False

    async def connect(self, *, timeout: float = _MESSAGE_TIMEOUT_SECONDS) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(self.socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise CoordinatorUnavailableError(f"no policy checkpoint coordinator at {self.socket_path}")
                await asyncio.sleep(0.1)
        self._channel = _Channel(reader, writer, self._handle)
        self._reader_task = asyncio.create_task(self._channel.run())
        self._reader_task.add_done_callback(self._connection_ended)
        state = await self.call("register", {"pid": os.getpid()})
        self.restored_cuts.keys = set(state["restored_keys"])
        self.gate.restored_targets = set(state["restored_targets"])
        self.retired.update(state["retired"])
        if not state["accepting"]:
            # A checkpoint is open: close at once and report, like every other worker did.
            self.generation = state["generation"]
            await self.gate.close(CheckpointRequest.model_validate(state["request"]))
            await self._changed()

    async def disconnect(self) -> None:
        self._disconnecting = True
        if self._channel is not None:
            self._channel.close()
        if self._reader_task is not None:
            await asyncio.gather(self._reader_task, return_exceptions=True)

    def _connection_ended(self, task: asyncio.Task) -> None:
        if self._disconnecting:
            return
        # The main process, and uvicorn's supervisor with it, is gone.
        # A worker left behind would hold the port against the server's restart
        # and could never take part in a checkpoint again.
        LOGGER.error("lost the policy checkpoint coordinator; shutting this worker down")
        self._channel = None
        self.on_coordinator_lost()

    async def call(
        self, kind: str, body: dict[str, Any], *, timeout: float = _MESSAGE_TIMEOUT_SECONDS
    ) -> dict[str, Any]:
        if self._channel is None:
            raise CoordinatorUnavailableError("this worker is not connected to the policy checkpoint coordinator")
        return await self._channel.call(kind, body, timeout=timeout)

    async def dispatch(self, operation: str, body: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Serve one checkpoint control route by forwarding it to the coordinator."""
        deadline_ts = (body or {}).get("deadline_ts")
        timeout = _MESSAGE_TIMEOUT_SECONDS if deadline_ts is None else max(0.0, deadline_ts - time.time())
        return await self.call(
            "control", {"operation": operation, "body": body}, timeout=timeout + _REPLY_GRACE_SECONDS
        )

    async def _handle(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        if kind == "close":
            self.generation = body["generation"]
            await self.gate.close(CheckpointRequest.model_validate(body["request"]))
            return self._numbered_report()
        if kind == "open":
            self.restored_cuts.keys = set(body["restored_keys"])
            if body["restored_targets"] is not None:
                self.gate.restored_targets = set(body["restored_targets"])
            self.gate.open()
            return {}
        if kind == "restored_pending":
            return {"targets": sorted(self.gate.restored_targets)}
        if kind == "snapshot":
            return self.gate.snapshot().model_dump(mode="json")
        if kind == "mark":
            # This worker refuses the attempts from now on, until the controller forgets the rollout.
            self.retired.mark([EpisodeId.model_validate(e) for e in body["episode_ids"]])
            return {}
        if kind == "retire":
            await self.gate.retire(EpisodeId.model_validate(body["episode_id"]))
            return {}
        if kind == "forget":
            self.retired.forget(body["rollout_ids"])
            return {}
        raise ControlError(f"unknown checkpoint message {kind!r}")

    async def _changed(self) -> None:
        """Report to the coordinator while a checkpoint is open; coalesce bursts of changes."""
        if self.gate.accepting or self._channel is None:
            return
        self._report_pending = True
        if self._report_task is None or self._report_task.done():
            self._report_task = asyncio.create_task(self._send_reports())

    async def _send_reports(self) -> None:
        while self._report_pending and not self.gate.accepting:
            self._report_pending = False
            try:
                await self.call("report", {"generation": self.generation, **self._numbered_report()})
            except ControlError:
                LOGGER.warning("could not report to the policy checkpoint coordinator", exc_info=True)
                return

    def _numbered_report(self) -> dict[str, Any]:
        self._report_seq += 1
        return {"seq": self._report_seq, "report": self.gate.report().model_dump(mode="json")}
