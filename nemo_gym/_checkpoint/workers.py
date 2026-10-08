# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A server with several uvicorn workers is still one checkpoint participant.

uvicorn's workers share one port, so a control call reaches an arbitrary worker.
One coordinator in the main process, which serves no requests, owns the participant:
the control plane (phases, checkpoint-ID fencing, lease, storage).
Each worker holds its own share of the state and:

- forwards every checkpoint control call to the coordinator;
- closes, reopens, and retires when the coordinator tells it to, and reports its readiness while a checkpoint is open.

Workers talk to the coordinator over a Unix socket: length-prefixed JSON frames,
with request and reply messages in both directions on one connection per worker.

Failures close the checkpoint rather than weaken it.
A worker that has not reported for the current checkpoint, or that disconnected while it was open,
blocks prepare and commit until the controller resumes.
A worker that starts, or restarts, while a checkpoint is open closes immediately.
A worker that loses the coordinator stops itself.

Each participant kind adds what its state needs: a ``CoordinatedParticipant`` subclass in the main process,
and a ``WorkerLink`` subclass in each worker.
"""

import asyncio
import json
import logging
import os
import signal
import threading
import time
import uuid
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Generic, Optional, TypeVar

from pydantic import BaseModel

from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRequest,
    ParticipantControlPlane,
    PrepareReport,
    RetiredAttempts,
    dispatch_control,
)
from nemo_gym._checkpoint.errors import ControlError
from nemo_gym.episode_types import EpisodeId
from nemo_gym.runtime_dir import server_runtime_dir


LOGGER = logging.getLogger(__name__)

# Commit and restore carry a worker's records in one message,
# which can reach hundreds of megabytes for servers with large sessions;
# the limit only rejects a corrupt length prefix.
_MAX_FRAME_BYTES = 1024 * 1024 * 1024
# How long past a control call's own deadline a worker waits for the coordinator's reply.
REPLY_GRACE_SECONDS = 10.0
# Timeout for messages that do no slow work (registration, reports, claims, reopening).
MESSAGE_TIMEOUT_SECONDS = 30.0


class CoordinatorUnavailableError(ControlError):
    status_code = 503
    code = "checkpoint_coordinator_unavailable"


class _RemoteControlError(ControlError):
    """A control error raised in the other process, re-raised here with the same status and code."""

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code


def coordinator_socket_path(prefix: str) -> str:
    return os.path.join(server_runtime_dir(), f"{prefix}-{uuid.uuid4().hex[:8]}.sock")


def _frame(message: dict[str, Any]) -> bytes:
    # The same encoder as the checkpoint writer:
    # orjson would refuse integers beyond 64 bits and silently turn NaN and infinities into null,
    # so a record could fail to reach a worker, or arrive changed.
    data = json.dumps(message, separators=(",", ":")).encode()
    if len(data) > _MAX_FRAME_BYTES:
        raise ValueError(f"checkpoint message of {len(data)} bytes exceeds the frame limit")
    return len(data).to_bytes(4, "big") + data


async def _read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    size = int.from_bytes(await reader.readexactly(4), "big")
    if size > _MAX_FRAME_BYTES:
        raise ValueError(f"checkpoint message of {size} bytes exceeds the frame limit")
    return json.loads(await reader.readexactly(size))


# Handles one incoming message kind and body; returns the reply body.
Handler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
# Handles a message kind at once, on the reading loop, or returns None to leave it to the ``Handler``.
ImmediateHandler = Callable[[str, dict[str, Any]], Optional[dict[str, Any]]]


class Channel:
    """Request and reply messages in both directions over one connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        handler: Handler,
        *,
        immediate: Optional[ImmediateHandler] = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._handler = handler
        self._immediate = immediate
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    async def call(self, kind: str, body: dict[str, Any], *, timeout: Optional[float]) -> dict[str, Any]:
        """Send a message and wait for its reply; a ``timeout`` of None waits as long as the caller does."""
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
                elif not self._reply_immediately(message):
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
            self._writer.write(_frame(message))
            await self._writer.drain()

    def _reply_immediately(self, message: dict[str, Any]) -> bool:
        """Answer a message without yielding to the event loop, so its reply precedes any later message.

        Each frame is written whole, so this reply never interleaves with a frame another task is sending.
        """
        if self._immediate is None:
            return False
        try:
            body = self._immediate(message["kind"], message["body"])
        except ControlError as error:
            reply = _error_reply(message, error.status_code, error.code, error.detail)
        else:
            if body is None:
                return False
            reply = {"reply_to": message["id"], "ok": True, "body": body}
        self._writer.write(_frame_reply(message, reply))
        return True

    async def _serve(self, message: dict[str, Any]) -> None:
        try:
            reply = {
                "reply_to": message["id"],
                "ok": True,
                "body": await self._handler(message["kind"], message["body"]),
            }
        except ControlError as error:
            reply = _error_reply(message, error.status_code, error.code, error.detail)
        except Exception as error:
            # A process boundary: report the failure to the caller instead of dropping the reply.
            LOGGER.exception("checkpoint message %r failed", message.get("kind"))
            reply = _error_reply(message, 500, "checkpoint_error", str(error))
        try:
            async with self._write_lock:
                self._writer.write(_frame_reply(message, reply))
                await self._writer.drain()
        except (ConnectionError, RuntimeError):
            pass


def _frame_reply(message: dict[str, Any], reply: dict[str, Any]) -> bytes:
    """Frame ``reply``, or a typed error if it cannot be framed: the caller must not wait until its deadline."""
    try:
        return _frame(reply)
    except (TypeError, ValueError) as error:
        LOGGER.exception("checkpoint reply to %r cannot be sent", message.get("kind"))
        return _frame(_error_reply(message, 422, "invalid_checkpoint_state", str(error)))


def _error_reply(message: dict[str, Any], status: int, code: str, detail: str) -> dict[str, Any]:
    return {"reply_to": message["id"], "ok": False, "error": {"status": status, "code": code, "detail": detail}}


# -- coordinator (main process) --------------------------------------------------------------------


ReportT = TypeVar("ReportT", bound=BaseModel)


class _Worker(Generic[ReportT]):
    def __init__(self, channel: Channel, registration: dict[str, Any]) -> None:
        self.channel = channel
        self.registration = registration
        self.report: Optional[ReportT] = None
        # Reports can arrive out of order across the reply and push paths; keep the newest.
        self.report_seq = -1

    def keep(self, seq: int, report: ReportT) -> None:
        if seq > self.report_seq:
            self.report_seq = seq
            self.report = report


class CoordinatedParticipant(CheckpointParticipant, Generic[ReportT]):
    """The participant of a server whose state is spread over several worker processes.

    A subclass names the report its workers send (``report_model``) and how to merge them,
    and adds the state the coordinator itself holds through the ``*_state`` hooks and ``handle``.
    """

    report_model: ClassVar[type[BaseModel]]
    # Names this server's workers in blockers and errors.
    worker_label: ClassVar[str]

    def __init__(self, *, expected_workers: int) -> None:
        super().__init__()
        self.expected_workers = expected_workers
        self.workers: dict[int, _Worker[ReportT]] = {}
        self.accepting = True
        # Increments with each close, so a report from before it is never mistaken for a current one.
        self.generation = 0
        self.request: Optional[CheckpointRequest] = None
        # A worker disconnected while admission was closed: what it held is unknown.
        self.lost_worker = False

    # -- per-kind hooks ---------------------------------------------------------------------------

    @abstractmethod
    def merge(self, reports: list[ReportT]) -> PrepareReport:
        """Combine the workers' reports into one."""

    def join_state(self, worker_index: int) -> dict[str, Any]:
        """Kind-specific state a joining worker adopts."""
        return {}

    def open_state(self) -> dict[str, Any]:
        """Kind-specific state every worker adopts on reopening."""
        return {}

    def retire_restored(self, episode_id: EpisodeId) -> None:
        """Discard restored state the coordinator holds for ``episode_id`` and earlier attempts."""

    async def handle(self, worker_index: int, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        """Serve a kind-specific message from a worker."""
        raise ControlError(f"unknown checkpoint message {kind!r}")

    def handle_now(self, worker_index: int, kind: str, body: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Serve a kind-specific message before any later message to that worker, or return None."""
        return None

    # -- workers ----------------------------------------------------------------------------------

    def join(self, worker_index: int, channel: Channel, registration: dict[str, Any]) -> dict[str, Any]:
        """Register a worker and return the state it must adopt."""
        self.workers[worker_index] = _Worker(channel, registration)
        return {
            "accepting": self.accepting,
            "generation": self.generation,
            "request": self.request.model_dump(mode="json") if self.request is not None else None,
            # A worker that starts after a retire must refuse that attempt's late requests too.
            "retired": self.retired.marks(),
            **self.join_state(worker_index),
        }

    async def leave(self, worker_index: int) -> None:
        if self.workers.pop(worker_index, None) is not None and not self.accepting:
            self.lost_worker = True
        await self.notify()

    async def receive_report(self, worker_index: int, generation: int, seq: int, report: dict[str, Any]) -> None:
        worker = self.workers.get(worker_index)
        if worker is not None and generation == self.generation and not self.accepting:
            worker.keep(seq, self.report_model.model_validate(report))
            await self.notify()

    async def broadcast(
        self, kind: str, body: dict[str, Any], *, timeout: Optional[float]
    ) -> dict[int, dict[str, Any]]:
        workers = dict(self.workers)
        return await self.call_each({worker_index: (kind, body) for worker_index in workers}, timeout=timeout)

    async def call_each(
        self, messages: dict[int, tuple[str, dict[str, Any]]], *, timeout: Optional[float]
    ) -> dict[int, dict[str, Any]]:
        """Send each worker its own message at once; fail if any worker does not complete it."""
        calls = {worker_index: message for worker_index, message in messages.items() if worker_index in self.workers}
        results = await asyncio.gather(
            *(
                self.workers[worker_index].channel.call(kind, body, timeout=timeout)
                for worker_index, (kind, body) in calls.items()
            ),
            return_exceptions=True,
        )
        replies = {}
        for (worker_index, (kind, _)), result in zip(calls.items(), results):
            if isinstance(result, ControlError):
                # Keep the worker's status and code, such as invalid_phase, so the controller can act on it.
                raise _RemoteControlError(
                    result.status_code,
                    result.code,
                    f"{self.worker_label} worker {worker_index} did not complete {kind}: {result.detail}",
                ) from result
            if isinstance(result, BaseException):
                raise ControlError(
                    f"{self.worker_label} worker {worker_index} did not complete {kind}: {result}"
                ) from result
            replies[worker_index] = result
        return replies

    # -- participant ------------------------------------------------------------------------------

    async def close_admission(self, request: CheckpointRequest) -> None:
        self.accepting = False
        self.generation += 1
        # Restore closes admission with a RestoreRequest; workers need only the checkpoint and deadline.
        request = CheckpointRequest(checkpoint_id=request.checkpoint_id, deadline_ts=request.deadline_ts)
        self.request = request
        for worker in self.workers.values():
            worker.report = None
        replies = await self.broadcast(
            "close",
            {"request": request.model_dump(mode="json"), "generation": self.generation},
            timeout=max(0.0, request.deadline_ts - time.time()) + REPLY_GRACE_SECONDS,
        )
        for worker_index, reply in replies.items():
            if worker_index in self.workers:
                self.workers[worker_index].keep(reply["seq"], self.report_model.model_validate(reply["report"]))

    async def open_admission(self) -> None:
        self.accepting = True
        self.request = None
        self.lost_worker = False
        for worker in self.workers.values():
            worker.report = None
        try:
            await self.broadcast("open", self.open_state(), timeout=MESSAGE_TIMEOUT_SECONDS)
        except ControlError:
            # A worker that cannot be reached is gone; one that restarts adopts the open state on joining.
            LOGGER.warning("a %s worker did not acknowledge reopening", self.worker_label, exc_info=True)

    def readiness(self) -> PrepareReport:
        reports = [worker.report for worker in self.workers.values() if worker.report is not None]
        merged = self.merge(reports)
        missing = []
        if len(reports) < self.expected_workers:
            missing.append(f"{self.worker_label}-workers-unreported:{self.expected_workers - len(reports)}")
        if self.lost_worker:
            missing.append(f"{self.worker_label}-worker-lost")
        blockers = missing + list(merged.blockers)
        return PrepareReport(
            ready=merged.ready and not missing,
            blockers=blockers,
            counts={**merged.counts, "workers": len(self.workers)},
            restarts=merged.restarts,
        )

    async def retire(self, episode_id: EpisodeId) -> None:
        self.retire_restored(episode_id)
        await self.broadcast(
            "retire", {"episode_id": episode_id.model_dump(mode="json")}, timeout=MESSAGE_TIMEOUT_SECONDS
        )

    async def mark_retired(self, episode_ids: list[EpisodeId]) -> None:
        self.retired.mark(episode_ids)
        await self.broadcast(
            "mark", {"episode_ids": [e.model_dump(mode="json") for e in episode_ids]}, timeout=MESSAGE_TIMEOUT_SECONDS
        )

    async def forget(self, rollout_ids: list[str]) -> None:
        # Every worker must be able to forget before any of them does, so a refusal leaves every refusal in place.
        await self.broadcast("check_forget", {"rollout_ids": rollout_ids}, timeout=MESSAGE_TIMEOUT_SECONDS)
        self.retired.forget(rollout_ids)
        await self.broadcast("forget", {"rollout_ids": rollout_ids}, timeout=MESSAGE_TIMEOUT_SECONDS)

    def status_extra(self) -> dict[str, Any]:
        return {"workers": len(self.workers), "expected_workers": self.expected_workers}


class WorkerCoordinator:
    """Serves one coordinated participant to the workers of one server."""

    def __init__(
        self,
        participant: CoordinatedParticipant,
        *,
        instance_name: str,
        lease_grace_seconds: float,
        socket_path: str,
    ) -> None:
        self.participant = participant
        self.controller = ParticipantControlPlane(
            participant, instance_name=instance_name, lease_grace_seconds=lease_grace_seconds
        )
        self.socket_path = socket_path
        self._next_worker_index = 0
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
        label = self.participant.worker_label

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

        threading.Thread(target=run, name=f"{label}-checkpoint-coordinator", daemon=True).start()
        if not ready.wait(timeout=MESSAGE_TIMEOUT_SECONDS):
            raise RuntimeError(f"the {label} checkpoint coordinator did not start")
        if failure:
            raise RuntimeError(f"the {label} checkpoint coordinator could not start") from failure[0]

    async def _connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._next_worker_index += 1
        worker_index = self._next_worker_index
        channel: Optional[Channel] = None

        async def handle(kind: str, body: dict[str, Any]) -> dict[str, Any]:
            if kind == "register":
                return self.participant.join(worker_index, channel, body)
            if kind == "report":
                await self.participant.receive_report(worker_index, body["generation"], body["seq"], body["report"])
                return {}
            if kind == "control":
                return await dispatch_control(self.controller, body["operation"], body.get("body"))
            return await self.participant.handle(worker_index, kind, body)

        def handle_now(kind: str, body: dict[str, Any]) -> Optional[dict[str, Any]]:
            return self.participant.handle_now(worker_index, kind, body)

        channel = Channel(reader, writer, handle, immediate=handle_now)
        try:
            await channel.run()
        finally:
            await self.participant.leave(worker_index)


# -- worker ---------------------------------------------------------------------------------------


def _terminate_this_worker() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


class WorkerLink:
    """One worker's share of a coordinated participant, driven by the coordinator in the main process.

    A subclass applies the coordinator's instructions to this worker's state through ``close_local``,
    ``open_local``, ``retire_local``, and ``handle``, and calls ``changed`` whenever its readiness may have changed.
    """

    # Names this server's workers in errors.
    worker_label: ClassVar[str]

    def __init__(self, *, socket_path: str, on_coordinator_lost: Optional[Callable[[], None]] = None) -> None:
        self.socket_path = socket_path
        self.on_coordinator_lost = on_coordinator_lost or _terminate_this_worker
        self._disconnecting = False
        self.retired = RetiredAttempts()
        self.generation = 0
        self._report_seq = 0
        self._channel: Optional[Channel] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._report_task: Optional[asyncio.Task] = None
        self._report_pending = False

    # -- per-kind hooks ---------------------------------------------------------------------------

    @property
    @abstractmethod
    def accepting(self) -> bool:
        """Whether this worker's admission is open."""

    @abstractmethod
    async def close_local(self, request: CheckpointRequest) -> None:
        """Close this worker's admission for a checkpoint."""

    @abstractmethod
    async def open_local(self, body: dict[str, Any]) -> None:
        """Reopen this worker's admission, adopting the coordinator's ``open_state``."""

    @abstractmethod
    async def retire_local(self, episode_id: EpisodeId) -> None:
        """Discard this worker's state of ``episode_id`` and earlier attempts."""

    def check_forget_local(self, rollout_ids: list[str]) -> None:
        """Raise if this worker cannot forget these rollouts yet; nothing changes either way."""

    async def forget_local(self, rollout_ids: list[str]) -> None:
        """Stop refusing these rollouts' retired attempts in this worker."""
        self.retired.forget(rollout_ids)

    @abstractmethod
    def report(self) -> BaseModel:
        """This worker's readiness."""

    def registration(self) -> dict[str, Any]:
        """What this worker tells the coordinator about itself when it joins."""
        return {"pid": os.getpid()}

    def adopt(self, state: dict[str, Any]) -> None:
        """Adopt the coordinator's ``join_state``."""

    async def handle(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        """Serve a kind-specific message from the coordinator."""
        raise ControlError(f"unknown checkpoint message {kind!r}")

    # -- connection -------------------------------------------------------------------------------

    async def connect(self, *, timeout: float = MESSAGE_TIMEOUT_SECONDS) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(self.socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise CoordinatorUnavailableError(
                        f"no {self.worker_label} checkpoint coordinator at {self.socket_path}"
                    )
                await asyncio.sleep(0.1)
        self._channel = Channel(reader, writer, self._handle)
        self._reader_task = asyncio.create_task(self._channel.run())
        self._reader_task.add_done_callback(self._connection_ended)
        state = await self.call("register", self.registration())
        self.retired.update(state["retired"])
        self.adopt(state)
        if not state["accepting"]:
            # A checkpoint is open: close at once and report, like every other worker did.
            self.generation = state["generation"]
            await self.close_local(CheckpointRequest.model_validate(state["request"]))
            await self.changed()

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
        LOGGER.error("lost the %s checkpoint coordinator; shutting this worker down", self.worker_label)
        self._channel = None
        self.on_coordinator_lost()

    async def call(
        self, kind: str, body: dict[str, Any], *, timeout: Optional[float] = MESSAGE_TIMEOUT_SECONDS
    ) -> dict[str, Any]:
        if self._channel is None:
            raise CoordinatorUnavailableError(
                f"this worker is not connected to the {self.worker_label} checkpoint coordinator"
            )
        return await self._channel.call(kind, body, timeout=timeout)

    async def dispatch(self, operation: str, body: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Serve one checkpoint control route by forwarding it to the coordinator."""
        deadline_ts = (body or {}).get("deadline_ts")
        timeout = MESSAGE_TIMEOUT_SECONDS if deadline_ts is None else max(0.0, deadline_ts - time.time())
        return await self.call(
            "control", {"operation": operation, "body": body}, timeout=timeout + REPLY_GRACE_SECONDS
        )

    async def _handle(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        if kind == "close":
            self.generation = body["generation"]
            await self.close_local(CheckpointRequest.model_validate(body["request"]))
            return self._numbered_report()
        if kind == "open":
            await self.open_local(body)
            return {}
        if kind == "mark":
            # This worker refuses the attempts from now on, until the controller forgets the rollout.
            self.retired.mark([EpisodeId.model_validate(e) for e in body["episode_ids"]])
            return {}
        if kind == "retire":
            await self.retire_local(EpisodeId.model_validate(body["episode_id"]))
            return {}
        if kind == "check_forget":
            self.check_forget_local(body["rollout_ids"])
            return {}
        if kind == "forget":
            await self.forget_local(body["rollout_ids"])
            return {}
        return await self.handle(kind, body)

    # -- reports ----------------------------------------------------------------------------------

    async def changed(self) -> None:
        """Report to the coordinator while a checkpoint is open; coalesce bursts of changes."""
        if self.accepting or self._channel is None:
            return
        self._report_pending = True
        if self._report_task is None or self._report_task.done():
            self._report_task = asyncio.create_task(self._send_reports())

    async def _send_reports(self) -> None:
        while self._report_pending and not self.accepting:
            self._report_pending = False
            try:
                await self.call("report", {"generation": self.generation, **self._numbered_report()})
            except ControlError:
                LOGGER.warning("could not report to the %s checkpoint coordinator", self.worker_label, exc_info=True)
                return

    def _numbered_report(self) -> dict[str, Any]:
        self._report_seq += 1
        return {"seq": self._report_seq, "report": self.report().model_dump(mode="json")}
