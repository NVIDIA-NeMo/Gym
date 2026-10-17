# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment, agent, and resources servers with several uvicorn workers.

Each worker runs the same participant a single-process server runs, for the episodes and sessions it
holds, linked to one coordinator in the main process (see ``nemo_gym._checkpoint.workers``). The
coordinator is the server's participant:

- **prepare** closes every worker and is ready once every worker reports ready.
  The restarts that earlier prepare stages reported reach every worker, including one that joins later,
  and each worker leaves them out of its blockers until the checkpoint resumes.
- **commit** asks every worker to export its records and writes the participant's one records file and
  manifest.
- **restore** places each record by where its state lives:

  - State that spans requests (resources sessions, agent sessions) is installed on a live worker,
    spread evenly one session at a time.
    That worker rewrites the session's routing claim to name itself (see ``nemo_gym.session_routing``),
    and every other worker drops any claim it held on the session.
    Cookies and MCP tokens name only the session ID, so one from before the crash,
    or from a single-process server, reaches the worker that holds the session now.
  - State that lives inside one request (environment episodes, an agent's own ``/run`` episodes) stays with
    the coordinator. The worker that receives the replacement attempt's ``/run`` claims it; exactly one
    claim succeeds. The claim leases the record to that worker, which settles it once it has installed the
    record, or returns it if the install failed. A record whose worker is lost goes back too. While a lease
    is open, prepare waits for it. A record nobody has claimed is exported again by the next checkpoint.

  A restore is all or nothing across the workers: if one worker's install fails,
  the workers that installed theirs delete what they installed, and release its claims, before the restore fails.
  A worker that holds restored sessions and is lost before the resume that follows the restore fails that resume:
  their state is gone, so the restore must be repeated on a restarted server.

- **retire** reaches every worker, and the coordinator retires what it holds for the attempt.

Claims are refused while a checkpoint is open, as a new episode is: the caller retries the episode later.
A claim is answered before any later message to its worker, so a worker that wins a claim starts the
episode before it can see the next checkpoint close, and that checkpoint waits for the episode instead of
missing it.
"""

import asyncio
import logging
import os
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from numbers import Number
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym._checkpoint.control import (
    PARSED_JSON_CONTEXT,
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    PrepareReport,
    install_control_routes,
    install_participant,
    next_attempt,
)
from nemo_gym._checkpoint.errors import AdmissionClosedError, ControlError, InvalidPhaseError
from nemo_gym._checkpoint.workers import (
    EXPORT_CHUNK_BYTES,
    MESSAGE_TIMEOUT_SECONDS,
    CoordinatedParticipant,
    CoordinatorUnavailableError,
    RawJSON,
    WorkerCoordinator,
    WorkerLink,
    WorkerLostError,
    coordinator_socket_path,
    encode_records,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import is_nemo_gym_fastapi_worker
from nemo_gym.session_routing import ROUTED_LIFESPANS_STATE, ResourcesSessionLifetimes, SessionClaims


LOGGER = logging.getLogger(__name__)

# Set by the main process for the workers it spawns.
COORDINATOR_SOCKET_ENV = "NEMO_GYM_CHECKPOINT_COORDINATOR_SOCKET"
# While a checkpoint is open, a worker re-reports this often if its readiness changed without a notification.
_REPORT_POLL_SECONDS = 0.5
# How long a status request waits for each worker's own status.
_STATUS_TIMEOUT_SECONDS = 2.0
# How many restored copies one worker deletes at once.
_DELETE_CONCURRENCY = 16
# A worker that could not settle a claimed record retries this often, until it succeeds or the connection closes.
_SETTLE_RETRY_SECONDS = 1.0


class CoordinatedServerParticipant(CoordinatedParticipant[PrepareReport]):
    """The participant of an environment, agent, or resources server with several workers."""

    report_model = PrepareReport

    def __init__(self, participant: CheckpointParticipant, *, expected_workers: int) -> None:
        super().__init__(expected_workers=expected_workers)
        # The coordinator stands in for the workers' participants, so it shares their kind and records.
        self.kind = participant.kind
        self.record_model = participant.record_model
        self.worker_label = participant.kind
        self.claim_key = participant.claim_key
        # The workers' participant kind, which also shapes the commit reply the controller reads.
        self._participant = participant
        # Restored records a replacement has not claimed yet, by claim key.
        self.restored: dict[str, CheckpointRecord] = {}
        # Claimed records the claiming worker has not settled yet, by claim key, with that worker's index.
        # A lease ends when the worker settles it, when the worker is lost, or when a retire discards its attempt.
        self.leases: dict[str, tuple[int, CheckpointRecord]] = {}
        # The records each export receives from each worker, while that export runs.
        self._exports: dict[int, dict[int, list[CheckpointRecord]]] = {}
        self._next_export = 0
        # Each worker's own status and readiness as of the last status request, which replaces them.
        self._worker_status: list[dict[str, Any]] = []
        self._worker_reports: list[PrepareReport] = []
        # The workers the last restore installed sessions on, from its install until the resume that reopens them,
        # and those of them lost meanwhile, whose restored sessions are gone.
        self._restored_workers: set[int] = set()
        self._restored_lost: set[int] = set()

    def merge(self, reports: list[PrepareReport]) -> PrepareReport:
        counts: Counter[str] = Counter()
        for report in reports:
            counts.update(report.counts)
        return PrepareReport(
            ready=all(report.ready for report in reports),
            blockers=sorted({blocker for report in reports for blocker in report.blockers}),
            counts=dict(counts),
            restarts=sorted({key for report in reports for key in report.restarts}),
        )

    def join_state(self, worker_index: int) -> dict[str, Any]:
        return {
            "restored_keys": sorted(self.restored),
            # A worker that joins while a checkpoint is open leaves out the restarts the others leave out.
            "carried_restarts": sorted(self.carried_restarts),
        }

    def open_state(self) -> dict[str, Any]:
        return {"restored_keys": sorted(self.restored)}

    async def carry_restarts(self, capture_keys: Iterable[str]) -> None:
        """Tell every worker as well: the episodes' sessions and requests live on the workers."""
        carried = self.carried_restarts | set(capture_keys)
        if carried == self.carried_restarts:
            return
        replies = await self.broadcast(
            "carry_restarts", {"restarts": sorted(carried)}, timeout=MESSAGE_TIMEOUT_SECONDS
        )
        # Kept only once every worker has them, so a prepare asked again after a failure sends them again.
        self.carried_restarts = carried
        for worker_index, reply in replies.items():
            if worker_index in self.workers:
                self.workers[worker_index].keep(reply["seq"], self.report_model.model_validate(reply["report"]))
        await self.notify()

    def retire_restored(self, episode_id: EpisodeId) -> None:
        def discarded(record: CheckpointRecord) -> bool:
            replacement = next_attempt(record.episode_id)
            return replacement.rollout_id == episode_id.rollout_id and replacement.attempt <= episode_id.attempt

        for key, record in list(self.restored.items()):
            if discarded(record):
                del self.restored[key]
        # The worker that holds a lease retires what it installed; the record must not come back if it fails.
        for key, (_, record) in list(self.leases.items()):
            if discarded(record):
                del self.leases[key]

    async def leave(self, worker_index: int) -> None:
        # What a lost worker claimed and had not settled may not have been installed anywhere: it can be claimed again.
        for key, (holder, record) in list(self.leases.items()):
            if holder == worker_index:
                del self.leases[key]
                self.restored[key] = record
        if worker_index in self._restored_workers and not self.accepting:
            self._restored_lost.add(worker_index)
        await super().leave(worker_index)

    async def open_admission(self) -> None:
        """Reopen every worker, unless a worker holding sessions the last restore installed was lost before it.

        Those sessions' state is gone, and their requests would get a 410.
        The resume fails instead, and keeps failing.
        The controller then repeats the restore on a restarted server, rather than continue those episodes without it.
        """
        if self._restored_lost:
            raise WorkerLostError(
                f"{self.kind} workers {sorted(self._restored_lost)} were lost after a restore installed sessions "
                "on them and before the resume, so the restored state of those sessions is gone; "
                "restart the server and restore again"
            )
        failed = await self.reopen_workers()
        lost = failed & self._restored_workers
        if lost:
            self._restored_lost |= lost
            self.accepting = False
            raise WorkerLostError(
                f"{self.kind} workers {sorted(lost)} hold restored sessions and did not reopen; "
                "restart the server and restore again"
            )
        self._restored_workers.clear()

    def readiness(self) -> PrepareReport:
        report = super().readiness()
        if not self.leases:
            return report
        # A claimed record is being installed on its worker: wait until it is settled one way or the other.
        unsettled = sorted(next_attempt(record.episode_id).capture_key for _, record in self.leases.values())
        return report.model_copy(update={"ready": False, "blockers": unsettled + list(report.blockers)})

    async def restored_pending(self) -> list[EpisodeId]:
        """Restored records no worker has claimed, and restored sessions no worker has used yet."""
        pending = {next_attempt(record.episode_id) for record in self.restored.values()}
        replies = await self.broadcast("restored_pending", {}, timeout=MESSAGE_TIMEOUT_SECONDS)
        for reply in replies.values():
            pending.update(EpisodeId.model_validate(episode_id) for episode_id in reply["episode_ids"])
        return sorted(pending, key=lambda episode_id: episode_id.capture_key)

    async def delete_restored(self, episode_id: EpisodeId) -> None:
        """Delete the restored state of ``episode_id`` here and on every worker, refusing nothing later.

        Each worker deletes only what no request has used, as a single process does;
        a retire would also free the sessions the replacement uses and refuse them until a forget that never comes.
        """
        self.retire_restored(episode_id)
        # As long as the deletion takes, as in a single process; a resume waits for it within its own deadline.
        await self.broadcast("delete_restored", {"episode_id": episode_id.model_dump(mode="json")}, timeout=None)

    def handle_now(self, worker_index: int, kind: str, body: dict[str, Any]) -> Optional[dict[str, Any]]:
        if kind != "claim":
            return None
        if not self.accepting:
            raise AdmissionClosedError(f"{self.kind} admission is closed for a checkpoint")
        key = body["key"]
        record = self.restored.pop(key, None)
        if record is None:
            return {"record": None}
        self.leases[key] = (worker_index, record)
        return {"record": record.to_json_record()}

    async def handle(self, worker_index: int, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        if kind == "settle_claim":
            lease = self.leases.get(body["key"])
            if lease is not None and lease[0] == worker_index:
                del self.leases[body["key"]]
                if not body["installed"]:
                    self.restored[body["key"]] = lease[1]
                await self.notify()
            return {}
        if kind == "export_chunk":
            received = self._exports.get(body["export"])
            if received is not None:
                # Checked as it arrives, off the event loop, while other workers are still exporting.
                records = await asyncio.to_thread(self._validate_records, body["records"])
                received.setdefault(worker_index, []).extend(records)
            return {}
        return await super().handle(worker_index, kind, body)

    def _validate_records(self, raw: list[dict[str, Any]]) -> list[CheckpointRecord]:
        # Built and checked on the worker, then parsed from its message: its payloads are JSON.
        return [self.record_model.model_validate(record, context=PARSED_JSON_CONTEXT) for record in raw]

    # -- commit and restore -------------------------------------------------------------------------

    def commit_reply(self, records: list[CheckpointRecord]) -> dict[str, Any]:
        # The coordinator answers the commit for its workers,
        # so the fields a participant kind adds to the reply are built here from the merged records.
        return self._participant.commit_reply(records)

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        raise NotImplementedError("a coordinated participant exports its workers' records through export()")

    async def export(self, episode_ids: Optional[list[EpisodeId]], *, checkpoint_id: str) -> list[CheckpointRecord]:
        self._next_export += 1
        export = self._next_export
        body = {
            "episode_ids": None if episode_ids is None else [e.model_dump(mode="json") for e in episode_ids],
            "export": export,
            "checkpoint_id": checkpoint_id,
        }
        # Each worker sends its records in chunks before it replies; they are kept here only while this export runs.
        self._exports[export] = {}
        try:
            # The controller bounds the export by the commit's deadline.
            replies = await self.collect("export", body, timeout=None)
            received = self._exports[export]
        finally:
            self._exports.pop(export, None)
        for worker_index, reply in replies.items():
            if len(received.get(worker_index, ())) != reply["count"]:
                raise ControlError(
                    f"{self.worker_label} worker {worker_index} exported {reply['count']} records, "
                    f"of which {len(received.get(worker_index, ()))} arrived"
                )
        records = await self._one_copy_per_session(received, replies)
        # An unclaimed restored episode still continues from its restored state while the commit's scope names it.
        scope = None if episode_ids is None else set(episode_ids)
        pending = [
            record.model_copy(update={"episode_id": next_attempt(record.episode_id)})
            for record in self.restored.values()
            if scope is None or next_attempt(record.episode_id) in scope
        ]
        return records + pending

    async def _one_copy_per_session(
        self, received: dict[int, list[CheckpointRecord]], replies: dict[int, dict[str, Any]]
    ) -> list[CheckpointRecord]:
        """The workers' records, with one record per piece of state, as the participant kind identifies it.

        A restored session can be seeded again on another worker,
        for example when an episode continues from before its seed and the seed is routed elsewhere.
        That worker's copy is live; the restored copy nothing used is stale, so it is left out and freed.
        Records the kind gives no identity, and records of different identities, are all kept.
        """
        copies: dict[Any, list[tuple[int, CheckpointRecord, bool]]] = defaultdict(list)
        for worker_index, records in received.items():
            unclaimed = set(replies[worker_index]["unclaimed"])
            for record in records:
                identity = self._participant.record_identity(record)
                key = identity if identity is not None else (worker_index, len(copies))
                copies[key].append((worker_index, record, identity in unclaimed))
        kept_records: list[CheckpointRecord] = []
        stale: dict[int, list[str]] = defaultdict(list)
        for key, found in copies.items():
            live = [copy for copy in found if not copy[2]]
            kept = (live or found)[0][1]
            kept_records.append(kept)
            for worker_index, record, unclaimed_copy in found:
                if unclaimed_copy and record is not kept:
                    stale[worker_index].append(key)
            if len(live) > 1:
                LOGGER.error("%s state %s is live on several workers; exporting one copy", self.kind, key)
        if stale:
            # Every worker deletes its stale copies at once, a bounded number at a time.
            await self.call_each(
                {worker_index: ("delete_unclaimed", {"identities": keys}) for worker_index, keys in stale.items()},
                timeout=None,
            )
        return kept_records

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        raise NotImplementedError("a coordinated participant restores through install()")

    async def install(self, records: list[CheckpointRecord], scope: list[EpisodeId]) -> None:
        """Install the records on the live workers, all or nothing.

        Each session goes to the live worker with the fewest sessions so far,
        which rewrites the session's routing claim to name itself.
        Every other worker drops any claim it held on a restored session.
        If a worker's install fails, or the worker is lost,
        the workers that installed theirs delete it again and release its claims.
        Only then does this raise.
        Admission reopens after a failed restore only once this has returned, so nothing of it can be used.
        The controller then retires the restored attempts, as it does after a failed restore on one process,
        and that retire finds nothing left to free.
        """
        if self.restored:
            raise ControlError(f"{self.kind} restore requires a server without restored state")
        held: dict[str, CheckpointRecord] = {}
        # The records of each session, which are installed together.
        sessions: dict[str, list[CheckpointRecord]] = defaultdict(list)
        for record in records:
            key = self.claim_key(record)
            if key is not None:
                held[key] = record
                continue
            identity = self._participant.record_identity(record)
            if identity is None:
                raise ControlError(
                    f"{self.kind} record of {record.episode_id.capture_key} names no session, "
                    "so its requests cannot be routed to the worker that would hold it"
                )
            sessions[identity].append(record)
        placement = self._place(sessions)
        installs: dict[int, list[dict[str, Any]]] = {worker_index: [] for worker_index in self.workers}
        for identity, worker_index in placement.items():
            installs[worker_index].extend(record.to_json_record() for record in sessions[identity])
        # Every worker installs, even with nothing to install, so each checks that it holds no live state,
        # and drops its claims on the sessions placed elsewhere.
        scope_json = [episode_id.model_dump(mode="json") for episode_id in scope]
        restored_sessions = sorted(sessions)
        messages = {
            worker_index: ("install", {"records": placed, "sessions": restored_sessions, "scope": scope_json})
            for worker_index, placed in installs.items()
        }
        outcomes = await self.call_each_settled(messages, timeout=None, bulk=True)
        failed = {index: result for index, (_, result) in outcomes.items() if isinstance(result, BaseException)}
        if failed:
            await self._uninstall([index for index in outcomes if index not in failed], scope_json)
            worker_index, error = next(iter(failed.items()))
            if isinstance(error, ControlError):
                raise ControlError(
                    f"{self.kind} worker {worker_index} did not complete install: {error.detail}"
                ) from error
            raise ControlError(f"{self.kind} worker {worker_index} did not complete install: {error}") from error
        self._restored_workers = set(placement.values())
        self._restored_lost.clear()
        self.restored = held

    async def _uninstall(self, workers: list[int], scope: list[dict[str, Any]]) -> None:
        """Delete what a failed restore installed on ``workers``, and release its claims.

        Admission is still closed, so no request has used it.
        A worker that cannot is logged: the controller retires the restored attempts after a failed restore,
        which frees what is left.
        """
        outcomes = await self.call_each_settled(
            {worker_index: ("uninstall", {"scope": scope}) for worker_index in workers}, timeout=None
        )
        for worker_index, (_, result) in outcomes.items():
            if isinstance(result, BaseException):
                LOGGER.error(
                    "%s worker %d could not delete what a failed restore installed: %s",
                    self.kind,
                    worker_index,
                    result,
                )

    def _place(self, sessions: dict[str, list[CheckpointRecord]]) -> dict[str, int]:
        """Assign each session to the live worker with the fewest records, then sessions, so far."""
        if not sessions:
            return {}
        workers = sorted(
            worker_index for worker_index, worker in self.workers.items() if worker.registration.get("routing_id")
        )
        if not workers or len(workers) < len(self.workers):
            raise ControlError(f"restoring {self.kind} sessions requires workers that route sessions to their owner")
        load = {worker_index: [0, 0] for worker_index in workers}
        placement = {}
        for identity in sorted(sessions, key=lambda identity: (-len(sessions[identity]), identity)):
            worker_index = min(workers, key=lambda worker_index: (*load[worker_index], worker_index))
            placement[identity] = worker_index
            load[worker_index][0] += len(sessions[identity])
            load[worker_index][1] += 1
        return placement

    async def refresh_status(self) -> None:
        """Ask every worker for its participant's status; a worker that does not answer in time is left out."""
        outcomes = await self.call_each_settled(
            {worker_index: ("status", {}) for worker_index in self.workers}, timeout=_STATUS_TIMEOUT_SECONDS
        )
        answers = [result for _, result in outcomes.values() if not isinstance(result, BaseException)]
        self._worker_status = [answer["extra"] for answer in answers]
        self._worker_reports = [PrepareReport.model_validate(answer["report"]) for answer in answers]

    def status_report(self) -> PrepareReport:
        if not self.accepting:
            # While a checkpoint is open, the workers' reports are current, and prepare reads the same.
            return self.readiness()
        # Otherwise workers report only when asked, as each status request does.
        merged = self.merge(self._worker_reports)
        return merged.model_copy(update={"counts": {**merged.counts, "workers": len(self.workers)}})

    def status_extra(self) -> dict[str, Any]:
        # The workers' participants hold the state: numbers add up, lists join, and flags hold for every worker.
        merged: dict[str, Any] = {}
        for status in self._worker_status:
            for key, value in status.items():
                merged[key] = value if key not in merged else _merge_status(merged[key], value)
        merged["restored_pending"] = sorted(set(merged.get("restored_pending") or ()) | set(self.restored))
        return {
            **merged,
            **super().status_extra(),
            "status_workers": len(self._worker_status),
        }


def _merge_status(first: Any, second: Any) -> Any:
    if isinstance(first, bool) and isinstance(second, bool):
        return first and second
    if isinstance(first, Number) and isinstance(second, Number):
        return first + second
    if isinstance(first, list) and isinstance(second, list):
        return sorted(set(first) | set(second), key=str)
    return first if first == second else sorted({str(first), str(second)})


class ParticipantWorkerLink(WorkerLink):
    """One worker's participant, driven by the server's coordinator.

    ``app`` is the worker's app, whose session claims a restore rewrites (see ``nemo_gym.session_routing``).
    """

    def __init__(
        self,
        participant: CheckpointParticipant,
        *,
        socket_path: str,
        app: FastAPI,
        on_coordinator_lost: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(socket_path=socket_path, on_coordinator_lost=on_coordinator_lost)
        self.worker_label = participant.kind
        self.participant = participant
        self.app = app
        # The participant and this worker share one set of retired attempts, which the coordinator keeps current.
        participant.retired = self.retired
        participant.on_change = self.changed
        participant.claim_restored = self.claim
        participant.on_session_freed = self._session_freed
        self._accepting = True
        # Claim keys the coordinator holds a restored record for, as of the last reopen.
        self.restored_keys: set[str] = set()
        self._last_report: Optional[dict[str, Any]] = None
        self._poll_task: Optional[asyncio.Task] = None
        # Settlements of claimed records still being sent; each ends once sent or once the connection closes.
        self._settling: set[asyncio.Task] = set()
        # The sessions the last restore installed here, whose claims a failed restore releases again.
        self._installed_sessions: set[str] = set()

    @property
    def accepting(self) -> bool:
        return self._accepting

    @property
    def routing_id(self) -> Optional[str]:
        """This worker's session routing ID, once the server has installed session routing."""
        return getattr(self.app.state, "nemo_gym_routing_id", None)

    def registration(self) -> dict[str, Any]:
        return {**super().registration(), "routing_id": self.routing_id}

    def adopt(self, state: dict[str, Any]) -> None:
        self.restored_keys = set(state["restored_keys"])
        self.participant.carried_restarts = set(state.get("carried_restarts") or ())

    def _claims(self) -> Optional[SessionClaims]:
        """This worker's session claims, with session routing across several workers."""
        return getattr(self.app.state, "nemo_gym_session_claims", None)

    def _lifetimes(self) -> Optional[ResourcesSessionLifetimes]:
        """How long this worker keeps its resources sessions' claims, on a resources server with session routing."""
        return getattr(self.app.state, "nemo_gym_resources_session_lifetimes", None)

    def _session_freed(self, session_id: str) -> None:
        # The participant freed the session outside a request, which the router would otherwise never see end.
        lifetimes = self._lifetimes()
        if lifetimes is not None:
            lifetimes.ended(session_id)

    def _route_restored(self, records: list[CheckpointRecord], restored: list[str]) -> None:
        """Claim the sessions installed here for this worker, and drop any claim on those installed elsewhere."""
        claims = self._claims()
        if claims is None:
            return
        installed = {self.participant.record_identity(record) for record in records} - {None}
        self._installed_sessions = installed
        for session_id in restored:
            if session_id not in installed:
                # A request for it before the restore may have claimed it here; the restore placed it elsewhere.
                claims.disown(session_id)
        lifetimes = self._lifetimes()
        for session_id in sorted(installed):
            claims.rewrite(session_id)
            if lifetimes is not None:
                # Open until its close or verify, as after a successful seed.
                lifetimes.opened(session_id)

    async def _uninstall_local(self, scope: list[dict[str, Any]]) -> None:
        """Delete what a failed restore installed here, and release its claims."""
        for episode_id in scope:
            await self.participant.delete_restored(next_attempt(EpisodeId.model_validate(episode_id)))
        claims, lifetimes = self._claims(), self._lifetimes()
        for session_id in self._installed_sessions:
            if lifetimes is not None:
                lifetimes.forget(session_id)
            if claims is not None:
                claims.release(session_id)
        self._installed_sessions = set()

    async def close_local(self, request: CheckpointRequest) -> None:
        self._accepting = False
        await self.participant.close_admission(request)
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_while_closed())

    async def open_local(self, body: dict[str, Any]) -> None:
        self._accepting = True
        self._last_report = None
        self.restored_keys = set(body["restored_keys"])
        # The restarts belonged to the checkpoint that reopens here.
        self.participant.carried_restarts.clear()
        await self.participant.open_admission()

    def mark_local(self, episode_ids: list[EpisodeId]) -> None:
        super().mark_local(episode_ids)
        # A participant that fences its sessions as their attempts are marked does it here as well,
        # so a retire that never reaches this worker still leaves them refused.
        self.participant.fence_retired(episode_ids)

    async def retire_local(self, episode_id: EpisodeId) -> None:
        await self.participant.retire(episode_id)

    def check_forget_local(self, rollout_ids: list[str]) -> None:
        self.participant.check_forget(rollout_ids)

    async def forget_local(self, rollout_ids: list[str]) -> None:
        await self.participant.forget(rollout_ids)

    def report(self) -> BaseModel:
        return self.participant.readiness()

    async def handle(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        if kind == "export":
            # The coordinator checked the readiness this worker last reported, which may have regressed since.
            if not self.participant.ready():
                raise InvalidPhaseError(f"{self.participant.kind} worker is no longer ready to commit")
            episode_ids = body["episode_ids"]
            records = await self.participant.export(
                None if episode_ids is None else [EpisodeId.model_validate(e) for e in episode_ids],
                checkpoint_id=body["checkpoint_id"],
            )
            unclaimed = self.participant.unclaimed_sessions()
            # Converted and encoded once, off the event loop, into messages of bounded size,
            # which the coordinator checks as they arrive.
            # The exported state does not change until the checkpoint resumes.
            chunks = await asyncio.to_thread(encode_records, records, chunk_bytes=EXPORT_CHUNK_BYTES)
            for chunk in chunks:
                await self.call(
                    "export_chunk", RawJSON(b'{"export":%d,"records":' % body["export"] + chunk + b"}"), timeout=None
                )
            return {"count": len(records), "unclaimed": unclaimed}
        if kind == "delete_restored":
            await self.participant.delete_restored(EpisodeId.model_validate(body["episode_id"]))
            return {}
        if kind == "delete_unclaimed":
            limit = asyncio.Semaphore(_DELETE_CONCURRENCY)

            async def delete(identity: str) -> None:
                async with limit:
                    await self.participant.delete_unclaimed_session(identity)

            await asyncio.gather(*(delete(identity) for identity in body["identities"]))
            return {}
        if kind == "uninstall":
            # A restore failed on another worker: what this worker installed goes, and so do its claims.
            await self._uninstall_local(body["scope"])
            return {}
        if kind == "status":
            return {"extra": self.participant.status_extra(), "report": self.participant.readiness().model_dump()}
        if kind == "install":
            # Read back from storage and parsed from the coordinator's message: their payloads are JSON.
            records = [
                self.participant.record_model.model_validate(record, context=PARSED_JSON_CONTEXT)
                for record in body["records"]
            ]
            scope = [EpisodeId.model_validate(episode_id) for episode_id in body["scope"]]
            await self.participant.install(records, scope)
            try:
                self._route_restored(records, body["sessions"])
            except BaseException:
                await self._uninstall_local(body["scope"])
                raise
            return {}
        if kind == "restored_pending":
            pending = await self.participant.restored_pending()
            return {"episode_ids": [episode_id.model_dump(mode="json") for episode_id in pending]}
        if kind == "carry_restarts":
            await self.participant.carry_restarts(body["restarts"])
            return self._numbered_report()
        return await super().handle(kind, body)

    @asynccontextmanager
    async def claim(self, key: str) -> AsyncIterator[Optional[dict[str, Any]]]:
        """Claim the restored record under ``key`` from the coordinator; None if there is none to claim.

        The coordinator leases the record to this worker.
        Install it inside the block: the lease is settled when the block ends,
        and the record goes back to the coordinator if the block raised.
        """
        if key not in self.restored_keys:
            yield None
            return
        try:
            reply = await self.call("claim", {"key": key})
        except BaseException:
            # The reply may be lost after the coordinator leased the record: give it back.
            self._settle(key, installed=False)
            raise
        record = reply["record"]
        if record is None:
            self.restored_keys.discard(key)
            yield None
            return
        self.restored_keys.discard(key)
        try:
            yield record
        except BaseException:
            # The record can be claimed again, here too.
            self.restored_keys.add(key)
            self._settle(key, installed=False)
            raise
        self._settle(key, installed=True)

    def _settle(self, key: str, *, installed: bool) -> None:
        """Tell the coordinator, in the background, whether the claimed record was installed; retry until it hears."""

        async def send() -> None:
            while self._channel is not None:
                try:
                    await self.call("settle_claim", {"key": key, "installed": installed})
                    return
                except CoordinatorUnavailableError:
                    LOGGER.warning("could not settle the claim of %s; retrying", key, exc_info=True)
                    await asyncio.sleep(_SETTLE_RETRY_SECONDS)

        task = asyncio.create_task(send())
        self._settling.add(task)
        task.add_done_callback(self._settling.discard)

    async def _poll_while_closed(self) -> None:
        # Some changes to readiness do not notify, such as a session ending;
        # a single process polls for them while it waits, so a worker re-reports them.
        while not self._accepting:
            await asyncio.sleep(_REPORT_POLL_SECONDS)
            if not self._accepting and self.report().model_dump(mode="json") != self._last_report:
                await self.changed()

    def reported(self, numbered: dict[str, Any]) -> None:
        # Only a report the coordinator received counts, so a failed one differs from the next poll and is sent again.
        self._last_report = numbered["report"]


def install_server_participant(
    app: FastAPI,
    participant: CheckpointParticipant,
    *,
    num_workers: int,
    auth_token: str,
    instance_name: str,
    lease_grace_seconds: float,
) -> None:
    """Take part in checkpoints with ``participant``, in one process or across ``num_workers`` workers.

    With several workers, the main process, which serves no requests, runs the coordinator,
    and each worker links ``participant`` to it.
    """
    if num_workers == 1:
        install_participant(
            app,
            participant,
            auth_token=auth_token,
            instance_name=instance_name,
            lease_grace_seconds=lease_grace_seconds,
        )
        return
    if not is_nemo_gym_fastapi_worker():
        coordinator = WorkerCoordinator(
            CoordinatedServerParticipant(participant, expected_workers=num_workers),
            instance_name=instance_name,
            lease_grace_seconds=lease_grace_seconds,
            socket_path=coordinator_socket_path(participant.kind),
        )
        coordinator.start_in_background()
        # Held for the life of the process, alongside uvicorn's supervisor.
        app.state.nemo_gym_checkpoint_coordinator = coordinator
        # The workers uvicorn is about to spawn inherit the socket path.
        os.environ[COORDINATOR_SOCKET_ENV] = coordinator.socket_path
        return
    link = ParticipantWorkerLink(participant, socket_path=os.environ[COORDINATOR_SOCKET_ENV], app=app)
    install_control_routes(app, link.dispatch, auth_token=auth_token)

    @asynccontextmanager
    async def joined(application: FastAPI) -> AsyncIterator[None]:
        await link.connect()
        try:
            yield
        finally:
            await link.disconnect()

    # A restore may place sessions on this worker as soon as it joins,
    # so a worker that routes sessions joins only once its private socket accepts the requests forwarded to it.
    # Session routing, installed after this, enters the lifespans in this list at that point.
    routed = getattr(app.state, ROUTED_LIFESPANS_STATE, None)
    if routed is None:
        routed = []
        setattr(app.state, ROUTED_LIFESPANS_STATE, routed)
    routed.append(joined)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_coordinator(application: FastAPI) -> AsyncIterator[Any]:
        if getattr(application.state, "nemo_gym_routing_id", None) is not None:
            async with original_lifespan(application) as state:
                yield state
            return
        # Without a private socket, the worker joins before its own startup, as nothing is forwarded to it.
        async with joined(application):
            async with original_lifespan(application) as state:
                yield state

    app.router.lifespan_context = lifespan_with_coordinator
