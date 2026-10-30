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
    A failed install goes back only once its worker has freed what it installed, so no two workers hold it.
    Until then a claim from any worker is told the record is still settling, and asks again,
    and prepare names the attempt as a blocker (see ``ParticipantWorkerLink.claim``).

  A restore is all or nothing across the workers: if one worker's install fails, or a worker the restore installs
  sessions on is lost before every install has finished, even after it acknowledged its own,
  the workers that installed theirs delete what they installed, and release its claims, before the restore fails.
  A worker that cannot delete it keeps every worker closed until a repeated rollback there succeeds,
  or that worker is lost.
  A failed restore whose install outlived it and lost such a worker afterwards is rolled back the same way
  before admission reopens.
  A worker that holds restored sessions and is lost after the restore and before the resume fails that resume:
  their state is gone, so the restore must be repeated on a restarted server.
  The workers reopen in two phases (see ``CoordinatedParticipant.reopen_workers``),
  so none serves until every worker holding restored sessions is ready.

- **retire** reaches every worker, and the coordinator retires what it holds for the attempt.
  A resources session a retire frees keeps its routing claim until the forget that ends its refusal,
  so a late request that carries only its cookie still reaches the worker that refuses it.

Claims are refused while a checkpoint is open, as a new episode is: the caller retries the episode later.
A claim is answered before any later message to its worker, so a worker that wins a claim starts the
episode before it can see the next checkpoint close, and that checkpoint waits for the episode instead of
missing it.
"""

import asyncio
import contextlib
import logging
import os
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from functools import partial
from numbers import Number
from typing import Any, NamedTuple, Optional
from uuid import uuid4

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
from nemo_gym.core.episode import EpisodeId
from nemo_gym.session_routing import ROUTED_LIFESPANS_STATE, ResourcesSessionLifetimes, SessionClaims
from nemo_gym.utils.server_process import is_nemo_gym_fastapi_worker


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
# One that could not tear down a record's rolled-back install retries it as well, waiting longer each time.
_SETTLE_RETRY_SECONDS = 1.0
_TEARDOWN_RETRY_MAX_SECONDS = 30.0
# Each attempt to tear down a rolled-back install is cancelled after this long, and tried again.
# A cancelled attempt that has not stopped this much later is no longer waited for.
_TEARDOWN_ATTEMPT_SECONDS = 60.0
_TEARDOWN_CANCEL_GRACE_SECONDS = 10.0
# Once the coordinator no longer holds the record's lease, for example after a retire,
# a teardown is tried at most this many more times and then given up.
_TEARDOWN_ATTEMPTS_WITHOUT_LEASE = 3
# A worker that stops cleanly waits this long for teardowns still running before it cancels them.
_TEARDOWN_GRACE_SECONDS = 30.0
# A claim of a record another claim still holds is sent again after this long, doubling up to the maximum.
_CLAIM_RETRY_SECONDS = 0.05
_CLAIM_RETRY_MAX_SECONDS = 1.0
# A rollback of a failed restore that did not complete on a worker is repeated after this long,
# doubling up to the maximum, for at most the budget each time a reopen or a restore needs it.
_ROLLBACK_RETRY_SECONDS = 0.1
_ROLLBACK_RETRY_MAX_SECONDS = 2.0
_ROLLBACK_RETRY_BUDGET_SECONDS = 30.0


class ClaimSettlingError(ControlError):
    """A restored record is leased to a claim that failed, and is on its way back: claim it again shortly."""

    status_code = 503
    code = "checkpoint_restored_claim_settling"


class RollbackIncompleteError(ControlError):
    """A worker could not yet delete what a failed restore installed on it, so admission stays closed: resume again."""

    status_code = 503
    code = "checkpoint_rollback_incomplete"


class Lease(NamedTuple):
    """A restored record claimed by a worker, until that worker settles the claim that ``lease_id`` names."""

    worker_index: int
    lease_id: str
    record: CheckpointRecord


class CoordinatedServerParticipant(CoordinatedParticipant[PrepareReport]):
    """The participant of an environment, agent, or resources server with several workers."""

    report_model = PrepareReport

    def __init__(self, participant: CheckpointParticipant, *, expected_workers: int) -> None:
        super().__init__(expected_workers=expected_workers)
        # The coordinator stands in for the workers' participants, so it shares their kind, mode and records.
        self.kind = participant.kind
        self.mode = participant.mode
        self.record_model = participant.record_model
        self.worker_label = participant.kind
        self.claim_key = participant.claim_key
        # The workers' participant kind, which also shapes the commit reply the controller reads.
        self._participant = participant
        # Restored records a replacement has not claimed yet, by claim key.
        self.restored: dict[str, CheckpointRecord] = {}
        # Claimed records the claiming worker has not settled yet, by claim key.
        # A lease ends when the worker settles it, when the worker is lost, or when a retire discards its attempt.
        # A claim of a leased record is refused as still settling, since a failed claim may be returning it.
        self.leases: dict[str, Lease] = {}
        # The records each export receives from each worker, while that export runs.
        self._exports: dict[int, dict[int, list[CheckpointRecord]]] = {}
        self._next_export = 0
        # Each worker's own status and readiness as of the last status request, which replaces them.
        self._worker_status: list[dict[str, Any]] = []
        self._worker_reports: list[PrepareReport] = []
        # The workers the last restore installs sessions on, from before it sends them until the resume after it,
        # those of them lost meanwhile, whose restored sessions are gone, and the restore's scope.
        # All three end with the reopen that follows, or with a failed restore that is rolled back.
        self._restored_workers: set[int] = set()
        self._restored_lost: set[int] = set()
        self._restored_scope: list[dict[str, Any]] = []
        # The workers that may still hold what a failed restore installed, because deleting it failed there,
        # each with that restore's scope.
        # Admission stays closed until each one's rollback is repeated successfully, or the worker is lost,
        # which takes what it held with it.
        # The control plane finishes this before another restore installs (see ``discard_failed_restore``).
        self._unremoved: dict[int, list[dict[str, Any]]] = {}

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
            "restored_keys": self._claimable_keys(),
            # A worker that joins while a checkpoint is open leaves out the restarts the others leave out.
            "carried_restarts": sorted(self.carried_restarts),
        }

    def open_state(self) -> dict[str, Any]:
        return {"restored_keys": self._claimable_keys()}

    def _claimable_keys(self) -> list[str]:
        # A leased record may still go back, for example once a stuck teardown succeeds, so its key stays claimable.
        return sorted(set(self.restored) | set(self.leases))

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
        for key, lease in list(self.leases.items()):
            if discarded(lease.record):
                del self.leases[key]

    async def leave(self, worker_index: int) -> None:
        # What a lost worker claimed and had not settled may not have been installed anywhere: it can be claimed again.
        for key, lease in list(self.leases.items()):
            if lease.worker_index == worker_index:
                del self.leases[key]
                self.restored[key] = lease.record
        if worker_index in self._restored_workers and not self.accepting:
            self._restored_lost.add(worker_index)
        self._unremoved.pop(worker_index, None)
        await super().leave(worker_index)

    async def open_admission(self) -> None:
        """Reopen every worker, unless a worker holding sessions the last restore installed was lost before it.

        Those sessions' state is gone, and their requests would get a 410.
        The resume fails instead, and keeps failing, with every worker still closed.
        The controller then repeats the restore on a restarted server, rather than continue those episodes without it.
        A holder that is not ready in the first phase of the reopen counts as lost.
        One lost in the second phase, once every worker was ready, is lost as after the resume.
        A failed restore whose rollback did not complete on a worker keeps every worker closed too,
        until a repeated rollback there succeeds or that worker is lost (see ``_finish_rollback``).
        """
        await self._finish_rollback()
        await self.reopen_workers()
        self._end_restore()

    def check_reopen(self, unready: set[int]) -> None:
        lost = self._restored_lost | (unready & self._restored_workers)
        if lost:
            self._restored_lost |= lost
            raise WorkerLostError(
                f"{self.kind} workers {sorted(lost)} were lost after a restore installed sessions on them "
                "and before the resume, so the restored state of those sessions is gone; "
                "restart the server and restore again"
            )

    def _end_restore(self) -> None:
        self._restored_workers = set()
        self._restored_lost = set()
        self._restored_scope = []

    async def discard_failed_restore(self) -> None:
        """Roll back a failed restore whose install finished, if a worker it installed sessions on was lost since.

        Those sessions are gone, so the restore cannot be served in part:
        every worker left deletes what it installed and releases its claims, and the coordinator drops its records.
        Without such a loss,
        what the install brought stays until the controller retires the restored attempts, as on one process.
        A rollback that does not complete on a worker, now or when the restore failed, is repeated there,
        and raises ``RollbackIncompleteError`` if it is still incomplete (see ``_finish_rollback``).
        """
        if self._restored_lost:
            try:
                await self._uninstall({worker_index: self._restored_scope for worker_index in self.workers})
            finally:
                # Admission is closed, so nothing claimed a record.
                self.restored = {}
                self._end_restore()
        await self._finish_rollback()

    def readiness(self) -> PrepareReport:
        report = super().readiness()
        if not self.leases:
            return report
        # A claimed record is being installed on its worker: wait until it is settled one way or the other.
        unsettled = sorted(next_attempt(lease.record.episode_id).capture_key for lease in self.leases.values())
        return report.model_copy(update={"ready": False, "blockers": unsettled + list(report.blockers)})

    async def unclaimed(self) -> list[EpisodeId]:
        """Restored records no worker has claimed, and restored sessions no worker has used yet."""
        unclaimed = {next_attempt(record.episode_id) for record in self.restored.values()}
        replies = await self.broadcast("unclaimed", {}, timeout=MESSAGE_TIMEOUT_SECONDS)
        for reply in replies.values():
            unclaimed.update(EpisodeId.model_validate(episode_id) for episode_id in reply["episode_ids"])
        return sorted(unclaimed, key=lambda episode_id: episode_id.capture_key)

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
        lease = self.leases.get(key)
        if lease is not None:
            raise ClaimSettlingError(
                f"the restored record of {key} is still claimed by {self.worker_label} worker {lease.worker_index}, "
                "which may be returning it; claim it again shortly"
            )
        record = self.restored.pop(key, None)
        if record is None:
            return {"record": None}
        self.leases[key] = Lease(worker_index, body["lease"], record)
        return {"record": record.to_json_record()}

    async def handle(self, worker_index: int, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        if kind == "settle_claim":
            lease = self.leases.get(body["key"])
            # Only the claim that holds the lease settles it.
            # A claim whose reply was lost settles a lease it never got, which changes nothing.
            if self._holds(lease, worker_index, body["lease"]):
                del self.leases[body["key"]]
                if not body["installed"]:
                    self.restored[body["key"]] = lease.record
                await self.notify()
            return {}
        if kind == "lease_held":
            return {"held": self._holds(self.leases.get(body["key"]), worker_index, body["lease"])}
        if kind == "export_chunk":
            received = self._exports.get(body["export"])
            if received is not None:
                # Checked as it arrives, off the event loop, while other workers are still exporting.
                records = await asyncio.to_thread(self._validate_records, body["records"])
                received.setdefault(worker_index, []).extend(records)
            return {}
        return await super().handle(worker_index, kind, body)

    @staticmethod
    def _holds(lease: Optional[Lease], worker_index: int, lease_id: str) -> bool:
        return lease is not None and (lease.worker_index, lease.lease_id) == (worker_index, lease_id)

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

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        self._next_export += 1
        export = self._next_export
        body = {
            "episode_ids": None if episode_ids is None else [e.model_dump(mode="json") for e in episode_ids],
            "export": export,
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
        unclaimed = [
            record.model_copy(update={"episode_id": next_attempt(record.episode_id)})
            for record in self.restored.values()
            if scope is None or next_attempt(record.episode_id) in scope
        ]
        return records + unclaimed

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
        # Recorded before any install is sent, and with no loss carried over from an earlier restore.
        # A holder lost from now on, even one that already acknowledged its install, fails the restore or the resume.
        self._restored_workers = set(placement.values())
        self._restored_lost = set()
        installs: dict[int, list[dict[str, Any]]] = {worker_index: [] for worker_index in self.workers}
        for identity, worker_index in placement.items():
            installs[worker_index].extend(record.to_json_record() for record in sessions[identity])
        # Every worker installs, even with nothing to install, so each checks that it holds no live state,
        # and drops its claims on the sessions placed elsewhere.
        scope_json = [episode_id.model_dump(mode="json") for episode_id in scope]
        self._restored_scope = scope_json
        restored_sessions = sorted(sessions)
        messages = {
            worker_index: ("install", {"records": placed, "sessions": restored_sessions, "scope": scope_json})
            for worker_index, placed in installs.items()
        }
        outcomes = await self.call_each_settled(messages, timeout=None, bulk=True)
        failed = {index: result for index, (_, result) in outcomes.items() if isinstance(result, BaseException)}
        # A holder lost after it acknowledged its install, while others still installed, took its sessions with it.
        lost = set(self._restored_lost)
        if failed or lost:
            await self._uninstall({index: scope_json for index in outcomes if index not in failed})
            # Nothing of this restore is left to lose.
            self._end_restore()
            if not failed:
                raise WorkerLostError(
                    f"{self.kind} workers {sorted(lost)} were lost while the restore installed sessions on them"
                )
            worker_index, error = next(iter(failed.items()))
            if isinstance(error, ControlError):
                raise ControlError(
                    f"{self.kind} worker {worker_index} did not complete install: {error.detail}"
                ) from error
            raise ControlError(f"{self.kind} worker {worker_index} did not complete install: {error}") from error
        self.restored = held

    async def _uninstall(self, scopes: dict[int, list[dict[str, Any]]]) -> None:
        """Delete what a failed restore with each scope installed on each worker, and release its claims.

        Admission is still closed, so no request has used it.
        Each connected worker counts as still holding it until it answers that it has deleted it,
        so a worker that fails, and every worker if this is cancelled, is left in ``_unremoved``.
        """
        connected = {worker_index: scope for worker_index, scope in scopes.items() if worker_index in self.workers}
        self._unremoved.update(connected)
        outcomes = await self.call_each_settled(
            {worker_index: ("uninstall", {"scope": scope}) for worker_index, scope in connected.items()}, timeout=None
        )
        for worker_index, (_, result) in outcomes.items():
            if not isinstance(result, BaseException):
                self._unremoved.pop(worker_index, None)
            elif worker_index in self.workers:
                LOGGER.error(
                    "%s worker %d could not delete what a failed restore installed, "
                    "so admission stays closed until it does: %s",
                    self.kind,
                    worker_index,
                    result,
                )

    async def _finish_rollback(self) -> None:
        """Repeat each rollback of a failed restore that did not complete, until each completes or its worker is lost.

        Each is retried with a growing wait, for at most ``_ROLLBACK_RETRY_BUDGET_SECONDS`` or the caller's deadline.
        If one is still incomplete then, this raises ``RollbackIncompleteError``,
        and the next reopen or restore tries again.
        """
        wait = _ROLLBACK_RETRY_SECONDS
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(_ROLLBACK_RETRY_BUDGET_SECONDS):
                while self._unremoved_connected():
                    await self._uninstall(dict(self._unremoved))
                    if not self._unremoved_connected():
                        break
                    await asyncio.sleep(wait)
                    wait = min(2 * wait, _ROLLBACK_RETRY_MAX_SECONDS)
        unremoved = self._unremoved_connected()
        if unremoved:
            raise RollbackIncompleteError(
                f"{self.kind} workers {unremoved} could not delete what a failed restore installed on them, "
                "so admission stays closed; resume again"
            )

    def _unremoved_connected(self) -> list[int]:
        # A worker that left took what it held with it.
        for worker_index in [index for index in self._unremoved if index not in self.workers]:
            del self._unremoved[worker_index]
        return sorted(self._unremoved)

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
        merged["unclaimed"] = sorted(set(merged.get("unclaimed") or ()) | set(self.restored))
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
        # Settlements of claimed records still being sent; each ends once sent, or once this worker disconnects.
        self._settling: set[asyncio.Task] = set()
        # The settlements among them still freeing a rolled-back install, by claim key, until each has freed it.
        self._tearing_down: dict[str, asyncio.Task] = {}
        # The teardown calls given up while still running, as they ignored their cancellation, by claim key.
        # Each keeps its key from being claimed here until it ends, since it could free what that claim opens.
        # One that never ends keeps its key unavailable on this worker for the worker's life.
        self._blocked: dict[str, asyncio.Task] = {}
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

    def _session_freed(self, session_id: str, *, until_forget: bool = False) -> None:
        # The participant freed the session outside a request, which the router would otherwise never see end.
        # A retired session keeps its claim until the forget, so its requests keep reaching this worker's refusal.
        lifetimes = self._lifetimes()
        if lifetimes is None:
            return
        if until_forget:
            lifetimes.retain(session_id)
        else:
            lifetimes.ended(session_id)

    async def _route_restored(self, records: list[CheckpointRecord], restored: list[str]) -> None:
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
        if lifetimes is not None:
            for session_id in installed:
                # Open until its close or verify, as after a successful seed.
                # Held before its claim is rewritten, so no release here can remove the rewritten claim.
                lifetimes.opened(session_id)
        # Off the event loop, in one batch, which a restore of many sessions needs.
        await claims.rewrite_all(sorted(installed))

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
        # A participant that marks its sessions as their attempts are marked does it here as well,
        # so a retire that never reaches this worker still leaves them refused.
        self.participant.mark_retired_sessions(episode_ids)

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
                None if episode_ids is None else [EpisodeId.model_validate(e) for e in episode_ids]
            )
            unclaimed = self.participant.unclaimed_sessions()
            await self._send_export(records, body["export"])
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
                await self._route_restored(records, body["sessions"])
            except BaseException:
                await self._uninstall_local(body["scope"])
                raise
            return {}
        if kind == "unclaimed":
            unclaimed = await self.participant.unclaimed()
            return {"episode_ids": [episode_id.model_dump(mode="json") for episode_id in unclaimed]}
        if kind == "carry_restarts":
            await self.participant.carry_restarts(body["restarts"])
            return self._numbered_report()
        return await super().handle(kind, body)

    async def _send_export(self, records: list[CheckpointRecord], export: int) -> None:
        """Send the records in messages of bounded size, each encoded off the event loop while the one before is sent.

        The coordinator checks each message as it arrives, before it answers.
        So encoding, sending and checking overlap, and this worker holds at most two encoded messages at once.
        The exported state does not change until the checkpoint resumes.
        """
        chunks = encode_records(records, chunk_bytes=EXPORT_CHUNK_BYTES)
        next_chunk = asyncio.ensure_future(asyncio.to_thread(next, chunks, None))
        try:
            while (chunk := await next_chunk) is not None:
                next_chunk = asyncio.ensure_future(asyncio.to_thread(next, chunks, None))
                await self.call(
                    "export_chunk", RawJSON(b'{"export":%d,"records":' % export + chunk + b"}"), timeout=None
                )
        finally:
            # A failed or cancelled send leaves the next message's encoding to finish in its thread, unused.
            next_chunk.cancel()

    @asynccontextmanager
    async def claim(
        self, key: str, *, teardown: Optional[Callable[[], Awaitable[None]]] = None
    ) -> AsyncIterator[Optional[dict[str, Any]]]:
        """Claim the restored record under ``key`` from the coordinator; None if there is none to claim.

        The coordinator leases the record to this worker.
        Install it inside the block: the lease is settled when the block ends.
        If the block raised, the record goes back to the coordinator, to be claimed again,
        but only once ``teardown``, which frees whatever the block left installed outside the participant, succeeds.
        Until then the lease holds the record, so no other worker installs a second copy,
        and prepare names the attempt as a blocker.
        The teardown is retried in the background (see ``_tear_down``),
        each attempt cancelled after ``_TEARDOWN_ATTEMPT_SECONDS``, until one succeeds or this worker disconnects.
        A teardown that never succeeds keeps the lease until the controller retires the attempt or this worker leaves.
        Once the lease is gone,
        the teardown gets ``_TEARDOWN_ATTEMPTS_WITHOUT_LEASE`` more attempts, and is then given up.
        A record another claim holds may be on its way back, so the claim is sent again, with a growing wait,
        for as long as the caller waits, and at most ``MESSAGE_TIMEOUT_SECONDS``.
        A rolled-back install of the same key still being freed here is waited for as long,
        whether or not a record is left to claim, so its teardown cannot free what this claim's caller opens next;
        past that, the claim raises ``ClaimSettlingError``, which its caller can retry.
        That includes a teardown given up while its call still runs: the key stays blocked until that call ends.
        """
        await self._wait_torn_down(key)
        if key not in self.restored_keys:
            yield None
            return
        lease_id, reply = await self._claim_lease(key)
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
            # Names the attempt the record continues, in a teardown that is given up.
            episode = next_attempt(EpisodeId.model_validate(record["episode_id"])).capture_key if teardown else key
            self._settle(key, lease_id, installed=False, teardown=teardown, episode=episode)
            raise
        self._settle(key, lease_id, installed=True)

    async def _wait_torn_down(self, key: str) -> None:
        loop = asyncio.get_running_loop()
        give_up_at = loop.time() + MESSAGE_TIMEOUT_SECONDS
        # A teardown given up while its call still runs blocks the key from then on, so both are waited for in turn.
        while pending := [task for task in (self._tearing_down.get(key), self._blocked.get(key)) if task is not None]:
            remaining = give_up_at - loop.time()
            if remaining <= 0:
                raise ClaimSettlingError(
                    f"this {self.worker_label} worker is still freeing a rolled-back install of {key}; try again later"
                )
            await asyncio.wait(pending, timeout=remaining)

    async def _claim_lease(self, key: str) -> tuple[str, dict[str, Any]]:
        """Ask the coordinator for the record under ``key``, again while another claim still holds it."""
        loop = asyncio.get_running_loop()
        give_up_at = loop.time() + MESSAGE_TIMEOUT_SECONDS
        wait = _CLAIM_RETRY_SECONDS
        while True:
            # Names this attempt's lease, so a settlement of an attempt whose reply was lost cannot end another one.
            lease_id = uuid4().hex
            try:
                return lease_id, await self.call("claim", {"key": key, "lease": lease_id})
            except ControlError as error:
                if error.code == ClaimSettlingError.code and loop.time() + wait <= give_up_at:
                    await asyncio.sleep(wait)
                    wait = min(2 * wait, _CLAIM_RETRY_MAX_SECONDS)
                    continue
                # The reply may be lost after the coordinator leased the record: give it back.
                # The coordinator ignores this for a lease it never granted.
                self._settle(key, lease_id, installed=False)
                raise
            except BaseException:
                self._settle(key, lease_id, installed=False)
                raise

    def _settle(
        self,
        key: str,
        lease_id: str,
        *,
        installed: bool,
        teardown: Optional[Callable[[], Awaitable[None]]] = None,
        episode: str = "",
    ) -> None:
        """Tell the coordinator, in the background, whether the claimed record was installed; retry until it hears.

        A record that was not installed goes back only after ``teardown`` has succeeded (see ``_tear_down``).
        """

        async def send() -> None:
            if teardown is not None:
                freed = await self._tear_down(key, lease_id, teardown, episode)
                # Freed or given up: a later claim of the key no longer waits for this settlement.
                self._torn_down(key, task)
                if not freed:
                    # The lease is gone, so there is nothing left to settle.
                    return
            while self._channel is not None:
                try:
                    await self.call("settle_claim", {"key": key, "lease": lease_id, "installed": installed})
                    return
                except CoordinatorUnavailableError:
                    LOGGER.warning("could not settle the claim of %s; retrying", key, exc_info=True)
                    await asyncio.sleep(_SETTLE_RETRY_SECONDS)

        task = asyncio.create_task(send())
        self._settling.add(task)
        task.add_done_callback(self._settling.discard)
        if teardown is not None:
            self._tearing_down[key] = task
            task.add_done_callback(partial(self._torn_down, key))

    def _torn_down(self, key: str, task: asyncio.Task) -> None:
        if self._tearing_down.get(key) is task:
            del self._tearing_down[key]

    async def _tear_down(self, key: str, lease_id: str, teardown: Callable[[], Awaitable[None]], episode: str) -> bool:
        """Free what a rolled-back install of ``key`` left outside the participant; return whether it did.

        Each attempt is cancelled after ``_TEARDOWN_ATTEMPT_SECONDS`` and tried again after a growing wait,
        as is one that fails or raises ``CancelledError`` on its own, so a teardown that hangs is tried again.
        An attempt that ignores its cancellation for ``_TEARDOWN_CANCEL_GRACE_SECONDS`` is not started again
        while it runs: the next try waits for it once more.
        While the coordinator holds the record's lease, the teardown is tried until it succeeds,
        so a stuck teardown never lets a second copy be installed.
        Once the lease is gone, for example after a retire, it gets ``_TEARDOWN_ATTEMPTS_WITHOUT_LEASE`` more tries.
        It is then given up and logged, since what the install left may be orphaned.
        Only cancelling this task, as a disconnect does, stops it otherwise.
        A call that ignored its cancellation may still run when this ends.
        It then blocks the key until it ends (see ``_block``).
        """
        wait = _SETTLE_RETRY_SECONDS
        attempt: Optional[asyncio.Task] = None
        tries_left: Optional[int] = None
        try:
            while True:
                if attempt is None:
                    attempt = asyncio.create_task(teardown())
                    # Its outcome is read below, or never, if the attempt is abandoned.
                    attempt.add_done_callback(lambda done: done.cancelled() or done.exception())
                if await self._attempt_ended(key, attempt):
                    if not attempt.cancelled() and attempt.exception() is None:
                        return True
                    LOGGER.warning(
                        "could not free the rolled-back install of %s; retrying",
                        key,
                        exc_info=None if attempt.cancelled() else attempt.exception(),
                    )
                    attempt = None
                if tries_left is None and not await self._lease_held(key, lease_id):
                    tries_left = _TEARDOWN_ATTEMPTS_WITHOUT_LEASE
                if tries_left is not None:
                    if tries_left == 0:
                        LOGGER.error(
                            "gave up freeing the rolled-back install of %s for %s after its lease ended; "
                            "what the install left outside the participant may be orphaned",
                            key,
                            episode,
                        )
                        if attempt is not None and not attempt.done():
                            LOGGER.error(
                                "the teardown of the rolled-back install of %s still runs after it was given up; "
                                "%s cannot be claimed on this worker until that call ends",
                                key,
                                key,
                            )
                            self._block(key, attempt)
                        return False
                    tries_left -= 1
                await asyncio.sleep(wait)
                wait = min(2 * wait, _TEARDOWN_RETRY_MAX_SECONDS)
        except asyncio.CancelledError:
            if attempt is not None:
                attempt.cancel()
                # Cancelled, it normally ends at once; one that ignores it keeps its key blocked.
                self._block(key, attempt)
            raise

    def _block(self, key: str, attempt: asyncio.Task) -> None:
        """Keep ``key`` from being claimed here until ``attempt``, a teardown call no longer waited for, ends.

        Teardowns act on the key alone, so the call could otherwise free what a later claim of the key opens here.
        A call that never ends keeps the key unavailable on this worker until the worker stops.
        """
        if attempt.done():
            return
        self._blocked[key] = attempt
        attempt.add_done_callback(partial(self._unblock, key))

    def _unblock(self, key: str, attempt: asyncio.Task) -> None:
        if self._blocked.get(key) is attempt:
            del self._blocked[key]

    async def _attempt_ended(self, key: str, attempt: asyncio.Task) -> bool:
        """Wait for a teardown attempt, cancelling it at its timeout; False if it ignores the cancellation."""
        done, _ = await asyncio.wait([attempt], timeout=_TEARDOWN_ATTEMPT_SECONDS)
        if done:
            return True
        attempt.cancel()
        done, _ = await asyncio.wait([attempt], timeout=_TEARDOWN_CANCEL_GRACE_SECONDS)
        if not done:
            LOGGER.error("the teardown of the rolled-back install of %s does not stop when cancelled", key)
        return bool(done)

    async def _lease_held(self, key: str, lease_id: str) -> bool:
        """Whether the coordinator still holds the record's lease for this claim; assumed so if it cannot answer."""
        try:
            reply = await self.call("lease_held", {"key": key, "lease": lease_id})
        except ControlError:
            return True
        return bool(reply["held"])

    async def disconnect(self) -> None:
        """Leave the coordinator, once the teardowns still running have finished or ``_TEARDOWN_GRACE_SECONDS`` passed.

        A teardown frees what a rolled-back install left outside the participant, such as a sandbox.
        One that finishes within the grace settles its claim before this worker leaves, so nothing is left behind.
        One still running then is cancelled, and the coordinator takes the record back as this worker leaves:
        the checkpoint stays consistent, but what the install left outside the participant may outlive it,
        alongside the copy the record's next claim installs elsewhere.
        """
        self._disconnecting = True
        tearing_down = list(self._tearing_down.values())
        if tearing_down:
            await asyncio.wait(tearing_down, timeout=_TEARDOWN_GRACE_SECONDS)
        await super().disconnect()

    def background_tasks(self) -> list[asyncio.Task]:
        polling = [self._poll_task] if self._poll_task is not None else []
        return [*super().background_tasks(), *polling, *self._settling]

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
