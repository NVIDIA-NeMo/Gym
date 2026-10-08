# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment, agent, and resources servers with several uvicorn workers.

Each worker runs the same participant a single-process server runs, for the episodes and sessions it
holds, linked to one coordinator in the main process (see ``nemo_gym._checkpoint.workers``). The
coordinator is the server's participant:

- **prepare** closes every worker and is ready once every worker reports ready.
- **commit** asks every worker to export its records and writes the participant's one records file and
  manifest. Each session record carries its routing owner: the worker ID in the session's cookie and MCP
  token (see ``nemo_gym.session_routing``).
- **restore** places each record by where its state lives:

  - State that spans requests (resources sessions, native agent sessions) is installed on a live worker.
    All the sessions of one pre-crash owner go to the same worker, owners are spread evenly, and every
    worker's router gets the alias table from old owner to new worker, so a request whose cookie or token
    names a pre-crash worker reaches the one that holds its session now. The manifest lists every owner
    a cookie may name at commit, so owners whose sessions exported nothing, such as those of a stateless
    resources server, are aliased too. Sessions checkpointed by a single-process server name no owner; each
    is placed on a live worker on its own, spread evenly, and every router gets the placement table
    from session ID to worker for them. A worker that joins later gets both tables when it registers.
  - State that lives inside one request (environment episodes, legacy agent ``/run`` episodes) stays with
    the coordinator. The worker that receives the replacement attempt's ``/run`` claims it; exactly one
    claim succeeds. A record nobody has claimed is exported again by the next checkpoint.

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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym._checkpoint.control import (
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
    MESSAGE_TIMEOUT_SECONDS,
    CoordinatedParticipant,
    WorkerCoordinator,
    WorkerLink,
    coordinator_socket_path,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import is_nemo_gym_fastapi_worker
from nemo_gym.session_routing import session_aliases, session_placements


LOGGER = logging.getLogger(__name__)

# Set by the main process for the workers it spawns.
COORDINATOR_SOCKET_ENV = "NEMO_GYM_CHECKPOINT_COORDINATOR_SOCKET"
# While a checkpoint is open, a worker re-reports this often if its readiness changed without a notification.
_REPORT_POLL_SECONDS = 0.5


class CoordinatedServerParticipant(CoordinatedParticipant[PrepareReport]):
    """The participant of an environment, agent, or resources server with several workers."""

    report_model = PrepareReport

    def __init__(
        self,
        participant: CheckpointParticipant,
        *,
        expected_workers: int,
        status: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(expected_workers=expected_workers)
        # The coordinator stands in for the workers' participants, so it shares their kind and records.
        self.kind = participant.kind
        self.record_model = participant.record_model
        self.worker_label = participant.kind
        self.claim_key = participant.claim_key
        self._status = status or {}
        # Restored records a replacement has not claimed yet, by claim key.
        self.restored: dict[str, CheckpointRecord] = {}
        # Pre-crash session owner -> the live worker that holds its sessions now.
        self.aliases: dict[str, str] = {}
        # Restored session ID -> the live worker that holds it, for sessions whose cookie may name no owner.
        # At most one entry per restored session; entries outlive their sessions until the next restore.
        self.placements: dict[str, str] = {}
        # From the manifest of the checkpoint being restored: every owner a cookie may name,
        # and the sessions that were placed without an owner and whose cookies may still name none.
        self._restoring_owners: list[str] = []
        self._restoring_placed: set[str] = set()
        # The owners named by the records of the last export, for its manifest.
        self._exported_owners: set[str] = set()

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
        return {"restored_keys": sorted(self.restored), "aliases": self.aliases, "placements": self.placements}

    def open_state(self) -> dict[str, Any]:
        # The routing tables too: a worker that joined while a restore was installing got them before they changed.
        return {"restored_keys": sorted(self.restored), "aliases": self.aliases, "placements": self.placements}

    def retire_restored(self, episode_id: EpisodeId) -> None:
        for key, record in list(self.restored.items()):
            replacement = next_attempt(record.episode_id)
            if replacement.rollout_id == episode_id.rollout_id and replacement.attempt <= episode_id.attempt:
                del self.restored[key]

    async def restored_pending(self) -> list[EpisodeId]:
        """Restored records no worker has claimed, and restored sessions no worker has used yet."""
        pending = {next_attempt(record.episode_id) for record in self.restored.values()}
        replies = await self.broadcast("restored_pending", {}, timeout=MESSAGE_TIMEOUT_SECONDS)
        for reply in replies.values():
            pending.update(EpisodeId.model_validate(episode_id) for episode_id in reply["episode_ids"])
        return sorted(pending, key=lambda episode_id: episode_id.capture_key)

    def handle_now(self, worker_index: int, kind: str, body: dict[str, Any]) -> Optional[dict[str, Any]]:
        if kind != "claim":
            return None
        if not self.accepting:
            raise AdmissionClosedError(f"{self.kind} admission is closed for a checkpoint")
        record = self.restored.pop(body["key"], None)
        return {"record": record.to_json_record() if record is not None else None}

    # -- commit and restore -------------------------------------------------------------------------

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        raise NotImplementedError("a coordinated participant exports its workers' records through export()")

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        body = {"episode_ids": None if episode_ids is None else [e.model_dump(mode="json") for e in episode_ids]}
        # The controller bounds the export by the commit's deadline.
        replies = await self.broadcast("export", body, timeout=None)
        records = await self._one_copy_per_session(replies)
        # An unclaimed restored episode still continues from its restored state while the commit's scope names it.
        scope = None if episode_ids is None else set(episode_ids)
        pending = [
            record.model_copy(update={"episode_id": next_attempt(record.episode_id)})
            for record in self.restored.values()
            if scope is None or next_attempt(record.episode_id) in scope
        ]
        exported = records + pending
        self._exported_owners = {owner for record in exported if (owner := getattr(record, "owner", None))}
        return exported

    async def _one_copy_per_session(self, replies: dict[int, dict[str, Any]]) -> list[CheckpointRecord]:
        """The workers' records, with one record per session.

        A restored session can be seeded again on another worker,
        for example when an episode continues from before its seed and the seed is routed elsewhere.
        That worker's copy is live; the restored copy nothing used is stale, so it is left out and freed.
        """
        copies: dict[Any, list[tuple[int, CheckpointRecord, bool]]] = defaultdict(list)
        for worker_index, reply in replies.items():
            unclaimed = set(reply.get("unclaimed") or ())
            for raw in reply["records"]:
                record = self.record_model.model_validate(raw)
                session_id = getattr(record, "session_id", None)
                key = session_id if session_id is not None else (worker_index, len(copies))
                copies[key].append((worker_index, record, session_id in unclaimed))
        records: list[CheckpointRecord] = []
        stale: list[tuple[int, str]] = []
        for key, found in copies.items():
            live = [copy for copy in found if not copy[2]]
            kept = (live or found)[0][1]
            records.append(kept)
            stale.extend(
                (worker_index, record.session_id)
                for worker_index, record, unclaimed_copy in found
                if unclaimed_copy and record is not kept
            )
            if len(live) > 1:
                LOGGER.error("%s session %s is live on several workers; exporting one copy", self.kind, key)
        for worker_index, session_id in stale:
            await self.call_each(
                {worker_index: ("delete_unclaimed", {"session_id": session_id})}, timeout=MESSAGE_TIMEOUT_SECONDS
            )
        return records

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        raise NotImplementedError("a coordinated participant restores through install()")

    def manifest_extra(self) -> dict[str, Any]:
        # The owners the checkpoint's sessions name, so a restore aliases them.
        # Every session whose cookie may outlive a restore exports a record, a stateless one without state,
        # so owners no continued session names are dropped: the list never outgrows the sessions,
        # however many restores a run goes through.
        return {"session_owners": sorted(self._exported_owners), "placed_sessions": sorted(self.placements)}

    def restore_manifest(self, manifest: dict[str, Any]) -> None:
        self._restoring_owners = list(manifest.get("session_owners") or [])
        self._restoring_placed = set(manifest.get("placed_sessions") or [])

    async def install(self, records: list[CheckpointRecord], scope: list[EpisodeId]) -> None:
        if self.restored:
            raise ControlError(f"{self.kind} restore requires a server without restored state")
        held: dict[str, CheckpointRecord] = {}
        # Records placed together: all the sessions of one owner, or one session that names no owner.
        groups: dict[tuple[str, str], list[CheckpointRecord]] = defaultdict(list)
        for record in records:
            key = self.claim_key(record)
            if key is not None:
                held[key] = record
                continue
            owner, session_id = getattr(record, "owner", None), getattr(record, "session_id", None)
            if owner is not None:
                groups["owner", owner].append(record)
            elif session_id is not None:
                # Checkpointed without routing, by a single-process server: the cookie names no worker.
                groups["session", session_id].append(record)
            else:
                raise ControlError(
                    f"{self.kind} record of {record.episode_id.capture_key} names neither a session owner nor a "
                    "session ID, so its requests cannot be routed to the worker that would hold it"
                )
        for owner in self._restoring_owners:
            groups.setdefault(("owner", owner), [])
        workers = self._place(groups)
        routing = {
            group: self.workers[worker_index].registration["routing_id"] for group, worker_index in workers.items()
        }
        aliases = {name: routing_id for (kind, name), routing_id in routing.items() if kind == "owner"}
        placements = {name: routing_id for (kind, name), routing_id in routing.items() if kind == "session"}
        # A session placed by an earlier restore may still send a cookie without an owner:
        # it follows its recorded owner, the worker it was placed on then.
        for (kind, owner), placed in groups.items():
            if kind == "owner":
                for record in placed:
                    if record.session_id in self._restoring_placed:
                        placements[record.session_id] = aliases[owner]
        installs: dict[int, list[dict[str, Any]]] = {worker_index: [] for worker_index in self.workers}
        for group, worker_index in workers.items():
            for record in groups[group]:
                if group[0] == "session":
                    # The worker that holds it now is its owner from here on, as for a session it created.
                    record = record.model_copy(update={"owner": routing[group]})
                installs[worker_index].append(record.to_json_record())
        # Every worker installs, even with nothing to install, so each checks that it holds no live state.
        messages = {
            worker_index: (
                "install",
                {
                    "records": placed,
                    "aliases": aliases,
                    "placements": placements,
                    "scope": [episode_id.model_dump(mode="json") for episode_id in scope],
                },
            )
            for worker_index, placed in installs.items()
        }
        await self.call_each(messages, timeout=None)
        self.aliases.update(aliases)
        self.placements = placements
        self.restored = held

    def _place(self, groups: dict[tuple[str, str], list[CheckpointRecord]]) -> dict[tuple[str, str], int]:
        """Assign each group to the live worker with the fewest records, then groups, so far."""
        if not groups:
            return {}
        workers = sorted(
            worker_index for worker_index, worker in self.workers.items() if worker.registration.get("routing_id")
        )
        if not workers or len(workers) < len(self.workers):
            raise ControlError(f"restoring {self.kind} sessions requires workers that route sessions to their owner")
        load = {worker_index: [0, 0] for worker_index in workers}
        placement = {}
        for group in sorted(groups, key=lambda group: (-len(groups[group]), group)):
            worker_index = min(workers, key=lambda worker_index: (*load[worker_index], worker_index))
            placement[group] = worker_index
            load[worker_index][0] += len(groups[group])
            load[worker_index][1] += 1
        return placement

    def status_extra(self) -> dict[str, Any]:
        return {
            **self._status,
            **super().status_extra(),
            "restored_pending": sorted(self.restored),
            "session_aliases": len(self.aliases),
            "session_placements": len(self.placements),
        }


class ParticipantWorkerLink(WorkerLink):
    """One worker's participant, driven by the server's coordinator.

    ``app`` is the worker's app, whose session router reads the tables this link keeps current.
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
        if hasattr(participant, "claim_restored"):
            participant.claim_restored = self.claim
        self._accepting = True
        # Claim keys the coordinator holds a restored record for, as of the last reopen.
        self.restored_keys: set[str] = set()
        self._last_report: Optional[dict[str, Any]] = None
        self._poll_task: Optional[asyncio.Task] = None

    @property
    def accepting(self) -> bool:
        return self._accepting

    @property
    def routing_id(self) -> Optional[str]:
        """This worker's session routing ID, once the server has installed session routing."""
        return getattr(self.app.state, "nemo_gym_routing_id", None)

    def registration(self) -> dict[str, Any]:
        if hasattr(self.participant, "owner"):
            self.participant.owner = self.routing_id
        return {**super().registration(), "routing_id": self.routing_id}

    def adopt(self, state: dict[str, Any]) -> None:
        self.restored_keys = set(state["restored_keys"])
        self._adopt_routing(state)

    def _adopt_routing(self, state: dict[str, Any]) -> None:
        session_aliases(self.app).update(state["aliases"])
        # Replaced, not merged: the table covers only the sessions the last restore installed.
        placements = session_placements(self.app)
        placements.clear()
        placements.update(state["placements"])

    async def close_local(self, request: CheckpointRequest) -> None:
        self._accepting = False
        await self.participant.close_admission(request)
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_while_closed())

    async def open_local(self, body: dict[str, Any]) -> None:
        self._accepting = True
        self._last_report = None
        self.restored_keys = set(body["restored_keys"])
        self._adopt_routing(body)
        await self.participant.open_admission()

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
            unclaimed = getattr(self.participant, "unclaimed_sessions", None)
            return {
                "records": [record.to_json_record() for record in records],
                "unclaimed": unclaimed() if unclaimed is not None else [],
            }
        if kind == "delete_unclaimed":
            await self.participant.delete_unclaimed_session(body["session_id"])
            return {}
        if kind == "install":
            records = [self.participant.record_model.model_validate(record) for record in body["records"]]
            scope = [EpisodeId.model_validate(episode_id) for episode_id in body["scope"]]
            await self.participant.install(records, scope)
            self._adopt_routing(body)
            return {}
        if kind == "restored_pending":
            pending = await self.participant.restored_pending()
            return {"episode_ids": [episode_id.model_dump(mode="json") for episode_id in pending]}
        return await super().handle(kind, body)

    async def claim(self, key: str) -> Optional[dict[str, Any]]:
        """Claim the restored record under ``key`` from the coordinator; None if there is none to claim."""
        if key not in self.restored_keys:
            return None
        reply = await self.call("claim", {"key": key})
        self.restored_keys.discard(key)
        return reply["record"]

    async def _poll_while_closed(self) -> None:
        # Some changes to readiness do not notify, such as a session ending;
        # a single process polls for them while it waits, so a worker re-reports them.
        while not self._accepting:
            await asyncio.sleep(_REPORT_POLL_SECONDS)
            if not self._accepting and self.report().model_dump(mode="json") != self._last_report:
                await self.changed()

    def _numbered_report(self) -> dict[str, Any]:
        numbered = super()._numbered_report()
        self._last_report = numbered["report"]
        return numbered


def install_server_participant(
    app: FastAPI,
    participant: CheckpointParticipant,
    *,
    num_workers: int,
    auth_token: str,
    instance_name: str,
    lease_grace_seconds: float,
    status: Optional[dict[str, Any]] = None,
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
            CoordinatedServerParticipant(participant, expected_workers=num_workers, status=status),
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
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_coordinator(application: FastAPI) -> AsyncIterator[Any]:
        await link.connect()
        try:
            async with original_lifespan(application) as state:
                yield state
        finally:
            await link.disconnect()

    app.router.lifespan_context = lifespan_with_coordinator
