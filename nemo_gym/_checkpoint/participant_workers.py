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
    resources server, are aliased too. A worker that joins later gets the table when it registers.
  - State that lives inside one request (environment episodes, legacy agent ``/run`` episodes) stays with
    the coordinator. The worker that receives the replacement attempt's ``/run`` claims it; exactly one
    claim succeeds. A record nobody has claimed is exported again by the next checkpoint.

- **retire** and attempt fences reach every worker, and the coordinator drops what it holds for a retired
  attempt.

Claims are refused while a checkpoint is open, as a new episode is: the caller retries the episode later.
A claim is answered before any later message to its worker, so a worker that wins a claim starts the
episode before it can see the next checkpoint close, and that checkpoint waits for the episode instead of
missing it.
"""

import asyncio
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
from nemo_gym._checkpoint.errors import AdmissionClosedError, ControlError
from nemo_gym._checkpoint.workers import (
    CoordinatedParticipant,
    WorkerCoordinator,
    WorkerLink,
    coordinator_socket_path,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import is_nemo_gym_fastapi_worker
from nemo_gym.session_routing import session_aliases


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
        # Session owners a restoring checkpoint names, including those whose sessions exported nothing.
        self._restoring_owners: list[str] = []

    def merge(self, reports: list[PrepareReport]) -> PrepareReport:
        counts: Counter[str] = Counter()
        for report in reports:
            counts.update(report.counts)
        return PrepareReport(
            ready=all(report.ready for report in reports),
            blockers=sorted({blocker for report in reports for blocker in report.blockers}),
            counts=dict(counts),
        )

    def join_state(self, worker_id: int) -> dict[str, Any]:
        return {"restored_keys": sorted(self.restored), "aliases": self.aliases}

    def open_state(self) -> dict[str, Any]:
        return {"restored_keys": sorted(self.restored)}

    def drop_restored(self, episode_id: EpisodeId) -> None:
        for key, record in list(self.restored.items()):
            replacement = next_attempt(record.episode_id)
            if replacement.rollout_id == episode_id.rollout_id and replacement.attempt <= episode_id.attempt:
                del self.restored[key]

    def handle_now(self, worker_id: int, kind: str, body: dict[str, Any]) -> Optional[dict[str, Any]]:
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
        records = [
            self.record_model.model_validate(record) for reply in replies.values() for record in reply["records"]
        ]
        # A restored episode whose replacement has not started still continues from its restored state.
        pending = [
            record.model_copy(update={"episode_id": next_attempt(record.episode_id)})
            for record in self.restored.values()
        ]
        return records + pending

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        raise NotImplementedError("a coordinated participant restores through install()")

    def manifest_extra(self) -> dict[str, Any]:
        # Every owner a live cookie or token may name: the workers' own IDs, and the pre-crash owners they serve.
        # A session that exported no record, such as one of a stateless resources server, still needs its
        # requests routed somewhere after a restore.
        owners = {worker.registration.get("routing_id") for worker in self.workers.values()} | set(self.aliases)
        return {"session_owners": sorted(owner for owner in owners if owner)}

    def restore_manifest(self, manifest: dict[str, Any]) -> None:
        self._restoring_owners = list(manifest.get("session_owners") or [])

    async def install(self, records: list[CheckpointRecord]) -> None:
        if self.restored:
            raise ControlError(f"{self.kind} restore requires a server without restored state")
        held: dict[str, CheckpointRecord] = {}
        by_owner: dict[str, list[CheckpointRecord]] = defaultdict(list)
        for record in records:
            key = self.claim_key(record)
            if key is not None:
                held[key] = record
                continue
            owner = getattr(record, "owner", None)
            if owner is None:
                raise ControlError(
                    f"{self.kind} record of {record.episode_id.capture_key} has no session owner, so its requests "
                    "cannot be routed to the worker that would hold it"
                )
            by_owner[owner].append(record)
        for owner in self._restoring_owners:
            by_owner.setdefault(owner, [])
        placement = self._place(by_owner)
        aliases = {owner: self.workers[worker_id].registration["routing_id"] for owner, worker_id in placement.items()}
        # Every worker installs, even with nothing to install, so each checks that it holds no live state.
        messages = {
            worker_id: (
                "install",
                {
                    "records": [
                        record.to_json_record()
                        for owner, placed in placement.items()
                        if placed == worker_id
                        for record in by_owner[owner]
                    ],
                    "aliases": aliases,
                },
            )
            for worker_id in self.workers
        }
        await self.call_each(messages, timeout=None)
        self.aliases.update(aliases)
        self.restored = held

    def _place(self, by_owner: dict[str, list[CheckpointRecord]]) -> dict[str, int]:
        """Assign each pre-crash owner to the live worker with the fewest records, then owners, so far."""
        if not by_owner:
            return {}
        workers = sorted(
            worker_id for worker_id, worker in self.workers.items() if worker.registration.get("routing_id")
        )
        if not workers or len(workers) < len(self.workers):
            raise ControlError(f"restoring {self.kind} sessions requires workers that route sessions to their owner")
        load = {worker_id: [0, 0] for worker_id in workers}
        placement = {}
        for owner in sorted(by_owner, key=lambda owner: (-len(by_owner[owner]), owner)):
            worker_id = min(workers, key=lambda worker_id: (*load[worker_id], worker_id))
            placement[owner] = worker_id
            load[worker_id][0] += len(by_owner[owner])
            load[worker_id][1] += 1
        return placement

    def status_extra(self) -> dict[str, Any]:
        return {
            **self._status,
            **super().status_extra(),
            "restored_pending": sorted(self.restored),
            "session_aliases": len(self.aliases),
        }


class ParticipantWorkerLink(WorkerLink):
    """One worker's participant, driven by the server's coordinator.

    ``app`` is the worker's app, whose session router reads the alias table this link keeps current.
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
        # The coordinator holds the fence; this worker's participant checks requests against a copy of it.
        participant.attempts = self.attempts
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
        return getattr(self.app.state, "nemo_gym_session_owner", None)

    def registration(self) -> dict[str, Any]:
        if hasattr(self.participant, "owner"):
            self.participant.owner = self.routing_id
        return {**super().registration(), "routing_id": self.routing_id}

    def adopt(self, state: dict[str, Any]) -> None:
        self.restored_keys = set(state["restored_keys"])
        session_aliases(self.app).update(state["aliases"])

    async def close_local(self, request: CheckpointRequest) -> None:
        self._accepting = False
        await self.participant.close_admission(request)
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_while_closed())

    async def open_local(self, body: dict[str, Any]) -> None:
        self._accepting = True
        self._last_report = None
        self.restored_keys = set(body["restored_keys"])
        await self.participant.open_admission()

    async def retire_local(self, episode_id: EpisodeId) -> None:
        await self.participant.retire(episode_id)

    def report(self) -> BaseModel:
        return self.participant.readiness()

    async def handle(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        if kind == "export":
            episode_ids = body["episode_ids"]
            records = await self.participant.export(
                None if episode_ids is None else [EpisodeId.model_validate(e) for e in episode_ids]
            )
            return {"records": [record.to_json_record() for record in records]}
        if kind == "install":
            records = [self.participant.record_model.model_validate(record) for record in body["records"]]
            await self.participant.install(records)
            session_aliases(self.app).update(body["aliases"])
            return {}
        return await super().handle(kind, body)

    async def claim(self, key: str) -> Optional[dict[str, Any]]:
        """Claim the restored record under ``key`` from the coordinator; None if there is none to claim."""
        if key not in self.restored_keys:
            return None
        reply = await self.call("claim", {"key": key})
        self.restored_keys.discard(key)
        return reply["record"]

    async def _poll_while_closed(self) -> None:
        # Some changes to readiness do not notify, such as a session ending; a single process polls for
        # them while it waits, so a worker re-reports them.
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

    With several workers, the main process, which serves no requests, runs the coordinator, and each
    worker links ``participant`` to it.
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
            socket_path=coordinator_socket_path(f"ng-{participant.kind}"),
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
