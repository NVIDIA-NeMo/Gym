# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A policy model server with several uvicorn workers is still one checkpoint participant.

uvicorn's workers share one port, so a control call reaches an arbitrary worker. One coordinator in the
main process, which serves no requests, owns the participant: the controller (phases, fencing, lease,
storage), the attempt fence, the restored generation cuts, and the ledger export and import through the
process-shared capture ledger. Each worker runs a ``PolicyGate`` for its own calls and:

- forwards every checkpoint control call to the coordinator;
- closes, reopens, and retires when the coordinator tells it to, and reports what it holds while a
  checkpoint is open (which responses are streaming, which calls are undelivered, and their cuts);
- claims a restored cut from the coordinator before running a re-issued call that may continue one.

The worker coordination itself (the socket, closing and reopening, readiness, fences, and losing a
worker) is ``nemo_gym._checkpoint.workers``.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Optional

from pydantic import BaseModel

from nemo_gym._checkpoint.control import CheckpointRecord, CheckpointRequest, PrepareReport
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
)
from nemo_gym._checkpoint.workers import (
    MESSAGE_TIMEOUT_SECONDS,
    CoordinatedParticipant,
    WorkerCoordinator,
    WorkerLink,
)
from nemo_gym._checkpoint.workers import coordinator_socket_path as _socket_path
from nemo_gym.episode_types import EpisodeId


LOGGER = logging.getLogger(__name__)

# Set by the main process for the workers it spawns.
COORDINATOR_SOCKET_ENV = "NEMO_GYM_POLICY_CHECKPOINT_SOCKET"


def coordinator_socket_path() -> str:
    return _socket_path("ng-policy")


# -- coordinator (main process) --------------------------------------------------------------------


class CoordinatedPolicyParticipant(CoordinatedParticipant[GateReport]):
    """The participant of a policy model server whose calls are spread over several worker processes."""

    kind = "model"
    record_model = ModelRecord
    report_model = GateReport
    worker_label = "policy"

    def __init__(self, ledger: Optional[CheckpointableLedger], *, expected_workers: int) -> None:
        super().__init__(expected_workers=expected_workers)
        self.ledger = ledger
        self.restored_cuts: dict[str, GenerationCutRecord] = {}

    def merge(self, reports: list[GateReport]) -> PrepareReport:
        return merge_reports(reports)

    def join_state(self, worker_id: int) -> dict[str, Any]:
        return {"restored_keys": sorted(self.restored_cuts)}

    def open_state(self) -> dict[str, Any]:
        return {"restored_keys": sorted(self.restored_cuts)}

    def drop_restored(self, episode_id: EpisodeId) -> None:
        for capture_key in [key for key in self.restored_cuts if covers(episode_id, key)]:
            del self.restored_cuts[capture_key]

    async def retire(self, episode_id: EpisodeId) -> None:
        await super().retire(episode_id)
        # After every worker cancelled the attempts' calls, so no late row recreates a ledger.
        await retire_ledgers(self.ledger, [episode_id])

    async def handle(self, worker_id: int, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        if kind == "claim_cut":
            record = self.claim_cut(body["capture_key"])
            return {"record": record.model_dump(mode="json") if record is not None else None}
        if kind == "release_cut":
            self.release_cut(body["capture_key"], GenerationCutRecord.model_validate(body["record"]))
            return {}
        return await super().handle(worker_id, kind, body)

    def claim_cut(self, capture_key: str) -> Optional[GenerationCutRecord]:
        return self.restored_cuts.pop(capture_key, None)

    def release_cut(self, capture_key: str, record: GenerationCutRecord) -> None:
        try:
            self.attempts.check(EpisodeId.from_capture_key(capture_key))
        except ControlError:
            return  # The attempt was retired while the call held the cut.
        self.restored_cuts.setdefault(capture_key, record)

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        raise NotImplementedError("the coordinated policy participant exports asynchronously; use export()")

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        if self.ledger is None:
            return []
        # Workers are closed, so what they hold is stable: fetch it once, then read the ledgers off the loop.
        replies = await self.broadcast("snapshot", {}, timeout=MESSAGE_TIMEOUT_SECONDS)
        snapshots = [GateSnapshot.model_validate(reply) for reply in replies.values()]
        return await asyncio.to_thread(
            export_model_records, self.ledger, episode_ids, snapshots, dict(self.restored_cuts)
        )

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        self._check_restorable()
        self.restored_cuts.update(import_model_records(self.ledger, records))

    async def install(self, records: list[CheckpointRecord]) -> None:
        self._check_restorable()
        self.restored_cuts.update(await asyncio.to_thread(import_model_records, self.ledger, records))
        # The restored attempts continue as the next attempt, so their own ledgers are no longer used.
        await retire_ledgers(self.ledger, [record.episode_id for record in records])

    async def restored_pending(self) -> list[EpisodeId]:
        return [EpisodeId.from_capture_key(key) for key in self.restored_cuts]

    def _check_restorable(self) -> None:
        if any(worker.report is None or worker.report.served for worker in self.workers.values()):
            raise ControlError("model restore requires freshly started policy workers; a worker has served calls")

    def commit_reply(self, records: list[CheckpointRecord]) -> dict[str, Any]:
        return {"staging_keys": retained_staging_keys(records)}

    def status_extra(self) -> dict[str, Any]:
        return {**super().status_extra(), "restored_generation_cuts": sorted(self.restored_cuts)}


class PolicyCoordinator(WorkerCoordinator):
    """Serves the coordinated participant to the workers of one policy model server."""

    participant: CoordinatedPolicyParticipant

    def __init__(
        self,
        ledger: Optional[CheckpointableLedger],
        *,
        expected_workers: int,
        instance_name: str,
        lease_grace_seconds: float,
        socket_path: str,
    ) -> None:
        super().__init__(
            CoordinatedPolicyParticipant(ledger, expected_workers=expected_workers),
            instance_name=instance_name,
            lease_grace_seconds=lease_grace_seconds,
            socket_path=socket_path,
        )


# -- worker ---------------------------------------------------------------------------------------


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
                "release_cut", {"capture_key": ticket.capture_key, "record": record.model_dump(mode="json")}
            )
        except ControlError:
            LOGGER.warning("could not release the restored cut of %s", ticket.capture_key)


class PolicyWorkerLink(WorkerLink):
    """One worker's gate, driven by the coordinator in the main process."""

    worker_label = "policy"

    def __init__(
        self,
        *,
        socket_path: str,
        server_name: str,
        cut_requester: Optional[CutRequester],
        on_coordinator_lost: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(socket_path=socket_path, on_coordinator_lost=on_coordinator_lost)
        self.restored_cuts = CoordinatedRestoredCuts(self)
        self.gate = PolicyGate(
            server_name=server_name,
            cut_requester=cut_requester,
            attempts=self.attempts,
            restored_cuts=self.restored_cuts,
            on_change=self.changed,
        )

    @property
    def accepting(self) -> bool:
        return self.gate.accepting

    def adopt(self, state: dict[str, Any]) -> None:
        self.restored_cuts.keys = set(state["restored_keys"])

    async def close_local(self, request: CheckpointRequest) -> None:
        await self.gate.close(request)

    async def open_local(self, body: dict[str, Any]) -> None:
        self.restored_cuts.keys = set(body["restored_keys"])
        self.gate.open()

    async def retire_local(self, episode_id: EpisodeId) -> None:
        await self.gate.retire(episode_id)

    def report(self) -> BaseModel:
        return self.gate.report()

    async def handle(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        if kind == "snapshot":
            return self.gate.snapshot().model_dump(mode="json")
        return await super().handle(kind, body)
