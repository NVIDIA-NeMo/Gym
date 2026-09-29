# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy model server participant: hold undelivered generations and checkpoint their lineage.

Only the model server that produces training tokens participates. Judge and simulator model servers
keep serving so that accepted tool and verifier calls can finish.

When a checkpoint closes admission, a new policy call is refused with 409 ``checkpoint_parked``. A call
already admitted keeps running, but its response is held until resume unless it had started before
admission closed. A call whose response the client has not received is *undelivered*: the agent that
issued it is still at the boundary it recorded before the call, so the checkpoint leaves the call's
ledger rows out and the restored agent re-issues it. Prepare therefore never waits for a generation to
finish; only a response already streaming must finish sending.

With generation cuts enabled, prepare also asks each inference worker to stage the prefix every
undelivered call has generated so far. Restore attaches that prefix to the re-issued call's admission,
so the worker continues it instead of generating from scratch. A call the worker could not cut is
simply regenerated.
"""

import asyncio
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, JsonValue

from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    ControlError,
    PrepareReport,
    next_attempt,
)
from nemo_gym._checkpoint.errors import AdmissionClosedError
from nemo_gym._checkpoint.generation_cut import (
    GENERATION_CUT_ROUTE,
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutPrefixAck,
    GenerationCutReceipt,
)
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.episode_types import EpisodeId
from nemo_gym.token_id_capture.fingerprint import conversation_digest
from nemo_gym.token_id_capture.sink import CaptureContext
from nemo_gym.token_id_capture.staging.records import CaptureAdmission, GenerationCutContinuation


LOGGER = logging.getLogger(__name__)

GENERATION_ROUTES = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
_ROLLOUT_PREFIX = re.compile(rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<capture_key>[^/]+)")

# Sends one inventory to the worker at ``backend`` and returns its receipt.
CutRequester = Callable[[str, GenerationCutInventory], Awaitable[GenerationCutReceipt]]


class GenerationCutRecord(BaseModel):
    """A durable prefix of an undelivered call, and the request it answers."""

    model_config = ConfigDict(extra="forbid")

    model_call_id: str
    request_digest: str
    continuation: GenerationCutContinuation


class ModelRecord(CheckpointRecord):
    """The token-free capture-ledger rows of one continued episode, up to the checkpoint.

    Rows name staged token coordinates that the training framework checkpoints itself; they carry no
    token arrays. Restore installs them as the ledger of the next attempt, so the replacement's first
    call resolves its parent from the restored rows exactly as a later turn of the same attempt would,
    and the next attempt's manifest carries the whole continued lineage.
    """

    rows: list[dict[str, JsonValue]]
    generation_cuts: list[GenerationCutRecord] = []


@runtime_checkable
class CheckpointableLedger(Protocol):
    def export_rows(self, rollout_id: str) -> list[dict]: ...

    def import_rows(self, rollout_id: str, rows: list[dict]) -> None: ...


@dataclass(eq=False)
class _Ticket:
    """One admitted generation request."""

    participant: "PolicyModelParticipant"
    capture_key: str
    task: Optional[asyncio.Task]
    ticket_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    admitted_at: float = field(default_factory=time.time)
    response_started: bool = False
    backend: Optional[str] = None
    capture: Optional[CaptureContext] = None
    cut: Optional[GenerationCutPrefixAck] = None

    @property
    def undelivered_call_id(self) -> Optional[str]:
        return self.capture.model_call_id if self.capture is not None and not self.response_started else None


_CURRENT_TICKET: ContextVar[Optional[_Ticket]] = ContextVar("nemo_gym_policy_ticket", default=None)


def attach_capture_context(context: CaptureContext) -> None:
    """Bind the current policy call's capture context to its checkpoint ticket.

    The capture middleware calls this for every captured call on every model server, so the checkpoint
    can leave an undelivered call's ledger rows out and continue a restored generation cut.
    """
    ticket = _CURRENT_TICKET.get()
    if ticket is not None:
        ticket.capture = context
        context.admission_hook = ticket.participant.continue_restored_cut


def note_generation_backend(base_url: Any) -> None:
    """Record which inference worker serves the current policy call, so a checkpoint can cut it there."""
    ticket = _CURRENT_TICKET.get()
    if ticket is not None and base_url is not None:
        ticket.backend = str(base_url)


class PolicyModelParticipant(CheckpointParticipant):
    kind = "model"
    record_model = ModelRecord

    def __init__(
        self,
        ledger: Optional[CheckpointableLedger] = None,
        *,
        server_name: str = "policy",
        cut_requester: Optional[CutRequester] = None,
    ) -> None:
        super().__init__()
        self.ledger = ledger
        self.server_name = server_name
        self.cut_requester = cut_requester
        self.accepting = True
        self._reopened = asyncio.Event()
        self._reopened.set()
        self._tickets: set[_Ticket] = set()
        self._restored_cuts: dict[str, GenerationCutRecord] = {}

    # -- data plane -----------------------------------------------------------------------------

    def admit(self, capture_key: Optional[str]) -> None:
        if capture_key is not None:
            self.attempts.check(EpisodeId.from_capture_key(capture_key))
        if not self.accepting:
            raise AdmissionClosedError("policy model admission is closed for a checkpoint")

    def enter(self, capture_key: str) -> _Ticket:
        ticket = _Ticket(participant=self, capture_key=capture_key, task=asyncio.current_task())
        self._tickets.add(ticket)
        return ticket

    async def exit(self, ticket: _Ticket) -> None:
        self._tickets.discard(ticket)
        await self.notify()

    async def release_response(self, ticket: _Ticket) -> None:
        """Hold a response until resume unless it may be delivered now."""
        while not self.accepting:
            await self.notify()
            await self._reopened.wait()
        ticket.response_started = True

    def continue_restored_cut(self, context: CaptureContext, admission: CaptureAdmission) -> CaptureAdmission:
        """Attach a restored generation cut to the re-issued call it belongs to."""
        record = self._restored_cuts.get(context.rollout_id)
        if record is None or conversation_digest(list(context.request_items or [])) != record.request_digest:
            return admission
        del self._restored_cuts[context.rollout_id]
        return admission.model_copy(update={"generation_cut": record.continuation})

    # -- checkpoint -----------------------------------------------------------------------------

    async def close_admission(self, request: CheckpointRequest) -> None:
        self.accepting = False
        self._reopened.clear()
        if self.cut_requester is not None:
            # One cut round for the calls already on a worker. A call that reaches a worker later is
            # still held; without a cut it is simply regenerated after a restore.
            await self._cut_all(request)

    async def open_admission(self) -> None:
        self.accepting = True
        self._reopened.set()
        for ticket in self._tickets:
            ticket.cut = None

    async def _cut_all(self, request: CheckpointRequest) -> None:
        by_backend: dict[str, list[_Ticket]] = {}
        for ticket in self._tickets:
            if ticket.backend is not None and ticket.undelivered_call_id is not None:
                by_backend.setdefault(ticket.backend, []).append(ticket)
        await asyncio.gather(*(self._cut(request, backend, tickets) for backend, tickets in by_backend.items()))

    async def _cut(self, request: CheckpointRequest, backend: str, tickets: list[_Ticket]) -> None:
        prefixes = {}
        for ticket in tickets:
            episode_id = EpisodeId.from_capture_key(ticket.capture_key)
            prefixes[ticket.ticket_id] = GenerationCutPrefix(
                ticket_id=ticket.ticket_id,
                rollout_id=episode_id.rollout_id,
                attempt_index=episode_id.attempt,
                model_call_id=ticket.capture.model_call_id,
                admitted_at=ticket.admitted_at,
            )
        inventory = GenerationCutInventory.build(
            checkpoint_id=request.checkpoint_id, server_name=self.server_name, active_prefixes=list(prefixes.values())
        )
        try:
            async with asyncio.timeout(max(0.0, request.deadline_ts - time.time())):
                receipt = await self.cut_requester(backend, inventory)
            receipt.validate_for(inventory)
            acks = {ack.ticket_id: ack for ack in receipt.prefixes}
        except Exception:
            # A missing cut only costs regeneration after restore; never block the checkpoint on it.
            LOGGER.warning("generation cut failed on %s; those calls regenerate after restore", backend, exc_info=True)
            acks = {ticket_id: GenerationCutPrefixAck.failure(prefix) for ticket_id, prefix in prefixes.items()}
        for ticket in tickets:
            ticket.cut = acks[ticket.ticket_id]

    def readiness(self) -> PrepareReport:
        # Held calls never block; only a response that started before admission closed must finish sending.
        streaming = sorted({ticket.capture_key for ticket in self._tickets if ticket.response_started})
        durable_cuts = sum(
            ticket.cut is not None and ticket.cut.disposition == "durable_prefix" for ticket in self._tickets
        )
        return PrepareReport(
            ready=not streaming,
            blockers=streaming,
            counts={
                "inflight": len(self._tickets),
                "held": sum(not ticket.response_started for ticket in self._tickets),
                "cut": durable_cuts,
            },
        )

    async def retire(self, episode_id: EpisodeId) -> None:
        for ticket in list(self._tickets):
            if _covers(episode_id, ticket.capture_key) and ticket.task is not None:
                ticket.task.cancel()
        for capture_key in [key for key in self._restored_cuts if _covers(episode_id, key)]:
            del self._restored_cuts[capture_key]

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        if self.ledger is None:
            return []
        if episode_ids is None:
            raise ControlError("model commit needs the episode_ids the controller continues from this checkpoint")
        undelivered = {
            ticket.undelivered_call_id: ticket for ticket in self._tickets if ticket.undelivered_call_id is not None
        }
        records = []
        for episode_id in episode_ids:
            rows = [
                row
                for row in self.ledger.export_rows(episode_id.capture_key)
                if row.get("model_call_id") not in undelivered
            ]
            cuts = [
                _cut_record(ticket)
                for ticket in undelivered.values()
                if ticket.capture_key == episode_id.capture_key
                and ticket.cut is not None
                and ticket.cut.disposition == "durable_prefix"
            ]
            # A restored cut that no re-issued call has consumed yet is still this episode's prefix.
            restored_cut = self._restored_cuts.get(episode_id.capture_key)
            if restored_cut is not None:
                cuts.append(restored_cut)
            if rows or cuts:
                records.append(ModelRecord(episode_id=episode_id, rows=rows, generation_cuts=cuts))
        return records

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        if self._tickets:
            raise ControlError("model restore requires a process that is not serving generations")
        if records and self.ledger is None:
            raise ControlError("checkpoint holds capture-ledger rows but this model server has no capture ledger")
        targets = [(record, next_attempt(record.episode_id).capture_key) for record in records]
        for record, target in targets:
            existing = self.ledger.export_rows(target)
            if existing and existing != record.rows:
                raise ControlError(f"capture ledger for {target} already holds rows from another execution")
            if len(record.generation_cuts) > 1:
                raise ControlError(f"episode {record.episode_id.capture_key} has more than one undelivered cut")
        for record, target in targets:
            self.ledger.import_rows(target, record.rows)
            if record.generation_cuts:
                self._restored_cuts[target] = record.generation_cuts[0]

    def status_extra(self) -> dict[str, Any]:
        return {"restored_generation_cuts": sorted(self._restored_cuts)}


def _cut_record(ticket: _Ticket) -> GenerationCutRecord:
    ack = ticket.cut
    return GenerationCutRecord(
        model_call_id=ack.model_call_id,
        request_digest=conversation_digest(list(ticket.capture.request_items or [])),
        continuation=GenerationCutContinuation(
            source_capture_key=ticket.capture_key,
            source_model_call_id=ack.model_call_id,
            staging_keys=ack.staging_keys,
            generation_token_count=ack.prefix_token_count,
            digest=ack.prefix_digest,
            effective_output_limit=ack.effective_output_limit,
            terminal_finish_reason=ack.terminal_finish_reason,
            terminal_stop_reason=ack.terminal_stop_reason,
        ),
    )


def _covers(retired: EpisodeId, capture_key: str) -> bool:
    """Whether retiring ``retired`` discards the attempt named by ``capture_key``."""
    other = EpisodeId.from_capture_key(capture_key)
    return other.rollout_id == retired.rollout_id and other.attempt <= retired.attempt


class PolicyAdmissionMiddleware:
    """Pure ASGI gate in front of generation routes."""

    def __init__(self, app: Any, participant: PolicyModelParticipant) -> None:
        self.app = app
        self.participant = participant

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not scope.get("path", "").endswith(GENERATION_ROUTES):
            await self.app(scope, receive, send)
            return
        match = _ROLLOUT_PREFIX.match(scope["path"])
        capture_key = match.group("capture_key") if match else None
        try:
            self.participant.admit(capture_key)
        except ControlError as error:
            await error.response()(scope, receive, send)
            return

        ticket = self.participant.enter(capture_key or "")

        async def gated_send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                await self.participant.release_response(ticket)
            await send(message)

        token = _CURRENT_TICKET.set(ticket)
        try:
            await self.app(scope, receive, gated_send)
        finally:
            _CURRENT_TICKET.reset(token)
            await self.participant.exit(ticket)


def generation_cut_requester(auth_token: str) -> CutRequester:
    """Request cuts from a worker's ``/ng-control/v1/generation-cut`` endpoint.

    ``backend`` is the worker's OpenAI-compatible base URL; the control route lives at its root.
    """

    async def request_cut(backend: str, inventory: GenerationCutInventory) -> GenerationCutReceipt:
        from nemo_gym.server_utils import get_response_json, raise_for_status, request

        root = backend.rstrip("/").removesuffix("/v1")
        response = await request(
            method="POST",
            url=f"{root}{GENERATION_CUT_ROUTE}",
            json=inventory.model_dump(mode="json"),
            headers={"authorization": f"Bearer {auth_token}"},
            _internal=True,
        )
        await raise_for_status(response)
        return GenerationCutReceipt.model_validate(await get_response_json(response))

    return request_cut
