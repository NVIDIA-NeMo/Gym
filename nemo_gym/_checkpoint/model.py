# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy model server participant: hold undelivered generations and checkpoint their lineage.

Only the model server that produces training tokens participates.
Judge and simulator model servers keep serving so that accepted tool and verifier calls can finish.

When a checkpoint closes admission, a new policy call waits until resume before it is admitted,
as a resources server's requests do.
A waiting call is not admitted, so it never holds up prepare:
the agent that sent it is still at the boundary it recorded before the call, and a restored agent sends it again.
A call already admitted keeps running,
but its response is held until resume unless it had started before admission closed.
A call whose response the client has not received is *undelivered*:
the agent that issued it is still at the boundary it recorded before the call,
so the checkpoint leaves the call's ledger rows out and the restored agent re-issues it.
Prepare therefore never waits for a generation to finish; only a response already streaming must finish sending.

With generation cuts enabled,
prepare also asks each inference worker to stage the prefix every undelivered call has generated so far.
Restore attaches that prefix to the re-issued call's admission,
so the worker continues it instead of generating from scratch.
A call the worker could not cut is simply regenerated.

The data plane of one server process is a ``PolicyGate``.
With one uvicorn worker, a ``PolicyModelParticipant`` owns the gate directly.
With several,
each worker runs a gate and one coordinator in the main process owns the participant
(``nemo_gym._checkpoint.model_workers``).
"""

import asyncio
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    ControlError,
    JsonPayload,
    PrepareReport,
    RetiredAttempts,
    next_attempt,
)
from nemo_gym._checkpoint.errors import InvalidRolloutIdError
from nemo_gym._checkpoint.generation_cut import (
    GENERATION_CUT_ROUTE,
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutPrefixAck,
    GenerationCutReceipt,
)
from nemo_gym._checkpoint.telemetry import checkpoint_span
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.episode_types import EpisodeId
from nemo_gym.telemetry.gym_metrics import record_checkpoint_event
from nemo_gym.token_id_capture.fingerprint import conversation_digest
from nemo_gym.token_id_capture.sink import CaptureContext
from nemo_gym.token_id_capture.staging.records import CaptureAdmission, GenerationCutContinuation


LOGGER = logging.getLogger(__name__)

GENERATION_ROUTES = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
_ROLLOUT_PREFIX = re.compile(rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<capture_key>[^/]+)")

# Request fields that compile into a structured decoder.
# A cut prefix cannot restore that decoder's state,
# so such a call restarts after a restore instead of continuing its prefix.
_STRUCTURED_GENERATION_FIELDS = (
    "guided_choice",
    "guided_grammar",
    "guided_json",
    "guided_regex",
    "guided_whitespace_pattern",
    "structural_tag",
    "structured_outputs",
)

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

    Rows name staged token coordinates that the training framework checkpoints itself; they carry no token arrays.
    Restore installs them as the ledger of the next attempt,
    so the replacement's first call resolves its parent from the restored rows exactly
    as a later turn of the same attempt would, and the next attempt's manifest carries the whole continued lineage.
    """

    rows: JsonPayload
    generation_cuts: list[GenerationCutRecord] = []


@runtime_checkable
class CheckpointableLedger(Protocol):
    def export_rows(self, rollout_id: str) -> list[dict]: ...

    def import_rows(self, rollout_id: str, rows: list[dict]) -> None: ...

    async def retire(self, rollout_ids: Sequence[str]) -> dict: ...


class EpisodeCut(BaseModel):
    """A durable cut of one undelivered call, keyed by the episode that will continue it."""

    model_config = ConfigDict(extra="forbid")

    capture_key: str
    record: GenerationCutRecord
    # When the cut call was admitted:
    # an episode can hold several undelivered calls when its client retried a call the server was still running,
    # and only the latest is still awaited.
    admitted_at: float = 0.0


class GateReport(BaseModel):
    """One process's readiness: cheap enough to recompute on every change while a checkpoint is open."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    streaming: list[str]
    inflight: int
    held: int
    cut: int = 0
    # Undelivered calls the worker could not cut, and calls not cut because their decoding is constrained.
    # Both regenerate after a restore; a worker that cuts nothing shows up here instead of silently.
    cut_failed: int = 0
    cut_skipped: int = 0
    # Whether this process has served any call since it started; restore refuses one that has.
    served: bool = False


class GateSnapshot(BaseModel):
    """What one process contributes to a commit; taken once, while admission is closed."""

    model_config = ConfigDict(extra="forbid")

    # Undelivered model calls (model_call_id -> capture key): the commit leaves their ledger rows out.
    undelivered: dict[str, str]
    cuts: list[EpisodeCut]


class RestoredCuts(Protocol):
    """Where a gate finds the restored generation cut a re-issued call continues."""

    async def prefetch(self, ticket: "_Ticket") -> None:
        """Claim the ticket's restored cut before the call runs, if claiming needs I/O."""

    def take(
        self, ticket: Optional["_Ticket"], capture_key: str, request_digest: str
    ) -> Optional[GenerationCutRecord]:
        """Return and consume the restored cut for this request, if it has one."""

    async def settle(self, ticket: "_Ticket") -> None:
        """Give back a claimed cut the call did not use."""


class LocalRestoredCuts:
    """Restored cuts held in this process; claiming needs no I/O."""

    def __init__(self) -> None:
        self.records: dict[str, GenerationCutRecord] = {}

    async def prefetch(self, ticket: "_Ticket") -> None:
        return None

    def take(
        self, ticket: Optional["_Ticket"], capture_key: str, request_digest: str
    ) -> Optional[GenerationCutRecord]:
        record = self.records.get(capture_key)
        if record is None or record.request_digest != request_digest:
            return None
        # Kept until the call delivers its response: a call still undelivered at the next checkpoint,
        # whose own cut fails, continues from this one again.
        if ticket is not None:
            ticket.claimed_cut, ticket.claimed_cut_used = record, True
        return record

    async def settle(self, ticket: "_Ticket") -> None:
        record, ticket.claimed_cut = ticket.claimed_cut, None
        if record is not None and ticket.claimed_cut_used and ticket.response_started:
            if self.records.get(ticket.capture_key) is record:
                del self.records[ticket.capture_key]


@dataclass(eq=False)
class _Ticket:
    """One admitted generation request."""

    gate: "PolicyGate"
    capture_key: str
    task: Optional[asyncio.Task]
    ticket_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    admitted_at: float = field(default_factory=time.time)
    response_started: bool = False
    backend: Optional[str] = None
    capture: Optional[CaptureContext] = None
    cut: Optional[GenerationCutPrefixAck] = None
    # The durable cut as a record, built once when the worker acknowledges it.
    cut_record: Optional[GenerationCutRecord] = None
    # Why this call must restart rather than continue a cut prefix (constrained decoding), if it must.
    cut_restart_reason: Optional[str] = None
    # A restored cut claimed from another process for this call, and whether the call used it.
    claimed_cut: Optional[GenerationCutRecord] = None
    claimed_cut_used: bool = False

    @property
    def undelivered_call_id(self) -> Optional[str]:
        return self.capture.model_call_id if self.capture is not None and not self.response_started else None


_CURRENT_TICKET: ContextVar[Optional[_Ticket]] = ContextVar("nemo_gym_policy_ticket", default=None)


def attach_capture_context(context: CaptureContext) -> None:
    """Bind the current policy call's capture context to its checkpoint ticket.

    The capture middleware calls this for every captured call on every model server,
    so the checkpoint can leave an undelivered call's ledger rows out and continue a restored generation cut.
    """
    ticket = _CURRENT_TICKET.get()
    if ticket is not None:
        ticket.capture = context
        context.admission_hook = ticket.gate.admission_hook(ticket)


def generation_cut_restart_reason(body: Mapping[str, Any]) -> Optional[str]:
    """Why a generation request cannot continue a cut prefix: its decoding is constrained."""
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "none"}:
            return f"tool_choice:{tool_choice}"
    elif tool_choice is not None:
        # Named function and allowed-tool choices compile into a structured decoder.
        return "tool_choice:constrained"
    response_format = body.get("response_format")
    if response_format is not None and (
        not isinstance(response_format, Mapping) or response_format.get("type") != "text"
    ):
        format_type = response_format.get("type", "unknown") if isinstance(response_format, Mapping) else "unknown"
        return f"response_format:{format_type}"
    for name in _STRUCTURED_GENERATION_FIELDS:
        if body.get(name) is not None:
            return name
    return None


def note_generation_request(body: Mapping[str, Any]) -> None:
    """Record the request the current policy call sends, so a checkpoint knows whether it may cut it."""
    ticket = _CURRENT_TICKET.get()
    if ticket is not None:
        ticket.cut_restart_reason = generation_cut_restart_reason(body)


def note_generation_backend(base_url: Any) -> None:
    """Record which inference worker serves the current policy call, so a checkpoint can cut it there."""
    ticket = _CURRENT_TICKET.get()
    if ticket is not None and base_url is not None:
        ticket.backend = str(base_url)


class PolicyGate:
    """The policy data plane of one server process: admission, held responses, and generation cuts."""

    def __init__(
        self,
        *,
        server_name: str,
        cut_requester: Optional[CutRequester],
        retired: RetiredAttempts,
        restored_cuts: RestoredCuts,
        on_change: Callable[[], Awaitable[None]],
    ) -> None:
        self.server_name = server_name
        self.cut_requester = cut_requester
        self.retired = retired
        self.restored_cuts = restored_cuts
        self.on_change = on_change
        self.accepting = True
        self._reopened = asyncio.Event()
        self._reopened.set()
        # Set, then replaced, whenever admission reopens or an attempt is retired, to wake waiting calls.
        self._admission_changed = asyncio.Event()
        self.tickets: set[_Ticket] = set()
        # Restore expects a freshly started server: a ledger this process wrote may belong to a live episode.
        self.served = False
        # Restored attempts no call has reached here yet, by capture key.
        self.restored_targets: set[str] = set()

    async def admit(self, capture_key: Optional[str]) -> None:
        """Wait while a checkpoint is open, then admit the call unless its attempt was retired.

        The caller must ``enter`` before it awaits anything else, so no checkpoint can close in between.
        """
        episode_id = None
        if capture_key is not None:
            try:
                episode_id = EpisodeId.from_capture_key(capture_key)
            except ValueError as error:
                raise InvalidRolloutIdError(f"{capture_key!r} is not a valid rollout id") from error
        while True:
            changed = self._admission_changed
            if episode_id is not None:
                self.retired.check(episode_id)
            if self.accepting:
                return
            await changed.wait()

    def _wake_admission(self) -> None:
        self._admission_changed.set()
        self._admission_changed = asyncio.Event()

    def enter(self, capture_key: str) -> _Ticket:
        ticket = _Ticket(gate=self, capture_key=capture_key, task=asyncio.current_task())
        self.served = True
        # The attempt has started here: a commit that no longer names it must not delete its ledger.
        self.restored_targets.discard(capture_key)
        self.tickets.add(ticket)
        return ticket

    async def exit(self, ticket: _Ticket) -> None:
        self.tickets.discard(ticket)
        try:
            await self.restored_cuts.settle(ticket)
        finally:
            # Reported even if giving back a claimed cut fails, or a coordinator keeps a stale report.
            await self.on_change()

    async def deliver_response(self, ticket: _Ticket) -> None:
        """Let a response start, holding it until resume while a checkpoint is open."""
        while not self.accepting:
            await self.on_change()
            await self._reopened.wait()
        ticket.response_started = True

    def admission_hook(self, ticket: _Ticket) -> Callable[[CaptureContext, CaptureAdmission], CaptureAdmission]:
        def continue_restored_cut(context: CaptureContext, admission: CaptureAdmission) -> CaptureAdmission:
            if ticket.cut_restart_reason is not None:
                return admission
            digest = conversation_digest(list(context.request_items or []))
            record = self.restored_cuts.take(ticket, context.rollout_id, digest)
            if record is None:
                return admission
            return admission.model_copy(update={"generation_cut": record.continuation})

        return continue_restored_cut

    async def close(self, request: CheckpointRequest) -> None:
        self.accepting = False
        self._reopened.clear()
        if self.cut_requester is not None:
            # One cut round for the calls already on a worker.
            # A call that reaches a worker later is still held; without a cut it is simply regenerated after a restore.
            await self._cut_all(request)

    def open(self) -> None:
        self.accepting = True
        self._reopened.set()
        self._wake_admission()
        for ticket in self.tickets:
            ticket.cut = None
            ticket.cut_record = None

    async def retire(self, episode_id: EpisodeId) -> None:
        """Refuse the attempts' waiting calls, cancel their admitted calls, and wait until those have stopped."""
        self._wake_admission()
        tasks = [
            ticket.task
            for ticket in list(self.tickets)
            # An untagged call (an eval, a judge, a health probe) belongs to no episode.
            if ticket.capture_key
            and covers(episode_id, ticket.capture_key)
            and ticket.task is not None
            and ticket.task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks)

    def report(self) -> GateReport:
        # Held calls never block; only a response that started before admission closed must finish sending.
        streaming = sorted({ticket.capture_key for ticket in self.tickets if ticket.response_started})
        undelivered = [ticket for ticket in self.tickets if ticket.undelivered_call_id is not None]
        return GateReport(
            ready=not streaming,
            streaming=streaming,
            inflight=len(self.tickets),
            served=self.served,
            held=len(self.tickets) - sum(ticket.response_started for ticket in self.tickets),
            cut=sum(ticket.cut_record is not None for ticket in undelivered),
            cut_failed=sum(ticket.cut is not None and ticket.cut_record is None for ticket in undelivered),
            cut_skipped=sum(ticket.cut_restart_reason is not None for ticket in undelivered),
        )

    def snapshot(self) -> GateSnapshot:
        undelivered = [ticket for ticket in self.tickets if ticket.undelivered_call_id is not None]
        return GateSnapshot(
            undelivered={ticket.undelivered_call_id: ticket.capture_key for ticket in undelivered},
            cuts=[
                EpisodeCut(capture_key=ticket.capture_key, record=ticket.cut_record, admitted_at=ticket.admitted_at)
                for ticket in undelivered
                if ticket.cut_record is not None
            ],
        )

    async def _cut_all(self, request: CheckpointRequest) -> None:
        by_backend: dict[str, list[_Ticket]] = {}
        for ticket in self.tickets:
            if (
                ticket.backend is not None
                and ticket.undelivered_call_id is not None
                and ticket.cut_restart_reason is None
            ):
                by_backend.setdefault(ticket.backend, []).append(ticket)
        await asyncio.gather(*(self._cut(request, backend, tickets) for backend, tickets in by_backend.items()))

    async def _cut(self, request: CheckpointRequest, backend: str, tickets: list[_Ticket]) -> None:
        prefixes = {}
        for ticket in tickets:
            episode_id = EpisodeId.from_capture_key(ticket.capture_key)
            prefixes[ticket.ticket_id] = GenerationCutPrefix(
                ticket_id=ticket.ticket_id,
                rollout_id=episode_id.rollout_id,
                attempt=episode_id.attempt,
                model_call_id=ticket.capture.model_call_id,
                admitted_at=ticket.admitted_at,
            )
        inventory = GenerationCutInventory.build(
            checkpoint_id=request.checkpoint_id, server_name=self.server_name, active_prefixes=list(prefixes.values())
        )
        with checkpoint_span("gym.checkpoint.generation_cut") as span:
            span.set(calls=len(prefixes))
            try:
                # Half of what remains: a cut is optional, and the stages after this one need time too.
                async with asyncio.timeout(max(0.0, request.deadline_ts - time.time()) / 2):
                    receipt = await self.cut_requester(backend, inventory)
                receipt.validate_for(inventory)
                acks = {ack.ticket_id: ack for ack in receipt.prefixes}
            except Exception:
                # A missing cut only costs regeneration after restore; never block the checkpoint on it.
                LOGGER.warning(
                    "generation cut failed on %s; those calls regenerate after restore", backend, exc_info=True
                )
                acks = {ticket_id: GenerationCutPrefixAck.failure(prefix) for ticket_id, prefix in prefixes.items()}
            dispositions = Counter(ack.disposition for ack in acks.values())
            span.set(**{f"cut_{disposition}": count for disposition, count in dispositions.items()})
        for disposition, count in dispositions.items():
            record_checkpoint_event("generation_cut", count, disposition=disposition)
        failed = sum(ack.disposition != "durable_prefix" for ack in acks.values())
        if failed:
            LOGGER.warning(
                "worker %s could not cut %d of %d calls; they regenerate after restore", backend, failed, len(acks)
            )
        for ticket in tickets:
            ticket.cut = acks[ticket.ticket_id]
            ticket.cut_record = _cut_record(ticket) if ticket.cut.disposition == "durable_prefix" else None


def merge_reports(reports: Iterable[GateReport]) -> PrepareReport:
    reports = list(reports)
    streaming = sorted({key for report in reports for key in report.streaming})
    return PrepareReport(
        ready=not streaming,
        blockers=streaming,
        counts={
            "inflight": sum(report.inflight for report in reports),
            "held": sum(report.held for report in reports),
            "cut": sum(report.cut for report in reports),
            "cut_failed": sum(report.cut_failed for report in reports),
            "cut_skipped": sum(report.cut_skipped for report in reports),
        },
    )


def export_model_records(
    ledger: CheckpointableLedger,
    episode_ids: Optional[list[EpisodeId]],
    snapshots: Iterable[GateSnapshot],
    restored_cuts: Mapping[str, GenerationCutRecord],
) -> list[ModelRecord]:
    """The rows and cuts of each continued episode, leaving out every undelivered call of every process."""
    if episode_ids is None:
        raise ControlError("model commit needs the episode_ids the controller continues from this checkpoint")
    snapshots = list(snapshots)
    undelivered = {call_id for snapshot in snapshots for call_id in snapshot.undelivered}
    # One cut per episode: when a client retried a call the server was still running,
    # both calls are undelivered and cut, but only the latest is still awaited and will be re-issued after a restore.
    latest: dict[str, EpisodeCut] = {}
    for snapshot in snapshots:
        for cut in snapshot.cuts:
            if cut.capture_key not in latest or cut.admitted_at > latest[cut.capture_key].admitted_at:
                latest[cut.capture_key] = cut
    cuts_by_key = {key: [cut.record] for key, cut in latest.items()}
    records = []
    for episode_id in episode_ids:
        key = episode_id.capture_key
        rows = [row for row in ledger.export_rows(key) if row.get("model_call_id") not in undelivered]
        cuts = list(cuts_by_key.get(key, []))
        # A restored cut that no re-issued call has consumed yet is still this episode's prefix,
        # unless a newer call of the episode was cut since.
        if key in restored_cuts and not cuts:
            cuts.append(restored_cuts[key])
        if rows or cuts:
            records.append(ModelRecord(episode_id=episode_id, rows=rows, generation_cuts=cuts))
    return records


def retained_staging_keys(records: Iterable[CheckpointRecord]) -> list[str]:
    """Every staged token-store key the checkpointed episodes still refer to.

    The training framework must keep these rows in its token store and may clear the rest.
    """
    keys: set[str] = set()
    for record in records:
        keys.update(row["staging_key"] for row in record.rows if row.get("staging_key"))
        for cut in record.generation_cuts:
            keys.update(cut.continuation.staging_keys)
    return sorted(keys)


def staging_keys_by_episode(records: Iterable[CheckpointRecord]) -> dict[str, list[str]]:
    """Each checkpointed episode's staged keys, keyed by its capture key.

    The controller keeps only the rows of episodes it continues,
    and a key's name does not always say whose it is: generation-cut keys name no rollout.
    A key can belong to more than one episode,
    since a restored attempt's record carries the rows of the attempt it continues.
    """
    return {record.episode_id.capture_key: retained_staging_keys([record]) for record in records}


async def retire_ledgers(ledger: Optional[CheckpointableLedger], episode_ids: Iterable[EpisodeId]) -> None:
    """Retire the ledgers of each episode and every earlier attempt of it, in one batch.

    A checkpoint retire stops these attempts, and a restore continues them as a later attempt,
    so nothing will write their ledgers again.
    """
    keys = [
        EpisodeId(rollout_id=episode_id.rollout_id, attempt=attempt).capture_key
        for episode_id in episode_ids
        for attempt in range(episode_id.attempt + 1)
    ]
    if ledger is not None and keys:
        await ledger.retire(keys)


def ledger_removal_refusal(app: Any) -> Optional[str]:
    """Why the capture ledger must not retire or delete now: a commit may still read any live episode's ledger."""
    gate = getattr(app.state, "nemo_gym_policy_gate", None)
    if gate is not None and not gate.accepting:
        return "a checkpoint is open on this model server; retry ledger retire and delete after it resumes"
    return None


def with_scope(
    records: list[CheckpointRecord], scope: list[EpisodeId], ledger: Optional[CheckpointableLedger]
) -> list[ModelRecord]:
    """``records`` plus an empty record for each in-scope episode without one,
    so the import also clears what dead executions left for those episodes' next attempts."""
    if ledger is None:
        return list(records)
    recorded = {record.episode_id for record in records}
    return [
        *records,
        *(ModelRecord(episode_id=episode_id, rows=[]) for episode_id in scope if episode_id not in recorded),
    ]


def import_model_records(
    ledger: Optional[CheckpointableLedger], records: list[ModelRecord]
) -> dict[str, GenerationCutRecord]:
    """Install every record's rows under its next attempt; return the restored cuts by capture key.

    Everything is validated before anything is written.
    The import replaces what dead executions left for a target attempt and later attempts, fences included:
    restoring a checkpoint again, after its replacement attempt made calls and Gym crashed,
    continues from the checkpoint's boundary.
    Each staged row keeps the capture key it was staged under,
    so a receipt of the next attempt verifies those calls against their own staged data.
    """
    if records and ledger is None:
        raise ControlError("checkpoint holds capture-ledger rows but this model server has no capture ledger")
    targets = [
        (record, next_attempt(record.episode_id).capture_key, _with_capture_key(record.rows, record.episode_id))
        for record in records
    ]
    for record, _, _ in targets:
        if len(record.generation_cuts) > 1:
            raise ControlError(f"episode {record.episode_id.capture_key} has more than one undelivered cut")
    import_many = getattr(ledger, "import_rows_many", None)
    try:
        if import_many is not None:
            import_many({target: rows for _, target, rows in targets})
        else:
            for _, target, rows in targets:
                ledger.import_rows(target, rows)
    except ValueError as error:
        raise ControlError(f"capture ledger refused the restore: {error}") from error
    return {target: record.generation_cuts[0] for record, target, _ in targets if record.generation_cuts}


def _with_capture_key(rows: list[dict[str, Any]], episode_id: EpisodeId) -> list[dict[str, Any]]:
    """Stamp each staged row with the capture key it was staged under.

    A row a restore already carried over keeps its stamp,
    so a chain of restores still names the attempt that staged the call.
    """
    return [
        {**row, "capture_key": episode_id.capture_key}
        if row.get("staging_key") is not None and row.get("capture_key") is None
        else row
        for row in rows
    ]


class PolicyModelParticipant(CheckpointParticipant):
    """The participant of a policy model server that runs in one process."""

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
        self._local_cuts = LocalRestoredCuts()
        self.gate = PolicyGate(
            server_name=server_name,
            cut_requester=cut_requester,
            retired=self.retired,
            restored_cuts=self._local_cuts,
            on_change=self.notify,
        )

    @property
    def _restored_cuts(self) -> dict[str, GenerationCutRecord]:
        return self._local_cuts.records

    async def close_admission(self, request: CheckpointRequest) -> None:
        await self.gate.close(request)

    async def open_admission(self) -> None:
        self.gate.open()

    def readiness(self) -> PrepareReport:
        return merge_reports([self.gate.report()])

    async def retire(self, episode_id: EpisodeId) -> None:
        await self.gate.retire(episode_id)
        for capture_key in [key for key in self._restored_cuts if covers(episode_id, key)]:
            del self._restored_cuts[capture_key]
        self.gate.restored_targets = {key for key in self.gate.restored_targets if not covers(episode_id, key)}
        await retire_ledgers(self.ledger, [episode_id])

    async def delete_restored(self, episode_id: EpisodeId) -> None:
        """Delete a restored attempt nothing started, without the fence a retire leaves:
        the controller may still start the rollout over as this attempt."""
        key = episode_id.capture_key
        self.gate.restored_targets.discard(key)
        self._restored_cuts.pop(key, None)
        if self.ledger is not None:
            await self.ledger.delete([key])

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        if self.ledger is None:
            return []
        return export_model_records(self.ledger, episode_ids, [self.gate.snapshot()], self._restored_cuts)

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        if self.ledger is None:
            return []
        # Snapshot on the event loop, then read every episode's ledger off it.
        snapshots, restored = [self.gate.snapshot()], dict(self._restored_cuts)
        return await asyncio.to_thread(export_model_records, self.ledger, episode_ids, snapshots, restored)

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        raise NotImplementedError("the policy participant restores through install(), which needs the restore scope")

    async def install(self, records: list[CheckpointRecord], scope: list[EpisodeId]) -> None:
        if self.gate.served:
            raise ControlError("model restore requires a freshly started model server; this one has served calls")
        self._restored_cuts.update(
            await asyncio.to_thread(import_model_records, self.ledger, with_scope(records, scope, self.ledger))
        )
        self.gate.restored_targets = {next_attempt(episode_id).capture_key for episode_id in scope}
        # The restored attempts continue as the next attempt, so their own ledgers are no longer used.
        await retire_ledgers(self.ledger, scope)

    async def restored_pending(self) -> list[EpisodeId]:
        return [EpisodeId.from_capture_key(key) for key in sorted(self.gate.restored_targets)]

    def commit_reply(self, records: list[CheckpointRecord]) -> dict[str, Any]:
        return {
            "staging_keys": retained_staging_keys(records),
            "staging_keys_by_episode": staging_keys_by_episode(records),
        }

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
            prefix_token_count=ack.prefix_token_count,
            prefix_digest=ack.prefix_digest,
            effective_output_limit=ack.effective_output_limit,
            terminal_finish_reason=ack.terminal_finish_reason,
            terminal_stop_reason=ack.terminal_stop_reason,
        ),
    )


def covers(retired: EpisodeId, capture_key: str) -> bool:
    """Whether retiring ``retired`` discards the attempt named by ``capture_key``."""
    other = EpisodeId.from_capture_key(capture_key)
    return other.rollout_id == retired.rollout_id and other.attempt <= retired.attempt


class PolicyAdmissionMiddleware:
    """Pure ASGI gate in front of generation routes."""

    def __init__(self, app: Any, gate: PolicyGate) -> None:
        self.app = app
        self.gate = gate

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not scope.get("path", "").endswith(GENERATION_ROUTES):
            await self.app(scope, receive, send)
            return
        match = _ROLLOUT_PREFIX.match(scope["path"])
        capture_key = match.group("capture_key") if match else None
        try:
            await self.gate.admit(capture_key)
        except ControlError as error:
            await error.response()(scope, receive, send)
            return

        ticket = self.gate.enter(capture_key or "")

        async def gated_send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                await self.gate.deliver_response(ticket)
            await send(message)

        token = _CURRENT_TICKET.set(ticket)
        try:
            await self.gate.restored_cuts.prefetch(ticket)
            await self.app(scope, receive, gated_send)
        finally:
            _CURRENT_TICKET.reset(token)
            await self.gate.exit(ticket)


def worker_control_root(base_url: str) -> str:
    """The root of the control routes a worker serves beside its OpenAI base URL."""
    return base_url.rstrip("/").removesuffix("/v1")


def generation_cut_requester(
    auth_token: str, *, control_root: Callable[[str], str] = worker_control_root
) -> CutRequester:
    """Request cuts from a worker's ``/ng-control/v1/generation-cut`` endpoint.

    ``backend`` is the worker's OpenAI-compatible base URL; ``control_root`` maps it to the root serving the route.
    """

    async def request_cut(backend: str, inventory: GenerationCutInventory) -> GenerationCutReceipt:
        from nemo_gym.server_utils import get_response_json, raise_for_status, request

        response = await request(
            method="POST",
            url=f"{control_root(backend)}{GENERATION_CUT_ROUTE}",
            json=inventory.model_dump(mode="json"),
            headers={"authorization": f"Bearer {auth_token}"},
            _internal=True,
            # The worker's data connections are busy with the very generations being cut.
            _control=True,
        )
        await raise_for_status(response)
        return GenerationCutReceipt.model_validate(await get_response_json(response))

    return request_cut
