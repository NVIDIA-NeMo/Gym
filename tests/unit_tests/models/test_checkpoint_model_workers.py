# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A multi-worker policy model server, as one coordinator and several worker gates over a Unix socket."""

import asyncio
import os
import shutil
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import pytest
from fastapi import FastAPI

from nemo_gym.core import EpisodeId
from nemo_gym.models._checkpoint import GenerationCutRecord, ModelRecord, PolicyGate, _Ticket
from nemo_gym.models._checkpoint_workers import PolicyCoordinator, PolicyWorkerLink
from nemo_gym.server_runtime._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ForgetRequest,
    RestoreRequest,
    RetireRequest,
    install_control_routes,
)
from nemo_gym.server_runtime._checkpoint.errors import ControlError, ExportFailedError, StaleAttemptError
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.sink import CaptureContext
from nemo_gym.token_id_capture.staging.records import GenerationCutContinuation
from tests.unit_tests.models.test_checkpoint_model import _staged_call, wait_until, watch_admission, watch_delivery
from tests.unit_tests.test_token_capture_ledger import ASSISTANT_1, ASSISTANT_2, USER_1, USER_2, _call_record, _commit


def control(checkpoint_id: str = "c1", *, timeout: float = 5, **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + timeout, **extra}


@asynccontextmanager
async def deployment(
    workers: int = 2, *, expected: Optional[int] = None, ledger: Optional[FileLineageStore] = None
) -> AsyncIterator[tuple[PolicyCoordinator, list[PolicyWorkerLink]]]:
    # AF_UNIX paths must be short, so the socket lives in its own directory under /tmp.
    directory = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
    socket_path = os.path.join(directory, "policy.sock")
    coordinator = PolicyCoordinator(
        ledger,
        expected_workers=expected or workers,
        instance_name="policy",
        lease_grace_seconds=60,
        socket_path=socket_path,
    )
    server = await coordinator.serve()
    links = [worker_link(socket_path) for _ in range(workers)]
    for link in links:
        await link.connect()
    try:
        yield coordinator, links
    finally:
        for link in links:
            await link.disconnect()
        server.close()
        await server.wait_closed()
        shutil.rmtree(directory, ignore_errors=True)


def worker_link(socket_path: str, lost: Optional[list[str]] = None) -> PolicyWorkerLink:
    # The real callback terminates the worker process, which here would be the test runner.
    return PolicyWorkerLink(
        socket_path=socket_path,
        server_name="policy",
        cut_requester=None,
        on_coordinator_lost=lambda: (lost if lost is not None else []).append("lost"),
    )


def held_call(gate: PolicyGate, capture_key: str, model_call_id: str, request_items: list) -> _Ticket:
    ticket = gate.enter(capture_key)
    ticket.capture = CaptureContext(
        rollout_id=capture_key, model_call_id=model_call_id, token_sink=None, request_items=request_items
    )
    return ticket


def restored_cut(model_call_id: str = "c2", digest: str = "e" * 64) -> GenerationCutRecord:
    continuation = GenerationCutContinuation(
        source_capture_key="r",
        source_model_call_id=model_call_id,
        staging_keys=(f"__generation_cut__/{model_call_id}",),
        prefix_token_count=3,
        prefix_digest="d" * 64,
        effective_output_limit=10,
    )
    return GenerationCutRecord(model_call_id=model_call_id, request_digest=digest, continuation=continuation)


async def test_prepare_closes_every_worker_and_waits_for_all_of_them() -> None:
    async with deployment() as (coordinator, (first, second)):
        streaming = second.gate.enter("r")
        streaming.response_started = True
        missed = await coordinator.controller.prepare(CheckpointRequest(**control(timeout=0.3)))
        closed = [first.gate.accepting, second.gate.accepting]
        await second.gate.exit(streaming)
        prepared = await coordinator.controller.prepare(CheckpointRequest(**control()))
        await coordinator.controller.resume(CheckpointRequest(**control()))
        reopened = [first.gate.accepting, second.gate.accepting]

    assert closed == [False, False]
    assert missed["report"]["blockers"] == ["r"]
    assert prepared["phase"] == "prepared"
    assert reopened == [True, True]


async def test_commit_leaves_out_undelivered_calls_on_every_worker(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "ledger")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",)))
    await ledger.record(
        _commit(
            _call_record("c2"), [USER_1, ASSISTANT_1, USER_2], [ASSISTANT_2], rollout_id="r", staging_chain=("r/c2",)
        )
    )
    async with deployment(ledger=ledger) as (coordinator, (_, second)):
        # The response to c2 is still on the second worker: the agent never received it.
        held_call(second.gate, "r", "c2", [USER_1, ASSISTANT_1, USER_2])
        await coordinator.controller.prepare(CheckpointRequest(**control()))
        await coordinator.controller.commit(
            CommitRequest(**control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}]))
        )
        [record] = await coordinator.participant.export([EpisodeId(rollout_id="r")])

    assert [row["model_call_id"] for row in record.rows] == ["c1"]


async def test_a_restored_cut_is_claimed_by_exactly_one_worker_and_returned_if_unclaimed() -> None:
    async with deployment() as (coordinator, (first, second)):
        coordinator.participant.restored_cuts["r-a1"] = restored_cut()
        await coordinator.controller.prepare(CheckpointRequest(**control()))
        await coordinator.controller.resume(CheckpointRequest(**control()))
        # Both workers now know a restored cut exists for r-a1; each re-issued call tries to claim it.
        mine, theirs = first.gate.enter("r-a1"), second.gate.enter("r-a1")
        await first.restored_cuts.prefetch(mine)
        await second.restored_cuts.prefetch(theirs)
        claimed = [mine.claimed_cut is not None, theirs.claimed_cut is not None]
        # The claiming call turns out to be a different request, so the cut goes back.
        unused = first.restored_cuts.take(mine, "r-a1", "f" * 64)
        await first.gate.exit(mine)
        await second.gate.exit(theirs)
        released = dict(coordinator.participant.restored_cuts)

    assert claimed == [True, False]
    assert unused is None
    assert released == {"r-a1": restored_cut()}


async def test_a_worker_lost_while_closed_blocks_the_checkpoint_until_resume() -> None:
    async with deployment() as (coordinator, (first, second)):
        await coordinator.controller.prepare(CheckpointRequest(**control()))
        # Its undelivered calls are unknown now, so their rows could be exported as delivered.
        await second.disconnect()
        report = coordinator.participant.readiness()
        # Final for this checkpoint: retrying later could export a later cut.
        with pytest.raises(ExportFailedError, match="no longer ready"):
            await coordinator.controller.commit(CommitRequest(**control(checkpoint_dir="/tmp/unused", episode_ids=[])))
        await coordinator.controller.resume(CheckpointRequest(**control()))
        # uvicorn restarts the worker; the next checkpoint proceeds.
        await second.connect()
        prepared = await coordinator.controller.prepare(CheckpointRequest(**control("c2")))

    assert report.blockers == ["policy-workers-unreported:1", "policy-worker-lost"]
    assert prepared["phase"] == "prepared"


async def test_a_worker_lost_between_readiness_and_snapshot_fails_the_commit(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "ledger")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",)))
    await ledger.record(
        _commit(
            _call_record("c2"), [USER_1, ASSISTANT_1, USER_2], [ASSISTANT_2], rollout_id="r", staging_chain=("r/c2",)
        )
    )
    async with deployment(ledger=ledger) as (coordinator, (_, second)):
        participant = coordinator.participant
        # The response to c2 is still on the second worker: the agent never received it.
        held_call(second.gate, "r", "c2", [USER_1, ASSISTANT_1, USER_2])
        await coordinator.controller.prepare(CheckpointRequest(**control()))
        export = participant.export

        async def export_after_the_worker_left(episode_ids: Optional[list[EpisodeId]]) -> list:
            # The commit has checked readiness; the worker leaves before the snapshot.
            await second.disconnect()
            async with asyncio.timeout(5):
                while len(participant.workers) > 1:
                    await asyncio.sleep(0.01)
            return await export(episode_ids)

        participant.export = export_after_the_worker_left
        commit = CommitRequest(**control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}]))
        with pytest.raises(ExportFailedError, match="worker"):
            await coordinator.controller.commit(commit)
        # Final for this checkpoint: retrying later could export a later cut.
        with pytest.raises(ExportFailedError):
            await coordinator.controller.commit(commit)

    assert not (tmp_path / "ckpt").exists()


async def test_a_worker_that_joins_during_a_checkpoint_closes_at_once() -> None:
    async with deployment(workers=1, expected=2) as (coordinator, (first,)):
        missed = await coordinator.controller.prepare(CheckpointRequest(**control(timeout=0.2)))
        late = worker_link(coordinator.socket_path)
        await late.connect()
        prepared = await coordinator.controller.prepare(CheckpointRequest(**control()))
        closed = late.gate.accepting is False
        await late.disconnect()

    assert missed["report"]["blockers"] == ["policy-workers-unreported:1"]
    assert closed
    assert prepared["phase"] == "prepared"


async def test_a_retired_attempt_is_refused_on_every_worker_and_one_that_joins_later_until_forget() -> None:
    async with deployment() as (coordinator, (first, second)):
        await coordinator.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r"}])))
        late = worker_link(coordinator.socket_path)
        await late.connect()
        refused = []
        for gate in (first.gate, second.gate, late.gate):
            with pytest.raises(StaleAttemptError):
                await gate.admit("r")
            refused.append(True)
        await coordinator.controller.forget(ForgetRequest(**control(rollout_ids=["r"])))
        for gate in (first.gate, second.gate, late.gate):
            await gate.admit("r")
        remaining = [len(link.retired) for link in (first, second, late)] + [len(coordinator.participant.retired)]
        await late.disconnect()

    assert refused == [True, True, True]
    assert remaining == [0, 0, 0, 0]


async def test_a_call_waiting_at_closed_admission_on_a_worker_stays_refused_after_a_cut_short_retire_and_forget() -> (
    None
):
    async with deployment() as (coordinator, (first, _second)):
        participant = coordinator.participant
        await participant.close_admission(CheckpointRequest(**control()))
        reached = watch_admission(first.gate)
        waiting = asyncio.create_task(first.gate.admit("r"))
        await asyncio.wait_for(reached.wait(), timeout=5)
        late = {**control("retire", episode_ids=[{"rollout_id": "r"}]), "deadline_ts": time.time() - 1}
        with pytest.raises(ControlError):
            await coordinator.controller.retire(RetireRequest(**late))
        await coordinator.controller.forget(ForgetRequest(**control("forget", rollout_ids=["r"])))
        await participant.open_admission()

        with pytest.raises(StaleAttemptError):
            await asyncio.wait_for(waiting, timeout=5)
        # The waiter left the gate's tracking with its refusal.
        assert not first.gate.waiters


async def test_a_worker_call_that_used_a_restored_cut_never_gives_it_back_to_a_retry_after_a_newer_cut(
    tmp_path: Path,
) -> None:
    from tests.unit_tests.models.test_checkpoint_model import _cutting_worker

    request_cut, _ = _cutting_worker()
    async with deployment(1, ledger=FileLineageStore(tmp_path / "ledger")) as (coordinator, _):
        cutting = PolicyWorkerLink(
            socket_path=coordinator.socket_path,
            server_name="policy",
            cut_requester=request_cut,
            on_coordinator_lost=list,
        )
        await cutting.connect()
        try:
            participant = coordinator.participant
            participant.restored_cuts["r-a1"] = restored_cut()
            await participant.close_admission(CheckpointRequest(**control("r1")))
            await participant.open_admission()
            resumed = held_call(cutting.gate, "r-a1", "c3", [])
            resumed.backend = "http://worker-0/v1"
            await cutting.restored_cuts.prefetch(resumed)
            continued = cutting.restored_cuts.take(resumed, "r-a1", "e" * 64)
            # The next checkpoint cuts the resumed call further; then the call ends before its response starts.
            await participant.close_admission(CheckpointRequest(**control("c2")))
            [exported] = await participant.export([EpisodeId(rollout_id="r", attempt=1)])
            await participant.open_admission()
            await cutting.gate.exit(resumed)
            retry = held_call(cutting.gate, "r-a1", "c4", [])
            await cutting.restored_cuts.prefetch(retry)
            received = cutting.restored_cuts.take(retry, "r-a1", "e" * 64)
        finally:
            await cutting.disconnect()

    assert continued == restored_cut()
    assert [cut.model_call_id for cut in exported.generation_cuts] == ["c3"]
    # The retry regenerates: it neither continues the stale restored cut nor holds one the commit dropped.
    assert received is None and "r-a1" not in participant.restored_cuts


async def test_worker_control_routes_forward_to_the_coordinator() -> None:
    async with deployment() as (_, (first, _second)):
        app = FastAPI()
        install_control_routes(app, first.dispatch, auth_token="t")
        headers = {"authorization": "Bearer t"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://w") as client:
            status = (await client.get("/ng-control/v1/checkpoint/status", headers=headers)).json()
            prepared = (await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=headers)).json()
            await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=headers)

    unconnected = worker_link("/tmp/ngc-missing.sock")
    app = FastAPI()
    install_control_routes(app, unconnected.dispatch, auth_token="t")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://w") as client:
        unavailable = await client.get("/ng-control/v1/checkpoint/status", headers={"authorization": "Bearer t"})

    assert (status["kind"], status["workers"]) == ("model", 2)
    assert prepared["phase"] == "prepared"
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "checkpoint_coordinator_unavailable"


async def test_a_worker_shuts_down_when_its_coordinator_is_gone() -> None:
    lost: list[str] = []
    async with deployment(workers=0, expected=1) as (coordinator, _):
        link = worker_link(coordinator.socket_path, lost)
        await link.connect()
        # The main process dies: the coordinator's end of every connection closes.
        coordinator._server.close()
        for worker in list(coordinator.participant.workers.values()):
            worker.channel.close()
        async with asyncio.timeout(5):
            while not lost:
                await asyncio.sleep(0.01)

    assert lost == ["lost"]


async def test_restore_across_workers_imports_once_and_offers_the_cut_to_every_worker(
    tmp_path: Path,
) -> None:
    ledger = FileLineageStore(tmp_path / "ledger")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",)))
    async with deployment(ledger=ledger) as (coordinator, _):
        coordinator.participant.restored_cuts["r"] = restored_cut()
        await coordinator.controller.prepare(CheckpointRequest(**control()))
        await coordinator.controller.commit(
            CommitRequest(**control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}]))
        )

    fresh = FileLineageStore(tmp_path / "fresh")
    async with deployment(ledger=fresh) as (coordinator, (first, second)):
        await coordinator.controller.restore(
            RestoreRequest(**control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}]))
        )
        closed = [first.gate.accepting, second.gate.accepting]
        await coordinator.controller.resume(CheckpointRequest(**control("r1")))
        offered = [first.restored_cuts.keys, second.restored_cuts.keys]

    assert closed == [False, False]
    assert [row["model_call_id"] for row in fresh.export_rows("r-a1")] == ["c1"]
    assert offered == [{"r-a1"}, {"r-a1"}]


async def test_restore_refuses_workers_that_have_served_calls(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "ledger")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",)))
    async with deployment(ledger=ledger) as (coordinator, _):
        await coordinator.controller.prepare(CheckpointRequest(**control()))
        await coordinator.controller.commit(
            CommitRequest(**control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}]))
        )

    fresh = FileLineageStore(tmp_path / "fresh")
    async with deployment(ledger=fresh) as (coordinator, (first, _)):
        # A ledger a worker wrote may belong to a live episode, which the import would replace.
        await first.gate.exit(first.gate.enter("r-a1"))
        with pytest.raises(ControlError, match="has served calls"):
            await coordinator.controller.restore(
                RestoreRequest(
                    **control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])
                )
            )

    assert fresh.export_rows("r-a1") == []


async def test_a_message_that_cannot_be_sent_or_is_never_answered_is_a_typed_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_gym.server_runtime._checkpoint import workers
    from nemo_gym.server_runtime._checkpoint.workers import Channel, CoordinatorUnavailableError

    # The call also cancels its handler on the other side, and gives up on a side that never answers.
    monkeypatch.setattr(workers, "CANCEL_GRACE_SECONDS", 0.2)

    class StuckWriter:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            # The socket buffer is full and the peer never reads.
            await asyncio.Event().wait()

    async def handler(kind, body):
        return {}

    channel = Channel(asyncio.StreamReader(), StuckWriter(), handler)
    with pytest.raises(CoordinatorUnavailableError):
        # Bounded from outside too, so a call that ignores its own timeout fails here instead of hanging.
        await asyncio.wait_for(channel.call("claim_cut", {"capture_key": "r-a1"}, timeout=0.1), 2)

    # The outer bound would raise TimeoutError instead, so the call stopped at its own timeout and the cancel's.


async def test_restore_is_refused_after_a_worker_left_even_if_its_replacement_is_fresh(tmp_path: Path) -> None:
    directory = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
    socket_path = os.path.join(directory, "policy.sock")
    coordinator = PolicyCoordinator(
        FileLineageStore(tmp_path / "ledger"),
        expected_workers=1,
        instance_name="policy",
        lease_grace_seconds=60,
        socket_path=socket_path,
    )
    server = await coordinator.serve()
    fresh = worker_link(socket_path)
    try:
        served = worker_link(socket_path)
        await served.connect()
        served.gate.enter("r")
        # uvicorn replaces the worker that served calls with a fresh one.
        await served.disconnect()
        # The coordinator notices the closed connection on its own schedule.
        async with asyncio.timeout(5):
            while coordinator.participant.workers:
                await asyncio.sleep(0.01)
        await fresh.connect()
        with pytest.raises(ControlError, match="has left"):
            coordinator.participant._check_restorable()
    finally:
        await fresh.disconnect()
        server.close()
        await server.wait_closed()
        shutil.rmtree(directory, ignore_errors=True)


async def test_a_restored_attempt_any_worker_started_is_no_longer_unclaimed(tmp_path: Path) -> None:
    async with deployment(2, ledger=FileLineageStore(tmp_path / "ledger")) as (coordinator, [first, second]):
        participant = coordinator.participant
        # A restore closes every worker, which report that they have served nothing, then installs.
        await participant.close_admission(CheckpointRequest(**control("r1")))
        await participant.install([], [EpisodeId(rollout_id="r"), EpisodeId(rollout_id="s")])
        await participant.open_admission()
        # The replacement of r starts on the second worker; s's replacement has not started anywhere.
        second.gate.enter("r-a1")

        pending = await participant.unclaimed()

    assert [episode_id.capture_key for episode_id in pending] == ["s-a1"]


async def test_a_reopen_does_not_make_a_started_restored_attempt_unclaimed_again(tmp_path: Path) -> None:
    async with deployment(2, ledger=FileLineageStore(tmp_path / "ledger")) as (coordinator, [first, second]):
        participant = coordinator.participant
        await participant.close_admission(CheckpointRequest(**control("r1")))
        await participant.install([], [EpisodeId(rollout_id="r"), EpisodeId(rollout_id="s")])
        await participant.open_admission()
        # The replacement of r starts on the second worker.
        second.gate.enter("r-a1")
        # A checkpoint closes and is resumed without a commit, so nothing asked the workers in between.
        await participant.close_admission(CheckpointRequest(**control("c1")))
        await participant.open_admission()

        pending = await participant.unclaimed()

    assert [episode_id.capture_key for episode_id in pending] == ["s-a1"]


async def test_after_a_restore_every_worker_refuses_the_replaced_attempt_even_one_that_joins_later(
    tmp_path: Path,
) -> None:
    from nemo_gym.server_runtime._checkpoint.store import write_participant_state

    write_participant_state(tmp_path / "ckpt", kind="model", instance="policy", checkpoint_id="c0", records=[])
    async with deployment(ledger=FileLineageStore(tmp_path / "ledger")) as (coordinator, links):
        await coordinator.controller.restore(
            RestoreRequest(**control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}]))
        )
        await coordinator.controller.resume(CheckpointRequest(**control("r1")))
        joined = worker_link(coordinator.socket_path)
        await joined.connect()
        try:
            for link in [*links, joined]:
                with pytest.raises(StaleAttemptError):
                    await link.gate.admit("r")
                await link.gate.admit("r-a1")
        finally:
            await joined.disconnect()


async def test_a_failed_restored_ledger_delete_keeps_the_attempt_pending_for_the_retry(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "ledger")
    async with deployment(2, ledger=ledger) as (coordinator, _):
        participant = coordinator.participant
        await participant.close_admission(CheckpointRequest(**control("r1")))
        await participant.install(
            [ModelRecord(episode_id=EpisodeId(rollout_id="r"), rows=[{"model_call_id": "c1"}])],
            [EpisodeId(rollout_id="r")],
        )
        await participant.open_admission()
        remove = ledger._remove
        failures = [OSError("ledger unavailable")]

        def failing_once(rollout_ids: list[str], **options: bool) -> dict:
            if failures:
                raise failures.pop()
            return remove(rollout_ids, **options)

        ledger._remove = failing_once
        with pytest.raises(OSError):
            await participant.delete_restored(EpisodeId(rollout_id="r", attempt=1))
        pending = await participant.unclaimed()
        await participant.delete_restored(EpisodeId(rollout_id="r", attempt=1))

    assert [episode_id.capture_key for episode_id in pending] == ["r-a1"]
    assert ledger.export_rows("r-a1") == []


async def test_a_worker_marks_a_restored_attempt_started_in_the_shared_ledger(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "ledger")
    async with deployment(1, ledger=ledger) as (coordinator, _):
        participant = coordinator.participant
        await participant.close_admission(CheckpointRequest(**control("r1")))
        await participant.install(
            [ModelRecord(episode_id=EpisodeId(rollout_id="r"), rows=[{"model_call_id": "c1"}])],
            [EpisodeId(rollout_id="r")],
        )
        await participant.open_admission()
        # A worker of another policy server capturing into the same ledger starts the attempt.
        other = PolicyWorkerLink(
            socket_path=coordinator.socket_path,
            server_name="other",
            cut_requester=None,
            ledger=FileLineageStore(tmp_path / "ledger"),
        )
        other.gate.restored_targets = {"r-a1"}
        await other.gate.admit("r-a1")
        other.gate.enter("r-a1")
        await participant.delete_restored(EpisodeId(rollout_id="r", attempt=1))

    assert ledger.export_rows("r-a1") == [{"model_call_id": "c1"}]


async def test_each_worker_publishes_a_call_s_ledger_row_only_once_its_response_starts(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "ledger")
    async with deployment(ledger=ledger) as (coordinator, (first, second)):
        delivered_sent, delivered = [], []
        await asyncio.wait_for(
            _staged_call(first.gate, FileLineageStore(tmp_path / "ledger"), delivered_sent, contexts=delivered),
            timeout=5,
        )
        held_sent, held, answer = [], [], asyncio.Event()
        delivering = watch_delivery(second.gate)
        call = _staged_call(
            second.gate, FileLineageStore(tmp_path / "ledger"), held_sent, contexts=held, answer=answer
        )
        await wait_until(lambda: len(held_sent) == 2)
        await coordinator.controller.prepare(CheckpointRequest(**control()))
        answer.set()
        # The second worker holds the completed response.
        await asyncio.wait_for(delivering.wait(), timeout=5)
        rows_while_held = [row["model_call_id"] for row in ledger.export_rows("r")]
        await coordinator.controller.commit(
            CommitRequest(**control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}]))
        )
        [record] = await coordinator.participant.export([EpisodeId(rollout_id="r")])
        await coordinator.controller.resume(CheckpointRequest(**control()))
        await asyncio.wait_for(call, timeout=5)

    assert rows_while_held == [delivered[0].model_call_id]
    assert [row["model_call_id"] for row in record.rows] == [delivered[0].model_call_id]
    assert [row["model_call_id"] for row in ledger.export_rows("r")] == [
        delivered[0].model_call_id,
        held[0].model_call_id,
    ]


async def crash(link: PolicyWorkerLink) -> None:
    """Drop the worker's connection the way a killed process does: without telling the coordinator."""
    link._disconnecting = True
    link._channel.close()
    await asyncio.gather(link._reader_task, return_exceptions=True)


async def wait_for_workers(coordinator: PolicyCoordinator, count: int) -> None:
    async with asyncio.timeout(5):
        while len(coordinator.participant.workers) != count:
            await asyncio.sleep(0.01)


def worker_logs(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    return [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if record.name == "nemo_gym.server_runtime._checkpoint.workers" and "policy worker" in record.getMessage()
    ]


async def test_workers_are_logged_by_pid_as_they_join_and_a_lost_one_as_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="nemo_gym.server_runtime._checkpoint.workers")
    async with deployment() as (coordinator, (_, second)):
        await crash(second)
        await wait_for_workers(coordinator, 1)
        # uvicorn starts a replacement.
        replacement = worker_link(coordinator.socket_path)
        await replacement.connect()
        try:
            await wait_for_workers(coordinator, 2)
        finally:
            await replacement.disconnect()

    pid = os.getpid()
    assert worker_logs(caplog)[:4] == [
        ("INFO", f"policy worker 1 (pid {pid}) joined; 1 of 2 workers connected"),
        ("INFO", f"policy worker 2 (pid {pid}) joined; 2 of 2 workers connected"),
        ("WARNING", f"policy worker 2 (pid {pid}) was lost; 1 of 2 workers connected; no checkpoint was open"),
        ("INFO", f"policy worker 3 (pid {pid}) joined; 2 of 2 workers connected"),
    ]


async def test_a_worker_lost_during_a_checkpoint_is_logged_as_blocking_its_commit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="nemo_gym.server_runtime._checkpoint.workers")
    async with deployment() as (coordinator, (_, second)):
        await coordinator.controller.prepare(CheckpointRequest(**control()))
        await crash(second)
        await wait_for_workers(coordinator, 1)
        await coordinator.controller.resume(CheckpointRequest(**control()))

    lost = f"policy worker 2 (pid {os.getpid()}) was lost; 1 of 2 workers connected; the open checkpoint cannot commit"
    assert ("WARNING", lost) in worker_logs(caplog)


async def test_a_worker_that_stops_cleanly_is_not_reported_as_lost(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="nemo_gym.server_runtime._checkpoint.workers")
    async with deployment() as (coordinator, (_, second)):
        await second.disconnect()
        await wait_for_workers(coordinator, 1)
        # Still a departure: what the worker served is unknown to a later restore.
        assert coordinator.participant.worker_left

    assert ("INFO", f"policy worker 2 (pid {os.getpid()}) stopped") in worker_logs(caplog)
    assert not [entry for entry in worker_logs(caplog) if entry[0] == "WARNING"]
