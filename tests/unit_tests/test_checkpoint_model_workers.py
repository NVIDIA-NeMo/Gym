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

from nemo_gym._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ForgetRequest,
    RestoreRequest,
    RetireRequest,
    install_control_routes,
)
from nemo_gym._checkpoint.errors import ControlError, InvalidPhaseError, StaleAttemptError
from nemo_gym._checkpoint.model import GenerationCutRecord, PolicyGate, _Ticket
from nemo_gym._checkpoint.model_workers import PolicyCoordinator, PolicyWorkerLink
from nemo_gym.episode_types import EpisodeId
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.sink import CaptureContext
from nemo_gym.token_id_capture.staging.records import GenerationCutContinuation
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
        with pytest.raises(InvalidPhaseError, match="no longer ready"):
            await coordinator.controller.commit(CommitRequest(**control(checkpoint_dir="/tmp/unused", episode_ids=[])))
        await coordinator.controller.resume(CheckpointRequest(**control()))
        # uvicorn restarts the worker; the next checkpoint proceeds.
        await second.connect()
        prepared = await coordinator.controller.prepare(CheckpointRequest(**control("c2")))

    assert report.blockers == ["policy-workers-unreported:1", "policy-worker-lost"]
    assert prepared["phase"] == "prepared"


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


async def test_worker_messages_carry_what_the_checkpoint_writer_carries() -> None:
    import math

    from nemo_gym._checkpoint.model_workers import _read_frame, _write_frame

    class Sink:
        data = b""

        def write(self, data: bytes) -> None:
            self.data += data

        async def drain(self) -> None:
            pass

    sink, reader = Sink(), asyncio.StreamReader()
    # A -inf logprob and an integer beyond 64 bits, which orjson would turn into null or refuse.
    await _write_frame(sink, {"logprobs": [-math.inf, -0.5], "nan": math.nan, "hash": 2**70})
    reader.feed_data(sink.data)

    message = await _read_frame(reader)

    assert message["logprobs"] == [-math.inf, -0.5] and math.isnan(message["nan"]) and message["hash"] == 2**70


async def test_a_message_that_cannot_be_sent_or_is_never_answered_is_a_typed_unavailable_error() -> None:
    from nemo_gym._checkpoint.model_workers import CoordinatorUnavailableError, _Channel

    class StuckWriter:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            # The socket buffer is full and the peer never reads.
            await asyncio.Event().wait()

    async def handler(kind, body):
        return {}

    channel = _Channel(asyncio.StreamReader(), StuckWriter(), handler)
    with pytest.raises(CoordinatorUnavailableError):
        # Bounded from outside too, so a call that ignores its own timeout fails here instead of hanging.
        await asyncio.wait_for(channel.call("claim_cut", {"capture_key": "r-a1"}, timeout=0.1), 2)

    # The outer bound would raise TimeoutError instead, so the call stopped at its own 0.1 s timeout.


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


async def test_a_restored_attempt_any_worker_started_is_no_longer_pending(tmp_path: Path) -> None:
    async with deployment(2, ledger=FileLineageStore(tmp_path / "ledger")) as (coordinator, [first, second]):
        participant = coordinator.participant
        # A restore closes every worker, which report that they have served nothing, then installs.
        await participant.close_admission(CheckpointRequest(**control("r1")))
        await participant.install([], [EpisodeId(rollout_id="r"), EpisodeId(rollout_id="s")])
        await participant.open_admission()
        # The replacement of r starts on the second worker; s's replacement has not started anywhere.
        second.gate.enter("r-a1")

        pending = await participant.restored_pending()

    assert [episode_id.capture_key for episode_id in pending] == ["s-a1"]


async def test_a_reopen_does_not_make_a_started_restored_attempt_pending_again(tmp_path: Path) -> None:
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

        pending = await participant.restored_pending()

    assert [episode_id.capture_key for episode_id in pending] == ["s-a1"]
