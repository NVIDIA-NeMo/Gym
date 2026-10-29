# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from nemo_gym._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ParticipantControlPlane,
    RestoreRequest,
    RetireRequest,
    install_participant,
)
from nemo_gym._checkpoint.errors import ControlError
from nemo_gym._checkpoint.generation_cut import GenerationCutInventory, GenerationCutPrefixAck, GenerationCutReceipt
from nemo_gym._checkpoint.model import (
    ModelRecord,
    PolicyAdmissionMiddleware,
    PolicyGate,
    PolicyModelParticipant,
    attach_capture_context,
    import_model_records,
    ledger_removal_refusal,
    note_generation_backend,
)
from nemo_gym.base_responses_api_model import _CaptureMiddleware
from nemo_gym.chat_streaming import SSE_KEEPALIVE
from nemo_gym.core import EpisodeId, current_rollout_id
from nemo_gym.responses_streaming import synthesize_responses_keepalive_sse
from nemo_gym.rollout_correlation import RolloutContextMiddleware
from nemo_gym.token_id_capture.control_routes import install_rollout_control_routes
from nemo_gym.token_id_capture.external_capture import VLLMWorkerCaptureHandler
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.protocols import CaptureAdmissionError, RolloutRetiredError
from nemo_gym.token_id_capture.records import ParentResolutionStatus
from nemo_gym.token_id_capture.sink import NG_COMMIT_COORDS_FIELD, CaptureContext, current_capture_context
from nemo_gym.token_id_capture.staging.records import (
    INVALID_COMMIT_COORDS_REASON,
    CaptureAdmission,
    CaptureLedgerCommit,
    CommitCoords,
    GenerationCutContinuation,
    RolloutManifest,
)
from tests.unit_tests.test_token_capture_ledger import (
    ASSISTANT_1,
    ASSISTANT_2,
    USER_1,
    USER_2,
    USER_3,
    _call_record,
    _commit,
)


AUTH = {"authorization": "Bearer t"}


def make_app() -> tuple[FastAPI, PolicyModelParticipant, asyncio.Event]:
    app = FastAPI()
    app.state.generation_done = asyncio.Event()
    participant = PolicyModelParticipant()
    release = asyncio.Event()

    async def stream():
        yield b"first "
        await release.wait()
        yield b"last"

    @app.post("/v1/chat/completions")
    async def chat() -> StreamingResponse:
        return StreamingResponse(stream())

    @app.post("/v1/responses")
    async def responses() -> dict:
        # A non-streaming generation: nothing is sent until it finishes.
        await app.state.generation_done.wait()
        return {"output": "done"}

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    install_participant(app, participant, auth_token="t", instance_name="policy", lease_grace_seconds=60)
    # Stands in for the model server's capture middleware, which strips the rollout prefix.
    app.add_middleware(RolloutContextMiddleware)
    app.add_middleware(PolicyAdmissionMiddleware, gate=participant.gate)
    return app, participant, release


def control(checkpoint_id: str = "c1", **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


async def wait_until(predicate: Callable[[], bool], timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def watch_admission(gate: PolicyGate) -> asyncio.Event:
    """Set once a call reaches admission, where a call waits while a checkpoint is open."""
    reached = asyncio.Event()
    admit = gate.admit

    async def admitting(capture_key: Optional[str]) -> None:
        reached.set()
        await admit(capture_key)

    gate.admit = admitting
    return reached


async def test_started_stream_drains_before_prepare_is_ready_and_new_calls_wait_for_resume() -> None:
    app, participant, release = make_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        stream = asyncio.create_task(client.post("/ng-rollout/r-a1/v1/chat/completions"))
        await wait_until(lambda: participant.readiness().blockers == ["r-a1"])
        prepare = asyncio.create_task(client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH))
        await wait_until(lambda: not participant.gate.accepting)
        await asyncio.sleep(0.05)
        assert not prepare.done()
        assert participant.readiness().blockers == ["r-a1"]

        reached = watch_admission(participant.gate)
        waiting = asyncio.create_task(client.post("/ng-rollout/other/v1/responses"))
        health = await client.get("/health")
        release.set()
        streamed = await stream
        prepared = (await prepare).json()
        await asyncio.wait_for(reached.wait(), timeout=5)
        await asyncio.sleep(0.05)
        # A waiting call is not admitted: it is no ticket, so a commit holds nothing of it.
        waited = not waiting.done() and len(participant.gate.tickets) == 0
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        app.state.generation_done.set()
        served = await asyncio.wait_for(waiting, timeout=5)
        reopened = await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)

    assert waited
    assert served.status_code == 200
    assert health.status_code == 200
    assert streamed.content == b"first last"
    assert prepared["phase"] == "prepared" and prepared["report"]["counts"]["inflight"] == 0
    assert reopened.json()["phase"] == "idle" and participant.gate.accepting


@pytest.mark.parametrize(
    "content_type", [b"application/json", b"text/event-stream; charset=utf-8", b"TEXT/EVENT-STREAM"]
)
async def test_headers_and_empty_bodies_pass_without_starting_an_answer(content_type: bytes) -> None:
    participant = PolicyModelParticipant()
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    headers_sent, answer_ready, answer_attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"Content-Type", content_type)]})
        await send({"type": "http.response.body", "body": b"", "more_body": True})
        headers_sent.set()
        await answer_ready.wait()
        answer_attempted.set()
        await send({"type": "http.response.body", "body": b"answer", "more_body": False})

    async def send(message):
        sent.append(message)

    call = asyncio.create_task(
        PolicyAdmissionMiddleware(app, participant.gate)(
            {"type": "http", "path": "/ng-rollout/r/v1/chat/completions"}, None, send
        )
    )
    try:
        await asyncio.wait_for(headers_sent.wait(), timeout=5)
        assert [message["type"] for message in sent] == ["http.response.start", "http.response.body"]
        prepared = await asyncio.wait_for(controller.prepare(CheckpointRequest(**control())), timeout=5)
        assert prepared["phase"] == "prepared"
        assert prepared["report"]["counts"]["held"] == 1
        answer_ready.set()
        await asyncio.wait_for(answer_attempted.wait(), timeout=5)
        assert not call.done()
        assert len(sent) == 2
        assert participant.readiness().ready
        await controller.resume(CheckpointRequest(**control()))
        await asyncio.wait_for(call, timeout=5)
        assert sent[-1]["body"] == b"answer"
    finally:
        await controller.resume(CheckpointRequest(**control()))
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)


async def test_keepalive_stream_is_cut_and_holds_model_output_until_resume(tmp_path: Path) -> None:
    """An SSE connection opened before backend completion must not deadlock checkpoint preparation."""
    ledger = FileLineageStore(tmp_path / "ledger")
    request_cut, requests = _cutting_worker()
    participant = PolicyModelParticipant(ledger, server_name="policy", cut_requester=request_cut)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    backend_done, keepalive_sent, answer_attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sent = []

    async def app(scope, receive, send):
        attach_capture_context(
            CaptureContext(rollout_id="r", model_call_id="c1", token_sink=None, request_items=[USER_1])
        )
        note_generation_backend("http://worker-0/v1")
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
        )
        await send({"type": "http.response.body", "body": b": keep-alive\n\n", "more_body": True})
        keepalive_sent.set()
        await backend_done.wait()
        # Liveness comments must still pass while the checkpoint holds the actual answer.
        await send({"type": "http.response.body", "body": b": keep-alive\n\n", "more_body": True})
        answer_attempted.set()
        await send({"type": "http.response.body", "body": b'data: {"answer":"done"}\n\n', "more_body": True})
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})

    async def send(message):
        sent.append(message)

    call = asyncio.create_task(
        PolicyAdmissionMiddleware(app, participant.gate)(
            {"type": "http", "path": "/ng-rollout/r/v1/chat/completions"}, None, send
        )
    )
    try:
        await asyncio.wait_for(keepalive_sent.wait(), timeout=1)
        prepared = await asyncio.wait_for(controller.prepare(CheckpointRequest(**control())), timeout=1)
        assert prepared["phase"] == "prepared"
        assert prepared["report"]["counts"] == {"inflight": 1, "held": 1, "cut": 1, "cut_failed": 0, "cut_skipped": 0}
        assert requests[0][1].active_prefixes[0].model_call_id == "c1"
        backend_done.set()
        await asyncio.wait_for(answer_attempted.wait(), timeout=1)
        assert [m.get("body") for m in sent] == [None, b": keep-alive\n\n", b": keep-alive\n\n"]
        assert participant.readiness().ready
        commit = await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
        assert commit["manifest"]["record_count"] == 1
        records = (tmp_path / "ckpt/gym/model/policy" / commit["manifest"]["records_file"]).read_text()
        assert json.loads(records.splitlines()[0])["generation_cuts"][0]["continuation"]["prefix_token_count"] == 3
        await controller.resume(CheckpointRequest(**control()))
        await asyncio.wait_for(call, timeout=1)
        assert sent[-2]["body"] == b'data: {"answer":"done"}\n\n'
        assert sent[-1]["body"] == b"data: [DONE]\n\n"
        assert not participant.gate.tickets
    finally:
        await controller.resume(CheckpointRequest(**control()))
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)


async def test_started_sse_model_output_finishes_while_admission_is_closed() -> None:
    participant = PolicyModelParticipant()
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    first_sent, finish = asyncio.Event(), asyncio.Event()
    sent = []

    async def app(scope, receive, send):
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
        )
        await send({"type": "http.response.body", "body": b"data: first\n\n", "more_body": True})
        first_sent.set()
        await finish.wait()
        await send({"type": "http.response.body", "body": b"data: last\n\n", "more_body": False})

    async def send(message):
        sent.append(message)

    call = asyncio.create_task(
        PolicyAdmissionMiddleware(app, participant.gate)(
            {"type": "http", "path": "/ng-rollout/r/v1/chat/completions"}, None, send
        )
    )
    try:
        await asyncio.wait_for(first_sent.wait(), timeout=1)
        prepare = asyncio.create_task(controller.prepare(CheckpointRequest(**control())))
        await wait_until(lambda: not participant.gate.accepting)
        assert not prepare.done()
        assert participant.readiness().blockers == ["r"]
        finish.set()
        await asyncio.wait_for(call, timeout=1)
        prepared = await asyncio.wait_for(prepare, timeout=1)
        assert prepared["phase"] == "prepared"
        assert sent[-1]["body"] == b"data: last\n\n"
        assert participant.readiness().ready
    finally:
        await controller.resume(CheckpointRequest(**control()))
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)


async def test_a_call_waiting_for_admission_is_refused_once_its_attempt_is_retired() -> None:
    app, participant, _ = make_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        reached = watch_admission(participant.gate)
        waiting = asyncio.create_task(client.post("/ng-rollout/r/v1/responses"))
        await asyncio.wait_for(reached.wait(), timeout=5)
        await participant.mark_retired([EpisodeId(rollout_id="r")])
        await participant.retire(EpisodeId(rollout_id="r"))
        refused = await asyncio.wait_for(waiting, timeout=5)

    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_stale_attempt"


async def test_a_call_waiting_at_closed_admission_stays_refused_after_its_retire_returns_and_the_rollout_is_forgotten() -> (
    None
):
    app, participant, _ = make_app()
    app.state.generation_done.set()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        await participant.close_admission(CheckpointRequest(**control()))
        reached = watch_admission(participant.gate)
        waiting = asyncio.create_task(client.post("/ng-rollout/r/v1/responses"))
        await asyncio.wait_for(reached.wait(), timeout=5)
        await participant.mark_retired([EpisodeId(rollout_id="r")])
        await participant.retire(EpisodeId(rollout_id="r"))
        # Once the retire returns, the controller may forget the rollout before the waiting call runs again.
        await participant.forget(["r"])
        await participant.open_admission()
        refused = await asyncio.wait_for(waiting, timeout=5)

    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert not participant.gate.served


async def test_a_call_waiting_at_closed_admission_stays_refused_after_a_retire_cut_short_by_its_deadline() -> None:
    app, participant, _ = make_app()
    app.state.generation_done.set()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        await participant.close_admission(CheckpointRequest(**control()))
        reached = watch_admission(participant.gate)
        waiting = asyncio.create_task(client.post("/ng-rollout/r/v1/responses"))
        await asyncio.wait_for(reached.wait(), timeout=5)
        late = {**control("retire", episode_ids=[{"rollout_id": "r"}]), "deadline_ts": time.time() - 1}
        retired = await client.post("/ng-control/v1/checkpoint/retire", json=late, headers=AUTH)
        await participant.forget(["r"])
        await participant.open_admission()
        refused = await asyncio.wait_for(waiting, timeout=5)

    assert retired.json()["error"]["code"] == "checkpoint_deadline_exceeded"
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert not participant.gate.served


async def test_undelivered_generation_does_not_block_prepare_and_is_held_until_resume() -> None:
    app, participant, _ = make_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        call = asyncio.create_task(client.post("/ng-rollout/r/v1/responses"))
        await wait_until(lambda: len(participant.gate.tickets) == 1)
        prepared = (await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)).json()
        delivering = asyncio.Event()
        deliver = participant.gate.deliver_response

        async def delivering_response(ticket) -> None:
            delivering.set()
            await deliver(ticket)

        participant.gate.deliver_response = delivering_response
        app.state.generation_done.set()
        await asyncio.wait_for(delivering.wait(), timeout=5)
        await asyncio.sleep(0.05)
        # The generation finished, but its response has not started.
        held = not call.done() and participant.gate.report().held == 1
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        delivered = await asyncio.wait_for(call, timeout=5)

    assert prepared["phase"] == "prepared" and prepared["report"]["counts"]["held"] == 1
    assert held
    assert delivered.json() == {"output": "done"}


async def test_retire_stops_a_running_generation_before_it_replies() -> None:
    app, participant, _ = make_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        running = asyncio.create_task(client.post("/ng-rollout/r/v1/responses"))
        await wait_until(lambda: len(participant.gate.tickets) == 1)
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        inflight_at_reply = participant.readiness().counts["inflight"]
        await asyncio.gather(running, return_exceptions=True)

    assert retired.status_code == 200
    assert inflight_at_reply == 0
    assert running.cancelled() or running.exception() is not None
    # The attempt stays refused until the controller forgets the rollout.
    assert len(participant.retired) == 1


async def _ledger_participant(root: Path) -> tuple[FileLineageStore, PolicyModelParticipant, ParticipantControlPlane]:
    ledger = FileLineageStore(root)
    participant = PolicyModelParticipant(ledger)
    return ledger, participant, ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)


def _commit_request(checkpoint_id: str, checkpoint_dir: Path, episode_ids: list | None = None) -> CommitRequest:
    return CommitRequest(**control(checkpoint_id, checkpoint_dir=str(checkpoint_dir), episode_ids=episode_ids))


def _restore_request(checkpoint_id: str, checkpoint_dir: Path, episode_ids: list) -> RestoreRequest:
    return RestoreRequest(**control(checkpoint_id, checkpoint_dir=str(checkpoint_dir), episode_ids=episode_ids))


async def test_restored_ledger_lets_the_next_attempt_resolve_its_parent(tmp_path: Path) -> None:
    ledger, _, controller = await _ledger_participant(tmp_path / "before")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",)))
    await ledger.close_admissions("r")
    await controller.prepare(CheckpointRequest(**control()))
    commit = await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
    # A row written after the cut belongs to an execution the restore replaces.
    await ledger.record(_commit(_call_record("late"), [USER_1, ASSISTANT_1, USER_2], [ASSISTANT_2], rollout_id="r"))

    restored_ledger, restored, restored_controller = await _ledger_participant(tmp_path / "after")
    await restored_controller.restore(_restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
    resolution = await restored_ledger.resolve("r-a1", [USER_1, ASSISTANT_1, USER_2])
    manifest = RolloutManifest.model_validate(await restored_ledger.manifest("r-a1"))

    assert commit["manifest"]["record_count"] == 1 and commit["episode_ids"] == ["r"]
    assert resolution.status == ParentResolutionStatus.RESOLVED
    assert (resolution.match.model_call_id, resolution.match.staging_chain) == ("c1", ("r/c1",))
    assert [record.model_call_id for record in manifest.records] == ["c1"]
    # The replaced attempt is refused until the controller forgets the rollout.
    assert restored.retired.marks() == {"r": 0}
    # The carried-over call stays staged under the attempt that made it.
    assert [record.source_capture_key for record in manifest.records] == ["r"]
    assert not any(row.get("admissions_closed") for row in restored_ledger.export_rows("r-a1"))
    await restored_ledger.begin_call("r-a1", "next")
    assert (await restored_ledger.manifest("r-a1"))["pending_call_ids"] == ["next"]
    with pytest.raises(CaptureAdmissionError, match="closed"):
        await ledger.begin_call("r", "still-closed")


async def test_restore_reopens_closed_admissions_from_older_checkpoint_rows(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "live")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r"))
    await ledger.close_admissions("r")
    record = ModelRecord(episode_id=EpisodeId(rollout_id="r"), rows=ledger.export_rows("r"))
    restored = FileLineageStore(tmp_path / "restored")
    import_model_records(restored, [record])
    await restored.begin_call("r-a1", "next")
    manifest = RolloutManifest.model_validate(await restored.manifest("r-a1"))
    assert [row.model_call_id for row in manifest.records] == ["c1"]
    assert manifest.pending_call_ids == ["next"] and not manifest.failures
    with pytest.raises(CaptureAdmissionError, match="closed"):
        await ledger.begin_call("r", "still-closed")


async def test_model_commit_requires_the_continued_episodes(tmp_path: Path) -> None:
    _, _, controller = await _ledger_participant(tmp_path / "store")
    await controller.prepare(CheckpointRequest(**control()))

    with pytest.raises(ControlError, match="episode_ids"):
        await controller.commit(_commit_request("c1", tmp_path / "ckpt"))


async def test_restore_refuses_a_model_server_that_has_served_calls(tmp_path: Path) -> None:
    ledger, _, controller = await _ledger_participant(tmp_path / "before")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r"))
    await ledger.record(_commit(_call_record("c9"), [USER_3], [ASSISTANT_1], rollout_id="s"))
    await controller.prepare(CheckpointRequest(**control()))
    await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}, {"rollout_id": "s"}]))

    # A ledger this process wrote may belong to a live episode, which the import would replace.
    restored_ledger, restored, restored_controller = await _ledger_participant(tmp_path / "after")
    await restored.gate.exit(restored.gate.enter("s-a1"))
    with pytest.raises(ControlError, match="has served calls"):
        await restored_controller.restore(
            _restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}, {"rollout_id": "s"}])
        )

    assert not await restored_ledger.has_rows("r-a1")


def _held_call(participant: PolicyModelParticipant, capture_key: str, model_call_id: str, request_items: list) -> None:
    ticket = participant.gate.enter(capture_key)
    ticket.backend = "http://worker-0/v1"
    ticket.capture = CaptureContext(
        rollout_id=capture_key, model_call_id=model_call_id, token_sink=None, request_items=request_items
    )


def _cutting_worker(cut: bool = True):
    requests: list[tuple[str, GenerationCutInventory]] = []

    async def request_cut(backend: str, inventory: GenerationCutInventory) -> GenerationCutReceipt:
        requests.append((backend, inventory))
        acks = [
            GenerationCutPrefixAck(
                **prefix.model_dump(),
                disposition="durable_prefix",
                cut_kind="active_prefix",
                frozen_buffer_id="buffer-1",
                staging_keys=(f"__generation_cut__/{prefix.model_call_id}",),
                prefix_token_count=3,
                prefix_digest="d" * 64,
                effective_output_limit=100,
            )
            if cut
            else GenerationCutPrefixAck.failure(prefix)
            for prefix in inventory.active_prefixes
        ]
        return GenerationCutReceipt(
            checkpoint_id=inventory.checkpoint_id,
            cut_id="cut-1",
            inventory_digest=inventory.inventory_digest,
            inventory=inventory,
            backend_snapshot_id="tq",
            prefixes=tuple(acks),
        )

    return request_cut, requests


async def test_undelivered_call_is_cut_and_continued_by_the_reissued_call(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "before")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",)))
    request_cut, requests = _cutting_worker()
    participant = PolicyModelParticipant(ledger, server_name="policy", cut_requester=request_cut)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    # c2 is in flight; its ledger row was written but the agent never received the response.
    _held_call(participant, "r", "c2", [USER_1, ASSISTANT_1, USER_2])
    await ledger.record(_commit(_call_record("c2"), [USER_1, ASSISTANT_1, USER_2], [ASSISTANT_2], rollout_id="r"))

    prepared = await controller.prepare(CheckpointRequest(**control()))
    await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))

    restored_ledger = FileLineageStore(tmp_path / "after")
    restored = PolicyModelParticipant(restored_ledger, server_name="policy")
    restored_controller = ParticipantControlPlane(restored, instance_name="policy", lease_grace_seconds=60)
    await restored_controller.restore(_restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
    await restored_controller.resume(CheckpointRequest(**control("r1")))
    admission = CaptureAdmission(capture_key="r-a1", model_call_id="c3", mode="text")
    other_request = CaptureContext(rollout_id="r-a1", model_call_id="c3", token_sink=None, request_items=[USER_3])
    reissued = CaptureContext(
        rollout_id="r-a1", model_call_id="c3", token_sink=None, request_items=[USER_1, ASSISTANT_1, USER_2]
    )

    assert prepared["report"]["counts"] == {"inflight": 1, "held": 1, "cut": 1, "cut_failed": 0, "cut_skipped": 0}
    assert [inventory.active_prefixes[0].model_call_id for _, inventory in requests] == ["c2"]
    assert [row["model_call_id"] for row in restored_ledger.export_rows("r-a1")] == ["c1"]
    ticket = restored.gate.enter("r-a1")
    hook = restored.gate.admission_hook(ticket)
    assert hook(other_request, admission).generation_cut is None
    continued = hook(reissued, admission).generation_cut
    assert (continued.source_model_call_id, continued.staging_keys) == ("c2", ("__generation_cut__/c2",))
    # The cut is kept until the call delivers its response, then freed.
    await restored.gate.deliver_response(ticket)
    await restored.gate.exit(ticket)
    later = restored.gate.enter("r-a1")
    assert restored.gate.admission_hook(later)(reissued, admission).generation_cut is None


async def test_failed_cut_regenerates_instead_of_blocking(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "before")
    request_cut, _ = _cutting_worker(cut=False)
    participant = PolicyModelParticipant(ledger, cut_requester=request_cut)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    _held_call(participant, "r", "c1", [USER_1])

    prepared = await controller.prepare(CheckpointRequest(**control()))
    commit = await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))

    assert prepared["phase"] == "prepared" and prepared["report"]["counts"]["cut"] == 0
    assert commit["manifest"]["record_count"] == 0


async def test_unreachable_worker_counts_as_a_failed_cut() -> None:
    async def unreachable(backend: str, inventory: GenerationCutInventory) -> GenerationCutReceipt:
        raise ConnectionError("worker is gone")

    participant = PolicyModelParticipant(cut_requester=unreachable)
    _held_call(participant, "r", "c1", [USER_1])

    prepared = await ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60).prepare(
        CheckpointRequest(**control())
    )

    assert prepared["phase"] == "prepared" and prepared["report"]["counts"]["cut"] == 0


def _captured_app(participant: PolicyModelParticipant, ledger: FileLineageStore | None, release: asyncio.Event):
    """A model app whose handler stands in for the capture middleware: it attaches a capture context,
    records the call's ledger row, then waits before responding."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict:
        capture_key = current_rollout_id()
        context = CaptureContext(
            rollout_id=capture_key, model_call_id="c2", token_sink=None, request_items=[USER_1, ASSISTANT_1, USER_2]
        )
        attach_capture_context(context)
        if ledger is not None:
            await ledger.record(
                _commit(_call_record("c2"), [USER_1, ASSISTANT_1, USER_2], [ASSISTANT_2], rollout_id=capture_key)
            )
        await release.wait()
        return {"hook": context.admission_hook is not None}

    # Stands in for the model server's capture middleware, which strips the rollout prefix.
    app.add_middleware(RolloutContextMiddleware)
    app.add_middleware(PolicyAdmissionMiddleware, gate=participant.gate)
    return app


async def test_undelivered_rows_are_excluded_on_any_capture_path(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "store")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",)))
    participant = PolicyModelParticipant(ledger)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    release = asyncio.Event()
    app = _captured_app(participant, ledger, release)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        call = asyncio.create_task(client.post("/ng-rollout/r/v1/chat/completions"))
        # The call has recorded its row and waits for release before the checkpoint opens.
        await wait_until(lambda: [row["model_call_id"] for row in ledger.export_rows("r")] == ["c1", "c2"])
        await controller.prepare(CheckpointRequest(**control()))
        release.set()
        await asyncio.sleep(0.05)
        live_rows = [row["model_call_id"] for row in ledger.export_rows("r")]
        commit = await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
        await controller.resume(CheckpointRequest(**control()))
        await call

    rows = (tmp_path / "ckpt" / "gym" / "model" / "policy" / commit["manifest"]["records_file"]).read_text()
    # The held call wrote its row to the live ledger, but the checkpoint leaves it out.
    assert live_rows == ["c1", "c2"]
    assert commit["manifest"]["record_count"] == 1
    assert '"c1"' in rows and '"c2"' not in rows


async def test_each_model_server_continues_only_its_own_restored_cut() -> None:
    first, second = PolicyModelParticipant(), PolicyModelParticipant()
    continuation = GenerationCutContinuation(
        source_capture_key="r",
        source_model_call_id="c2",
        staging_keys=("__generation_cut__/c2",),
        prefix_token_count=3,
        prefix_digest="d" * 64,
        effective_output_limit=10,
    )
    from nemo_gym._checkpoint.model import GenerationCutRecord
    from nemo_gym.token_id_capture.fingerprint import conversation_digest

    first._restored_cuts["r"] = GenerationCutRecord(
        model_call_id="c2",
        request_digest=conversation_digest([USER_1, ASSISTANT_1, USER_2]),
        continuation=continuation,
    )
    responses = []
    for participant in (second, first):
        release = asyncio.Event()
        release.set()
        app = _captured_app(participant, None, release)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
            responses.append((await client.post("/ng-rollout/r/v1/chat/completions")).json())
    admission = CaptureAdmission(capture_key="r", model_call_id="c9", mode="text")
    context = CaptureContext(
        rollout_id="r", model_call_id="c9", token_sink=None, request_items=[USER_1, ASSISTANT_1, USER_2]
    )

    assert responses == [{"hook": True}, {"hook": True}]
    assert second.gate.admission_hook(second.gate.enter("r"))(context, admission).generation_cut is None
    assert first.gate.admission_hook(first.gate.enter("r"))(context, admission).generation_cut == continuation


async def test_a_restored_cut_not_yet_reused_survives_the_next_checkpoint(tmp_path: Path) -> None:
    from nemo_gym._checkpoint.model import GenerationCutRecord

    continuation = GenerationCutContinuation(
        source_capture_key="r",
        source_model_call_id="c2",
        staging_keys=("__generation_cut__/c2",),
        prefix_token_count=3,
        prefix_digest="d" * 64,
        effective_output_limit=10,
    )
    cut = GenerationCutRecord(model_call_id="c2", request_digest="e" * 64, continuation=continuation)
    ledger, participant, controller = await _ledger_participant(tmp_path / "ledger")
    participant._restored_cuts["r-a1"] = cut
    # The replacement attempt has not re-issued its call when the next checkpoint lands.
    await controller.prepare(CheckpointRequest(**control("c2")))
    await controller.commit(_commit_request("c2", tmp_path / "ckpt", [{"rollout_id": "r", "attempt": 1}]))

    _, again, again_controller = await _ledger_participant(tmp_path / "again")
    await again_controller.restore(_restore_request("r2", tmp_path / "ckpt", [{"rollout_id": "r", "attempt": 1}]))

    assert again._restored_cuts == {"r-a2": cut}


async def test_a_newer_cut_replaces_the_restored_cut_a_resumed_call_keeps_for_its_retry(tmp_path: Path) -> None:
    from nemo_gym._checkpoint.model import GenerationCutRecord
    from nemo_gym.token_id_capture.fingerprint import conversation_digest

    request_items = [USER_1, ASSISTANT_1, USER_2]
    continuation = GenerationCutContinuation(
        source_capture_key="r",
        source_model_call_id="c2",
        staging_keys=("__generation_cut__/c2",),
        prefix_token_count=3,
        prefix_digest="d" * 64,
        effective_output_limit=10,
    )
    restored_cut = GenerationCutRecord(
        model_call_id="c2", request_digest=conversation_digest(request_items), continuation=continuation
    )
    request_cut, _ = _cutting_worker()
    participant = PolicyModelParticipant(FileLineageStore(tmp_path / "ledger"), cut_requester=request_cut)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    participant._restored_cuts["r-a1"] = restored_cut
    admission = CaptureAdmission(capture_key="r-a1", model_call_id="c3", mode="text")
    # The resumed call re-issues the cut request and continues the restored cut.
    _held_call(participant, "r-a1", "c3", request_items)
    [resumed] = participant.gate.tickets
    continued = participant.gate.admission_hook(resumed)(resumed.capture, admission).generation_cut

    # The next checkpoint cuts the resumed call further, and exports that newer cut.
    await controller.prepare(CheckpointRequest(**control("c2")))
    commit = await controller.commit(_commit_request("c2", tmp_path / "ckpt", [{"rollout_id": "r", "attempt": 1}]))
    await controller.resume(CheckpointRequest(**control("c2")))
    # The resumed call ends before its response starts, and the agent retries it.
    await participant.gate.exit(resumed)
    retry = participant.gate.enter("r-a1")
    retried = CaptureContext(rollout_id="r-a1", model_call_id="c4", token_sink=None, request_items=request_items)
    received = participant.gate.admission_hook(retry)(retried, admission).generation_cut

    assert continued == continuation
    assert commit["staging_keys"] == ["__generation_cut__/c3"]
    # The retry continues the cut the checkpoint kept, not the one whose staged rows it no longer retains.
    assert received.staging_keys == ("__generation_cut__/c3",)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({}, None),
        ({"tool_choice": "auto", "tools": [{"type": "function"}]}, None),
        ({"tool_choice": "none", "tools": [{"type": "function"}]}, None),
        ({"tool_choice": "required"}, "tool_choice:required"),
        ({"tool_choice": {"type": "function", "function": {"name": "f"}}}, "tool_choice:constrained"),
        ({"response_format": {"type": "text"}}, None),
        ({"response_format": {"type": "json_schema"}}, "response_format:json_schema"),
        ({"guided_json": {}}, "guided_json"),
        ({"structured_outputs": {}}, "structured_outputs"),
    ],
)
def test_constrained_decoding_must_regenerate_instead_of_continuing_a_cut(body: dict, reason: str | None) -> None:
    from nemo_gym._checkpoint.model import generation_cut_regenerate_reason

    assert generation_cut_regenerate_reason(body) == reason


async def test_a_constrained_call_is_not_cut_and_never_continues_a_restored_cut(tmp_path: Path) -> None:
    request_cut, requests = _cutting_worker()
    participant = PolicyModelParticipant(FileLineageStore(tmp_path / "ledger"), cut_requester=request_cut)
    _held_call(participant, "r", "c1", [USER_1])
    constrained = participant.gate.enter("s")
    constrained.backend = "http://worker-0/v1"
    constrained.capture = CaptureContext(rollout_id="s", model_call_id="c2", token_sink=None, request_items=[USER_1])
    constrained.cut_regenerate_reason = "tool_choice:required"
    await participant.close_admission(CheckpointRequest(**control()))
    report = participant.readiness()

    participant._restored_cuts["s"] = restored = participant.gate.snapshot().cuts[0].record.model_copy()
    admission = CaptureAdmission(capture_key="s", model_call_id="c9", mode="text")
    context = CaptureContext(rollout_id="s", model_call_id="c9", token_sink=None, request_items=[USER_1])
    attached = participant.gate.admission_hook(constrained)(context, admission).generation_cut

    [(_, inventory)] = requests
    assert [prefix.model_call_id for prefix in inventory.active_prefixes] == ["c1"]
    assert (report.counts["cut"], report.counts["cut_skipped"]) == (1, 1)
    assert attached is None
    assert participant._restored_cuts == {"s": restored}


async def test_calls_the_worker_could_not_cut_are_counted(tmp_path: Path) -> None:
    request_cut, _ = _cutting_worker(cut=False)
    participant = PolicyModelParticipant(FileLineageStore(tmp_path / "ledger"), cut_requester=request_cut)
    _held_call(participant, "r", "c1", [USER_1])
    await participant.close_admission(CheckpointRequest(**control()))

    assert participant.readiness().counts["cut_failed"] == 1


async def test_commit_reply_lists_the_staged_keys_the_checkpoint_keeps(tmp_path: Path) -> None:
    ledger, _, controller = await _ledger_participant(tmp_path / "ledger")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",)))
    await controller.prepare(CheckpointRequest(**control()))
    reply = await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
    [row] = ledger.export_rows("r")

    assert reply["staging_keys"] == [row["staging_key"]]


async def test_commit_reply_groups_the_staged_keys_by_episode(tmp_path: Path) -> None:
    """The controller keeps only the rows of episodes it continues, so it needs each key's owner."""
    ledger, _, controller = await _ledger_participant(tmp_path / "ledger")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r"))
    await ledger.record(_commit(_call_record("c9"), [USER_3], [ASSISTANT_1], rollout_id="s"))
    await controller.prepare(CheckpointRequest(**control()))
    reply = await controller.commit(
        _commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}, {"rollout_id": "s"}])
    )
    [r_row], [s_row] = ledger.export_rows("r"), ledger.export_rows("s")

    assert reply["staging_keys_by_episode"] == {"r": [r_row["staging_key"]], "s": [s_row["staging_key"]]}
    assert sorted(reply["staging_keys"]) == sorted([r_row["staging_key"], s_row["staging_key"]])


async def test_generation_cut_keys_are_grouped_under_their_episode(tmp_path: Path) -> None:
    """A cut's continuation keys name no rollout; only the record says whose they are."""
    request_cut, _ = _cutting_worker()
    participant = PolicyModelParticipant(FileLineageStore(tmp_path / "ledger"), cut_requester=request_cut)
    _held_call(participant, "s", "c2", [USER_1])
    await participant.close_admission(CheckpointRequest(**control()))
    [cut] = participant.gate.snapshot().cuts
    carried = {"model_call_id": "c1", "staging_key": "r/c1"}
    records = [
        ModelRecord(episode_id=EpisodeId(rollout_id="r"), rows=[carried]),
        # A restored attempt's record carries the rows of the attempt it continues.
        ModelRecord(episode_id=EpisodeId(rollout_id="r", attempt=1), rows=[carried]),
        ModelRecord(episode_id=EpisodeId(rollout_id="s"), rows=[], generation_cuts=[cut.record]),
    ]

    assert participant.commit_reply(records)["staging_keys_by_episode"] == {
        "r": ["r/c1"],
        "r-a1": ["r/c1"],
        "s": ["__generation_cut__/c2"],
    }


async def test_ledger_export_and_import_run_off_the_event_loop(tmp_path: Path) -> None:
    import threading

    class RecordingLedger:
        def __init__(self) -> None:
            self.threads: list[int] = []
            self.rows: dict[str, list[dict]] = {}

        def export_rows(self, rollout_id: str) -> list[dict]:
            self.threads.append(threading.get_ident())
            # Only the source episode starts with rows; the restore target starts empty.
            default = [{"model_call_id": "c1", "staging_key": "r/c1"}] if rollout_id == "r" else []
            return self.rows.get(rollout_id, default)

        def import_rows(self, rollout_id: str, rows: list[dict]) -> None:
            self.threads.append(threading.get_ident())
            self.rows[rollout_id] = rows

        async def retire(self, rollout_ids: list[str]) -> dict:
            return {"removed": [], "absent": list(rollout_ids)}

    source = RecordingLedger()
    participant = PolicyModelParticipant(source)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    await controller.prepare(CheckpointRequest(**control()))
    await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
    target = RecordingLedger()
    restored = PolicyModelParticipant(target)
    await ParticipantControlPlane(restored, instance_name="policy", lease_grace_seconds=60).restore(
        _restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}])
    )

    loop_thread = threading.get_ident()
    assert source.threads and loop_thread not in source.threads
    assert target.threads and loop_thread not in target.threads
    assert target.rows["r-a1"] == [{"model_call_id": "c1", "staging_key": "r/c1", "source_capture_key": "r"}]


def test_a_batched_ledger_import_syncs_the_directory_per_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nemo_gym.token_id_capture.lineage as lineage

    ledger = FileLineageStore(tmp_path / "ledger")
    directory_opens = []
    real_open = lineage.os.open

    def counting_open(path, flags, *args):
        if flags == os.O_RDONLY:
            directory_opens.append(path)
        return real_open(path, flags, *args)

    monkeypatch.setattr(lineage.os, "open", counting_open)
    ledger.import_rows_many({"single-a1": [{"model_call_id": "c"}]})
    per_batch = len(directory_opens)
    rows = {f"r{index}-a1": [{"model_call_id": f"c{index}"}] for index in range(20)}
    ledger.import_rows_many(rows)

    assert len(directory_opens) == 2 * per_batch
    assert ledger.export_rows("r7-a1") == [{"model_call_id": "c7"}]


def test_restoring_a_checkpoint_again_after_its_replacement_made_a_call_continues_from_the_checkpoint(
    tmp_path: Path,
) -> None:
    row = {"model_call_id": "c0", "staging_key": "roll-1/c0"}
    record = ModelRecord(episode_id=EpisodeId.from_capture_key("roll-1"), rows=[row])
    import_model_records(FileLineageStore(tmp_path), [record])
    # The replacement attempt commits a call, then Gym crashes before the next checkpoint.
    with (tmp_path / "roll-1-a1.lineage.jsonl").open("a") as handle:
        handle.write('{"model_call_id":"c1","staging_key":"roll-1-a1/c1"}\n')

    import_model_records(FileLineageStore(tmp_path), [record])

    # Only the checkpointed call, stamped with the attempt that staged it.
    assert FileLineageStore(tmp_path).export_rows("roll-1-a1") == [{**row, "source_capture_key": "roll-1"}]


async def test_a_ledger_import_deletes_dead_executions_and_fences_of_the_target_and_later_attempts(
    tmp_path: Path,
) -> None:
    ledger = FileLineageStore(tmp_path)
    ledger.import_rows_many(
        {
            "r-a1": [{"model_call_id": "dead"}],
            "r-a2": [{"model_call_id": "dead-later"}],
            "r1-a2": [{"model_call_id": "other-rollout"}],
            "r-a1x-a3": [{"model_call_id": "look-alike"}],
        }
    )
    (tmp_path / "r-a2.tokens.jsonl").write_text("{}\n")
    (tmp_path / "r-a2.tokens.incomplete").write_text("x\n")
    # The trainer finished or abandoned the dead execution's attempts 1 and 3 and retired them, leaving fences.
    await ledger.retire(["r-a1", "r-a3"])
    # A restored attempt 1 owns the rollout from here on: attempt 2 of the dead execution must not survive.
    fresh = FileLineageStore(tmp_path)
    fresh.import_rows_many({"r-a1": [{"model_call_id": "checkpointed"}]})

    assert fresh.export_rows("r-a1") == [{"model_call_id": "checkpointed"}]
    assert fresh.export_rows("r-a2") == []
    assert not (tmp_path / "r-a2.tokens.jsonl").exists() and not (tmp_path / "r-a2.tokens.incomplete").exists()
    # A fence left in place would silently discard the restored episode's rows when it reaches that attempt.
    assert not (tmp_path / "r-a1.lineage.retired").exists() and not (tmp_path / "r-a3.lineage.retired").exists()
    assert fresh.export_rows("r1-a2") == [{"model_call_id": "other-rollout"}]
    assert fresh.export_rows("r-a1x-a3") == [{"model_call_id": "look-alike"}]


async def test_readiness_stays_cheap_while_many_cut_calls_are_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nemo_gym._checkpoint.model as model

    request_cut, _ = _cutting_worker()
    participant = PolicyModelParticipant(FileLineageStore(tmp_path / "ledger"), cut_requester=request_cut)
    for index in range(50):
        _held_call(participant, f"r{index}", f"c{index}", [USER_1])
    await participant.close_admission(CheckpointRequest(**control()))
    digests = []
    real_digest = model.conversation_digest
    monkeypatch.setattr(model, "conversation_digest", lambda items: digests.append(1) or real_digest(items))
    for _ in range(20):
        report = participant.readiness()

    # Each cut record is built once, when the cut is acknowledged; readiness only counts.
    assert digests == []
    assert report.counts["cut"] == 50


async def test_an_episode_with_a_retried_call_keeps_only_the_latest_cut(tmp_path: Path) -> None:
    request_cut, _ = _cutting_worker()
    ledger = FileLineageStore(tmp_path / "ledger")
    participant = PolicyModelParticipant(ledger, cut_requester=request_cut)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    # The client gave up on c1 and retried it as c2; the server is still running both.
    _held_call(participant, "r", "c1", [USER_1])
    _held_call(participant, "r", "c2", [USER_1])
    [first, second] = sorted(participant.gate.tickets, key=lambda ticket: ticket.capture.model_call_id)
    first.admitted_at, second.admitted_at = 1.0, 2.0
    await controller.prepare(CheckpointRequest(**control()))
    await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
    restored = PolicyModelParticipant(FileLineageStore(tmp_path / "restored"))
    await ParticipantControlPlane(restored, instance_name="policy", lease_grace_seconds=60).restore(
        _restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}])
    )

    assert restored._restored_cuts["r-a1"].model_call_id == "c2"


async def test_ledger_retire_and_delete_wait_out_an_open_checkpoint(tmp_path: Path) -> None:
    ledger, participant, controller = await _ledger_participant(tmp_path)
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r"))
    app = FastAPI()
    install_rollout_control_routes(app, ledger, auth_token="t", refuse_removal=lambda: ledger_removal_refusal(app))
    app.state.nemo_gym_policy_gate = participant.gate
    headers = {"authorization": "Bearer t"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        await controller.prepare(CheckpointRequest(**control()))
        # A commit still reads the ledger of every live episode, so removing one now would empty its lineage.
        refused = await client.post(
            "/training-token-capture/control/rollouts/retire", json={"rollout_ids": ["r"]}, headers=headers
        )
        deleted = await client.post(
            "/training-token-capture/control/rollouts/delete", json={"rollout_ids": ["r"]}, headers=headers
        )
        commit = await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
        await controller.resume(CheckpointRequest(**control()))
        retired = await client.post(
            "/training-token-capture/control/rollouts/retire", json={"rollout_ids": ["r"]}, headers=headers
        )

    assert refused.status_code == 409 and deleted.status_code == 409
    assert commit["manifest"]["record_count"] == 1
    assert retired.status_code == 200 and retired.json()["removed"] == ["r"]


async def test_retiring_an_attempt_retires_the_ledgers_of_it_and_every_earlier_attempt(tmp_path: Path) -> None:
    ledger, _, controller = await _ledger_participant(tmp_path)
    for key in ("r", "r-a1", "r-a2"):
        await ledger.record(_commit(_call_record(f"call-{key}"), [USER_1], [ASSISTANT_1], rollout_id=key))

    await controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r", "attempt": 1}])))

    # A retired ledger has no rows and refuses reads, so a reader cannot mistake it for an empty live one.
    for key in ("r", "r-a1"):
        with pytest.raises(RolloutRetiredError):
            await ledger.has_rows(key)
    assert (tmp_path / "r.lineage.retired").exists() and (tmp_path / "r-a1.lineage.retired").exists()
    assert await ledger.has_rows("r-a2")


async def test_a_restore_retires_the_ledgers_of_the_attempts_it_continues(tmp_path: Path) -> None:
    ledger, _, controller = await _ledger_participant(tmp_path / "ledger")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r"))
    await controller.prepare(CheckpointRequest(**control()))
    await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))

    # A crash: a fresh model server over the same capture directory restores the checkpoint.
    restored_ledger, _, restored = await _ledger_participant(tmp_path / "ledger")
    await restored.restore(_restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}]))

    # Attempt 0 continues as attempt 1, so attempt 0's ledger is freed and fenced against late rows.
    assert [row["model_call_id"] for row in restored_ledger.export_rows("r-a1")] == ["c1"]
    with pytest.raises(RolloutRetiredError):
        await restored_ledger.has_rows("r")
    assert (tmp_path / "ledger" / "r.lineage.retired").exists()


def test_a_chain_of_restores_keeps_each_row_staged_under_its_own_attempt() -> None:
    """Restoring attempt 1 again stamps its own calls and keeps attempt 0's stamps."""

    class DictLedger:
        def __init__(self) -> None:
            self.rows: dict[str, list[dict]] = {}

        def export_rows(self, rollout_id: str) -> list[dict]:
            return self.rows.get(rollout_id, [])

        def import_rows(self, rollout_id: str, rows: list[dict]) -> None:
            self.rows[rollout_id] = rows

    ledger = DictLedger()
    failure = {"model_call_id": "lost", "failure_reason": "capture_failed"}
    import_model_records(
        ledger,
        [
            ModelRecord(
                episode_id=EpisodeId(rollout_id="r"), rows=[{"model_call_id": "c1", "staging_key": "r/c1"}, failure]
            )
        ],
    )
    import_model_records(
        ledger,
        [
            ModelRecord(
                episode_id=EpisodeId(rollout_id="r", attempt=1),
                rows=[*ledger.rows["r-a1"], {"model_call_id": "c2", "staging_key": "r-a1/c2"}],
            )
        ],
    )

    assert ledger.rows["r-a1"] == [{"model_call_id": "c1", "staging_key": "r/c1", "source_capture_key": "r"}, failure]
    assert ledger.rows["r-a2"] == [
        {"model_call_id": "c1", "staging_key": "r/c1", "source_capture_key": "r"},
        failure,
        {"model_call_id": "c2", "staging_key": "r-a1/c2", "source_capture_key": "r-a1"},
    ]


def _restored_cut_checkpoint(directory: Path) -> None:
    """A checkpoint of rollout r with one delivered call and an undelivered call that was cut."""
    from nemo_gym._checkpoint.model import GenerationCutRecord, ModelRecord
    from nemo_gym._checkpoint.store import write_participant_state

    continuation = GenerationCutContinuation(
        source_capture_key="r",
        source_model_call_id="c2",
        staging_keys=("__generation_cut__/c2",),
        prefix_token_count=3,
        prefix_digest="d" * 64,
        effective_output_limit=10,
    )
    cut = GenerationCutRecord(model_call_id="c2", request_digest="e" * 64, continuation=continuation)
    record = ModelRecord(episode_id=EpisodeId(rollout_id="r"), rows=[{"model_call_id": "c1"}], generation_cuts=[cut])
    write_participant_state(
        directory, kind="model", instance="policy", checkpoint_id="c0", records=[record.to_json_record()]
    )


async def test_a_commit_that_no_longer_names_a_started_replacement_leaves_its_ledger_alone(tmp_path: Path) -> None:
    _restored_cut_checkpoint(tmp_path / "ckpt")
    ledger, participant, controller = await _ledger_participant(tmp_path / "ledger")
    await controller.restore(_restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
    await controller.resume(CheckpointRequest(**control("r1")))
    # The replacement starts; its re-issued call renders differently, so the restored cut is not used.
    participant.gate.enter("r-a1")
    ledger.import_rows_many({"r-a1": [{"model_call_id": "c1"}, {"model_call_id": "replacement-call"}]})

    # The controller stops naming a restored rollout once it has dispatched it.
    await controller.prepare(CheckpointRequest(**control("c2")))
    await controller.commit(_commit_request("c2", tmp_path / "ckpt2", []))

    assert [row["model_call_id"] for row in ledger.export_rows("r-a1")] == ["c1", "replacement-call"]
    assert not (tmp_path / "ledger" / "r-a1.lineage.retired").exists()


async def test_a_commit_deletes_a_restored_attempt_nothing_started_without_fencing_it(tmp_path: Path) -> None:
    _restored_cut_checkpoint(tmp_path / "ckpt")
    ledger, participant, controller = await _ledger_participant(tmp_path / "ledger")
    await controller.restore(_restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
    await controller.resume(CheckpointRequest(**control("r1")))

    await controller.prepare(CheckpointRequest(**control("c2")))
    await controller.commit(_commit_request("c2", tmp_path / "ckpt2", []))

    # Gone, and not fenced: the controller may still start the rollout over as r-a1.
    assert ledger.export_rows("r-a1") == [] and participant._restored_cuts == {}
    assert not (tmp_path / "ledger" / "r-a1.lineage.retired").exists()
    assert await participant.unclaimed() == []


async def test_restoring_again_clears_a_dead_execution_of_an_in_scope_episode_without_a_record(tmp_path: Path) -> None:
    ledger, participant, controller = await _ledger_participant(tmp_path / "before")
    ledger.import_rows_many({"r": [{"model_call_id": "c1"}]})
    await controller.prepare(CheckpointRequest(**control()))
    # Episode e is continued but made no calls, so the model has no record of it.
    await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}, {"rollout_id": "e"}]))

    after = FileLineageStore(tmp_path / "after")
    # What an earlier restore's replacement of e wrote before Gym crashed again.
    after.import_rows_many({"e-a1": [{"model_call_id": "dead"}]})
    restored = PolicyModelParticipant(after)
    await ParticipantControlPlane(restored, instance_name="policy", lease_grace_seconds=60).restore(
        _restore_request("r2", tmp_path / "ckpt", [{"rollout_id": "r"}, {"rollout_id": "e"}])
    )

    assert after.export_rows("e-a1") == []
    assert [row["model_call_id"] for row in after.export_rows("r-a1")] == ["c1"]


async def test_untagged_calls_neither_break_a_retire_nor_survive_as_untyped_errors() -> None:
    participant = PolicyModelParticipant()
    # An eval, a judge, or a health probe carries no rollout prefix.
    untagged = participant.gate.enter("")
    stopped = asyncio.Event()

    async def tagged_call() -> None:
        participant.gate.enter("r")
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    task = asyncio.create_task(tagged_call())
    await asyncio.sleep(0)
    await participant.retire(EpisodeId(rollout_id="r"))

    assert stopped.is_set() and task.cancelled() and untagged in participant.gate.tickets
    for malformed in ("bad!id", "-a1"):
        with pytest.raises(ControlError) as refused:
            await participant.gate.admit(malformed)
        assert refused.value.status_code == 400 and refused.value.code == "checkpoint_invalid_rollout_id"


async def test_a_ticket_leaving_is_reported_even_if_giving_back_its_cut_fails() -> None:
    participant = PolicyModelParticipant()
    reported = []

    async def on_change() -> None:
        reported.append(True)

    async def failing_settle(ticket) -> None:
        raise ControlError("coordinator unavailable")

    participant.gate.on_change = on_change
    participant.gate.restored_cuts.settle = failing_settle
    ticket = participant.gate.enter("r")
    with pytest.raises(ControlError):
        await participant.gate.exit(ticket)

    assert reported and ticket not in participant.gate.tickets


async def test_a_hung_cut_leaves_the_rest_of_the_prepare_deadline_for_the_stages_after_it() -> None:
    async def hung(backend: str, inventory: GenerationCutInventory) -> GenerationCutReceipt:
        await asyncio.Event().wait()

    participant = PolicyModelParticipant(server_name="policy", cut_requester=hung)
    _held_call(participant, "r", "c1", [USER_1])
    started = time.monotonic()
    await participant.gate.close(CheckpointRequest(**{**control(), "deadline_ts": time.time() + 3.0}))

    # Half of the remaining 3 s, with a second to spare on a loaded machine.
    assert time.monotonic() - started < 2.5
    assert participant.gate.report().cut_failed == 1


async def test_after_a_restore_a_late_call_of_the_replaced_attempt_is_refused(tmp_path: Path) -> None:
    from nemo_gym._checkpoint.store import write_participant_state

    # This model server holds no record of the rollout; the restore continues it all the same.
    write_participant_state(tmp_path, kind="model", instance="policy", checkpoint_id="c0", records=[])
    app, _, _ = make_app()
    app.state.generation_done.set()
    restore = control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        restored = await client.post("/ng-control/v1/checkpoint/restore", json=restore, headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        # A harness of the crashed attempt that outlived the crash.
        late = await client.post("/ng-rollout/r/v1/responses")
        replacement = await client.post("/ng-rollout/r-a1/v1/responses")

    assert restored.status_code == 200
    assert late.status_code == 409 and late.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert replacement.status_code == 200


async def _restored_servers_sharing_one_ledger(
    tmp_path: Path,
) -> tuple[FileLineageStore, list[tuple[PolicyModelParticipant, ParticipantControlPlane]]]:
    """Two policy servers capturing into one ledger, as policy_model and policy_model_reasoning_off do,
    restored from a checkpoint that continues rollout r from its call c1."""
    from nemo_gym._checkpoint.store import write_participant_state

    record = ModelRecord(episode_id=EpisodeId(rollout_id="r"), rows=[{"model_call_id": "c1"}])
    servers = []
    for name in ("policy", "policy_reasoning_off"):
        # Each server exports the shared ledger, so both checkpoints hold the same rows.
        write_participant_state(
            tmp_path / "ckpt", kind="model", instance=name, checkpoint_id="c0", records=[record.to_json_record()]
        )
        participant = PolicyModelParticipant(FileLineageStore(tmp_path / "ledger"), server_name=name)
        controller = ParticipantControlPlane(participant, instance_name=name, lease_grace_seconds=60)
        await controller.restore(_restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
        await controller.resume(CheckpointRequest(**control("r1")))
        servers.append((participant, controller))
    return FileLineageStore(tmp_path / "ledger"), servers


@pytest.mark.parametrize("recorded", [True, False])
async def test_a_restored_attempt_started_on_one_policy_server_keeps_its_ledger_on_another(
    tmp_path: Path, recorded: bool
) -> None:
    ledger, [(first, _), (_, second_controller)] = await _restored_servers_sharing_one_ledger(tmp_path)
    release = asyncio.Event()
    # Without a ledger the handler records no row: the call is admitted and still running, with nothing written.
    app = _captured_app(first, first.ledger if recorded else None, release)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        call = asyncio.create_task(client.post("/ng-rollout/r-a1/v1/chat/completions"))
        await wait_until(lambda: first.gate.tickets and "r-a1" not in first.gate.restored_targets)
        if recorded:
            await wait_until(lambda: [row["model_call_id"] for row in ledger.export_rows("r-a1")] == ["c1", "c2"])
        # The second server never saw the attempt start, and the commit no longer names it.
        await second_controller.prepare(CheckpointRequest(**control("c2")))
        await second_controller.commit(_commit_request("c2", tmp_path / "ckpt2", []))
        release.set()
        await call

    assert [row["model_call_id"] for row in ledger.export_rows("r-a1")] == (["c1", "c2"] if recorded else ["c1"])


async def test_a_restored_attempt_no_policy_server_started_is_deleted_with_no_marker_left(tmp_path: Path) -> None:
    ledger, [(_, first_controller), (_, second_controller)] = await _restored_servers_sharing_one_ledger(tmp_path)
    for controller in (first_controller, second_controller):
        await controller.prepare(CheckpointRequest(**control("c2")))
        await controller.commit(_commit_request("c2", tmp_path / "ckpt2", []))

    assert ledger.export_rows("r-a1") == []
    assert not list((tmp_path / "ledger").glob("r-a1.lineage.*"))


async def test_the_started_marker_ends_with_the_attempt_s_ledger(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path)
    for remove in (ledger.retire, ledger.delete):
        ledger.import_rows_many({"r-a1": [{"model_call_id": "c1"}]})
        ledger.mark_started("r-a1")
        await remove(["r-a1"])
        assert not (tmp_path / "r-a1.lineage.started").exists()
    # Restoring again replaces what a dead execution left, its marker included.
    ledger.mark_started("r-a1")
    ledger.import_rows_many({"r-a1": [{"model_call_id": "c1"}]})
    restored_again_marked = (tmp_path / "r-a1.lineage.started").exists()
    # A retired attempt's calls are refused, so it is never marked.
    await ledger.retire(["r-a1"])
    ledger.mark_started("r-a1")

    assert not restored_again_marked
    assert not (tmp_path / "r-a1.lineage.started").exists()


async def test_a_failed_restored_ledger_delete_is_retried_by_the_next_commit_attempt(tmp_path: Path) -> None:
    _restored_cut_checkpoint(tmp_path / "ckpt")
    ledger, participant, controller = await _ledger_participant(tmp_path / "ledger")
    await controller.restore(_restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
    await controller.resume(CheckpointRequest(**control("r1")))
    remove = ledger._remove
    failures = [OSError("ledger unavailable")]

    def failing_once(rollout_ids: list[str], **options: bool) -> dict:
        if failures:
            raise failures.pop()
        return remove(rollout_ids, **options)

    ledger._remove = failing_once
    await controller.prepare(CheckpointRequest(**control("c2")))
    with pytest.raises(OSError):
        await controller.commit(_commit_request("c2", tmp_path / "ckpt2", []))
    await controller.commit(_commit_request("c2", tmp_path / "ckpt2", []))

    assert ledger.export_rows("r-a1") == []
    assert await participant.unclaimed() == []


async def test_a_retried_retire_waits_for_a_cancelled_call_s_cleanup_without_cancelling_it_again() -> None:
    participant = PolicyModelParticipant()
    generating, cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancelled_in, cleaned_up = [], []

    async def app(scope, receive, send):
        try:
            generating.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_in.append("generation")
            cleanup_started.set()
            try:
                # Stands in for releasing the downstream generation and session after the cancellation.
                await release_cleanup.wait()
            except asyncio.CancelledError:
                cancelled_in.append("cleanup")
                raise
            cleaned_up.append(True)
            raise

    async def send(message):
        pass

    call = asyncio.create_task(
        PolicyAdmissionMiddleware(app, participant.gate)(
            {"type": "http", "path": "/ng-rollout/r/v1/chat/completions"}, None, send
        )
    )
    await asyncio.wait_for(generating.wait(), timeout=5)
    first = asyncio.create_task(participant.retire(EpisodeId(rollout_id="r")))
    await asyncio.wait_for(cleanup_started.wait(), timeout=5)
    # The retire's deadline cuts it short while the call is still cleaning up.
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    retrying = asyncio.Event()
    refuse_waiters = participant.gate.refuse_waiters

    def refusing(episode_ids):
        retrying.set()
        return refuse_waiters(episode_ids)

    participant.gate.refuse_waiters = refusing
    retry = asyncio.create_task(participant.retire(EpisodeId(rollout_id="r")))
    # The retry has passed every cancellation it would make before this test runs again.
    await asyncio.wait_for(retrying.wait(), timeout=5)
    release_cleanup.set()
    await asyncio.wait_for(retry, timeout=5)
    await asyncio.gather(call, return_exceptions=True)

    assert cancelled_in == ["generation"]
    assert cleaned_up == [True]
    assert not participant.gate.tickets


# What a Responses stream with ``stream_keepalive_s`` sends while generation still runs, one event per chunk.
RESPONSES_LIFECYCLE_EVENTS = tuple(
    event.encode() for event in synthesize_responses_keepalive_sse("resp_r", include_created=True)
)
RESPONSE_COMPLETED_EVENT = (
    b'event: response.completed\ndata: {"type": "response.completed", "response": {"id": "resp_r"}}\n\n'
)


def _staged_call(
    gate: PolicyGate,
    ledger: FileLineageStore,
    sent: list[dict],
    *,
    contexts: Optional[list[CaptureContext]] = None,
    finalized: Optional[asyncio.Event] = None,
    answer: Optional[asyncio.Event] = None,
    lifecycle: Optional[asyncio.Event] = None,
) -> asyncio.Task:
    """A policy call of rollout r through the gate and the model server's capture middleware.

    Its worker staged the tokens, so the call finalizes its capture before it responds.
    It then sends headers and a keep-alive comment, and its answer once ``answer`` is set.
    With ``lifecycle``, the call instead opens its stream with Responses lifecycle events, as
    ``stream_keepalive_s`` does before generation finishes, and finalizes its capture and sends
    its terminal event once ``lifecycle`` is set.
    """

    async def app(scope, receive, send):
        context = current_capture_context()
        context.request_items = [USER_1, ASSISTANT_1, USER_2]
        context.capture_admission = CaptureAdmission(
            capture_key=context.rollout_id, model_call_id=context.model_call_id, mode="text"
        )
        if contexts is not None:
            contexts.append(context)
        coords = CommitCoords(
            capture_key=context.rollout_id,
            model_call_id=context.model_call_id,
            prev_len=0,
            weight_version=1,
            delta_len=3,
            cum_len=3,
            digest="0" * 64,
            extras_digest="1" * 64,
            staging_key=f"{context.rollout_id}/{context.model_call_id}",
            chain_hash="2" * 64,
            cumulative_hash="3" * 64,
        )
        payload = {
            "id": f"chatcmpl-{context.model_call_id}",
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
            NG_COMMIT_COORDS_FIELD: coords.model_dump(mode="json"),
        }
        if lifecycle is not None:
            await send(
                {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
            )
            for event in RESPONSES_LIFECYCLE_EVENTS:
                await send({"type": "http.response.body", "body": event, "more_body": True})
            await lifecycle.wait()
        handler = VLLMWorkerCaptureHandler()
        handler.prepare_response(payload)
        await handler.finalize_response(payload)
        if finalized is not None:
            finalized.set()
        if lifecycle is not None:
            await send({"type": "http.response.body", "body": RESPONSE_COMPLETED_EVENT, "more_body": False})
            return
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
        )
        await send({"type": "http.response.body", "body": SSE_KEEPALIVE, "more_body": True})
        if answer is not None:
            await answer.wait()
        await send({"type": "http.response.body", "body": b'data: {"answer":"done"}\n\n', "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    capture = _CaptureMiddleware(
        app,
        store=None,
        model_server_name="policy",
        lineage_store=ledger,
        external_staging=True,
        token_capture_enabled=True,
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/ng-rollout/r/training-token-capture/v1/chat/completions",
        "headers": [],
    }
    return asyncio.create_task(PolicyAdmissionMiddleware(capture, gate)(scope, receive, send))


def _answers(sent: list[dict]) -> list[bytes]:
    return [message["body"] for message in sent if message.get("body", b"").startswith(b"data:")]


def watch_delivery(gate: PolicyGate) -> asyncio.Event:
    """Set once a call's first answer chunk reaches the gate, where a checkpoint holds it."""
    delivering = asyncio.Event()
    deliver = gate.deliver_response

    async def delivering_response(ticket) -> None:
        delivering.set()
        await deliver(ticket)

    gate.deliver_response = delivering_response
    return delivering


async def test_a_call_s_ledger_row_is_published_at_its_first_answer_chunk_not_at_headers_or_keepalives(
    tmp_path: Path,
) -> None:
    ledger = FileLineageStore(tmp_path / "ledger")
    participant = PolicyModelParticipant(ledger)
    sent, contexts, answer = [], [], asyncio.Event()
    call = _staged_call(participant.gate, ledger, sent, contexts=contexts, answer=answer)
    await wait_until(lambda: len(sent) == 2)
    rows_before_the_answer = ledger.export_rows("r")
    answer.set()
    await asyncio.wait_for(call, timeout=5)

    assert [message.get("body") for message in sent[:2]] == [None, SSE_KEEPALIVE]
    assert rows_before_the_answer == []
    assert [row["model_call_id"] for row in ledger.export_rows("r")] == [contexts[0].model_call_id]
    assert contexts[0].committed
    assert _answers(sent) == [b'data: {"answer":"done"}\n\n']


async def test_responses_lifecycle_events_are_liveness_not_the_answer(tmp_path: Path) -> None:
    """``stream_keepalive_s`` opens a Responses stream long before generation finishes.

    Its ``response.created`` and ``response.in_progress`` events start no answer: a checkpoint holds the call
    instead of waiting for it, and the call's ledger row is published before its terminal event leaves.
    """
    ledger = FileLineageStore(tmp_path / "ledger")
    participant = PolicyModelParticipant(ledger)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    sent, contexts, lifecycle = [], [], asyncio.Event()
    delivering = watch_delivery(participant.gate)
    sent_at_publication = []
    publish = participant.gate.publish

    async def recording_publish(ticket) -> None:
        sent_at_publication.append(len(sent))
        await publish(ticket)

    participant.gate.publish = recording_publish
    call = _staged_call(participant.gate, ledger, sent, contexts=contexts, lifecycle=lifecycle)
    try:
        await wait_until(lambda: len(sent) == 1 + len(RESPONSES_LIFECYCLE_EVENTS))
        assert not delivering.is_set()
        prepared = await asyncio.wait_for(controller.prepare(CheckpointRequest(**control())), timeout=5)
        assert prepared["phase"] == "prepared"
        assert prepared["report"]["counts"]["held"] == 1
        lifecycle.set()
        await asyncio.wait_for(delivering.wait(), timeout=5)
        assert len(sent) == 1 + len(RESPONSES_LIFECYCLE_EVENTS), "the terminal event waits for resume"
        await controller.resume(CheckpointRequest(**control()))
        await asyncio.wait_for(call, timeout=5)
    finally:
        await controller.resume(CheckpointRequest(**control()))
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)

    assert [message.get("body") for message in sent[1:]] == [*RESPONSES_LIFECYCLE_EVENTS, RESPONSE_COMPLETED_EVENT]
    assert sent_at_publication == [1 + len(RESPONSES_LIFECYCLE_EVENTS)], "published before the terminal event"
    assert [row["model_call_id"] for row in ledger.export_rows("r")] == [contexts[0].model_call_id]
    assert contexts[0].committed


async def test_a_call_that_leaves_before_its_response_starts_leaves_no_row_and_is_not_poisoned(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "ledger")
    participant = PolicyModelParticipant(ledger)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    sent, contexts, answer = [], [], asyncio.Event()
    delivering = watch_delivery(participant.gate)
    call = _staged_call(participant.gate, ledger, sent, contexts=contexts, answer=answer)
    await wait_until(lambda: len(sent) == 2)
    await controller.prepare(CheckpointRequest(**control()))
    answer.set()
    await asyncio.wait_for(delivering.wait(), timeout=5)
    # The client disconnects while the checkpoint holds the answer.
    call.cancel()
    await asyncio.gather(call, return_exceptions=True)
    await controller.resume(CheckpointRequest(**control()))

    # No row and no failure row: the agent makes the call again.
    assert ledger.export_rows("r") == []
    assert not contexts[0].committed
    assert _answers(sent) == []
    assert not participant.gate.tickets


class _FailingLedger(FileLineageStore):
    async def record(self, commit: CaptureLedgerCommit) -> None:
        raise OSError("ledger unavailable")


async def test_a_failed_publication_poisons_the_call_and_sends_no_answer(tmp_path: Path) -> None:
    ledger = _FailingLedger(tmp_path / "ledger")
    participant = PolicyModelParticipant(ledger)
    sent, contexts = [], []
    [outcome] = await asyncio.gather(
        _staged_call(participant.gate, ledger, sent, contexts=contexts), return_exceptions=True
    )

    assert isinstance(outcome, OSError)
    assert _answers(sent) == []
    assert ledger.export_rows("r") == [
        {"model_call_id": contexts[0].model_call_id, "failure_reason": INVALID_COMMIT_COORDS_REASON}
    ]
    assert not participant.gate.tickets


class _SlowLedger(FileLineageStore):
    """A ledger whose writes wait until the test lets them proceed."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.writing, self.proceed = asyncio.Event(), asyncio.Event()
        self.written: list[str] = []

    async def record(self, commit: CaptureLedgerCommit) -> None:
        self.writing.set()
        await self.proceed.wait()
        await super().record(commit)
        self.written.append(commit.record.model_call_id)


async def test_a_retire_during_publication_waits_for_it_to_finish(tmp_path: Path) -> None:
    ledger = _SlowLedger(tmp_path / "ledger")
    participant = PolicyModelParticipant(ledger)
    sent, contexts = [], []
    call = _staged_call(participant.gate, ledger, sent, contexts=contexts)
    await asyncio.wait_for(ledger.writing.wait(), timeout=5)
    retiring = asyncio.Event()
    refuse_waiters = participant.gate.refuse_waiters

    def refusing(episode_ids):
        retiring.set()
        return refuse_waiters(episode_ids)

    participant.gate.refuse_waiters = refusing
    retire = asyncio.create_task(participant.retire(EpisodeId(rollout_id="r")))
    await asyncio.wait_for(retiring.wait(), timeout=5)
    ledger.proceed.set()
    await asyncio.wait_for(retire, timeout=5)
    written_when_the_retire_returned = list(ledger.written)
    await asyncio.gather(call, return_exceptions=True)

    assert written_when_the_retire_returned == [contexts[0].model_call_id]
    assert _answers(sent) == []
    assert not participant.gate.tickets
    # The retire then removed the attempt's ledger, the published row with it.
    assert ledger.export_rows("r") == []


async def test_a_checkpoint_that_closes_during_publication_waits_for_it_and_keeps_the_row(tmp_path: Path) -> None:
    ledger = _SlowLedger(tmp_path / "ledger")
    participant = PolicyModelParticipant(ledger)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    sent, contexts = [], []
    call = _staged_call(participant.gate, ledger, sent, contexts=contexts)
    await asyncio.wait_for(ledger.writing.wait(), timeout=5)
    prepare = asyncio.create_task(controller.prepare(CheckpointRequest(**control())))
    await wait_until(lambda: not participant.gate.accepting)
    # The response has started, so the call blocks the checkpoint until it has published its row and left.
    assert participant.readiness().blockers == ["r"]
    ledger.proceed.set()
    prepared = await asyncio.wait_for(prepare, timeout=5)
    await asyncio.wait_for(call, timeout=5)
    committed = await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
    await controller.resume(CheckpointRequest(**control()))

    rows = (tmp_path / "ckpt" / "gym" / "model" / "policy" / committed["manifest"]["records_file"]).read_text()
    assert prepared["phase"] == "prepared"
    assert _answers(sent) == [b'data: {"answer":"done"}\n\n']
    assert contexts[0].model_call_id in rows


async def test_a_response_held_on_one_policy_server_is_in_no_snapshot_of_the_ledger_it_shares(tmp_path: Path) -> None:
    servers = []
    for name in ("policy", "policy_reasoning_off"):
        participant = PolicyModelParticipant(FileLineageStore(tmp_path / "ledger"), server_name=name)
        servers.append((participant, ParticipantControlPlane(participant, instance_name=name, lease_grace_seconds=60)))
    [(first, _), _] = servers
    await first.ledger.record(
        _commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",))
    )
    sent, contexts, answer = [], [], asyncio.Event()
    delivering = watch_delivery(first.gate)
    call = _staged_call(first.gate, first.ledger, sent, contexts=contexts, answer=answer)
    await wait_until(lambda: len(sent) == 2)
    for _, controller in servers:
        await controller.prepare(CheckpointRequest(**control()))
    answer.set()
    # The first server holds the completed response; the second server has no ticket for it.
    await asyncio.wait_for(delivering.wait(), timeout=5)
    snapshots = []
    for (participant, controller), name in zip(servers, ("policy", "policy_reasoning_off")):
        committed = await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
        snapshots.append(
            (tmp_path / "ckpt" / "gym" / "model" / name / committed["manifest"]["records_file"]).read_text()
        )
    for _, controller in servers:
        await controller.resume(CheckpointRequest(**control()))
    await asyncio.wait_for(call, timeout=5)

    held = contexts[0].model_call_id
    assert all('"c1"' in snapshot and held not in snapshot for snapshot in snapshots)
    # Restoring both into fresh servers sharing one ledger, whichever writes last.
    restored_rows = []
    for name in ("policy", "policy_reasoning_off"):
        participant = PolicyModelParticipant(FileLineageStore(tmp_path / "restored"), server_name=name)
        controller = ParticipantControlPlane(participant, instance_name=name, lease_grace_seconds=60)
        await controller.restore(_restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
        restored_rows.append([row["model_call_id"] for row in participant.ledger.export_rows("r-a1")])
    assert restored_rows == [["c1"], ["c1"]]


async def test_a_call_whose_capture_attaches_after_the_snapshot_cannot_land_its_row_in_the_checkpoint(
    tmp_path: Path,
) -> None:
    ledger = FileLineageStore(tmp_path / "ledger")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",)))
    participant = PolicyModelParticipant(ledger)
    controller = ParticipantControlPlane(participant, instance_name="policy", lease_grace_seconds=60)
    prefetching, prefetched = asyncio.Event(), asyncio.Event()

    async def paused_prefetch(ticket) -> None:
        # Stands in for claiming a restored cut from a coordinator: the ticket has no capture context yet.
        prefetching.set()
        await prefetched.wait()

    participant.gate.restored_cuts.prefetch = paused_prefetch
    reading, call_finalized = threading.Event(), threading.Event()
    export_rows = ledger.export_rows

    def export_rows_once_the_call_finalized(rollout_id: str) -> list[dict]:
        reading.set()
        call_finalized.wait(timeout=5)
        return export_rows(rollout_id)

    ledger.export_rows = export_rows_once_the_call_finalized
    sent, contexts, finalized = [], [], asyncio.Event()
    call = _staged_call(participant.gate, ledger, sent, contexts=contexts, finalized=finalized)
    await asyncio.wait_for(prefetching.wait(), timeout=5)
    prepared = await controller.prepare(CheckpointRequest(**control()))
    commit = asyncio.create_task(controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}])))
    # The commit has taken its snapshot, without this call, and is about to read the ledger.
    await wait_until(reading.is_set)
    prefetched.set()
    await asyncio.wait_for(finalized.wait(), timeout=5)
    call_finalized.set()
    committed = await asyncio.wait_for(commit, timeout=5)
    await controller.resume(CheckpointRequest(**control()))
    await asyncio.wait_for(call, timeout=5)

    rows = (tmp_path / "ckpt" / "gym" / "model" / "policy" / committed["manifest"]["records_file"]).read_text()
    assert prepared["phase"] == "prepared"
    assert '"c1"' in rows and contexts[0].model_call_id not in rows
    assert [row["model_call_id"] for row in export_rows("r")] == ["c1", contexts[0].model_call_id]


def test_a_misspelled_model_server_checkpoint_setting_fails_at_startup() -> None:
    from pydantic import ValidationError

    from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig

    with pytest.raises(ValidationError, match="polciy"):
        BaseResponsesAPIModelConfig.model_validate(
            {"host": "", "port": 0, "entrypoint": "", "name": "m", "checkpoint": {"polciy": True}}
        )
    # The block itself is accepted.
    config = BaseResponsesAPIModelConfig.model_validate(
        {"host": "", "port": 0, "entrypoint": "", "name": "m", "checkpoint": {"policy": True, "generation_cuts": True}}
    )
    assert config.checkpoint.policy and config.checkpoint.generation_cuts


def _model_server_starting(name: str, global_config: dict, ledger: FileLineageStore) -> None:
    """Run a model server's checkpoint setup, as its startup does."""
    from types import SimpleNamespace

    from nemo_gym.base_responses_api_model import ModelCheckpointConfig, SimpleResponsesAPIModel

    entry = next(iter(global_config[name]["responses_api_models"].values()))
    server = SimpleNamespace(
        server_client=SimpleNamespace(global_config_dict=global_config),
        config=SimpleNamespace(
            name=name, checkpoint=ModelCheckpointConfig.model_validate(entry.get("checkpoint", {}))
        ),
    )
    SimpleResponsesAPIModel.setup_policy_checkpoint(server, FastAPI(), ledger)


def test_a_model_server_outside_the_checkpoint_gate_that_writes_the_policy_ledger_is_rejected_at_startup(
    tmp_path: Path,
) -> None:
    def model(**config) -> dict:
        return {"responses_api_models": {"vllm_model": config}}

    def ref(name: str) -> dict:
        return {"type": "responses_api_models", "name": name}

    run = {
        "checkpoint": {"enabled": True, "control_auth_token": "t"},
        "token_id_capture": {"enabled": True, "all_agents": True, "external_staging": True, "rebuild_response": False},
        "policy_model": model(checkpoint={"policy": True}),
        "user_model": model(),
        "judge_model": model(),
        # The agent sends the training-token capture prefix to every model server it calls.
        "agent": {
            "responses_api_agents": {"simple_agent": {"model_server": ref("policy_model"), "user": ref("user_model")}}
        },
        # Only a resources server calls the judge, without the prefix, so the judge never writes the ledger.
        "resources": {"resources_servers": {"judge": {"judge_model_server": ref("judge_model")}}},
    }
    ledger = FileLineageStore(tmp_path / "ledger")

    with pytest.raises(ValueError, match="'user_model'.*'agent'.*'policy_model'"):
        _model_server_starting("user_model", run, ledger)
    _model_server_starting("judge_model", {**run, "user_model": model(checkpoint={"policy": True})}, ledger)
    _model_server_starting(
        "judge_model", {**run, "token_id_capture": {**run["token_id_capture"], "all_agents": False}}, ledger
    )


async def test_a_restore_after_a_crash_removes_the_started_markers_the_dead_execution_left(tmp_path: Path) -> None:
    from nemo_gym._checkpoint.store import write_participant_state

    # Restoring the checkpoint again, and restoring a later one that continues the restored attempt.
    for checkpoint_id, episode in (("c0", {"rollout_id": "r"}), ("c1", {"rollout_id": "r", "attempt": 1})):
        record = ModelRecord(episode_id=EpisodeId(**episode), rows=[{"model_call_id": "c1"}])
        write_participant_state(
            tmp_path / checkpoint_id,
            kind="model",
            instance="policy",
            checkpoint_id=checkpoint_id,
            records=[record.to_json_record()],
        )
        # The execution that crashed had started the restored attempt r-a1.
        FileLineageStore(tmp_path / "ledger").mark_started("r-a1")
        _, _, controller = await _ledger_participant(tmp_path / "ledger")
        await controller.restore(_restore_request("r1", tmp_path / checkpoint_id, [episode]))

        assert not (tmp_path / "ledger" / "r-a1.lineage.started").exists()


def test_marking_a_restored_attempt_started_does_not_sync_the_ledger_directory(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path)
    syncs = []
    ledger._fsync_ledger_root = lambda: syncs.append(True)
    ledger.mark_started("r-a1")

    assert (tmp_path / "r-a1.lineage.started").exists()
    assert syncs == []
