# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
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
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.rollout_correlation import RolloutContextMiddleware, current_rollout_id
from nemo_gym.token_id_capture.control_routes import install_rollout_control_routes
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.protocols import RolloutRetiredError
from nemo_gym.token_id_capture.records import ParentResolutionStatus
from nemo_gym.token_id_capture.sink import CaptureContext
from nemo_gym.token_id_capture.staging.records import CaptureAdmission, GenerationCutContinuation, RolloutManifest
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

    assert refused.status_code == 409 and refused.json()["error"]["code"] == "stale_attempt"


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
    assert len(restored.retired) == 0
    # The carried-over call stays staged under the attempt that made it.
    assert [record.capture_key for record in manifest.records] == ["r"]


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
    admission = CaptureAdmission(rollout_id="r-a1", model_call_id="c3", mode="text")
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
    admission = CaptureAdmission(rollout_id="r", model_call_id="c9", mode="text")
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
def test_constrained_decoding_must_restart_instead_of_continuing_a_cut(body: dict, reason: str | None) -> None:
    from nemo_gym._checkpoint.model import generation_cut_restart_reason

    assert generation_cut_restart_reason(body) == reason


async def test_a_constrained_call_is_not_cut_and_never_continues_a_restored_cut(tmp_path: Path) -> None:
    request_cut, requests = _cutting_worker()
    participant = PolicyModelParticipant(FileLineageStore(tmp_path / "ledger"), cut_requester=request_cut)
    _held_call(participant, "r", "c1", [USER_1])
    constrained = participant.gate.enter("s")
    constrained.backend = "http://worker-0/v1"
    constrained.capture = CaptureContext(rollout_id="s", model_call_id="c2", token_sink=None, request_items=[USER_1])
    constrained.cut_restart_reason = "tool_choice:required"
    await participant.close_admission(CheckpointRequest(**control()))
    report = participant.readiness()

    participant._restored_cuts["s"] = restored = participant.gate.snapshot().cuts[0].record.model_copy()
    admission = CaptureAdmission(rollout_id="s", model_call_id="c9", mode="text")
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
    assert target.rows["r-a1"] == [{"model_call_id": "c1", "staging_key": "r/c1", "capture_key": "r"}]


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
    assert FileLineageStore(tmp_path).export_rows("roll-1-a1") == [{**row, "capture_key": "roll-1"}]


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

    assert ledger.rows["r-a1"] == [{"model_call_id": "c1", "staging_key": "r/c1", "capture_key": "r"}, failure]
    assert ledger.rows["r-a2"] == [
        {"model_call_id": "c1", "staging_key": "r/c1", "capture_key": "r"},
        failure,
        {"model_call_id": "c2", "staging_key": "r-a1/c2", "capture_key": "r-a1"},
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
    assert await participant.restored_pending() == []


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
        assert refused.value.status_code == 400 and refused.value.code == "invalid_rollout_id"


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
