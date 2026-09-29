# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from nemo_gym._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ParticipantController,
    RestoreRequest,
    install_participant,
)
from nemo_gym._checkpoint.errors import ControlError, StaleAttemptError
from nemo_gym._checkpoint.generation_cut import GenerationCutInventory, GenerationCutPrefixAck, GenerationCutReceipt
from nemo_gym._checkpoint.model import PolicyAdmissionMiddleware, PolicyModelParticipant, attach_capture_context
from nemo_gym.rollout_correlation import RolloutContextMiddleware, current_rollout_id
from nemo_gym.token_id_capture.lineage import FileLineageStore
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
    app.add_middleware(PolicyAdmissionMiddleware, participant=participant)
    return app, participant, release


def control(checkpoint_id: str = "c1", **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


async def test_started_stream_drains_before_prepare_is_ready_and_new_calls_are_parked() -> None:
    app, participant, release = make_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        stream = asyncio.create_task(client.post("/ng-rollout/r-a1/v1/chat/completions"))
        await asyncio.sleep(0.05)
        prepare = asyncio.create_task(client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH))
        await asyncio.sleep(0.05)
        assert not prepare.done()
        assert participant.readiness().blockers == ["r-a1"]

        parked = await client.post("/ng-rollout/other/v1/chat/completions")
        health = await client.get("/health")
        release.set()
        streamed = await stream
        prepared = (await prepare).json()
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        reopened = await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)

    assert parked.status_code == 409 and parked.json()["error"]["code"] == "checkpoint_parked"
    assert health.status_code == 200
    assert streamed.content == b"first last"
    assert prepared["phase"] == "prepared"
    assert reopened.json()["phase"] == "idle" and participant.accepting


async def test_undelivered_generation_does_not_block_prepare_and_is_held_until_resume() -> None:
    app, participant, _ = make_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        call = asyncio.create_task(client.post("/ng-rollout/r/v1/responses"))
        await asyncio.sleep(0.05)
        prepared = (await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)).json()
        app.state.generation_done.set()
        await asyncio.sleep(0.05)
        held = not call.done()
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        delivered = await asyncio.wait_for(call, timeout=5)

    assert prepared["phase"] == "prepared" and prepared["report"]["counts"]["held"] == 1
    assert held
    assert delivered.json() == {"output": "done"}


async def test_retire_cancels_a_held_generation_and_fences_its_attempt() -> None:
    app, participant, _ = make_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        held = asyncio.create_task(client.post("/ng-rollout/r/v1/responses"))
        await asyncio.sleep(0.05)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post(
            "/ng-control/v1/checkpoint/retire", json=control(episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        stale = await client.post("/ng-rollout/r/v1/responses")
        await asyncio.gather(held, return_exceptions=True)

    assert stale.status_code == 409 and stale.json()["error"]["code"] == "stale_attempt"
    assert participant.readiness().counts["inflight"] == 0


async def _ledger_participant(root: Path) -> tuple[FileLineageStore, PolicyModelParticipant, ParticipantController]:
    ledger = FileLineageStore(root)
    participant = PolicyModelParticipant(ledger)
    return ledger, participant, ParticipantController(participant, instance_name="policy", lease_grace_seconds=60)


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
    with pytest.raises(StaleAttemptError):
        restored.admit("r")


async def test_model_commit_requires_the_continued_episodes(tmp_path: Path) -> None:
    _, _, controller = await _ledger_participant(tmp_path / "store")
    await controller.prepare(CheckpointRequest(**control()))

    with pytest.raises(ControlError, match="episode_ids"):
        await controller.commit(_commit_request("c1", tmp_path / "ckpt"))


async def test_restore_refuses_to_merge_into_a_foreign_ledger(tmp_path: Path) -> None:
    ledger, _, controller = await _ledger_participant(tmp_path / "before")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r"))
    await ledger.record(_commit(_call_record("c9"), [USER_3], [ASSISTANT_1], rollout_id="s"))
    await controller.prepare(CheckpointRequest(**control()))
    await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}, {"rollout_id": "s"}]))

    restored_ledger, _, restored_controller = await _ledger_participant(tmp_path / "after")
    await restored_ledger.record(_commit(_call_record("other"), [USER_2], [ASSISTANT_2], rollout_id="s-a1"))
    with pytest.raises(ControlError, match="s-a1"):
        await restored_controller.restore(
            _restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}, {"rollout_id": "s"}])
        )

    assert not await restored_ledger.has_rows("r-a1")


def _held_call(participant: PolicyModelParticipant, capture_key: str, model_call_id: str, request_items: list) -> None:
    ticket = participant.enter(capture_key)
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
    controller = ParticipantController(participant, instance_name="policy", lease_grace_seconds=60)
    # c2 is in flight; its ledger row was written but the agent never received the response.
    _held_call(participant, "r", "c2", [USER_1, ASSISTANT_1, USER_2])
    await ledger.record(_commit(_call_record("c2"), [USER_1, ASSISTANT_1, USER_2], [ASSISTANT_2], rollout_id="r"))

    prepared = await controller.prepare(CheckpointRequest(**control()))
    await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))

    restored_ledger = FileLineageStore(tmp_path / "after")
    restored = PolicyModelParticipant(restored_ledger, server_name="policy")
    await ParticipantController(restored, instance_name="policy", lease_grace_seconds=60).restore(
        _restore_request("r1", tmp_path / "ckpt", [{"rollout_id": "r"}])
    )
    admission = CaptureAdmission(rollout_id="r-a1", model_call_id="c3", mode="text")
    other_request = CaptureContext(rollout_id="r-a1", model_call_id="c3", token_sink=None, request_items=[USER_3])
    reissued = CaptureContext(
        rollout_id="r-a1", model_call_id="c3", token_sink=None, request_items=[USER_1, ASSISTANT_1, USER_2]
    )

    assert prepared["report"]["counts"] == {"inflight": 1, "held": 1, "cut": 1}
    assert [inventory.active_prefixes[0].model_call_id for _, inventory in requests] == ["c2"]
    assert [row["model_call_id"] for row in restored_ledger.export_rows("r-a1")] == ["c1"]
    assert restored.continue_restored_cut(other_request, admission).generation_cut is None
    continued = restored.continue_restored_cut(reissued, admission).generation_cut
    assert (continued.source_model_call_id, continued.staging_keys) == ("c2", ("__generation_cut__/c2",))
    assert restored.continue_restored_cut(reissued, admission).generation_cut is None


async def test_failed_cut_regenerates_instead_of_blocking(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "before")
    request_cut, _ = _cutting_worker(cut=False)
    participant = PolicyModelParticipant(ledger, cut_requester=request_cut)
    controller = ParticipantController(participant, instance_name="policy", lease_grace_seconds=60)
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

    prepared = await ParticipantController(participant, instance_name="policy", lease_grace_seconds=60).prepare(
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
    app.add_middleware(PolicyAdmissionMiddleware, participant=participant)
    return app


async def test_undelivered_rows_are_excluded_on_any_capture_path(tmp_path: Path) -> None:
    ledger = FileLineageStore(tmp_path / "store")
    await ledger.record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1], rollout_id="r", staging_chain=("r/c1",)))
    participant = PolicyModelParticipant(ledger)
    controller = ParticipantController(participant, instance_name="policy", lease_grace_seconds=60)
    release = asyncio.Event()
    app = _captured_app(participant, ledger, release)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        call = asyncio.create_task(client.post("/ng-rollout/r/v1/chat/completions"))
        await asyncio.sleep(0.05)
        await controller.prepare(CheckpointRequest(**control()))
        release.set()
        await asyncio.sleep(0.05)
        live_rows = [row["model_call_id"] for row in ledger.export_rows("r")]
        commit = await controller.commit(_commit_request("c1", tmp_path / "ckpt", [{"rollout_id": "r"}]))
        await controller.resume(CheckpointRequest(**control()))
        await call

    rows = (tmp_path / "ckpt" / "gym" / "model" / "policy" / "records.jsonl").read_text()
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
        generation_token_count=3,
        digest="d" * 64,
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
    assert second.continue_restored_cut(context, admission).generation_cut is None
    assert first.continue_restored_cut(context, admission).generation_cut == continuation


async def test_a_restored_cut_not_yet_reused_survives_the_next_checkpoint(tmp_path: Path) -> None:
    from nemo_gym._checkpoint.model import GenerationCutRecord

    continuation = GenerationCutContinuation(
        source_capture_key="r",
        source_model_call_id="c2",
        staging_keys=("__generation_cut__/c2",),
        generation_token_count=3,
        digest="d" * 64,
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
