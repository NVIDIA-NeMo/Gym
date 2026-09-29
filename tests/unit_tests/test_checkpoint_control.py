# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym._checkpoint.control import (
    AttemptFence,
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    PrepareReport,
    StaleAttemptError,
    install_participant,
)
from nemo_gym.episode_types import EpisodeId


TOKEN = "secret"
AUTH = {"authorization": f"Bearer {TOKEN}"}


class Record(CheckpointRecord):
    value: int


class FakeParticipant(CheckpointParticipant):
    """Executions are either running or parked; prepare is ready once none are running."""

    kind = "fake"
    record_model = Record

    def __init__(self) -> None:
        super().__init__()
        self.accepting = True
        self.executions: dict[str, dict] = {}
        self.fail_close = False

    async def close_admission(self, request: CheckpointRequest) -> None:
        if self.fail_close:
            raise RuntimeError("close failed")
        self.accepting = False

    async def open_admission(self) -> None:
        self.accepting = True

    def readiness(self) -> PrepareReport:
        running = [key for key, execution in self.executions.items() if not execution["parked"]]
        return PrepareReport(ready=not running, blockers=running, counts={"live": len(self.executions)})

    async def retire(self, episode_id: EpisodeId) -> None:
        self.executions.pop(episode_id.capture_key, None)

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[BaseModel]:
        return [
            Record(episode_id=EpisodeId.from_capture_key(key), value=execution["value"])
            for key, execution in self.executions.items()
        ]

    def restore_records(self, records: list[BaseModel]) -> None:
        for record in records:
            if record.value < 0:
                raise ValueError("invalid record")
        for record in records:
            next_id = record.episode_id.model_copy(update={"attempt": record.episode_id.attempt + 1})
            self.executions[next_id.capture_key] = {"parked": True, "value": record.value}

    async def park(self, key: str) -> None:
        self.executions[key]["parked"] = True
        await self.notify()


def make_client(participant: FakeParticipant, *, lease_grace: float = 60) -> httpx.AsyncClient:
    app = FastAPI()
    install_participant(app, participant, auth_token=TOKEN, instance_name="fake-1", lease_grace_seconds=lease_grace)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", headers=AUTH)


def body(checkpoint_id: str = "ckpt-1", *, timeout: float = 5.0, **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + timeout, **extra}


async def test_prepare_waits_for_parking_and_replays() -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": False, "value": 1}
    async with make_client(participant) as client:
        pending = asyncio.create_task(client.post("/ng-control/v1/checkpoint/prepare", json=body()))
        await asyncio.sleep(0.05)
        assert not pending.done()
        assert participant.accepting is False

        await participant.park("r")
        first = (await pending).json()
        again = (await client.post("/ng-control/v1/checkpoint/prepare", json=body())).json()

    assert first["phase"] == "prepared"
    assert again == first


async def test_prepare_that_misses_deadline_stays_closed_until_straggler_is_retired() -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": False, "value": 1}
    participant.executions["s-a1"] = {"parked": True, "value": 2}
    async with make_client(participant) as client:
        missed = (await client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=0.1))).json()
        assert missed["phase"] == "preparing"
        assert missed["report"]["blockers"] == ["r"]
        assert participant.accepting is False

        retired = await client.post(
            "/ng-control/v1/checkpoint/retire", json=body(episode_ids=[{"rollout_id": "r", "attempt": 0}])
        )
        assert retired.status_code == 200
        ready = (await client.post("/ng-control/v1/checkpoint/prepare", json=body())).json()

    assert ready["phase"] == "prepared"
    with pytest.raises(StaleAttemptError):
        participant.attempts.check(EpisodeId(rollout_id="r", attempt=0))


async def test_resume_interrupts_a_waiting_prepare_and_retires_the_checkpoint() -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": False, "value": 1}
    async with make_client(participant) as client:
        pending = asyncio.create_task(client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=30)))
        await asyncio.sleep(0.05)
        resumed = (await client.post("/ng-control/v1/checkpoint/resume", json=body())).json()
        interrupted = (await asyncio.wait_for(pending, timeout=5)).json()
        stale = await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        replayed_resume = (await client.post("/ng-control/v1/checkpoint/resume", json=body())).json()

    assert resumed["phase"] == "idle"
    assert interrupted["phase"] == "idle"
    assert participant.accepting is True
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "stale_checkpoint"
    assert replayed_resume["idempotent"] is True


async def test_failed_prepare_reopens_admission() -> None:
    participant = FakeParticipant()
    participant.fail_close = True
    async with make_client(participant) as client:
        with pytest.raises(RuntimeError, match="close failed"):
            await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        status = (await client.get("/ng-control/v1/checkpoint/status")).json()

    assert status["phase"] == "idle" and status["checkpoint_id"] is None
    assert participant.accepting is True


async def test_commit_restore_round_trip_and_immutable_manifest(tmp_path: Path) -> None:
    source = FakeParticipant()
    source.executions["r-a2"] = {"parked": True, "value": 7}
    async with make_client(source) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        commit = (
            await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))
        ).json()
        retry = (await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))).json()
        await client.post("/ng-control/v1/checkpoint/resume", json=body())

    assert commit == retry
    assert commit["manifest"]["record_count"] == 1

    other = FakeParticipant()
    other.executions["q"] = {"parked": True, "value": 1}
    async with make_client(other) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body("ckpt-2"))
        clobber = await client.post(
            "/ng-control/v1/checkpoint/commit", json=body("ckpt-2", checkpoint_dir=str(tmp_path))
        )
    assert clobber.status_code == 422 and clobber.json()["error"]["code"] == "invalid_checkpoint_state"

    restored = FakeParticipant()
    async with make_client(restored) as client:
        result = (
            await client.post(
                "/ng-control/v1/checkpoint/restore",
                json=body("restore-1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r", "attempt": 2}]),
            )
        ).json()
        assert restored.accepting is False
        await client.post("/ng-control/v1/checkpoint/resume", json=body("restore-1"))

    assert result == {"phase": "restored", "source_checkpoint_id": "ckpt-1", "restored": ["r-a3"]}
    assert restored.executions == {"r-a3": {"parked": True, "value": 7}}
    assert restored.accepting is True
    with pytest.raises(StaleAttemptError):
        restored.attempts.check(EpisodeId(rollout_id="r", attempt=2))


async def test_restore_rejects_corrupt_records_without_installing(tmp_path: Path) -> None:
    source = FakeParticipant()
    source.executions["r"] = {"parked": True, "value": 7}
    async with make_client(source) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))
    records = tmp_path / "gym" / "fake" / "fake-1" / "records.jsonl"
    records.write_text(json.dumps({"episode_id": {"rollout_id": "r", "attempt": 0}, "value": 8}) + "\n")

    restored = FakeParticipant()
    async with make_client(restored) as client:
        response = await client.post(
            "/ng-control/v1/checkpoint/restore",
            json=body("restore-1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
        )

    assert response.status_code == 422 and "digest" in response.json()["error"]["detail"]
    assert restored.executions == {}
    assert restored.accepting is True


async def test_invalid_record_leaves_participant_open_and_idle(tmp_path: Path) -> None:
    source = FakeParticipant()
    source.executions["r"] = {"parked": True, "value": -1}
    async with make_client(source) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))

    restored = FakeParticipant()
    async with make_client(restored) as client:
        with pytest.raises(ValueError, match="invalid record"):
            await client.post(
                "/ng-control/v1/checkpoint/restore",
                json=body("restore-1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            )
        status = (await client.get("/ng-control/v1/checkpoint/status")).json()

    assert status["phase"] == "idle"
    assert restored.accepting is True and restored.executions == {}


async def test_commit_requires_prepared_and_routes_require_bearer(tmp_path: Path) -> None:
    participant = FakeParticipant()
    async with make_client(participant) as client:
        early = await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))
        unauthorized = await client.get("/ng-control/v1/checkpoint/status", headers={"authorization": "Bearer no"})

    assert early.status_code == 409 and early.json()["error"]["code"] == "invalid_phase"
    assert unauthorized.status_code == 401


def test_attempt_fence_keeps_only_the_minimum_live_attempt() -> None:
    fence = AttemptFence()
    fence.retire(EpisodeId(rollout_id="r", attempt=0))
    fence.retire(EpisodeId(rollout_id="r", attempt=1))
    fence.retire(EpisodeId(rollout_id="r", attempt=0))

    fence.check(EpisodeId(rollout_id="r", attempt=2))
    fence.check(EpisodeId(rollout_id="other", attempt=0))
    for attempt in (0, 1):
        with pytest.raises(StaleAttemptError):
            fence.check(EpisodeId(rollout_id="r", attempt=attempt))


async def test_retire_during_a_waiting_prepare_unblocks_it() -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": False, "value": 1}
    async with make_client(participant) as client:
        started = time.time()
        pending = asyncio.create_task(client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=30)))
        await asyncio.sleep(0.05)
        await client.post("/ng-control/v1/checkpoint/retire", json=body(episode_ids=[{"rollout_id": "r"}]))
        prepared = (await asyncio.wait_for(pending, timeout=5)).json()

    assert prepared["phase"] == "prepared"
    assert time.time() - started < 5


async def test_commit_refuses_a_participant_that_is_no_longer_ready(tmp_path: Path) -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        participant.executions["r"]["parked"] = False
        refused = await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))

    assert refused.status_code == 409 and "no longer ready" in refused.json()["error"]["detail"]
    assert not (tmp_path / "gym").exists()


async def test_lease_expiry_resumes_a_participant_whose_controller_went_quiet() -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}
    async with make_client(participant, lease_grace=0.2) as client:
        prepared = (await client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=0.1))).json()
        closed_after_prepare = participant.accepting is False
        await asyncio.sleep(0.5)
        status = (await client.get("/ng-control/v1/checkpoint/status")).json()
        stale = await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir="/unused"))

    assert prepared["phase"] == "prepared" and closed_after_prepare
    assert status["phase"] == "idle" and participant.accepting
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "stale_checkpoint"


async def test_renew_keeps_a_slow_but_live_controller_in_charge() -> None:
    participant = FakeParticipant()
    async with make_client(participant, lease_grace=0.3) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=0.05))
        for _ in range(4):
            await asyncio.sleep(0.15)
            renewed = await client.post("/ng-control/v1/checkpoint/renew", json=body(timeout=0.05))
            assert renewed.status_code == 200
        status = (await client.get("/ng-control/v1/checkpoint/status")).json()

    assert status["phase"] == "prepared" and participant.accepting is False


async def test_restore_installs_only_the_episodes_the_controller_continues(tmp_path: Path) -> None:
    source = FakeParticipant()
    source.executions["keep"] = {"parked": True, "value": 1}
    source.executions["drop"] = {"parked": True, "value": 2}
    async with make_client(source) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))

    restored = FakeParticipant()
    async with make_client(restored) as client:
        result = (
            await client.post(
                "/ng-control/v1/checkpoint/restore",
                json=body("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "keep"}]),
            )
        ).json()

    assert result["restored"] == ["keep-a1"]
    assert restored.executions == {"keep-a1": {"parked": True, "value": 1}}
    restored.attempts.check(EpisodeId(rollout_id="drop"))
    with pytest.raises(StaleAttemptError):
        restored.attempts.check(EpisodeId(rollout_id="keep"))


async def test_restored_state_can_be_retired_after_resume(tmp_path: Path) -> None:
    source = FakeParticipant()
    source.executions["r"] = {"parked": True, "value": 1}
    async with make_client(source) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))

    restored = FakeParticipant()
    async with make_client(restored) as client:
        scope = [{"rollout_id": "r"}]
        await client.post(
            "/ng-control/v1/checkpoint/restore", json=body("r1", checkpoint_dir=str(tmp_path), episode_ids=scope)
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=body("r1"))
        # The controller decided not to continue r after all.
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire", json=body("cleanup", episode_ids=[{"rollout_id": "r", "attempt": 1}])
        )

    assert retired.status_code == 200
    assert restored.executions == {}
    with pytest.raises(StaleAttemptError):
        restored.attempts.check(EpisodeId(rollout_id="r", attempt=1))


async def test_commit_io_past_the_deadline_fails_and_a_retry_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nemo_gym._checkpoint.control as control

    real_write = control.write_participant_state
    slow = {"on": True}

    def maybe_slow_write(*args, **kwargs):
        if slow["on"]:
            time.sleep(0.4)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(control, "write_participant_state", maybe_slow_write)
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        late = await client.post(
            "/ng-control/v1/checkpoint/commit", json=body(timeout=0.1, checkpoint_dir=str(tmp_path))
        )
        await asyncio.sleep(0.5)
        slow["on"] = False
        retried = await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))

    assert late.status_code == 409 and late.json()["error"]["code"] == "deadline_exceeded"
    assert retried.status_code == 200 and retried.json()["manifest"]["record_count"] == 1


def test_prepare_report_caps_listed_blockers_but_counts_all() -> None:
    report = PrepareReport(ready=False, blockers=[f"r{i}" for i in range(5000)])

    assert len(report.blockers) == 100 and report.blocker_count == 5000
    assert PrepareReport.model_validate(report.model_dump()) == report


def test_seed_reply_reports_the_verify_mode_and_defaults_to_wait() -> None:
    from multidict import CIMultiDict

    from nemo_gym._checkpoint.steps import seed_verify_mode

    assert seed_verify_mode(CIMultiDict({"X-NG-Checkpoint-Verify": "replay"})) == "replay"
    assert seed_verify_mode({"x-ng-checkpoint-verify": "wait"}) == "wait"
    assert seed_verify_mode({"x-ng-checkpoint-verify": "unexpected"}) == "wait"
    assert seed_verify_mode({}) == "wait"
    assert seed_verify_mode(None) == "wait"


def test_a_fence_copied_to_another_process_only_rises() -> None:
    from nemo_gym._checkpoint.control import AttemptFence

    source, copy = AttemptFence(), AttemptFence()
    source.retire(EpisodeId(rollout_id="r", attempt=1))
    copy.retire(EpisodeId(rollout_id="s", attempt=3))
    copy.raise_to(source.minimums())
    copy.raise_to({"s": 1})

    assert copy.minimums() == {"r": 2, "s": 4}


async def test_an_episode_woken_by_resume_stays_parked_if_a_new_checkpoint_closes_first() -> None:
    from nemo_gym._checkpoint.steps import EpisodeSteps

    async def notify() -> None:
        return None

    steps = EpisodeSteps(notify)
    progressed = asyncio.Event()

    async def episode() -> None:
        steps.begin("r")
        await steps.boundary("r", {"next": "b"})
        progressed.set()

    steps.close()
    task = asyncio.create_task(episode())
    await asyncio.sleep(0)
    # Resume and a new close before the parked episode gets to run again.
    steps.open()
    steps.close()
    await asyncio.sleep(0.05)
    parked_through_second_checkpoint = not progressed.is_set() and steps.blockers() == []
    steps.open()
    await asyncio.wait_for(task, 1)

    assert parked_through_second_checkpoint
    assert progressed.is_set()


async def test_control_calls_use_a_reserved_pool_when_data_calls_fill_theirs() -> None:
    from aiohttp import web

    import nemo_gym.server_utils as server_utils

    release = asyncio.Event()

    async def slow(request: web.Request) -> web.Response:
        await release.wait()
        return web.json_response({"slow": True})

    async def fast(request: web.Request) -> web.Response:
        return web.json_response({"fast": True})

    app = web.Application()
    app.router.add_get("/slow", slow)
    app.router.add_get("/fast", fast)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    saved = (server_utils._GLOBAL_AIOHTTP_CLIENT, server_utils._GLOBAL_AIOHTTP_CONTROL_CLIENT)
    server_utils._GLOBAL_AIOHTTP_CLIENT = server_utils._GLOBAL_AIOHTTP_CONTROL_CLIENT = None
    try:
        # One data connection per host, and a long call holding it.
        server_utils.set_global_aiohttp_client(
            server_utils.GlobalAIOHTTPAsyncClientConfig(global_aiohttp_connector_limit_per_host=1)
        )
        held = asyncio.create_task(server_utils.request("GET", f"http://127.0.0.1:{port}/slow"))
        await asyncio.sleep(0.1)
        control = await asyncio.wait_for(
            server_utils.request("GET", f"http://127.0.0.1:{port}/fast", _control=True), 2
        )
        data = asyncio.create_task(server_utils.request("GET", f"http://127.0.0.1:{port}/fast"))
        await asyncio.sleep(0.2)
        data_waited = not data.done()
        release.set()
        await held
        await data
    finally:
        await server_utils._GLOBAL_AIOHTTP_CLIENT.close()
        if server_utils._GLOBAL_AIOHTTP_CONTROL_CLIENT is not None:
            await server_utils._GLOBAL_AIOHTTP_CONTROL_CLIENT.close()
        server_utils._GLOBAL_AIOHTTP_CLIENT, server_utils._GLOBAL_AIOHTTP_CONTROL_CLIENT = saved
        await runner.cleanup()

    assert control.status == 200
    assert data_waited


def test_stored_records_keep_their_key_order(tmp_path: Path) -> None:
    """A restored episode must see its state exactly as exported, including the order of every dict's keys."""
    from nemo_gym._checkpoint.store import read_participant_state, write_participant_state

    state = {"zeta": 1, "alpha": {"tools": [{"name": "b", "args": {"y": 2, "x": 1}}], "beta": None}, "mid": "m"}
    record = {"episode_id": {"rollout_id": "r", "attempt": 0}, "state": state}
    write_participant_state(tmp_path, kind="resources", instance="r", checkpoint_id="c1", records=[record])

    _, [restored] = read_participant_state(tmp_path, kind="resources", instance="r")

    assert json.dumps(restored["state"]) == json.dumps(state)
    assert list(restored) == list(record)
