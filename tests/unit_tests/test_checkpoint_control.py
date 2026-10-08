# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError

from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    JsonPayload,
    PrepareReport,
    RetiredAttempts,
    StaleAttemptError,
    install_participant,
)
from nemo_gym._checkpoint.errors import CheckpointStateError
from nemo_gym._checkpoint.steps import EpisodeSteps
from nemo_gym._checkpoint.store import participant_dir, read_participant_state, write_participant_state
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
        scope = None if episode_ids is None else {episode_id.capture_key for episode_id in episode_ids}
        return [
            Record(episode_id=EpisodeId.from_capture_key(key), value=execution["value"])
            for key, execution in self.executions.items()
            # Restored state outside the commit's scope is not exported: the commit retires it.
            if scope is None or not execution.get("restored") or key in scope
        ]

    def restore_records(self, records: list[BaseModel]) -> None:
        for record in records:
            if record.value < 0:
                raise ValueError("invalid record")
        for record in records:
            next_id = record.episode_id.model_copy(update={"attempt": record.episode_id.attempt + 1})
            self.executions[next_id.capture_key] = {"parked": True, "value": record.value}

    async def restored_pending(self) -> list[EpisodeId]:
        return [
            EpisodeId.from_capture_key(key) for key, execution in self.executions.items() if execution.get("restored")
        ]

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
        await _until(lambda: participant.accepting is False, "the prepare closed admission")
        assert not pending.done()

        await participant.park("r")
        first = (await pending).json()
        again = (await client.post("/ng-control/v1/checkpoint/prepare", json=body())).json()

    assert first["phase"] == "prepared"
    assert again == first


async def test_a_straggler_is_retired_after_resume_not_while_a_checkpoint_is_open() -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": False, "value": 1}
    participant.executions["s-a1"] = {"parked": True, "value": 2}
    straggler = {"episode_ids": [{"rollout_id": "r", "attempt": 0}]}
    async with make_client(participant) as client:
        missed = (await client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=0.1))).json()
        assert missed["phase"] == "preparing"
        assert missed["report"]["blockers"] == ["r"]
        assert participant.accepting is False

        refused = await client.post("/ng-control/v1/checkpoint/retire", json=body(**straggler))
        await client.post("/ng-control/v1/checkpoint/resume", json=body())
        retired = await client.post("/ng-control/v1/checkpoint/retire", json=body("retire", **straggler))
        ready = (await client.post("/ng-control/v1/checkpoint/prepare", json=body("ckpt-2"))).json()

    assert refused.status_code == 409 and refused.json()["error"]["code"] == "invalid_phase"
    assert retired.status_code == 200
    assert ready["phase"] == "prepared"
    # Stopped and freed; only the rollout's refusal remains, until the controller forgets it.
    assert participant.executions.keys() == {"s-a1"}
    assert len(participant.retired) == 1


async def test_resume_interrupts_a_waiting_prepare_and_retires_the_checkpoint() -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": False, "value": 1}
    async with make_client(participant) as client:
        pending = asyncio.create_task(client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=30)))
        await _until(lambda: participant.accepting is False, "the prepare closed admission")
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
    # A restore runs in a fresh process, so no replaced attempt is running and nothing needs refusing.
    assert len(restored.retired) == 0


async def test_restore_rejects_corrupt_records_without_installing(tmp_path: Path) -> None:
    source = FakeParticipant()
    source.executions["r"] = {"parked": True, "value": 7}
    async with make_client(source) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))
    directory = tmp_path / "gym" / "fake" / "fake-1"
    records = directory / json.loads((directory / "manifest.json").read_text())["records_file"]
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


def test_a_retired_attempt_stays_refused_until_its_rollout_is_forgotten() -> None:
    retired = RetiredAttempts()
    retired.mark([EpisodeId(rollout_id="r", attempt=1)])
    # A later retire of an earlier attempt never lowers the mark.
    retired.mark([EpisodeId(rollout_id="r", attempt=0)])
    for attempt in (0, 1):
        with pytest.raises(StaleAttemptError):
            retired.check(EpisodeId(rollout_id="r", attempt=attempt))
    retired.check(EpisodeId(rollout_id="r", attempt=2))
    retired.check(EpisodeId(rollout_id="other", attempt=0))
    assert len(retired) == 1

    joined = RetiredAttempts()
    joined.update(retired.marks())
    with pytest.raises(StaleAttemptError):
        joined.check(EpisodeId(rollout_id="r", attempt=1))

    retired.forget(["r", "never-retired"])
    retired.check(EpisodeId(rollout_id="r", attempt=0))
    assert len(retired) == 0


async def test_a_request_arriving_after_retire_finished_is_refused_until_forget() -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": False, "value": 1}
    async with make_client(participant) as client:
        retired = await client.post("/ng-control/v1/checkpoint/retire", json=body(episode_ids=[{"rollout_id": "r"}]))
        # The caller sent this request before its own retire; it reaches this server only now.
        with pytest.raises(StaleAttemptError):
            participant.retired.check(EpisodeId(rollout_id="r"))
        forgotten = await client.post("/ng-control/v1/checkpoint/forget", json=body(rollout_ids=["r"]))
        status = (await client.get("/ng-control/v1/checkpoint/status")).json()

    assert retired.status_code == 200 and forgotten.json() == {"forgotten": ["r"]}
    participant.retired.check(EpisodeId(rollout_id="r"))
    assert status["retired_rollouts"] == 0


async def test_forget_is_refused_while_a_checkpoint_is_open() -> None:
    participant = FakeParticipant()
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/retire", json=body("retire", episode_ids=[{"rollout_id": "r"}]))
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        refused = await client.post("/ng-control/v1/checkpoint/forget", json=body(rollout_ids=["r"]))

    assert refused.status_code == 409 and refused.json()["error"]["code"] == "invalid_phase"
    assert len(participant.retired) == 1


async def test_a_failed_retire_keeps_its_attempts_refused_and_a_retry_stops_them() -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": False, "value": 1}
    participant.executions["s"] = {"parked": False, "value": 2}
    original_retire = participant.retire
    fail = {"s"}

    async def flaky_retire(episode_id: EpisodeId) -> None:
        if episode_id.capture_key in fail:
            raise RuntimeError("could not stop")
        await original_retire(episode_id)

    participant.retire = flaky_retire
    batch = {"episode_ids": [{"rollout_id": "r"}, {"rollout_id": "s"}]}
    async with make_client(participant) as client:
        with pytest.raises(RuntimeError):
            await client.post("/ng-control/v1/checkpoint/retire", json=body(**batch))
        for key in ("r", "s"):
            with pytest.raises(StaleAttemptError):
                participant.retired.check(EpisodeId(rollout_id=key))
        fail.clear()
        retried = await client.post("/ng-control/v1/checkpoint/retire", json=body(**batch))

    assert retried.status_code == 200 and participant.executions == {}
    with pytest.raises(StaleAttemptError):
        participant.retired.check(EpisodeId(rollout_id="s"))


async def test_retire_replies_only_after_the_participant_stopped_the_attempt() -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": False, "value": 1}
    stopping = asyncio.Event()
    finish = asyncio.Event()
    refused_while_stopping = []
    original_retire = participant.retire

    async def slow_retire(episode_id: EpisodeId) -> None:
        stopping.set()
        await finish.wait()
        await original_retire(episode_id)

    participant.retire = slow_retire
    async with make_client(participant) as client:
        retire = asyncio.create_task(
            client.post("/ng-control/v1/checkpoint/retire", json=body(episode_ids=[{"rollout_id": "r"}]))
        )
        await stopping.wait()
        try:
            participant.retired.check(EpisodeId(rollout_id="r"))
        except StaleAttemptError:
            refused_while_stopping.append(True)
        replied_early = retire.done()
        finish.set()
        retired = await retire

    assert refused_while_stopping == [True] and not replied_early
    assert retired.status_code == 200 and participant.executions == {}
    assert len(participant.retired) == 1


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
    async with make_client(participant, lease_grace=2.0) as client:
        prepared = (await client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=0.1))).json()
        closed_after_prepare = participant.accepting is False
        await _until(lambda: participant.accepting, "the lease expired")
        status = (await client.get("/ng-control/v1/checkpoint/status")).json()
        stale = await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir="/unused"))

    assert prepared["phase"] == "prepared" and closed_after_prepare
    assert status["phase"] == "idle" and participant.accepting
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "stale_checkpoint"


async def test_renew_keeps_a_slow_but_live_controller_in_charge() -> None:
    participant = FakeParticipant()
    async with make_client(participant, lease_grace=2.0) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=0.05))
        first_lease = (await client.get("/ng-control/v1/checkpoint/status")).json()["lease_expires_at"]
        # Each renew comes well inside the lease, leaving most of it as slack for a loaded machine.
        # At least one renew comes after the first lease ran out, when nothing else could have kept it.
        while time.time() < first_lease + 1.0:
            await asyncio.sleep(0.5)
            renewed = await client.post("/ng-control/v1/checkpoint/renew", json=body(timeout=0.05))
            assert renewed.status_code == 200
        status = (await client.get("/ng-control/v1/checkpoint/status")).json()

    # The controller stayed in charge past the lease its prepare set, so only the renewals kept it there.
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
    assert len(restored.retired) == 1


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


async def test_an_episode_woken_by_resume_stays_parked_if_a_new_checkpoint_closes_first() -> None:
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
    parked_through_second_checkpoint = not progressed.is_set() and steps.blocker_count() == 0
    steps.open()
    await asyncio.wait_for(task, 10)

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
            server_utils.request("GET", f"http://127.0.0.1:{port}/fast", _control=True), 10
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


async def test_a_commit_retires_restored_episodes_its_scope_leaves_out(tmp_path: Path) -> None:
    participant = FakeParticipant()
    participant.executions["kept-a1"] = {"parked": True, "value": 1, "restored": True}
    participant.executions["dropped-a1"] = {"parked": True, "value": 2, "restored": True}
    participant.executions["live"] = {"parked": True, "value": 3}
    scope = [{"rollout_id": "kept", "attempt": 1}, {"rollout_id": "live"}]
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        await client.post(
            "/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path), episode_ids=scope)
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=body())

    # The controller no longer continues "dropped", so its restored state is retired.
    assert set(participant.executions) == {"kept-a1", "live"}


async def test_a_commit_that_fails_while_retiring_out_of_scope_state_can_be_retried(tmp_path: Path) -> None:
    participant = FakeParticipant()
    participant.executions["kept-a1"] = {"parked": True, "value": 1, "restored": True}
    participant.executions["x-a1"] = {"parked": True, "value": 2, "restored": True}
    participant.executions["y-a1"] = {"parked": True, "value": 3, "restored": True}
    original_retire = participant.retire
    fail_on = {"y-a1"}

    async def flaky_retire(episode_id: EpisodeId) -> None:
        if episode_id.capture_key in fail_on:
            raise RuntimeError("could not free")
        await original_retire(episode_id)

    participant.retire = flaky_retire
    commit = body(checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "kept", "attempt": 1}])
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        with pytest.raises(RuntimeError):
            await client.post("/ng-control/v1/checkpoint/commit", json=commit)
        fail_on.clear()
        retried = await client.post("/ng-control/v1/checkpoint/commit", json=commit)

    # The retry exported the same in-scope records, so the existing manifest matched.
    assert retried.status_code == 200 and retried.json()["episode_ids"] == ["kept-a1"]
    assert set(participant.executions) == {"kept-a1"}


async def test_restore_validates_only_the_records_in_scope(tmp_path: Path) -> None:
    source = FakeParticipant()
    source.executions["keep"] = {"parked": True, "value": 1}
    source.executions["other"] = {"parked": True, "value": -1}
    async with make_client(source) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))
    validated: list[str] = []
    original = Record.model_validate

    def counting(value, *args, **kwargs):
        validated.append(EpisodeId.model_validate(value["episode_id"]).capture_key)
        return original(value, *args, **kwargs)

    restored = FakeParticipant()
    Record.model_validate = counting
    try:
        async with make_client(restored) as client:
            result = await client.post(
                "/ng-control/v1/checkpoint/restore",
                json=body("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "keep"}]),
            )
    finally:
        Record.model_validate = original

    assert result.status_code == 200 and validated == ["keep"]


async def _no_notify() -> None:
    return None


async def test_retire_waits_for_the_episode_and_a_cut_short_retire_leaves_it_tracked() -> None:
    steps = EpisodeSteps(_no_notify)
    release = asyncio.Event()
    started = asyncio.Event()

    async def episode() -> None:
        steps.begin("r")
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Cleanup that outlives the cancellation, like closing sessions.
            await release.wait()
            raise
        finally:
            await steps.end("r")

    task = asyncio.create_task(episode())
    await started.wait()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(steps.retire("r"), 0.05)
    # The task is still running: it stays tracked, a duplicate start is refused, and it still blocks a checkpoint.
    with pytest.raises(ValueError, match="already running"):
        steps.begin("r")
    assert steps.keys() == ["r"] and steps.blocker_count() == 1

    retry = asyncio.create_task(steps.retire("r"))
    await asyncio.sleep(0.01)
    assert not retry.done()
    release.set()
    await asyncio.wait_for(retry, 10)

    assert task.done() and steps.keys() == [] and steps.blocker_count() == 0


async def test_a_replay_step_that_ends_during_a_checkpoint_does_not_block_it_again() -> None:
    steps = EpisodeSteps(_no_notify)
    finish_step = asyncio.Event()
    stepped = asyncio.Event()
    record_boundary = asyncio.Event()

    async def episode() -> None:
        steps.begin("r")
        await steps.boundary("r", {"next": "verify"})
        async with steps.step("r", "replay"):
            await finish_step.wait()
        stepped.set()
        await record_boundary.wait()
        await steps.boundary("r", {"next": "close"})
        await steps.end("r")

    task = asyncio.create_task(episode())
    await asyncio.sleep(0)
    steps.close()
    assert steps.blocker_count() == 0
    finish_step.set()
    await stepped.wait()
    # Between the end of the step and its next boundary, the episode still counts at the boundary before the step.
    assert steps.blocker_count() == 0 and steps.exported() == {"r": {"next": "verify"}}
    record_boundary.set()
    await asyncio.sleep(0)
    assert steps.blocker_count() == 0 and steps.exported() == {"r": {"next": "close"}}
    steps.open()
    await asyncio.wait_for(task, 10)


async def test_every_live_episode_is_exported_with_or_without_a_boundary() -> None:
    steps = EpisodeSteps(_no_notify)
    in_first_step = asyncio.Event()
    hold = asyncio.Event()

    async def fresh() -> None:
        steps.begin("fresh")
        async with steps.step("fresh", "replay"):
            in_first_step.set()
            await hold.wait()

    async def restored() -> None:
        steps.begin("restored-a1", continuation={"next": "verify"})
        async with steps.step("restored-a1", "replay"):
            await hold.wait()

    tasks = [asyncio.create_task(fresh()), asyncio.create_task(restored())]
    await in_first_step.wait()
    await asyncio.sleep(0)
    steps.close()

    # The fresh episode starts over from its input; the restored one continues from its restored boundary.
    assert steps.exported() == {"fresh": None, "restored-a1": {"next": "verify"}}
    assert steps.blocker_count() == 0
    hold.set()
    steps.open()
    await asyncio.gather(*tasks)


async def test_blockers_are_counted_in_full_and_listed_up_to_a_limit() -> None:
    steps = EpisodeSteps(_no_notify)
    for key in ("c", "a", "b"):
        steps.begin(key)

    assert steps.blocker_count() == 3
    assert steps.blockers(2) == ["a", "b"]


def test_records_stream_to_disk_and_a_retried_commit_leaves_no_temporary_file(tmp_path: Path) -> None:
    from nemo_gym._checkpoint.errors import CheckpointStateError
    from nemo_gym._checkpoint.store import participant_dir, read_participant_state, write_participant_state

    def records(value: int):
        for attempt in range(3):
            yield {"episode_id": {"rollout_id": "r", "attempt": attempt}, "value": value}

    first = write_participant_state(tmp_path, kind="fake", instance="f", checkpoint_id="c1", records=records(1))
    again = write_participant_state(tmp_path, kind="fake", instance="f", checkpoint_id="c1", records=records(1))
    with pytest.raises(CheckpointStateError, match="different commit"):
        write_participant_state(tmp_path, kind="fake", instance="f", checkpoint_id="c1", records=records(2))

    manifest, restored = read_participant_state(tmp_path, kind="fake", instance="f")
    directory = participant_dir(tmp_path, kind="fake", instance="f")
    assert first == again == manifest and manifest["record_count"] == 3
    assert [record["episode_id"]["attempt"] for record in restored] == [0, 1, 2]
    assert sorted(path.name for path in directory.iterdir()) == ["manifest.json", manifest["records_file"]]
    assert manifest["records_file"] == f"records-{manifest['records_sha256']}.jsonl"


def test_a_corrupt_records_file_is_a_checkpoint_state_error(tmp_path: Path) -> None:
    from nemo_gym._checkpoint.errors import CheckpointStateError
    from nemo_gym._checkpoint.store import participant_dir, read_participant_state, write_participant_state

    manifest = write_participant_state(
        tmp_path, kind="fake", instance="f", checkpoint_id="c1", records=[{"episode_id": {"rollout_id": "r"}}]
    )
    (participant_dir(tmp_path, kind="fake", instance="f") / manifest["records_file"]).write_bytes(b"{not json\n")

    with pytest.raises(CheckpointStateError):
        read_participant_state(tmp_path, kind="fake", instance="f")


def _gated_write(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, list[None], list[None]]:
    """Make every write wait for the returned gate; count the writes started and finished."""
    import nemo_gym._checkpoint.control as control

    real_write = control.write_participant_state
    gate, started, finished = threading.Event(), [], []

    def gated(*args, **kwargs):
        started.append(None)
        gate.wait(5)
        try:
            return real_write(*args, **kwargs)
        finally:
            finished.append(None)

    monkeypatch.setattr(control, "write_participant_state", gated)
    return gate, started, finished


async def _until(condition: Callable[[], object], what: str) -> None:
    # Polls instead of sleeping a fixed time, so a loaded machine does not turn the wait into a false result.
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting until {what}")


async def _past_deadline(
    client: httpx.AsyncClient, operation: str, until: Callable[[], object], what: str, **fields: object
) -> httpx.Response:
    """Send ``operation`` with a deadline that passes while the test holds its work up, once that work has started.

    On a loaded machine the deadline can pass before the held work starts, for example during the export.
    The same call again, as a controller retries, starts that work or awaits the run already under way.
    """
    for _ in range(100):
        late = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body(timeout=0.1, **fields))
        assert late.json()["error"]["code"] == "deadline_exceeded"
        if until():
            return late
    raise AssertionError(f"timed out waiting until {what}")


async def test_a_commit_retried_while_its_write_runs_awaits_that_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate, started, _ = _gated_write(monkeypatch)
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        late = await _past_deadline(
            client, "commit", lambda: started, "the write started", checkpoint_dir=str(tmp_path)
        )
        # The exported state changes after the failed call; a second export would write different records.
        participant.executions["r"]["value"] = 2
        retried = asyncio.create_task(
            client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))
        )
        await asyncio.sleep(0.05)
        gate.set()
        retried = await retried

    assert late.json()["error"]["code"] == "deadline_exceeded"
    assert retried.status_code == 200 and len(started) == 1
    _, records = read_participant_state(tmp_path, kind="fake", instance="fake-1")
    assert [record["value"] for record in records] == [1]


async def test_resume_stops_a_write_that_outlived_its_commit_before_it_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate, started, finished = _gated_write(monkeypatch)
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        # The write must have outlived its commit, or this checks nothing.
        await _past_deadline(client, "commit", lambda: started, "the write started", checkpoint_dir=str(tmp_path))
        await client.post("/ng-control/v1/checkpoint/resume", json=body())
        gate.set()
        await _until(lambda: finished, "the write finished")

    directory = participant_dir(tmp_path, kind="fake", instance="fake-1")
    assert not directory.exists() or list(directory.iterdir()) == []


def test_a_losing_writer_neither_replaces_the_published_records_nor_leaves_its_own(tmp_path: Path) -> None:
    def records(value: int) -> list[dict]:
        return [{"episode_id": {"rollout_id": "r"}, "value": value}]

    published = write_participant_state(tmp_path, kind="fake", instance="f", checkpoint_id="c1", records=records(1))
    with pytest.raises(CheckpointStateError, match="different commit"):
        write_participant_state(tmp_path, kind="fake", instance="f", checkpoint_id="c1", records=records(2))

    manifest, restored = read_participant_state(tmp_path, kind="fake", instance="f")
    directory = participant_dir(tmp_path, kind="fake", instance="f")
    assert manifest == published and restored == records(1)
    assert sorted(path.name for path in directory.iterdir()) == ["manifest.json", published["records_file"]]


class PayloadRecord(CheckpointRecord):
    state: JsonPayload


def test_payloads_are_checked_by_what_the_writer_can_write(tmp_path: Path) -> None:
    big = PayloadRecord(episode_id=EpisodeId(rollout_id="big"), state={"hash": 2**70})
    write_participant_state(tmp_path, kind="fake", instance="f", checkpoint_id="c1", records=[big.to_json_record()])
    assert read_participant_state(tmp_path, kind="fake", instance="f")[1][0]["state"] == {"hash": 2**70}

    with pytest.raises(ValidationError, match="not JSON"):
        PayloadRecord(episode_id=EpisodeId(rollout_id="r"), state={"handle": object()})

    # The fast check accepts a datetime that the writer cannot write; the writer names the episode.
    dated = PayloadRecord(episode_id=EpisodeId(rollout_id="dated"), state={"at": datetime.now(timezone.utc)})
    with pytest.raises(CheckpointStateError, match="episode .*dated.* is not JSON"):
        write_participant_state(
            tmp_path / "other", kind="fake", instance="f", checkpoint_id="c1", records=[dated.to_json_record()]
        )


async def test_a_restore_whose_close_fails_part_way_reopens_admission(tmp_path: Path) -> None:
    write_participant_state(
        tmp_path,
        kind="fake",
        instance="fake-1",
        checkpoint_id="c1",
        records=[{"episode_id": {"rollout_id": "r"}, "value": 1}],
    )

    class HalfClosing(FakeParticipant):
        async def close_admission(self, request: CheckpointRequest) -> None:
            # One worker closed, another failed.
            self.accepting = False
            raise RuntimeError("close failed")

    participant = HalfClosing()
    async with make_client(participant) as client:
        with pytest.raises(RuntimeError, match="close failed"):
            await client.post(
                "/ng-control/v1/checkpoint/restore",
                json=body("restore-1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            )
        status = (await client.get("/ng-control/v1/checkpoint/status")).json()

    assert status["phase"] == "idle" and participant.accepting is True


async def test_a_restored_record_that_fails_validation_is_a_typed_error_naming_its_episode(tmp_path: Path) -> None:
    # Written before a schema change: the record lacks a field the participant now requires.
    write_participant_state(
        tmp_path, kind="fake", instance="fake-1", checkpoint_id="c1", records=[{"episode_id": {"rollout_id": "r"}}]
    )
    participant = FakeParticipant()
    async with make_client(participant) as client:
        response = await client.post(
            "/ng-control/v1/checkpoint/restore",
            json=body("restore-1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_checkpoint_state"
    assert "episode 'r' is invalid" in response.json()["error"]["detail"]
    assert participant.accepting is True and participant.executions == {}


async def test_an_episode_whose_retire_was_cut_short_blocks_the_next_checkpoint_and_is_not_exported() -> None:
    steps = EpisodeSteps(_no_notify)
    release = asyncio.Event()

    async def episode() -> None:
        steps.begin("r")
        try:
            await steps.boundary("r", {"step": 1})
        except asyncio.CancelledError:
            # Slow final cleanup, like closing sessions.
            await release.wait()
            raise
        finally:
            await steps.end("r")

    steps.close()
    task = asyncio.create_task(episode())
    await asyncio.sleep(0.01)
    steps.open()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(steps.retire("r"), 0.05)

    # A new checkpoint before the retire is retried: the episode is still parked, but it is being stopped.
    steps.close()
    assert steps.blocker_count() == 1 and steps.blockers(10) == ["r"]
    assert steps.exported() == {}

    release.set()
    await asyncio.wait([task], timeout=10)
    assert task.cancelled() and steps.blocker_count() == 0 and steps.keys() == []


async def test_a_commit_retried_after_its_write_failed_stores_the_same_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nemo_gym._checkpoint.control as control

    real_write = control.write_participant_state
    failures = [OSError("disk full")]

    def failing_once(*args, **kwargs):
        if failures:
            raise CheckpointStateError(str(failures.pop()))
        return real_write(*args, **kwargs)

    monkeypatch.setattr(control, "write_participant_state", failing_once)
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        failed = await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))
        # The exported state moves after the failure; the retry must still store the first export.
        participant.executions["r"]["value"] = 2
        elsewhere = await client.post(
            "/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path / "other"))
        )
        retried = await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))

    assert failed.status_code == 422
    assert elsewhere.json()["error"]["code"] == "checkpoint_conflict"
    assert retried.status_code == 200
    _, records = read_participant_state(tmp_path, kind="fake", instance="fake-1")
    assert [record["value"] for record in records] == [1]


async def test_resume_waits_for_a_publication_under_way_so_none_appears_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nemo_gym._checkpoint.store as store

    real_create = store._create
    publishing, release = threading.Event(), threading.Event()

    def slow_create(*args, **kwargs):
        publishing.set()
        release.wait(5)
        return real_create(*args, **kwargs)

    monkeypatch.setattr(store, "_create", slow_create)
    manifest = participant_dir(tmp_path, kind="fake", instance="fake-1") / "manifest.json"
    real_stop = store.WriteStop.stop
    manifest_when_stopped: list[bool] = []

    def recording_stop(self) -> None:
        real_stop(self)
        manifest_when_stopped.append(manifest.exists())

    monkeypatch.setattr(store.WriteStop, "stop", recording_stop)
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        await _past_deadline(
            client, "commit", publishing.is_set, "the write reached its publication", checkpoint_dir=str(tmp_path)
        )
        resume = asyncio.create_task(client.post("/ng-control/v1/checkpoint/resume", json=body()))
        await asyncio.sleep(0.1)
        # The writer passed its last stop check: resume waits for the manifest instead of racing it.
        assert not resume.done() and not manifest.exists()
        release.set()
        await resume

    assert manifest.exists() and participant.accepting is True
    # The stop returned only once the publication under way had finished.
    assert manifest_when_stopped == [True]


def test_a_read_keeps_only_the_selected_records_but_verifies_the_whole_file(tmp_path: Path) -> None:
    rows = [{"episode_id": {"rollout_id": f"r{index}"}, "value": index} for index in range(5)]
    write_participant_state(tmp_path, kind="fake", instance="f", checkpoint_id="c", records=rows)

    manifest, kept = read_participant_state(
        tmp_path, kind="fake", instance="f", select=lambda row: row["value"] if row["value"] % 2 else None
    )

    assert kept == [1, 3] and manifest["record_count"] == 5


async def test_prepare_asks_whether_ready_and_lists_blockers_only_for_its_reply() -> None:
    class Counting(FakeParticipant):
        listings = 0

        def readiness(self) -> PrepareReport:
            Counting.listings += 1
            return super().readiness()

        def ready(self) -> bool:
            return all(execution["parked"] for execution in self.executions.values())

    participant = Counting()
    for index in range(20):
        participant.executions[f"r{index}"] = {"parked": False, "value": index}
    async with make_client(participant) as client:
        prepare = asyncio.create_task(client.post("/ng-control/v1/checkpoint/prepare", json=body()))
        # Admission closes right before the readiness wait, so each park below wakes a waiting prepare.
        await _until(lambda: participant.accepting is False, "the prepare closed admission")
        for index in range(20):
            await participant.park(f"r{index}")
        reply = (await prepare).json()

    assert reply["phase"] == "prepared"
    assert Counting.listings <= 2


async def test_a_retire_after_a_restore_that_ran_out_of_time_waits_for_its_install(tmp_path: Path) -> None:
    write_participant_state(
        tmp_path,
        kind="fake",
        instance="fake-1",
        checkpoint_id="c1",
        records=[{"episode_id": {"rollout_id": "r"}, "value": 1}],
    )
    release = threading.Event()
    installing: list[None] = []
    installed = asyncio.Event()

    class SlowInstall(FakeParticipant):
        async def install(self, records, scope) -> None:
            installing.append(None)
            # Like the model ledger import: a thread the restore's deadline cannot stop.
            await asyncio.to_thread(release.wait, 5)
            self.restore_records(records)
            installed.set()

    participant = SlowInstall()
    async with make_client(participant) as client:
        restore = await _past_deadline(
            client,
            "restore",
            lambda: installing,
            "the install started",
            checkpoint_id="r1",
            checkpoint_dir=str(tmp_path),
            episode_ids=[{"rollout_id": "r"}],
        )
        retire = asyncio.create_task(
            client.post(
                "/ng-control/v1/checkpoint/retire", json=body("r1", episode_ids=[{"rollout_id": "r", "attempt": 1}])
            )
        )
        await asyncio.sleep(0.1)
        assert not retire.done()
        release.set()
        retired = await retire
        # A retire that did not wait would leave behind what the install brings.
        await asyncio.wait_for(installed.wait(), 10)

    assert restore.json()["error"]["code"] == "deadline_exceeded"
    assert retired.status_code == 200 and participant.executions == {}


async def test_admission_reopens_only_once_an_install_that_outlived_its_restore_has_finished(tmp_path: Path) -> None:
    write_participant_state(
        tmp_path,
        kind="fake",
        instance="fake-1",
        checkpoint_id="c1",
        records=[{"episode_id": {"rollout_id": "r"}, "value": 1}],
    )
    release = threading.Event()
    installing: list[None] = []

    class SlowInstall(FakeParticipant):
        async def install(self, records, scope) -> None:
            installing.append(None)
            await asyncio.to_thread(release.wait, 5)
            self.restore_records(records)

    participant = SlowInstall()
    async with make_client(participant) as client:
        restore = await _past_deadline(
            client,
            "restore",
            lambda: installing,
            "the install started",
            checkpoint_id="r1",
            checkpoint_dir=str(tmp_path),
            episode_ids=[{"rollout_id": "r"}],
        )
        # The install is still changing state: nothing may be admitted against it yet.
        closed_while_installing = not participant.accepting
        release.set()
        await _until(lambda: participant.accepting, "admission reopened")

    assert restore.json()["error"]["code"] == "deadline_exceeded"
    assert closed_while_installing and participant.accepting


async def test_install_is_given_every_episode_the_restore_continues(tmp_path: Path) -> None:
    write_participant_state(
        tmp_path,
        kind="fake",
        instance="fake-1",
        checkpoint_id="c1",
        records=[{"episode_id": {"rollout_id": "r"}, "value": 1}],
    )
    seen: list = []

    class Recording(FakeParticipant):
        async def install(self, records, scope) -> None:
            seen.append(sorted(episode_id.capture_key for episode_id in scope))
            self.restore_records(records)

    async with make_client(Recording()) as client:
        await client.post(
            "/ng-control/v1/checkpoint/restore",
            json=body("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}, {"rollout_id": "e"}]),
        )

    # "e" has no record here, but a participant that keeps files per episode must still clear its own.
    assert seen == [["e", "r"]]


async def test_a_commit_deletes_unscoped_restored_state_without_retiring_it(tmp_path: Path) -> None:
    deleted: list[str] = []

    class Deleting(FakeParticipant):
        async def delete_restored(self, episode_id: EpisodeId) -> None:
            deleted.append(episode_id.capture_key)
            await super().delete_restored(episode_id)

        async def retire(self, episode_id: EpisodeId) -> None:
            deleted.append(f"retired {episode_id.capture_key}")
            await super().retire(episode_id)

    participant = Deleting()
    participant.executions["r-a1"] = {"parked": True, "value": 1, "restored": True}
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path), episode_ids=[]))

    # The default delete_restored retires; a participant whose retire fences overrides it.
    assert deleted == ["r-a1", "retired r-a1"]


async def test_a_restart_episode_never_blocks_parks_or_is_exported_and_is_reported() -> None:
    steps = EpisodeSteps(_no_notify)
    finish_step = asyncio.Event()
    reached_boundary = asyncio.Event()

    async def episode() -> None:
        steps.begin("r")
        await steps.mark_restart("r")
        async with steps.step("r", "wait"):
            await finish_step.wait()
        # A restart does not park: nothing of it is in the checkpoint, so it just keeps running.
        await steps.boundary("r", {"next": "verify"})
        reached_boundary.set()

    task = asyncio.create_task(episode())
    await asyncio.sleep(0)
    steps.close()

    # Inside a wait step, which would hold up any other episode's checkpoint.
    assert steps.blocker_count() == 0 and steps.exported() == {} and steps.restarts() == ["r"]
    finish_step.set()
    await asyncio.wait_for(reached_boundary.wait(), 10)
    assert steps.exported() == {} and steps.restarts() == ["r"]
    await asyncio.wait_for(task, 10)


async def test_a_restart_episode_being_retired_still_blocks_until_it_has_stopped() -> None:
    steps = EpisodeSteps(_no_notify)
    release = asyncio.Event()

    async def episode() -> None:
        steps.begin("r")
        await steps.mark_restart("r")
        try:
            await asyncio.Event().wait()
        finally:
            await release.wait()

    task = asyncio.create_task(episode())
    await asyncio.sleep(0)
    retire = asyncio.create_task(steps.retire("r"))
    await asyncio.sleep(0)

    assert steps.blocker_count() == 1 and steps.restarts() == []
    release.set()
    await asyncio.wait_for(retire, 10)
    assert steps.blocker_count() == 0
    await asyncio.gather(task, return_exceptions=True)


class RestartingParticipant(FakeParticipant):
    """Executions marked ``restart`` cannot be captured: they never block and are reported as restarts."""

    def readiness(self) -> PrepareReport:
        running = [key for key, e in self.executions.items() if not e["parked"] and not e.get("restart")]
        restarts = sorted(key for key, e in self.executions.items() if e.get("restart"))
        return PrepareReport(ready=not running, blockers=running, restarts=restarts)

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[BaseModel]:
        return [
            record
            for record in super().export_records(episode_ids)
            if not self.executions[record.episode_id.capture_key].get("restart")
        ]


async def test_a_commit_refuses_a_scope_naming_a_restart_and_commits_one_without_it(tmp_path: Path) -> None:
    participant = RestartingParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}
    # Still running, and never captured: it does not hold up the checkpoint.
    participant.executions["s"] = {"parked": False, "value": 2, "restart": True}
    async with make_client(participant) as client:
        prepare = await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        assert prepare.json()["phase"] == "prepared"
        assert prepare.json()["report"]["restarts"] == ["s"]

        refused = await client.post(
            "/ng-control/v1/checkpoint/commit",
            json=body(checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}, {"rollout_id": "s"}]),
        )
        committed = await client.post(
            "/ng-control/v1/checkpoint/commit",
            json=body(checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
        )

    assert refused.status_code == 409 and refused.json()["error"]["code"] == "restart_in_scope"
    assert committed.status_code == 200 and committed.json()["episode_ids"] == ["r"]


def test_a_seed_reply_says_whether_its_session_restarts() -> None:
    from nemo_gym._checkpoint.steps import CHECKPOINT_RESTART_HEADER, seed_restarts

    assert seed_restarts({CHECKPOINT_RESTART_HEADER: "1"})
    assert not seed_restarts({}) and not seed_restarts(None)


async def test_an_episode_can_begin_as_a_restart() -> None:
    steps = EpisodeSteps(_no_notify)
    steps.begin("r", restart=True)
    steps.close()

    assert steps.blocker_count() == 0 and steps.restarts() == ["r"]
    await steps.end("r")
    assert steps.restarts() == []


async def test_an_episode_becomes_a_restart_only_once_no_checkpoint_is_open() -> None:
    steps = EpisodeSteps(_no_notify)
    seeded = asyncio.Event()

    async def episode() -> None:
        steps.begin("r")
        await steps.boundary("r", {"next": "seed"})
        async with steps.step("r", "replay"):
            await seeded.wait()
            # The seed reply says the session cannot be captured, but it arrived during a checkpoint.
            await steps.mark_restart("r")

    task = asyncio.create_task(episode())
    await asyncio.sleep(0)
    steps.close()
    seeded.set()
    await asyncio.sleep(0.01)
    # The checkpoint still exports the boundary before the seed, which a crash would start over from anyway.
    during = (steps.restarts(), dict(steps.exported()))
    steps.open()
    await asyncio.wait_for(task, 10)

    assert during == ([], {"r": {"next": "seed"}})


async def test_an_export_that_outlives_its_commit_is_awaited_by_the_retry_not_repeated(tmp_path: Path) -> None:
    exports: list[str] = []
    release = asyncio.Event()

    class SlowExport(FakeParticipant):
        async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[BaseModel]:
            exports.append("started")
            await release.wait()
            return self.export_records(episode_ids)

    participant = SlowExport()
    participant.executions["r"] = {"parked": True, "value": 1}
    commit = body(checkpoint_dir=str(tmp_path), timeout=0.2)
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        late = await client.post("/ng-control/v1/checkpoint/commit", json=commit)
        # The export finishes after its commit gave up; the state it read is the checkpoint's cut.
        release.set()
        await asyncio.sleep(0.05)
        participant.executions["r"]["value"] = 2
        retried = await client.post("/ng-control/v1/checkpoint/commit", json=commit | {"deadline_ts": time.time() + 5})

    assert late.json()["error"]["code"] == "deadline_exceeded"
    assert retried.status_code == 200 and exports == ["started"]
    _, records = read_participant_state(tmp_path, kind="fake", instance="fake-1")
    assert [record["value"] for record in records] == [1]


async def test_resume_waits_for_a_cleanup_that_outlived_its_commit_before_reopening(tmp_path: Path) -> None:
    order: list[str] = []
    release = asyncio.Event()
    deleting: list[None] = []

    class SlowCleanup(FakeParticipant):
        async def delete_restored(self, episode_id: EpisodeId) -> None:
            deleting.append(None)
            await release.wait()
            order.append(f"deleted {episode_id.capture_key}")
            await super().delete_restored(episode_id)

        async def open_admission(self) -> None:
            order.append("reopened")
            await super().open_admission()

    participant = SlowCleanup()
    participant.executions["r-a1"] = {"parked": True, "value": 1, "restored": True}
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        # The resume below must find the deletion under way, or it has nothing to wait for.
        late = await _past_deadline(
            client, "commit", lambda: deleting, "the cleanup started", checkpoint_dir=str(tmp_path), episode_ids=[]
        )
        resume = asyncio.create_task(client.post("/ng-control/v1/checkpoint/resume", json=body()))
        await asyncio.sleep(0.1)
        waited = not resume.done()
        release.set()
        resumed = await resume

    # A replacement may start as the attempt being deleted, so admission reopens only once the deletion finished.
    assert late.json()["error"]["code"] == "deadline_exceeded"
    assert waited and resumed.status_code == 200
    assert order == ["deleted r-a1", "reopened"]


async def test_a_completed_commit_or_restore_retried_with_other_arguments_conflicts(tmp_path: Path) -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}
    commit = body(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        first = await client.post("/ng-control/v1/checkpoint/commit", json=commit)
        same = await client.post("/ng-control/v1/checkpoint/commit", json=commit)
        other_dir = await client.post(
            "/ng-control/v1/checkpoint/commit", json=commit | {"checkpoint_dir": str(tmp_path)}
        )
        other_scope = await client.post("/ng-control/v1/checkpoint/commit", json=commit | {"episode_ids": []})
        await client.post("/ng-control/v1/checkpoint/resume", json=body())

    restored = FakeParticipant()
    restore = body("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])
    async with make_client(restored) as client:
        await client.post("/ng-control/v1/checkpoint/restore", json=restore)
        same_restore = await client.post("/ng-control/v1/checkpoint/restore", json=restore)
        other_restore = await client.post("/ng-control/v1/checkpoint/restore", json=restore | {"episode_ids": []})

    assert first.status_code == same.status_code == same_restore.status_code == 200
    assert same.json() == first.json()
    for refused in (other_dir, other_scope, other_restore):
        assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_conflict"


def test_the_control_connection_pool_refuses_negative_limits() -> None:
    from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig

    for field in ("global_aiohttp_control_connector_limit", "global_aiohttp_control_connector_limit_per_host"):
        with pytest.raises(ValidationError):
            GlobalAIOHTTPAsyncClientConfig.model_validate({field: -1})
        assert getattr(GlobalAIOHTTPAsyncClientConfig.model_validate({field: 0}), field) == 0


async def test_resume_does_not_wait_for_an_export_nothing_will_use(tmp_path: Path) -> None:
    class HungExport(FakeParticipant):
        async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[BaseModel]:
            await asyncio.Event().wait()
            return []

    participant = HungExport()
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path), timeout=0.1))
        resumed = await asyncio.wait_for(client.post("/ng-control/v1/checkpoint/resume", json=body()), timeout=10)

    assert resumed.status_code == 200 and participant.accepting


async def test_resume_past_its_deadline_on_a_hung_cleanup_changes_nothing_and_can_be_retried(tmp_path: Path) -> None:
    release = asyncio.Event()
    deleting: list[None] = []

    class HungCleanup(FakeParticipant):
        async def delete_restored(self, episode_id: EpisodeId) -> None:
            deleting.append(None)
            await release.wait()
            await super().delete_restored(episode_id)

    participant = HungCleanup()
    participant.executions["r-a1"] = {"parked": True, "value": 1, "restored": True}
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        # The resume below must find the deletion under way, or it has nothing to wait for.
        await _past_deadline(
            client, "commit", lambda: deleting, "the cleanup started", checkpoint_dir=str(tmp_path), episode_ids=[]
        )
        late = await client.post("/ng-control/v1/checkpoint/resume", json=body(timeout=0.2))
        status = (await client.get("/ng-control/v1/checkpoint/status")).json()
        release.set()
        resumed = await client.post("/ng-control/v1/checkpoint/resume", json=body())

    # Admission stays closed while the deletion runs: a replacement may start as the attempt it deletes.
    assert late.json()["error"]["code"] == "deadline_exceeded"
    assert status["phase"] == "prepared"
    assert resumed.status_code == 200 and participant.accepting
    assert "r-a1" not in participant.executions


async def test_a_retry_naming_the_same_scope_in_another_order_awaits_the_same_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate, _, _ = _gated_write(monkeypatch)
    participant = FakeParticipant()
    participant.executions["a"] = {"parked": True, "value": 1}
    participant.executions["b"] = {"parked": True, "value": 2}
    scope = [{"rollout_id": "a"}, {"rollout_id": "b"}]
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        first = await client.post(
            "/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path), episode_ids=scope, timeout=0.2)
        )
        gate.set()
        retried = await client.post(
            "/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path), episode_ids=scope[::-1])
        )

    assert first.json()["error"]["code"] == "deadline_exceeded"
    assert retried.status_code == 200 and retried.json()["episode_ids"] == ["a", "b"]
