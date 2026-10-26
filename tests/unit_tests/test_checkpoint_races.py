# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control calls that race a participant's own background work.

A race here does not raise an error: it leaves the participant in the wrong state.
So each test holds the background work at a gate, lets the racing call through, then checks the state:
the phase, whether admission is open, the lease, what is exported or installed, and what a later checkpoint sees.
Only lease expiries and the short deadlines that make a call time out wait on the clock.
The work they race is held at a gate until the test releases it.
"""

import asyncio
import threading
from pathlib import Path

import httpx
import pytest

from nemo_gym._checkpoint.control import StaleAttemptError
from nemo_gym._checkpoint.store import read_participant_state, write_participant_state
from nemo_gym.episode_types import EpisodeId
from tests.unit_tests.test_checkpoint_control import (
    FakeParticipant,
    _gated_write,
    _past_deadline,
    _until,
    body,
    make_client,
)


ROUTE = "/ng-control/v1/checkpoint"


class HeldWait(FakeParticipant):
    """A participant whose prepare waits at a gate once it has found the participant not ready.

    It stands for a prepare whose wait for readiness outlives anything else the test lets happen meanwhile.
    """

    def __init__(self) -> None:
        super().__init__()
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()

    async def wait_changed(self, timeout: float) -> None:
        self.waiting.set()
        await self.release.wait()


async def status(client: httpx.AsyncClient) -> dict:
    return (await client.get(f"{ROUTE}/status")).json()


async def committed_records(client: httpx.AsyncClient, checkpoint_id: str, directory: Path) -> list[dict]:
    """Prepare and commit ``checkpoint_id``, resume it, and return the records it stored."""
    prepared = (await client.post(f"{ROUTE}/prepare", json=body(checkpoint_id))).json()
    assert prepared["phase"] == "prepared", prepared
    committed = await client.post(f"{ROUTE}/commit", json=body(checkpoint_id, checkpoint_dir=str(directory)))
    assert committed.status_code == 200, committed.text
    await client.post(f"{ROUTE}/resume", json=body(checkpoint_id))
    _, records, _ = read_participant_state(directory, kind="fake", instance="fake-1")
    return records


async def test_a_resume_while_prepare_waits_for_readiness_aborts_the_checkpoint_and_reopens(tmp_path: Path) -> None:
    participant = HeldWait()
    participant.executions["r"] = {"parked": False, "value": 1}
    async with make_client(participant) as client:
        prepare = asyncio.create_task(client.post(f"{ROUTE}/prepare", json=body(timeout=30)))
        await asyncio.wait_for(participant.waiting.wait(), 5)
        resumed = await client.post(f"{ROUTE}/resume", json=body())
        reopened_before_prepare_returned = participant.accepting and not prepare.done()
        participant.release.set()
        interrupted = (await asyncio.wait_for(prepare, 5)).json()
        after = await status(client)
        stale_commit = await client.post(f"{ROUTE}/commit", json=body(checkpoint_dir=str(tmp_path / "1")))
        # The episode that blocked the aborted checkpoint finishes its step; a later checkpoint captures it.
        participant.executions["r"] = {"parked": True, "value": 2}
        later = await committed_records(client, "ckpt-2", tmp_path / "2")

    assert resumed.status_code == 200 and reopened_before_prepare_returned
    assert interrupted["phase"] == "idle"
    assert after["phase"] == "idle" and after["checkpoint_id"] is None and after["lease_expires_at"] is None
    assert stale_commit.json()["error"]["code"] == "checkpoint_stale"
    assert not (tmp_path / "1").exists()
    assert [record["value"] for record in later] == [2]


async def test_a_lease_expiry_while_prepare_waits_for_readiness_aborts_the_checkpoint_and_reopens(
    tmp_path: Path,
) -> None:
    participant = HeldWait()
    participant.executions["r"] = {"parked": False, "value": 1}
    async with make_client(participant, lease_grace=0.1) as client:
        # The prepare's own deadline passes while its wait is held, so only the lease watcher can end the checkpoint.
        prepare = asyncio.create_task(client.post(f"{ROUTE}/prepare", json=body(timeout=0.1)))
        await asyncio.wait_for(participant.waiting.wait(), 5)
        await _until(lambda: participant.accepting, "the lease expired and reopened admission")
        expired = await status(client)
        participant.release.set()
        interrupted = (await asyncio.wait_for(prepare, 5)).json()
        stale_prepare = await client.post(f"{ROUTE}/prepare", json=body())
        participant.executions["r"] = {"parked": True, "value": 2}
        later = await committed_records(client, "ckpt-2", tmp_path / "2")

    assert expired["phase"] == "idle" and expired["checkpoint_id"] is None and expired["lease_expires_at"] is None
    assert interrupted["phase"] == "idle"
    # A controller that only now asks again cannot reopen the expired checkpoint.
    assert stale_prepare.json()["error"]["code"] == "checkpoint_stale"
    assert [record["value"] for record in later] == [2]


async def test_a_lease_expiry_during_a_commits_cleanup_reopens_only_once_the_cleanup_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nemo_gym._checkpoint.control as control

    monkeypatch.setattr(control, "_LEASE_REOPEN_WAIT_SECONDS", 0.05)
    release = asyncio.Event()
    deleting: list[str] = []
    reopen_attempts: list[None] = []
    real_reopen = control.ParticipantControlPlane._reopen

    async def counted_reopen(self, **kwargs):
        reopen_attempts.append(None)
        return await real_reopen(self, **kwargs)

    monkeypatch.setattr(control.ParticipantControlPlane, "_reopen", counted_reopen)

    class HeldCleanup(FakeParticipant):
        async def delete_restored(self, episode_id: EpisodeId) -> None:
            deleting.append(episode_id.capture_key)
            await release.wait()
            await super().delete_restored(episode_id)

    participant = HeldCleanup()
    participant.executions["dropped-a1"] = {"parked": True, "value": 1, "restored": True}
    participant.executions["live"] = {"parked": True, "value": 2}
    scope = {"checkpoint_dir": str(tmp_path / "1"), "episode_ids": [{"rollout_id": "live"}]}
    async with make_client(participant, lease_grace=0.5) as client:
        await client.post(f"{ROUTE}/prepare", json=body(timeout=0.1))
        # The write finishes; the deletion of the restored state the scope leaves out is held.
        late = await _past_deadline(client, "commit", lambda: deleting, "the cleanup started", **scope)
        await _until(lambda: len(reopen_attempts) >= 2, "the lease expired and a reopen ran out of time")
        held = await status(client)
        closed_while_deleting = not participant.accepting
        release.set()
        await _until(lambda: participant.accepting, "the lease recovery reopened admission")
        after = await status(client)
        stale_commit = await client.post(f"{ROUTE}/commit", json=body(**scope))
        later = await committed_records(client, "ckpt-2", tmp_path / "2")

    assert late.json()["error"]["code"] == "checkpoint_deadline_exceeded"
    # A replacement may start as the attempt being deleted, so admission stays closed until the deletion finished.
    assert closed_while_deleting and held["phase"] == "prepared" and held["checkpoint_id"] == "ckpt-1"
    assert after["phase"] == "idle" and after["checkpoint_id"] is None and after["lease_expires_at"] is None
    assert stale_commit.json()["error"]["code"] == "checkpoint_stale"
    assert deleting == ["dropped-a1"] and set(participant.executions) == {"live"}
    # The deleted state is not retired: the controller may still start that rollout over as that attempt.
    participant.retired.check(EpisodeId(rollout_id="dropped", attempt=1))
    assert [record["episode_id"]["rollout_id"] for record in later] == ["live"]


async def test_a_commit_retried_while_its_timed_out_write_still_runs_stores_the_first_export_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nemo_gym._checkpoint.control as control

    gate, started, finished = _gated_write(monkeypatch)
    calls: list[None] = []
    real_start_write = control.ParticipantControlPlane._start_write

    async def counted_start_write(self, request):
        write = await real_start_write(self, request)
        # The call now waits for this write.
        calls.append(None)
        return write

    monkeypatch.setattr(control.ParticipantControlPlane, "_start_write", counted_start_write)
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}
    commit = {"checkpoint_dir": str(tmp_path / "1")}
    async with make_client(participant) as client:
        await client.post(f"{ROUTE}/prepare", json=body())
        first = await _past_deadline(client, "commit", lambda: started, "the write started", **commit)
        # The state changes after the export; a second export would store the new value.
        participant.executions["r"]["value"] = 2
        second = await client.post(f"{ROUTE}/commit", json=body(timeout=0.1, **commit))
        held = await status(client)
        waiting = len(calls)
        retry = asyncio.create_task(client.post(f"{ROUTE}/commit", json=body(**commit)))
        await _until(lambda: len(calls) > waiting, "the retry waits for the write")
        gate.set()
        retried = await asyncio.wait_for(retry, 10)
        replayed = await client.post(f"{ROUTE}/commit", json=body(**commit))
        committed = await status(client)
        closed_until_resume = not participant.accepting
        await client.post(f"{ROUTE}/resume", json=body())
        after = await status(client)
        writes = (len(started), len(finished))
        later = await committed_records(client, "ckpt-2", tmp_path / "2")

    assert first.json()["error"]["code"] == "checkpoint_deadline_exceeded"
    assert second.json()["error"]["code"] == "checkpoint_deadline_exceeded"
    assert held["phase"] == "prepared"
    assert retried.status_code == 200 and replayed.json() == retried.json()
    # One write ran for the three calls, and it stored the export the first call took.
    assert writes == (1, 1)
    _, records, _ = read_participant_state(tmp_path / "1", kind="fake", instance="fake-1")
    assert [record["value"] for record in records] == [1]
    assert committed["phase"] == "committed" and closed_until_resume
    assert after["phase"] == "idle" and after["lease_expires_at"] is None and participant.accepting
    assert [record["value"] for record in later] == [2]


async def test_a_restore_that_fails_while_its_install_runs_then_races_a_resume_leaves_the_attempts_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nemo_gym._checkpoint.control as control

    resume_waiting = asyncio.Event()
    real_retry = control.ParticipantControlPlane._retry_pending_reopen

    async def signalled_retry(self, request):
        resume_waiting.set()
        return await real_retry(self, request)

    monkeypatch.setattr(control.ParticipantControlPlane, "_retry_pending_reopen", signalled_retry)
    write_participant_state(
        tmp_path / "source",
        kind="fake",
        instance="fake-1",
        checkpoint_id="c1",
        records=[{"episode_id": {"rollout_id": "r"}, "value": 1}],
    )
    release = threading.Event()
    installing: list[None] = []

    class HeldInstall(FakeParticipant):
        def __init__(self) -> None:
            super().__init__()
            self.opened = 0

        async def install(self, records, scope) -> None:
            installing.append(None)
            # Like the model ledger import: a thread the restore's deadline cannot stop.
            await asyncio.to_thread(release.wait, 10)
            self.restore_records(records)

        async def open_admission(self) -> None:
            self.opened += 1
            await super().open_admission()

    participant = HeldInstall()
    restored_attempt = {"rollout_id": "r", "attempt": 1}
    async with make_client(participant) as client:
        failed = await _past_deadline(
            client,
            "restore",
            lambda: installing,
            "the install started",
            checkpoint_id="r1",
            checkpoint_dir=str(tmp_path / "source"),
            episode_ids=[{"rollout_id": "r"}],
        )
        # The controller retires the attempt the failed restore brought, as coordination does,
        # and that retire runs out of time waiting for the install; it still refuses the attempt.
        cut_short = await client.post(f"{ROUTE}/retire", json=body("r1", timeout=0.1, episode_ids=[restored_attempt]))
        resume = asyncio.create_task(client.post(f"{ROUTE}/resume", json=body("r1")))
        # The resume finds admission left closed by the failed restore, and waits for the install first.
        await asyncio.wait_for(resume_waiting.wait(), 5)
        closed_while_installing = not participant.accepting and not resume.done()
        release.set()
        resumed = await asyncio.wait_for(resume, 10)
        after = await status(client)
        installed_before_retry = set(participant.executions)
        retried = await client.post(f"{ROUTE}/retire", json=body("r1", episode_ids=[restored_attempt]))
        # The install's own reopen, scheduled when it finished, waited for the lock ahead of this retire,
        # and found admission already reopened.
        opened = participant.opened
        later = await committed_records(client, "ckpt-2", tmp_path / "2")

    assert failed.json()["error"]["code"] == "checkpoint_deadline_exceeded"
    assert cut_short.json()["error"]["code"] == "checkpoint_deadline_exceeded"
    assert closed_while_installing
    assert resumed.status_code == 200 and resumed.json()["phase"] == "idle"
    assert after["phase"] == "idle" and after["checkpoint_id"] is None and participant.accepting
    assert opened == 1
    # Both the replaced attempt and the attempt the failed restore brought stay refused until forget.
    for attempt in (0, 1):
        with pytest.raises(StaleAttemptError):
            participant.retired.check(EpisodeId(rollout_id="r", attempt=attempt))
    # The install finished after its restore failed, so its state is there until the retire is retried.
    assert installed_before_retry == {"r-a1"}
    assert retried.status_code == 200 and participant.executions == {}
    assert later == []
