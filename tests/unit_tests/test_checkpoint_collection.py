# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollout collection as the checkpoint controller of an evaluation run.

Most tests replace coordination with a fake.
The ones that check what the servers are left with use a real environment participant behind its control routes.
"""

import asyncio
import json
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiohttp import ClientResponseError
from fastapi import FastAPI
from omegaconf import DictConfig, OmegaConf

import nemo_gym.rollout_collection
from nemo_gym._checkpoint import collection as collection_module
from nemo_gym._checkpoint.collection import CollectionCheckpointer, CollectionStopped
from nemo_gym._checkpoint.control import install_participant
from nemo_gym._checkpoint.coordination import CoordinationError, PrepareResult
from nemo_gym._checkpoint.environment import EnvironmentParticipant
from nemo_gym._checkpoint.errors import ControlError
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.rollout_collection import (
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _episode_rollout_id,
)
from nemo_gym.server_utils import ServerClient
from tests.unit_tests.test_checkpoint_control import _until
from tests.unit_tests.test_checkpoint_coordination import Recorder
from tests.unit_tests.test_rollout_collection import (  # noqa: F401 - empty_global_config is a fixture
    FakeResponse,
    empty_global_config,
    install_fake_server_client,
)


class FakeCoordination:
    """Records every coordination call; commit writes a marker where participants would write records."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.prepared = True
        self.restore_fails = False
        # Retire calls that fail before one succeeds.
        self.retire_failures = 0
        # Forget calls refused before one succeeds, as a server refuses while it still holds a session of the rollout.
        self.forget_failures = 0
        # Capture keys a participant reports it cannot capture.
        self.restarts: list[str] = []
        self.prepare_started = asyncio.Event()
        self.release_prepare = asyncio.Event()
        self.release_prepare.set()

    async def discover(self, client: Any, *, auth_token: str) -> str:
        return "participants"

    async def prepare(self, participants: Any, checkpoint_id: str, *, deadline_ts: float) -> PrepareResult:
        self.calls.append(("prepare", checkpoint_id))
        self.prepare_started.set()
        await self.release_prepare.wait()
        reply = {
            "phase": "prepared" if self.prepared else "preparing",
            "report": {"blockers": ["busy"], "restarts": self.restarts},
        }
        return PrepareResult(prepared=self.prepared, replies={"environment": reply})

    async def commit(
        self, participants: Any, checkpoint_id: str, checkpoint_dir: str, episodes: list, *, deadline_ts: float
    ) -> dict:
        self.calls.append(("commit", sorted((episode.rollout_id, episode.attempt) for episode in episodes)))
        Path(checkpoint_dir, "gym").mkdir(parents=True)
        return {}

    async def restore(
        self, participants: Any, restore_id: str, checkpoint_dir: str, episodes: list, *, deadline_ts: float
    ) -> dict:
        self.calls.append(("restore", sorted((episode.rollout_id, episode.attempt) for episode in episodes)))
        if self.restore_fails:
            raise CoordinationError("restore", {"agent": "boom"})
        return {}

    async def resume(self, participants: Any, checkpoint_id: str, *, deadline_ts: float) -> None:
        self.calls.append(("resume", checkpoint_id))

    async def retire(self, participants: Any, checkpoint_id: str, episodes: list, *, deadline_ts: float) -> None:
        self.calls.append(("retire", sorted((episode.rollout_id, episode.attempt) for episode in episodes)))
        if self.retire_failures:
            self.retire_failures -= 1
            raise CoordinationError("retire", {"model": "unreachable"})

    async def forget(self, participants: Any, checkpoint_id: str, rollout_ids: list, *, deadline_ts: float) -> None:
        self.calls.append(("forget", sorted(rollout_ids)))
        if self.forget_failures:
            self.forget_failures -= 1
            raise CoordinationError("forget", {"resources": "retire_incomplete"})


@pytest.fixture
def coordination(monkeypatch: pytest.MonkeyPatch) -> FakeCoordination:
    fake = FakeCoordination()
    for name in ("discover", "prepare", "commit", "restore", "resume", "retire", "forget"):
        monkeypatch.setattr(collection_module.coordination, name, getattr(fake, name))
    return fake


def checkpointer(path: Path) -> CollectionCheckpointer:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    return CollectionCheckpointer(path, server_client)


def started(path: Path) -> CollectionCheckpointer:
    """A fresh run, which starts its dispatch log before it dispatches, as rollout collection does."""
    checkpoints = checkpointer(path)
    checkpoints.reset()
    return checkpoints


def kinds(fake: FakeCoordination) -> list[str]:
    return [kind for kind, _ in fake.calls]


async def test_a_checkpoint_commits_the_rows_in_flight_and_publishes_them(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)
    await checkpoints.before_dispatch("1-0", 2)
    await checkpoints.before_dispatch("2-0", 0)
    checkpoints.after_dispatch("2-0")

    published = await checkpoints.checkpoint()

    manifest = json.loads((published / "collection.json").read_text())
    assert kinds(coordination) == ["prepare", "commit", "resume"]
    assert coordination.calls[1][1] == [("0-0", 0), ("1-0", 2)]
    assert (tmp_path / "LATEST").read_text() == published.name
    assert [(row["rollout_id"], row["attempt"]) for row in manifest["continued"]] == [("0-0", 0), ("1-0", 2)]


async def test_no_row_starts_while_a_checkpoint_is_open(tmp_path: Path, coordination: FakeCoordination) -> None:
    checkpoints = started(tmp_path)
    coordination.release_prepare.clear()
    running = asyncio.create_task(checkpoints.checkpoint())
    await coordination.prepare_started.wait()
    dispatch = asyncio.create_task(checkpoints.before_dispatch("0-0", 0))
    await asyncio.sleep(0.05)
    waited = not dispatch.done()
    coordination.release_prepare.set()
    await running
    await dispatch

    assert waited
    # The row started after the checkpoint, so that checkpoint did not continue it.
    assert coordination.calls[1] == ("commit", [])


async def test_a_checkpoint_that_does_not_prepare_publishes_nothing_and_resumes(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)
    coordination.prepared = False

    published = await checkpoints.checkpoint()
    await asyncio.wait_for(checkpoints.before_dispatch("1-0", 0), timeout=1)

    assert published is None
    assert kinds(coordination) == ["prepare", "resume"]
    assert not (tmp_path / "LATEST").exists()


async def test_a_checkpoint_that_stops_does_not_resume_or_start_more_rows(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)

    published = await checkpoints.checkpoint(stop=True)
    blocked = asyncio.create_task(checkpoints.before_dispatch("1-0", 0))
    await asyncio.sleep(0.05)

    assert published is not None and checkpoints.stopped
    assert kinds(coordination) == ["prepare", "commit"]
    assert not blocked.done()
    blocked.cancel()


async def test_only_the_latest_two_checkpoints_are_kept(tmp_path: Path, coordination: FakeCoordination) -> None:
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)
    published = [await checkpoints.checkpoint() for _ in range(3)]

    assert [path.exists() for path in published] == [False, True, True]


async def test_restore_continues_only_unfinished_rows_as_their_next_attempt(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    await source.before_dispatch("1-0", 1)
    await source.checkpoint()
    coordination.calls.clear()

    # Row 1-0 finished after the checkpoint, so it is already in the output and not restored.
    # Row 5-0 is new.
    attempts = await checkpointer(tmp_path).restore({"0-0": 0, "5-0": 0})

    assert attempts == {"0-0": 1, "5-0": 0}
    assert coordination.calls[0] == ("restore", [("0-0", 0)])
    assert kinds(coordination) == ["restore", "resume"]


async def test_a_failed_restore_restarts_its_rows_past_the_retired_attempt(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    await source.checkpoint()
    coordination.restore_fails = True

    restored = checkpointer(tmp_path)
    attempts = await restored.restore({"0-0": 0})

    # Coordination retired attempt 1 everywhere when the restore failed, so the row starts over as attempt 2.
    assert attempts == {"0-0": 2}
    # The log carries the retired attempt 1, so if this run crashes before it sends attempt 2,
    # the next run retires attempt 1 again and sends attempt 2 without restoring.
    coordination.calls.clear()
    assert await checkpointer(tmp_path).restore({"0-0": 0}) == {"0-0": 2}
    assert coordination.calls == [("retire", [("0-0", 1)])]


async def test_a_fresh_run_forgets_earlier_checkpoints(tmp_path: Path, coordination: FakeCoordination) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    published = await source.checkpoint()
    (tmp_path / "notes.txt").write_text("not a checkpoint")

    fresh = started(tmp_path)

    assert fresh.latest() is None
    assert not published.exists()
    assert (tmp_path / "notes.txt").exists()


def test_a_checkpoint_dir_needs_checkpointing_enabled(tmp_path: Path) -> None:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({})

    with pytest.raises(ValueError, match="checkpoint:"):
        CollectionCheckpointer(tmp_path, server_client)


def test_episode_ids_are_the_same_for_every_attempt_of_a_row() -> None:
    from nemo_gym.rollout_collection import _episode_request_body, _episode_rollout_id

    row = {"_ng_task_index": 3, "_ng_rollout_index": 1, "_ng_attempt_index": 2}
    native = {"task_id": {"taskset": "t", "task_id": "x"}, "task_input": {}} | row

    assert _episode_rollout_id(row) == _episode_rollout_id(row | {"_ng_attempt_index": 0}) == "3-1"
    assert _episode_request_body(native)["episode_id"] == EpisodeId(rollout_id="3-1", attempt=2).model_dump()


async def test_the_timer_checkpoints_while_rows_are_in_flight(tmp_path: Path, coordination: FakeCoordination) -> None:
    checkpoints = started(tmp_path)
    checkpoints.every_s = 0.05
    ticks = 0

    class Watched(dict):
        def __len__(self) -> int:
            nonlocal ticks
            ticks += 1
            return super().__len__()

    # Counts the timer's checks for rows in flight, so the idle claim is checked only after it ticked.
    checkpoints._in_flight = Watched()
    collection = asyncio.create_task(asyncio.sleep(10))
    checkpoints.start(collection)
    async with asyncio.timeout(5):
        while ticks < 2:
            await asyncio.sleep(0.01)
    idle = list(coordination.calls)
    await checkpoints.before_dispatch("0-0", 0)
    # Wait for the timer to checkpoint the row; closing earlier would stop the timer before its next tick.
    async with asyncio.timeout(5):
        while ("commit", [("0-0", 0)]) not in coordination.calls:
            await asyncio.sleep(0.01)
    await checkpoints.close()
    collection.cancel()

    # Nothing is in flight at first, so the timer skips; once a row runs, it checkpoints that row.
    assert idle == []
    assert ("commit", [("0-0", 0)]) in coordination.calls
    assert (tmp_path / "collector.pid").exists()


def test_the_checkpoint_timer_needs_a_checkpoint_dir() -> None:
    from nemo_gym.rollout_collection import RolloutCollectionConfig

    with pytest.raises(ValueError, match="checkpoint_every_s needs checkpoint_dir"):
        RolloutCollectionConfig(input_jsonl_fpath="in.jsonl", output_jsonl_fpath="out.jsonl", checkpoint_every_s=60)


async def test_restored_rows_not_dispatched_yet_are_continued_by_the_next_checkpoint(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    await source.before_dispatch("1-0", 0)
    await source.checkpoint()
    restored = checkpointer(tmp_path)
    await restored.restore({"0-0": 0, "1-0": 0})
    coordination.calls.clear()

    # Row 0-0's replacement has started; row 1-0's has not, so participants still hold its restored state.
    await restored.before_dispatch("0-0", 1)
    await restored.checkpoint()

    assert coordination.calls[1] == ("commit", [("0-0", 1), ("1-0", 1)])


async def test_rows_that_fail_are_retired_and_rows_that_succeed_are_not(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)
    await checkpoints.before_dispatch("1-0", 2)
    checkpoints.after_dispatch("0-0")
    checkpoints.after_dispatch("1-0", failed=True)
    await checkpoints.close()

    # Nothing will dispatch the retired row again once the run closes, so it is forgotten then.
    assert coordination.calls == [("retire", [("1-0", 2)]), ("forget", ["1-0"])]


async def test_a_retired_row_is_forgotten_once_a_later_attempt_replies(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("r", 0)
    checkpoints.after_dispatch("r", failed=True)
    await asyncio.sleep(0)
    await checkpoints.before_dispatch("r", 1)
    checkpoints.after_dispatch("r")
    for _ in range(3):
        await asyncio.sleep(0)
    calls_before_close = list(coordination.calls)
    await checkpoints.close()

    assert calls_before_close == [("retire", [("r", 0)]), ("forget", ["r"])]
    assert coordination.calls == calls_before_close


async def test_a_refused_forget_retires_the_rollout_again_before_the_next_checkpoint(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    """A session a protocol route started after the retire makes the server refuse the forget until it is freed."""
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("r", 0)
    checkpoints.after_dispatch("r", failed=True)
    await asyncio.sleep(0)
    coordination.forget_failures = 1
    await checkpoints.before_dispatch("r", 1)
    checkpoints.after_dispatch("r")
    for _ in range(3):
        await asyncio.sleep(0)
    await checkpoints.before_dispatch("s", 0)
    coordination.calls.clear()
    await checkpoints.checkpoint()

    # Before the checkpoint prepares, the refused rollout is retired again at the attempt it retired, then forgotten.
    assert coordination.calls[:2] == [("retire", [("r", 0)]), ("forget", ["r"])]
    assert coordination.calls[2][0] == "prepare"
    coordination.calls.clear()
    await checkpoints.close()
    assert ("forget", ["r"]) not in coordination.calls


async def test_a_forget_refused_when_the_run_closes_is_retired_and_forgotten_again(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("1-0", 2)
    checkpoints.after_dispatch("1-0", failed=True)
    coordination.forget_failures = 1
    await checkpoints.close()

    assert coordination.calls == [
        ("retire", [("1-0", 2)]),
        ("forget", ["1-0"]),
        ("retire", [("1-0", 2)]),
        ("forget", ["1-0"]),
    ]


async def test_publishing_removes_partial_checkpoints_and_temporary_files(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    (tmp_path / "ckpt-20260101T000000-deadbeef" / "gym").mkdir(parents=True)
    (tmp_path / ".LATEST.0123abcd").write_text("stale")
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)

    published = await checkpoints.checkpoint()

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(["LATEST", "dispatched.jsonl", published.name])


async def test_a_checkpoint_leaves_restarts_out_of_its_scope_and_records_them(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)
    await checkpoints.before_dispatch("1-0", 2)
    # A restart-only agent runs 1-0, so nothing of it is captured.
    coordination.restarts = ["1-0-a2"]

    published = await checkpoints.checkpoint()

    manifest = json.loads((published / "collection.json").read_text())
    assert coordination.calls[1] == ("commit", [("0-0", 0)])
    assert [(row["rollout_id"], row["attempt"]) for row in manifest["continued"]] == [("0-0", 0)]
    assert [(row["rollout_id"], row["attempt"]) for row in manifest["restarted"]] == [("1-0", 2)]


async def test_restore_retires_unfinished_restarts_and_starts_them_over_as_their_next_attempt(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    await source.before_dispatch("1-0", 2)
    await source.before_dispatch("2-0", 0)
    coordination.restarts = ["1-0-a2", "2-0"]
    await source.checkpoint()
    coordination.calls.clear()

    # Row 2-0 finished after the checkpoint, so it is neither retired nor sent again.
    restored = checkpointer(tmp_path)
    attempts = await restored.restore({"0-0": 0, "1-0": 2})
    await restored.close()

    assert attempts == {"0-0": 1, "1-0": 3}
    # A harness of attempt 2 that outlived the crash is refused before the row starts over.
    assert coordination.calls[0] == ("retire", [("1-0", 2)])
    assert ("restore", [("0-0", 0)]) in coordination.calls
    # The run closes: the retired row, and the restored row whose replaced attempt is refused, are forgotten.
    assert coordination.calls[-1] == ("forget", ["0-0", "1-0"])


async def test_restore_retires_rows_sent_after_the_checkpoint_and_sends_them_as_a_later_attempt(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    await source.checkpoint()
    # Sent after the checkpoint, then the run crashed: its harness may still be running attempt 0.
    await source.before_dispatch("1-0", 0)
    coordination.calls.clear()

    attempts = await checkpointer(tmp_path).restore({"0-0": 0, "1-0": 0})

    assert attempts == {"0-0": 1, "1-0": 1}
    assert coordination.calls[0] == ("retire", [("1-0", 0)])
    assert ("restore", [("0-0", 0)]) in coordination.calls


async def test_a_checkpoint_older_than_the_last_run_is_not_restored(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    await source.checkpoint()
    # The next run restores the row and sends it as attempt 1, then crashes before its own first checkpoint.
    second = checkpointer(tmp_path)
    assert await second.restore({"0-0": 0}) == {"0-0": 1}
    await second.before_dispatch("0-0", 1)
    coordination.calls.clear()

    attempts = await checkpointer(tmp_path).restore({"0-0": 0})

    # Restoring the first checkpoint again would install attempt 1 next to the one still running.
    assert "restore" not in kinds(coordination)
    assert coordination.calls == [("retire", [("0-0", 1)])]
    assert attempts == {"0-0": 2}


async def test_a_run_without_a_dispatch_log_restores_nothing_and_keeps_its_attempts(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    assert await checkpointer(tmp_path).restore({"0-0": 0, "1-0": 2}) == {"0-0": 0, "1-0": 2}
    assert coordination.calls == []


async def test_a_failed_retire_is_repeated_before_the_next_checkpoint_and_at_close(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)
    await checkpoints.before_dispatch("1-0", 0)
    coordination.retire_failures = 2
    checkpoints.after_dispatch("0-0", failed=True)
    await asyncio.sleep(0)
    await checkpoints.checkpoint()
    await checkpoints.close()

    # The first retire fails, the one before the checkpoint fails again, and the one at close succeeds.
    retire = ("retire", [("0-0", 0)])
    assert [call if call[0] == "retire" else call[0] for call in coordination.calls] == [
        retire,
        retire,
        "prepare",
        "commit",
        "resume",
        retire,
        "forget",
    ]


async def test_only_rows_a_run_sent_are_fenced_after_it_crashes(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    coordination.calls.clear()

    # Row 1-0 was never sent, so nothing of it can be running: it keeps its attempt and needs no retire.
    attempts = await checkpointer(tmp_path).restore({"0-0": 0, "1-0": 0})

    assert attempts == {"0-0": 1, "1-0": 0}
    assert coordination.calls == [("retire", [("0-0", 0)])]


async def test_a_fence_is_carried_through_a_run_that_crashes_before_sending_the_row(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    await checkpointer(tmp_path).restore({"0-0": 0})
    coordination.calls.clear()

    # The second run fenced attempt 0 and crashed before sending attempt 1.
    # Servers refuse attempt 0 until the rollout is forgotten, so the third run must not send it again.
    attempts = await checkpointer(tmp_path).restore({"0-0": 0})

    assert attempts == {"0-0": 1}
    assert coordination.calls == [("retire", [("0-0", 0)])]


async def test_a_partial_last_line_of_a_killed_run_is_ignored(tmp_path: Path, coordination: FakeCoordination) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    with open(tmp_path / "dispatched.jsonl", "ab") as log:
        log.write(b'["1-0",')

    assert await checkpointer(tmp_path).restore({"0-0": 0, "1-0": 0}) == {"0-0": 1, "1-0": 0}


async def test_a_restored_row_is_forgotten_once_its_replacement_replies_or_when_the_run_closes(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = started(tmp_path)
    await source.before_dispatch("0-0", 0)
    await source.before_dispatch("1-0", 0)
    await source.checkpoint()
    restored = checkpointer(tmp_path)
    await restored.restore({"0-0": 0, "1-0": 0})
    coordination.calls.clear()

    # Every participant refuses the attempts the restore replaced until the controller forgets their rollouts.
    await restored.before_dispatch("0-0", 1)
    restored.after_dispatch("0-0")
    async with asyncio.timeout(5):
        while ("forget", ["0-0"]) not in coordination.calls:
            await asyncio.sleep(0.01)
    replied = list(coordination.calls)
    await restored.close()

    assert replied == [("forget", ["0-0"])]
    # Row 1-0's replacement was never sent; nothing dispatches it once the run closes, so it is forgotten then.
    assert coordination.calls == [*replied, ("forget", ["1-0"])]


# -- a deployment with a real environment participant and control plane -----------------------------------------------


class _Reply:
    """The parts of aiohttp's ``ClientResponse`` that rollout collection and coordination read."""

    def __init__(self, response: httpx.Response) -> None:
        self.status = response.status_code
        self.ok = response.is_success
        self._body = response.content
        self.content = SimpleNamespace(read=self.read)

    async def read(self) -> bytes:
        return self._body

    def release(self) -> None:
        pass

    def raise_for_status(self) -> None:
        if not self.ok:
            request_info = SimpleNamespace(method="POST", url="http://env/run", real_url="http://env/run", headers={})
            raise ClientResponseError(request_info=request_info, history=(), status=self.status, message="refused")


class Deployment:
    """An environment server and a policy model, reached in process.

    The environment server is the real ``EnvironmentParticipant`` behind the real control routes.
    Its ``/run`` begins the row's episode, which runs until the test opens the row's gate.
    The model is a participant that is always ready, because coordination needs a policy model.
    ``request_slots`` bounds the ``/run`` requests in flight, as an HTTP connection pool does.
    """

    def __init__(self, *, request_slots: int = 8) -> None:
        self.participant = EnvironmentParticipant()
        self.gates: dict[str, asyncio.Event] = {}
        self.refusals: list[tuple[str, str]] = []
        self.slots = asyncio.Semaphore(request_slots)
        environment = FastAPI()
        install_participant(
            environment, self.participant, auth_token="t", instance_name="environment", lease_grace_seconds=60
        )

        @environment.post("/run")
        async def run(row: dict) -> dict:
            episode_id = EpisodeId(rollout_id=_episode_rollout_id(row), attempt=row.get(ATTEMPT_INDEX_KEY_NAME, 0))
            try:
                self.participant.begin(episode_id, row["responses_create_params"], None)
            except ControlError as error:
                self.refusals.append((episode_id.capture_key, error.code))
                raise
            try:
                await self.gate(episode_id.rollout_id).wait()
            finally:
                self.participant.finishing(episode_id)
                await self.participant.end(episode_id)
            return {"reward": 1.0, "response": {}}

        model = FastAPI()
        install_participant(
            model, Recorder("model", [], "policy"), auth_token="t", instance_name="policy", lease_grace_seconds=60
        )
        self.apps = {"my_environment_server": environment, "policy": model, "my_agent": FastAPI()}
        self.client = MagicMock(spec=ServerClient)
        self.client.global_config_dict = OmegaConf.create(
            {
                "checkpoint": {"enabled": True, "control_auth_token": "t"},
                "my_agent": {"responses_api_agents": {"impl": {}}},
                "my_environment_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}
                },
                "policy": {"responses_api_models": {"policy": {}}},
            }
        )
        self.client.request = self._request
        self.client.post = self._post

    def gate(self, rollout_id: str) -> asyncio.Event:
        return self.gates.setdefault(rollout_id, asyncio.Event())

    async def _send(self, server_name: str, method: str, url_path: str, **kwargs: Any) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.apps[server_name])
        async with httpx.AsyncClient(transport=transport, base_url=f"http://{server_name}", timeout=60) as client:
            return await client.request(method, url_path, **kwargs)

    async def _request(self, server_name: str, url_path: str, method: str, **kwargs: Any) -> Any:
        response = await self._send(
            server_name, method, url_path, content=kwargs.get("data"), headers=kwargs.get("headers")
        )
        return _Reply(response)

    async def _post(self, server_name: str, url_path: str, json: dict) -> _Reply:
        async with self.slots:
            return _Reply(await self._send(server_name, "POST", url_path, json=json))

    def checkpointer(self, path: Path, **kwargs: Any) -> CollectionCheckpointer:
        return CollectionCheckpointer(path, self.client, **kwargs)

    async def status(self) -> dict:
        response = await self._send(
            "my_environment_server", "GET", "/ng-control/v1/checkpoint/status", headers={"Authorization": "Bearer t"}
        )
        return response.json()


def collection_row(task_index: int) -> dict:
    return {
        AGENT_REF_KEY_NAME: {"name": "my_agent"},
        TASK_INDEX_KEY_NAME: task_index,
        ROLLOUT_INDEX_KEY_NAME: 0,
        "responses_create_params": {"input": [{"role": "user", "content": f"task {task_index}"}]},
    }


@pytest.mark.parametrize("route_failures_to_sidecar", [False, True])
async def test_a_row_refused_because_a_checkpoint_closed_admission_runs_after_the_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route_failures_to_sidecar: bool
) -> None:
    """A row that passed the dispatch gate but reached the environment server after prepare is sent again.

    Rows 0 and 1 take both request slots, so row 2 waits for one after it passed ``before_dispatch``.
    A checkpoint closes the environment server's admission while rows 0 and 1 run.
    Row 0 then finishes, and row 2 takes its slot and is refused with ``admission_closed``.
    Once row 1 finishes, the checkpoint prepares, commits and resumes, and row 2 is sent again as the same attempt.
    """
    deployment = Deployment(request_slots=2)
    monkeypatch.setattr(
        nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: deployment.client
    )
    checkpoints = deployment.checkpointer(tmp_path / "checkpoints")
    checkpoints.reset()
    rows = [collection_row(task_index) for task_index in range(3)]
    completions = RolloutCollectionHelper()._run_examples_with_metadata(
        rows, route_failures_to_sidecar=route_failures_to_sidecar, checkpointer=checkpoints
    )
    collection = asyncio.gather(*completions, return_exceptions=True)
    await _until(lambda: len(checkpoints._in_flight) == 3, "every row passed the dispatch gate")

    checkpoint = asyncio.create_task(checkpoints.checkpoint())
    await _until(lambda: deployment.participant.steps.closed, "the checkpoint closed admission")
    deployment.gate("0-0").set()
    await _until(lambda: deployment.refusals, "row 2 was refused")
    deployment.gate("1-0").set()
    published = await asyncio.wait_for(checkpoint, 10)
    deployment.gate("2-0").set()
    results = await asyncio.wait_for(collection, 10)
    await checkpoints.close()

    assert deployment.refusals == [("2-0", "admission_closed")]
    assert published is not None
    assert [completed.result.get("reward") for completed in results] == [1.0, 1.0, 1.0]
    assert sorted(completed.row[TASK_INDEX_KEY_NAME] for completed in results) == [0, 1, 2]
    # Row 2 was sent again as attempt 0, and nothing counted it as failed: no attempt of it was retired.
    assert all(completed.row.get(ATTEMPT_INDEX_KEY_NAME, 0) == 0 for completed in results)
    assert checkpoints._failed == [] and checkpoints._unretired == [] and checkpoints._retired == {}
    assert (await deployment.status())["phase"] == "idle"


async def test_sigterm_with_the_timing_report_on_checkpoints_stops_and_prints_the_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    coordination: FakeCoordination,
    capsys: pytest.CaptureFixture[str],
    empty_global_config: MagicMock,
) -> None:
    """The checkpoint's SIGTERM path and the timing report's SIGTERM print both run when both are enabled."""
    input_jsonl_fpath = tmp_path / "input.jsonl"
    input_jsonl_fpath.write_text(json.dumps(collection_row(0) | {"x": 0}) + "\n")
    running = asyncio.Event()

    async def post(server_name: str, url_path: str, json: dict, **kwargs: Any) -> FakeResponse:
        running.set()
        await asyncio.Event().wait()

    server_client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    server_client.global_config_dict = OmegaConf.merge(
        server_client.global_config_dict, {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    )
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(input_jsonl_fpath),
        output_jsonl_fpath=str(tmp_path / "output.jsonl"),
        disable_health_check=True,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        dispatch_start_report_within_s=60,
        timing_summary_interval_s=0,
    )
    before = signal.getsignal(signal.SIGTERM)

    async def preempt() -> None:
        await running.wait()
        os.kill(os.getpid(), signal.SIGTERM)

    preemption = asyncio.create_task(preempt())
    with pytest.raises(CollectionStopped) as stopped:
        await asyncio.wait_for(RolloutCollectionHelper().run_from_config(config), 10)
    await preemption

    assert kinds(coordination) == ["prepare", "commit"]
    assert stopped.value.checkpoint_dir is not None and (stopped.value.checkpoint_dir / "collection.json").exists()
    # SIGTERM printed the timing table before the checkpoint published; the run prints it again as it ends.
    out = capsys.readouterr().out
    assert out.index("Run timing summary") < out.index(f"Checkpoint {stopped.value.checkpoint_dir.name} published")
    # Collection gave SIGTERM back to whatever handled it before.
    assert signal.getsignal(signal.SIGTERM) == before


def delay_prepare_replies(monkeypatch: pytest.MonkeyPatch) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold each prepare reply after every participant prepared; returns the prepared and release events."""
    prepared = asyncio.Event()
    reply = asyncio.Event()
    real_prepare = collection_module.coordination.prepare

    async def delayed_prepare(*args: Any, **kwargs: Any) -> PrepareResult:
        result = await real_prepare(*args, **kwargs)
        prepared.set()
        await reply.wait()
        return result

    monkeypatch.setattr(collection_module.coordination, "prepare", delayed_prepare)
    return prepared, reply


async def test_closing_during_a_timed_checkpoint_leaves_every_participant_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run ends while a timer checkpoint waits for its prepare reply; closing must not leave servers prepared."""
    deployment = Deployment()
    checkpoints = deployment.checkpointer(tmp_path / "checkpoints", every_s=0.01)
    checkpoints.reset()
    prepared, reply = delay_prepare_replies(monkeypatch)
    collection = asyncio.create_task(asyncio.sleep(10))
    checkpoints.start(collection)
    await checkpoints.before_dispatch("0-0", 0)
    await asyncio.wait_for(prepared.wait(), 5)
    # The last row replies while the prepare reply is still on its way, so the run ends.
    checkpoints.after_dispatch("0-0")
    collection.cancel()
    closing = asyncio.create_task(checkpoints.close())
    await asyncio.sleep(0)
    reply.set()
    await asyncio.wait_for(closing, 10)

    # Closing waited for the checkpoint, which published and resumed.
    assert (tmp_path / "checkpoints" / "LATEST").exists()
    status = await deployment.status()
    assert status["phase"] == "idle" and not deployment.participant.steps.closed
    # The next episode is admitted.
    deployment.participant.begin(EpisodeId(rollout_id="1-0"), {"input": []}, None)


async def test_a_checkpoint_cancelled_while_prepared_resumes_every_participant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment = Deployment()
    checkpoints = deployment.checkpointer(tmp_path / "checkpoints")
    checkpoints.reset()
    prepared, _ = delay_prepare_replies(monkeypatch)
    await checkpoints.before_dispatch("0-0", 0)
    checkpoint = asyncio.create_task(checkpoints.checkpoint())
    await asyncio.wait_for(prepared.wait(), 5)

    checkpoint.cancel()
    with pytest.raises(asyncio.CancelledError):
        await checkpoint

    # The resume finished before the cancellation reached the caller.
    assert (await deployment.status())["phase"] == "idle" and not deployment.participant.steps.closed
    assert not (tmp_path / "checkpoints" / "LATEST").exists()
    # Rows start again.
    await asyncio.wait_for(checkpoints.before_dispatch("1-0", 0), 1)


async def test_a_checkpoint_signal_during_a_resumed_start_waits_for_the_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SIGUSR1 while a resumed run reads its dispatch log must not publish over the checkpoint it restores.

    The previous run checkpointed row 0-0 at a boundary, then crashed.
    The new run gets the signal while it reads the previous run's dispatch log.
    """
    directory = tmp_path / "checkpoints"
    previous = Deployment()
    in_step = asyncio.Event()

    async def episode() -> None:
        # The episode records a boundary, then runs a replay step, which a checkpoint does not wait for.
        episode_id = EpisodeId(rollout_id="0-0")
        previous.participant.begin(episode_id, {"input": []}, None)
        await previous.participant.boundary(episode_id, {"step": 1})
        async with previous.participant.step(episode_id, "replay"):
            in_step.set()
            await asyncio.Event().wait()

    running = asyncio.create_task(episode())
    await asyncio.wait_for(in_step.wait(), 5)
    source = previous.checkpointer(directory)
    source.reset()
    await source.before_dispatch("0-0", 0)
    seeded = await source.checkpoint()
    # The previous run crashes.
    running.cancel()
    assert seeded is not None

    deployment = Deployment()
    checkpoints = deployment.checkpointer(directory)
    published = threading.Event()
    real_publish = CollectionCheckpointer._publish
    real_read_log = CollectionCheckpointer._read_log

    def watched_publish(self: CollectionCheckpointer, target: Path, manifest: Any) -> None:
        real_publish(self, target, manifest)
        published.set()

    def read_log_during_a_signal(self: CollectionCheckpointer) -> Any:
        os.kill(os.getpid(), signal.SIGUSR1)
        # Long enough for a checkpoint that does not wait for the restore to publish first.
        published.wait(1)
        return real_read_log(self)

    monkeypatch.setattr(CollectionCheckpointer, "_publish", watched_publish)
    monkeypatch.setattr(CollectionCheckpointer, "_read_log", read_log_during_a_signal)
    collection = asyncio.create_task(asyncio.sleep(10))
    checkpoints.start(collection)
    attempts = await checkpoints.restore({"0-0": 0})
    await _until(lambda: (directory / "LATEST").read_text() != seeded.name, "the signal's checkpoint published")
    latest = checkpoints.latest()
    await checkpoints.close()
    collection.cancel()

    # The row continues from its boundary as attempt 1 instead of starting over.
    assert attempts == {"0-0": 1}
    assert "0-0-a1" in (await deployment.status())["restored_pending"]
    # The signal's checkpoint ran after the restore, so it continues the restored row too.
    assert latest is not None
    assert [(row.rollout_id, row.attempt) for row in latest[1].continued] == [("0-0", 1)]


async def test_retention_keeps_the_published_checkpoint_when_the_clock_goes_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, coordination: FakeCoordination
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(
        collection_module,
        "time",
        SimpleNamespace(time=lambda: clock[0], monotonic=time.monotonic, strftime=time.strftime),
    )
    checkpoints = started(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)
    await checkpoints.checkpoint()
    clock[0] = 1001.0
    await checkpoints.checkpoint()
    # The clock steps back, so the newest checkpoint has the oldest creation time.
    clock[0] = 500.0
    newest = await checkpoints.checkpoint()

    assert (tmp_path / "LATEST").read_text() == newest.name
    assert newest.exists()
    assert checkpoints.latest()[0] == newest
    assert len([path for path in tmp_path.iterdir() if (path / "collection.json").exists()]) == 2
