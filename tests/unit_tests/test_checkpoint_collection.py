# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollout collection as the checkpoint controller of an evaluation run, with coordination replaced by a fake."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig

from nemo_gym._checkpoint import collection as collection_module
from nemo_gym._checkpoint.collection import CollectionCheckpointer
from nemo_gym._checkpoint.coordination import CoordinationError, PrepareResult
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import ServerClient


class FakeCoordination:
    """Records every coordination call; commit writes a marker where participants would write records."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.prepared = True
        self.restore_fails = False
        # Retire calls that fail before one succeeds.
        self.retire_failures = 0
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
    # Closing cancels a timer checkpoint still under way, so wait for its commit first.
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
    assert coordination.calls[-1] == ("forget", ["1-0"])


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
