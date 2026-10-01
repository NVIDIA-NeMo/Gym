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
        self.prepare_started = asyncio.Event()
        self.release_prepare = asyncio.Event()
        self.release_prepare.set()

    async def discover(self, client: Any, *, auth_token: str) -> str:
        return "participants"

    async def prepare(self, participants: Any, checkpoint_id: str, *, deadline_ts: float) -> PrepareResult:
        self.calls.append(("prepare", checkpoint_id))
        self.prepare_started.set()
        await self.release_prepare.wait()
        reply = {"phase": "prepared" if self.prepared else "preparing", "report": {"blockers": ["busy"]}}
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


@pytest.fixture
def coordination(monkeypatch: pytest.MonkeyPatch) -> FakeCoordination:
    fake = FakeCoordination()
    for name in ("discover", "prepare", "commit", "restore", "resume"):
        monkeypatch.setattr(collection_module.coordination, name, getattr(fake, name))
    return fake


def checkpointer(path: Path) -> CollectionCheckpointer:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    return CollectionCheckpointer(path, server_client)


def kinds(fake: FakeCoordination) -> list[str]:
    return [kind for kind, _ in fake.calls]


async def test_a_checkpoint_commits_the_rows_in_flight_and_publishes_them(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    checkpoints = checkpointer(tmp_path)
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
    checkpoints = checkpointer(tmp_path)
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
    checkpoints = checkpointer(tmp_path)
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
    checkpoints = checkpointer(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)

    published = await checkpoints.checkpoint(stop=True)
    blocked = asyncio.create_task(checkpoints.before_dispatch("1-0", 0))
    await asyncio.sleep(0.05)

    assert published is not None and checkpoints.stopped
    assert kinds(coordination) == ["prepare", "commit"]
    assert not blocked.done()
    blocked.cancel()


async def test_only_the_latest_two_checkpoints_are_kept(tmp_path: Path, coordination: FakeCoordination) -> None:
    checkpoints = checkpointer(tmp_path)
    await checkpoints.before_dispatch("0-0", 0)
    published = [await checkpoints.checkpoint() for _ in range(3)]

    assert [path.exists() for path in published] == [False, True, True]


async def test_restore_continues_only_unfinished_rows_as_their_next_attempt(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = checkpointer(tmp_path)
    await source.before_dispatch("0-0", 0)
    await source.before_dispatch("1-0", 1)
    await source.checkpoint()
    coordination.calls.clear()

    # Row 1-0 finished after the checkpoint, so it is already in the output and not restored.
    attempts = await checkpointer(tmp_path).restore(["0-0", "5-0"])

    assert attempts == {"0-0": 1}
    assert coordination.calls[0] == ("restore", [("0-0", 0)])
    assert kinds(coordination) == ["restore", "resume"]


async def test_a_failed_restore_restarts_its_rows_past_the_retired_attempt(
    tmp_path: Path, coordination: FakeCoordination
) -> None:
    source = checkpointer(tmp_path)
    await source.before_dispatch("0-0", 0)
    await source.checkpoint()
    coordination.restore_fails = True

    attempts = await checkpointer(tmp_path).restore(["0-0"])

    # Coordination retired attempt 1 everywhere when the restore failed, so the row starts over as attempt 2.
    assert attempts == {"0-0": 2}


async def test_a_fresh_run_forgets_earlier_checkpoints(tmp_path: Path, coordination: FakeCoordination) -> None:
    source = checkpointer(tmp_path)
    await source.before_dispatch("0-0", 0)
    published = await source.checkpoint()
    (tmp_path / "notes.txt").write_text("not a checkpoint")

    fresh = checkpointer(tmp_path)
    fresh.reset()

    assert fresh.latest() is None
    assert not published.exists()
    assert (tmp_path / "notes.txt").exists()
    assert await fresh.restore(["0-0"]) == {}


def test_a_checkpoint_dir_needs_checkpointing_enabled(tmp_path: Path) -> None:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({})

    with pytest.raises(ValueError, match="checkpoint:"):
        CollectionCheckpointer(tmp_path, server_client)


def test_episode_ids_are_the_same_for_every_attempt_of_a_row() -> None:
    from nemo_gym.rollout_collection import _episode_rollout_id, _native_episode_request_body

    row = {"_ng_task_index": 3, "_ng_rollout_index": 1, "_ng_attempt_index": 2}
    native = {"task_id": {"taskset": "t", "task_id": "x"}, "task_input": {}} | row

    assert _episode_rollout_id(row) == _episode_rollout_id(row | {"_ng_attempt_index": 0}) == "3-1"
    assert _native_episode_request_body(native)["episode_id"] == EpisodeId(rollout_id="3-1", attempt=2).model_dump()


async def test_the_timer_checkpoints_while_rows_are_in_flight(tmp_path: Path, coordination: FakeCoordination) -> None:
    checkpoints = checkpointer(tmp_path)
    checkpoints.every_s = 0.05
    collection = asyncio.create_task(asyncio.sleep(10))
    checkpoints.start(collection)
    await asyncio.sleep(0.12)
    idle = list(coordination.calls)
    await checkpoints.before_dispatch("0-0", 0)
    await asyncio.sleep(0.12)
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
