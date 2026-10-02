# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment server participant: record where each episode can continue.

An environment server owns the protocol of its episodes: which step runs next and what it needs. The
protocol records a boundary after each completed step and runs each step in ``wait`` or ``replay`` mode
(see ``nemo_gym._checkpoint.steps``). Prepare closes episode admission and is ready once every live
episode is parked at a boundary or inside a replay step. A restored boundary is handed to the
replacement attempt's ``/run``, which continues from it instead of starting over.
"""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional

from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    JsonPayload,
    PrepareReport,
    next_attempt,
)
from nemo_gym._checkpoint.errors import AdmissionClosedError, ControlError
from nemo_gym._checkpoint.steps import Boundary, EpisodeSteps, StepMode
from nemo_gym.episode_types import EpisodeId


class EpisodeRecord(CheckpointRecord):
    """One episode at its latest boundary."""

    task_digest: str
    boundary: JsonPayload


def task_digest(task: Any) -> str:
    return hashlib.sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class EnvironmentParticipant(CheckpointParticipant):
    kind = "environment"
    record_model = EpisodeRecord

    def __init__(self) -> None:
        super().__init__()
        self.steps = EpisodeSteps(self.notify)
        self._task_digests: dict[str, str] = {}
        self._restored: dict[str, EpisodeRecord] = {}

    def begin(self, episode_id: EpisodeId, task: Any, deadline: Optional[asyncio.Timeout]) -> None:
        """Register a starting episode, attaching its restored boundary if it continues one."""
        self.attempts.check(episode_id)
        key = episode_id.capture_key
        digest = task_digest(task)
        restored = self._restored.pop(key, None)
        if self.steps.closed and restored is None:
            raise AdmissionClosedError("episode admission is closed for a checkpoint")
        if restored is not None and restored.task_digest != digest:
            raise ControlError(f"episode {key} continues a checkpoint recorded for a different task")
        try:
            self.steps.begin(key, continuation=restored.boundary if restored else None, deadline=deadline)
        except ValueError as error:
            raise ControlError(str(error)) from error
        self._task_digests[key] = digest

    def finishing(self, episode_id: EpisodeId) -> None:
        """Call before the episode's final cleanup, so a retire cannot interrupt it."""
        self.steps.finishing(episode_id.capture_key)

    async def end(self, episode_id: EpisodeId) -> None:
        self._task_digests.pop(episode_id.capture_key, None)
        await self.steps.end(episode_id.capture_key)

    def continuation(self, episode_id: EpisodeId) -> Optional[Boundary]:
        return self.steps.continuation(episode_id.capture_key)

    async def boundary(self, episode_id: EpisodeId, state: Boundary) -> None:
        await self.steps.boundary(episode_id.capture_key, state)

    @asynccontextmanager
    async def step(self, episode_id: EpisodeId, mode: StepMode) -> AsyncIterator[None]:
        async with self.steps.step(episode_id.capture_key, mode):
            yield

    async def close_admission(self, request: CheckpointRequest) -> None:
        self.steps.close()

    async def open_admission(self) -> None:
        self.steps.open()

    def readiness(self) -> PrepareReport:
        blockers = self.steps.blockers()
        return PrepareReport(ready=not blockers, blockers=blockers, counts={"episodes": len(self.steps.keys())})

    async def retire(self, episode_id: EpisodeId) -> None:
        for key in [key for key in self.steps.keys() if _covers(episode_id, key)]:
            self._task_digests.pop(key, None)
            await self.steps.retire(key)
        for key in [key for key in self._restored if _covers(episode_id, key)]:
            del self._restored[key]

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        live = [
            EpisodeRecord(
                episode_id=EpisodeId.from_capture_key(key),
                task_digest=self._task_digests[key],
                boundary=boundary,
            )
            for key, boundary in self.steps.exported().items()
        ]
        # A restored episode whose replacement has not started yet still continues from its restored
        # boundary, so a checkpoint taken before the replacement runs must carry it forward.
        pending = [
            EpisodeRecord(
                episode_id=EpisodeId.from_capture_key(key), task_digest=record.task_digest, boundary=record.boundary
            )
            for key, record in self._restored.items()
        ]
        return live + pending

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        if self.steps.keys() or self._restored:
            raise ControlError("environment restore requires a process without live episodes")
        self._restored = {next_attempt(record.episode_id).capture_key: record for record in records}

    async def restored_pending(self) -> list[EpisodeId]:
        return [EpisodeId.from_capture_key(key) for key in self._restored]

    def status_extra(self) -> dict[str, Any]:
        return {"restored_pending": sorted(self._restored)}


def _covers(retired: EpisodeId, capture_key: str) -> bool:
    """Whether retiring ``retired`` discards the attempt named by ``capture_key``."""
    other = EpisodeId.from_capture_key(capture_key)
    return other.rollout_id == retired.rollout_id and other.attempt <= retired.attempt
