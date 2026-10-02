# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Episode boundaries and in-flight step modes, shared by every owner of an episode protocol.

An episode is a sequence of steps. After each completed step its owner records a boundary: the next
step and the results that step needs. A checkpoint can always be taken at a boundary. A step that is
still running when a checkpoint closes admission has one of two modes:

- ``wait`` (the default): the checkpoint waits for the step. The owner then records the boundary after
  it, with the step's result, and parks there. Use it for any step that is unsafe to run twice.
- ``replay``: the checkpoint does not wait. The episode counts as parked at the boundary before the
  step; the step keeps running and the owner parks at the next boundary until resume. After a crash
  the step runs again from the earlier boundary. Use it only for steps declared safe to re-run, such
  as idempotent seeds, pure or judge-based verification, and agent activations, which continue from
  the agent's own boundaries.

Time an episode spends parked for a checkpoint does not count against its deadline.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import JsonValue


StepMode = Literal["wait", "replay"]
Boundary = dict[str, JsonValue]


# A resources server reports on its /seed_session reply whether its /verify may be replayed. Every
# episode seeds before it verifies, so the caller learns the mode without another request.
CHECKPOINT_VERIFY_HEADER = "x-ng-checkpoint-verify"


def seed_verify_mode(headers: Optional[Mapping[str, str]]) -> StepMode:
    """The mode for a resources server's ``/verify``, from its seed reply; ``wait`` if it reported none."""
    return "replay" if (headers or {}).get(CHECKPOINT_VERIFY_HEADER) == "replay" else "wait"


@dataclass
class _Episode:
    task: Optional[asyncio.Task]
    deadline: Optional[asyncio.Timeout]
    continuation: Optional[Boundary]
    state: Literal["running", "wait_step", "replay_step", "parked"] = "running"
    boundary: Optional[Boundary] = None
    resume: asyncio.Event = field(default_factory=asyncio.Event)
    suspended_remaining: Optional[float] = None

    def suspend_deadline(self) -> None:
        if self.deadline is None or self.deadline.when() is None or self.suspended_remaining is not None:
            return
        self.suspended_remaining = self.deadline.when() - asyncio.get_running_loop().time()
        self.deadline.reschedule(None)

    def resume_deadline(self) -> None:
        if self.suspended_remaining is None or self.deadline is None:
            return
        self.deadline.reschedule(asyncio.get_running_loop().time() + self.suspended_remaining)
        self.suspended_remaining = None


class EpisodeSteps:
    """Track the boundaries and in-flight steps of the episodes one owner runs."""

    def __init__(self, notify: Callable[[], Awaitable[None]]) -> None:
        self._notify = notify
        self._episodes: dict[str, _Episode] = {}
        self.closed = False

    def begin(
        self,
        key: str,
        *,
        continuation: Optional[Boundary] = None,
        deadline: Optional[asyncio.Timeout] = None,
    ) -> None:
        if key in self._episodes:
            raise ValueError(f"episode {key} is already running")
        self._episodes[key] = _Episode(task=asyncio.current_task(), deadline=deadline, continuation=continuation)

    def continuation(self, key: str) -> Optional[Boundary]:
        """Return the restored boundary this episode continues from, once."""
        episode = self._episodes[key]
        continuation, episode.continuation = episode.continuation, None
        return continuation

    def finishing(self, key: str) -> None:
        """The episode's protocol is over and its owner is releasing what it created.

        A retire from here on stops tracking the episode but does not cancel its task: the owner's cleanup,
        which closes the episode's sessions, must run to completion.
        """
        episode = self._episodes.get(key)
        if episode is not None:
            episode.task = None

    async def end(self, key: str) -> None:
        self._episodes.pop(key, None)
        await self._notify()

    async def boundary(self, key: str, state: Boundary) -> None:
        """Record a completed step; park here until resume if a checkpoint is open."""
        episode = self._episodes[key]
        episode.boundary = state
        if not self.closed:
            return
        episode.state = "parked"
        episode.resume.clear()
        episode.suspend_deadline()
        await self._notify()
        while True:
            await episode.resume.wait()
            if self._episodes.get(key) is not episode:
                raise asyncio.CancelledError(f"episode {key} was retired while parked")
            if not self.closed:
                break
            # A new checkpoint closed admission before this episode woke: it stays parked for that one too.
            episode.resume.clear()
        episode.state = "running"

    @asynccontextmanager
    async def step(self, key: str, mode: StepMode) -> AsyncIterator[None]:
        """Run one step in ``mode``; see the module docstring."""
        episode = self._episodes[key]
        episode.state = "wait_step" if mode == "wait" else "replay_step"
        if mode == "replay" and self.closed:
            episode.suspend_deadline()
        await self._notify()
        try:
            yield
        finally:
            if self._episodes.get(key) is episode:
                episode.state = "running"
                episode.resume_deadline()
            await self._notify()

    def close(self) -> None:
        self.closed = True
        for episode in self._episodes.values():
            if episode.state in ("parked", "replay_step"):
                episode.suspend_deadline()

    def open(self) -> None:
        self.closed = False
        for episode in self._episodes.values():
            episode.resume_deadline()
            episode.resume.set()

    def blockers(self) -> list[str]:
        """Episodes between boundaries or inside a wait step: a checkpoint must wait for them."""
        return sorted(key for key, episode in self._episodes.items() if episode.state in ("running", "wait_step"))

    def exported(self) -> dict[str, Boundary]:
        """The latest boundary of every episode that is parked or inside a replay step."""
        return {
            key: episode.boundary
            for key, episode in self._episodes.items()
            if episode.state in ("parked", "replay_step") and episode.boundary is not None
        }

    def keys(self) -> list[str]:
        return list(self._episodes)

    async def retire(self, key: str) -> None:
        episode = self._episodes.pop(key, None)
        if episode is None:
            return
        episode.resume.set()
        if episode.task is not None and episode.task is not asyncio.current_task():
            episode.task.cancel()
        await self._notify()
