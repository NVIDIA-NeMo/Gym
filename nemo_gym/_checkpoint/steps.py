# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Episode boundaries and in-flight step modes, shared by every owner of an episode protocol.

An episode is a sequence of steps.
After each completed step its owner records a boundary: the next step and the results that step needs.
A checkpoint can always be taken at a boundary.
A step that is still running when a checkpoint closes admission has one of two modes:

- ``wait`` (the default): the checkpoint waits for the step.
  The owner then records the boundary after it, with the step's result, and parks there.
  Use it for any step that is unsafe to run twice.
- ``replay``: the checkpoint does not wait.
  The episode counts as parked at the boundary before the step;
  the step keeps running and the owner parks at the next boundary until resume.
  After a crash the step runs again from the earlier boundary.
  Use it only for steps declared safe to re-run, such as idempotent seeds, pure or judge-based verification,
  and agent activations, which continue from the agent's own boundaries.

Time an episode spends parked for a checkpoint does not count against its deadline.

Every live episode is checkpointable: one that has not recorded a boundary yet,
such as an episode inside its first replay step,
is exported with no boundary and starts over from its input after a crash;
a restored episode starts at the boundary it was restored from.

An episode that uses a server which cannot capture its part, such as a restart-only agent, is a restart:
it never blocks a checkpoint, never parks, and is never exported.
The owner reports it, so the controller leaves it out of the commit and, after a crash,
starts the rollout over from its input.
If nothing crashes, it just keeps running.
An episode becomes a restart only while no checkpoint is open,
so the episodes a checkpoint exports and the restarts it reports never change during it.
"""

import asyncio
import heapq
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import JsonValue


StepMode = Literal["wait", "replay"]
Boundary = dict[str, JsonValue]
_State = Literal["running", "wait_step", "replay_step", "parked"]
# States in which a checkpoint must wait for the episode.
_BLOCKING: frozenset[str] = frozenset({"running", "wait_step"})


# A resources server reports on its /seed_session reply whether its /verify may be replayed.
# Every episode seeds before it verifies, so the caller learns the mode without another request.
CHECKPOINT_VERIFY_HEADER = "x-ng-checkpoint-verify"


def seed_verify_mode(headers: Optional[Mapping[str, str]]) -> StepMode:
    """The mode for a resources server's ``/verify``, from its seed reply; ``wait`` if it reported none."""
    return "replay" if (headers or {}).get(CHECKPOINT_VERIFY_HEADER) == "replay" else "wait"


# A server that cannot capture a session, such as a restart-only agent or resources server,
# says so on its seed reply, so the episode that seeded it becomes a restart.
CHECKPOINT_RESTART_HEADER = "x-ng-checkpoint-restart"


def seed_restarts(headers: Optional[Mapping[str, str]]) -> bool:
    """Whether a seed reply says its session cannot be captured, so its episode must start over after a crash."""
    return (headers or {}).get(CHECKPOINT_RESTART_HEADER) == "1"


@dataclass
class _Episode:
    task: Optional[asyncio.Task]
    deadline: Optional[asyncio.Timeout]
    continuation: Optional[Boundary]
    state: _State = "running"
    boundary: Optional[Boundary] = None
    # Set by retire: the episode is being stopped, and must not continue past a boundary.
    retired: bool = False
    # A server this episode uses cannot capture its part: the episode is never exported and starts over after a crash.
    restart: bool = False
    resume: asyncio.Event = field(default_factory=asyncio.Event)
    suspended_remaining: Optional[float] = None
    # Set when the owner's final cleanup starts: a retire then waits for the episode without cancelling it.
    finishing: bool = False

    def suspend_deadline(self) -> None:
        if self.deadline is None or self.deadline.when() is None or self.suspended_remaining is not None:
            return
        self.suspended_remaining = self.deadline.when() - asyncio.get_running_loop().time()
        self.deadline.reschedule(None)

    def resume_deadline(self) -> None:
        if self.suspended_remaining is None or self.deadline is None:
            return
        remaining, self.suspended_remaining = self.suspended_remaining, None
        try:
            self.deadline.reschedule(asyncio.get_running_loop().time() + remaining)
        except RuntimeError:
            # The episode's deadline scope has already exited, for example while a failing step unwinds;
            # there is nothing left to time, and raising here would hide the original error.
            pass


def _blocks(episode: _Episode) -> bool:
    return episode.retired or (not episode.restart and episode.state in _BLOCKING)


class EpisodeSteps:
    """Track the boundaries and in-flight steps of the episodes one owner runs."""

    def __init__(self, notify: Callable[[], Awaitable[None]]) -> None:
        self._notify = notify
        self._episodes: dict[str, _Episode] = {}
        self._blocking = 0
        self.closed = False
        self._open = asyncio.Event()
        self._open.set()

    def _set_state(self, episode: _Episode, state: _State) -> None:
        blocked = _blocks(episode)
        episode.state = state
        self._blocking += _blocks(episode) - blocked

    def _forget(self, key: str, episode: _Episode) -> None:
        if self._episodes.get(key) is episode:
            del self._episodes[key]
            self._blocking -= _blocks(episode)

    async def _changed(self) -> None:
        # Only a prepare waits for readiness, and only while admission is closed.
        if self.closed:
            await self._notify()

    def begin(
        self,
        key: str,
        *,
        continuation: Optional[Boundary] = None,
        deadline: Optional[asyncio.Timeout] = None,
        restart: bool = False,
    ) -> None:
        if key in self._episodes:
            raise ValueError(f"episode {key} is already running")
        # A restored episode is at the boundary it was restored from until it records a new one.
        episode = _Episode(
            task=asyncio.current_task(),
            deadline=deadline,
            continuation=continuation,
            boundary=continuation,
            restart=restart,
        )
        self._episodes[key] = episode
        self._blocking += _blocks(episode)

    async def mark_restart(self, key: str) -> None:
        """Make this episode a restart from now on, once no checkpoint is open; see the module docstring."""
        while self.closed:
            await self._open.wait()
        episode = self._episodes.get(key)
        if episode is None:
            return
        blocked = _blocks(episode)
        episode.restart = True
        self._blocking += _blocks(episode) - blocked
        await self._changed()

    def continuation(self, key: str) -> Optional[Boundary]:
        """Return the restored boundary this episode continues from, once."""
        episode = self._episodes[key]
        continuation, episode.continuation = episode.continuation, None
        return continuation

    def finishing(self, key: str) -> None:
        """The episode's protocol is over and its owner is releasing what it created.

        A retire from here on waits for the episode but does not cancel its task: the owner's cleanup,
        which closes the episode's sessions, must run to completion.
        """
        episode = self._episodes.get(key)
        if episode is not None:
            episode.finishing = True

    async def end(self, key: str) -> None:
        episode = self._episodes.get(key)
        if episode is not None:
            self._forget(key, episode)
        await self._changed()

    async def boundary(self, key: str, state: Boundary) -> None:
        """Record a completed step; park here until resume if a checkpoint is open."""
        episode = self._episodes[key]
        episode.boundary = state
        if not self.closed or episode.restart:
            if episode.state != "running":
                self._set_state(episode, "running")
            return
        self._set_state(episode, "parked")
        episode.resume.clear()
        episode.suspend_deadline()
        await self._notify()
        while True:
            await episode.resume.wait()
            if episode.retired:
                raise asyncio.CancelledError(f"episode {key} was retired while parked")
            if not self.closed:
                break
            # A new checkpoint closed admission before this episode woke: it stays parked for that one too.
            episode.resume.clear()
        self._set_state(episode, "running")

    @asynccontextmanager
    async def step(self, key: str, mode: StepMode) -> AsyncIterator[None]:
        """Run one step in ``mode``; see the module docstring."""
        episode = self._episodes[key]
        self._set_state(episode, "wait_step" if mode == "wait" else "replay_step")
        if mode == "replay" and self.closed:
            episode.suspend_deadline()
        await self._changed()
        completed = False
        try:
            yield
            completed = True
        finally:
            if self._episodes.get(key) is episode:
                if mode == "replay" and self.closed and completed:
                    # The checkpoint already counts this episode at the boundary before the step; it stays there,
                    # not a blocker, until its next boundary parks it with the step's result.
                    pass
                elif mode == "replay" and self.closed:
                    # The step raised, so the episode is failing, and its cleanup may close its sessions before the
                    # commit.
                    # Continuing it from the boundary before the step would need those sessions, so it is exported
                    # with no boundary and starts over from its input after a crash.
                    # It stays out of the blockers, so one failure does not fail the whole checkpoint.
                    episode.boundary = None
                else:
                    # A step that raised has no next boundary: the episode is failing, so it blocks until it ends.
                    self._set_state(episode, "running")
                    episode.resume_deadline()
            await self._changed()

    def close(self) -> None:
        self.closed = True
        self._open.clear()
        for episode in self._episodes.values():
            if episode.state in ("parked", "replay_step"):
                episode.suspend_deadline()

    def open(self) -> None:
        self.closed = False
        self._open.set()
        for episode in self._episodes.values():
            episode.resume_deadline()
            episode.resume.set()

    def blocker_count(self) -> int:
        """How many episodes are between boundaries, inside a wait step, or retired but not yet stopped:
        a checkpoint must wait for them."""
        return self._blocking

    def blockers(self, limit: int) -> list[str]:
        """The first ``limit`` blocking episodes, by key; empty without a scan when nothing blocks."""
        if not self._blocking:
            return []
        return heapq.nsmallest(limit, (key for key, episode in self._episodes.items() if _blocks(episode)))

    def exported(self) -> dict[str, Optional[Boundary]]:
        """The latest boundary of every episode that is parked or inside a replay step, and not retired.

        ``None`` for an episode that has not recorded a boundary yet: it starts over from its input.
        """
        return {
            key: episode.boundary
            for key, episode in self._episodes.items()
            if episode.state in ("parked", "replay_step") and not episode.retired and not episode.restart
        }

    def restarts(self) -> list[str]:
        """The episodes that start over after a crash, by key; the commit must leave them out."""
        return sorted(key for key, episode in self._episodes.items() if episode.restart and not episode.retired)

    def keys(self) -> list[str]:
        return list(self._episodes)

    async def retire(self, key: str) -> None:
        """Stop the episode: cancel it unless its final cleanup already started, then wait until it has ended.

        The episode stays tracked until then, so if this wait is cut short,
        a later retire finds it and waits again, and a duplicate start of the same attempt is refused.
        Until then it blocks a checkpoint whatever its state,
        so a prepare reports an episode that is still being stopped instead of exporting it.
        """
        episode = self._episodes.get(key)
        if episode is None:
            return
        first = not episode.retired
        blocked = _blocks(episode)
        episode.retired = True
        self._blocking += _blocks(episode) - blocked
        episode.resume.set()
        if episode.task is not None and episode.task is not asyncio.current_task():
            # Cancel once, and never once the final cleanup started: cancelling would interrupt that cleanup.
            if first and not episode.finishing:
                episode.task.cancel()
            await asyncio.wait([episode.task])
        self._forget(key, episode)
        await self._changed()
