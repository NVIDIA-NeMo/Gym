# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated agent-side prototype; not integrated into live Gym."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

import anyio


logger = logging.getLogger(__name__)
T = TypeVar("T")


class Failure(Protocol):
    reason: str


class Manifest(Protocol):
    rollout_id: str
    failures: Sequence[Failure]


@dataclass(frozen=True)
class CaptureAbortAcknowledgement:
    rollout_id: str
    reason: str
    sandbox_stopped: Literal[True] = True
    verification_performed: Literal[False] = False


async def run_until_capture_failure(
    response: Awaitable[T],
    read_manifest: Callable[[], Awaitable[Manifest]],
    stop_sandbox: Callable[[], Awaitable[None]],
    *,
    rollout_id: str,
    uncommitted_reason: str,
    poll_interval_s: float = 5.0,
    cleanup_timeout_s: float = 60.0,
) -> T | CaptureAbortAcknowledgement:
    """Caller retains ordinary-error/disconnect cleanup responsibility.

    read_manifest must use the bounded authenticated control client.
    A cleanup error must propagate; caller must never translate it to an ack
    or retry the same AsyncSandbox.stop object and assume a no-op is success.
    """
    task = asyncio.ensure_future(response)
    try:
        if not rollout_id or poll_interval_s <= 0 or cleanup_timeout_s <= 0:
            raise ValueError("identity and positive polling/cleanup intervals required")
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_interval_s)
            if done:
                return task.result()
            try:
                manifest = await read_manifest()
            except (RuntimeError, OSError, ValueError):
                logger.warning("Capture manifest unavailable; retaining active response")
                continue
            if task.done():
                return task.result()
            if manifest.rollout_id != rollout_id:
                logger.warning("Capture manifest identity mismatch; retaining active response")
                continue
            reason = next((f.reason for f in manifest.failures if f.reason and f.reason != uncommitted_reason), None)
            if reason is None:
                continue
            if not task.cancel():
                return task.result()
            # Stop before draining: a response task waiting on sandbox I/O
            # may only settle after the actual provider has stopped it.
            with anyio.fail_after(cleanup_timeout_s, shield=True):
                await stop_sandbox()
                await asyncio.gather(task, return_exceptions=True)
            return CaptureAbortAcknowledgement(rollout_id, reason)
    finally:
        if not task.done():
            task.cancel()
        with anyio.move_on_after(cleanup_timeout_s, shield=True):
            await asyncio.gather(task, return_exceptions=True)
