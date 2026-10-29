# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one supervised harness launch until it finishes, parking it in place at every checkpoint."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import JsonValue

from nemo_gym.agent_utils.sandbox_session import HarnessPauseResult, SandboxSession
from nemo_gym.agent_utils.supervisor_client import CleanupReceipt


LOG = logging.getLogger(__name__)
# A continue that fails this many times falls back to stopping the harness.
CONTINUE_ATTEMPTS = 3


class ParkingActivation(Protocol):
    """The agent checkpoint participant's handle for one running activation (``nemo_gym._checkpoint.agent``)."""

    @property
    def park_requested(self) -> bool: ...

    async def wait_park_requested(self) -> None:
        """Return once a checkpoint asks this activation to park; raise once its session is retired or closed."""

    async def boundary(self, snapshot: Callable[[], dict[str, JsonValue]]) -> None:
        """Park here until the checkpoint resumes; return at once when none asks."""

    def queued(self) -> AbstractAsyncContextManager[None]: ...


@dataclass(frozen=True)
class Finished[T]:
    """The launch ended on its own, possibly after pauses in place; ``result`` is what ``launch`` returned."""

    result: T


@dataclass(frozen=True)
class Stopped:
    """A checkpoint stopped the launch because pausing or continuing it failed, and the activation parked.

    The harness did not finish: relaunch it, resuming its saved conversation. ``receipt`` is the supervisor's cleanup
    receipt; ``error`` is what the stopped launch raised while it collected its artifacts, if anything.
    """

    receipt: CleanupReceipt | None
    error: BaseException | None = None


async def run_until_parked[T](
    session: SandboxSession[Any],
    activation: ParkingActivation,
    *,
    launch: Callable[[], Awaitable[T]],
    on_parked: Callable[[], dict[str, JsonValue]],
    stop_timeout: float,
) -> Finished[T] | Stopped:
    """Run ``launch`` (the adapter's ``session.execute(...)``) and park its harness at every checkpoint.

    When a checkpoint asks the activation to park, the harness is paused in place and the activation parks at
    ``on_parked``, the adapter's boundary snapshot, until the checkpoint resumes; then the harness continues as if
    its model call were slow, and the launch keeps running for later checkpoints. If the pause is not confirmed, or
    the continue fails ``CONTINUE_ATTEMPTS`` times, the harness is stopped (TERM, so it can save its state), the
    activation parks after a pause failure, and ``Stopped`` tells the adapter to relaunch. A launch that ends on
    its own, also as a park arrives, returns ``Finished``. An error of ``launch`` propagates.

    Cancelled, for example when its session is retired or closed, it stops the harness, shielded and bounded by
    ``stop_timeout``, before it re-raises.
    """
    launched = asyncio.ensure_future(launch())
    parking: asyncio.Task[None] | None = None
    try:
        while True:
            parking = asyncio.ensure_future(activation.wait_park_requested())
            await asyncio.wait({launched, parking}, return_when=asyncio.FIRST_COMPLETED)
            if launched.done():
                parking.cancel()
                return Finished(launched.result())
            # A retired or closed session raises here; the handler below stops the harness.
            parking.result()
            parking = None
            try:
                paused = await session.pause_harness()
            except Exception as error:
                # A request that failed in transport leaves the pause unknown: stopping is the safe fallback.
                paused = HarnessPauseResult(status="unconfirmed", error=f"pause request failed: {error!r}")
            if paused.status == "finished":
                return Finished(await launched)
            if paused.confirmed:
                await activation.boundary(on_parked)
                if await _continue(session):
                    continue
                LOG.warning("%s harness did not continue after a checkpoint; stopping it", session.harness)
                return await _stop(session, launched, stop_timeout=stop_timeout)
            LOG.warning("%s harness pause was not confirmed (%s); stopping it", session.harness, paused.error)
            stopped = await _stop(session, launched, stop_timeout=stop_timeout)
            if isinstance(stopped, Stopped):
                await activation.boundary(on_parked)
            return stopped
    except BaseException:
        if not launched.done():
            # Cancelling the launch stops the harness and collects its artifacts (SandboxSession.execute).
            launched.cancel()
            waiter = asyncio.ensure_future(asyncio.wait({launched}, timeout=stop_timeout))
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                pass
        raise
    finally:
        if parking is not None and not parking.done():
            parking.cancel()


async def _continue(session: SandboxSession[Any]) -> bool:
    for attempt in range(1, CONTINUE_ATTEMPTS + 1):
        try:
            result = await session.continue_harness()
        except Exception as error:
            result = HarnessPauseResult(status="unconfirmed", error=f"continue request failed: {error!r}")
        if result.status in ("confirmed", "finished"):
            return True
        LOG.warning(
            "%s harness continue attempt %d/%d failed: %s", session.harness, attempt, CONTINUE_ATTEMPTS, result.error
        )
    return False


async def _stop[T](
    session: SandboxSession[Any], launched: asyncio.Future[T], *, stop_timeout: float
) -> Finished[T] | Stopped:
    """Stop the harness and wait for the launch to collect; a harness that ended on its own first is ``Finished``."""
    await session.stop_harness(timeout=stop_timeout)
    try:
        result = await launched
    except Exception as error:
        receipt = session.cleanup
        if receipt is not None and not receipt.get("stopped", False):
            raise
        return Stopped(receipt=receipt, error=error)
    receipt = session.cleanup
    if receipt is not None and not receipt.get("stopped", False):
        return Finished(result)
    return Stopped(receipt=receipt)
