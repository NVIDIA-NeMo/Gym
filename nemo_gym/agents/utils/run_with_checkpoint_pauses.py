# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one supervised harness launch to its end, pausing the harness in place at every checkpoint."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import JsonValue

from nemo_gym.agents.utils.sandbox_session import HarnessPauseResult, SandboxSession
from nemo_gym.agents.utils.supervisor_client import CleanupReceipt


LOG = logging.getLogger(__name__)
# A pause request that fails in transport is sent again this many times in all, then the harness is stopped.
PAUSE_ATTEMPTS = 4
# A continue that fails this many times falls back to stopping the harness.
CONTINUE_ATTEMPTS = 3
# Wait before the second attempt of a pause or a continue; doubled before each later one.
RETRY_DELAY_S = 1.0


class ParkingActivation(Protocol):
    """The agent checkpoint participant's handle for one running activation.

    Implemented by ``nemo_gym.server_runtime._checkpoint.agent.Activation``.
    """

    @property
    def park_requested(self) -> bool: ...

    async def wait_park_requested(self) -> None:
        """Return once a checkpoint asks this activation to park; raise once its session is retired or closed."""

    async def boundary(self, snapshot: Callable[[], dict[str, JsonValue]]) -> None:
        """Park here until the checkpoint releases its participants; return at once when none asks."""

    def queued(self) -> AbstractAsyncContextManager[None]: ...


@dataclass(frozen=True)
class LaunchFinished[T]:
    """The launch ended on its own, possibly after pauses in place; ``result`` is what ``launch`` returned."""

    result: T


@dataclass(frozen=True)
class LaunchStopped:
    """A checkpoint stopped the launch because pausing or continuing the harness failed, and the activation parked.

    The harness did not finish: launch it again from the state it saved. ``receipt`` is the supervisor's cleanup
    receipt; ``error`` is what the stopped launch raised while it collected its artifacts, if anything.
    """

    receipt: CleanupReceipt | None
    error: BaseException | None = None


async def run_with_checkpoint_pauses[T](
    session: SandboxSession[Any],
    activation: ParkingActivation,
    *,
    launch: Callable[[], Awaitable[T]],
    on_parked: Callable[[], dict[str, JsonValue]],
    stop_timeout: float,
) -> LaunchFinished[T] | LaunchStopped:
    """Run ``launch`` (the adapter's ``session.execute(...)``), pausing its harness in place at every checkpoint.

    When a checkpoint asks the activation to park, the harness is paused in place and the activation parks at
    ``on_parked``, the adapter's boundary snapshot, until the checkpoint releases it; then the harness is continued
    and the launch keeps running for later checkpoints. A pause stops the harness's processes, not its clocks: a
    tool that runs across the pause sees the paused time, so its own timeout can fire when the harness continues.
    Model calls wait at the model server and are not affected.

    A pause or continue request that fails in transport is repeated after a delay, with the same sequence number,
    so the supervisor serves it once. If the pause is still not confirmed, or the continue is refused by the
    supervisor or fails ``CONTINUE_ATTEMPTS`` times, the harness is stopped: the supervisor sends TERM and, after the cleanup grace
    period, KILL to what is left, so a harness that ignores TERM costs the whole grace period. The activation
    parks after a pause failure, and ``LaunchStopped`` tells the adapter to launch again. A supervisor that died
    during this stop, leaving a receipt the provider had to prove (``receipt_missing``), counts as stopped too. A
    launch that ends on its own, also as a park arrives, returns ``LaunchFinished``. An error of ``launch``
    propagates, and a launch that is still staging and does not end when cancelled raises: the activation must not
    park while anything of it can still write to the sandbox. Nor may it fail meanwhile, since a failed activation
    no longer holds up the checkpoint: any error is raised only once the launch has ended and its staging no longer
    runs, however long that takes. Until then the checkpoint waits for the session, up to its own deadline.

    Cancelled, for example when its session is retired or closed, it stops the harness, shielded and bounded by
    ``stop_timeout``, before it re-raises.
    """
    launched = asyncio.ensure_future(launch())
    parking: asyncio.Task[None] | None = None
    try:
        while True:
            parking = asyncio.ensure_future(activation.wait_park_requested())
            # A park request can fail, with its session closing, while this call is being cancelled: it is then
            # done, so nothing below cancels or reads it.
            parking.add_done_callback(_retrieve_exception)
            await asyncio.wait({launched, parking}, return_when=asyncio.FIRST_COMPLETED)
            if launched.done():
                parking.cancel()
                return LaunchFinished(launched.result())
            # A retired or closed session raises here; the handler below stops the harness.
            parking.result()
            parking = None
            paused = await _pause(session)
            if paused.status == "finished":
                return LaunchFinished(await launched)
            if paused.confirmed:
                await activation.boundary(on_parked)
                if await _continue(session):
                    continue
                LOG.warning("%s harness did not continue after a checkpoint; stopping it", session.harness)
                return await _stop(session, launched, stop_timeout=stop_timeout)
            LOG.warning("%s harness pause was not confirmed (%s); stopping it", session.harness, paused.error)
            stopped = await _stop(session, launched, stop_timeout=stop_timeout)
            if isinstance(stopped, LaunchStopped):
                await activation.boundary(on_parked)
            return stopped
    except BaseException as error:
        if not launched.done():
            # Cancelling the launch stops the harness and collects its artifacts (SandboxSession.execute).
            launched.cancel()
            # A launch that outlives the wait is left behind; whatever it raises later is not this caller's.
            launched.add_done_callback(_retrieve_exception)
            waiter = asyncio.ensure_future(asyncio.wait({launched}, timeout=stop_timeout))
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                pass
        if isinstance(error, Exception):
            # The activation is about to fail, and its session then no longer holds up a checkpoint. A staging
            # that did not end when cancelled may still be writing to the sandbox: fail only once it has ended.
            await _wait_for_staging(session, launched, stop_timeout=stop_timeout)
        raise
    finally:
        if parking is not None and not parking.done():
            parking.cancel()


async def _wait_for_staging(
    session: SandboxSession[Any], launched: asyncio.Future[Any], *, stop_timeout: float
) -> None:
    """Return once the launch has ended and its staging no longer runs; warn every ``stop_timeout`` until then.

    Not bounded: while the caller waits here its activation is unfinished, so a checkpoint waits for the session
    and fails at its own deadline instead of taking a snapshot beside the staging's writes. A retire or a close of
    the session cancels the wait.
    """
    while not launched.done() or session.staging:
        LOG.warning(
            "%s launch is still staging after it was cancelled; its activation holds up checkpoints until it ends",
            session.harness,
        )
        if not launched.done():
            await asyncio.wait({launched}, timeout=stop_timeout)
        else:
            await session.wait_staged(timeout=stop_timeout)


def _retrieve_exception(task: asyncio.Future[Any]) -> None:
    """Read a finished task's exception, so asyncio does not log it as never retrieved."""
    if not task.cancelled():
        task.exception()


async def _retry_delay(attempt: int) -> None:
    await asyncio.sleep(RETRY_DELAY_S * 2 ** (attempt - 1))


async def _pause(session: SandboxSession[Any]) -> HarnessPauseResult:
    """Pause the harness, repeating a request that failed in transport; an answer of the supervisor is final.

    A request that timed out is not repeated: the provider had no capacity for it, or the supervisor did not
    answer within the pause timeout, and neither clears in a few seconds.
    """
    for attempt in range(1, PAUSE_ATTEMPTS + 1):
        try:
            return await session.pause_harness()
        except Exception as error:
            # A request that failed in transport leaves the pause unknown: stopping is the safe fallback.
            if isinstance(error, TimeoutError) or attempt == PAUSE_ATTEMPTS:
                return HarnessPauseResult(status="unconfirmed", error=f"pause request failed: {error!r}")
            LOG.warning("%s harness pause attempt %d/%d failed: %r", session.harness, attempt, PAUSE_ATTEMPTS, error)
        await _retry_delay(attempt)
    raise AssertionError("unreachable")


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
        if result.acknowledged:
            # The supervisor tried and refused. The same request would read that same answer again.
            return False
        if attempt < CONTINUE_ATTEMPTS:
            await _retry_delay(attempt)
    return False


def _ended_by_stop(receipt: CleanupReceipt | None) -> bool:
    """Whether a launch this module stopped ended by that stop, not on its own.

    The supervisor records the stop it served. A supervisor that died while stopping left no receipt; the one the
    provider's proof stands in for (``receipt_missing``) has no ``stopped``, and the harness did not finish either.
    """
    return receipt is None or receipt.get("stopped", False) or receipt.get("receipt_missing", False)


async def _stop[T](
    session: SandboxSession[Any], launched: asyncio.Future[T], *, stop_timeout: float
) -> LaunchFinished[T] | LaunchStopped:
    """Stop the harness and wait for the launch to collect.

    A harness that ended on its own first is ``LaunchFinished``. A launch that had not started its harness is cancelled, and an error of a launch no stop ended propagates.
    """
    await session.stop_harness(timeout=stop_timeout)
    if not session.launch_started and not launched.done():
        # The launch is still staging, so the stop had no harness to end. Left alone it would run the harness
        # unpaused, or wait for the continue of a pause request that did reach the sandbox. Cancelling ends it
        # before it launches (SandboxSession.execute).
        launched.cancel()
        await asyncio.wait({launched}, timeout=stop_timeout)
        if not launched.done():
            # Staging may still be writing to the sandbox: the caller must not park beside it.
            raise RuntimeError(
                f"{session.harness} launch did not end within {stop_timeout:g}s of being cancelled while staging"
            )
        if session.staging:
            # The launch gave up waiting for its staging (SandboxSession.execute), which is still running.
            raise RuntimeError(f"{session.harness} launch was cancelled while staging, and its staging has not ended")
        if launched.cancelled():
            return LaunchStopped(receipt=None)
    try:
        result = await launched
    except Exception as error:
        receipt = session.cleanup
        # Only a receipt of the stop makes this the stopped launch's error. Without one, for example when the
        # sandbox died, it is the launch's own failure.
        if receipt is None or not _ended_by_stop(receipt):
            raise
        return LaunchStopped(receipt=receipt, error=error)
    receipt = session.cleanup
    if not _ended_by_stop(receipt):
        return LaunchFinished(result)
    return LaunchStopped(receipt=receipt)
