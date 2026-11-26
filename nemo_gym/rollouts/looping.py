# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-owned process execution for the severe text looping assessment."""

import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Mapping

from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.metrics.looping import (
    LoopingAssessment,
    LoopingLimits,
    assess_text,
    collect_response_text,
)
from nemo_gym.metrics.looping import LoopingReason as LoopingReason


logger = logging.getLogger(__name__)


class LoopingAssessor:
    """One persistent spawned pool with at most max_workers submitted jobs.

    Reserve capacity before preparing text. Caller timeout/cancellation leaves that slot
    reserved until the underlying future finishes; no inline CPU fallback is used.
    Work/size limits bound normal jobs; this adapter does not supervise individual OS processes.
    Awaiting a label remains a dependency of the caller's rollout completion.
    """

    def __init__(
        self, max_workers: int = 1, *, limits: LoopingLimits = LoopingLimits(), timeout_s: float = 30
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be greater than 0")
        self._limits = limits
        self._timeout_s = timeout_s
        self._max_workers = max_workers
        self._capacity = asyncio.Semaphore(max_workers)
        self._pool: ProcessPoolExecutor | None = None
        self._closed = False

    async def assess(self, response: NeMoGymResponse | Mapping[str, object] | None) -> LoopingAssessment:
        """Assess returned episode text from a Gym response or its serialized mapping."""
        try:
            async with asyncio.timeout(self._timeout_s):
                await self._capacity.acquire()
                submitted = False
                try:
                    if self._closed:
                        return LoopingAssessment()
                    chunks = collect_response_text(response, self._limits)
                    if chunks is None:
                        return LoopingAssessment()
                    if self._pool is None:
                        self._pool = ProcessPoolExecutor(
                            max_workers=self._max_workers, mp_context=multiprocessing.get_context("spawn")
                        )
                    future = self._pool.submit(assess_text, chunks, self._limits)
                    loop = asyncio.get_running_loop()

                    def release(_):
                        if not loop.is_closed():
                            loop.call_soon_threadsafe(self._capacity.release)

                    future.add_done_callback(release)
                    submitted = True
                    return await asyncio.shield(asyncio.wrap_future(future))
                finally:
                    if not submitted:
                        self._capacity.release()
        except Exception:
            logger.warning("Severe text looping assessment unavailable.", exc_info=True)
            return LoopingAssessment()

    def close(self) -> None:
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
