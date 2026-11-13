# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded retries for judge requests without changing their contents."""

import asyncio

from aiohttp import ClientConnectionError

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NgFailureEvent
from nemo_gym.verification import call_judge, is_judge_failure


RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


async def request_judgment(
    server_client: ServerClient,
    *,
    server_name: str,
    params: NeMoGymResponseCreateParamsNonStreaming,
    attempts: int = 3,
) -> NeMoGymResponse:
    """Retry transient HTTP and connection failures, waiting five then ten seconds."""
    if attempts < 1:
        raise ValueError("Judge transport attempts must be positive")
    for attempt in range(attempts):
        try:
            return await call_judge(
                server_client,
                server_name=server_name,
                url_path="/v1/responses",
                json=params,
                response_model=NeMoGymResponse,
                max_attempts=1,
            )
        except NgFailureEvent as exc:
            if not is_judge_failure(exc):
                raise
            retryable = (
                exc.http_status in RETRYABLE_STATUSES
                if exc.http_status is not None
                else isinstance(exc.__cause__, (TimeoutError, ConnectionError, ClientConnectionError))
            )
            if not retryable or attempt + 1 == attempts:
                raise
        await asyncio.sleep(min(60, 5 * 2**attempt))
    raise RuntimeError("Judge transport completed without a response")
