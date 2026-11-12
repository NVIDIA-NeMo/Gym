# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise HTTP classification, connection recovery, and bounded retry delays."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import FailureBoundary, NeMoGymResponseCreateParamsNonStreaming, NgFailureEvent
from resources_servers.vision2code import judge_transport


def failure(status: int | None) -> NgFailureEvent:
    return NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind="judge_failed",
        failure_reason="Judge request failed",
        http_status=status,
    )


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
async def test_transient_status_retries_unchanged_request(monkeypatch, status):
    expected = object()
    call = AsyncMock(side_effect=[failure(status), failure(status), expected])
    sleep = AsyncMock()
    monkeypatch.setattr(judge_transport, "call_judge", call)
    monkeypatch.setattr(judge_transport.asyncio, "sleep", sleep)
    params = NeMoGymResponseCreateParamsNonStreaming(input=[], temperature=0.7)
    result = await judge_transport.request_judgment(MagicMock(spec=ServerClient), server_name="judge", params=params)
    assert result is expected
    assert [args.args[0] for args in sleep.await_args_list] == [5, 10]
    assert call.await_count == 3
    assert all(args.kwargs["json"] is params and args.kwargs["max_attempts"] == 1 for args in call.await_args_list)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 501, 505])
async def test_permanent_status_does_not_retry(monkeypatch, status):
    error = failure(status)
    call, sleep = AsyncMock(side_effect=error), AsyncMock()
    monkeypatch.setattr(judge_transport, "call_judge", call)
    monkeypatch.setattr(judge_transport.asyncio, "sleep", sleep)
    with pytest.raises(NgFailureEvent) as caught:
        await judge_transport.request_judgment(
            MagicMock(spec=ServerClient), server_name="judge", params=NeMoGymResponseCreateParamsNonStreaming(input=[])
        )
    assert caught.value is error
    assert call.await_count == 1
    sleep.assert_not_awaited()


async def test_connection_failure_exhausts_budget(monkeypatch):
    error = failure(None)
    error.__cause__ = TimeoutError("Judge timed out")
    call, sleep = AsyncMock(side_effect=error), AsyncMock()
    monkeypatch.setattr(judge_transport, "call_judge", call)
    monkeypatch.setattr(judge_transport.asyncio, "sleep", sleep)
    with pytest.raises(NgFailureEvent):
        await judge_transport.request_judgment(
            MagicMock(spec=ServerClient), server_name="judge", params=NeMoGymResponseCreateParamsNonStreaming(input=[])
        )
    assert call.await_count == 3
    assert [args.args[0] for args in sleep.await_args_list] == [5, 10]
