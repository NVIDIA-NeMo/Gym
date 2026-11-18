# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Lean4 sandbox that does not answer is recorded on the request, and its error result is unchanged.

The error result is scored as before. Recording it means the rollout is no longer indistinguishable from one whose
proof was compiled and found wrong.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nemo_gym.core import RequestFailureLog
from resources_servers.math_formal_lean.sandbox_client import Lean4SandboxClient


def _client_answering(post: AsyncMock) -> Lean4SandboxClient:
    client = Lean4SandboxClient()
    client._client = MagicMock(post=post)
    return client


def _reply(status_code: int = 200, body: object = None, *, bad_json: bool = False) -> MagicMock:
    reply = MagicMock(status_code=status_code)
    if bad_json:
        reply.json.side_effect = json.JSONDecodeError("bad", "doc", 0)
    else:
        reply.json.return_value = body
    return reply


async def _execute(post: AsyncMock) -> tuple[dict, list]:
    with RequestFailureLog.capture() as log:
        result = await _client_answering(post).execute_lean4("theorem t : True := trivial", timeout=10.0)
    return result, [e for e in log.report.events.values() if e.operation == "scored_despite_failure"]


@pytest.mark.parametrize(
    ("post", "expected_status", "expected_in_reason"),
    [
        (AsyncMock(return_value=_reply(502)), "error", "HTTP 502"),
        (AsyncMock(side_effect=httpx.ReadTimeout("slow")), "timeout", "no reply within 15.0 s"),
        (AsyncMock(side_effect=httpx.ConnectError("refused")), "error", "ConnectError"),
        (AsyncMock(return_value=_reply(bad_json=True)), "error", "not valid JSON"),
    ],
    ids=["http_502", "client_timeout", "connect_error", "invalid_json"],
)
async def test_a_sandbox_that_does_not_answer_is_recorded_and_its_result_is_unchanged(
    post: AsyncMock, expected_status: str, expected_in_reason: str
) -> None:
    result, recorded = await _execute(post)

    assert result["process_status"] == expected_status
    (event,) = recorded
    assert (event.boundary.value, event.failure_kind, event.stage) == (
        "resources_server",
        "verifier_error",
        "verification",
    )
    assert expected_in_reason in event.failure_reason


@pytest.mark.parametrize("process_status", ["completed", "timeout"])
async def test_a_compile_the_sandbox_reports_is_not_recorded(process_status: str) -> None:
    # A timeout the sandbox itself reports means the proof took too long, which is a result, not a missing answer.
    body = {"process_status": process_status, "stdout": "", "stderr": ""}

    result, recorded = await _execute(AsyncMock(return_value=_reply(200, body)))

    assert result == body
    assert recorded == []
