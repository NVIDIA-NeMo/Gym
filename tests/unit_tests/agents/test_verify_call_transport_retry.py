# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An agent's verify call retries a reply lost in transport, within the judge call's bound, before it masks a row."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientPayloadError, ClientResponseError, ServerDisconnectedError

from nemo_gym.agents.utils import verify_call
from nemo_gym.agents.utils.verify_call import VERIFY_CALL_MAX_ATTEMPTS, verify_with_transport_retry
from nemo_gym.core.failure_transport import RequestFailureLog
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.verification.judge import JUDGE_CALL_MAX_ATTEMPTS


@pytest.fixture
def stateless(monkeypatch):
    mode = AsyncMock(return_value=True)
    monkeypatch.setattr(verify_call, "_stateless_verifier", mode)
    monkeypatch.setattr(verify_call, "_retry_wait_s", lambda attempt: 0)
    return mode


def _calls(*outcomes):
    """A verify call that raises or returns each of ``outcomes`` in turn."""
    remaining = list(outcomes)
    calls = []

    async def call():
        calls.append(1)
        outcome = remaining.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return call, calls


def _verify_events(log: RequestFailureLog) -> list:
    return [event for event in log.report.events.values() if event.operation == "verify"]


async def test_a_payload_error_is_retried_within_the_bound_and_the_grade_is_kept(stateless) -> None:
    assert VERIFY_CALL_MAX_ATTEMPTS == JUDGE_CALL_MAX_ATTEMPTS == 3
    call, calls = _calls(ClientPayloadError("cut"), ServerDisconnectedError(), {"reward": 1.0})
    with RequestFailureLog.capture() as log:
        result = await verify_with_transport_retry(call, server_client=MagicMock(), server_name="rs")
    assert result == {"reward": 1.0} and len(calls) == 3
    events = _verify_events(log)
    # Each failed attempt is observed, as a retried judge call's attempts are: one operation, one attempt id each.
    assert [event.failure_kind for event in events] == ["agent_route_failed"] * 2
    assert len({event.operation_id for event in events}) == 1
    assert sorted(event.attempt_id.rsplit("-", 1)[1] for event in events) == ["1", "2"]
    assert {event.foreign_error.exception_type.rsplit(".", 1)[1] for event in events} == {
        "ClientPayloadError",
        "ServerDisconnectedError",
    }


async def test_a_persistent_transport_failure_still_fails_after_the_bound(stateless) -> None:
    call, calls = _calls(*(ClientPayloadError("cut") for _ in range(5)))
    with RequestFailureLog.capture() as log, pytest.raises(NgFailureEvent) as raised:
        await verify_with_transport_retry(call, server_client=MagicMock(), server_name="rs")
    assert len(calls) == VERIFY_CALL_MAX_ATTEMPTS
    assert raised.value.failure_kind == "agent_route_failed" and raised.value.operation == "verify"
    assert raised.value.failure_reason == "Agent operation verify failed: ClientPayloadError"
    assert len(_verify_events(log)) == VERIFY_CALL_MAX_ATTEMPTS


async def test_a_verifier_reply_is_never_retried(stateless) -> None:
    call, calls = _calls(ClientResponseError(None, (), status=503, message="busy"), {"reward": 1.0})
    with RequestFailureLog.capture(), pytest.raises(NgFailureEvent):
        await verify_with_transport_retry(call, server_client=MagicMock(), server_name="rs")
    assert len(calls) == 1 and not stateless.await_count


async def test_a_verifier_that_is_not_stateless_is_never_sent_a_delivered_request_again(monkeypatch) -> None:
    client = MagicMock()
    client.get = AsyncMock(side_effect=AssertionError("mode unknown"))
    call, calls = _calls(ClientPayloadError("cut"), {"reward": 1.0})
    with RequestFailureLog.capture(), pytest.raises(NgFailureEvent):
        await verify_with_transport_retry(call, server_client=client, server_name="rs")
    assert len(calls) == 1
