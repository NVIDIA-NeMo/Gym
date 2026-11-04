# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An agent's verify call, with a bounded retry of a verify call lost in transport.

A verify call whose reply was lost (the body cut off mid-read, the peer dropped the connection, a timeout) has no
verdict, and an agent that masks its row then keeps a completed answer ungraded. The call is tried again, up to
``VERIFY_CALL_MAX_ATTEMPTS`` attempts with the judge call's full-jitter backoff, before the failure reaches the agent.
A verify request that may have been delivered is sent again only to a verifier that reports itself ``STATELESS``
(``GET /reverify_mode``): grading the same saved answer again is then safe. An HTTP reply (any status) is the
verifier's own answer and is never retried here.
"""

import asyncio
import random
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import ClientConnectionError, ClientPayloadError

from nemo_gym.client.http_client import get_response_json, raise_for_status
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.failure_kinds import AGENT_ROUTE_FAILED
from nemo_gym.core.failure_records import FailureBoundary
from nemo_gym.core.failure_transport import observe_failure
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.server_runtime.base_resources_server import ReverifyMode
from nemo_gym.verification.judge import JUDGE_CALL_MAX_ATTEMPTS, JUDGE_RETRY_BASE_WAIT_S, JUDGE_RETRY_MAX_WAIT_S


VERIFY_CALL_MAX_ATTEMPTS = JUDGE_CALL_MAX_ATTEMPTS
"""Attempts of one verify call (the first one included) before its transport failure reaches the agent."""

# A lost reply: a body cut off mid-read, a dropped or refused connection, a timeout. Not an HTTP status.
_TRANSPORT_ERRORS = (ClientPayloadError, ClientConnectionError, ConnectionError, TimeoutError)


def is_verify_transport_error(error: BaseException) -> bool:
    """Whether a failed verify call lost its reply in transport, so another attempt can succeed."""
    return isinstance(error, _TRANSPORT_ERRORS)


def _retry_wait_s(attempt: int) -> float:
    """Full-jitter exponential backoff after failed attempt ``attempt`` (1-based), as a judge call waits."""
    return random.uniform(0, min(JUDGE_RETRY_MAX_WAIT_S, JUDGE_RETRY_BASE_WAIT_S * 2 ** (attempt - 1)))


async def _stateless_verifier(server_client: ServerClient, server_name: str) -> bool:
    """Whether the verifier reports ``STATELESS``; a verifier that cannot say is not resent a delivered request."""
    try:
        response = await server_client.get(server_name=server_name, url_path="/reverify_mode")
        await raise_for_status(response)
        return ReverifyMode(await get_response_json(response)) == ReverifyMode.STATELESS
    except Exception:  # noqa: BLE001 - an unknown mode never admits a resend
        return False


def verify_route_failure(error: BaseException, *, operation_id: str, attempt: int) -> NgFailureEvent:
    """The agent boundary's failure of one verify attempt, caused by ``error``; ``agent_route_failure`` for verify with
    the attempt named, so a retried call keeps every attempt as an observation."""
    failure = NgFailureEvent(
        boundary=FailureBoundary.AGENT,
        failure_kind=AGENT_ROUTE_FAILED,
        failure_reason=f"Agent operation verify failed: {type(error).__name__}",
        operation="verify",
        operation_id=operation_id,
        attempt_id=f"{operation_id}-{attempt}",
        stage="verification",
    )
    failure.__cause__ = error  # set before recording, so the observation carries the cause chain
    return failure


async def verify_with_transport_retry(
    call: Callable[[], Awaitable[Any]], *, server_client: ServerClient, server_name: str
) -> Any:
    """Await ``call`` (one verify request and its parsed reply), retrying a reply lost in transport.

    Every failed attempt is observed as its own ``agent_route_failed`` event of operation ``verify`` (one
    ``operation_id`` per call, the attempt in ``attempt_id``). A transport failure is tried again, up to
    ``VERIFY_CALL_MAX_ATTEMPTS`` attempts, only against a ``STATELESS`` verifier. The last failure is raised as that
    event, ``from`` its cause; any other failure is raised the same way at once.
    """
    operation_id = uuid.uuid4().hex
    resend: bool | None = None
    for attempt in range(1, VERIFY_CALL_MAX_ATTEMPTS + 1):
        try:
            return await call()
        except Exception as error:
            failure = verify_route_failure(error, operation_id=operation_id, attempt=attempt)
            observe_failure(failure)
            if attempt < VERIFY_CALL_MAX_ATTEMPTS and is_verify_transport_error(error):
                if resend is None:
                    resend = await _stateless_verifier(server_client, server_name)
                if resend:
                    await asyncio.sleep(_retry_wait_s(attempt))
                    continue
            raise failure from error
    raise AssertionError("unreachable")  # pragma: no cover
