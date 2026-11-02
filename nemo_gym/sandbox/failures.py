# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox runtime failures: facts for caller policies, without grading decisions."""

from nemo_gym.core.failure_kinds import (
    PROVIDER_CREDENTIALS_REJECTED,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_REQUEST_FAILED,
    PROVIDER_UNAVAILABLE,
)
from nemo_gym.exception_inspection import bounded_text, error_to_http_status
from nemo_gym.server_runtime.episode_types import FailureBoundary, SandboxFailureDetails
from nemo_gym.typed_exceptions import NgFailureEvent, sdk_error_details


def sandbox_failure(
    *,
    operation: str,
    sandbox_id: str,
    cause: BaseException | None = None,
    failure_kind: str | None = None,
    reason: str | None = None,
    execution_id: str | None = None,
    evidence_source: str | None = None,
) -> NgFailureEvent:
    """Create a sandbox-boundary event; callers raise it ``from cause``.

    Only this operation's HTTP status is inspected. Message text cannot establish
    OOM, credentials rejection, command termination, or safe command replay.
    """
    try:
        from httpx import TimeoutException, TransportError

        timeout_types = (TimeoutError, TimeoutException)
        transport_types = (ConnectionError, TransportError)
    except ImportError:
        timeout_types, transport_types = (TimeoutError,), (ConnectionError,)
    status = error_to_http_status(cause) if cause is not None else None
    if failure_kind is None:
        if status in (401, 403):
            failure_kind = PROVIDER_CREDENTIALS_REJECTED
        elif isinstance(cause, timeout_types):
            failure_kind = PROVIDER_IO_TIMEOUT
        elif status in (410, 502, 503, 504) or isinstance(cause, transport_types):
            failure_kind = PROVIDER_UNAVAILABLE
        else:
            failure_kind = PROVIDER_REQUEST_FAILED
    failure = NgFailureEvent(
        boundary=FailureBoundary.SANDBOX,
        operation=operation,
        failure_kind=failure_kind,
        failure_reason=bounded_text(reason or f"Sandbox {operation} failed; remote outcome is unknown", 2000),
        # When wrapping an already typed failure, the child retains its HTTP facts.
        http_status=status if not isinstance(cause, NgFailureEvent) else None,
        foreign_error=sdk_error_details(cause)
        if cause is not None and not isinstance(cause, NgFailureEvent)
        else None,
        sandbox=SandboxFailureDetails(
            sandbox_id=sandbox_id, execution_id=execution_id, evidence_source=evidence_source
        ),
    )
    failure.__cause__ = cause
    return failure
