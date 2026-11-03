# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox runtime failures: facts for caller policies, without grading decisions."""

from nemo_gym.core.failure_kinds import (
    PROVIDER_CREATE_TIMEOUT,
    PROVIDER_CREDENTIALS_REJECTED,
    PROVIDER_INTERNAL_ERROR,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_NOT_READY,
    PROVIDER_REQUEST_FAILED,
    PROVIDER_UNAVAILABLE,
)
from nemo_gym.exception_inspection import bounded_text, error_to_http_status
from nemo_gym.server_runtime.episode_types import FailureBoundary, SandboxFailureDetails
from nemo_gym.typed_exceptions import NgFailureEvent, sdk_error_details


def sandbox_failure(
    *,
    operation: str,
    sandbox_id: str | None = None,
    cause: BaseException | None = None,
    failure_kind: str | None = None,
    reason: str | None = None,
    execution_id: str | None = None,
    evidence_source: str | None = None,
    http_status: int | None = None,
) -> NgFailureEvent:
    """Create a sandbox-boundary event; callers raise it ``from cause``.

    Only this operation's HTTP status is inspected. Message text cannot establish
    OOM, credentials rejection, command termination, or safe command replay.
    """
    from aiohttp import ClientConnectionError

    try:
        from httpx import TimeoutException, TransportError

        timeout_types = (TimeoutError, TimeoutException)
        transport_types = (ConnectionError, ClientConnectionError, TransportError)
    except ImportError:
        timeout_types, transport_types = (TimeoutError,), (ConnectionError, ClientConnectionError)
    # Normalize only the SDK's known transport wrapper, not arbitrary cause chains.
    observed = cause
    sdk_kind = None
    try:
        from opensandbox.exceptions import (
            SandboxInternalException,
            SandboxReadyTimeoutException,
            SandboxUnhealthyException,
        )
    except ImportError:
        pass
    else:
        if isinstance(cause, SandboxInternalException):
            sdk_kind = PROVIDER_INTERNAL_ERROR
            if isinstance(cause.__cause__, (*timeout_types, *transport_types)):
                observed = cause.__cause__
                sdk_kind = None
        elif isinstance(cause, SandboxReadyTimeoutException):
            sdk_kind = PROVIDER_CREATE_TIMEOUT
        elif isinstance(cause, SandboxUnhealthyException):
            sdk_kind = PROVIDER_NOT_READY
    status = (
        http_status
        if http_status is not None
        else (error_to_http_status(cause) if cause is not None and not isinstance(cause, NgFailureEvent) else None)
    )
    if failure_kind is None:
        if status in (401, 403):
            failure_kind = PROVIDER_CREDENTIALS_REJECTED
        elif isinstance(observed, timeout_types):
            failure_kind = PROVIDER_IO_TIMEOUT
        elif status in (410, 502, 503, 504) or isinstance(observed, transport_types):
            failure_kind = PROVIDER_UNAVAILABLE
        else:
            failure_kind = sdk_kind or PROVIDER_REQUEST_FAILED
    details = None
    if cause is not None and not isinstance(cause, NgFailureEvent):
        try:
            message = str(cause)
        except Exception:
            message = None  # A broken SDK formatter must not replace the observed failure.
        details = sdk_error_details(cause, message=message)
    failure = NgFailureEvent(
        boundary=FailureBoundary.SANDBOX,
        operation=operation,
        failure_kind=failure_kind,
        failure_reason=bounded_text(reason or f"Sandbox {operation} failed; remote outcome is unknown", 2000),
        # When wrapping an already typed failure, the child retains its HTTP facts.
        http_status=status,
        foreign_error=details,
        sandbox=SandboxFailureDetails(
            sandbox_id=sandbox_id, execution_id=execution_id, evidence_source=evidence_source
        )
        if sandbox_id is not None
        else None,
    )
    failure.__cause__ = cause
    return failure
