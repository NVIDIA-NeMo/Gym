# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate OpenSandbox SDK and transport observations into registered failure kinds."""

from nemo_gym.core.failure_kinds import (
    PROVIDER_CREATE_TIMEOUT,
    PROVIDER_CREDENTIALS_REJECTED,
    PROVIDER_INTERNAL_ERROR,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_NOT_READY,
    PROVIDER_REQUEST_FAILED,
    PROVIDER_UNAVAILABLE,
)
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.failures import sandbox_failure
from nemo_gym.utils.exception_inspection import error_to_http_status


def opensandbox_failure(
    *,
    operation: str,
    sandbox_id: str | None = None,
    cause: BaseException | None = None,
    failure_kind: str | None = None,
    reason: str | None = None,
    execution_id: str | None = None,
    provider_status_reason: str | None = None,
    exit_code: int | None = None,
    http_status: int | None = None,
) -> NgFailureEvent:
    """Classify this operation's evidence, then construct its sandbox event.

    Message text cannot establish OOM, credentials rejection, command termination,
    or safe command replay. Callers raise the event ``from cause``.
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
            SandboxApiException,
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
        elif isinstance(cause, SandboxApiException) and cause.error.code == "KUBERNETES::POD_IP_NOT_AVAILABLE":
            # A structured endpoint-availability observation, even when returned as HTTP 404.
            sdk_kind = PROVIDER_UNAVAILABLE
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
    return sandbox_failure(
        operation=operation,
        failure_kind=failure_kind,
        sandbox_id=sandbox_id,
        cause=cause,
        reason=reason,
        execution_id=execution_id,
        provider="opensandbox",
        provider_status_reason=provider_status_reason,
        exit_code=exit_code,
        http_status=status,
    )
