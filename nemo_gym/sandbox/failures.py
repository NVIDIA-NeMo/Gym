# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox runtime failures: facts for caller policies, without grading decisions."""

from nemo_gym.core.failure_kinds import FailureStage
from nemo_gym.core.failure_records import FailureBoundary, SandboxFailureDetails
from nemo_gym.core.typed_exceptions import NgFailureEvent, sdk_error_details
from nemo_gym.utils.exception_inspection import bounded_text


def sandbox_failure(
    *,
    operation: str,
    failure_kind: str,
    sandbox_id: str | None = None,
    cause: BaseException | None = None,
    reason: str | None = None,
    execution_id: str | None = None,
    provider: str | None = None,
    provider_status_reason: str | None = None,
    exit_code: int | None = None,
    http_status: int | None = None,
    container: str | None = None,
    stage: FailureStage | None = None,
) -> NgFailureEvent:
    """Construct an event from provider-selected facts; callers raise it ``from cause``.

    SDK classification belongs to the provider. HTTP status must describe this
    operation, not a child failure; causes retain their own evidence.
    """
    foreign_error = None
    if cause is not None and not isinstance(cause, NgFailureEvent):
        try:
            message = str(cause)
        except Exception:
            message = None  # A broken SDK formatter must not replace the observed failure.
        foreign_error = sdk_error_details(cause, message=message)
    failure = NgFailureEvent(
        boundary=FailureBoundary.SANDBOX,
        operation=operation,
        failure_kind=failure_kind,
        failure_reason=bounded_text(reason or f"Sandbox {operation} failed; remote outcome is unknown", 2000),
        # When wrapping an already typed failure, the child retains its HTTP facts.
        http_status=http_status,
        foreign_error=foreign_error,
        stage=stage,
        details=SandboxFailureDetails(
            sandbox_id=sandbox_id,
            execution_id=execution_id,
            provider=provider,
            provider_status_reason=bounded_text(provider_status_reason, 2000)
            if provider_status_reason is not None
            else None,
            exit_code=exit_code,
            container=container,
        )
        if any(
            value is not None
            for value in (sandbox_id, execution_id, provider, provider_status_reason, exit_code, container)
        )
        else None,
    )
    failure.__cause__ = cause
    return failure
