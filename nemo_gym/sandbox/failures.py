# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox runtime failures: facts for caller policies, without grading decisions."""

from nemo_gym.exception_inspection import bounded_text
from nemo_gym.server_runtime.episode_types import FailureBoundary, SandboxFailureDetails
from nemo_gym.typed_exceptions import NgFailureEvent, sdk_error_details


def sandbox_failure(
    *,
    operation: str,
    failure_kind: str,
    sandbox_id: str | None = None,
    cause: BaseException | None = None,
    reason: str | None = None,
    execution_id: str | None = None,
    evidence_source: str | None = None,
    http_status: int | None = None,
) -> NgFailureEvent:
    """Construct an event from provider-selected facts; callers raise it ``from cause``.

    SDK classification belongs to the provider. HTTP status must describe this
    operation, not a child failure; causes retain their own evidence.
    """
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
        http_status=http_status,
        foreign_error=details,
        sandbox=SandboxFailureDetails(
            sandbox_id=sandbox_id, execution_id=execution_id, evidence_source=evidence_source
        )
        if sandbox_id is not None
        else None,
    )
    failure.__cause__ = cause
    return failure
