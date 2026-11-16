# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox observations built only from lifecycle facts an AsyncSandbox actually saw."""

from __future__ import annotations

from typing import Any, Literal

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.trajectory.trajectory_record import SandboxObservation


def lifecycle_observation(
    sandbox: AsyncSandbox | None, *, role: Literal["agent", "verifier", "environment"]
) -> dict[str, Any] | None:
    """The owner's observation of one created sandbox, or None when there is no AsyncSandbox to observe.

    ``outcome`` is ``sandbox_error`` only when every provider close failed; otherwise it stays ``unknown``: a
    confirmed teardown says nothing about the command or the grade. Cleanup failures stay listed after a retry,
    with the first as ``error_type``. Resource usage and remote lifetime are never estimated.
    """
    if not isinstance(sandbox, AsyncSandbox) or sandbox.sandbox_id is None:
        return None
    provider = getattr(getattr(sandbox, "_handle", None), "provider_name", None)
    closed = sandbox.teardown_confirmed
    errors = list(sandbox.cleanup_errors)
    return SandboxObservation(
        role=role,
        provider=provider if isinstance(provider, str) else None,
        sandbox_id=sandbox.sandbox_id,
        outcome="sandbox_error" if closed is False else "unknown",
        error_type=errors[0] if errors else None,
        observed_client_interval_s=sandbox.observed_client_interval_s,
        teardown_confirmed=closed,
        cleanup_errors=errors,
    ).model_dump(mode="json")
