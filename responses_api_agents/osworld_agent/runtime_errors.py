# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed operation failures shared by OSWorld transport and rollout layers."""

from __future__ import annotations


class OSWorldOperationTimeoutError(TimeoutError):
    """Base class for a bounded rollout operation that exceeded its deadline."""


class OSWorldModelTimeoutError(OSWorldOperationTimeoutError):
    """The policy model did not complete within its per-call deadline."""


class OSWorldActionTimeoutError(OSWorldOperationTimeoutError):
    """An environment action did not complete within its per-action deadline."""


class OSWorldGuestStartTimeoutError(TimeoutError):
    """The OSWorld guest in a new sandbox did not become ready within its deadline.

    ``provider`` names the sandbox provider and ``http_status`` is the status of the guest's last reply (``None``
    when the last probe got no reply). ``sandbox_id`` is the provider's id of the sandbox, ``url`` the readiness URL
    polled (without credentials) and ``body_head`` the first 512 bytes of the last reply's body.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        http_status: int | None,
        sandbox_id: str | None = None,
        url: str | None = None,
        body_head: str = "",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.http_status = http_status
        self.sandbox_id = sandbox_id
        self.url = url
        self.body_head = body_head

    @property
    def is_infrastructure_failure(self) -> bool:
        """A fresh sandbox can succeed: no reply, a 5xx, 408 or 429 reply, or an empty success.

        A guest that kept answering another 4xx is a task or image problem a new sandbox repeats.
        """
        status = self.http_status
        return status is None or not 400 <= status < 500 or status in (408, 429)
