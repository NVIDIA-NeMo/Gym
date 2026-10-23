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
