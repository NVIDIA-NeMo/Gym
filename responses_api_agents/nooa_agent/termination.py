# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA termination reasons shared by runtime and response projection."""

from enum import StrEnum


class NOOATerminationReason(StrEnum):
    """Stable wire values describing why a NOOA rollout stopped."""

    POLICY_BUDGET_EXCEEDED = "policy_budget_exceeded"
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"
    MAX_OUTPUT_TOKENS_EXCEEDED = "max_output_tokens_exceeded"
    INVALID_POLICY_OUTPUT = "invalid_policy_output"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    CANCELLED = "cancelled"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
