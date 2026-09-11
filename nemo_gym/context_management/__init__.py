# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in context management shared by sequential Responses agents."""

from nemo_gym.context_management.client import ContextGuardRejected, ContextManagedResponsesClient
from nemo_gym.context_management.config import ContextHistoryConfig
from nemo_gym.context_management.result import LogicalCCResult, LogicalCCSegment, SelectedAction


__all__ = [
    "ContextGuardRejected",
    "ContextHistoryConfig",
    "ContextManagedResponsesClient",
    "LogicalCCResult",
    "LogicalCCSegment",
    "SelectedAction",
]
