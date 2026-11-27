# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-local v4 action/execution handoff; no v3 or v3.5 state is changed."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any

from .protocol import Action


@dataclass
class Session:
    pending_action: Action | None = None
    previous_failure_signature: str | None = None
    consecutive_identical_failure_count: int = 0
    reference: Any = None


CURRENT_SESSION: contextvars.ContextVar[Session] = contextvars.ContextVar("nemotron_v4_session")
