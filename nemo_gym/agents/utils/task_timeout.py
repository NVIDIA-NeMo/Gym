# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the per-task agent budget carried by materialized task rows."""

import logging
import math
from collections.abc import Mapping


LOG = logging.getLogger(__name__)
AGENT_TIMEOUT_METADATA_KEY = "agent_timeout_sec"


def resolve_agent_timeout(metadata: Mapping[str, str] | None, *, ceiling: float) -> float:
    """Cap a positive, finite task timeout by the configured execution ceiling.

    Missing or invalid metadata keeps the configured timeout, matching Hermes.
    The execution budget includes model and tool calls. Their individual timeouts,
    and the separate setup and cleanup limits, remain unchanged.
    """
    budget = float(ceiling)
    raw = (metadata or {}).get(AGENT_TIMEOUT_METADATA_KEY)
    if raw is None:
        return budget
    try:
        requested = float(raw)
    except (TypeError, ValueError):
        requested = math.nan
    if not math.isfinite(requested) or requested <= 0:
        LOG.warning("Ignoring invalid %s=%r; using the configured %ss", AGENT_TIMEOUT_METADATA_KEY, raw, budget)
        return budget
    return min(budget, requested)
