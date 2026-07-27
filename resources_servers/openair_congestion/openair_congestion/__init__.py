# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain implementation for the OpenAir congestion resource server.

The package is colocated with its NeMo Gym resource server so a clean checkout
contains the action schemas, deterministic replay dynamics, guardrails, reward
function, telemetry rendering, and related utilities.

``ENV_NAME`` and ``SCHEMA_VERSION`` identify the stable environment contract
used by generated data and model checkpoints.
"""

from __future__ import annotations


ENV_NAME: str = "openair_congestion_v1"
SCHEMA_VERSION: str = "1.0.0"
__version__: str = "0.1.0"

__all__ = ["ENV_NAME", "SCHEMA_VERSION", "__version__"]
