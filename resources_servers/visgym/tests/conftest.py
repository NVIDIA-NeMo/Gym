# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import importlib.util

import pytest


def _has_upstream_visgym() -> bool:
    if importlib.util.find_spec("gymnasium") is None:
        return False
    try:
        gym = importlib.import_module("gymnasium")
        importlib.import_module("gymnasium.envs")
    except ImportError:
        return False
    try:
        gym.spec("maze_2d/easy")
    except Exception:
        return False
    return True


requires_visgym = pytest.mark.skipif(
    not _has_upstream_visgym(),
    reason="VisGym's gymnasium fork is not installed in this environment.",
)
