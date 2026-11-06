# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed sandbox-provider distribution metadata, without loading providers."""

from importlib.metadata import entry_points


ENTRY_POINT_GROUP = "nemo_gym.sandbox_providers"


def plugin_distributions() -> list[str]:
    """Names of the installed distributions that publish providers as entry points."""
    return sorted({ep.dist.name for ep in entry_points(group=ENTRY_POINT_GROUP) if getattr(ep, "dist", None)})
