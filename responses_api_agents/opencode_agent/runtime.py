# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stock OpenCode plus the same observability patch, locally and in sandboxes."""

from pathlib import Path


OPENCODE_VERSION = "1.17.11"
OBSERVABILITY_PATCH = Path(__file__).with_name("assistant_message_header.js")


def apply_observability_patch(config: dict, *, plugin_path: Path = OBSERVABILITY_PATCH) -> None:
    """Register the shared correlation plugin at its execution-side location."""
    config.setdefault("plugin", []).append(plugin_path.as_uri())
