# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared classification for agent-produced deliverable artifacts."""

from pathlib import Path


# Run state may live beside the agent's output, but must never be graded or
# treated as a reusable deliverable.
IGNORE_FILES = frozenset(
    {
        "finish_params.json",
        "history.json",
        "history.pkl",
        "inprogress_history.json",
        "metadata.json",
        "log.txt",
        "reference_files",
    }
)


def is_deliverable(path: Path) -> bool:
    """Return whether *path* is an agent-produced output artifact."""
    return path.is_file() and path.name not in IGNORE_FILES
