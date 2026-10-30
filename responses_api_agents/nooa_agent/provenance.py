# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which NOOA and adapter revisions produced a rollout.

The sandbox runtime installs NOOA from the commit pinned in this directory's requirements.txt; nothing inside the
sandbox knows that commit (NOOA's own git metadata describes the task repository it runs in). The agent server
knows the pin, so it stamps it on every response's metadata, next to the Gym checkout's commit.
"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path


NOOA_COMMIT_KEY = "nooa_commit"
GYM_ADAPTER_COMMIT_KEY = "gym_adapter_commit"

_AGENT_DIR = Path(__file__).resolve().parent
_PIN = re.compile(r"^nooa\s*@\s*\S*?([0-9a-f]{40})(?:\b|$)", re.MULTILINE)


def nooa_pin(requirements: Path = _AGENT_DIR / "requirements.txt") -> str | None:
    """The 40-hex NOOA commit the `nooa @ ...` requirement names, or None without one."""
    try:
        text = requirements.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _PIN.search(text)
    return match.group(1) if match else None


def git_commit(root: Path = _AGENT_DIR.parents[1]) -> str | None:
    """HEAD of the checkout at root, or None outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", commit) else None


@lru_cache(maxsize=1)
def provenance_metadata() -> dict[str, str]:
    """String metadata for a response: the NOOA pin and the adapter checkout, each omitted when unknown."""
    values = {NOOA_COMMIT_KEY: nooa_pin(), GYM_ADAPTER_COMMIT_KEY: git_commit()}
    return {key: value for key, value in values.items() if value}
