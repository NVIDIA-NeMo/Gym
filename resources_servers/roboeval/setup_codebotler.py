# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch the pinned CodeBotler checkout that RoboEval's tasks, simulator, and tracer are imported from.

RoboEval is not vendored: its license forbids using the benchmark contents as ML training data, so the checkout is
fetched at a pinned commit into a gitignored directory next to this module.
"""

import fcntl
import os
import subprocess
from pathlib import Path


CODEBOTLER_URL = "https://github.com/ut-amrl/codebotler"
CODEBOTLER_COMMIT = "26d202a05a2153a15353a62a4854e6fb0be73865"
DEFAULT_ROOT = Path(__file__).parent / ".codebotler"


def ensure_codebotler(root: str | os.PathLike | None = None) -> Path:
    """Return a CodeBotler checkout, cloning the pinned commit into ``DEFAULT_ROOT`` when no root is given.

    ``root`` (or ``ROBOEVAL_ROOT``) points at an existing checkout and is used as-is. Concurrent callers share one
    clone through a file lock.
    """
    root = root or os.environ.get("ROBOEVAL_ROOT")
    if root:
        path = Path(root).resolve()
        if not (path / "benchmark" / "tasks").is_dir():
            raise FileNotFoundError(f"{path} is not a CodeBotler checkout (no benchmark/tasks)")
        return path

    DEFAULT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_ROOT.parent / ".codebotler.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (DEFAULT_ROOT / "benchmark" / "tasks").is_dir():
            subprocess.run(["git", "clone", "--quiet", CODEBOTLER_URL, str(DEFAULT_ROOT)], check=True)
            subprocess.run(["git", "-C", str(DEFAULT_ROOT), "checkout", "--quiet", CODEBOTLER_COMMIT], check=True)
        head = subprocess.run(
            ["git", "-C", str(DEFAULT_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if head != CODEBOTLER_COMMIT:
            raise RuntimeError(f"{DEFAULT_ROOT} is at {head}, expected pinned CodeBotler commit {CODEBOTLER_COMMIT}")
    return DEFAULT_ROOT
