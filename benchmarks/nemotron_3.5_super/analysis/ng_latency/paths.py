# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the latency pipeline's own files live: ``<run dir>/metrics/``, with a read-time fallback to the old layout.

Gym's native files (``rollouts.jsonl`` and friends) stay at the run directory root. Everything this package
writes goes under ``metrics/``. Readers call :func:`locate` so runs recorded before the folder existed are
still found at the root.
"""

from __future__ import annotations

from pathlib import Path


METRICS_DIR_NAME = "metrics"


def metrics_dir(run_dir: Path, *, create: bool = True) -> Path:
    """The folder for the files this package writes under ``run_dir``; created unless ``create`` is false."""
    path = Path(run_dir) / METRICS_DIR_NAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def locate(run_dir: Path, name: str) -> Path:
    """Where ``name`` is under ``run_dir``: ``metrics/<name>`` if present, else the old root location if
    present, else the ``metrics/`` path (for a file that is about to be written). Never creates anything."""
    run_dir = Path(run_dir)
    preferred = run_dir / METRICS_DIR_NAME / name
    if preferred.exists():
        return preferred
    legacy = run_dir / name
    if legacy.exists():
        return legacy
    return preferred
