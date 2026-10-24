# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where persisted deliverables live: ``<root>/task_<task_id>/repeat_<n>/``.

/verify writes this layout; the comparison scorer, multistage reuse and the transport assignment read it,
including reference trees from earlier runs. ``finish_params.json`` in a repeat directory marks a finished run.
"""

import re
from pathlib import Path
from typing import List, Union


FINISH_PARAMS_FILE = "finish_params.json"
_ROLLOUT_ID = re.compile(r"(\d+)-(\d+)")


def task_dir(root: Union[str, Path], task_id: str) -> Path:
    return Path(root) / f"task_{task_id}"


def repeat_dir(root: Union[str, Path], task_id: str, repeat: int) -> Path:
    return task_dir(root, task_id) / f"repeat_{repeat}"


def repeat_from_rollout_id(rollout_id: str) -> int:
    """Read the repeat index from a rollout id, which Gym derives as ``"<task_index>-<rollout_index>"``."""
    match = _ROLLOUT_ID.fullmatch(rollout_id)
    if match is None:
        raise ValueError(f"Rollout id {rollout_id!r} is not '<task_index>-<rollout_index>' with two integers")
    return int(match.group(2))


def repeat_dirs(task_dir: Path) -> List[Path]:
    """All deliverable dirs of a task, supporting both layouts.

    New: ``task_<id>/repeat_<n>/`` — return every repeat dir, sorted. Old: flat ``task_<id>/`` — return
    ``[task_dir]``. Missing → ``[]``.
    """
    if not task_dir.is_dir():
        return []
    repeats = sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("repeat_"))
    return repeats or [task_dir]
