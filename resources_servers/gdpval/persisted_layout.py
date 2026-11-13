# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where persisted deliverables live: ``<root>/task_<task_id>/repeat_<n>/``.

``<n>`` is the rollout index of one of Gym's own rollouts, or ``id_<rollout_id>`` for a rollout another
orchestrator named (see ``repeat_key_from_rollout_id``). /verify writes this layout; the comparison scorer,
multistage reuse and the transport assignment read it, including reference trees from earlier runs. ``finish_params.json`` in a repeat directory marks a finished run.
"""

import re
from pathlib import Path
from typing import List, Union


FINISH_PARAMS_FILE = "finish_params.json"
_ROLLOUT_ID = re.compile(r"(\d+)-(\d+)")


def task_dir(root: Union[str, Path], task_id: str) -> Path:
    return Path(root) / f"task_{task_id}"


def repeat_dir(root: Union[str, Path], task_id: str, repeat: Union[int, str]) -> Path:
    return task_dir(root, task_id) / f"repeat_{repeat}"


def repeat_key_from_rollout_id(rollout_id: str) -> str:
    """Name a rollout's repeat directory after its rollout id.

    Gym derives its own rollout ids as ``"<task_index>-<rollout_index>"``; their key is the rollout index, so a task's
    repeats are ``repeat_0``, ``repeat_1``, ... An id that another orchestrator assigns, such as
    ``job-<job_id>-sample-<n>``, keeps the whole id under an ``id_`` prefix, so each such rollout still gets a
    directory of its own and none can share one with a Gym repeat: ``"0"`` keys ``id_0``, not ``0`` like ``"7-0"``.
    ``EpisodeId`` allows only path-safe characters in a rollout id.
    """
    match = _ROLLOUT_ID.fullmatch(rollout_id)
    return str(int(match.group(2))) if match else f"id_{rollout_id}"


def repeat_dirs(task_dir: Path) -> List[Path]:
    """All deliverable dirs of a task, supporting both layouts.

    New: ``task_<id>/repeat_<n>/`` — return every repeat dir, sorted. Old: flat ``task_<id>/`` — return
    ``[task_dir]``. Missing → ``[]``.
    """
    if not task_dir.is_dir():
        return []
    repeats = sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("repeat_"))
    return repeats or [task_dir]
