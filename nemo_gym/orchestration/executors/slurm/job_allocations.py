# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Record a resumable job's own running allocation from inside its driver step.

The job script records allocations with `sacct` (see `resume_script`), and the job that ends the chain does so only
after its driver step. Exports run inside that step, so an export of the job directory would miss the exporting job's
own allocation. The driver step has no `sacct`, but Slurm's job environment describes the running allocation; this
module appends that row to `JOB_ALLOCATIONS_NAME` just before the exports, in the job script's row format. The job
script's later record of the same incarnation then adds nothing (it keeps one row per job, restart count and
`ended`). Earlier jobs of the chain are already recorded: each job's prologue records the ones that ended.
"""

import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from nemo_gym.orchestration.executors.slurm.resume_script import (
    JOB_ALLOCATIONS_LOCK_NAME,
    JOB_ALLOCATIONS_NAME,
    JOB_CHAIN_NAME,
)


logger = logging.getLogger(__name__)

# What `source` says of a row built from Slurm's job environment rather than from `sacct`.
SLURM_ENVIRONMENT_SOURCE = "slurm_environment"
# The job script's lock protocol: a lock older than a minute is stale, and a writer gives up after about 15 s.
_STALE_LOCK_S = 60
_LOCK_TRIES = 50
_LOCK_WAIT_S = 0.3
_DIGITS = re.compile(r"[0-9]{1,15}")
# The characters the job script keeps in a text field.
_UNSAFE_TEXT = re.compile(r"[^][A-Za-z0-9_.,=:/+()* -]")


def _count(value: str | None) -> int | None:
    """A count from a Slurm variable such as `4` or `gb300:4` (a typed request), or None."""
    match = re.fullmatch(r"(?:[A-Za-z0-9_-]+:)?([0-9]{1,6})", value or "")
    return int(match.group(1)) if match else None


def own_allocation_row(environ: Mapping[str, str], now: float) -> dict[str, object] | None:
    """This job's running allocation as a `JOB_ALLOCATIONS_NAME` row, or None when the environment lacks it."""
    job_id = environ.get("SLURM_JOB_ID", "")
    start = environ.get("SLURM_JOB_START_TIME", "")
    nodes = _count(environ.get("SLURM_JOB_NUM_NODES"))
    if not (_DIGITS.fullmatch(job_id) and _DIGITS.fullmatch(start) and nodes):
        return None
    # Set when the job asked for the same GPUs on every node (`--gpus-per-node`). The step's own GPU count would be
    # wrong for a job whose node pools differ, so without it the GPUs are unknown (null), never a guess.
    per_node = _count(environ.get("SLURM_GPUS_PER_NODE"))
    restarts = environ.get("SLURM_RESTART_COUNT", "0")
    return {
        # The allocation's full TRES string is only in the scheduler's accounting.
        "alloc_tres": "",
        "elapsed_s": max(0, int(now) - int(start)),
        "end_epoch_s": None,
        "ended": False,
        "gpus": per_node * nodes if per_node is not None else None,
        "job_id": job_id,
        "node_list": _UNSAFE_TEXT.sub("", environ.get("SLURM_JOB_NODELIST", "")),
        "nodes": nodes,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "restarts": int(restarts) if _DIGITS.fullmatch(restarts) else 0,
        "source": SLURM_ENVIRONMENT_SOURCE,
        "start_epoch_s": int(start),
        "state": "RUNNING",
    }


def _already_recorded(path: Path, row: Mapping[str, object]) -> bool:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return False
    for line in lines:
        try:
            existing = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(existing, dict) and all(existing.get(k) == row[k] for k in ("job_id", "restarts", "ended")):
            return True
    return False


def record_own_allocation(
    job_dir: Path,
    *,
    environ: Mapping[str, str] = os.environ,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Append this job's running allocation to the job directory's `JOB_ALLOCATIONS_NAME`, once; True if added.

    Only a resumable Slurm job records allocations (its directory has `JOB_CHAIN_NAME`). Best-effort, like the job
    script: missing environment, a lock that stays held, or a write error records nothing and is logged.
    """
    if not (job_dir / JOB_CHAIN_NAME).is_file():
        return False
    row = own_allocation_row(environ, now())
    if row is None:
        logger.warning(
            "Slurm's job environment does not describe this allocation; %s gets no row.", JOB_ALLOCATIONS_NAME
        )
        return False
    lock = job_dir / JOB_ALLOCATIONS_LOCK_NAME
    for _ in range(_LOCK_TRIES):
        try:
            if now() - lock.stat().st_mtime > _STALE_LOCK_S:
                lock.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            sleep(_LOCK_WAIT_S)
    else:
        logger.warning(
            "%s stayed held; %s records no row for job %s now.", lock.name, JOB_ALLOCATIONS_NAME, row["job_id"]
        )
        return False
    try:
        path = job_dir / JOB_ALLOCATIONS_NAME
        if _already_recorded(path, row):
            return False
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return True
    except OSError as exc:
        logger.warning("Cannot append to %s: %s", JOB_ALLOCATIONS_NAME, exc)
        return False
    finally:
        lock.unlink(missing_ok=True)
