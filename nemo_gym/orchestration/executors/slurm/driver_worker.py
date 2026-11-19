# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render an owned driver step whose worker waits for local and scheduler cleanup."""

import shlex
from pathlib import Path

from nemo_gym.utils import process_worker


# Only the step carrying this invocation's random name belongs to the worker. Never cancel its allocation:
# other drivers and shared services can still be running there. An unreadable scheduler is not an empty queue.
_CLEANUP_STEP = r"""set -euo pipefail
: "${SLURM_JOB_ID:?}" "${GYM_DRIVER_STEP_NAME:?}"
empty=0
for attempt in $(seq 1 20); do
    rows=$(squeue --steps --jobs="$SLURM_JOB_ID" --noheader --format='%.64i|%.128j')
    found=0
    while IFS='|' read -r step name; do
        step=${step//[[:space:]]/}
        name=${name//[[:space:]]/}
        if [[ "$name" == "$GYM_DRIVER_STEP_NAME" ]]; then
            [[ "$step" =~ ^[0-9]+\.[0-9]+$ ]] || { echo "Invalid driver step id: $step" >&2; exit 1; }
            [[ "${step%%.*}" == "$SLURM_JOB_ID" ]] || exit 1
            found=1
            sig=TERM
            (( attempt <= 10 )) || sig=KILL
            # A step can finish between squeue and scancel. Only a subsequent successful query confirms that.
            scancel --signal="$sig" "$step" || true
        fi
    done <<< "$rows"
    if (( found == 0 )); then
        empty=$((empty + 1))
        (( empty < 2 )) || exit 0
    else
        empty=0
    fi
    sleep 1
done
echo "Driver step termination was not confirmed" >&2
exit 1
"""


def render_driver_worker(driver_srun: str, bench_dir: Path) -> tuple[str, str]:
    """Return the worker's prelude and command, leaving service watches in Gym's existing renderer.

    The submitting Gym embeds its worker so the batch host and driver's pinned Gym need no matching module.
    ``driver_srun`` is a single foreground command, including its environment assignments and shell-quoted argv.
    """
    source = Path(process_worker.__file__).read_text()
    prelude = (
        "GYM_DRIVER_STEP_NAME=$(cat /proc/sys/kernel/random/uuid) || exit 70\n"
        'export GYM_DRIVER_STEP_NAME="gym-driver-$GYM_DRIVER_STEP_NAME"'
    )
    receipt = shlex.quote(str(bench_dir / "logs" / "driver-worker")) + '."$SLURM_JOB_ID"."$GYM_DRIVER_STEP_NAME".json'
    command = (
        f"python3 -c {shlex.quote(source)} --receipt {receipt} "
        f"--cleanup-command {shlex.quote(_CLEANUP_STEP)} -- env {driver_srun}"
    )
    return prelude, command
