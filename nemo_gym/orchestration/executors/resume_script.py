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

"""The auto-resume prologue: a self-chaining Slurm job.

Rendered at the top of a benchmark's sbatch script, ahead of every
service/driver command, when `BenchmarkRunConfig.resumable` is set. Each job
in the chain inspects the PREVIOUS job (passed as `$1`) via `sacct`, decides
whether to continue, and unconditionally queues its own successor with an
`afternotok` dependency before doing any real work. A job that COMPLETEs
leaves its queued successor with a dependency that can never be satisfied, so
Slurm cancels it on its own -- the chain needs no explicit "we're done" step
on the success path.
"""

from nemo_gym.orchestration.api import ResumeConfig


# `_stage` in slurm.py always writes this benchmark's script to exactly this
# filename inside its job dir, which `#SBATCH --chdir` makes `$PWD` at runtime.
_JOB_SCRIPT_NAME = "job.sh"

_PROLOGUE_TEMPLATE = """\
# --- Auto-resume chain ---
OUTPUT_DIR="$PWD"
_this_script="$OUTPUT_DIR/{job_script_name}"
_prev_slurm_job_id="${{1:-}}"
_walltime_file="$OUTPUT_DIR/.gym_accumulated_walltime"
_retry_file="$OUTPUT_DIR/.gym_infra_retries"

if [[ "$_prev_slurm_job_id" != "" ]]; then
    for _sacct_try in 1 2 3 4 5; do
        _prev_state=$(sacct -j $_prev_slurm_job_id -P -n -o State | head -n 1)
        [[ -n "$_prev_state" ]] && break
        sleep 2
    done
    _prev_elapsed=$(sacct -j $_prev_slurm_job_id -P -n -o ElapsedRaw | head -n 1)
    _prev_elapsed=${{_prev_elapsed:-0}}
    _gym_accumulated=$(cat "$_walltime_file" 2>/dev/null || echo 0)
    _gym_accumulated=$((_gym_accumulated + _prev_elapsed))
    echo $_gym_accumulated > "$_walltime_file"

    if [[ $_prev_state == 'COMPLETED' ]]; then
        echo "Previous job $_prev_slurm_job_id completed successfully. Exiting."
        exit 0
    elif [[ $_prev_state == CANCELLED* ]]; then
        echo "Previous job $_prev_slurm_job_id was cancelled. Stopping chain."
        exit 0
    elif [[ $_prev_state == 'TIMEOUT' || $_prev_state == 'PREEMPTED' || $_prev_state == 'NODE_FAIL' ]]; then
        echo "Previous job $_prev_slurm_job_id: $_prev_state. Resuming..."
{max_walltime_check}
    else
        _retries=$(cat "$_retry_file" 2>/dev/null || echo 0)
        _retries=$((_retries + 1))
        echo $_retries > "$_retry_file"
        if [[ $_retries -ge {max_retries} ]]; then
            echo "Infra retry limit ({max_retries}) reached after $_prev_state. Stopping."
            exit 1
        fi
        echo "Previous job $_prev_slurm_job_id: $_prev_state. Infra retry $_retries/{max_retries}..."
    fi
fi

echo "$SLURM_JOB_ID" >> "$OUTPUT_DIR/.gym_job_chain"
_next_output=$(sbatch --dependency=afternotok:$SLURM_JOB_ID "$_this_script" $SLURM_JOB_ID 2>&1) && {{
    _next_id=$(echo "$_next_output" | grep -oE '[0-9]+')
    if [[ -n "$_next_id" ]]; then
        echo "Auto-resume follow-up queued: $_next_id (afternotok:$SLURM_JOB_ID)"
    fi
}} || echo "WARNING: Failed to submit auto-resume follow-up. Chain will NOT continue on failure."
"""

_MAX_WALLTIME_CHECK_TEMPLATE = """\
        if (( _gym_accumulated >= {seconds} )); then
            echo "Accumulated walltime (${{_gym_accumulated}}s) reached max_walltime cap ({raw} = {seconds}s). Stopping chain."
            exit 0
        fi"""


def _walltime_to_seconds(duration: str) -> int:
    """Convert a Slurm-style duration string to seconds.

    Covers the subset of Slurm's `--time` grammar that `max_walltime` accepts:
    `MM`, `MM:SS`, `HH:MM:SS`, `D-HH`, `D-HH:MM`, `D-HH:MM:SS` -- the same
    spellings Slurm itself takes, so a value can be copied straight from
    `compute.walltime`.
    """
    text = duration.strip()
    days = 0
    rest = text
    if "-" in text:
        day_str, _, rest = text.partition("-")
        if not day_str.isdigit():
            raise ValueError(f"Invalid max_walltime {duration!r}: day component must be an integer.")
        days = int(day_str)

    parts = rest.split(":")
    if not (1 <= len(parts) <= 3) or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"Invalid max_walltime {duration!r}: expected Slurm duration syntax, e.g. '48:00:00' or '2-00:00:00'."
        )
    values = [int(p) for p in parts]

    if days:
        # D-HH, D-HH:MM, or D-HH:MM:SS
        values += [0] * (3 - len(values))
        hours, minutes, seconds = values
    elif len(values) == 1:
        hours, minutes, seconds = 0, values[0], 0
    elif len(values) == 2:
        hours, minutes, seconds = 0, values[0], values[1]
    else:
        hours, minutes, seconds = values

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def render_resume_prologue(resume: ResumeConfig) -> str:
    """Render the auto-resume prologue for one benchmark's sbatch script."""
    max_walltime_check = ""
    if resume.max_walltime is not None:
        seconds = _walltime_to_seconds(resume.max_walltime)
        max_walltime_check = _MAX_WALLTIME_CHECK_TEMPLATE.format(seconds=seconds, raw=resume.max_walltime)

    return _PROLOGUE_TEMPLATE.format(
        job_script_name=_JOB_SCRIPT_NAME,
        max_retries=resume.max_retries,
        max_walltime_check=max_walltime_check,
    )
