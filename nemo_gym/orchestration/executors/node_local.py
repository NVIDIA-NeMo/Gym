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

"""The driver's node-local directory, archived into the job dir once per Slurm job (`node_local`).

Files that many small writes produce, such as per-rollout evidence, stay on the driver node's own disk while the
driver runs. The batch script creates `<base_dir>/gym-node-local-$SLURM_JOB_ID` on the driver's node, optionally
extracts the newest earlier archive into it, and mounts it into the driver container at a path that is the same for
every job of the run, so absolute paths the run saves stay valid in its later jobs. When the driver step ends, the
directory is written to `<job dir>/node-local/<SLURM_JOB_ID>.tar.gz` (one file, renamed into place once complete) and
removed. With restore each archive holds the whole directory, so only the newest two are kept.

A step at its time limit is killed within moments (its `srun` gets TERM and kills the step), too fast to archive. So
the watch loop stops the driver `stop_before_walltime_seconds` before the job's end time, archives, and then waits
for the limit, so the job still ends TIMEOUT and the auto-resume chain reads it as it reads any segment.
"""

import shlex
from pathlib import Path

from nemo_gym.orchestration.api import NODE_LOCAL_STOP_WAIT_SECONDS, NodeLocalConfig


NODE_LOCAL_DIR_ENV = "GYM_NODE_LOCAL_DIR"
ARCHIVES_DIRNAME = "node-local"
# Inside the driver container: one path per run and benchmark, the same in every job of the run.
_CONTAINER_ROOT = "/tmp/gym-node-local"

# Run on the driver's node as `bash -c SCRIPT _ DIR ARCHIVES RESTORE`.
_RESTORE_SCRIPT = """\
dir=$1 archives=$2
rm -rf -- "$dir" && mkdir -p -m 700 -- "$dir" || { echo "ERROR: cannot create the node-local dir $dir"; exit 1; }
[ "$3" = 1 ] || exit 0
for name in $(ls -1 "$archives" 2>/dev/null | grep -xE '[0-9]{1,15}[.]tar[.]gz' | sort -rn); do
    if tar -xzf "$archives/$name" -C "$dir"; then
        echo "Restored the node-local dir from $archives/$name"
        exit 0
    fi
    echo "WARNING: cannot extract $archives/$name; trying an older archive"
    rm -rf -- "$dir" && mkdir -p -m 700 -- "$dir" || exit 1
done
echo "No earlier archive in $archives: the node-local dir starts empty"
"""

# Run on the driver's node as `bash -c SCRIPT _ DIR ARCHIVES JOB_ID TIMEOUT_SECONDS RESTORE`.
_ARCHIVE_SCRIPT = """\
# Ignored by tar too: the TERM Slurm sends a job at its end must not cut the archive short.
trap '' TERM
dir=$1 archives=$2 job=$3 limit=$4 restore=$5
[ -d "$dir" ] || { echo "WARNING: no node-local dir $dir to archive"; exit 0; }
tmp="$archives/.$job.tar.gz.tmp"
rc=2
if mkdir -p -- "$archives"; then
    tar -czf "$tmp" -C "$dir" . &
    pid=$!
    for _ in $(seq "$limit"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null
    wait "$pid"
    rc=$?
fi
# GNU tar exits 1 when a file changed while it was read: the archive is still complete.
if [ "$rc" -le 1 ] && mv -f -- "$tmp" "$archives/$job.tar.gz"; then
    echo "Archived the node-local dir to $archives/$job.tar.gz"
    # With restore each archive holds the whole dir: keep the newest two, the second as restore's fallback.
    if [ "$restore" = 1 ]; then
        for name in $(ls -1 "$archives" | grep -xE '[0-9]{1,15}[.]tar[.]gz' | sort -rn | tail -n +3); do
            rm -f -- "$archives/$name" && echo "Removed the superseded archive $archives/$name"
        done
    fi
else
    rm -f -- "$tmp"
    echo "WARNING: archiving the node-local dir failed (exit $rc); its files from this job are lost"
fi
rm -rf -- "$dir"
"""

_SETUP_TEMPLATE = """\
# --- Node-local dir, archived into the job dir when the driver step ends ---
_gym_nl_dir={host_dir}
_gym_nl_archives={archives}
_gym_nl_done=""
_gym_nl_deadline=""
_gym_nl_run() {{ {on_driver_node}bash -c "$1" _gym_node_local "${{@:2}}"; }}
_gym_nl_archive() {{
    [[ -z "$_gym_nl_done" ]] || return 0
    _gym_nl_done=1
    _gym_nl_run {archive_script} "$_gym_nl_dir" "$_gym_nl_archives" "$SLURM_JOB_ID" {archive_timeout} {restore}
}}
_gym_nl_run {restore_script} "$_gym_nl_dir" "$_gym_nl_archives" {restore}"""

_DEADLINE_TEMPLATE = """\
_gym_nl_job=$(SLURM_TIME_FORMAT=%s scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null) || _gym_nl_job=""
if [[ ! "$_gym_nl_job" =~ (^|[[:space:]])EndTime=([0-9]{{1,15}})([[:space:]]|$) ]]; then
    echo "WARNING: the job's end time is unknown: the driver is not stopped before the time limit, and the node-local dir is archived only if the driver step ends first."
elif (( BASH_REMATCH[2] - {stop} <= $(date +%s) )); then
    echo "WARNING: less than {stop}s remain: the driver is not stopped before the time limit, and the node-local dir is archived only if the driver step ends first."
else
    _gym_nl_deadline=$((BASH_REMATCH[2] - {stop}))
fi
_gym_nl_stop_for_walltime() {{
    _gym_nl_deadline=""
    # A finished evaluation writes nothing more there: archive and leave the exports their time.
    if [ -e "${{GYM_EVAL_SUCCEEDED_MARKER:-}}" ]; then
        echo "The time limit is {stop}s away and the evaluation has finished: archiving the node-local dir."
        _gym_nl_archive
        return
    fi
    echo "The time limit is {stop}s away: stopping the driver and archiving the node-local dir."
    kill -TERM "$_gym_driver_pid" 2>/dev/null
    for _i in $(seq 1 {stop_wait}); do kill -0 "$_gym_driver_pid" 2>/dev/null || break; sleep 1; done
    _gym_nl_archive
    echo "Waiting for the time limit: the job ends TIMEOUT, as a segment that ran to its limit does."
    while :; do sleep 60 & wait $!; done
}}"""

# Only where the batch shell runs on the driver's node: a step can no longer start once the job is ending.
_TERM_TRAP = """\
_gym_nl_on_term() {
    trap '' TERM
    echo "The job got TERM: archiving the node-local dir before Slurm kills the job."
    _gym_nl_archive
    exit 143
}
trap _gym_nl_on_term TERM"""

# Watch-loop line: once the deadline passes, stop the driver and archive (see _DEADLINE_TEMPLATE).
DEADLINE_CHECK = '[[ -z "$_gym_nl_deadline" ]] || (( $(date +%s) < _gym_nl_deadline )) || _gym_nl_stop_for_walltime'


def container_dir(remote_bench_dir: Path) -> str:
    """The directory's path inside the driver container: `<root>/<gym job id>/<benchmark>`."""
    return f"{_CONTAINER_ROOT}/{remote_bench_dir.parent.name}/{remote_bench_dir.name}"


def driver_mount(remote_bench_dir: Path) -> str:
    """The `--container-mounts` entry, already shell-quoted: the host dir expands when the driver step starts."""
    return f'"$_gym_nl_dir":{shlex.quote(container_dir(remote_bench_dir))}'


def driver_env_prefix(remote_bench_dir: Path) -> str:
    return f"{NODE_LOCAL_DIR_ENV}={shlex.quote(container_dir(remote_bench_dir))} "


def render_setup(config: NodeLocalConfig, remote_bench_dir: Path, driver_node_flags: str) -> str:
    """Batch-script lines that run before the driver: create (and restore) the dir and define the archive step.

    `driver_node_flags` are the driver step's own `srun` node flags; empty when the batch shell runs on its node.
    """
    on_driver_node = f"srun --overlap{driver_node_flags} " if driver_node_flags else ""
    blocks = [
        _SETUP_TEMPLATE.format(
            host_dir=f'{config.base_dir}/gym-node-local-"$SLURM_JOB_ID"',
            archives=shlex.quote(str(remote_bench_dir / ARCHIVES_DIRNAME)),
            on_driver_node=on_driver_node,
            archive_script=shlex.quote(_ARCHIVE_SCRIPT),
            archive_timeout=config.archive_timeout_seconds,
            restore_script=shlex.quote(_RESTORE_SCRIPT),
            restore=int(config.restore),
        )
    ]
    if config.stop_before_walltime_seconds:
        blocks.append(
            _DEADLINE_TEMPLATE.format(stop=config.stop_before_walltime_seconds, stop_wait=NODE_LOCAL_STOP_WAIT_SECONDS)
        )
    if not driver_node_flags:
        blocks.append(_TERM_TRAP)
    return "\n".join(blocks)


def render_after_driver(driver_node_flags: str) -> str:
    """Batch-script lines right after the driver step: archive, then restore the driver step's status in `$?`."""
    return "\n".join(
        [
            "_gym_nl_rc=$?",
            "_gym_nl_archive",
            *([] if driver_node_flags else ["trap - TERM"]),
            '(exit "$_gym_nl_rc")',
        ]
    )
