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
removed. With exports, the archive is written as soon as the evaluation has succeeded instead, and the exports wait for
it, so an exporter of the job dir finds it. With restore each archive holds the whole directory, so only it and the
archive it was restored from are kept.
"""

import shlex
from pathlib import Path

from nemo_gym.orchestration.api import NodeLocalConfig
from nemo_gym.orchestration.executors.script_templates import NODE_LOCAL_ARCHIVED_ENV


NODE_LOCAL_DIR_ENV = "GYM_NODE_LOCAL_DIR"
# The barrier file (NODE_LOCAL_ARCHIVED_ENV) the job writes once it has archived; the exports wait for it.
_ARCHIVED_NAME = ".gym_node_local_archived"
# The exports' wait for that barrier beyond archive_timeout_seconds: the watch polls every 5 s, then starts a step.
ARCHIVE_WAIT_MARGIN_SECONDS = 60
ARCHIVES_DIRNAME = "node-local"
# Inside the driver container: one path per run and benchmark, the same in every job of the run.
_CONTAINER_ROOT = "/tmp/gym-node-local"

# Run on the driver's node as `bash -c SCRIPT _ DIR ARCHIVES RESTORE`. `DIR.restored-from` names the archive it
# extracted, for the archive step's pruning.
_RESTORE_SCRIPT = """\
dir=$1 archives=$2
rm -f -- "$dir.restored-from"
rm -rf -- "$dir" && mkdir -p -m 700 -- "$dir" || { echo "ERROR: cannot create the node-local dir $dir"; exit 1; }
[ "$3" = 1 ] || exit 0
found=0
for name in $(ls -1 "$archives" 2>/dev/null | grep -xE '[0-9]{1,15}[.]tar[.]gz' | sort -rn); do
    found=1
    if tar -xzf "$archives/$name" -C "$dir"; then
        echo "Restored the node-local dir from $archives/$name"
        echo "$name" > "$dir.restored-from"
        exit 0
    fi
    echo "WARNING: cannot extract $archives/$name; trying an older archive"
    rm -rf -- "$dir" && mkdir -p -m 700 -- "$dir" || exit 1
done
# Saved results may name files in those archives: an empty dir would publish a run without them.
if [ "$found" = 1 ]; then
    echo "ERROR: no node-local archive in $archives could be restored; the driver does not start"
    exit 1
fi
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
    # With restore each archive holds the whole dir: of the older ones keep only the one this job restored from,
    # a known-good fallback (without a record, the newest older one).
    if [ "$restore" = 1 ]; then
        keep=$(cat "$dir.restored-from" 2>/dev/null)
        for name in $(ls -1 "$archives" | grep -xE '[0-9]{1,15}[.]tar[.]gz' | sort -rn); do
            (( 10#${name%%.*} < 10#$job )) || continue
            [ -n "$keep" ] || keep=$name
            [ "$name" != "$keep" ] || continue
            rm -f -- "$archives/$name" && echo "Removed the superseded archive $archives/$name"
        done
    fi
else
    rm -f -- "$tmp"
    echo "WARNING: archiving the node-local dir failed (exit $rc); its files from this job are lost"
fi
"""

# Run on the driver's node once the driver step has ended: exports may still read the dir until then.
_CLEANUP_SCRIPT = 'rm -rf -- "$1" "$1.restored-from"'

_SETUP_TEMPLATE = """\
# --- Node-local dir, archived into the job dir when the driver step ends ---
_gym_nl_dir={host_dir}
_gym_nl_archives={archives}
_gym_nl_done=""
_gym_nl_run() {{ {on_driver_node}bash -c "$1" _gym_node_local "${{@:2}}"; }}
_gym_nl_archive() {{
    [[ -z "$_gym_nl_done" ]] || return 0
    _gym_nl_done=1
    _gym_nl_run {archive_script} "$_gym_nl_dir" "$_gym_nl_archives" "$SLURM_JOB_ID" {archive_timeout} {restore}{release}
}}
_gym_nl_cleanup() {{ _gym_nl_run {cleanup_script} "$_gym_nl_dir"; }}
# Once the evaluation has succeeded, archive before the exports read the job dir.
_gym_nl_watch() {{
    if [[ -z "$_gym_nl_done" && -e "${{GYM_EVAL_SUCCEEDED_MARKER:-}}" ]]; then
        echo "The evaluation succeeded: archiving the node-local dir before the exports read the job dir."
        _gym_nl_archive
    fi
}}
_gym_nl_run {restore_script} "$_gym_nl_dir" "$_gym_nl_archives" {restore} || exit $?"""

# Only where the batch shell runs on the driver's node: a step can no longer start once the job is ending.
_TERM_TRAP = """\
_gym_nl_on_term() {
    trap '' TERM
    if [[ -z "$_gym_nl_done" ]]; then
        echo "The job got TERM: archiving the node-local dir before Slurm kills the job."
        _gym_nl_archive
    fi
    _gym_nl_cleanup
    exit 143
}
trap _gym_nl_on_term TERM"""

# Watch-loop line: archive once the evaluation succeeded. It also keeps the driver in the background, so a TERM to
# the batch shell runs its trap at once.
WATCH_CHECK = "_gym_nl_watch"


def container_dir(remote_bench_dir: Path) -> str:
    """The directory's path inside the driver container: `<root>/<gym job id>/<benchmark>`."""
    return f"{_CONTAINER_ROOT}/{remote_bench_dir.parent.name}/{remote_bench_dir.name}"


def driver_mount(remote_bench_dir: Path) -> str:
    """The `--container-mounts` entry, already shell-quoted: the host dir expands when the driver step starts."""
    return f'"$_gym_nl_dir":{shlex.quote(container_dir(remote_bench_dir))}'


def _archived_barrier(remote_bench_dir: Path) -> str:
    return f'{shlex.quote(str(remote_bench_dir / _ARCHIVED_NAME))}."$SLURM_JOB_ID"'


def driver_env_prefix(remote_bench_dir: Path, *, exports: bool) -> str:
    """The driver's env: the dir, and with exports the barrier file that says the dir has been archived."""
    barrier = f"{NODE_LOCAL_ARCHIVED_ENV}={_archived_barrier(remote_bench_dir)} " if exports else ""
    return f"{NODE_LOCAL_DIR_ENV}={shlex.quote(container_dir(remote_bench_dir))} {barrier}"


def archive_wait_seconds(config: NodeLocalConfig) -> int:
    """How long the exports wait for the barrier before they run anyway."""
    return config.archive_timeout_seconds + ARCHIVE_WAIT_MARGIN_SECONDS


def render_setup(config: NodeLocalConfig, remote_bench_dir: Path, driver_node_flags: str, *, exports: bool) -> str:
    """Batch-script lines that run before the driver: create (and restore) the dir and define the archive step.

    `driver_node_flags` are the driver step's own `srun` node flags; empty when the batch shell runs on its node.
    With `exports`, every archive (or failed one) also releases the exports' barrier.
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
            release=f"\n    : > {_archived_barrier(remote_bench_dir)}" if exports else "",
            cleanup_script=shlex.quote(_CLEANUP_SCRIPT),
        )
    ]
    if not driver_node_flags:
        blocks.append(_TERM_TRAP)
    return "\n".join(blocks)


def render_after_driver(driver_node_flags: str) -> str:
    """Batch-script lines right after the driver step: archive, remove the dir, restore the step's status in `$?`."""
    return "\n".join(
        [
            "_gym_nl_rc=$?",
            "_gym_nl_archive",
            "_gym_nl_cleanup",
            *([] if driver_node_flags else ["trap - TERM"]),
            '(exit "$_gym_nl_rc")',
        ]
    )
