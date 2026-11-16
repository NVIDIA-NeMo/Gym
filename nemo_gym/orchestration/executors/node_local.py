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
extracts every earlier archive of the run into it, and mounts it into the driver container at a path that is the same
for every job of the run, so absolute paths the run saves stay valid in its later jobs. When the driver step ends, the
files the job added or changed are written to `<job dir>/node-local/<SLURM_JOB_ID>.<SLURM_RESTART_COUNT>.tar.gz` (one
file, renamed into place once complete) and the directory is removed. With exports, the archive is written as soon as
the evaluation has succeeded instead, and the exports wait for it, so an exporter of the job dir finds it. Each archive
holds one job attempt's files, so none is ever removed, and a restore extracts them all in the order the jobs ran.
"""

import shlex
from pathlib import Path

from nemo_gym.orchestration.api import NodeLocalConfig
from nemo_gym.orchestration.executors.resume_script import JOB_CHAIN_NAME
from nemo_gym.orchestration.executors.script_templates import NODE_LOCAL_ARCHIVE_DONE_MARKER_ENV


NODE_LOCAL_DIR_ENV = "GYM_NODE_LOCAL_DIR"
# The marker (NODE_LOCAL_ARCHIVE_DONE_MARKER_ENV) the job creates once its archive step has ended; the exports wait
# for it.
_ARCHIVE_DONE_MARKER_NAME = ".gym_node_local_archive_done"
# The exports' wait for that marker beyond archive_timeout_seconds: the watch polls every 5 s, then starts a step.
ARCHIVE_WAIT_MARGIN_SECONDS = 60
ARCHIVES_DIRNAME = "node-local"
# Inside the driver container: one path per run and benchmark, the same in every job of the run.
_CONTAINER_ROOT = "/tmp/gym-node-local"

# Run on the driver's node as `bash -c SCRIPT _ DIR ARCHIVES CHAIN JOB_ID RESTORE`. Archives are named
# `<job id>.<restart count>.tar.gz`: a requeued attempt keeps its job id. `DIR.stamp` marks the end of the restore.
_RESTORE_SCRIPT = """\
set -f
IFS=$'\\n'
dir=$1 archives=$2 chain=$3 job=$4 restore=$5
# Saved rows may name files in any archive of the run: a job that cannot restore them all does not start the driver.
fail() {
    echo "ERROR: $1; the driver does not start, since saved rows may name files in the run's node-local archives."
    rm -rf -- "$dir"
    exit 1
}
rm -rf -- "$dir" "$dir.stamp" "$dir.list" && mkdir -p -m 700 -- "$dir" || { echo "ERROR: cannot create the node-local dir $dir"; exit 1; }
names=""
# Only a missing dir means that no job has archived yet.
if [ -e "$archives" ] && ! names=$(ls -1A -- "$archives"); then
    [ "$restore" != 1 ] || fail "cannot list $archives"
fi
# Left by a job cut off mid-archive: no two jobs of a run are live at once.
for name in $names; do
    case $name in .*.tar.gz.tmp) rm -f -- "$archives/$name" && echo "Removed the partial archive $archives/$name" ;; esac
done
if [ "$restore" = 1 ]; then
    ids=""
    if [ -e "$chain" ]; then ids=$(cat -- "$chain") || fail "cannot read the job chain $chain"; fi
    found=$(printf '%s\\n' $names | grep -xE '[0-9]{1,15}[.][0-9]{1,6}[.]tar[.]gz' | sort -t. -k1,1n -k2,2n)
    restored=0
    # The job chain's order, then archives it does not name (a run without a chain) by job id; attempts in order.
    for id in $({ printf '%s\\n' $ids; printf '%s\\n' $found | cut -d. -f1; } | awk '/^[0-9]+$/ && !seen[$0]++'); do
        mine=$(printf '%s\\n' $found | grep "^$id[.]")
        [ -n "$mine" ] || [ "$id" = "$job" ] || echo "NOTE: job $id of the run left no node-local archive: any files it wrote there are lost"
        for name in $mine; do
            if ! tar -xzf "$archives/$name" -C "$dir"; then
                fail "cannot extract $archives/$name (repair or remove it, or set node_local.restore to false)"
            fi
            echo "Restored $archives/$name"
            restored=$((restored + 1))
        done
    done
    [ "$restored" -gt 0 ] || echo "No earlier archive in $archives: the node-local dir starts empty"
fi
# Extraction set the ctime of every restored file before this: the archive step takes what changes after it.
touch -- "$dir.stamp" || { rm -rf -- "$dir"; echo "ERROR: cannot create $dir.stamp"; exit 1; }
"""

# Run on the driver's node as `bash -c SCRIPT _ DIR ARCHIVES NAME TIMEOUT_SECONDS`.
_ARCHIVE_SCRIPT = """\
# Ignored by tar too: the TERM Slurm sends a job at its end must not cut the archive short.
trap '' TERM
dir=$1 archives=$2 name=$3 limit=$4
[ -d "$dir" ] || { echo "WARNING: no node-local dir $dir to archive"; exit 0; }
tmp="$archives/.$name.tar.gz.tmp"
stamp=""
[ ! -e "$dir.stamp" ] || stamp="$dir.stamp"
rc=2
if mkdir -p -- "$archives"; then
    # Only what this job added or changed (all of it without the stamp); a failed listing fails the archive.
    (
        cd -- "$dir" && find . ${stamp:+-cnewer "$stamp"} -print0 > "$dir.list" || exit 2
        # GNU tar fails on a listed file removed before it reads it, such as a temporary file; BSD tar warns.
        ignore=""
        if tar --version 2>/dev/null | grep -q 'GNU tar'; then ignore=--ignore-failed-read; fi
        exec tar --null --no-recursion $ignore -czf "$tmp" -T "$dir.list"
    ) &
    pid=$!
    for _ in $(seq "$limit"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null
    wait "$pid"
    rc=$?
fi
# GNU tar exits 1 when a file changed while it was read: the archive is still complete.
if [ "$rc" -le 1 ] && mv -f -- "$tmp" "$archives/$name.tar.gz"; then
    echo "Archived the node-local files this job added or changed to $archives/$name.tar.gz"
else
    rm -f -- "$tmp"
    echo "WARNING: archiving the node-local dir failed (exit $rc); its files from this job are lost"
fi
"""

# Run on the driver's node once the driver step has ended: exports may still read the dir until then.
_CLEANUP_SCRIPT = 'rm -rf -- "$1" "$1.stamp" "$1.list"'

_SETUP_TEMPLATE = """\
# --- Node-local dir, archived into the job dir when the driver step ends ---
_gym_nl_dir={host_dir}
_gym_nl_archives={archives}
_gym_nl_done=""
_gym_nl_run() {{ {on_driver_node}bash -c "$1" _gym_node_local "${{@:2}}"; }}
_gym_nl_archive() {{
    [[ -z "$_gym_nl_done" ]] || return 0
    _gym_nl_done=1
    _gym_nl_run {archive_script} "$_gym_nl_dir" "$_gym_nl_archives" "$SLURM_JOB_ID.${{SLURM_RESTART_COUNT:-0}}" {archive_timeout}{create_marker}
}}
_gym_nl_cleanup() {{ _gym_nl_run {cleanup_script} "$_gym_nl_dir"; }}
# Once the evaluation has succeeded, archive before the exports read the job dir.
_gym_nl_watch() {{
    if [[ -z "$_gym_nl_done" && -e "${{GYM_EVAL_SUCCEEDED_MARKER:-}}" ]]; then
        echo "The evaluation succeeded: archiving the node-local dir before the exports read the job dir."
        _gym_nl_archive
    fi
}}
_gym_nl_run {restore_script} "$_gym_nl_dir" "$_gym_nl_archives" {chain} "$SLURM_JOB_ID" {restore} || exit $?"""

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


def _archive_done_marker(remote_bench_dir: Path) -> str:
    return f'{shlex.quote(str(remote_bench_dir / _ARCHIVE_DONE_MARKER_NAME))}."$SLURM_JOB_ID"'


def driver_env_prefix(remote_bench_dir: Path, *, exports: bool) -> str:
    """The driver's env: the dir, and with exports the marker that says the archive step has ended."""
    marker = f"{NODE_LOCAL_ARCHIVE_DONE_MARKER_ENV}={_archive_done_marker(remote_bench_dir)} " if exports else ""
    return f"{NODE_LOCAL_DIR_ENV}={shlex.quote(container_dir(remote_bench_dir))} {marker}"


def archive_wait_seconds(config: NodeLocalConfig) -> int:
    """How long the exports wait for the marker before they run anyway."""
    return config.archive_timeout_seconds + ARCHIVE_WAIT_MARGIN_SECONDS


def render_setup(config: NodeLocalConfig, remote_bench_dir: Path, driver_node_flags: str, *, exports: bool) -> str:
    """Batch-script lines that run before the driver: create (and restore) the dir and define the archive step.

    `driver_node_flags` are the driver step's own `srun` node flags; empty when the batch shell runs on its node.
    With `exports`, every archive step (also a failed one) then creates the marker the exports wait for.
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
            chain=shlex.quote(str(remote_bench_dir / JOB_CHAIN_NAME)),
            restore=int(config.restore),
            create_marker=f"\n    : > {_archive_done_marker(remote_bench_dir)}" if exports else "",
            cleanup_script=shlex.quote(_CLEANUP_SCRIPT),
        )
    ]
    if not driver_node_flags:
        blocks.append(_TERM_TRAP)
    return "\n".join(blocks)


def render_after_driver(remote_bench_dir: Path, driver_node_flags: str, *, exports: bool) -> str:
    """Batch-script lines right after the driver step: archive, remove the dir, restore the step's status in `$?`.

    With `exports` the exports have run by then: the marker goes too, should their wait have timed out before it came.
    """
    return "\n".join(
        [
            "_gym_nl_rc=$?",
            "_gym_nl_archive",
            "_gym_nl_cleanup",
            *([f"rm -f -- {_archive_done_marker(remote_bench_dir)}"] if exports else []),
            *([] if driver_node_flags else ["trap - TERM"]),
            '(exit "$_gym_nl_rc")',
        ]
    )
