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

"""The Slurm commands the executor's parts share: sbatch and its reply, partition limits, live checks."""

import os
import re
import shlex
from pathlib import Path
from typing import Any

from nemo_gym.eval.recovery.decision import (
    LIVE_JOB_STATES,
)
from nemo_gym.orchestration.api import (
    SlurmComputeConfig,
    SubmitConfig,
)
from nemo_gym.orchestration.executors.connection import (
    Connection,
)
from nemo_gym.orchestration.executors.otel import (
    resolve_token,
    validate_destination,
)
from nemo_gym.orchestration.executors.resume_script import (
    RESUBMIT_LOCK_NAME,
    _walltime_to_seconds,
)


def _slurm_time_limit_seconds(text: str) -> int | None:
    """A partition time limit as sinfo prints it (``%l``) in seconds; None for ``UNLIMITED`` or ``infinite``.

    Takes Slurm's duration spellings (``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``) and
    raises ValueError for anything else, such as ``n/a``.
    """
    value = text.strip()
    if value.upper() in ("UNLIMITED", "INFINITE"):
        return None
    return _walltime_to_seconds(value)


def _partition_time_limit(conn: Connection, compute: SlurmComputeConfig) -> dict[str, Any]:
    """The partition a job of ``compute`` queues on and its MaxTime, read with sinfo over ``conn``.

    The partitions are the node pools' ones, else ``extra_args.partition`` (a list ``a,b`` allowed), else the
    cluster's default one. Every one must be listed, and the lowest limit applies: ``partition`` names it,
    ``max_time`` is sinfo's text and ``max_time_s`` its seconds (None when unlimited); ``error`` says why they are
    unknown.
    """
    pools = sorted({pool.partition for pool in compute.node_pools.values()})
    partition = ",".join(pools) if pools else compute.extra_args.get("partition")
    limit: dict[str, Any] = {"partition": partition, "max_time": None, "max_time_s": None, "error": None}
    try:
        if partition is not None:
            names = str(partition).split(",")
            rows = [
                line.split()
                for line in conn.run([f"sinfo -h -p {shlex.quote(str(partition))} -o '%P %l'"]).splitlines()
            ]
            listed = {row[0].rstrip("*"): row[1] for row in rows if len(row) == 2}
            missing = [name for name in names if name not in listed]
            if missing:
                raise ValueError(f"sinfo lists no partition {', '.join(missing)}")
            parsed = [(_slurm_time_limit_seconds(listed[name]), listed[name], name) for name in names]
        else:
            rows = [line.split() for line in conn.run(["sinfo -h -o '%P %l'"]).splitlines()]
            default = [row for row in rows if len(row) == 2 and row[0].endswith("*")]
            if not default:
                raise ValueError("sinfo names no default partition")
            name, text = default[0][0].rstrip("*"), default[0][1]
            parsed = [(_slurm_time_limit_seconds(text), text, name)]
    except (RuntimeError, OSError, ValueError) as error:
        return limit | {"error": str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__}
    bounded = [item for item in parsed if item[0] is not None]
    seconds, text, name = min(bounded) if bounded else (None, parsed[0][1], parsed[0][2])
    return limit | {"partition": name, "max_time": text, "max_time_s": seconds}


# Each sbatch reports its own result on a line that names its benchmark, so a
# failure cannot shift the benchmarks after it onto the wrong job ids.
_MARKER = "__GYM_JOB:"

# Benchmark names are interpolated into the marker line, so they must not carry
# its delimiter or anything the shell would act on.
_VALID_BENCHMARK_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Real `sbatch --parsable` output is a bare "<id>" or a federated "<id>;<cluster>".
# stderr is merged onto the same line (see `_sbatch_command`), and a *successful*
# sbatch can still print a warning (job_submit plugin notices, QOS/ntasks
# adjustments are routine on production Slurm) — so only the payload's last
# whitespace-delimited token is checked against this, not the whole line.
_JOB_ID_RE = re.compile(r"^\d+(;\S+)?$")

# Held by `gym eval resubmit` from its scheduler check to its sbatch, so two resubmits cannot both queue a job.
_RESUBMIT_LOCK = RESUBMIT_LOCK_NAME


def _chain_live_check(chain: list[str], live_error: str) -> list[str]:
    """Shell lines that fail unless the scheduler confirms that no job of `chain` waits or runs.

    Slurm drops a finished job from the controller after MinJobAge, and `squeue -j` then exits 1 with "Invalid job
    id specified" once it knows none of the ids. That is no answer about the jobs, so sacct, which keeps finished
    jobs, decides: a job whose latest incarnation (highest Restarts; sacct -D lists every requeue) is in one of the
    `LIVE_JOB_STATES` is live, and one it does not know is not. Any other squeue failure, or a failing sacct, leaves
    the scheduler unknown.
    """
    ids = shlex.quote(",".join(chain))
    live_states = "|".join(sorted(LIVE_JOB_STATES))
    return [
        f"if _gym_queue=$(squeue --all --noheader -j {ids} --format=%A 2>&1); then "
        "live=$(printf '%s\n' \"$_gym_queue\" | grep -E '^[0-9]' || true); "
        "elif printf '%s' \"$_gym_queue\" | grep -q 'Invalid job id specified'; then "
        f"_gym_accounting=$(sacct -D -n -P -X -j {ids} --format=JobIDRaw,State,Restarts) || fail 'scheduler unknown'; "
        "live=$(printf '%s\n' \"$_gym_accounting\" | awk -F'|' "
        '\'NF >= 2 {split($2, s, " "); state = s[1]; sub(/[+]$/, "", state); '
        "if (!($1 in restarts) || $3 + 0 >= restarts[$1]) {restarts[$1] = $3 + 0; latest[$1] = state}} "
        f"END {{for (id in latest) if (latest[id] ~ /^({live_states})$/) print id}}'); "
        "else fail 'scheduler unknown'; fi",
        f'[ -z "$live" ] || fail "{live_error}"',
    ]


def _sbatch_command(benchmark: str, script: Path) -> str:
    """One `sbatch` that reports its own benchmark, exit status and output.

    `rc` is captured immediately: `$?` after the `tr` pipeline would be *tr's*
    status, which is zero however badly sbatch failed. `tr` flattens sbatch's
    multi-line error messages so the whole result stays on one marker line.
    """
    if not _VALID_BENCHMARK_NAME.match(benchmark):
        raise ValueError(
            f"Invalid benchmark name {benchmark!r}: names must match {_VALID_BENCHMARK_NAME.pattern} "
            "so they can be reported back from the submit script."
        )
    return (
        f"out=$(sbatch --parsable {shlex.quote(str(script))} 2>&1); rc=$?; "
        "out=$(echo \"$out\" | tr '\\n' ' '); "
        f'echo "{_MARKER}{benchmark}:$rc:$out"'
    )


def _parse_sbatch_results(output: str) -> dict[str, tuple[str | None, str | None]]:
    """Benchmark name to `(job_id, error)`, exactly one of which is set."""
    results: dict[str, tuple[str | None, str | None]] = {}
    for line in output.splitlines():
        if not line.startswith(_MARKER):
            continue
        benchmark, _, rest = line[len(_MARKER) :].partition(":")
        status, _, payload = rest.partition(":")
        payload = payload.strip()
        if status == "0":
            tokens = payload.split()
            candidate = tokens[-1] if tokens else ""
            if _JOB_ID_RE.match(candidate):
                # A federated sbatch answers "jobid;cluster"; the ledger wants the id.
                results[benchmark] = (candidate.split(";")[0], None)
            else:
                # Exit 0 but the last token isn't an id: no output at all, or a
                # warning with nothing that looks like a job id after it. Either
                # way there is no id to trust, so this is a failure, not a
                # success with garbage (or an empty string) in job_id.
                results[benchmark] = (None, payload or "sbatch exited 0 with no output")
        else:
            results[benchmark] = (None, payload or f"sbatch exited {status}")
    return results


def _submission_env_exports(compute: SlurmComputeConfig) -> list[str]:
    """Forward explicitly selected values over SSH stdin, never into artifacts."""
    missing = [name for name in compute.forward_env if name not in os.environ]
    if missing:
        raise ValueError(f"compute.forward_env variables are unset: {', '.join(missing)}")
    return [f"export {name}={shlex.quote(os.environ[name])}" for name in sorted(set(compute.forward_env))]


def _submission_exports(config: SubmitConfig) -> list[str]:
    """What `run()` exports ahead of sbatch: the forwarded values, then the telemetry token when it is on."""
    exports = _submission_env_exports(next(iter(config.compute.values())))
    token = resolve_token(config)
    validate_destination(config)
    if token is not None:
        exports.append(f"export {config.otel.token_env}={shlex.quote(token)}")
    return exports
