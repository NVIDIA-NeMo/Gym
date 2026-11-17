# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Generate a standalone Slurm job for post-evaluation exports."""

import re
import shlex
from pathlib import Path


def render_export_job(
    *,
    bench_dir: Path,
    partition: str,
    account: str,
    walltime: str,
    export_command: str,
    extra_args: dict[str, str] | None = None,
) -> str:
    """Render a CPU-only Slurm export job without submitting it."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", partition):
        raise ValueError("Invalid Slurm export partition")

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", account):
        raise ValueError("Invalid Slurm account")

    if not re.fullmatch(r"[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]", walltime):
        raise ValueError("Invalid Slurm export walltime")

    extra_directives = []
    for key, value in (extra_args or {}).items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", key) or not re.fullmatch(r"[^\r\n\x00]*", value):
            raise ValueError("Invalid Slurm export extra argument")
        allowed_extra_args = {
            "constraint",
            "qos",
            "reservation",
            "mail-type",
            "mail-user",
            "comment",
        }
        if key not in allowed_extra_args:
            raise ValueError("Unsupported Slurm export extra argument")
        extra_directives.append(f"#SBATCH --{key}={value}")

    if not export_command.strip():
        raise ValueError("Export command cannot be empty")

    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --account={account}",
            f"#SBATCH --time={walltime}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=1",
            *extra_directives,
            "set -euo pipefail",
            f"cd {shlex.quote(str(bench_dir))}",
            export_command,
            "",
        ]
    )


def render_export_submission(*, bench_dir: Path, job_script: str) -> str:
    """Submit a prepared CPU job once, without changing the evaluation exit status.

    A lock serializes concurrent completion paths. A failed submission remains
    retryable; only a successfully parsed Slurm job id is persisted.
    """
    import base64

    bench = shlex.quote(str(bench_dir))
    payload = base64.b64encode(job_script.encode()).decode("ascii")
    return "\n".join(
        [
            f"_gym_export_dir={bench}",
            'mkdir -p "$_gym_export_dir" || true',
            "(",
            '  _gym_lock="$_gym_export_dir/.exports.submit.lock"',
            '  if ! mkdir "$_gym_lock" 2>/dev/null; then exit 0; fi',
            "  trap 'rmdir \"$_gym_lock\" 2>/dev/null || true' EXIT",
            '  if [[ -s "$_gym_export_dir/exports.job_id" ]]; then exit 0; fi',
            f'  printf %s {shlex.quote(payload)} | base64 -d > "$_gym_export_dir/.exports.sbatch.tmp.$$" || exit 0',
            '  mv "$_gym_export_dir/.exports.sbatch.tmp.$$" "$_gym_export_dir/.exports.sbatch" || exit 0',
            '  _gym_export_id=$(sbatch --parsable "$_gym_export_dir/.exports.sbatch") || { echo "WARNING: CPU export sbatch failed" >&2; exit 0; }',
            "  _gym_export_id=${_gym_export_id%%;*}",
            '  if [[ "$_gym_export_id" =~ ^[0-9]+$ ]]; then',
            '    printf "%s\\n" "$_gym_export_id" > "$_gym_export_dir/exports.job_id" || true',
            "  else",
            '    echo "WARNING: invalid CPU export sbatch job id" >&2',
            "  fi",
            ") || true",
        ]
    )
