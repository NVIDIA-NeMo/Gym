# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit one batch job per model/language, gated by the model's Hindi pilot."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--pilots-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Resubmit only terminal failed jobs")
    parser.add_argument("--models", nargs="+", help="Limit submission or retry to these manifest model keys")
    return parser.parse_args()


def slurm_options(config: dict[str, str]) -> list[str]:
    """Translate the supported manifest Slurm fields to sbatch arguments."""
    options = []
    for field, flag in (
        ("account", "account"),
        ("partition", "partition"),
        ("constraint", "constraint"),
        ("time", "time"),
    ):
        if config.get(field):
            options.append(f"--{flag}={config[field]}")
    return options


def accounting_states(jobs: dict[str, dict]) -> dict[str, str]:
    """Return normalized Slurm states for every recorded job ID."""
    job_ids = ",".join(job["job_id"] for job in jobs.values())
    if not job_ids:
        return {}
    output = subprocess.check_output(["sacct", "-n", "-X", "-j", job_ids, "--format=JobID,State", "-P"], text=True)
    states = {}
    for line in output.splitlines():
        if "|" not in line:
            continue
        job_id, state = line.split("|", maxsplit=1)
        states[job_id] = state.split()[0].rstrip("+")
    return states


def main() -> None:
    args = parse_args()
    run = args.run.expanduser().resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    (run / "logs").mkdir(exist_ok=True)
    with (run / ".launch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = run / "jobs.json"
        jobs = json.loads(ledger.read_text()) if ledger.exists() else {}
        states = accounting_states(jobs) if args.retry_failed else {}
        active = []
        if not args.dry_run:
            active = subprocess.check_output(
                ["squeue", "-h", "-u", getpass.getuser(), "-o", "%j"], text=True
            ).splitlines()
        for model, spec in manifest["models"].items():
            if args.models and model not in args.models:
                continue
            languages = ["hi"]
            if not args.pilots_only:
                languages.extend(language for language in manifest["languages"] if language != "hi")
            for language in languages:
                key = f"{model}/{language}"
                if (run / "results" / model / language / "DONE").exists():
                    continue
                previous = jobs.get(key)
                if previous and (not args.retry_failed or states.get(previous["job_id"]) not in FAILURE_STATES):
                    continue
                if not previous and args.retry_failed:
                    continue
                name = f"gpqa5pre-{model}-{language}"
                if name in active:
                    raise RuntimeError(f"Unrecorded live job {name}; inspect before submitting again")
                command = [
                    "sbatch",
                    "--parsable",
                    f"--job-name={name}",
                    "--nodes=1",
                    "--ntasks=1",
                    "--cpus-per-task=32",
                    f"--gpus-per-node={spec['gpus']}",
                    f"--mem={spec['memory_gb']}G",
                    f"--output={run}/logs/{model}-{language}-%j.log",
                    *slurm_options(manifest.get("slurm", {})),
                ]
                if language != "hi" and not (run / "results" / model / "hi" / "DONE").exists():
                    pilot = jobs.get(f"{model}/hi", {}).get("job_id", "PILOT" if args.dry_run else None)
                    if pilot is None:
                        raise RuntimeError(f"Missing pilot for {model}")
                    command.extend([f"--dependency=afterok:{pilot}", "--kill-on-invalid-dep=yes"])
                command.extend([str(run / "source" / "job.sbatch"), str(run), model, language])
                if args.dry_run:
                    print(json.dumps(command))
                    continue
                job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
                if not job_id.isdigit():
                    raise RuntimeError(f"Unexpected sbatch output: {job_id}")
                history = []
                if previous:
                    history = previous.get("history", []) + [
                        {field: value for field, value in previous.items() if field != "history"}
                    ]
                jobs[key] = {
                    "job_id": job_id,
                    "model": model,
                    "language": language,
                    "command": command,
                    "submitted_utc": datetime.now(timezone.utc).isoformat(),
                    "history": history,
                }
                temporary = ledger.with_suffix(".tmp")
                temporary.write_text(json.dumps(jobs, indent=2) + "\n")
                temporary.replace(ledger)
                print(key, job_id, flush=True)


if __name__ == "__main__":
    main()
