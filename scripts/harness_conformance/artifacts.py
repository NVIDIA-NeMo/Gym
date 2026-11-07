# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free downloads shared by the P0 reporter and read-only docs job."""

import json
import subprocess
from pathlib import Path


def _list_items(endpoint: str, key: str) -> list[dict]:
    listing = subprocess.check_output(["gh", "api", "--paginate", endpoint, "--jq", f".{key}[] | @json"], text=True)
    return [json.loads(line) for line in listing.splitlines()]


def _job_artifact(harness: str, jobs: list[dict], files: list[dict], *, run_id: str, attempt: str) -> dict | None:
    jobs = [job for job in jobs if job["name"].rsplit(" / ", 1)[-1] == f"{harness} conformance probes"]
    if len(jobs) != 1 or jobs[0]["status"] != "completed":
        return None
    job = jobs[0]
    if not job.get("started_at") or not job.get("completed_at"):
        return None
    prefix = f"harness-conformance-{harness}-{run_id}-"
    candidates = []
    for artifact in files:
        suffix = artifact["name"].removeprefix(prefix)
        if (
            artifact["name"].startswith(prefix)
            and suffix.isdecimal()
            and 1 <= int(suffix) <= int(attempt)
            and not artifact["expired"]
            # GitHub retains successful jobs on partial reruns, copying their
            # original times but assigning the new run_attempt. Match the actual
            # execution so a missing upload cannot reuse a previous run's result.
            and job["started_at"] <= artifact["created_at"] <= job["completed_at"]
        ):
            candidates.append(artifact)
    return max(candidates, key=lambda artifact: int(artifact["name"].removeprefix(prefix)), default=None)


def download_artifacts(harnesses: list[str], *, repo: str, run_id: str, attempt: str, output: Path) -> dict[str, int]:
    """Download each selected job's evidence and return its artifact ID for links."""
    if not harnesses:
        return {}
    endpoint = f"repos/{repo}/actions/runs/{run_id}"
    try:
        jobs = _list_items(f"{endpoint}/attempts/{attempt}/jobs?per_page=100", "jobs")
        files = _list_items(f"{endpoint}/artifacts?per_page=100", "artifacts")
    except (subprocess.CalledProcessError, ValueError):
        print("::warning::Could not list conformance jobs and artifacts; measurements are unavailable.")
        jobs, files = [], []
    downloaded = {}
    for harness in harnesses:
        result = output / harness / "probes" / "ci-result.json"
        result.unlink(missing_ok=True)
        try:
            artifact = _job_artifact(harness, jobs, files, run_id=run_id, attempt=attempt)
        except (KeyError, TypeError, ValueError):
            artifact = None
        if artifact is None:
            print(f"::warning::No evidence matches the selected {harness} job; its measurement is unavailable.")
            continue
        try:
            subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    run_id,
                    "--repo",
                    repo,
                    "--name",
                    artifact["name"],
                    "--dir",
                    str(output / harness),
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            # Do not let partially extracted or older files qualify this attempt.
            result.unlink(missing_ok=True)
            print(f"::warning::Could not download {harness} evidence; its measurement is unavailable.")
        else:
            downloaded[harness] = artifact["id"]
    return downloaded
