# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard-library helpers for CI check suites that run in GitHub Actions.

Workflows import this module with bare `python3` before Gym is installed, so it must not import `nemo_gym` or
third-party packages.

`download_artifacts` and `post_check_run` run the `gh` CLI, which reads `GH_TOKEN`. The
token needs `actions: read` to download and list artifacts, and `checks: write` to publish a check run.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, get_args


CompareMode = Literal["branch", "commit"]
COMPARE_BRANCH: CompareMode = "branch"
COMPARE_COMMIT: CompareMode = "commit"
COMPARE_MODES = get_args(CompareMode)


def _merge_base(base: str, *, root: Path) -> str:
    """The merge base with `base`; a CI checkout has other branches only as `origin/<name>`."""
    command = ["git", "merge-base", "--"]
    try:
        output = subprocess.check_output([*command, base, "HEAD"], cwd=root, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        output = subprocess.check_output(
            [*command, f"origin/{base}", "HEAD"], cwd=root, text=True, stderr=subprocess.PIPE
        )
    return output.strip()


def changed_paths(base: str, *, root: Path, subject: str, compare: CompareMode = COMPARE_BRANCH) -> list[str] | None:
    """List the paths changed from `base` to HEAD, including deleted and renamed ones.

    With `branch`, `base` is a branch and the whole branch since its merge base is compared, as for a PR. For a
    stacked PR `base` is the parent branch it targets, so only that layer is covered. With `commit`, `base` is a
    commit, such as the one before a push, and is compared directly with HEAD, which stays correct when history is
    rewritten.

    Returns None when there is no base or the diff is unavailable; the caller then runs all `subject`.
    """
    if not base:
        return None
    try:
        ancestor = _merge_base(base, root=root) if compare == COMPARE_BRANCH else base
        paths = subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", "-z", ancestor, "HEAD", "--"], cwd=root
        ).decode("utf-8", errors="replace")
    except subprocess.CalledProcessError:
        print(f"::warning::Could not determine changed files; running all {subject}.")
        return None
    return [path for path in paths.split("\0") if path]


CI_RESULT = "ci-result.json"
STATUSES = ("passed", "failed", "unavailable")


def read_ci_result(path: Path, *, subject: str, sha: str, subject_key: str = "subject") -> dict | None:
    """Read one measurement only when it belongs to this subject and tested commit."""
    try:
        result = json.loads(path.read_text())
        if result[subject_key] == subject and result["sha"] == sha and result["status"] in STATUSES:
            return result
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def tested_sha() -> str:
    """The commit the checks ran on: `TESTED_SHA` when a PR mirror run tests its merge commit, else `GITHUB_SHA`."""
    return os.environ.get("TESTED_SHA") or os.environ.get("GITHUB_SHA", "local")


def write_ci_result(output: Path, *, subject: str, status: str, subject_key: str = "subject") -> None:
    """Record the status of one subject for the tested commit."""
    output.mkdir(parents=True, exist_ok=True)
    (output / CI_RESULT).write_text(json.dumps({subject_key: subject, "sha": tested_sha(), "status": status}) + "\n")


def _list_items(endpoint: str, key: str) -> list[dict]:
    listing = subprocess.check_output(["gh", "api", "--paginate", endpoint, "--jq", f".{key}[] | @json"], text=True)
    return [json.loads(line) for line in listing.splitlines()]


def _job_artifact(
    subject: str,
    job_name: str,
    jobs: list[dict],
    files: list[dict],
    *,
    artifact_prefix: str,
    run_id: str,
    attempt: str,
) -> dict | None:
    jobs = [job for job in jobs if job["name"].rsplit(" / ", 1)[-1] == job_name]
    if len(jobs) != 1 or jobs[0]["status"] != "completed":
        return None
    job = jobs[0]
    if not job.get("started_at") or not job.get("completed_at"):
        return None
    prefix = f"{artifact_prefix}-{subject}-{run_id}-"
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


def download_artifacts(
    subjects: list[str],
    *,
    artifact_prefix: str,
    job_names: dict[str, str],
    repo: str,
    run_id: str,
    attempt: str,
    output: Path,
    result_file: str,
) -> dict[str, int]:
    """Download each selected job's evidence and return its artifact ID for links.

    `job_names` maps each artifact subject to its job name without reusable-workflow prefixes.
    Match the selected job's execution window, including retained jobs from earlier attempts.
    """
    if not subjects:
        return {}
    endpoint = f"repos/{repo}/actions/runs/{run_id}"
    try:
        jobs = _list_items(f"{endpoint}/attempts/{attempt}/jobs?per_page=100", "jobs")
        files = _list_items(f"{endpoint}/artifacts?per_page=100", "artifacts")
    except (subprocess.CalledProcessError, ValueError):
        print("::warning::Could not list check jobs and artifacts; measurements are unavailable.")
        jobs, files = [], []
    downloaded = {}
    for subject in subjects:
        result = output / subject / result_file
        result.unlink(missing_ok=True)
        try:
            artifact = _job_artifact(
                subject,
                job_names[subject],
                jobs,
                files,
                artifact_prefix=artifact_prefix,
                run_id=run_id,
                attempt=attempt,
            )
        except (KeyError, TypeError, ValueError):
            artifact = None
        if artifact is None:
            print(f"::warning::No evidence matches the selected {subject} job; its measurement is unavailable.")
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
                    str(output / subject),
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            # Do not let partially extracted or older files qualify this attempt.
            result.unlink(missing_ok=True)
            print(f"::warning::Could not download {subject} evidence; its measurement is unavailable.")
        else:
            downloaded[subject] = artifact["id"]
    return downloaded


class CheckOutput(TypedDict):
    title: str
    summary: str


class CheckRun(TypedDict):
    name: str
    head_sha: str
    status: str
    conclusion: str
    output: CheckOutput
    details_url: NotRequired[str]
    external_id: NotRequired[str]


def markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table with no trailing newline."""
    lines = [f"| {' | '.join(columns)} |", "|" + "---|" * len(columns)]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def artifact_download_urls(artifact_ids: dict[str, int], *, repo: str, run_id: str, server_url: str) -> dict[str, str]:
    """Link exactly the uploads selected by `download_artifacts`, including retained jobs."""
    return {
        subject: f"{server_url}/{repo}/actions/runs/{run_id}/artifacts/{artifact_id}"
        for subject, artifact_id in artifact_ids.items()
    }


def post_check_run(payload: CheckRun, *, repo: str) -> None:
    """Publish the detailed check before the workflow job reflects its result."""
    subprocess.run(
        ["gh", "api", "--method", "POST", f"repos/{repo}/check-runs", "--input", "-"],
        input=json.dumps(payload),
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def job_exit_code(payload: CheckRun, *, probe_result: str) -> int:
    """Fail the reporting job when the check or any probe job did not succeed."""
    if payload["conclusion"] != "success" or probe_result not in ("success", "skipped"):
        print(
            f"::error title={payload['name']}::{payload['output']['title']}. "
            f"Probe jobs: {probe_result}. See the job summary for failures, rerun commands, and artifact links."
        )
        return 1
    return 0


def table_cell(text: str) -> str:
    """Collapse whitespace and escape pipes so external text cannot break a Markdown table row."""
    return " ".join(text.split()).replace("|", "&#124;")


def annotation_data(text: str) -> str:
    """Escape the characters that end a workflow command's message."""
    return text.replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
