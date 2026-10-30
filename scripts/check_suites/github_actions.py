# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard-library helpers for CI check suites that run in GitHub Actions.

Workflows import this module with bare `python3` before Gym is installed, so it must not import `nemo_gym` or
third-party packages.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import NotRequired, TypedDict


def changed_paths(base: str, *, root: Path, subject: str) -> list[str] | None:
    """Compare the entire branch since its merge base, including deleted/renamed paths.

    Returns None when there is no base or the diff is unavailable; the caller then runs all `subject`.
    """
    if not base:
        return None
    try:
        ancestor = subprocess.check_output(
            ["git", "merge-base", "--", base, "HEAD"], cwd=root, text=True, stderr=subprocess.PIPE
        ).strip()
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


def write_ci_result(output: Path, *, subject: str, status: str, subject_key: str = "subject") -> None:
    """Record the status of one subject for the tested commit."""
    output.mkdir(parents=True, exist_ok=True)
    (output / CI_RESULT).write_text(
        json.dumps({subject_key: subject, "sha": os.environ.get("GITHUB_SHA", "local"), "status": status}) + "\n"
    )


def download_artifacts(
    subjects: list[str],
    *,
    artifact_prefix: str,
    repo: str,
    run_id: str,
    attempt: str,
    output: Path,
    result_file: str,
) -> None:
    """Download one attempt only; failed/partial downloads invalidate that subject's `result_file`."""
    for subject in subjects:
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
                    f"{artifact_prefix}-{subject}-{run_id}-{attempt}",
                    "--dir",
                    str(output / subject),
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            # Do not let partially extracted or older files qualify this attempt.
            (output / subject / result_file).unlink(missing_ok=True)
            print(f"::warning::Could not download {subject} evidence; its measurement is unavailable.")


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


def artifact_download_urls(
    subjects: list[str], *, artifact_prefix: str, repo: str, run_id: str, attempt: str, server_url: str
) -> dict[str, str]:
    """Map each subject to the download page of its unexpired artifact in this run attempt."""
    urls = {}
    try:
        listing = subprocess.check_output(
            [
                "gh",
                "api",
                "--paginate",
                f"repos/{repo}/actions/runs/{run_id}/artifacts",
                "--jq",
                ".artifacts[] | select(.expired == false) | [.name, .id] | @tsv",
            ],
            text=True,
        )
        by_name = dict(line.split("\t") for line in listing.splitlines())
        for subject in subjects:
            artifact_id = by_name.get(f"{artifact_prefix}-{subject}-{run_id}-{attempt}")
            if artifact_id:
                urls[subject] = f"{server_url}/{repo}/actions/runs/{run_id}/artifacts/{artifact_id}"
    except (subprocess.CalledProcessError, ValueError):
        print("::warning::Could not link individual artifacts; use the workflow artifacts page.")
    return urls


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
