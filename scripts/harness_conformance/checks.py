# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the existing P0 suite as a GitHub check, independently of probe execution."""

import json
import os
import subprocess
from pathlib import Path
from typing import NotRequired, TypedDict

from .registry import HARNESSES


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


def p0_check(harnesses: list[str] | None, *, artifacts: Path, sha: str) -> CheckRun:
    """Require a complete result for every selected harness; missing evidence fails P0."""
    rows = []
    counts = dict.fromkeys(("passed", "failed", "unavailable"), 0)
    for harness in harnesses or []:
        status = "unavailable"
        try:
            result = json.loads((artifacts / harness / "probes" / "ci-result.json").read_text())
            if result["harness"] == harness and result["sha"] == sha and result["status"] in counts:
                status = result["status"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        counts[status] += 1
        rows.append(f"| {harness} | {status} |")
    if harnesses is None:
        title = "Could not select affected harnesses"
        summary = "P0 could not be evaluated because harness selection failed."
    elif not harnesses:
        title = "No affected harnesses"
        summary = "No registered harnesses or their dependencies were changed."
    else:
        title = ", ".join(f"{count} {status}" for status, count in counts.items())
        summary = "| Harness | P0 result |\n|---|---|\n" + "\n".join(rows)
        summary += "\n\nUnavailable means setup, tests, execution, or evidence collection failed."
    return {
        "name": "Harness conformance P0",
        "head_sha": sha,
        "status": "completed",
        "conclusion": "failure" if harnesses is None or counts["failed"] or counts["unavailable"] else "success",
        "output": {"title": title, "summary": summary},
    }


def main() -> None:
    # No adapter imports or runtime installation: only this reporting job has checks:write.
    repo = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    artifacts = Path(os.environ["RUNNER_TEMP"]) / "conformance-evidence"
    try:
        harnesses = json.loads(os.environ["SELECTED_HARNESSES"])
        if not isinstance(harnesses, list) or any(h not in HARNESSES for h in harnesses):
            raise ValueError("Invalid selection")
    except (ValueError, TypeError, KeyError):
        harnesses = None
    for harness in harnesses or []:
        # Download each exact artifact separately: gh extracts a single artifact into --dir.
        # The attempt suffix prevents a partial rerun from reusing older passing evidence.
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
                    f"harness-conformance-{harness}-{run_id}-{attempt}",
                    "--dir",
                    str(artifacts / harness),
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            # A failed download may have extracted only part of the artifact.
            (artifacts / harness / "probes" / "ci-result.json").unlink(missing_ok=True)
            print(f"::warning::Could not download {harness} evidence; P0 will report it as unavailable.")
    payload = p0_check(harnesses, artifacts=artifacts, sha=os.environ["GITHUB_SHA"])
    run_url = f"{os.environ['GITHUB_SERVER_URL']}/{repo}/actions/runs/{run_id}/attempts/{attempt}"
    payload["details_url"] = run_url
    payload["external_id"] = f"harness-conformance-p0-{run_id}-{attempt}"
    output = payload["output"]
    output["summary"] += f"\n\n[Probe summaries and downloadable evidence]({run_url})."
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as handle:
        handle.write(f"## Harness conformance P0\n\n{output['title']}\n\n{output['summary']}\n")
    # A failed P0 conclusion is a check result, not a failure to publish it.
    subprocess.run(
        ["gh", "api", "--method", "POST", f"repos/{repo}/check-runs", "--input", "-"],
        input=json.dumps(payload),
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    main()
