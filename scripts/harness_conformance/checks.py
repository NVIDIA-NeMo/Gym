# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the existing P0 suite as a GitHub check, independently of probe execution."""

import json
import os
import subprocess
from pathlib import Path
from typing import NotRequired, TypedDict

from .feedback import failure_details, reproduction
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


def p0_check(
    harnesses: list[str] | None, *, artifacts: Path, sha: str, artifact_urls: dict[str, str] | None = None
) -> CheckRun:
    """Require a complete result for every selected harness; missing evidence fails P0."""
    rows = []
    counts = dict.fromkeys(("passed", "failed", "unavailable"), 0)
    rerun = []
    for harness in harnesses or []:
        status = "unavailable"
        valid = False
        try:
            result = json.loads((artifacts / harness / "probes" / "ci-result.json").read_text())
            if result["harness"] == harness and result["sha"] == sha and result["status"] in counts:
                status = result["status"]
                valid = True
        except (OSError, ValueError, KeyError, TypeError):
            pass
        counts[status] += 1
        count, first = (
            failure_details(harness, artifacts / harness / "probes")
            if valid
            else ("Unknown", "Evidence is missing, invalid, or from another commit. Check the probe job logs.")
        )
        artifact_url = (artifact_urls or {}).get(harness)
        evidence = f"[Download report and logs]({artifact_url})" if artifact_url else "See workflow artifacts/logs"
        rows.append(f"| {harness} | {status} | {count} | {first} | {evidence} |")
        if status != "passed":
            rerun.append(harness)
    if harnesses is None:
        title = "Could not select affected harnesses"
        summary = "P0 could not be evaluated because harness selection failed."
    elif not harnesses:
        title = "No affected harnesses"
        summary = "No registered harnesses or their dependencies were changed."
    else:
        title = ", ".join(f"{count} {status}" for status, count in counts.items())
        summary = (
            "| Harness | P0 result | Failed scenario checks | First failure | Evidence |\n|---|---|---|---|---|\n"
        )
        summary += "\n".join(rows)
        summary += (
            "\n\nCounts include execution errors and count each scenario once. "
            "Unavailable means setup, tests, execution, or evidence collection failed.\n\n"
            "Each artifact contains `probes/conformance_report.md`, `setup.log`, `tests.log`, "
            "and per-scenario `probes/<harness>/<scenario>/scenario_result.json` and `episode.log`.\n\n"
        )
        if rerun:
            summary += reproduction(rerun)
    summary += f"\n\nTested commit: `{sha}`."
    return {
        "name": "Harness conformance P0",
        "head_sha": sha,
        "status": "completed",
        "conclusion": "failure" if harnesses is None or counts["failed"] or counts["unavailable"] else "success",
        "output": {"title": title, "summary": summary},
    }


def main() -> int:
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
    run_url = f"{os.environ['GITHUB_SERVER_URL']}/{repo}/actions/runs/{run_id}/attempts/{attempt}"
    artifact_urls = {}
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
        for harness in harnesses or []:
            artifact_id = by_name.get(f"harness-conformance-{harness}-{run_id}-{attempt}")
            if artifact_id:
                artifact_urls[harness] = (
                    f"{os.environ['GITHUB_SERVER_URL']}/{repo}/actions/runs/{run_id}/artifacts/{artifact_id}"
                )
    except (subprocess.CalledProcessError, ValueError):
        print("::warning::Could not link individual artifacts; use the workflow artifacts page.")
    payload = p0_check(harnesses, artifacts=artifacts, sha=os.environ["GITHUB_SHA"], artifact_urls=artifact_urls)
    payload["details_url"] = run_url
    payload["external_id"] = f"harness-conformance-p0-{run_id}-{attempt}"
    output = payload["output"]
    output["summary"] += f"\n\n[Probe summaries and downloadable evidence]({run_url})."
    setup_url = f"{os.environ['GITHUB_SERVER_URL']}/{repo}/blob/{os.environ['GITHUB_SHA']}/scripts/harness_conformance/README.md#run-diagnostic-probes"
    output["summary"] += f" [Local setup instructions]({setup_url})."
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as handle:
        handle.write(f"## Harness conformance P0\n\n{output['title']}\n\n{output['summary']}\n")
    # Publish the detailed check before making the workflow job reflect its result.
    subprocess.run(
        ["gh", "api", "--method", "POST", f"repos/{repo}/check-runs", "--input", "-"],
        input=json.dumps(payload),
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    probe_result = os.environ.get("PROBE_JOB_RESULT", "success")
    if payload["conclusion"] != "success" or probe_result not in ("success", "skipped"):
        print(
            f"::error title=Harness conformance P0::{output['title']}. "
            f"Probe jobs: {probe_result}. See the job summary for failures, rerun commands, and artifact links."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
