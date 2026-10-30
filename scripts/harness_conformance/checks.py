# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the existing P0 suite as a GitHub check, independently of probe execution."""

import json
import os
from pathlib import Path

from scripts.check_suites.github_actions import (
    CI_RESULT,
    STATUSES,
    CheckRun,
    artifact_download_urls,
    download_artifacts,
    job_exit_code,
    markdown_table,
    post_check_run,
    read_ci_result,
)

from .feedback import failure_details, reproduction
from .registry import HARNESSES


CHECK_NAME = "Harness conformance (non-blocking)"
ARTIFACT_PREFIX = "harness-conformance"
RESULT_FILE = f"probes/{CI_RESULT}"


def p0_check(
    harnesses: list[str] | None, *, artifacts: Path, sha: str, artifact_urls: dict[str, str] | None = None
) -> CheckRun:
    """Require a complete result for every selected harness; missing evidence fails P0."""
    rows = []
    counts = dict.fromkeys(STATUSES, 0)
    rerun = []
    for harness in harnesses or []:
        result = read_ci_result(artifacts / harness / RESULT_FILE, subject=harness, sha=sha, subject_key="harness")
        status = result["status"] if result else "unavailable"
        counts[status] += 1
        count, first = (
            failure_details(harness, artifacts / harness / "probes")
            if result
            else ("Unknown", "Evidence is missing, invalid, or from another commit. Check the probe job logs.")
        )
        artifact_url = (artifact_urls or {}).get(harness)
        evidence = f"[Download report and logs]({artifact_url})" if artifact_url else "See workflow artifacts/logs"
        rows.append([harness, status, count, first, evidence])
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
        summary = markdown_table(["Harness", "P0 result", "Failed scenario checks", "First failure", "Evidence"], rows)
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
        "name": CHECK_NAME,
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
    download_artifacts(
        harnesses or [],
        artifact_prefix=ARTIFACT_PREFIX,
        repo=repo,
        run_id=run_id,
        attempt=attempt,
        output=artifacts,
        result_file=RESULT_FILE,
    )
    run_url = f"{os.environ['GITHUB_SERVER_URL']}/{repo}/actions/runs/{run_id}/attempts/{attempt}"
    artifact_urls = artifact_download_urls(
        harnesses or [],
        artifact_prefix=ARTIFACT_PREFIX,
        repo=repo,
        run_id=run_id,
        attempt=attempt,
        server_url=os.environ["GITHUB_SERVER_URL"],
    )
    payload = p0_check(harnesses, artifacts=artifacts, sha=os.environ["GITHUB_SHA"], artifact_urls=artifact_urls)
    payload["details_url"] = run_url
    payload["external_id"] = f"harness-conformance-p0-{run_id}-{attempt}"
    output = payload["output"]
    output["summary"] += f"\n\n[Probe summaries and downloadable evidence]({run_url})."
    setup_url = f"{os.environ['GITHUB_SERVER_URL']}/{repo}/blob/{os.environ['GITHUB_SHA']}/scripts/harness_conformance/README.md#run-diagnostic-probes"
    output["summary"] += f" [Local setup instructions]({setup_url})."
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as handle:
        handle.write(f"## {CHECK_NAME}\n\n{output['title']}\n\n{output['summary']}\n")
    # Publish the detailed check before making the workflow job reflect its result.
    post_check_run(payload, repo=repo)
    return job_exit_code(payload, probe_result=os.environ.get("PROBE_JOB_RESULT", "success"))


if __name__ == "__main__":
    raise SystemExit(main())
