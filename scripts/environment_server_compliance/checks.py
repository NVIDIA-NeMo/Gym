# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the compliance findings as a GitHub check, independently of the component jobs."""

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
    table_cell,
    tested_sha,
)

from .ci import FINDINGS_FILE, subject
from .engine import KINDS


CHECK_NAME = "Environment Server compliance (non-blocking)"
ARTIFACT_PREFIX = "environment-server-compliance"
RESULT_FILE = CI_RESULT
RESULT_LABELS = {"passed": "compliant", "failed": "with findings", "unavailable": "unavailable"}
MAX_LISTED_FINDINGS = 50


def _findings(path: Path) -> list[dict]:
    try:
        findings = json.loads(path.read_text())
        return findings if isinstance(findings, list) else []
    except (OSError, ValueError):
        return []


def compliance_check(
    components: list[str] | None, *, artifacts: Path, sha: str, artifact_urls: dict[str, str] | None = None
) -> CheckRun:
    """Summarize one result per selected component; missing or foreign evidence counts as unavailable."""
    rows, listed = [], []
    counts = dict.fromkeys(STATUSES, 0)
    for component in components or []:
        name = subject(component)
        result = read_ci_result(artifacts / name / RESULT_FILE, subject=name, sha=sha)
        status = result["status"] if result else "unavailable"
        findings = _findings(artifacts / name / FINDINGS_FILE) if result else []
        counts[status] += 1
        artifact_url = (artifact_urls or {}).get(name)
        evidence = f"[Download]({artifact_url})" if artifact_url else "See workflow artifacts/logs"
        rows.append(
            [component, RESULT_LABELS[status], ", ".join(sorted({f["rule"] for f in findings})) or "-", evidence]
        )
        listed += [
            f"- `{finding['path']}:{finding['line']}` **{finding['rule']}**: {table_cell(finding['message'])}"
            for finding in findings
        ]
    if components is None:
        title = "Could not select components"
        summary = "Compliance could not be evaluated because component selection failed."
    elif not components:
        title = "No changed components"
        summary = "No agent servers were changed."
    else:
        title = ", ".join(f"{counts[status]} {RESULT_LABELS[status]}" for status in STATUSES)
        summary = markdown_table(["Component", "Result", "Rules", "Evidence"], rows)
        summary += "\n\nFindings are warnings and do not block the change. Unavailable means the component could not "
        summary += "be installed or imported, or its evidence is missing.\n"
        if listed:
            summary += "\n" + "\n".join(listed[:MAX_LISTED_FINDINGS])
            if len(listed) > MAX_LISTED_FINDINGS:
                summary += f"\n- and {len(listed) - MAX_LISTED_FINDINGS} more; see the evidence artifacts."
    summary += f"\n\nTested commit: `{sha}`."
    return {
        "name": CHECK_NAME,
        "head_sha": sha,
        "status": "completed",
        "conclusion": "failure" if components is None or counts["failed"] or counts["unavailable"] else "success",
        "output": {"title": title, "summary": summary},
    }


def main() -> int:
    # No Gym imports or installation: only this reporting job has checks:write.
    repo = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    server_url = os.environ["GITHUB_SERVER_URL"]
    artifacts = Path(os.environ["RUNNER_TEMP"]) / "compliance-evidence"
    try:
        components = json.loads(os.environ["SELECTED_COMPONENTS"])
        if not isinstance(components, list) or any(
            not isinstance(c, str) or c.split("/", 1)[0] not in KINDS for c in components
        ):
            raise ValueError("Invalid selection")
    except (ValueError, TypeError, KeyError):
        components = None
    names = [subject(component) for component in components or []]
    artifact_ids = download_artifacts(
        names,
        artifact_prefix=ARTIFACT_PREFIX,
        job_names={subject(component): f"{component} compliance" for component in components or []},
        repo=repo,
        run_id=run_id,
        attempt=attempt,
        output=artifacts,
        result_file=RESULT_FILE,
    )
    artifact_urls = artifact_download_urls(artifact_ids, repo=repo, run_id=run_id, server_url=server_url)
    payload = compliance_check(components, artifacts=artifacts, sha=tested_sha(), artifact_urls=artifact_urls)
    payload["head_sha"] = os.environ["GITHUB_SHA"]
    run_url = f"{server_url}/{repo}/actions/runs/{run_id}/attempts/{attempt}"
    payload["details_url"] = run_url
    payload["external_id"] = f"environment-server-compliance-{run_id}-{attempt}"
    output = payload["output"]
    output["summary"] += f"\n\n[Component summaries and downloadable evidence]({run_url})."
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as handle:
        handle.write(f"## {CHECK_NAME}\n\n{output['title']}\n\n{output['summary']}\n")
    # Publish the detailed check before making the workflow job reflect its result.
    post_check_run(payload, repo=repo)
    return job_exit_code(payload, probe_result=os.environ.get("CHECK_JOB_RESULT", "success"))


if __name__ == "__main__":
    raise SystemExit(main())
