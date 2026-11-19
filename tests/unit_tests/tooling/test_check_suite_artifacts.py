# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared artifact downloads do not reuse partial results or skip other harnesses."""

import json
import subprocess

import pytest
from scripts.check_suites import github_actions as artifacts
from scripts.harness_conformance import checks


def job(harness, *, start=10, end=20, status="completed"):
    # GitHub copies retained jobs into the new attempt with their original times.
    return {
        "name": f"Harness conformance (non-blocking) / {harness} conformance probes",
        "run_attempt": 2,
        "status": status,
        "started_at": f"2026-11-06T15:{start:02}:00Z",
        "completed_at": f"2026-11-06T15:{end:02}:00Z",
    }


def evidence(harness, attempt, *, minute=15, expired=False):
    return {
        "id": attempt * 100,
        "name": f"harness-conformance-{harness}-12-{attempt}",
        "created_at": f"2026-11-06T15:{minute:02}:00Z",
        "expired": expired,
    }


def mock_listing(monkeypatch, jobs, files):
    def listing(command, **kwargs):
        assert "--paginate" in command
        if "/attempts/2/jobs?per_page=100" in command[3]:
            values = jobs
        else:
            assert command[3].endswith("/artifacts?per_page=100")
            values = files
        return "\n".join(json.dumps(value) for value in values)

    monkeypatch.setattr(artifacts.subprocess, "check_output", listing)


def test_partial_retry_retains_successful_job_evidence_and_uses_rerun_failure(tmp_path, monkeypatch):
    files = [evidence("pi", 1), evidence("hermes", 1), evidence("hermes", 2, minute=35), evidence("codex", 1)]
    mock_listing(monkeypatch, [job("pi"), job("hermes", start=30, end=40), job("codex", start=30, end=40)], files)
    # Codex reran without an upload; neither its old artifact nor local result is valid.
    stale = tmp_path / "codex" / checks.RESULT_FILE
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"harness": "codex", "sha": "sha", "status": "passed"}))
    downloaded = []

    def download(command, **kwargs):
        name = command[command.index("--name") + 1]
        downloaded.append(name)
        path = tmp_path / ("pi" if "-pi-" in name else "hermes") / "probes" / "ci-result.json"
        path.parent.mkdir(parents=True)
        status = "passed" if name.endswith("-1") else "failed"
        path.write_text(json.dumps({"harness": path.parent.parent.name, "sha": "sha", "status": status}))

    monkeypatch.setattr(artifacts.subprocess, "run", download)
    artifacts.download_artifacts(
        ["pi", "hermes", "codex"],
        artifact_prefix=checks.ARTIFACT_PREFIX,
        job_names=checks.JOB_NAMES,
        repo="example/Gym",
        run_id="12",
        attempt="2",
        output=tmp_path,
        result_file=checks.RESULT_FILE,
    )
    assert downloaded == [files[0]["name"], files[2]["name"]]
    assert not stale.exists()
    result = checks.p0_check(["pi", "hermes", "codex"], artifacts=tmp_path, sha="sha")
    assert result["output"]["title"] == "1 passed, 1 failed, 1 unavailable"


def test_failed_download_invalidates_partial_result_and_continues(tmp_path, monkeypatch):
    calls = []
    mock_listing(monkeypatch, [job("pi"), job("hermes")], [evidence("pi", 2), evidence("hermes", 2)])

    def download(command, **kwargs):
        calls.append(command)
        if "harness-conformance-pi-12-2" in command:
            partial = tmp_path / "pi" / "probes" / "ci-result.json"
            partial.parent.mkdir(parents=True)
            partial.write_text('"partial result"')
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(artifacts.subprocess, "run", download)
    artifacts.download_artifacts(
        ["pi", "hermes"],
        artifact_prefix=checks.ARTIFACT_PREFIX,
        job_names=checks.JOB_NAMES,
        repo="example/Gym",
        run_id="12",
        attempt="2",
        output=tmp_path,
        result_file=checks.RESULT_FILE,
    )
    assert len(calls) == 2 and "harness-conformance-hermes-12-2" in calls[1]
    assert not (tmp_path / "pi" / "probes" / "ci-result.json").exists()


@pytest.mark.parametrize(
    "jobs,files",
    [
        ([job("pi")], [evidence("pi", 3)]),  # A newer attempt must not affect this report.
        ([job("pi")], [evidence("pi", 1, expired=True)]),
        ([job("pi", status="in_progress")], [evidence("pi", 2)]),
        ([job("pi")], [{**evidence("pi", 2), "name": "2"}]),  # Unrelated numeric artifact name.
        ([], [evidence("pi", 1)]),
    ],
)
def test_job_artifact_rejects_ineligible_evidence(jobs, files):
    assert (
        artifacts._job_artifact(
            "pi",
            "pi conformance probes",
            jobs,
            files,
            artifact_prefix=checks.ARTIFACT_PREFIX,
            run_id="12",
            attempt="2",
        )
        is None
    )
