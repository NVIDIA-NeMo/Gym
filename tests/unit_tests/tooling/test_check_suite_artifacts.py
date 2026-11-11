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
    files = [evidence("pi", 1), evidence("hermes", 1), evidence("hermes", 2, minute=35)]
    mock_listing(monkeypatch, [job("pi"), job("hermes", start=30, end=40)], files)
    downloaded = []

    def download(command, **kwargs):
        name = command[command.index("--name") + 1]
        downloaded.append(name)
        path = tmp_path / ("pi" if "-pi-" in name else "hermes") / "probes" / "ci-result.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "harness": path.parent.parent.name,
                    "sha": "sha",
                    "status": "passed" if name.endswith("-1") else "failed",
                }
            )
        )

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
    assert downloaded == [files[0]["name"], files[2]["name"]]
    result = checks.p0_check(["pi", "hermes"], artifacts=tmp_path, sha="sha")
    assert result["output"]["title"] == "1 passed, 1 failed, 0 unavailable"


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
        ([job("pi", start=30, end=40)], [evidence("pi", 1)]),  # Rerun lost its upload.
        ([job("pi")], [evidence("pi", 3)]),  # A newer attempt must not affect this report.
        ([job("pi")], [evidence("pi", 1, expired=True)]),
        ([job("pi", status="in_progress")], [evidence("pi", 2)]),
        ([job("pi")], [{**evidence("pi", 2), "name": "2"}]),  # Unrelated numeric artifact name.
        ([], [evidence("pi", 1)]),
    ],
)
def test_unmatched_evidence_never_reuses_an_older_local_result(tmp_path, monkeypatch, jobs, files):
    mock_listing(monkeypatch, jobs, files)
    stale = tmp_path / "pi" / "probes" / "ci-result.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"harness": "pi", "sha": "sha", "status": "passed"}))
    monkeypatch.setattr(artifacts.subprocess, "run", lambda *a, **kw: pytest.fail("must not download stale evidence"))
    assert (
        artifacts.download_artifacts(
            ["pi"],
            artifact_prefix=checks.ARTIFACT_PREFIX,
            job_names=checks.JOB_NAMES,
            repo="example/Gym",
            run_id="12",
            attempt="2",
            output=tmp_path,
            result_file=checks.RESULT_FILE,
        )
        == {}
    )
    assert (
        checks.p0_check(["pi"], artifacts=tmp_path, sha="sha")["output"]["title"]
        == "0 passed, 0 failed, 1 unavailable"
    )
