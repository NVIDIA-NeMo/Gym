# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compliance publication retains unchanged jobs' evidence after partial retries."""

import json
import subprocess
from pathlib import Path

import pytest
from scripts.environment_server_compliance import checks


@pytest.mark.parametrize("missing_rerun_upload", [False, True])
def test_compliance_publisher_matches_each_job_execution(tmp_path, monkeypatch, missing_rerun_upload):
    components = ["responses_api_agents/clean", "responses_api_agents/legacy"]
    env = {
        "GITHUB_REPOSITORY": "example/Gym",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": "tested-sha",
        "GITHUB_SERVER_URL": "https://github.com",
        "RUNNER_TEMP": str(tmp_path),
        "SELECTED_COMPONENTS": json.dumps(components),
        "CHECK_JOB_RESULT": "success",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    uploads = []
    jobs = []
    for index, component in enumerate(components):
        name = checks.subject(component)
        start, end = ("00", "10") if index == 0 else ("20", "30")
        jobs.append(
            {
                "name": f"Environment Server compliance (non-blocking) / {component} compliance",
                "run_attempt": 2,
                "status": "completed",
                "started_at": f"2026-11-10T15:{start}:00Z",
                "completed_at": f"2026-11-10T15:{end}:00Z",
            }
        )
        uploads.append(
            {
                "name": f"environment-server-compliance-{name}-123-1",
                "id": 100 + index,
                "expired": False,
                "created_at": "2026-11-10T15:05:00Z",
            }
        )
    if not missing_rerun_upload:
        uploads.append(
            {**uploads[-1], "name": uploads[-1]["name"][:-1] + "2", "id": 202, "created_at": "2026-11-10T15:25:00Z"}
        )

    def listing(command, **kwargs):
        values = jobs if "/attempts/2/jobs?per_page=100" in command[3] else uploads
        return "\n".join(json.dumps(value) for value in values)

    posted, downloaded = [], []

    def run(command, **kwargs):
        if command[:3] == ["gh", "run", "download"]:
            name = command[command.index("--name") + 1]
            downloaded.append(name)
            directory = Path(command[command.index("--dir") + 1])
            directory.mkdir(parents=True)
            status = "passed" if directory.name.endswith("clean") else "failed"
            (directory / "ci-result.json").write_text(
                json.dumps(
                    {
                        "subject": directory.name,
                        "sha": "tested-sha",
                        "status": status,
                    }
                )
            )
        else:
            assert command[:4] == ["gh", "api", "--method", "POST"]
            posted.append(json.loads(kwargs["input"]))

    monkeypatch.setattr(subprocess, "check_output", listing)
    monkeypatch.setattr(subprocess, "run", run)
    assert checks.main() == 1
    assert len(posted) == 1
    output = posted[0]["output"]
    expected = (
        "1 compliant, 0 with findings, 1 unavailable"
        if missing_rerun_upload
        else "1 compliant, 1 with findings, 0 unavailable"
    )
    assert output["title"] == expected
    assert downloaded[0] == uploads[0]["name"]
    assert uploads[1]["name"] not in downloaded
    assert "actions/runs/123/artifacts/100" in output["summary"]
    assert ("actions/runs/123/artifacts/202" in output["summary"]) is not missing_rerun_upload
    assert len(downloaded) == (1 if missing_rerun_upload else 2)
