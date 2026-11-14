# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compliance publication wires job names and selected artifact links to shared reporting."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from scripts.check_suites.github_actions import write_ci_result
from scripts.environment_server_compliance import checks


def test_compliance_publisher_wires_job_names_and_artifact_links(tmp_path, monkeypatch):
    component = "responses_api_agents/example"
    name = checks.subject(component)
    env = {
        "GITHUB_REPOSITORY": "example/Gym",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": "tested-sha",
        "GITHUB_SERVER_URL": "https://github.com",
        "RUNNER_TEMP": str(tmp_path),
        "SELECTED_COMPONENTS": json.dumps([component]),
        "CHECK_JOB_RESULT": "success",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    download = Mock(return_value={name: 202})
    post = Mock()
    monkeypatch.setattr(checks, "download_artifacts", download)
    monkeypatch.setattr(checks, "post_check_run", post)

    checks.main()

    download.assert_called_once_with(
        [name],
        artifact_prefix="environment-server-compliance",
        job_names={name: f"{component} compliance"},
        repo="example/Gym",
        run_id="123",
        attempt="2",
        output=tmp_path / "compliance-evidence",
        result_file="ci-result.json",
    )
    assert post.call_count == 1
    assert (
        "https://github.com/example/Gym/actions/runs/123/artifacts/202" in post.call_args.args[0]["output"]["summary"]
    )


@pytest.fixture
def advisory_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Mock]:
    component = "responses_api_agents/example"
    name = checks.subject(component)
    for key, value in {
        "GITHUB_REPOSITORY": "example/Gym",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": "event-sha",
        "TESTED_SHA": "tested-sha",
        "GITHUB_SERVER_URL": "https://github.com",
        "RUNNER_TEMP": str(tmp_path),
        "SELECTED_COMPONENTS": json.dumps([component]),
        "CHECK_JOB_RESULT": "success",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    }.items():
        monkeypatch.setenv(key, value)
    evidence = tmp_path / "compliance-evidence" / name
    write_ci_result(evidence, subject=name, status="failed")
    (evidence / checks.FINDINGS_FILE).write_text(
        json.dumps([{"rule": "A1", "path": f"{component}/app.py", "line": 42, "message": "Keep the warning visible"}])
    )
    monkeypatch.setattr(checks, "download_artifacts", Mock(return_value={name: 202}))
    post = Mock()
    monkeypatch.setattr(checks, "post_check_run", post)
    return evidence, post


@pytest.mark.parametrize("probe_result", ["success", "skipped"])
def test_advisory_findings_publish_before_a_successful_reporter_exit(
    advisory_publication, tmp_path, monkeypatch, capsys, probe_result
):
    _, post = advisory_publication
    monkeypatch.setenv("CHECK_JOB_RESULT", probe_result)

    assert checks.main() == 0

    post.assert_called_once()
    check = post.call_args.args[0]
    assert check["conclusion"] == "neutral"
    assert check["head_sha"] == "event-sha"
    summary = check["output"]["summary"]
    assert "responses_api_agents/example/app.py:42" in summary
    assert "Keep the warning visible" in summary
    assert "Tested commit: `tested-sha`" in summary
    assert "https://github.com/example/Gym/actions/runs/123/artifacts/202" in summary
    assert "Keep the warning visible" in (tmp_path / "summary.md").read_text()
    stdout = capsys.readouterr().out
    assert "::warning" in stdout
    assert "::error" not in stdout


@pytest.mark.parametrize("probe_result", ["failure", "cancelled"])
def test_an_advisory_result_does_not_hide_a_failed_probe(advisory_publication, monkeypatch, capsys, probe_result):
    _, post = advisory_publication
    monkeypatch.setenv("CHECK_JOB_RESULT", probe_result)

    assert checks.main() == 1

    post.assert_called_once()
    assert post.call_args.args[0]["conclusion"] == "neutral"
    assert f"Probe jobs: {probe_result}" in capsys.readouterr().out


@pytest.mark.parametrize("fault", ["unavailable", "missing-findings", "selection"])
def test_reporter_infrastructure_failures_publish_failure_and_exit_nonzero(advisory_publication, monkeypatch, fault):
    evidence, post = advisory_publication
    if fault == "unavailable":
        write_ci_result(evidence, subject=checks.subject("responses_api_agents/example"), status="unavailable")
    elif fault == "missing-findings":
        (evidence / checks.FINDINGS_FILE).unlink()
    else:
        monkeypatch.setenv("SELECTED_COMPONENTS", "null")

    assert checks.main() == 1

    post.assert_called_once()
    assert post.call_args.args[0]["conclusion"] == "failure"


def test_advisory_findings_do_not_hide_a_publication_failure(advisory_publication):
    _, post = advisory_publication
    post.side_effect = subprocess.CalledProcessError(1, ["gh", "api"])

    with pytest.raises(subprocess.CalledProcessError):
        checks.main()

    post.assert_called_once()
    assert post.call_args.args[0]["conclusion"] == "neutral"
