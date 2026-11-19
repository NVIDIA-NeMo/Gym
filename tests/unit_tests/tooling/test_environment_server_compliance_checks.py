# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compliance publication wires job names and selected artifact links to shared reporting."""

import json
from unittest.mock import Mock

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
