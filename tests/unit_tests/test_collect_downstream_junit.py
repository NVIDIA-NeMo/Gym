# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Gym-to-NeMo JUnit collection hop."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = REPO_ROOT / ".gitlab" / "ci" / "collect_downstream_junit.py"
PIPELINE_PATH = REPO_ROOT / ".gitlab-ci.yml"


def _load_collector():
    spec = importlib.util.spec_from_file_location("gym_collect_downstream_junit", COLLECTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collector = _load_collector()


def _artifact(entries: dict[str, bytes | str]) -> bytes:
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, content in entries.items():
            output.writestr(name, content.encode() if isinstance(content, str) else content)
    return archive.getvalue()


def _junit(suite: str) -> str:
    return f'<testsuite name="{suite}" tests="1"><testcase classname="ci" name="ok"/></testsuite>'


def _bridge(
    bridge_id: int = 11,
    *,
    pipeline_id: int = 22,
    project_id: int = 99,
    status: str = "success",
    name: str = "gym-cpu-ci",
) -> dict:
    return {
        "id": bridge_id,
        "name": name,
        "downstream_pipeline": {
            "id": pipeline_id,
            "project_id": project_id,
            "status": status,
        },
    }


def _report_job(job_id: int, *, status: str = "success", junit: bool = True) -> dict:
    artifacts = [{"file_type": "junit", "filename": "junit.xml.gz"}] if junit else []
    return {
        "id": job_id,
        "name": "nemo_gym_collect_junit",
        "status": status,
        "artifacts": artifacts,
    }


class _FakeApi:
    def __init__(self, *, bridges, jobs, archives):
        self.bridges = bridges
        self.jobs = jobs
        self.archives = archives
        self.bridge_calls = []
        self.job_calls = []
        self.artifact_calls = []

    def list_bridges(self, project_id, pipeline_id):
        self.bridge_calls.append((project_id, pipeline_id))
        return self.bridges

    def list_pipeline_jobs(self, project_id, pipeline_id):
        self.job_calls.append((project_id, pipeline_id))
        return self.jobs

    def download_job_artifacts(self, project_id, job_id):
        self.artifact_calls.append((project_id, job_id))
        value = self.archives[job_id]
        if isinstance(value, Exception):
            raise value
        return value


def test_collects_partial_junit_from_failed_parent_and_latest_retries(tmp_path):
    api = _FakeApi(
        bridges=[
            _bridge(10, pipeline_id=20),
            _bridge(12, pipeline_id=24, status="failed"),
        ],
        jobs=[
            _report_job(100, status="failed"),
            {"id": 105, "name": "unrelated", "status": "success", "artifacts": []},
            _report_job(110, status="failed"),
        ],
        archives={
            110: _artifact(
                {
                    "gym-junit/gym_core_unit_tests-80/core.xml": _junit("core"),
                    "gym-junit/gym_server_tests-90/server.xml": _junit("server"),
                    "coverage.xml": "<coverage/>",
                }
            )
        },
    )
    output_dir = tmp_path / "collected"

    result = collector.collect_downstream_junit(
        api,
        gym_project_id=77,
        gym_pipeline_id=88,
        bridge_name="gym-cpu-ci",
        report_job_name="nemo_gym_collect_junit",
        output_dir=output_dir,
    )

    assert result.downstream_project_id == 99
    assert result.downstream_pipeline_id == 24
    assert result.report_job_id == 110
    assert result.report_job_status == "failed"
    assert result.report_count == 2
    assert api.bridge_calls == [(77, 88)]
    assert api.job_calls == [(99, 24)]
    assert api.artifact_calls == [(99, 110)]
    assert len(list(output_dir.glob("**/*.xml"))) == 2


def test_latest_bridge_without_downstream_never_falls_back():
    bridges = [_bridge(10), {"id": 12, "name": "gym-cpu-ci", "downstream_pipeline": None}]

    with pytest.raises(collector.CollectorError, match="latest bridge.*has no downstream"):
        collector.resolve_downstream_pipeline(bridges, bridge_name="gym-cpu-ci")


@pytest.mark.parametrize("status", ["running", "manual", "unknown", ""])
def test_nonterminal_or_unknown_downstream_status_is_rejected(status):
    with pytest.raises(collector.CollectorError, match="non-terminal or unknown status"):
        collector.resolve_downstream_pipeline([_bridge(status=status)], bridge_name="gym-cpu-ci")


def test_latest_nonterminal_report_attempt_does_not_publish_stale_retry():
    jobs = [_report_job(100), _report_job(110, status="running")]

    with pytest.raises(collector.CollectorError, match="report job.*non-terminal"):
        collector.select_report_job(jobs, report_job_name="nemo_gym_collect_junit")


def test_successful_report_job_without_junit_creates_empty_directory(tmp_path):
    api = _FakeApi(
        bridges=[_bridge(status="failed")],
        jobs=[_report_job(110, junit=False)],
        archives={},
    )
    output_dir = tmp_path / "collected"

    result = collector.collect_downstream_junit(
        api,
        gym_project_id=77,
        gym_pipeline_id=88,
        bridge_name="gym-cpu-ci",
        report_job_name="nemo_gym_collect_junit",
        output_dir=output_dir,
    )

    assert result.report_count == 0
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
    assert api.artifact_calls == []


@pytest.mark.parametrize("status", ["failed", "canceled", "skipped"])
def test_unsuccessful_report_job_without_junit_fails_closed(tmp_path, status):
    api = _FakeApi(
        bridges=[_bridge(status="failed")],
        jobs=[_report_job(110, status=status, junit=False)],
        archives={},
    )

    with pytest.raises(collector.CollectorError, match=f"ended {status} without a JUnit artifact"):
        collector.collect_downstream_junit(
            api,
            gym_project_id=77,
            gym_pipeline_id=88,
            bridge_name="gym-cpu-ci",
            report_job_name="nemo_gym_collect_junit",
            output_dir=tmp_path / "collected",
        )


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ({"gym-junit/../../escape.xml": _junit("bad")}, "unsafe XML path"),
        ({"gym-junit/report.xml": "<testsuite>"}, "is malformed"),
        (
            {"gym-junit/report.xml": '<!DOCTYPE x [<!ENTITY y "z">]><testsuite name="&y;"/>'},
            "forbidden DTD/entity",
        ),
        ({"other/report.xml": _junit("wrong-prefix")}, "contains no validated XML"),
    ],
)
def test_malformed_or_unsafe_xml_is_rejected_atomically(tmp_path, entries, message):
    output_dir = tmp_path / "collected"
    api = _FakeApi(
        bridges=[_bridge()],
        jobs=[_report_job(110)],
        archives={110: _artifact(entries)},
    )

    with pytest.raises(collector.CollectorError, match=message):
        collector.collect_downstream_junit(
            api,
            gym_project_id=77,
            gym_pipeline_id=88,
            bridge_name="gym-cpu-ci",
            report_job_name="nemo_gym_collect_junit",
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert list(tmp_path.glob(".collected.tmp-*")) == []
    assert not (tmp_path / "escape.xml").exists()


def test_artifact_download_error_is_contextual_and_atomic(tmp_path):
    api = _FakeApi(
        bridges=[_bridge()],
        jobs=[_report_job(110)],
        archives={110: collector.CollectorError("HTTP 404: expired")},
    )

    with pytest.raises(collector.CollectorError, match="report job 110 artifacts.*expired"):
        collector.collect_downstream_junit(
            api,
            gym_project_id=77,
            gym_pipeline_id=88,
            bridge_name="gym-cpu-ci",
            report_job_name="nemo_gym_collect_junit",
            output_dir=tmp_path / "collected",
        )


class _Response:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None):
        self._body = BytesIO(body)
        self.headers = headers or {}

    def read(self, size=-1):
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _OpenSequence:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _json_response(value, headers=None):
    return _Response(json.dumps(value).encode(), headers)


def _request_headers(request: urllib.request.Request) -> dict[str, str]:
    return {key.lower(): value for key, value in request.header_items()}


def test_gitlab_client_paginates_cross_project_and_uses_split_auth():
    opener = _OpenSequence(
        [
            _json_response([_bridge(10)], {"X-Next-Page": "2"}),
            _json_response([_bridge(11)], {"X-Next-Page": ""}),
            _json_response([_report_job(110)]),
            _Response(b"artifact zip"),
        ]
    )
    api = collector.GitLabApi(
        "https://gitlab.example/api/v4/",
        job_token="job-secret",
        metadata_token="read-secret",
        open_url=opener,
    )

    assert [bridge["id"] for bridge in api.list_bridges(77, 88)] == [10, 11]
    assert api.list_pipeline_jobs(99, 22)[0]["id"] == 110
    assert api.download_job_artifacts(99, 110) == b"artifact zip"

    first_url = urllib.parse.urlsplit(opener.requests[0][0].full_url)
    assert first_url.path.endswith("/projects/77/pipelines/88/bridges")
    assert urllib.parse.parse_qs(first_url.query) == {"per_page": ["100"], "page": ["1"]}
    jobs_url = urllib.parse.urlsplit(opener.requests[2][0].full_url)
    assert jobs_url.path.endswith("/projects/99/pipelines/22/jobs")
    assert "include_retried" not in jobs_url.query
    assert _request_headers(opener.requests[0][0])["private-token"] == "read-secret"
    assert _request_headers(opener.requests[3][0])["job-token"] == "job-secret"


def test_job_token_artifact_denial_is_actionable_and_does_not_leak_token():
    error = urllib.error.HTTPError(
        "https://gitlab.example/api/v4/projects/99/jobs/110/artifacts",
        403,
        "Forbidden",
        {},
        BytesIO(b'{"message":"403 Forbidden"}'),
    )
    opener = _OpenSequence([error])
    api = collector.GitLabApi(
        "https://gitlab.example/api/v4",
        job_token="do-not-print-this",
        open_url=opener,
    )

    with pytest.raises(collector.CollectorError) as caught:
        api.download_job_artifacts(99, 110)

    message = str(caught.value)
    assert "owner-approved target-project job-token allowlist" in message
    assert "do-not-print-this" not in message


def test_job_token_metadata_denial_names_the_safe_fallback_without_leaking_token():
    error = urllib.error.HTTPError(
        "https://gitlab.example/api/v4/projects/77/pipelines/88/bridges",
        403,
        "Forbidden",
        {},
        BytesIO(b'{"message":"403 Forbidden"}'),
    )
    opener = _OpenSequence([error])
    api = collector.GitLabApi(
        "https://gitlab.example/api/v4",
        job_token="do-not-print-this",
        open_url=opener,
    )

    with pytest.raises(collector.CollectorError) as caught:
        api.list_bridges(77, 88)

    message = str(caught.value)
    assert "RO_API_TOKEN or GITLAB_API_TOKEN" in message
    assert "read_api access to both projects" in message
    assert "do-not-print-this" not in message


@pytest.mark.parametrize(
    ("environment", "metadata_token", "uses_job_token"),
    [
        (
            {
                "CI_API_V4_URL": "https://gitlab.example/api/v4",
                "CI_JOB_TOKEN": "job",
                "RO_API_TOKEN": "read-only",
                "GITLAB_API_TOKEN": "fallback",
            },
            "read-only",
            False,
        ),
        (
            {
                "CI_API_V4_URL": "https://gitlab.example/api/v4",
                "CI_JOB_TOKEN": "job",
                "GITLAB_API_TOKEN": "fallback",
            },
            "fallback",
            False,
        ),
        (
            {
                "CI_API_V4_URL": "https://gitlab.example/api/v4",
                "CI_JOB_TOKEN": "job",
            },
            "job",
            True,
        ),
    ],
)
def test_api_from_env_auth_priority(environment, metadata_token, uses_job_token):
    api = collector.api_from_env(environment)

    assert api.metadata_token == metadata_token
    assert api.metadata_uses_job_token is uses_job_token
    assert api.job_token == "job"


def test_cross_origin_redirect_drops_all_authentication_headers():
    request = urllib.request.Request(
        "https://gitlab.example/api/v4/projects/99/jobs/110/artifacts",
        headers={
            "JOB-TOKEN": "job-secret",
            "PRIVATE-TOKEN": "read-secret",
            "Authorization": "Bearer secret",
        },
    )

    redirected = collector.AuthSafeRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://object-storage.example/signed-artifact",
    )

    headers = _request_headers(redirected)
    assert "job-token" not in headers
    assert "private-token" not in headers
    assert "authorization" not in headers


def test_main_requires_ci_job_token_without_network(tmp_path):
    environment = {
        "CI_API_V4_URL": "https://gitlab.example/api/v4",
        "CI_PROJECT_ID": "77",
        "CI_PIPELINE_ID": "88",
    }

    with pytest.raises(SystemExit, match="CI_JOB_TOKEN is required"):
        collector.main(["--output-dir", os.fspath(tmp_path / "reports")], environment)


def test_pipeline_keeps_bridge_required_and_collector_always_running():
    pipeline = PIPELINE_PATH.read_text()

    assert "gym-cpu-ci:\n  stage: test" in pipeline
    assert "gym-cpu-ci-junit:\n  stage: report" in pipeline
    assert "--report-job-name nemo_gym_collect_junit" in pipeline
    assert "junit: collected-junit/**/*.xml" in pipeline
    collector_job = pipeline.split("gym-cpu-ci-junit:", 1)[1]
    assert "when: always" in collector_job
    assert "allow_failure: false" in collector_job
    bridge_job = pipeline.split("gym-cpu-ci:", 1)[1].split("gym-cpu-ci-junit:", 1)[0]
    assert "allow_failure" not in bridge_job
