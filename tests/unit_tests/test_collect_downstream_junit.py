# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the receipt-bound NeMo-to-Gym JUnit relay."""

from __future__ import annotations

import hashlib
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
SOURCE_SHA = "a" * 40


def _load_collector():
    spec = importlib.util.spec_from_file_location("gym_collect_downstream_junit", COLLECTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collector = _load_collector()


def _identity():
    return collector.GymIdentity(
        project_id=191584,
        project_path="dl/nemo/gym",
        pipeline_id=60672261,
        mr_iid=452,
        source_sha=SOURCE_SHA,
    )


def _environment(**updates):
    environment = {
        "CI_API_V4_URL": "https://gitlab.example/api/v4",
        "CI_SERVER_URL": "https://gitlab.example",
        "CI_JOB_TOKEN": "job-secret",
        "CI_PROJECT_ID": "191584",
        "CI_PROJECT_PATH": "dl/nemo/gym",
        "CI_PIPELINE_ID": "60672261",
        "CI_MERGE_REQUEST_IID": "452",
        "CI_COMMIT_SHA": SOURCE_SHA,
    }
    environment.update(updates)
    return environment


def _junit(suite: str) -> bytes:
    return (f'<testsuite name="{suite}" tests="1"><testcase classname="ci" name="ok"/></testsuite>').encode()


def _receipt(reports: dict[str, bytes], **updates):
    identity = _identity()
    receipt = {
        "schema_version": 1,
        "package_name": "nemo-gym-ci-junit",
        "package_version": identity.package_version,
        "package_filename": "reports.zip",
        "gym_project_id": identity.project_id,
        "gym_project_path": identity.project_path,
        "gym_pipeline_id": identity.pipeline_id,
        "gym_mr_iid": identity.mr_iid,
        "gym_source_sha": identity.source_sha,
        "nemo_project_id": 65523,
        "nemo_pipeline_id": 60672262,
        "nemo_job_id": 381778367,
        "nemo_child_pipeline_id": 60672286,
        "report_count": len(reports),
        "reports": [
            {
                "path": path,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in sorted(reports.items())
        ],
    }
    receipt.update(updates)
    return receipt


def _relay(reports: dict[str, bytes] | None = None, *, receipt_updates=None, extra=None) -> bytes:
    reports = reports or {}
    receipt = _receipt(reports, **(receipt_updates or {}))
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("receipt.json", json.dumps(receipt, sort_keys=True))
        for path, payload in reports.items():
            output.writestr(path, payload)
        for path, payload in (extra or {}).items():
            output.writestr(path, payload)
    return archive.getvalue()


def test_valid_relay_is_extracted_atomically(tmp_path):
    reports = {
        "junit/core/core.xml": _junit("core"),
        "junit/server-0/server.xml": _junit("server"),
    }
    output_dir = tmp_path / "collected"

    result = collector.extract_relay(_relay(reports), identity=_identity(), output_dir=output_dir)

    assert result.package_version == f"pipeline-60672261-{SOURCE_SHA}"
    assert result.nemo_pipeline_id == 60672262
    assert result.nemo_job_id == 381778367
    assert result.nemo_child_pipeline_id == 60672286
    assert result.report_count == 2
    assert (output_dir / "core" / "core.xml").read_bytes() == reports["junit/core/core.xml"]
    assert (output_dir / "server-0" / "server.xml").read_bytes() == reports["junit/server-0/server.xml"]
    assert list(tmp_path.glob(".collected.tmp-*")) == []


def test_zero_report_relay_creates_empty_output(tmp_path):
    output_dir = tmp_path / "collected"

    result = collector.extract_relay(_relay(), identity=_identity(), output_dir=output_dir)

    assert result.report_count == 0
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": 2}, "schema_version mismatch"),
        ({"gym_project_id": 1}, "gym_project_id mismatch"),
        ({"gym_project_path": "other/project"}, "gym_project_path mismatch"),
        ({"gym_pipeline_id": 1}, "gym_pipeline_id mismatch"),
        ({"gym_mr_iid": 1}, "gym_mr_iid mismatch"),
        ({"gym_source_sha": "b" * 40}, "gym_source_sha mismatch"),
        ({"nemo_project_id": 1}, "nemo_project_id mismatch"),
        ({"nemo_pipeline_id": 0}, "nemo_pipeline_id must be a positive integer"),
    ],
)
def test_receipt_identity_mismatch_is_rejected_atomically(tmp_path, updates, message):
    output_dir = tmp_path / "collected"

    with pytest.raises(collector.CollectorError, match=message):
        collector.extract_relay(
            _relay({"junit/core.xml": _junit("core")}, receipt_updates=updates),
            identity=_identity(),
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert list(tmp_path.glob(".collected.tmp-*")) == []


@pytest.mark.parametrize(
    ("reports", "receipt_updates", "extra", "message"),
    [
        ({"junit/../escape.xml": _junit("bad")}, None, None, "unsafe JUnit path"),
        ({"other/report.xml": _junit("bad")}, None, None, "unsafe JUnit path"),
        ({"junit/report.xml": b"<testsuite>"}, None, None, "malformed"),
        (
            {"junit/report.xml": b'<!DOCTYPE x [<!ENTITY y "z">]><testsuite name="&y;"/>'},
            None,
            None,
            "forbidden DTD/entity",
        ),
        (
            {
                "junit/report.xml": (
                    '<?xml version="1.0" encoding="UTF-16"?>'
                    '<!DOCTYPE testsuite [<!ENTITY y "expanded">]>'
                    '<testsuite name="&y;"/>'
                ).encode("utf-16")
            },
            None,
            None,
            "forbidden DTD/entity",
        ),
        (
            {"junit/report.xml": _junit("ok")},
            None,
            {"unexpected.txt": b"x"},
            "members do not match receipt",
        ),
    ],
)
def test_unsafe_or_malformed_relay_is_rejected_atomically(tmp_path, reports, receipt_updates, extra, message):
    output_dir = tmp_path / "collected"

    with pytest.raises(collector.CollectorError, match=message):
        collector.extract_relay(
            _relay(reports, receipt_updates=receipt_updates, extra=extra),
            identity=_identity(),
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_hash_mismatch_is_rejected(tmp_path):
    reports = {"junit/report.xml": _junit("ok")}
    receipt = _receipt(reports)
    receipt["reports"][0]["sha256"] = "0" * 64
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("receipt.json", json.dumps(receipt))
        output.writestr("junit/report.xml", reports["junit/report.xml"])

    with pytest.raises(collector.CollectorError, match="SHA-256 mismatch"):
        collector.extract_relay(archive.getvalue(), identity=_identity(), output_dir=tmp_path / "collected")


def test_output_must_be_fresh(tmp_path):
    output_dir = tmp_path / "collected"
    output_dir.mkdir()

    with pytest.raises(collector.CollectorError, match="refusing to overwrite"):
        collector.extract_relay(_relay(), identity=_identity(), output_dir=output_dir)


class _Response:
    def __init__(self, body: bytes):
        self._body = BytesIO(body)

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


def _request_headers(request: urllib.request.Request) -> dict[str, str]:
    return {key.lower(): value for key, value in request.header_items()}


def test_package_client_downloads_exact_same_project_coordinate_with_job_token():
    opener = _OpenSequence([_Response(b"relay")])
    client = collector.GitLabPackageClient(
        "https://gitlab.example/api/v4/",
        server_url="https://gitlab.example",
        job_token="job-secret",
        open_url=opener,
    )

    result = client.download_relay(project_id=191584, package_version=_identity().package_version)

    assert result == b"relay"
    request, timeout = opener.requests[0]
    parsed = urllib.parse.urlsplit(request.full_url)
    assert parsed.path.endswith(
        f"/projects/191584/packages/generic/nemo-gym-ci-junit/pipeline-60672261-{SOURCE_SHA}/reports.zip"
    )
    assert parsed.query == ""
    assert _request_headers(request)["job-token"] == "job-secret"
    assert "private-token" not in _request_headers(request)
    assert timeout == collector.REQUEST_TIMEOUT_SECONDS


def test_missing_package_error_does_not_leak_token():
    url = "https://gitlab.example/api/v4/projects/191584/packages/generic/x/y/z"
    error = urllib.error.HTTPError(url, 404, "Not Found", {}, BytesIO(b'{"message":"404"}'))
    client = collector.GitLabPackageClient(
        "https://gitlab.example/api/v4",
        server_url="https://gitlab.example",
        job_token="do-not-print-this",
        open_url=_OpenSequence([error]),
    )

    with pytest.raises(collector.CollectorError) as caught:
        client.download_relay(project_id=191584, package_version=_identity().package_version)

    assert "missing for this exact pipeline" in str(caught.value)
    assert "do-not-print-this" not in str(caught.value)


def test_cross_origin_redirect_drops_all_authentication_headers():
    request = urllib.request.Request(
        "https://gitlab.example/api/v4/projects/191584/packages/generic/x/y/z",
        headers={
            "JOB-TOKEN": "job-secret",
            "PRIVATE-TOKEN": "read-secret",
            "Authorization": "Bearer secret",
        },
    )

    redirected = collector.AuthSafeRedirectHandler().redirect_request(
        request, None, 302, "Found", {}, "https://object-storage.example/signed-package"
    )

    headers = _request_headers(redirected)
    assert "job-token" not in headers
    assert "private-token" not in headers
    assert "authorization" not in headers


def test_main_requires_ci_job_token_without_network(tmp_path):
    environment = _environment()
    environment.pop("CI_JOB_TOKEN")

    with pytest.raises(SystemExit, match="CI_JOB_TOKEN is required"):
        collector.main(["--output-dir", os.fspath(tmp_path / "reports")], environment)


def test_identity_requires_full_sha():
    with pytest.raises(collector.CollectorError, match="full 40-character"):
        collector.identity_from_env(_environment(CI_COMMIT_SHA="abc"))


def test_package_client_rejects_noncanonical_api_origin():
    with pytest.raises(collector.CollectorError, match="must equal CI_SERVER_URL"):
        collector.GitLabPackageClient(
            "https://evil.example/api/v4",
            server_url="https://gitlab.example",
            job_token="secret",
        )


def test_receipt_rejects_duplicate_json_keys_atomically(tmp_path):
    receipt = json.dumps(_receipt({})).encode()
    duplicate = receipt[:-1] + b',"schema_version":1}'
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("receipt.json", duplicate)

    with pytest.raises(collector.CollectorError, match="duplicate key"):
        collector.extract_relay(archive.getvalue(), identity=_identity(), output_dir=tmp_path / "collected")


def test_pipeline_forwards_identity_and_keeps_bridge_status_separate():
    pipeline = PIPELINE_PATH.read_text()

    assert 'GYM_GITLAB_PROJECT_ID: "$CI_PROJECT_ID"' in pipeline
    assert 'GYM_GITLAB_PIPELINE_ID: "$CI_PIPELINE_ID"' in pipeline
    assert "gym-cpu-ci:\n  stage: test" in pipeline
    assert "gym-cpu-ci-junit:\n  stage: report" in pipeline
    assert "collect_downstream_junit.py" in pipeline
    assert "junit: collected-junit/**/*.xml" in pipeline
    collector_job = pipeline.split("gym-cpu-ci-junit:", 1)[1]
    assert "when: always" in collector_job
    assert "allow_failure: false" in collector_job
    bridge_job = pipeline.split("gym-cpu-ci:", 1)[1].split("gym-cpu-ci-junit:", 1)[0]
    assert "allow_failure" not in bridge_job
    assert "RO_API_TOKEN" not in collector_job
    assert "GITLAB_API_TOKEN" not in collector_job
