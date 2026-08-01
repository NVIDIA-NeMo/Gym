#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Republish JUnit XML from the exact NeMo CI pipeline triggered by Gym.

This collector runs in the originating Gym pipeline after ``gym-cpu-ci`` has
reached a terminal state. It resolves that bridge's immutable downstream
pipeline ID, selects the latest terminal ``nemo_gym_collect_junit`` attempt,
downloads its artifact by immutable job ID, and copies only validated JUnit
XML below ``gym-junit/`` into a fresh directory for Gym to publish.

Metadata requests prefer ``RO_API_TOKEN``, then ``GITLAB_API_TOKEN``. If
neither is set, they use ``CI_JOB_TOKEN``. Artifact downloads always use
``CI_JOB_TOKEN`` so cross-project access remains governed by GitLab's job-token
policy instead of an implicit credential fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


DEFAULT_BRIDGE_NAME = "gym-cpu-ci"
DEFAULT_REPORT_JOB_NAME = "nemo_gym_collect_junit"
DEFAULT_OUTPUT_DIR = "collected-junit"
EXPECTED_REPORT_PREFIX = PurePosixPath("gym-junit")
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
MAX_XML_BYTES = 64 * 1024 * 1024
MAX_TOTAL_XML_BYTES = 256 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30
TERMINAL_PIPELINE_STATUSES = frozenset({"canceled", "failed", "skipped", "success"})
TERMINAL_JOB_STATUSES = frozenset({"canceled", "failed", "skipped", "success"})
AUTH_HEADER_NAMES = ("Authorization", "JOB-TOKEN", "PRIVATE-TOKEN")


class CollectorError(RuntimeError):
    """A bridge, API, artifact, or report validation failure."""


@dataclass(frozen=True)
class DownstreamPipeline:
    """The exact cross-project pipeline selected from the Gym bridge."""

    project_id: int
    pipeline_id: int
    bridge_id: int
    status: str


@dataclass(frozen=True)
class CollectionResult:
    """Summary of one successful downstream report collection."""

    downstream_project_id: int
    downstream_pipeline_id: int
    report_job_id: int
    report_job_status: str
    report_count: int
    output_dir: Path


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port


class AuthSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward GitLab credentials to cross-origin artifact storage."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and _origin(req.full_url) != _origin(newurl):
            protected = {name.lower() for name in AUTH_HEADER_NAMES}
            for header_store in (redirected.headers, redirected.unredirected_hdrs):
                for header_name in list(header_store):
                    if header_name.lower() in protected:
                        header_store.pop(header_name)
        return redirected


def _default_open_url(request: urllib.request.Request, *, timeout: int):
    opener = urllib.request.build_opener(AuthSafeRedirectHandler())
    return opener.open(request, timeout=timeout)


class GitLabApi:
    """Small stdlib GitLab client for bridge, job, and artifact reads."""

    def __init__(
        self,
        api_url: str,
        *,
        job_token: str,
        metadata_token: str | None = None,
        open_url: Callable[..., Any] = _default_open_url,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        parsed = urllib.parse.urlsplit(self.api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CollectorError("CI_API_V4_URL must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise CollectorError("CI_API_V4_URL must not contain a query string or fragment")
        if not job_token:
            raise CollectorError("CI_JOB_TOKEN is required to download NeMo CI job artifacts")

        self.job_token = job_token
        self.metadata_token = metadata_token or job_token
        self.metadata_uses_job_token = metadata_token is None
        self._open_url = open_url

    def _url(self, path: str, query: Mapping[str, str | int] | None = None) -> str:
        url = f"{self.api_url}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        return url

    @staticmethod
    def _error_body(error: urllib.error.HTTPError) -> str:
        try:
            body = error.read(2049)
        except OSError:
            return ""
        return body[:2048].decode("utf-8", errors="replace").strip()

    def _authorization_error(self, *, url: str, status: int, metadata: bool) -> CollectorError:
        if metadata and self.metadata_uses_job_token:
            return CollectorError(
                f"GitLab metadata request was denied with HTTP {status}: {url}. "
                "This GitLab version may not allow CI_JOB_TOKEN to list pipeline bridges/jobs; "
                "set masked RO_API_TOKEN or GITLAB_API_TOKEN with read_api access to both projects. "
                "Tokens were not logged."
            )
        if metadata:
            return CollectorError(
                f"GitLab metadata request was denied with HTTP {status}: {url}. "
                "Verify the read-API token can read both the Gym and NeMo CI projects. "
                "Tokens were not logged."
            )
        return CollectorError(
            f"GitLab artifact request was denied with HTTP {status}: {url}. "
            "The originating Gym CI_JOB_TOKEN must be authorized to read NeMo CI job artifacts; "
            "an owner-approved target-project job-token allowlist entry may be required. "
            "Tokens were not logged."
        )

    def _get(
        self,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
        metadata: bool,
        max_bytes: int,
    ) -> tuple[bytes, Mapping[str, str]]:
        url = self._url(path, query)
        if metadata:
            header_name = "JOB-TOKEN" if self.metadata_uses_job_token else "PRIVATE-TOKEN"
            token = self.metadata_token
        else:
            header_name = "JOB-TOKEN"
            token = self.job_token
        request = urllib.request.Request(
            url,
            headers={
                header_name: token,
                "Accept": "application/json" if metadata else "application/zip",
            },
        )

        try:
            with self._open_url(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read(max_bytes + 1)
                response_headers = response.headers
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise self._authorization_error(url=url, status=exc.code, metadata=metadata) from exc
            detail = self._error_body(exc)
            suffix = f": {detail}" if detail else ""
            raise CollectorError(f"GitLab request failed with HTTP {exc.code}: {url}{suffix}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CollectorError(f"GitLab request failed for {url}: {exc}") from exc

        if len(body) > max_bytes:
            raise CollectorError(f"GitLab response exceeded the {max_bytes}-byte safety limit: {url}")
        return body, response_headers

    def _get_json_page(
        self,
        path: str,
        *,
        query: Mapping[str, str | int],
    ) -> tuple[list[dict[str, Any]], Mapping[str, str]]:
        body, headers = self._get(path, query=query, metadata=True, max_bytes=MAX_JSON_BYTES)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CollectorError(f"GitLab returned invalid JSON for {self._url(path, query)}") from exc
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise CollectorError(f"GitLab returned an unexpected payload for {self._url(path, query)}")
        return payload, headers

    def _get_all(self, path: str) -> list[dict[str, Any]]:
        page = 1
        items: list[dict[str, Any]] = []
        while True:
            query = {"per_page": 100, "page": page}
            payload, headers = self._get_json_page(path, query=query)
            items.extend(payload)
            next_page = str(headers.get("X-Next-Page", "")).strip()
            if not next_page:
                return items
            if not next_page.isdigit() or int(next_page) <= page:
                raise CollectorError(f"GitLab returned invalid X-Next-Page={next_page!r} for {self._url(path, query)}")
            page = int(next_page)

    def list_bridges(self, project_id: int, pipeline_id: int) -> list[dict[str, Any]]:
        return self._get_all(f"projects/{project_id}/pipelines/{pipeline_id}/bridges")

    def list_pipeline_jobs(self, project_id: int, pipeline_id: int) -> list[dict[str, Any]]:
        # The default endpoint excludes superseded retry attempts. Do not set include_retried=true.
        return self._get_all(f"projects/{project_id}/pipelines/{pipeline_id}/jobs")

    def download_job_artifacts(self, project_id: int, job_id: int) -> bytes:
        body, _ = self._get(
            f"projects/{project_id}/jobs/{job_id}/artifacts",
            metadata=False,
            max_bytes=MAX_ARCHIVE_BYTES,
        )
        return body


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CollectorError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CollectorError(f"{label} must be a positive integer") from exc
    if number <= 0 or str(value).strip() != str(number):
        raise CollectorError(f"{label} must be a positive integer")
    return number


def _record_id(record: Mapping[str, Any], label: str) -> int:
    return _positive_integer(record.get("id"), f"{label} id")


def _record_status(record: Mapping[str, Any], label: str, terminal_statuses: frozenset[str]) -> str:
    status = str(record.get("status", "")).strip()
    if status not in terminal_statuses:
        shown = status or "missing"
        raise CollectorError(f"{label} has a non-terminal or unknown status: {shown}")
    return status


def resolve_downstream_pipeline(
    bridges: Iterable[Mapping[str, Any]],
    *,
    bridge_name: str,
) -> DownstreamPipeline:
    """Select the latest named bridge attempt and its exact downstream pipeline."""
    records = list(bridges)
    matches = [bridge for bridge in records if bridge.get("name") == bridge_name]
    if not matches:
        available = sorted({str(bridge.get("name")) for bridge in records if bridge.get("name")})
        available_text = ", ".join(available) if available else "none"
        raise CollectorError(
            f"bridge {bridge_name!r} was not found in the Gym pipeline; available bridges: {available_text}"
        )

    bridge = max(matches, key=lambda item: _record_id(item, "bridge"))
    bridge_id = _record_id(bridge, "bridge")
    downstream = bridge.get("downstream_pipeline")
    if not isinstance(downstream, dict):
        raise CollectorError(f"latest bridge {bridge_name!r} (job {bridge_id}) has no downstream pipeline")
    pipeline_id = _positive_integer(
        downstream.get("id"),
        f"bridge {bridge_name!r} downstream pipeline id",
    )
    project_id = _positive_integer(
        downstream.get("project_id"),
        f"bridge {bridge_name!r} downstream project id",
    )
    status = _record_status(
        downstream,
        f"bridge {bridge_name!r} downstream pipeline {pipeline_id}",
        TERMINAL_PIPELINE_STATUSES,
    )
    return DownstreamPipeline(
        project_id=project_id,
        pipeline_id=pipeline_id,
        bridge_id=bridge_id,
        status=status,
    )


def select_report_job(
    jobs: Iterable[Mapping[str, Any]],
    *,
    report_job_name: str,
) -> dict[str, Any]:
    """Select the newest terminal attempt of the NeMo parent report job."""
    matches: list[dict[str, Any]] = []
    available: set[str] = set()
    for item in jobs:
        job = dict(item)
        name = job.get("name")
        if isinstance(name, str) and name.strip():
            available.add(name)
        if name == report_job_name:
            _record_id(job, "report job")
            matches.append(job)

    if not matches:
        available_text = ", ".join(sorted(available)) if available else "none"
        raise CollectorError(
            f"report job {report_job_name!r} was not found in the NeMo parent pipeline; "
            f"available jobs: {available_text}"
        )

    latest = max(matches, key=lambda item: _record_id(item, "report job"))
    latest_id = _record_id(latest, "report job")
    _record_status(latest, f"report job {report_job_name!r} ({latest_id})", TERMINAL_JOB_STATUSES)
    return latest


def _has_junit_artifact(job: Mapping[str, Any]) -> bool:
    artifacts = job.get("artifacts", [])
    if artifacts is None:
        return False
    if not isinstance(artifacts, list):
        raise CollectorError(f"report job {_record_id(job, 'report job')} has malformed artifact metadata")
    return any(isinstance(artifact, dict) and artifact.get("file_type") == "junit" for artifact in artifacts)


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "report"


def _validate_xml_member(info: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(info.filename)
    if (
        not info.filename
        or "\\" in info.filename
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CollectorError(f"artifact archive contains an unsafe XML path: {info.filename!r}")
    unix_mode = (info.external_attr >> 16) & 0o177777
    file_type = stat.S_IFMT(unix_mode)
    if file_type and not stat.S_ISREG(unix_mode):
        raise CollectorError(f"artifact XML member is not a regular file: {info.filename!r}")
    if info.flag_bits & 0x1:
        raise CollectorError(f"artifact XML member is encrypted: {info.filename!r}")
    if info.file_size > MAX_XML_BYTES:
        raise CollectorError(f"artifact XML member {info.filename!r} exceeds the {MAX_XML_BYTES}-byte safety limit")
    return path


def _is_junit_xml(xml_bytes: bytes, member_name: str) -> bool:
    if re.search(rb"<!\s*(?:DOCTYPE|ENTITY)\b", xml_bytes, flags=re.IGNORECASE):
        raise CollectorError(f"artifact XML member contains a forbidden DTD/entity: {member_name!r}")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise CollectorError(f"artifact XML member is malformed: {member_name!r}: {exc}") from exc
    if not isinstance(root.tag, str):
        return False
    return root.tag.rsplit("}", 1)[-1] in {"testsuite", "testsuites"}


def extract_junit_reports(
    archive: bytes,
    destination: Path,
    *,
    job_id: int,
    job_name: str,
) -> int:
    """Validate an artifact ZIP and write only expected NeMo collector XML."""
    try:
        artifact = zipfile.ZipFile(BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise CollectorError(f"report job {job_id} artifact is not a valid ZIP archive") from exc

    with artifact:
        members = artifact.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise CollectorError(
                f"report job {job_id} artifact has {len(members)} members; limit is {MAX_ARCHIVE_MEMBERS}"
            )
        seen_names: set[str] = set()
        junit_documents: list[tuple[PurePosixPath, bytes]] = []
        total_xml_bytes = 0
        for info in members:
            if info.is_dir() or not info.filename.lower().endswith(".xml"):
                continue
            path = _validate_xml_member(info)
            if info.filename in seen_names:
                raise CollectorError(f"artifact archive contains duplicate XML path: {info.filename!r}")
            seen_names.add(info.filename)
            if len(path.parts) < 2 or path.parts[0] != EXPECTED_REPORT_PREFIX.name:
                continue
            try:
                with artifact.open(info) as member:
                    xml_bytes = member.read(MAX_XML_BYTES + 1)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise CollectorError(
                    f"could not read XML member {info.filename!r} from report job {job_id}: {exc}"
                ) from exc
            if len(xml_bytes) > MAX_XML_BYTES:
                raise CollectorError(
                    f"artifact XML member {info.filename!r} exceeds the {MAX_XML_BYTES}-byte safety limit"
                )
            total_xml_bytes += len(xml_bytes)
            if total_xml_bytes > MAX_TOTAL_XML_BYTES:
                raise CollectorError(
                    f"report job {job_id} artifact XML exceeds the {MAX_TOTAL_XML_BYTES}-byte total safety limit"
                )
            if not _is_junit_xml(xml_bytes, info.filename):
                continue
            junit_documents.append((path, xml_bytes))

    if not junit_documents:
        raise CollectorError(
            f"report job {job_id} ({job_name}) advertises JUnit but its artifact contains no "
            f"validated XML below {EXPECTED_REPORT_PREFIX}/"
        )

    job_dir = destination / f"{_safe_component(job_name)}-{job_id}"
    job_dir.mkdir()
    for index, (path, xml_bytes) in enumerate(junit_documents, start=1):
        basename = _safe_component(path.stem)
        (job_dir / f"{index:04d}-{basename}.xml").write_bytes(xml_bytes)
    return len(junit_documents)


def collect_downstream_junit(
    api: GitLabApi,
    *,
    gym_project_id: int,
    gym_pipeline_id: int,
    bridge_name: str,
    report_job_name: str,
    output_dir: Path,
) -> CollectionResult:
    """Collect the exact NeMo parent report attempt into a fresh Gym directory."""
    if not bridge_name.strip():
        raise CollectorError("bridge name must not be empty")
    if not report_job_name.strip():
        raise CollectorError("report job name must not be empty")
    if os.path.lexists(output_dir):
        raise CollectorError(f"output path already exists; refusing to overwrite it: {output_dir}")

    downstream = resolve_downstream_pipeline(
        api.list_bridges(gym_project_id, gym_pipeline_id),
        bridge_name=bridge_name,
    )
    report_job = select_report_job(
        api.list_pipeline_jobs(downstream.project_id, downstream.pipeline_id),
        report_job_name=report_job_name,
    )
    report_job_id = _record_id(report_job, "report job")
    report_job_status = str(report_job["status"])
    has_junit = _has_junit_artifact(report_job)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    report_count = 0
    try:
        if has_junit:
            try:
                archive = api.download_job_artifacts(downstream.project_id, report_job_id)
            except CollectorError as exc:
                raise CollectorError(f"could not download NeMo report job {report_job_id} artifacts: {exc}") from exc
            report_count = extract_junit_reports(
                archive,
                staging_dir,
                job_id=report_job_id,
                job_name=report_job_name,
            )
        elif report_job_status != "success":
            raise CollectorError(f"NeMo report job {report_job_id} ended {report_job_status} without a JUnit artifact")
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return CollectionResult(
        downstream_project_id=downstream.project_id,
        downstream_pipeline_id=downstream.pipeline_id,
        report_job_id=report_job_id,
        report_job_status=report_job_status,
        report_count=report_count,
        output_dir=output_dir,
    )


def _required_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise CollectorError(f"{name} is required")
    return value


def api_from_env(environ: Mapping[str, str]) -> GitLabApi:
    """Create a GitLab client with split metadata and artifact credentials."""
    metadata_token = environ.get("RO_API_TOKEN", "").strip()
    if not metadata_token:
        metadata_token = environ.get("GITLAB_API_TOKEN", "").strip()
    return GitLabApi(
        _required_env(environ, "CI_API_V4_URL"),
        job_token=_required_env(environ, "CI_JOB_TOKEN"),
        metadata_token=metadata_token or None,
    )


def create_parser(environ: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser, defaulting identifiers from GitLab CI."""
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description="Collect JUnit XML from the exact NeMo CI pipeline triggered by Gym.")
    parser.add_argument(
        "--project-id",
        default=environ.get("CI_PROJECT_ID"),
        help="Originating Gym project ID (default: $CI_PROJECT_ID)",
    )
    parser.add_argument(
        "--pipeline-id",
        default=environ.get("CI_PIPELINE_ID"),
        help="Originating Gym pipeline ID (default: $CI_PIPELINE_ID)",
    )
    parser.add_argument(
        "--bridge-name",
        default=DEFAULT_BRIDGE_NAME,
        help=f"Gym bridge job name (default: {DEFAULT_BRIDGE_NAME})",
    )
    parser.add_argument(
        "--report-job-name",
        default=DEFAULT_REPORT_JOB_NAME,
        help=f"NeMo parent report job name (default: {DEFAULT_REPORT_JOB_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=f"Fresh Gym report directory to create (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    """Command entry point."""
    environ = os.environ if environ is None else environ
    args = create_parser(environ).parse_args(argv)
    try:
        result = collect_downstream_junit(
            api_from_env(environ),
            gym_project_id=_positive_integer(args.project_id, "CI_PROJECT_ID/--project-id"),
            gym_pipeline_id=_positive_integer(args.pipeline_id, "CI_PIPELINE_ID/--pipeline-id"),
            bridge_name=args.bridge_name,
            report_job_name=args.report_job_name,
            output_dir=args.output_dir,
        )
    except CollectorError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    if result.report_count == 0:
        print(
            f"WARNING: NeMo report job {result.report_job_id} published no JUnit XML; "
            f"created empty report directory {result.output_dir}",
            file=sys.stderr,
        )
    print(
        f"Collected {result.report_count} JUnit report(s) from NeMo project "
        f"{result.downstream_project_id}, pipeline {result.downstream_pipeline_id}, "
        f"job {result.report_job_id} ({result.report_job_status}) into {result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    main()
