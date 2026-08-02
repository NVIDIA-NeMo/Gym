#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and republish the JUnit relay for the originating Gym pipeline.

The trusted NeMo CI parent collector publishes one receipt-bound Generic
Package file into the Gym project. This job downloads that exact coordinate
from its own project with ``CI_JOB_TOKEN`` and never queries cross-project
pipeline metadata or receives a persistent API token.
"""

from __future__ import annotations

import argparse
import hashlib
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
from typing import Any, Callable, Mapping, Sequence


DEFAULT_OUTPUT_DIR = "collected-junit"
PACKAGE_NAME = "nemo-gym-ci-junit"
PACKAGE_FILENAME = "reports.zip"
RECEIPT_FILENAME = "receipt.json"
RECEIPT_SCHEMA_VERSION = 1
EXPECTED_GITLAB_API_URL = "https://gitlab-master.nvidia.com/api/v4"
EXPECTED_GYM_PROJECT_ID = 191584
EXPECTED_GYM_PROJECT_PATH = "dl/nemo/gym"
NEMO_PROJECT_ID = 65523
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_XML_BYTES = 64 * 1024 * 1024
MAX_TOTAL_XML_BYTES = 256 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
AUTH_HEADER_NAMES = ("Authorization", "JOB-TOKEN", "PRIVATE-TOKEN")
RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "package_name",
        "package_version",
        "package_filename",
        "gym_project_id",
        "gym_project_path",
        "gym_pipeline_id",
        "gym_mr_iid",
        "gym_source_sha",
        "nemo_project_id",
        "nemo_pipeline_id",
        "nemo_job_id",
        "nemo_child_pipeline_id",
        "report_count",
        "reports",
    }
)
REPORT_FIELDS = frozenset({"path", "size", "sha256"})


class CollectorError(RuntimeError):
    """A relay download, receipt, archive, or report validation failure."""


@dataclass(frozen=True)
class GymIdentity:
    """Immutable identity of the originating Gym MR pipeline."""

    project_id: int
    project_path: str
    pipeline_id: int
    mr_iid: int
    source_sha: str

    @property
    def package_version(self) -> str:
        return f"pipeline-{self.pipeline_id}-{self.source_sha}"


@dataclass(frozen=True)
class CollectionResult:
    """Summary of one validated relay collection."""

    package_version: str
    nemo_pipeline_id: int
    nemo_job_id: int
    nemo_child_pipeline_id: int
    report_count: int
    output_dir: Path


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port


class AuthSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward GitLab credentials to cross-origin package storage."""

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
    return urllib.request.build_opener(AuthSafeRedirectHandler()).open(request, timeout=timeout)


class GitLabPackageClient:
    """Small stdlib client for one same-project Generic Package download."""

    def __init__(
        self,
        api_url: str,
        *,
        job_token: str,
        open_url: Callable[..., Any] = _default_open_url,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        parsed = urllib.parse.urlsplit(self.api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CollectorError("CI_API_V4_URL must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise CollectorError("CI_API_V4_URL must not contain a query string or fragment")
        if self.api_url != EXPECTED_GITLAB_API_URL:
            raise CollectorError(f"Gym report relay requires CI_API_V4_URL={EXPECTED_GITLAB_API_URL}")
        if not job_token:
            raise CollectorError("CI_JOB_TOKEN is required to download the Gym JUnit relay")
        self.job_token = job_token
        self._open_url = open_url

    def download_relay(self, *, project_id: int, package_version: str) -> bytes:
        components = (
            "projects",
            str(project_id),
            "packages",
            "generic",
            PACKAGE_NAME,
            package_version,
            PACKAGE_FILENAME,
        )
        path = "/".join(urllib.parse.quote(component, safe="") for component in components)
        url = f"{self.api_url}/{path}"
        request = urllib.request.Request(
            url,
            headers={"JOB-TOKEN": self.job_token, "Accept": "application/zip"},
        )
        try:
            with self._open_url(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read(MAX_ARCHIVE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            detail = _error_body(exc)
            suffix = f": {detail}" if detail else ""
            if exc.code in {401, 403}:
                raise CollectorError(
                    f"Gym relay package request was denied with HTTP {exc.code}: {url}. "
                    "The current Gym CI_JOB_TOKEN must be allowed to read its own package. "
                    f"Tokens were not logged{suffix}"
                ) from exc
            if exc.code == 404:
                raise CollectorError(f"Gym relay package is missing for this exact pipeline: {url}{suffix}") from exc
            raise CollectorError(f"Gym relay package request failed with HTTP {exc.code}: {url}{suffix}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CollectorError(f"Gym relay package request failed for {url}: {exc}") from exc
        if len(body) > MAX_ARCHIVE_BYTES:
            raise CollectorError(f"Gym relay package exceeds the {MAX_ARCHIVE_BYTES}-byte safety limit")
        return body


def _error_body(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read(2049)
    except OSError:
        return ""
    return body[:2048].decode("utf-8", errors="replace").strip()


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


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CollectorError(f"{label} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CollectorError(f"{label} must be a non-negative integer") from exc
    if number < 0 or str(value).strip() != str(number):
        raise CollectorError(f"{label} must be a non-negative integer")
    return number


def _required_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise CollectorError(f"{name} is required")
    return value


def identity_from_env(environ: Mapping[str, str]) -> GymIdentity:
    source_sha = _required_env(environ, "CI_COMMIT_SHA")
    if not COMMIT_SHA_PATTERN.fullmatch(source_sha):
        raise CollectorError("CI_COMMIT_SHA must be a full 40-character lowercase hexadecimal SHA")
    identity = GymIdentity(
        project_id=_positive_integer(_required_env(environ, "CI_PROJECT_ID"), "CI_PROJECT_ID"),
        project_path=_required_env(environ, "CI_PROJECT_PATH"),
        pipeline_id=_positive_integer(_required_env(environ, "CI_PIPELINE_ID"), "CI_PIPELINE_ID"),
        mr_iid=_positive_integer(_required_env(environ, "CI_MERGE_REQUEST_IID"), "CI_MERGE_REQUEST_IID"),
        source_sha=source_sha,
    )
    if identity.project_id != EXPECTED_GYM_PROJECT_ID:
        raise CollectorError(f"CI_PROJECT_ID must be {EXPECTED_GYM_PROJECT_ID}")
    if identity.project_path != EXPECTED_GYM_PROJECT_PATH:
        raise CollectorError(f"CI_PROJECT_PATH must be {EXPECTED_GYM_PROJECT_PATH}")
    return identity


def _validate_member(info: zipfile.ZipInfo, *, label: str, max_size: int) -> PurePosixPath:
    path = PurePosixPath(info.filename)
    if (
        not info.filename
        or "\\" in info.filename
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CollectorError(f"relay archive contains an unsafe {label} path: {info.filename!r}")
    unix_mode = (info.external_attr >> 16) & 0o177777
    file_type = stat.S_IFMT(unix_mode)
    if file_type and not stat.S_ISREG(unix_mode):
        raise CollectorError(f"relay {label} member is not a regular file: {info.filename!r}")
    if info.flag_bits & 0x1:
        raise CollectorError(f"relay {label} member is encrypted: {info.filename!r}")
    if info.file_size > max_size:
        raise CollectorError(f"relay {label} member {info.filename!r} exceeds the {max_size}-byte limit")
    return path


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, max_size: int) -> bytes:
    try:
        with archive.open(info) as member:
            payload = member.read(max_size + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise CollectorError(f"could not read relay member {info.filename!r}: {exc}") from exc
    if len(payload) > max_size:
        raise CollectorError(f"relay member {info.filename!r} exceeds the {max_size}-byte limit")
    return payload


def _parse_receipt(payload: bytes) -> dict[str, Any]:
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise CollectorError(f"relay receipt contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        receipt = json.loads(payload, object_pairs_hook=object_pairs)
    except CollectorError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CollectorError("relay receipt is not valid UTF-8 JSON") from exc
    if not isinstance(receipt, dict):
        raise CollectorError("relay receipt must be a JSON object")
    fields = set(receipt)
    if fields != RECEIPT_FIELDS:
        missing = sorted(RECEIPT_FIELDS - fields)
        extra = sorted(fields - RECEIPT_FIELDS)
        raise CollectorError(f"relay receipt fields do not match schema: missing={missing}, extra={extra}")
    return receipt


def _validate_receipt_identity(receipt: Mapping[str, Any], identity: GymIdentity) -> tuple[int, int, int]:
    numeric_expected = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "gym_project_id": identity.project_id,
        "gym_pipeline_id": identity.pipeline_id,
        "gym_mr_iid": identity.mr_iid,
        "nemo_project_id": NEMO_PROJECT_ID,
    }
    for field, expected_value in numeric_expected.items():
        actual_value = _positive_integer(receipt.get(field), f"receipt {field}")
        if actual_value != expected_value:
            raise CollectorError(
                f"relay receipt {field} mismatch: expected {expected_value!r}, got {receipt.get(field)!r}"
            )
    string_expected = {
        "package_name": PACKAGE_NAME,
        "package_version": identity.package_version,
        "package_filename": PACKAGE_FILENAME,
        "gym_project_path": identity.project_path,
        "gym_source_sha": identity.source_sha,
    }
    for field, expected_value in string_expected.items():
        if receipt.get(field) != expected_value:
            raise CollectorError(
                f"relay receipt {field} mismatch: expected {expected_value!r}, got {receipt.get(field)!r}"
            )
    return (
        _positive_integer(receipt.get("nemo_pipeline_id"), "receipt nemo_pipeline_id"),
        _positive_integer(receipt.get("nemo_job_id"), "receipt nemo_job_id"),
        _positive_integer(receipt.get("nemo_child_pipeline_id"), "receipt nemo_child_pipeline_id"),
    )


def _validate_junit(payload: bytes, path: str) -> None:
    if re.search(rb"<!\s*(?:DOCTYPE|ENTITY)\b", payload, flags=re.IGNORECASE):
        raise CollectorError(f"relay XML contains a forbidden DTD/entity: {path!r}")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise CollectorError(f"relay XML is malformed: {path!r}: {exc}") from exc
    if not isinstance(root.tag, str) or root.tag.rsplit("}", 1)[-1] not in {"testsuite", "testsuites"}:
        raise CollectorError(f"relay XML does not have a JUnit root: {path!r}")


def extract_relay(archive_bytes: bytes, *, identity: GymIdentity, output_dir: Path) -> CollectionResult:
    """Validate a receipt-bound ZIP and atomically publish its JUnit files."""
    if os.path.lexists(output_dir):
        raise CollectorError(f"output path already exists; refusing to overwrite it: {output_dir}")
    try:
        archive = zipfile.ZipFile(BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise CollectorError("Gym relay package is not a valid ZIP archive") from exc

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        with archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise CollectorError(f"relay archive has {len(members)} members; limit is {MAX_ARCHIVE_MEMBERS}")
            if any(info.is_dir() for info in members):
                raise CollectorError("relay archive must not contain directory entries")
            by_name: dict[str, zipfile.ZipInfo] = {}
            for info in members:
                if info.filename in by_name:
                    raise CollectorError(f"relay archive contains duplicate path: {info.filename!r}")
                by_name[info.filename] = info
            receipt_info = by_name.get(RECEIPT_FILENAME)
            if receipt_info is None:
                raise CollectorError(f"relay archive is missing {RECEIPT_FILENAME}")
            _validate_member(receipt_info, label="receipt", max_size=MAX_RECEIPT_BYTES)
            receipt = _parse_receipt(_read_member(archive, receipt_info, max_size=MAX_RECEIPT_BYTES))
            nemo_pipeline_id, nemo_job_id, nemo_child_pipeline_id = _validate_receipt_identity(receipt, identity)

            reports = receipt.get("reports")
            if not isinstance(reports, list):
                raise CollectorError("relay receipt reports must be a list")
            report_count = _nonnegative_integer(receipt.get("report_count"), "receipt report_count")
            if report_count != len(reports):
                raise CollectorError("relay receipt report_count does not match reports")

            expected_paths: list[str] = []
            validated: list[tuple[PurePosixPath, bytes]] = []
            total_xml_bytes = 0
            for index, report in enumerate(reports):
                if not isinstance(report, dict) or set(report) != REPORT_FIELDS:
                    raise CollectorError(f"relay receipt report {index} does not match schema")
                path_value = report.get("path")
                if not isinstance(path_value, str):
                    raise CollectorError(f"relay receipt report {index} path must be a string")
                path = PurePosixPath(path_value)
                if (
                    "\\" in path_value
                    or path.is_absolute()
                    or len(path.parts) < 2
                    or path.parts[0] != "junit"
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or path.suffix.lower() != ".xml"
                ):
                    raise CollectorError(f"relay receipt contains an unsafe JUnit path: {path_value!r}")
                expected_paths.append(path_value)
                info = by_name.get(path_value)
                if info is None:
                    raise CollectorError(f"relay archive is missing declared JUnit file: {path_value!r}")
                _validate_member(info, label="XML", max_size=MAX_XML_BYTES)
                expected_size = _positive_integer(report.get("size"), f"report {path_value!r} size")
                expected_sha = report.get("sha256")
                if not isinstance(expected_sha, str) or not SHA256_PATTERN.fullmatch(expected_sha):
                    raise CollectorError(f"report {path_value!r} sha256 must be lowercase hexadecimal")
                payload = _read_member(archive, info, max_size=MAX_XML_BYTES)
                if len(payload) != expected_size:
                    raise CollectorError(f"relay report size mismatch: {path_value!r}")
                if hashlib.sha256(payload).hexdigest() != expected_sha:
                    raise CollectorError(f"relay report SHA-256 mismatch: {path_value!r}")
                total_xml_bytes += len(payload)
                if total_xml_bytes > MAX_TOTAL_XML_BYTES:
                    raise CollectorError(f"relay XML exceeds the {MAX_TOTAL_XML_BYTES}-byte total safety limit")
                _validate_junit(payload, path_value)
                validated.append((path, payload))

            if expected_paths != sorted(expected_paths) or len(expected_paths) != len(set(expected_paths)):
                raise CollectorError("relay receipt report paths must be sorted and unique")
            actual_names = set(by_name)
            declared_names = {RECEIPT_FILENAME, *expected_paths}
            if actual_names != declared_names:
                raise CollectorError(
                    f"relay archive members do not match receipt: unexpected={sorted(actual_names - declared_names)}, "
                    f"missing={sorted(declared_names - actual_names)}"
                )

            for path, payload in validated:
                destination = staging_dir.joinpath(*path.parts[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return CollectionResult(
        package_version=identity.package_version,
        nemo_pipeline_id=nemo_pipeline_id,
        nemo_job_id=nemo_job_id,
        nemo_child_pipeline_id=nemo_child_pipeline_id,
        report_count=report_count,
        output_dir=output_dir,
    )


def create_parser(environ: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description="Validate and republish the exact Gym JUnit relay package.")
    parser.add_argument(
        "--output-dir",
        default=environ.get("GYM_CI_RELAY_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        type=Path,
        help=f"Fresh Gym report directory to create (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    environ = os.environ if environ is None else environ
    args = create_parser(environ).parse_args(argv)
    try:
        identity = identity_from_env(environ)
        client = GitLabPackageClient(
            _required_env(environ, "CI_API_V4_URL"),
            job_token=_required_env(environ, "CI_JOB_TOKEN"),
        )
        archive = client.download_relay(
            project_id=identity.project_id,
            package_version=identity.package_version,
        )
        result = extract_relay(archive, identity=identity, output_dir=args.output_dir)
    except CollectorError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    if result.report_count == 0:
        print(
            f"WARNING: relay from NeMo pipeline {result.nemo_pipeline_id} contains no JUnit XML; "
            f"created empty report directory {result.output_dir}",
            file=sys.stderr,
        )
    print(
        f"Collected {result.report_count} JUnit report(s) from NeMo pipeline "
        f"{result.nemo_pipeline_id}, child {result.nemo_child_pipeline_id}, job "
        f"{result.nemo_job_id} via package {result.package_version} into {result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    main()
