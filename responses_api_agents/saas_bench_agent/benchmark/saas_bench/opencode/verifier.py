# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/saas_bench/opencode/verifier.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Gym adaptation: preserve task checks while separating model and evaluation failures.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative grader diagnostics, separate from task achievement and raw score."""

from __future__ import annotations

import re

# These establish a broken grading dependency, not merely an absent task entity.
_INFRA = re.compile(
    r"unknown column|(?:table .+ doesn't exist)|no such (?:table|column)|"
    r"(?:401|403|50[0-9]) (?:client|server) error|"
    r"404 client error[^\n]+/api/2\.0/authentication|"
    r"(?:authentication|auth) failed.*(?:404|401|403)|"
    r"connection refused|failed to establish a new connection|"
    r"(?:connect|read) timed out|expecting value: line 1 column 1",
    re.IGNORECASE | re.DOTALL,
)


def verifier_diagnostics(verify: dict) -> dict:
    """Mask an unavailable grader; retain its measured checks without awarding points.

    Generic exceptions (e.g. indexing an empty task-state result) remain ordinary
    failed checks with a warning. We do not assume every verifier exception is an
    infrastructure problem or remove failed checkpoint weights from denominators.
    """
    infra, warnings = [], []
    for check in verify.get("checks", []):
        if check.get("passed"):
            continue
        detail = check.get("detail", "")
        record = {
            "label": check.get("label"),
            "weight": check.get("weight"),
            "detail": detail,
        }
        if _INFRA.search(detail):
            infra.append(record)
        elif "exception:" in detail.lower() or "error" in detail.lower():
            warnings.append(record)
    unavailable = (
        verify.get("status") not in {"PASS", "FAIL"}
        or bool(infra)
        or bool(verify.get("judge_failures"))
        or bool(verify.get("error"))
    )
    return {
        "unavailable": unavailable,
        "infrastructure_checks": infra,
        "judge_failures": verify.get("judge_failures", []),
        "warnings": warnings,
        "raw_score": verify.get("raw_score", verify.get("score")),
        "raw_status": verify.get("status"),
        "policy": "mask_entire_sample_on_unavailable_grader",
    }


class VerifierUnavailable(RuntimeError):
    """A required grading API is unavailable before the episode starts."""


def probe_onlyoffice_api(base_url: str, *, timeout: float = 30) -> dict:
    """Require a JSON authentication route, rather than a generic ready HTML page.

    Empty credentials do not log in or change app state. A JSON client error is
    enough to prove routing to the API; a 404, HTML welcome page or redirect is
    not. The normal slot readiness probe has already finished before this runs.
    """
    import json
    import time
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    deadline = time.monotonic() + timeout
    result = {
        "available": False,
        "endpoint": base_url.rstrip("/") + "/api/2.0/authentication",
    }
    while True:
        request = Request(
            result["endpoint"],
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            try:
                response = urlopen(
                    request, timeout=max(0.1, min(5, deadline - time.monotonic()))
                )
            except HTTPError as exc:
                response = exc
            with response:
                result["status"] = response.status
                result["content_type"] = response.headers.get("Content-Type", "")
                body = response.read(65536)
                try:
                    payload = json.loads(body)
                    json_response = True
                except (ValueError, UnicodeDecodeError):
                    json_response = False
                # This bundled Community Server reports rejected credentials as
                # HTTP 500 with a specific JSON error, including empty credentials.
                auth_rejected = (
                    json_response
                    and isinstance(payload, dict)
                    and isinstance(payload.get("error"), dict)
                    and payload["error"].get("message") == "User authentication failed"
                )
                result["expected_auth_rejection"] = (
                    response.status == 500 and auth_rejected
                )
                result["available"] = (
                    (
                        response.status in {200, 201, 400, 401, 403, 405, 422}
                        or response.status == 500
                        and auth_rejected
                    )
                    and json_response
                    and response.geturl().rstrip("/") == result["endpoint"].rstrip("/")
                )
                if result["available"]:
                    return result
        except (URLError, TimeoutError, OSError) as exc:
            result["error"] = str(exc)[:500]
        if time.monotonic() >= deadline:
            return result
        time.sleep(min(1, max(0, deadline - time.monotonic())))
