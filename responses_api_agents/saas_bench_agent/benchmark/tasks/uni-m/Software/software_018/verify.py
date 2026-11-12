# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_018/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-018-I1: CVE Remediation Sprint for todo-api and blog-engine

Checks: 14 weighted checks across code-server, baserow, openproject.
Strategy: Baserow REST API, code-server docker exec, OpenProject REST API.

Required env vars:
  SERVER_HOSTNAME, CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import os
import sys
import subprocess
import json

try:
    import requests as req_lib
except ImportError:
    print("FATAL: 'requests' library not available", file=sys.stderr)
    sys.exit(1)

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

REQUIRED_VARS = [
    "CODE_SERVER_PORT", "CODE_SERVER_CONTAINER",
    "BASEROW_PORT", "BASEROW_CONTAINER", "BASEROW_DB_CONTAINER",
    "OPENPROJECT_PORT", "OPENPROJECT_CONTAINER",
]
for _var in REQUIRED_VARS:
    if not os.environ.get(_var):
        print(f"FATAL: {_var} not set", file=sys.stderr)
        sys.exit(1)

CODE_SERVER_CONTAINER = os.environ["CODE_SERVER_CONTAINER"]
BASEROW_PORT = os.environ["BASEROW_PORT"]
BASEROW_DB_CONTAINER = os.environ["BASEROW_DB_CONTAINER"]
OPENPROJECT_PORT = os.environ["OPENPROJECT_PORT"]
OPENPROJECT_CONTAINER = os.environ["OPENPROJECT_CONTAINER"]

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"
OP_URL = f"http://{HOST}:{OPENPROJECT_PORT}"

# ── Expected data ─────────────────────────────────────────────────────────────
EXPECTED_CVES = {
    "Flask":      {"cve_id": "CVE-2023-30861", "cvss": 7.5, "fixed": "2.2.5",   "vuln": "2.0.1",    "project": "todo-api",    "severity": "High"},
    "Jinja2":     {"cve_id": "CVE-2024-22195", "cvss": 5.4, "fixed": "3.1.3",   "vuln": "3.0.1",    "project": "todo-api",    "severity": "Medium"},
    "SQLAlchemy": {"cve_id": "CVE-2023-27479", "cvss": 4.3, "fixed": "1.4.49",  "vuln": "1.4.22",   "project": "todo-api",    "severity": "Medium"},
    "requests":   {"cve_id": "CVE-2023-32681", "cvss": 6.1, "fixed": "2.31.0",  "vuln": "2.25.1",   "project": "todo-api",    "severity": "Medium"},
    "express":    {"cve_id": "CVE-2022-24999", "cvss": 7.5, "fixed": "4.17.3",  "vuln": "4.17.1",   "project": "blog-engine", "severity": "High"},
    "ejs":        {"cve_id": "CVE-2022-29078", "cvss": 9.8, "fixed": "3.1.7",   "vuln": "3.1.6",    "project": "blog-engine", "severity": "Critical"},
    "marked":     {"cve_id": "CVE-2022-21680", "cvss": 7.5, "fixed": "4.0.10",  "vuln": "2.0.0",    "project": "blog-engine", "severity": "High"},
    "lodash":     {"cve_id": "CVE-2021-23337", "cvss": 7.2, "fixed": "4.17.21", "vuln": "4.17.20",  "project": "blog-engine", "severity": "High"},
}

EXPECTED_CVE_IDS = {v["cve_id"] for v in EXPECTED_CVES.values()}

# Work packages expected for Critical + High only (5 total)
EXPECTED_WPS: dict[str, dict] = {}
for _lib, _info in EXPECTED_CVES.items():
    if _info["severity"] in ("Critical", "High"):
        _subject = f"[{_info['project']}] Upgrade {_lib}: {_info['vuln']} \u2192 {_info['fixed']} ({_info['cve_id']})"
        _priority = "High" if _info["severity"] == "Critical" else "Normal"
        _desc = f"CVSS: {_info['cvss']}; Severity: {_info['severity']}; Discovered: 2025-02-10"
        EXPECTED_WPS[_info["cve_id"]] = {
            "subject": _subject, "priority": _priority,
            "description": _desc, "library": _lib,
        }


# ── Result accumulator ────────────────────────────────────────────────────────
_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


# ── Helpers ───────────────────────────────────────────────────────────────────
def docker_exec(container: str, *args: str, timeout: int = 15) -> tuple[int, str, str]:
    r = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


def baserow_auth() -> dict:
    """Get Baserow auth token and return headers."""
    resp = req_lib.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    resp.raise_for_status()
    token = resp.json()["token"]
    return {"Authorization": f"JWT {token}"}


def op_auth() -> tuple:
    """Return (username, password) for OpenProject basic auth."""
    return ("admin", "AdminPass123!")


def op_get(path: str, params: dict | None = None):
    resp = req_lib.get(f"{OP_URL}{path}", auth=op_auth(), params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ── Shared state across Baserow checks ────────────────────────────────────────
_br_headers: dict | None = None
_br_table_id: int | None = None
_br_fields: dict | None = None   # field_name -> field_info
_br_rows: list | None = None


def _init_baserow():
    global _br_headers
    if _br_headers is not None:
        return
    _br_headers = baserow_auth()


def _get_field_value(row: dict, field_name: str):
    """Extract a field's value from a Baserow row by field name."""
    if not _br_fields:
        return None
    field = _br_fields.get(field_name)
    if not field:
        return None
    val = row.get(f"field_{field['id']}")
    if isinstance(val, dict) and "value" in val:
        return val["value"]
    return val


# ── Baserow checks ────────────────────────────────────────────────────────────

def check_1_baserow_db_exists() -> None:
    """Verify Baserow database 'Dependency Security Audit 2025Q1' exists."""
    global _br_table_id, _br_fields
    try:
        _init_baserow()
        resp = req_lib.get(f"{BASEROW_URL}/api/applications/",
                           headers=_br_headers, timeout=15)
        resp.raise_for_status()
        db = None
        for app in resp.json():
            if (app.get("name") == "Dependency Security Audit 2025Q1"
                    and app.get("type") == "database"):
                db = app
                break
        if not db:
            check("1. Baserow DB exists", 1, False, "database not found")
            return
        check("1. Baserow DB exists", 1, True)

        # Also find the CVE Registry table for subsequent checks
        tables_resp = req_lib.get(
            f"{BASEROW_URL}/api/database/tables/database/{db['id']}/",
            headers=_br_headers, timeout=15,
        )
        tables_resp.raise_for_status()
        for t in tables_resp.json():
            if t["name"] == "CVE Registry":
                _br_table_id = t["id"]
                break
    except Exception as e:
        check("1. Baserow DB exists", 1, False, f"exception: {e}")


def check_2_baserow_table_and_fields() -> None:
    """Verify table 'CVE Registry' exists with required fields."""
    global _br_fields
    try:
        if _br_table_id is None:
            check("2. CVE Registry table with fields", 1, False, "table not found")
            return
        fields_resp = req_lib.get(
            f"{BASEROW_URL}/api/database/fields/table/{_br_table_id}/",
            headers=_br_headers, timeout=15,
        )
        fields_resp.raise_for_status()
        _br_fields = {f["name"]: f for f in fields_resp.json()}

        required = ["CVE ID", "Project", "Library Name", "Vulnerable Version",
                     "Fixed Version", "CVSS Score", "Severity", "Discovered Date"]
        missing = [f for f in required if f not in _br_fields]
        check("2. CVE Registry table with fields", 1, len(missing) == 0,
              f"missing fields: {missing}" if missing else "")
    except Exception as e:
        check("2. CVE Registry table with fields", 1, False, f"exception: {e}")



# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_baserow_db_exists()
    check_2_baserow_table_and_fields()

    total = sum(w for _, w, _, _ in _checks)
    earned = sum(w for _, w, p, _ in _checks if p)
    all_pass = all(p for _, _, p, _ in _checks) and bool(_checks)
    score = (earned / total) if total else 0.0

    print(
        f"SCORE: {score:.3f}  PASS: {all_pass}  ({earned}/{total})",
        file=sys.stderr,
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
