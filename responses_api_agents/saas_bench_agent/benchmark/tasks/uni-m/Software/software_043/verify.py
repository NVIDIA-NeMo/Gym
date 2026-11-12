# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_043/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-043-I4: Audit Prometheus alerts and reconcile SLOs across
code-server, Baserow, and OpenProject.

Checks: 14 weighted checks (21 points total).
Strategy: docker exec (code-server), REST API (Baserow, OpenProject).

Required env vars:
  SERVER_HOSTNAME,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import os
import sys
import re
import subprocess
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_PORT = os.environ.get("CODE_SERVER_PORT")
CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")
OPENPROJECT_PORT = os.environ.get("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

_missing = []
for _var in [
    "CODE_SERVER_PORT", "CODE_SERVER_CONTAINER",
    "BASEROW_PORT", "BASEROW_CONTAINER",
    "OPENPROJECT_PORT", "OPENPROJECT_CONTAINER",
]:
    if not os.environ.get(_var):
        _missing.append(_var)
if _missing:
    print(f"FATAL: missing env vars: {', '.join(_missing)}", file=sys.stderr)
    sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"

# ── Constants from task spec ──────────────────────────────────────────────────
SERVICE_LIST = ["analytics-service", "media-service", "reporting-service"]  # sorted

SLO_TARGETS = {
    "analytics-service::EventIngestLagHigh": 10.000,
    "analytics-service::QueryLatencyP95": 1.500,
    "reporting-service::ReportRenderLatencyP99": 3.000,
    "reporting-service::ReportGenerationErrorRate": 0.004,
    "media-service::UploadLatencyP95": 0.900,
    "media-service::TranscodeFailureRateHigh": 0.020,
}

SEVERITY_MAP = {
    "EventIngestLagHigh": "Ticket",
    "QueryLatencyP95": "Info",
    "ReportRenderLatencyP99": "Ticket",
    "ReportGenerationErrorRate": "Page",
    "UploadLatencyP95": "Ticket",
    "TranscodeFailureRateHigh": "Page",
}

MISMATCH_TOLERANCE = 0.002
AUDIT_DATE = "2026-07-22"

# Expected order: Service alphabetical, then Alert Name alphabetical within service
EXPECTED_ORDER = [
    ("analytics-service", "EventIngestLagHigh"),
    ("analytics-service", "QueryLatencyP95"),
    ("media-service", "TranscodeFailureRateHigh"),
    ("media-service", "UploadLatencyP95"),
    ("reporting-service", "ReportGenerationErrorRate"),
    ("reporting-service", "ReportRenderLatencyP99"),
]

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
    resp = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    return {"Authorization": f"JWT {token}"}


def op_sql(query: str) -> str:
    """Run a SQL query against the OpenProject embedded Postgres DB."""
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER,
        "psql", "postgres://openproject:openproject@127.0.0.1/openproject",
        "-t", "-A", "-c", query,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"op_sql failed: {err.strip()}")
    return out.strip()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_audit_file_exists() -> None:
    """Check that alert-audit-2026-07-22.md exists in code-server."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "test", "-f", "/home/coder/project/devops-configs/docs/alert-audit-2026-07-22.md",
        )
        check("1. Audit markdown file exists in code-server", 1, rc == 0,
              "file not found" if rc != 0 else "")
    except Exception as e:
        check("1. Audit markdown file exists in code-server", 1, False, f"exception: {e}")


def check_2_audit_file_content() -> None:
    """Check that audit file has correct 3-line structure."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "cat", "/home/coder/project/devops-configs/docs/alert-audit-2026-07-22.md",
        )
        if rc != 0:
            check("2. Audit file has correct 3-line content", 2, False, "file not readable")
            return

        lines = out.strip().split("\n")
        issues = []

        if len(lines) < 3:
            issues.append(f"expected 3 lines, got {len(lines)}")
        else:
            # Line 1: heading
            if "Alert Rule Audit" not in lines[0] or "2026-07-22" not in lines[0]:
                issues.append(f"line 1 mismatch: {lines[0]!r}")
            # Line 2: services scanned
            if "Services scanned:" not in lines[1]:
                issues.append(f"line 2 missing 'Services scanned:': {lines[1]!r}")
            else:
                # Check that all 3 services are mentioned
                for svc in SERVICE_LIST:
                    if svc not in lines[1]:
                        issues.append(f"line 2 missing service {svc}")
            # Line 3: counts
            if not re.search(r"Total rules:\s*\d+", lines[2]):
                issues.append(f"line 3 missing 'Total rules': {lines[2]!r}")
            if not re.search(r"Mismatched:\s*\d+", lines[2]):
                issues.append(f"line 3 missing 'Mismatched': {lines[2]!r}")
            if not re.search(r"Paging rules:\s*\d+", lines[2]):
                issues.append(f"line 3 missing 'Paging rules': {lines[2]!r}")

        check("2. Audit file has correct 3-line content", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("2. Audit file has correct 3-line content", 2, False, f"exception: {e}")


# ── Baserow checks ───────────────────────────────────────────────────────────

_baserow_headers = None
_baserow_table_id = None
_baserow_rows = None


def _init_baserow() -> bool:
    """Authenticate with Baserow. Returns True if auth succeeded."""
    global _baserow_headers
    if _baserow_headers is not None:
        return bool(_baserow_headers)
    try:
        _baserow_headers = baserow_auth()
        return True
    except Exception:
        _baserow_headers = {}
        return False


def _find_baserow_db_and_table() -> tuple[bool, str]:
    """Find 'Analytics Stack Alert Audit' database and 'Alert Rule Audit' table."""
    global _baserow_table_id
    try:
        # List all applications (databases)
        resp = requests.get(
            f"{BASEROW_URL}/api/applications/",
            headers=_baserow_headers, timeout=15,
        )
        resp.raise_for_status()
        apps = resp.json()

        db = None
        for app in apps:
            if app.get("name") == "Analytics Stack Alert Audit":
                db = app
                break

        if not db:
            return False, "database 'Analytics Stack Alert Audit' not found"

        # Find table
        tables = db.get("tables", [])
        if not tables:
            # Fetch tables explicitly
            resp2 = requests.get(
                f"{BASEROW_URL}/api/database/tables/database/{db['id']}/",
                headers=_baserow_headers, timeout=15,
            )
            resp2.raise_for_status()
            tables = resp2.json()

        for t in tables:
            if t.get("name") == "Alert Rule Audit":
                _baserow_table_id = t["id"]
                return True, ""

        return False, "table 'Alert Rule Audit' not found in database"
    except Exception as e:
        return False, f"exception: {e}"


def _get_baserow_rows() -> list[dict]:
    """Fetch all rows from the Alert Rule Audit table."""
    global _baserow_rows
    if _baserow_rows is not None:
        return _baserow_rows
    if _baserow_table_id is None:
        return []
    try:
        resp = requests.get(
            f"{BASEROW_URL}/api/database/rows/table/{_baserow_table_id}/",
            headers=_baserow_headers,
            params={"size": 100},
            timeout=15,
        )
        resp.raise_for_status()
        _baserow_rows = resp.json().get("results", [])
        return _baserow_rows
    except Exception:
        _baserow_rows = []
        return []


def _get_baserow_fields() -> list[dict]:
    """Fetch field definitions for the table."""
    if _baserow_table_id is None:
        return []
    try:
        resp = requests.get(
            f"{BASEROW_URL}/api/database/fields/table/{_baserow_table_id}/",
            headers=_baserow_headers, timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def _field_name_map(fields: list[dict]) -> dict[str, dict]:
    """Map field name -> field metadata."""
    return {f["name"]: f for f in fields}


def _row_value(row: dict, field_name: str, fields: list[dict]) -> object:
    """Get a row's value by field name. Handles Baserow's field_<id> keys."""
    for f in fields:
        if f["name"] == field_name:
            key = f"field_{f['id']}"
            return row.get(key)
    return None


def check_3_baserow_database_exists() -> None:
    """Check that Baserow database 'Analytics Stack Alert Audit' exists."""
    try:
        if not _init_baserow():
            check("3. Baserow database exists", 1, False, "auth failed")
            return

        # List all applications (databases)
        resp = requests.get(
            f"{BASEROW_URL}/api/applications/",
            headers=_baserow_headers, timeout=15,
        )
        resp.raise_for_status()
        apps = resp.json()
        db_names = [a.get("name") for a in apps]
        found = "Analytics Stack Alert Audit" in db_names
        check("3. Baserow database exists", 1, found,
              f"databases: {db_names}" if not found else "")
    except Exception as e:
        check("3. Baserow database exists", 1, False, f"exception: {e}")


def check_4_baserow_table_fields() -> None:
    """Check table has required fields with correct types."""
    try:
        if _baserow_table_id is None:
            check("4. Table 'Alert Rule Audit' with correct fields", 2, False,
                  "table not found")
            return
        fields = _get_baserow_fields()
        field_map = _field_name_map(fields)

        required = {
            "Rule ID": "text",
            "Service": "single_select",
            "Alert Name": "text",
            "Threshold Value": "number",
            "SLO Target": "number",
            "Mismatch": "boolean",
            "Severity": "single_select",
            "Captured At": "date",
        }

        issues = []
        for fname, expected_type in required.items():
            if fname not in field_map:
                issues.append(f"missing field '{fname}'")
            else:
                actual_type = field_map[fname].get("type", "")
                if expected_type and actual_type != expected_type:
                    issues.append(f"'{fname}' type={actual_type}, expected {expected_type}")

        check("4. Table 'Alert Rule Audit' with correct fields", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("4. Table 'Alert Rule Audit' with correct fields", 2, False, f"exception: {e}")


def check_5_baserow_row_count_and_ids() -> None:
    """Check 6 rows with Rule IDs AR-01..AR-06 in correct order."""
    try:
        rows = _get_baserow_rows()
        fields = _get_baserow_fields()
        if not rows:
            check("5. 6 rows with correct Rule IDs", 2, False, "no rows found")
            return

        issues = []
        if len(rows) != 6:
            issues.append(f"expected 6 rows, got {len(rows)}")

        # Check Rule IDs
        rule_ids = []
        for r in rows:
            rid = _row_value(r, "Rule ID", fields)
            rule_ids.append(rid)

        expected_ids = [f"AR-{i:02d}" for i in range(1, 7)]
        if rule_ids != expected_ids:
            issues.append(f"Rule IDs: {rule_ids}, expected {expected_ids}")

        # Check ordering: service alpha then alert name alpha
        for i, r in enumerate(rows):
            svc_val = _row_value(r, "Service", fields)
            svc = svc_val.get("value", "") if isinstance(svc_val, dict) else str(svc_val or "")
            alert = _row_value(r, "Alert Name", fields) or ""
            if i < len(EXPECTED_ORDER):
                exp_svc, exp_alert = EXPECTED_ORDER[i]
                if svc != exp_svc or alert != exp_alert:
                    issues.append(f"row {i+1}: got ({svc}, {alert}), expected ({exp_svc}, {exp_alert})")

        check("5. 6 rows with correct Rule IDs and order", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("5. 6 rows with correct Rule IDs and order", 2, False, f"exception: {e}")


def check_6_slo_targets() -> None:
    """Check SLO Target values match the spec."""
    try:
        rows = _get_baserow_rows()
        fields = _get_baserow_fields()
        if not rows:
            check("6. SLO Target values correct", 2, False, "no rows")
            return

        issues = []
        for r in rows:
            svc_val = _row_value(r, "Service", fields)
            svc = svc_val.get("value", "") if isinstance(svc_val, dict) else str(svc_val or "")
            alert = _row_value(r, "Alert Name", fields) or ""
            slo_actual = _row_value(r, "SLO Target", fields)
            key = f"{svc}::{alert}"
            expected_slo = SLO_TARGETS.get(key)
            if expected_slo is None:
                issues.append(f"unknown key {key}")
                continue
            try:
                slo_num = float(slo_actual) if slo_actual is not None else None
            except (TypeError, ValueError):
                slo_num = None
            if slo_num is None or abs(slo_num - expected_slo) > 0.0005:
                issues.append(f"{key}: SLO={slo_actual}, expected {expected_slo}")

        check("6. SLO Target values correct", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("6. SLO Target values correct", 2, False, f"exception: {e}")


def check_7_mismatch_values() -> None:
    """Check Mismatch = abs(Threshold - SLO) > 0.002."""
    try:
        rows = _get_baserow_rows()
        fields = _get_baserow_fields()
        if not rows:
            check("7. Mismatch values computed correctly", 2, False, "no rows")
            return

        issues = []
        for r in rows:
            svc_val = _row_value(r, "Service", fields)
            svc = svc_val.get("value", "") if isinstance(svc_val, dict) else str(svc_val or "")
            alert = _row_value(r, "Alert Name", fields) or ""
            threshold = _row_value(r, "Threshold Value", fields)
            slo = _row_value(r, "SLO Target", fields)
            mismatch = _row_value(r, "Mismatch", fields)

            try:
                t_val = float(threshold)
                s_val = float(slo)
            except (TypeError, ValueError):
                issues.append(f"{svc}/{alert}: non-numeric threshold/slo")
                continue

            expected_mismatch = abs(t_val - s_val) > MISMATCH_TOLERANCE
            actual_mismatch = bool(mismatch)
            if actual_mismatch != expected_mismatch:
                issues.append(
                    f"{svc}/{alert}: mismatch={actual_mismatch}, "
                    f"expected={expected_mismatch} (threshold={t_val}, slo={s_val})"
                )

        check("7. Mismatch values computed correctly", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("7. Mismatch values computed correctly", 2, False, f"exception: {e}")


def check_8_severity_values() -> None:
    """Check Severity matches the severity map."""
    try:
        rows = _get_baserow_rows()
        fields = _get_baserow_fields()
        if not rows:
            check("8. Severity values correct", 1, False, "no rows")
            return

        issues = []
        for r in rows:
            alert = _row_value(r, "Alert Name", fields) or ""
            sev_val = _row_value(r, "Severity", fields)
            sev = sev_val.get("value", "") if isinstance(sev_val, dict) else str(sev_val or "")
            expected = SEVERITY_MAP.get(alert, "Ticket")
            if sev != expected:
                issues.append(f"{alert}: severity={sev!r}, expected {expected!r}")

        check("8. Severity values correct", 1, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("8. Severity values correct", 1, False, f"exception: {e}")


def check_9_captured_at() -> None:
    """Check Captured At = 2026-07-22 for all rows."""
    try:
        rows = _get_baserow_rows()
        fields = _get_baserow_fields()
        if not rows:
            check("9. Captured At dates correct", 1, False, "no rows")
            return

        issues = []
        for r in rows:
            rid = _row_value(r, "Rule ID", fields) or "?"
            cap = _row_value(r, "Captured At", fields)
            cap_str = str(cap or "")
            if not cap_str.startswith(AUDIT_DATE):
                issues.append(f"{rid}: date={cap_str!r}, expected {AUDIT_DATE}")

        check("9. Captured At dates correct", 1, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("9. Captured At dates correct", 1, False, f"exception: {e}")


def check_10_mismatched_rules_view() -> None:
    """Check 'Mismatched Rules' Grid view exists filtered to Mismatch=true."""
    try:
        if _baserow_table_id is None:
            check("10. 'Mismatched Rules' view exists", 2, False, "table not found")
            return

        resp = requests.get(
            f"{BASEROW_URL}/api/database/views/table/{_baserow_table_id}/",
            headers=_baserow_headers, timeout=15,
        )
        resp.raise_for_status()
        views = resp.json()

        target_view = None
        for v in views:
            if v.get("name") == "Mismatched Rules":
                target_view = v
                break

        if not target_view:
            check("10. 'Mismatched Rules' view exists", 2, False,
                  f"view not found among {[v.get('name') for v in views]}")
            return

        # Check it's a grid view
        issues = []
        if target_view.get("type") != "grid":
            issues.append(f"type={target_view.get('type')}, expected grid")

        # Check filters
        view_id = target_view["id"]
        try:
            filter_resp = requests.get(
                f"{BASEROW_URL}/api/database/views/{view_id}/filters/",
                headers=_baserow_headers, timeout=15,
            )
            filter_resp.raise_for_status()
            filters = filter_resp.json()

            # There should be a filter on Mismatch = true
            fields = _get_baserow_fields()
            mismatch_field = None
            for f in fields:
                if f["name"] == "Mismatch":
                    mismatch_field = f
                    break

            has_mismatch_filter = False
            for flt in filters:
                if mismatch_field and flt.get("field") == mismatch_field["id"]:
                    has_mismatch_filter = True
                    break

            if not has_mismatch_filter:
                issues.append("no filter on Mismatch field found")
        except Exception as e:
            issues.append(f"could not check filters: {e}")

        check("10. 'Mismatched Rules' view exists with filter", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("10. 'Mismatched Rules' view exists with filter", 2, False, f"exception: {e}")


# ── OpenProject checks (via docker exec DB) ──────────────────────────────────

def check_11_immediate_priority() -> None:
    """Check 'Immediate' priority enumeration exists in OpenProject."""
    try:
        out = op_sql("SELECT name FROM enumerations WHERE type='IssuePriority' ORDER BY position;")
        names = [n.strip() for n in out.split("\n") if n.strip()]
        found = "Immediate" in names
        check("11. 'Immediate' priority exists in OpenProject", 1, found,
              f"priorities: {names}" if not found else "")
    except Exception as e:
        check("11. 'Immediate' priority exists in OpenProject", 1, False, f"exception: {e}")


def _get_mismatch_rows() -> list[tuple[str, str, float, float, str]]:
    """Get (service, alert_name, threshold, slo, severity) for Mismatch=true rows."""
    rows = _get_baserow_rows()
    fields = _get_baserow_fields()
    result = []
    for r in rows:
        mismatch = _row_value(r, "Mismatch", fields)
        if not mismatch:
            continue
        svc_val = _row_value(r, "Service", fields)
        svc = svc_val.get("value", "") if isinstance(svc_val, dict) else str(svc_val or "")
        alert = _row_value(r, "Alert Name", fields) or ""
        threshold = float(_row_value(r, "Threshold Value", fields) or 0)
        slo = float(_row_value(r, "SLO Target", fields) or 0)
        sev_val = _row_value(r, "Severity", fields)
        sev = sev_val.get("value", "") if isinstance(sev_val, dict) else str(sev_val or "")
        result.append((svc, alert, threshold, slo, sev))
    return result


def check_12_op_work_packages_exist() -> None:
    """Check Bug work packages exist for each mismatched row with correct subjects."""
    try:
        mismatch_rows = _get_mismatch_rows()
        if not mismatch_rows:
            check("12. Bug work packages for mismatched rows", 2, False,
                  "no mismatch rows from Baserow to verify against")
            return

        # Query work packages in api-gateway project that start with 'Reconcile alert:'
        out = op_sql(
            "SELECT wp.subject, t.name AS type_name "
            "FROM work_packages wp "
            "JOIN types t ON wp.type_id = t.id "
            "JOIN projects p ON wp.project_id = p.id "
            "WHERE p.identifier = 'api-gateway' "
            "AND wp.subject LIKE 'Reconcile alert:%' "
            "ORDER BY wp.subject;"
        )
        # Parse pipe-separated rows
        wp_data = {}
        for line in out.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                wp_data[parts[0].strip()] = parts[1].strip()

        issues = []
        for svc, alert, _, _, _ in mismatch_rows:
            expected_subject = f"Reconcile alert: {svc}/{alert}"
            if expected_subject not in wp_data:
                issues.append(f"missing WP: {expected_subject!r}")
            elif wp_data[expected_subject] != "Bug":
                issues.append(f"WP '{expected_subject}' type={wp_data[expected_subject]}, expected Bug")

        check("12. Bug work packages for mismatched rows", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("12. Bug work packages for mismatched rows", 2, False, f"exception: {e}")


def check_13_op_assignee_priority() -> None:
    """Check work packages have correct assignee and priority."""
    try:
        mismatch_rows = _get_mismatch_rows()
        if not mismatch_rows:
            check("13. WP assignee and priority correct", 2, False, "no data to verify")
            return

        # Build expected: subject -> priority
        expected = {}
        for svc, alert, _, _, sev in mismatch_rows:
            subj = f"Reconcile alert: {svc}/{alert}"
            expected[subj] = "Immediate" if sev == "Page" else "High"

        out = op_sql(
            "SELECT wp.subject, e.name AS priority, u.login AS assignee "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "LEFT JOIN enumerations e ON wp.priority_id = e.id "
            "LEFT JOIN users u ON wp.assigned_to_id = u.id "
            "WHERE p.identifier = 'api-gateway' "
            "AND wp.subject LIKE 'Reconcile alert:%' "
            "ORDER BY wp.subject;"
        )

        wp_map = {}
        for line in out.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                wp_map[parts[0].strip()] = {
                    "priority": parts[1].strip(),
                    "assignee": parts[2].strip(),
                }

        issues = []
        for subj, exp_priority in expected.items():
            if subj not in wp_map:
                continue  # Already caught in check 12
            actual = wp_map[subj]
            if actual["priority"] != exp_priority:
                issues.append(f"'{subj}': priority={actual['priority']!r}, expected {exp_priority!r}")
            if actual["assignee"] != "observability.sre":
                issues.append(f"'{subj}': assignee={actual['assignee']!r}, expected 'observability.sre'")

        check("13. WP assignee and priority correct", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("13. WP assignee and priority correct", 2, False, f"exception: {e}")


def check_14_op_description() -> None:
    """Check work package descriptions match the exact format."""
    try:
        mismatch_rows = _get_mismatch_rows()
        if not mismatch_rows:
            check("14. WP descriptions correct", 2, False, "no data to verify")
            return

        # Build expected description fragments per subject
        expected_descs = {}
        for svc, alert, threshold, slo, _ in mismatch_rows:
            subj = f"Reconcile alert: {svc}/{alert}"
            t_str = f"{threshold:.3f}"
            s_str = f"{slo:.3f}"
            expected_descs[subj] = {
                "threshold": t_str,
                "slo": s_str,
            }

        out = op_sql(
            "SELECT wp.subject, wp.description "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "WHERE p.identifier = 'api-gateway' "
            "AND wp.subject LIKE 'Reconcile alert:%' "
            "ORDER BY wp.subject;"
        )

        issues = []
        for line in out.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) < 2:
                continue
            subj = parts[0].strip()
            desc = parts[1].strip()

            if subj not in expected_descs:
                continue

            exp = expected_descs[subj]
            desc_clean = " ".join(desc.split())

            key_parts_to_check = [
                ("Current threshold:", exp["threshold"]),
                ("SLO target:", exp["slo"]),
                ("analytics_reporting_media_alerts.yml", None),
                ("Audit: 2026-07-22", None),
            ]
            missing = []
            for label, val in key_parts_to_check:
                search = f"{label} {val}" if val else label
                if search not in desc_clean and label not in desc_clean:
                    missing.append(label)

            if missing:
                issues.append(
                    f"'{subj}': description missing: {missing}; "
                    f"got: {desc_clean[:120]!r}"
                )

        check("14. WP descriptions correct", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("14. WP descriptions correct", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # code-server checks
    check_1_audit_file_exists()
    check_2_audit_file_content()

    # Baserow checks (init once, reuse)
    _init_baserow()
    _find_baserow_db_and_table()
    _get_baserow_rows()

    check_3_baserow_database_exists()
    check_4_baserow_table_fields()
    check_5_baserow_row_count_and_ids()
    check_6_slo_targets()
    check_7_mismatch_values()
    check_8_severity_values()
    check_9_captured_at()
    check_10_mismatched_rules_view()

    # OpenProject checks
    check_11_immediate_priority()
    check_12_op_work_packages_exist()
    check_13_op_assignee_priority()
    check_14_op_description()

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
