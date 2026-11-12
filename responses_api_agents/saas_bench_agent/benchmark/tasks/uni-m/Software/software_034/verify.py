# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_034/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-034-I2: Multi-Project Coverage Audit across code-server, Baserow, OpenProject.

Checks: 14 weighted checks across code-server, baserow, openproject.
Strategy: Baserow REST API, code-server docker exec filesystem, OpenProject REST API.

Required env vars:
  SERVER_HOSTNAME,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER.
"""

import os
import sys
import json
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

for var_name, var_val in [
    ("CODE_SERVER_PORT", CODE_SERVER_PORT),
    ("CODE_SERVER_CONTAINER", CODE_SERVER_CONTAINER),
    ("BASEROW_PORT", BASEROW_PORT),
    ("BASEROW_CONTAINER", BASEROW_CONTAINER),
    ("BASEROW_DB_CONTAINER", BASEROW_DB_CONTAINER),
    ("OPENPROJECT_PORT", OPENPROJECT_PORT),
    ("OPENPROJECT_CONTAINER", OPENPROJECT_CONTAINER),
]:
    if not var_val:
        print(f"FATAL: {var_name} not set", file=sys.stderr)
        sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"

# ── Expected data ─────────────────────────────────────────────────────────────
MODULES_BY_PROJECT = {
    "blog-engine": ["src/middleware/logger.js", "src/services/markdownRenderer.js"],
    "tabler": ["core/js/tabler.ts", "core/scss/"],
    "weather-dashboard": ["src/services/geocoding.ts", "src/utils/constants.ts"],
}

# Rows ordered by project alpha then module alpha
EXPECTED_ROWS = []
_idx = 1
for proj in sorted(MODULES_BY_PROJECT.keys()):
    for mod in sorted(MODULES_BY_PROJECT[proj]):
        EXPECTED_ROWS.append({
            "entry_id": f"CV-{_idx:03d}",
            "project": proj,
            "module_path": mod,
        })
        _idx += 1

PROJECTS = sorted(MODULES_BY_PROJECT.keys())
DB_NAME = "Coverage Audit Sprint 14 2026"
TABLE1_NAME = "Coverage By Module"
TABLE2_NAME = "Project Coverage Summary"
VIEW_NAME = "Remediation Queue"
AUDIT_DATE = "2026-05-20"
REPORT_PATH = "devops-configs/docs/coverage-audit-2026-05-20.md"
OP_PROJECT_NAME = "API Gateway"
QA_OWNER = "Bob Martinez"
THRESHOLD = 70

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


def baserow_auth() -> str:
    """Get Baserow JWT token."""
    resp = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["token"]


def baserow_get(token: str, path: str, params: dict | None = None) -> dict:
    resp = requests.get(
        f"{BASEROW_URL}/api/{path}",
        headers={"Authorization": f"JWT {token}"},
        params=params or {},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ── Shared state across checks ───────────────────────────────────────────────
_bstate: dict = {}


def _init_baserow():
    """Auth and find the database + tables. Populates _bstate."""
    if _bstate:
        return
    token = baserow_auth()
    _bstate["token"] = token

    # List all applications (databases)
    apps_resp = requests.get(
        f"{BASEROW_URL}/api/applications/",
        headers={"Authorization": f"JWT {token}"},
        timeout=15,
    )
    apps_resp.raise_for_status()
    apps = apps_resp.json()
    db = None
    for a in apps:
        if a.get("name") == DB_NAME and a.get("type") == "database":
            db = a
            break
    _bstate["db"] = db
    if not db:
        return

    # List tables
    tables_resp = baserow_get(token, f"database/tables/database/{db['id']}/")
    _bstate["tables"] = tables_resp
    for t in tables_resp:
        if t["name"] == TABLE1_NAME:
            _bstate["table1"] = t
        elif t["name"] == TABLE2_NAME:
            _bstate["table2"] = t


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_baserow_db_exists() -> None:
    """Baserow database 'Coverage Audit Sprint 14 2026' exists."""
    try:
        _init_baserow()
        db = _bstate.get("db")
        check("1. Baserow database exists", 2, db is not None,
              f"expected '{DB_NAME}'" if not db else f"found id={db['id']}")
    except Exception as e:
        check("1. Baserow database exists", 2, False, f"exception: {e}")


def check_2_table1_fields() -> None:
    """Table 'Coverage By Module' exists with correct fields."""
    try:
        _init_baserow()
        t1 = _bstate.get("table1")
        if not t1:
            check("2. Coverage By Module table + fields", 2, False, "table not found")
            return
        token = _bstate["token"]
        fields_resp = baserow_get(token, f"database/fields/table/{t1['id']}/")
        field_names = {f["name"] for f in fields_resp}
        expected_fields = {"Entry ID", "Project", "Module Path", "Coverage Pct", "Captured At", "Below Threshold"}
        missing = expected_fields - field_names
        check("2. Coverage By Module table + fields", 2, len(missing) == 0,
              f"missing fields: {missing}" if missing else f"all {len(expected_fields)} fields present")
    except Exception as e:
        check("2. Coverage By Module table + fields", 2, False, f"exception: {e}")


def check_3_table1_row_count_and_ids() -> None:
    """Coverage By Module has exactly 6 rows with correct Entry IDs and ordering."""
    try:
        _init_baserow()
        t1 = _bstate.get("table1")
        if not t1:
            check("3. Coverage By Module rows + IDs", 2, False, "table not found")
            return
        token = _bstate["token"]
        rows_resp = baserow_get(token, f"database/rows/table/{t1['id']}/",
                                params={"size": 200, "user_field_names": "true"})
        rows = rows_resp.get("results", [])
        _bstate["table1_rows"] = rows

        if len(rows) != 6:
            check("3. Coverage By Module rows + IDs", 2, False,
                  f"expected 6 rows, got {len(rows)}")
            return

        actual_ids = [r.get("Entry ID", "") for r in rows]
        expected_ids = [er["entry_id"] for er in EXPECTED_ROWS]
        ok = actual_ids == expected_ids
        check("3. Coverage By Module rows + IDs", 2, ok,
              f"IDs: {actual_ids}" if not ok else "6 rows, IDs CV-001..CV-006 in order")
    except Exception as e:
        check("3. Coverage By Module rows + IDs", 2, False, f"exception: {e}")


def check_4_table1_project_module() -> None:
    """Coverage By Module rows have correct Project and Module Path values."""
    try:
        rows = _bstate.get("table1_rows", [])
        if not rows:
            check("4. Coverage By Module Project+Module values", 2, False, "no rows loaded")
            return
        mismatches = []
        for i, row in enumerate(rows):
            exp = EXPECTED_ROWS[i] if i < len(EXPECTED_ROWS) else None
            if not exp:
                mismatches.append(f"row {i}: unexpected extra row")
                continue
            # Project may be a dict (single-select) or string
            proj_val = row.get("Project", "")
            if isinstance(proj_val, dict):
                proj_val = proj_val.get("value", "")
            mod_val = row.get("Module Path", "")
            if proj_val != exp["project"]:
                mismatches.append(f"row {i}: project expected '{exp['project']}', got '{proj_val}'")
            if mod_val != exp["module_path"]:
                mismatches.append(f"row {i}: module expected '{exp['module_path']}', got '{mod_val}'")
        ok = len(mismatches) == 0
        check("4. Coverage By Module Project+Module values", 2, ok,
              "; ".join(mismatches[:3]) if mismatches else "all 6 rows match")
    except Exception as e:
        check("4. Coverage By Module Project+Module values", 2, False, f"exception: {e}")


def check_5_table1_captured_at() -> None:
    """Coverage By Module rows have Captured At = 2026-05-20."""
    try:
        rows = _bstate.get("table1_rows", [])
        if not rows:
            check("5. Coverage By Module Captured At", 1, False, "no rows loaded")
            return
        bad = []
        for i, row in enumerate(rows):
            val = str(row.get("Captured At", ""))
            if not val.startswith(AUDIT_DATE):
                bad.append(f"row {i}: '{val}'")
        ok = len(bad) == 0
        check("5. Coverage By Module Captured At", 1, ok,
              f"wrong dates: {'; '.join(bad[:3])}" if bad else f"all rows have {AUDIT_DATE}")
    except Exception as e:
        check("5. Coverage By Module Captured At", 1, False, f"exception: {e}")


def check_6_table1_below_threshold() -> None:
    """Coverage By Module Below Threshold correctly set (true when Coverage Pct < 70)."""
    try:
        rows = _bstate.get("table1_rows", [])
        if not rows:
            check("6. Coverage By Module Below Threshold logic", 2, False, "no rows loaded")
            return
        bad = []
        for i, row in enumerate(rows):
            cov = row.get("Coverage Pct")
            bt = row.get("Below Threshold")
            if cov is None:
                bad.append(f"row {i}: Coverage Pct missing")
                continue
            try:
                cov_f = float(cov)
            except (TypeError, ValueError):
                bad.append(f"row {i}: Coverage Pct not numeric: {cov}")
                continue
            expected_bt = cov_f < THRESHOLD
            if isinstance(bt, bool):
                actual_bt = bt
            elif bt in (True, "true", "True", 1):
                actual_bt = True
            else:
                actual_bt = False
            if actual_bt != expected_bt:
                bad.append(f"row {i}: cov={cov_f}, BT expected {expected_bt} got {actual_bt}")
        ok = len(bad) == 0
        check("6. Coverage By Module Below Threshold logic", 2, ok,
              "; ".join(bad[:3]) if bad else "all rows correct")
    except Exception as e:
        check("6. Coverage By Module Below Threshold logic", 2, False, f"exception: {e}")


def check_7_table2_exists_and_rows() -> None:
    """Project Coverage Summary exists with 3 rows, correct Module Count."""
    try:
        _init_baserow()
        t2 = _bstate.get("table2")
        if not t2:
            check("7. Project Coverage Summary table + rows", 2, False, "table not found")
            return
        token = _bstate["token"]
        rows_resp = baserow_get(token, f"database/rows/table/{t2['id']}/",
                                params={"size": 200, "user_field_names": "true"})
        rows = rows_resp.get("results", [])
        _bstate["table2_rows"] = rows

        if len(rows) != 3:
            check("7. Project Coverage Summary table + rows", 2, False,
                  f"expected 3 rows, got {len(rows)}")
            return

        bad = []
        for row in rows:
            proj_val = row.get("Project", "")
            if isinstance(proj_val, dict):
                proj_val = proj_val.get("value", "")
            mc = row.get("Module Count")
            expected_mc = len(MODULES_BY_PROJECT.get(proj_val, []))
            if expected_mc == 0:
                bad.append(f"unknown project '{proj_val}'")
            elif mc is not None and int(mc) != expected_mc:
                bad.append(f"{proj_val}: Module Count expected {expected_mc}, got {mc}")

        ok = len(bad) == 0
        check("7. Project Coverage Summary table + rows", 2, ok,
              "; ".join(bad) if bad else "3 rows, Module Count correct")
    except Exception as e:
        check("7. Project Coverage Summary table + rows", 2, False, f"exception: {e}")


def check_8_table2_avg_coverage() -> None:
    """Project Coverage Summary Avg Coverage Pct matches average of module coverages."""
    try:
        t1_rows = _bstate.get("table1_rows", [])
        t2_rows = _bstate.get("table2_rows", [])
        if not t1_rows or not t2_rows:
            check("8. Project Coverage Summary Avg Coverage", 2, False, "rows not loaded")
            return

        # Compute expected averages from table 1
        project_covs: dict[str, list[float]] = {}
        for row in t1_rows:
            proj_val = row.get("Project", "")
            if isinstance(proj_val, dict):
                proj_val = proj_val.get("value", "")
            cov = row.get("Coverage Pct")
            if cov is not None:
                try:
                    project_covs.setdefault(proj_val, []).append(float(cov))
                except (TypeError, ValueError):
                    pass

        bad = []
        for row in t2_rows:
            proj_val = row.get("Project", "")
            if isinstance(proj_val, dict):
                proj_val = proj_val.get("value", "")
            avg = row.get("Avg Coverage Pct")
            if avg is None:
                bad.append(f"{proj_val}: Avg Coverage Pct missing")
                continue
            covs = project_covs.get(proj_val, [])
            if not covs:
                bad.append(f"{proj_val}: no coverage data in table 1")
                continue
            expected_avg = round(sum(covs) / len(covs), 2)
            try:
                actual_avg = round(float(avg), 2)
            except (TypeError, ValueError):
                bad.append(f"{proj_val}: avg not numeric: {avg}")
                continue
            if abs(actual_avg - expected_avg) > 0.02:
                bad.append(f"{proj_val}: expected avg {expected_avg}, got {actual_avg}")

        ok = len(bad) == 0
        check("8. Project Coverage Summary Avg Coverage", 2, ok,
              "; ".join(bad) if bad else "all averages match")
    except Exception as e:
        check("8. Project Coverage Summary Avg Coverage", 2, False, f"exception: {e}")


def check_9_remediation_queue_view() -> None:
    """Remediation Queue view exists on Coverage By Module."""
    try:
        _init_baserow()
        t1 = _bstate.get("table1")
        if not t1:
            check("9. Remediation Queue view exists", 2, False, "table1 not found")
            return
        token = _bstate["token"]
        views_resp = baserow_get(token, f"database/views/table/{t1['id']}/")
        view_names = [v["name"] for v in views_resp]
        found = VIEW_NAME in view_names
        check("9. Remediation Queue view exists", 2, found,
              f"views: {view_names}" if not found else "found")
    except Exception as e:
        check("9. Remediation Queue view exists", 2, False, f"exception: {e}")


def check_10_report_file_exists() -> None:
    """File devops-configs/docs/coverage-audit-2026-05-20.md exists in code-server."""
    try:
        rc, out, err = docker_exec(CODE_SERVER_CONTAINER,
                                   "test", "-f", f"/home/coder/project/{REPORT_PATH}")
        found = rc == 0
        if not found:
            # Try alternate paths
            rc2, out2, _ = docker_exec(CODE_SERVER_CONTAINER,
                                       "find", "/home/coder", "-path",
                                       f"*/{REPORT_PATH}", "-type", "f")
            if rc2 == 0 and out2.strip():
                found = True
                _bstate["report_real_path"] = out2.strip().split("\n")[0]
        check("10. Report file exists in code-server", 1, found,
              "" if found else f"file not found at {REPORT_PATH}")
    except Exception as e:
        check("10. Report file exists in code-server", 1, False, f"exception: {e}")


def check_11_report_file_content() -> None:
    """Report file has correct structure: header, projects line, per-project lines."""
    try:
        rpath = _bstate.get("report_real_path", f"/home/coder/project/{REPORT_PATH}")
        rc, out, err = docker_exec(CODE_SERVER_CONTAINER, "cat", rpath)
        if rc != 0:
            check("11. Report file content structure", 2, False, f"cannot read file: {err.strip()}")
            return

        lines = out.strip().split("\n")
        bad = []

        # Line 1: # Coverage Audit — 2026-05-20
        if len(lines) < 1 or "Coverage Audit" not in lines[0] or AUDIT_DATE not in lines[0]:
            bad.append(f"line 1 wrong: '{lines[0] if lines else ''}'")

        # Line 2: Projects: blog-engine,tabler,weather-dashboard (comma-separated sorted)
        if len(lines) < 2:
            bad.append("line 2 missing")
        else:
            l2 = lines[1]
            if not l2.startswith("Projects:"):
                bad.append(f"line 2 doesn't start with 'Projects:': '{l2}'")
            else:
                for p in PROJECTS:
                    if p not in l2:
                        bad.append(f"line 2 missing project '{p}'")

        # Lines 3-5: one per project in alpha order
        if len(lines) < 5:
            bad.append(f"expected at least 5 lines, got {len(lines)}")
        else:
            for i, proj in enumerate(PROJECTS):
                line = lines[2 + i]
                if proj not in line:
                    bad.append(f"line {3+i} missing project '{proj}': '{line}'")
                if "avg" not in line.lower() and "%" not in line:
                    bad.append(f"line {3+i} missing coverage info")

        ok = len(bad) == 0
        check("11. Report file content structure", 2, ok,
              "; ".join(bad[:3]) if bad else "correct 5-line structure")
    except Exception as e:
        check("11. Report file content structure", 2, False, f"exception: {e}")


def _op_query(sql: str) -> str:
    """Run a SQL query against OpenProject's embedded Postgres."""
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER, "bash", "-c",
        f"PGPASSWORD=openproject psql -U openproject -h 127.0.0.1 -d openproject -t -A -c \"{sql}\"",
        timeout=20,
    )
    if rc != 0:
        raise RuntimeError(f"psql failed: {err.strip()}")
    return out.strip()


def check_12_op_work_packages_exist() -> None:
    """OpenProject 'API Gateway' has 5 Task work packages with 'Raise coverage' subjects."""
    try:
        # Find project id
        pid_out = _op_query(f"SELECT id FROM projects WHERE name = '{OP_PROJECT_NAME}';")
        if not pid_out:
            check("12. OpenProject 5 Task work packages exist", 2, False,
                  f"project '{OP_PROJECT_NAME}' not found in DB")
            return
        pid = pid_out.strip().split("\n")[0]

        # Find Task-type work packages with 'Raise coverage' in subject
        sql = (
            f"SELECT wp.id, wp.subject "
            f"FROM work_packages wp "
            f"JOIN types t ON t.id = wp.type_id "
            f"WHERE wp.project_id = {pid} "
            f"AND t.name = 'Task' "
            f"AND wp.subject LIKE 'Raise coverage:%' "
            f"ORDER BY wp.id;"
        )
        rows_out = _op_query(sql)
        rows = [line for line in rows_out.split("\n") if line.strip()] if rows_out else []
        _bstate["op_coverage_wp_ids"] = [r.split("|")[0] for r in rows]
        _bstate["op_coverage_wp_subjects"] = [r.split("|", 1)[1] if "|" in r else "" for r in rows]
        _bstate["op_project_id"] = pid

        ok = len(rows) == 5
        check("12. OpenProject 5 Task work packages exist", 2, ok,
              f"found {len(rows)} 'Raise coverage' Task WPs" if not ok else "5 found")
    except Exception as e:
        check("12. OpenProject 5 Task work packages exist", 2, False, f"exception: {e}")


def check_13_op_assignee_priority() -> None:
    """Work packages have assignee=Bob Martinez and priority=High."""
    try:
        wp_ids = _bstate.get("op_coverage_wp_ids", [])
        if not wp_ids:
            check("13. WP assignee + priority", 2, False, "no coverage WPs found")
            return

        ids_csv = ",".join(wp_ids)
        sql = (
            f"SELECT wp.id, "
            f"(SELECT u.firstname || ' ' || u.lastname FROM users u WHERE u.id = wp.assigned_to_id) as assignee, "
            f"(SELECT e.name FROM enumerations e WHERE e.id = wp.priority_id) as priority "
            f"FROM work_packages wp "
            f"WHERE wp.id IN ({ids_csv});"
        )
        rows_out = _op_query(sql)
        rows = [line for line in rows_out.split("\n") if line.strip()] if rows_out else []

        bad = []
        for row in rows:
            parts = row.split("|")
            if len(parts) < 3:
                bad.append(f"unexpected row format: {row}")
                continue
            wp_id, assignee, priority = parts[0], parts[1], parts[2]
            if QA_OWNER not in assignee:
                bad.append(f"WP {wp_id}: assignee='{assignee}' expected '{QA_OWNER}'")
            if priority != "High":
                bad.append(f"WP {wp_id}: priority='{priority}' expected 'High'")

        ok = len(bad) == 0
        check("13. WP assignee + priority", 2, ok,
              "; ".join(bad[:3]) if bad else f"all WPs: assignee={QA_OWNER}, priority=High")
    except Exception as e:
        check("13. WP assignee + priority", 2, False, f"exception: {e}")


def check_14_op_description() -> None:
    """Work packages have description matching 'Current: <X>%; Target: 70%; Audit: 2026-05-20'."""
    try:
        wp_ids = _bstate.get("op_coverage_wp_ids", [])
        if not wp_ids:
            check("14. WP description format", 2, False, "no coverage WPs found")
            return

        ids_csv = ",".join(wp_ids)
        # OpenProject stores description as markdown in description column (text)
        sql = (
            f"SELECT wp.id, wp.subject, "
            f"COALESCE(j.data->>'description', '') as desc_raw "
            f"FROM work_packages wp "
            f"LEFT JOIN (SELECT journable_id, data FROM journals "
            f"WHERE journable_type = 'WorkPackage' ORDER BY id DESC LIMIT 1000) j "
            f"ON j.journable_id = wp.id "
            f"WHERE wp.id IN ({ids_csv});"
        )
        # Simpler approach: just query the work_packages table directly
        sql2 = (
            f"SELECT id, subject, "
            f"COALESCE(description, '') "
            f"FROM work_packages WHERE id IN ({ids_csv});"
        )
        rows_out = _op_query(sql2)
        rows = [line for line in rows_out.split("\n") if line.strip()] if rows_out else []

        bad = []
        for row in rows:
            parts = row.split("|", 2)
            if len(parts) < 3:
                bad.append(f"unexpected row format: {row}")
                continue
            wp_id, subj, desc_text = parts[0], parts[1], parts[2]
            if "Target: 70%" not in desc_text:
                bad.append(f"WP {wp_id}: missing 'Target: 70%' in desc")
            elif f"Audit: {AUDIT_DATE}" not in desc_text:
                bad.append(f"WP {wp_id}: missing 'Audit: {AUDIT_DATE}' in desc")
            elif "Current:" not in desc_text:
                bad.append(f"WP {wp_id}: missing 'Current:' in desc")

        ok = len(bad) == 0
        check("14. WP description format", 2, ok,
              "; ".join(bad[:3]) if bad else "all descriptions match expected format")
    except Exception as e:
        check("14. WP description format", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_baserow_db_exists()
    check_2_table1_fields()
    check_3_table1_row_count_and_ids()
    check_4_table1_project_module()
    check_5_table1_captured_at()
    check_6_table1_below_threshold()
    check_7_table2_exists_and_rows()
    check_8_table2_avg_coverage()
    check_9_remediation_queue_view()
    check_10_report_file_exists()
    check_11_report_file_content()
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
