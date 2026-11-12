# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_002/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-002-I3: Test Execution Audit for data-analyzer and todo-api

Checks: 12 weighted checks across code-server, baserow, openproject.
Strategy: code-server=filesystem, baserow=REST API, openproject=docker exec psql

Required env vars:
  SERVER_HOSTNAME,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import os
import sys
import json
import re
import subprocess
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

_missing = []
if not CODE_SERVER_CONTAINER:
    _missing.append("CODE_SERVER_CONTAINER")
if not BASEROW_PORT:
    _missing.append("BASEROW_PORT")
if not BASEROW_CONTAINER:
    _missing.append("BASEROW_CONTAINER")
if not OPENPROJECT_CONTAINER:
    _missing.append("OPENPROJECT_CONTAINER")
if _missing:
    print(f"FATAL: missing env vars: {', '.join(_missing)}", file=sys.stderr)
    sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"

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
    """Authenticate to Baserow and return JWT access token."""
    resp = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    # token-auth returns access_token or token depending on version
    return data.get("access_token", data.get("token", ""))


def baserow_get(path: str, token: str) -> dict:
    resp = requests.get(
        f"{BASEROW_URL}{path}",
        headers={"Authorization": f"JWT {token}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def op_sql(query: str) -> str:
    """Run a SQL query against OpenProject's embedded postgres."""
    # Use env var for password and pass query via -c to avoid shell escaping issues
    r = subprocess.run(
        ["docker", "exec", "-e", "PGPASSWORD=openproject",
         OPENPROJECT_CONTAINER,
         "psql", "-h", "localhost", "-U", "openproject", "-d", "openproject",
         "-t", "-A", "-c", query],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"psql error: {r.stderr.strip()}")
    return r.stdout.strip()


# ── Shared state for Baserow checks ──────────────────────────────────────────
_br_token = None
_br_db_id = None
_br_table_id = None
_br_fields = {}   # field name -> field object
_br_rows = []     # list of row dicts


def _init_baserow():
    """Fetch Baserow DB, table, fields, and rows. Called once."""
    global _br_token, _br_db_id, _br_table_id, _br_fields, _br_rows
    _br_token = baserow_auth()

    # Find the database
    apps = baserow_get("/api/applications/", _br_token)
    for app in apps:
        if app.get("name") == "Regression Test Audit March 2026" and app.get("type") == "database":
            _br_db_id = app["id"]
            break

    if not _br_db_id:
        return

    # Find the table
    tables = baserow_get(f"/api/database/tables/database/{_br_db_id}/", _br_token)
    for t in tables:
        if t.get("name") == "Test Execution Audit":
            _br_table_id = t["id"]
            break

    if not _br_table_id:
        return

    # Get fields
    fields_list = baserow_get(f"/api/database/fields/table/{_br_table_id}/", _br_token)
    for f in fields_list:
        _br_fields[f["name"]] = f

    # Get rows
    rows_resp = baserow_get(f"/api/database/rows/table/{_br_table_id}/?user_field_names=true", _br_token)
    _br_rows = rows_resp.get("results", [])


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_projects_exist():
    """Verify data-analyzer and todo-api project dirs exist in code-server."""
    try:
        rc1, out1, _ = docker_exec(CODE_SERVER_CONTAINER, "test", "-d", "/home/coder/data-analyzer")
        rc2, out2, _ = docker_exec(CODE_SERVER_CONTAINER, "test", "-d", "/home/coder/todo-api")
        both = (rc1 == 0 and rc2 == 0)
        detail = ""
        if rc1 != 0:
            detail += "data-analyzer not found; "
        if rc2 != 0:
            detail += "todo-api not found; "
        check("1. Project dirs exist in code-server", 1, both, detail.rstrip("; "))
    except Exception as e:
        check("1. Project dirs exist in code-server", 1, False, f"exception: {e}")


def check_2_baserow_db_exists():
    """Verify Baserow database 'Regression Test Audit March 2026' exists."""
    try:
        _init_baserow()
        check("2. Baserow DB 'Regression Test Audit March 2026' exists", 1,
              _br_db_id is not None,
              "" if _br_db_id else "database not found")
    except Exception as e:
        check("2. Baserow DB 'Regression Test Audit March 2026' exists", 1, False, f"exception: {e}")


def check_3_baserow_table_exists():
    """Verify table 'Test Execution Audit' exists in the DB."""
    check("3. Table 'Test Execution Audit' exists", 1,
          _br_table_id is not None,
          "" if _br_table_id else "table not found")


def check_4_exactly_two_rows():
    """Verify the table has exactly 2 rows."""
    n = len(_br_rows)
    check("4. Table has exactly 2 rows", 2,
          n == 2,
          f"found {n} rows")


def _find_row(project_name: str) -> dict | None:
    """Find a row by Project field value (case-insensitive)."""
    for row in _br_rows:
        val = row.get("Project", "")
        if isinstance(val, str) and val.strip().lower() == project_name.lower():
            return row
    return None


def check_5_data_analyzer_counts():
    """Verify data-analyzer row has integer Tests Passed and Tests Failed."""
    try:
        row = _find_row("data-analyzer")
        if not row:
            check("5. data-analyzer row has Tests Passed/Failed", 2, False, "row not found")
            return
        tp = row.get("Tests Passed")
        tf = row.get("Tests Failed")
        # Baserow may return as string or number
        tp_ok = tp is not None and str(tp).replace("-", "").isdigit()
        tf_ok = tf is not None and str(tf).replace("-", "").isdigit()
        passed = tp_ok and tf_ok
        detail = f"Tests Passed={tp}, Tests Failed={tf}"
        check("5. data-analyzer row has Tests Passed/Failed", 2, passed, detail)
    except Exception as e:
        check("5. data-analyzer row has Tests Passed/Failed", 2, False, f"exception: {e}")


def check_6_todo_api_counts():
    """Verify todo-api row has integer Tests Passed and Tests Failed."""
    try:
        row = _find_row("todo-api")
        if not row:
            check("6. todo-api row has Tests Passed/Failed", 2, False, "row not found")
            return
        tp = row.get("Tests Passed")
        tf = row.get("Tests Failed")
        tp_ok = tp is not None and str(tp).replace("-", "").isdigit()
        tf_ok = tf is not None and str(tf).replace("-", "").isdigit()
        passed = tp_ok and tf_ok
        detail = f"Tests Passed={tp}, Tests Failed={tf}"
        check("6. todo-api row has Tests Passed/Failed", 2, passed, detail)
    except Exception as e:
        check("6. todo-api row has Tests Passed/Failed", 2, False, f"exception: {e}")


def check_7_pass_rate_correct():
    """Verify Pass Rate is correctly computed as passed/(passed+failed)*100 rounded to 2 decimals."""
    try:
        all_ok = True
        details = []
        for proj in ["data-analyzer", "todo-api"]:
            row = _find_row(proj)
            if not row:
                all_ok = False
                details.append(f"{proj}: row not found")
                continue
            tp = row.get("Tests Passed")
            tf = row.get("Tests Failed")
            pr = row.get("Pass Rate")
            if tp is None or tf is None or pr is None:
                all_ok = False
                details.append(f"{proj}: missing fields (tp={tp}, tf={tf}, pr={pr})")
                continue
            try:
                tp_i = int(float(str(tp)))
                tf_i = int(float(str(tf)))
                pr_f = float(str(pr))
            except (ValueError, TypeError):
                all_ok = False
                details.append(f"{proj}: non-numeric values (tp={tp}, tf={tf}, pr={pr})")
                continue
            total = tp_i + tf_i
            if total == 0:
                all_ok = False
                details.append(f"{proj}: total tests is 0")
                continue
            expected_rate = round(tp_i / total * 100, 2)
            if abs(pr_f - expected_rate) > 0.01:
                all_ok = False
                details.append(f"{proj}: expected rate {expected_rate}, got {pr_f}")
            else:
                details.append(f"{proj}: rate={pr_f} correct")
        check("7. Pass Rate correctly computed", 2, all_ok, "; ".join(details))
    except Exception as e:
        check("7. Pass Rate correctly computed", 2, False, f"exception: {e}")


def check_8_pass_fail_threshold():
    """Verify Pass/Fail is set correctly per 85.00 threshold."""
    try:
        all_ok = True
        details = []
        for proj in ["data-analyzer", "todo-api"]:
            row = _find_row(proj)
            if not row:
                all_ok = False
                details.append(f"{proj}: row not found")
                continue
            pr = row.get("Pass Rate")
            pf = row.get("Pass/Fail")
            if pr is None or pf is None:
                all_ok = False
                details.append(f"{proj}: missing fields (pr={pr}, pf={pf})")
                continue
            try:
                pr_f = float(str(pr))
            except (ValueError, TypeError):
                all_ok = False
                details.append(f"{proj}: non-numeric pass rate {pr}")
                continue
            # Pass/Fail may be a dict (single_select) or string
            pf_val = pf
            if isinstance(pf, dict):
                pf_val = pf.get("value", "")
            pf_str = str(pf_val).strip()
            expected_pf = "Pass" if pr_f >= 85.00 else "Fail"
            if pf_str.lower() != expected_pf.lower():
                all_ok = False
                details.append(f"{proj}: rate={pr_f}, expected {expected_pf}, got {pf_str}")
            else:
                details.append(f"{proj}: {pf_str} correct for rate {pr_f}")
        check("8. Pass/Fail correct per 85.00 threshold", 2, all_ok, "; ".join(details))
    except Exception as e:
        check("8. Pass/Fail correct per 85.00 threshold", 2, False, f"exception: {e}")


def check_9_captured_at():
    """Verify Captured At date is populated for both rows."""
    try:
        all_ok = True
        details = []
        for proj in ["data-analyzer", "todo-api"]:
            row = _find_row(proj)
            if not row:
                all_ok = False
                details.append(f"{proj}: row not found")
                continue
            ca = row.get("Captured At")
            if ca is None or str(ca).strip() == "":
                all_ok = False
                details.append(f"{proj}: Captured At empty")
            else:
                details.append(f"{proj}: {ca}")
        check("9. Captured At date populated", 1, all_ok, "; ".join(details))
    except Exception as e:
        check("9. Captured At date populated", 1, False, f"exception: {e}")


def check_10_op_work_package_exists():
    """Verify OpenProject has a Task WP 'Test Execution Audit Report' in 'product-catalog'."""
    try:
        result = op_sql(
            "SELECT wp.id, wp.subject, t.name AS type_name "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "WHERE p.identifier = 'product-catalog' "
            "AND wp.subject = 'Test Execution Audit Report' "
            "AND t.name = 'Task'"
        )
        found = len(result.strip()) > 0 if result else False
        check("10. OpenProject Task WP 'Test Execution Audit Report' exists", 2,
              found,
              f"query returned: {result[:100]}" if result else "not found")
    except Exception as e:
        check("10. OpenProject Task WP 'Test Execution Audit Report' exists", 2, False, f"exception: {e}")


def check_11_op_desc_data_analyzer():
    """Verify WP description mentions data-analyzer with pass rate and counts."""
    try:
        result = op_sql(
            "SELECT wp.description FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "WHERE p.identifier = 'product-catalog' "
            "AND wp.subject = 'Test Execution Audit Report'"
        )
        if not result:
            check("11. WP description has data-analyzer metrics", 2, False, "WP not found")
            return
        desc_lower = result.lower()
        has_project = "data-analyzer" in desc_lower
        # Look for numbers near data-analyzer mention — pass rate, passed, failed
        has_numbers = bool(re.search(r'\d+', result))
        passed = has_project and has_numbers
        detail = f"has 'data-analyzer': {has_project}, has numbers: {has_numbers}"
        check("11. WP description has data-analyzer metrics", 2, passed, detail)
    except Exception as e:
        check("11. WP description has data-analyzer metrics", 2, False, f"exception: {e}")


def check_12_op_desc_todo_api():
    """Verify WP description mentions todo-api with pass rate and counts."""
    try:
        result = op_sql(
            "SELECT wp.description FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "WHERE p.identifier = 'product-catalog' "
            "AND wp.subject = 'Test Execution Audit Report'"
        )
        if not result:
            check("12. WP description has todo-api metrics", 2, False, "WP not found")
            return
        desc_lower = result.lower()
        has_project = "todo-api" in desc_lower
        has_numbers = bool(re.search(r'\d+', result))
        # Check for pass/fail outcome mention
        has_outcome = "pass" in desc_lower or "fail" in desc_lower
        passed = has_project and has_numbers and has_outcome
        detail = f"has 'todo-api': {has_project}, has numbers: {has_numbers}, has pass/fail: {has_outcome}"
        check("12. WP description has todo-api metrics", 2, passed, detail)
    except Exception as e:
        check("12. WP description has todo-api metrics", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_projects_exist()
    check_2_baserow_db_exists()
    check_3_baserow_table_exists()
    check_4_exactly_two_rows()
    check_5_data_analyzer_counts()
    check_6_todo_api_counts()
    check_7_pass_rate_correct()
    check_8_pass_fail_threshold()
    check_9_captured_at()
    check_10_op_work_package_exists()
    check_11_op_desc_data_analyzer()
    check_12_op_desc_todo_api()

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
